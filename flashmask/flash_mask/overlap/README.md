# FM-4 Overlap Bridge

把 FM-3（FA3 / sm_90a）已验证的 NVSHMEM SM 级通信-overlap runtime，接到 FM-4（FA4 / cute-DSL / SM100）前向路径上的 native bridge。不重写 NVSHMEM 成 cute-DSL，而是保留 FM-3 native runtime，用一个很薄的 `extern "C"` bridge 把 `OverlapCommunicator` 的 SRBuffer 裸指针包成 cute 能消费的 tensor view，并把 forward all-gather 的 launch / 跨 iter 重置 / readiness 信号透传给 cute kernel 的 per-tile gate，做到真正的通信-计算重叠。

## 目录结构

```
flash_mask/overlap/
├── __init__.py            软 import 守卫（import 永不失败，.so 懒加载）
├── overlap_runtime.py     ctypes 加载 + uid bootstrap + init + AG launch + SrKvView 构造
├── README.md              本文件
└── csrc/
    ├── CMakeLists.txt      照搬 distributed/CMakeLists.txt 的 NVSHMEM 链接配方 + 自建 nvshmem target + 符号隔离
    ├── fm4_overlap_bridge.cu   extern "C" 薄封装
    └── fm4_overlap.map     version script，只导出 fm4_overlap_*
```

bridge `.so` 把三个 TU 编进一个 shared lib：`fm4_overlap_bridge.cu`（新增）、`overlap_comm.cu`、`sep_sr_buffer.cu`（后两个从 `csrc/flashmask_v2/distributed/` 复用，不改），静态链接预编译的 `libnvshmem.a` 加 UID bootstrap host `.so`。

## 调用链

前向 overlap 由 `cute/interface.py::_flash_attn_fwd` 在 `group is not None` 时驱动：

1. `overlap_runtime.ensure_initialized(k, v, group, mask_head)` —— 进程级 bootstrap unique id（仅一次），随后每步把 shape/topology 转发给 C++ 单例，由 `init_singleton_instance` 自行判断是否需要 reconfigure / 重新分配 SRBuffer。
2. `overlap_runtime.start_forward_ag(k, v, startend_row_indices, compute_stream)` —— 照搬 FM-3 forward launch template 的顺序发起稀疏 all-gather，**不做任何阻塞等待**：
   - `wait_sr_buffer_empty`：compute stream 通知 comm stream 上一步的 SRBuffer 消费完，可复用。
   - `compute_chunk_mask_sparse`：在 compute stream 上算 `copy_chunk_mask`，让 AG kernel 跳过全掩码 KV chunk。
   - `update_kv`：把本地 K/V `cudaMemcpyAsync` 进 SRBuffer 的本地 chunk 区。
   - `reset_ag_counter` + `wait_wptr_init`：在 compute stream 重置 AG 动态调度 counter（`block_cnt_semaphore`）为 1 并 record `wptr_init`，comm stream 等该 event 落地——这是 cute 路对 `prepare_flashmask` 重置那半的等价替身（cute kernel 不走 PHI scheduler，不会调 `prepare_flashmask`）。
   - `run_ag`：在 comm stream launch remote-get kernel。
   - `wait_reset_stream_coordinator`：compute stream 等 comm kernel 真正占住 SM（防 compute 抢光 SM 的死锁）。
   - 返回 `SrKvView`（SRBuffer K/V 裸指针 + 完整 `(B, S_total, H, D)` 维度）和 `write_ptr`（gate 自旋的 int32 counter）。
3. cute kernel（`flash_fwd_sm100.py`）用 `SrKvView` 的裸地址 + 运行时 Int32 维度构造 mK/mV，load warp 在每个远端 KV tile 前用 `_overlap_gate` 自旋 `write_ptr`，达成 overlap。

## 设计要点

- **cute 不需要知道底层是 NVSHMEM 内存**：只要拿到合法 device pointer 加 dtype/shape/stride 就能像普通 tensor 读。数据 readiness 不由 cute tensor 表达，而由 FM-3 已有的 `write_ptr` / event / semaphore 保证。
- **指针过界一律 `uint64_t`**：Python 只见 int，`reinterpret_cast` 回真实类型，位等价。
- **`OverlapCommunicator` 不暴露给 Python**：生命周期完全在 C++ static `unique_ptr` 单例里，到进程退出。
- **init 不解引用 k/v**：构造函数虽接收 k/v 指针但只存 shape，所以 `fm4_overlap_init` 内部传 nullptr，本地 KV 拷贝发生在 `update_kv`。
- **unique id 进程级缓存**：id 进程内恒定，`bootstrap_unique_id` 只在首次 broadcast，之后复用，避免每步一次 collective。
- **跨 iter counter 重置**：`block_cnt_semaphore` 是单例内一次分配、地址稳定的成员；不每步重置则 iter≥2 首个 `atomicAdd` 即超过 `total_chunks`，remote-get 主循环一行不搬，SRBuffer 远端区残留上一步数据。`reset_ag_counter` 补上这一步，且只在本 bridge、只被 FM-4 调，对 FM-3（走 `prepare_flashmask`）零影响。
- **AG 前必须算 chunk mask**：`copy_chunk_mask` 是 `block_work_ids` 里 `cudaMallocAsync` 分到的子区域，不清零；AG remote-get kernel 读它判断 256/512 行的 KV chunk 能否整段跳过（`remote_get_kernel.cuh:383`）。mask 张量必须存活到 `run_ag` 消费完（kernel 异步跑在 comm stream），故 `start_forward_ag` 持有 keepalive 到 launch 之后。

## 构建

bridge 是 opt-in 组件，不会被 `FLASHMASK_BUILD=all` 带上（需要 NVSHMEM 和 H100/B200）。必须显式请求 `ovl`。C++ 改动后必须重编 `.so`（cute kernel 是运行时 JIT，改 `.py` 立即生效）。

### 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `NVSHMEM_HOME` | `/root/work/Paddle/build/third_party/install/nvshmem` | NVSHMEM 安装根目录（含 `include/` 与 `lib/`）；传给子 CMake 的 `NVSHMEM_INSTALL_DIR` |
| `FM4_OVERLAP_CUDA_ARCH` | `90a` | 目标 sm arch，B200 设 `100a`（要求 CUDA >= 12.8） |
| `FM4_OVERLAP_CUTLASS_INC` | 自动探测 | cutlass include 目录（须含 `cutlass/bfloat16.h`） |

### H100（复用预编译 sm_90 NVSHMEM）

```bash
cd flashmask
FLASHMASK_BUILD=fa4,ovl pip install -e . --no-build-isolation
python -c "from flash_mask.overlap import overlap_runtime as r; r._load(); print('loaded OK')"
```

### B200（需要 sm_100 NVSHMEM）

Paddle 预编译的 `libnvshmem.a` 设备码只有 sm_90，B200 上须先备好一份 sm_100 的 NVSHMEM（重编，或指向 Paddle 自带的 sm_100 third_party），再：

```bash
cd flashmask
NVSHMEM_HOME=<sm100 NVSHMEM 路径> \
FM4_OVERLAP_CUDA_ARCH=100a \
FLASHMASK_BUILD=fa4,ovl pip install -e . --no-build-isolation
```

## 已知遗留风险

- 符号隔离（version-script + `-Bsymbolic` + `--exclude-libs,ALL`）只能挡 ELF interposition，挡不住 NVSHMEM plugin `.so` 静态状态共享这个更深的根因。当前单实例、不要求与 DeepEP 共存，故记录但不处理。
- `100a` 路径本地（A100）无法编译验证，靠静态审查保证。
