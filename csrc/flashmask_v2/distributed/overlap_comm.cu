#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include "overlap_comm.cuh"
#include "remote_get_kernel.cuh"
#include "remote_put_kernel.cuh"
#include "dk_dv_reduce_bf16.cuh"
#include "cp_heuristic.cuh"
#include "layout_transpose.cuh"

namespace flashmask {

// whether should we manually manage nvshmem related environment setups
// deprecation warning: will be removed in the future
static constexpr bool SHOULD_MANAGE_NVSHMEM = true;
// no team_bar but fine-grained signaling, good for single node CUDA IPC
static constexpr bool USE_SEMAPHORES = true;
// whether to use stream coordinator to make sure the scheduling order of comm & comp kernels
// Note(heqianyue): If we can make sure CUDA_DEVICE_MAX_CONNECTION=1, stream_coord is actually not required
static constexpr bool USE_STREAM_COORD = true;
// RS buffer capacity when per-stage buffering is disabled (FLASHMASK_PER_STAGE_BUFFER=false).
// When per-stage is enabled, capacity is num_segments instead (one slot per segment).
static constexpr int RS_BUFFER_CAPACITY = 1;
// SM_MARGIN works for 128K CP16, but for 32K CP4, the performance is a bit degraded.
static constexpr int OVERLAP_SM_MARGIN = 0;

// allowed value: [16, 32, 64] (larger number is generally better for KV with larger num_head and higher CP)
static constexpr int RDMA_ROW_PER_WARP = 32;
static constexpr int STREAM_COORD_OFFSET = USE_STREAM_COORD ? 2 : 0;   // do not adjust the value if you don't know what ur doing

// allowed value: 16 or 8 (16 warps are generally better for KV with larger num_head and higher CP)
static constexpr int num_warps = 16;    // making the grid larger is generally better
static constexpr int num_blocks = 32;   // 32 reg, 256 thread, one SM of H800 can hold 8 blocks, we use 4 SM

// RS-overlap: each segment has 4 chunks.
static constexpr int rs_overlap_min_h_k = 1;

// ====== S_local Static Dispatch ======
// S_local (= S_chunk) must be a power of 2 and divisible by row_per_block (= num_warps * RDMA_ROW_PER_WARP = 512).
// Host-side dispatch selects the compile-time S_chunk constant for GPU kernels.
// Supported values: {4096, 8192, 16384, 32768, 65536, 131072}.
// NOTE: For S_local >= 32768, RS-overlap is automatically disabled (num_chunks=1 from heuristic,
//       which makes the splitted AG/RS code paths degenerate). AG-only overlap still works.
// WARNING: For very large S_local (131072) with high nranks (16), S_full * H * D may approach
//          INT_MAX in kernel address calculations. Known limitation of the existing kernel code.

#define SChunkCase(MACRO_FUNC, _S_chunk, ...) \
    case _S_chunk: { MACRO_FUNC(_S_chunk, ##__VA_ARGS__); break; }

#define SChunkDispatch(MACRO_FUNC, _S_local, ...)                                \
    switch (_S_local) {                                                          \
        SChunkCase(MACRO_FUNC, 4096, ##__VA_ARGS__)                              \
        SChunkCase(MACRO_FUNC, 8192, ##__VA_ARGS__)                              \
        SChunkCase(MACRO_FUNC, 16384, ##__VA_ARGS__)                             \
        SChunkCase(MACRO_FUNC, 32768, ##__VA_ARGS__)                             \
        SChunkCase(MACRO_FUNC, 65536, ##__VA_ARGS__)                             \
        SChunkCase(MACRO_FUNC, 131072, ##__VA_ARGS__)                            \
    default:                                                                     \
        throw std::invalid_argument(                                             \
            "[FlashMask Overlap] S_local must be one of {4096, 8192, 16384, 32768, 65536, 131072}, got: " \
            + std::to_string(_S_local));                                         \
    }

template <typename Ty>
void dump_sr_buffer(const Ty* const src, int num_elem, int rank, std::string buffer_name) {
    // used only when debugging, dump the SR buffer to a .bin binary and can be read by numpy
    if (src == nullptr || num_elem <= 0) return;
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());

    size_t size_in_bytes = num_elem * sizeof(Ty);

    // 1. Allocate CPU staging memory
    std::vector<char> host_buffer(size_in_bytes);

    // 2. Copy data from GPU (Global Memory) to CPU
    cudaError_t err = cudaMemcpy(host_buffer.data(), src, size_in_bytes, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        std::cerr << "CUDA Memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 3. Construct filename and write to binary file
    std::string filename = buffer_name + std::to_string(rank) + ".bin";
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);

    if (outfile.is_open()) {
        outfile.write(host_buffer.data(), size_in_bytes);
        outfile.close();
        std::cout << "Successfully dumped " << num_elem << " elements to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open file: " << filename << std::endl;
    }
}

void get_nvshmem_info(int& my_pe, int& n_pes) {
    my_pe = nvshmem_my_pe();
    n_pes = nvshmem_n_pes();
}

void init_with_unique_id(
    std::vector<uint8_t>&& root_unique_id_val,
    int rank,
    int num_ranks
) {       // adopted from DeepEP
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    std::memcpy(
        &root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    WARN_PRINT("Start to set unique ID args...\n");
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    WARN_PRINT("Start to set init attr...\n");
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    // TODO(heqianyue): Do we need to bar here?
    WARN_PRINT("%d / %d bars before completing the init.\n", rank, num_ranks);
    nvshmem_barrier_all();
}

void init_distributed_environment(
    int rank,
    int nranks,
    int& my_pe, 
    int& n_pes,
    const uint8_t* unique_id_ptr
) {
    if (unique_id_ptr == nullptr) {
        throw std::runtime_error("unique_id_ptr is null: NVSHMEM initialization requires a valid unique ID.");
    }

    bool all_zeros = std::all_of(unique_id_ptr, unique_id_ptr + 128, [](uint8_t x) { return x == 0; });
    if (all_zeros) {
        throw std::runtime_error("invalid unique_id: The provided NVSHMEM unique ID consists entirely of zeros.");
    }

    WARN_PRINT("[FlashMask Overlap] Initializing NVSHMEM... Rank: %d / %d, PE ID: %d / %d\n", rank, nranks, my_pe, n_pes);
    std::vector<uint8_t> unique_id_val;
    WARN_PRINT("Extracting unique ID...");
    unique_id_val.resize(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(unique_id_val.data(), unique_id_ptr, sizeof(nvshmemx_uniqueid_t));
    init_with_unique_id(std::move(unique_id_val), rank, nranks);
    get_nvshmem_info(my_pe, n_pes);
    WARN_PRINT("[FlashMask Overlap] NVSHMEM initialized. Rank: %d / %d, PE ID: %d / %d\n", rank, nranks, my_pe, n_pes);
}

void finalize_distributed_environment() {
    WARN_PRINT("[FlashMask Overlap] Finalizing...\n");
    nvshmem_finalize();
    WARN_PRINT("[FlashMask Overlap] NVSHMEM env finalized.\n");
}

template <typename KVType>
OverlapCommunicator<KVType>::OverlapCommunicator(
    const KVType* const k_data,
    const KVType* const v_data,
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    const uint8_t* unique_id_ptr,
    int mask_head,
    bool overlap_rs
    // Maybe we should manage the following by ourselves? Do not pass as parameters
): kv_buffer(nullptr),
   dkv_buffer(nullptr),
   B(b_kv),
   S_local(s_kv),
   H(h_kv),
   H_mask(mask_head),
   D(d_kv),
   num_chunks(get_num_chunk_per_segment(s_kv, nranks, h_kv)),
   block_work_ids(nullptr),
   block_cnt_semaphore(nullptr),
   copy_chunk_mask(nullptr),
   put_streams(nullptr),
   bwd_done_events(nullptr),
   rs_block_cnt(nullptr),
   _num_put_streams(0)
{
    if constexpr (SHOULD_MANAGE_NVSHMEM) {
        init_distributed_environment(rank, nranks, _my_pe, _total_n_pes, unique_id_ptr);
    } else {
        get_nvshmem_info(_my_pe, _total_n_pes);     // get info if nvshmem is already avaliable
    }

    // Hierarchical overlap topology discovery
    _gpus_per_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE); 
    _my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    _num_nodes = _total_n_pes / _gpus_per_node;

    _flags = OverlapFeatureFlags::from_env();

    // Fallback 1: hierarchical mode is a no-op on a single node — fall back to circular shift.
    if (_flags.use_hierarchical && !hier::hier_is_effective(_total_n_pes, _gpus_per_node)) {
        WARN_PRINT("[FlashMask Overlap] FLASHMASK_USE_HIERARCHICAL=true but single node (%d PEs, %d per node). "
                   "Hierarchical mode has no effect, falling back to circular shift.\n",
                   _total_n_pes, _gpus_per_node);
        _flags.use_hierarchical = false;
    }

    // Fallback 2: BHSD + hierarchical requires B*H <= 64 (sema_intra bitmask limit).
    if (_flags.use_bhsd_layout && _flags.use_hierarchical && b_kv * h_kv > 64) {
        printf("[FlashMask Overlap] BHSD + hierarchical requires B*H <= 64 (sema_intra bitmask limit), "
               "got B*H=%d. Falling back to BSHD layout.\n", b_kv * h_kv);
        _flags.use_bhsd_layout = false;
    }

    // Log the effective (post-fallback) switch state.
    _flags.print(rank);

    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    cudaStreamCreateWithPriority(&comm_stream, 
                                  cudaStreamNonBlocking,
                                  greatest_priority);
    cudaEventCreateWithFlags(&wptr_init, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&sr_usable, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ag_done, cudaEventDisableTiming);
    cudaEventRecord(sr_usable, comm_stream);                    // set initial status for SR buffer
    cudaEventRecord(ag_done, comm_stream);                      // initial state: no AG pending
    _local_batch_stride = s_kv * h_kv * d_kv;
    _total_numel = _local_batch_stride * b_kv * nranks;             // won't overflow, but should be careful

    // This variable is simply a int32_t, so can be passed by value
    nvshmem_team_t cp_team = NVSHMEM_TEAM_WORLD;
    const int sema_count = USE_SEMAPHORES ? (_flags.use_hierarchical ? _num_nodes + _total_n_pes : _total_n_pes) : 0;
    _sema_inter_size = _flags.use_hierarchical ? _num_nodes : 0;
    kv_buffer = std::make_unique<SRBuffer<KVType>>(_total_numel, cp_team, sema_count);
    if constexpr (USE_SEMAPHORES) {
        cudaMemset(kv_buffer->semaphores(), 0, sizeof(int64_t) * sema_count);
    }
    const int num_copy_chunks = b_kv * s_kv * nranks / (RDMA_ROW_PER_WARP * num_warps);
    // FWD uses num_blocks slots (reduce-based wptr). BWD uses work_done[0..N] bitmap (per-work flag).
    // Allocate max of both so the same buffer serves both paths.
    // BHSD: bitmap indexed by effective_batch (B*H) × works_per_batch, need more entries.
    const int bitmap_entries = _flags.use_bhsd_layout
        ? (b_kv * h_kv * s_kv * nranks / (RDMA_ROW_PER_WARP * num_warps))
        : num_copy_chunks;
    _bitmap_region_size = ((bitmap_entries + 1) + 3) & ~3;
    const int head_region_size = std::max(num_blocks, _bitmap_region_size);
    // block_cnt_semaphore (AG overlap only requires only 1 extra int, 2 is for RS overlap, 2 more for padding)
    // rank_commit_counters: nranks (for RS P2P commit)
    // rank_empty_counters: effective_batch * nranks (for hierarchical per-batch Phase 1 AG counters)
    //   BHSD: effective_batch = B*H; BSHD: effective_batch = B
    const int effective_batch_alloc = _flags.use_bhsd_layout ? b_kv * h_kv : b_kv;
    const int rank_counters_size = nranks + effective_batch_alloc * nranks;
    CUDA_DEBUG_CHECK(cudaMallocAsync(&block_work_ids, sizeof(int) * (head_region_size + num_copy_chunks + 4 + rank_counters_size), comm_stream));
    copy_chunk_mask = block_work_ids + head_region_size;
    block_cnt_semaphore = copy_chunk_mask + num_copy_chunks;
    stream_coordinator = block_cnt_semaphore + STREAM_COORD_OFFSET;
    rank_commit_counters = block_cnt_semaphore + 4;
    rank_empty_counters = rank_commit_counters + nranks;
    if constexpr (USE_STREAM_COORD) {
        // allocate stream coordinator and set the value to 0, so that comp stream will wait until comm stream have set this
        cudaMemsetAsync(stream_coordinator, 0, sizeof(int), comm_stream);
    }
    if (overlap_rs) {
        // auxilary stream for RS-overlap (for used in reduce)
        cudaStreamCreateWithPriority(&aux_p_stream, cudaStreamNonBlocking, std::min(greatest_priority + 1, least_priority));
        cudaStreamCreateWithPriority(&aux_c_stream, cudaStreamNonBlocking, std::min(greatest_priority + 1, least_priority));
        cudaEventCreateWithFlags(&bwd_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&reduce_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&local_moved, cudaEventDisableTiming);
        const int num_stages = _total_n_pes / num_chunks;
        const int rs_capacity = _flags.per_stage_buffer ? num_stages : RS_BUFFER_CAPACITY;
        dkv_buffer = std::make_unique<SepSRBuffer<KVType>>(
            _local_batch_stride * b_kv,      // single chunk K numel (B * S_local * H * D)
            _total_n_pes,
            num_chunks,
            rs_capacity,
            cp_team
        );
        dkv_buffer->initialize_buffer(_my_pe, _flags.per_stage_buffer);

        // Per-segment producer streams for parallel put (per-stage buffering only)
        if (_flags.per_stage_buffer) {
            _num_put_streams = num_stages;
            put_streams = new cudaStream_t[num_stages];
            bwd_done_events = new cudaEvent_t[num_stages];
            for (int i = 0; i < num_stages; i++) {
                cudaStreamCreateWithPriority(&put_streams[i], cudaStreamNonBlocking,
                    std::min(greatest_priority + 1, least_priority));
                cudaEventCreateWithFlags(&bwd_done_events[i], cudaEventDisableTiming);
            }
            CUDA_DEBUG_CHECK(cudaMalloc(&rs_block_cnt, sizeof(int) * num_stages));
        }

        printf("[FlashMask Overlap] Using RS-Overlap, buffer capacity: %d, num_chunks: %d\n",
                rs_capacity, num_chunks);
    }
    // Ensure initialize_buffer memset completes before barrier so that after
    // team_bar returns, all PEs' semaphores are guaranteed to be zeroed.
    kv_buffer->team_bar();
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    if (overlap_rs && _flags.per_stage_buffer) {
        prime_rs_semaphores();
    }
    // Initialize configuration tracking
    _config = {b_kv, s_kv, h_kv, mask_head, d_kv, nranks, overlap_rs, _flags.use_bhsd_layout};
    _sr_buffer_capacity = _total_numel;
    _dkv_single_k_numel_capacity = overlap_rs ? (_local_batch_stride * b_kv) : 0;
    _dkv_num_chunks = overlap_rs ? num_chunks : 0;
    _num_copy_chunks = b_kv * s_kv * nranks / (RDMA_ROW_PER_WARP * num_warps);
    // copy to the last position of the SR buffer
    WARN_PRINT("SR buffer valid: %d, B, S, H, D: %d, %d, %d, %d, nranks: %d\n", int(kv_buffer->is_valid()), B, S_local, H, D, nranks);
    WARN_PRINT("[FlashMask Overlap] constructor rank: %d, nranks: %d\n", rank, nranks);
}

template <typename KVType>
OverlapCommunicator<KVType>::~OverlapCommunicator() {
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    CUDA_DEBUG_CHECK(cudaFreeAsync(block_work_ids, comm_stream));
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    CUDA_DEBUG_CHECK(cudaEventDestroy(wptr_init));
    CUDA_DEBUG_CHECK(cudaEventDestroy(sr_usable));
    CUDA_DEBUG_CHECK(cudaEventDestroy(ag_done));
    CUDA_DEBUG_CHECK(cudaStreamDestroy(comm_stream));
    kv_buffer->release();           // do not depend on auto-release
    if (dkv_buffer) {
        CUDA_DEBUG_CHECK(cudaEventDestroy(bwd_done));
        CUDA_DEBUG_CHECK(cudaEventDestroy(reduce_done));
        CUDA_DEBUG_CHECK(cudaEventDestroy(local_moved));
        CUDA_DEBUG_CHECK(cudaStreamDestroy(aux_p_stream));
        CUDA_DEBUG_CHECK(cudaStreamDestroy(aux_c_stream));
        if (_num_put_streams > 0) {
            for (int i = 0; i < _num_put_streams; i++) {
                CUDA_DEBUG_CHECK(cudaStreamDestroy(put_streams[i]));
                CUDA_DEBUG_CHECK(cudaEventDestroy(bwd_done_events[i]));
            }
            delete[] put_streams;
            delete[] bwd_done_events;
            CUDA_DEBUG_CHECK(cudaFree(rs_block_cnt));
            put_streams = nullptr;
            bwd_done_events = nullptr;
            rs_block_cnt = nullptr;
            _num_put_streams = 0;
        }
        dkv_buffer->release();
    }
    if constexpr (SHOULD_MANAGE_NVSHMEM && MANUAL_CLEANUP) {
        finalize_distributed_environment();
    }
}

template <typename KVType>
void OverlapCommunicator<KVType>::wait_wptr_init() {
    cudaStreamWaitEvent(comm_stream, wptr_init);
}

__global__ void WaitAndResetStreamCoordKernel(
    int* const stream_coordinator
) {
    do {
        int old_wptr_val = 0;
        asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(old_wptr_val) : "l"(stream_coordinator));
        if (old_wptr_val == 0xffffffff) break;
        __nanosleep(10);
    } while (true);
    *stream_coordinator = 0;
    __threadfence();
}

template <typename KVType>
void OverlapCommunicator<KVType>::wait_reset_stream_coordinator(cudaStream_t stream) {
    static_assert(num_blocks <= 32, "To correctly use stream coordinator, num CTAs for comm kernels cannot exceed 32.");
    if constexpr (USE_STREAM_COORD) {
        WaitAndResetStreamCoordKernel<<<1, 1, 0, stream>>>(stream_coordinator);
    }
}

template <typename KVType>
void OverlapCommunicator<KVType>::wait_sr_buffer_empty(cudaStream_t stream) {
    // TBH, since there is enough time between two attention call,
    // for an unsafe impl, we actually don't need to wait.
    // This would save some time, but not entirely safe if
    // the attention's workload is too too small. Yet currently
    // we haven't trigger unsafe problems even once
    cudaEventRecord(sr_usable, stream);             // in-case there is previously unfinished comp stream ops
    cudaStreamWaitEvent(comm_stream, sr_usable);
    if constexpr (USE_SEMAPHORES) {
        WARN_PRINT("Before wait_self_empty\n");
        // block if computation that uses the local chunk is not finished
        // so that we won't corrupt the local buffer being used
        if (_flags.use_hierarchical) {
            sema::ag::wait_self_empty_hier(
                sema_intra(),
                _my_pe,
                _gpus_per_node,
                _num_nodes,
                comm_stream
            );
        } else {
            sema::ag::wait_self_empty(
                sema_intra(),
                _my_pe,
                comm_stream
            );
        }
        WARN_PRINT_SYNC(comm_stream, "After wait_self_empty\n");
    }
}

template <typename KVType>
void OverlapCommunicator<KVType>::update_kv_buffer(
    const KVType* const new_k_data,
    const KVType* const new_v_data,
    const bool fwd
) {
    // remember to pair `update_kv_buffer` with `wait_sr_buffer_empty` (except from the constructor call)
    // this `cudaMemcpyAsync` itself won't introduce too much overhead
    // yet, `team_bar` (nvshmem_sync_team) is the culprit
    WARN_PRINT("Before cudaMemcpyAsync... is fwd: %d\n", int(fwd));
    // bwd copies the data to the start chunk of the SR, while fwd copies to the last chunk
    const int local_offset = fwd ? (_local_batch_stride * (_total_n_pes - 1)) : 0;
    // for bwd RS-overlap splitted AG, the batch stride is num_chunks * S_local * S_stride
    int batch_stride = _local_batch_stride * _total_n_pes;
    if (fwd == false && dkv_buffer) {
        batch_stride = _local_batch_stride * num_chunks;
        // Non-hierarchical BWD: save local KV at the SR tail so remote ranks can Phase-1 fetch it.
        // Hierarchical BWD: local KV lives at segment-0 offset-0 (SR base) and is never overwritten,
        // so no tail backup is needed.
        if (!_flags.use_hierarchical) {
            if (_flags.use_bhsd_layout) {
                // BHSD: transpose local KV into the tail backup region as (B,H,S_local,D)
                layout::transpose_copy_to_sr(
                    reinterpret_cast<const __nv_bfloat16*>(new_k_data),
                    reinterpret_cast<__nv_bfloat16*>(local_k_data()),
                    B, S_local, H, S_local, 0, comm_stream);
                layout::transpose_copy_to_sr(
                    reinterpret_cast<const __nv_bfloat16*>(new_v_data),
                    reinterpret_cast<__nv_bfloat16*>(local_v_data()),
                    B, S_local, H, S_local, 0, comm_stream);
            } else {
                const size_t copy_bytes = B * _local_batch_stride * sizeof(KVType);
                CUDA_DEBUG_CHECK(cudaMemcpyAsync(local_k_data(), new_k_data, copy_bytes,
                                        cudaMemcpyDeviceToDevice, comm_stream));
                CUDA_DEBUG_CHECK(cudaMemcpyAsync(local_v_data(), new_v_data, copy_bytes,
                                        cudaMemcpyDeviceToDevice, comm_stream));
            }
        }
    }
    if (_flags.use_bhsd_layout) {
        // BHSD path: transpose-copy BSHD src → BHSD SR buffer at correct S offset
        const int S_dst = fwd ? (S_local * _total_n_pes) : (S_local * num_chunks);
        const int s_offset = fwd ? (S_dst - S_local) : 0;
        layout::transpose_copy_to_sr(
            reinterpret_cast<const __nv_bfloat16*>(new_k_data),
            reinterpret_cast<__nv_bfloat16*>(kv_buffer->k_data()),
            B, S_local, H, S_dst, s_offset, comm_stream);
        layout::transpose_copy_to_sr(
            reinterpret_cast<const __nv_bfloat16*>(new_v_data),
            reinterpret_cast<__nv_bfloat16*>(kv_buffer->v_data()),
            B, S_local, H, S_dst, s_offset, comm_stream);
        cudaError_t transpose_err = cudaGetLastError();
        if (transpose_err != cudaSuccess) {
            fprintf(stderr, "[BHSD DEBUG] CUDA error AFTER transpose_copy_to_sr: %s\n",
                    cudaGetErrorString(transpose_err));
        }
    } else {
        // BSHD path: per-batch flat memcpy into SR buffer
        for (int bid = 0; bid < B; bid++) {
            CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->k_data() + local_offset + bid * batch_stride,
                                    new_k_data + bid * _local_batch_stride,
                                    _local_batch_stride * sizeof(KVType),
                                    cudaMemcpyDeviceToDevice,
                                    comm_stream));
            CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->v_data() + local_offset + bid * batch_stride,
                                    new_v_data + bid * _local_batch_stride,
                                    _local_batch_stride * sizeof(KVType),
                                    cudaMemcpyDeviceToDevice,
                                    comm_stream));
        }
    }
    if constexpr (USE_SEMAPHORES) {
        // notify all other PEs that the local data is ready (1. set self to be `total_pes - 1`. 2. broadcast to other PEs)
        // BHSD: effective_batch = B*H (each (b,h) uses one bit in sema_intra int64)
        const int effective_batch = _flags.use_bhsd_layout ? B * H : B;
        if (_flags.use_hierarchical) {
            sema::ag::notify_full_hier(
                sema_inter(),
                sema_intra(),
                _my_pe, _total_n_pes,
                _gpus_per_node, _num_nodes,
                effective_batch, comm_stream
            );
        } else {
            sema::ag::notify_full(
                kv_buffer->semaphores(),
                _my_pe, _total_n_pes,
                kv_buffer->team(), comm_stream
            );
        }
    } else {
        // bar, so that comm_stream will finish transfering data before starting to get data from remote
        // this can be slow if the pace difference of PEs is large  
        kv_buffer->team_bar_on_stream(comm_stream);
    }
    if constexpr (!USE_STREAM_COORD && OVERLAP_SM_MARGIN == 0) {
        // sync, otherwise it might hang unexceptedly. The culprit is that computation CTAs
        // are scheduled before communication kernels, occupying all the SMs, causing deadlock.
        CUDA_DEBUG_CHECK(cudaStreamSynchronize(comm_stream));
    }
    WARN_PRINT("After cudaMemcpyAsync and notify full\n");
}

// Guard: if constexpr (_S > S_chunk) prevents template instantiation for degenerate
// S==S_chunk (nranks=1) combinations generated by SChunkDispatch × SeqlenDispatch.
// These are unreachable at runtime (overlap requires nranks >= 2, so S_full >= 2*S_chunk).
// Eliminating them saves ~40% of kernel binary from combinatorial expansion.

#define LaunchSparseKernel(_S, bwd, _hier)                                            \
    SparseLargeKVChunkRemoteGetKernel<KVType, _S, S_chunk,                            \
        num_warps, RDMA_ROW_PER_WARP, USE_STREAM_COORD, USE_SEMAPHORES, bwd,          \
        _hier>                                                                        \
        <<<num_blocks, num_warps * 32, 0, comm_stream>>>(                             \
                        kv_buffer->k_data(),                                          \
                        kv_buffer->v_data(),                                          \
                        write_ptr,                                                    \
                        block_work_ids,                                               \
                        block_cnt_semaphore,                                          \
                        stream_coordinator,                                           \
                        copy_chunk_mask,                                              \
                        _my_pe,                                                       \
                        _total_n_pes,                                                 \
                        effective_batch,                                              \
                        effective_s_stride,                                           \
                        kv_buffer->semaphores(),                                      \
                        rank_empty_counters,                                          \
                        _gpus_per_node,                                               \
                        _sema_inter_size,                                             \
                        effective_num_heads                                           \
                    )

#define SparseLargeChunkKernel(_S, bwd)                                             \
    if constexpr ((_S) > S_chunk) {                                                 \
        if (_flags.use_hierarchical) {                                              \
            LaunchSparseKernel(_S, bwd, true);                                      \
        } else {                                                                    \
            LaunchSparseKernel(_S, bwd, false);                                     \
        }                                                                           \
    }

#define SeqlenCase(MACRO_FUNC, _S, KernelTraits, ...)       \
    case _S: {                                              \
        MACRO_FUNC(_S, KernelTraits, ##__VA_ARGS__); break; \
    }

#define SeqlenDispatch(MACRO_FUNC, _S, ...)                 \
    switch (_S) {                                           \
        SeqlenCase(MACRO_FUNC, 8192, ##__VA_ARGS__)         \
        SeqlenCase(MACRO_FUNC, 16384, ##__VA_ARGS__)        \
        SeqlenCase(MACRO_FUNC, 32768, ##__VA_ARGS__)        \
        SeqlenCase(MACRO_FUNC, 65536, ##__VA_ARGS__)        \
        SeqlenCase(MACRO_FUNC, 131072, ##__VA_ARGS__)       \
        SeqlenCase(MACRO_FUNC, 262144, ##__VA_ARGS__)       \
        SeqlenCase(MACRO_FUNC, 524288, ##__VA_ARGS__)       \
        SeqlenCase(MACRO_FUNC, 1048576, ##__VA_ARGS__)      \
        SeqlenCase(MACRO_FUNC, 2097152, ##__VA_ARGS__)      \
    default:                                                \
        throw std::invalid_argument(                        \
            "[FlashMask Overlap] Full seqlen (S_local * nranks) must be one of " \
            "{8192..2097152}, got: " + std::to_string(_S)); \
    }

template <typename KVType>
void OverlapCommunicator<KVType>::compute_chunk_mask(
    const int* const lt_start_ptr,
    const int* const lt_end_ptr,
    const int* const ut_start_ptr,
    const int* const ut_end_ptr,
    cudaStream_t stream,
    const bool fwd
) {
    if (lt_end_ptr || ut_start_ptr) {
        std::cerr << "[Warning] FlashMask Overlap does not support mask with lt_end and ut_start ptrs yet. Will be added soon.\n";
    }
    if (ut_end_ptr == nullptr || lt_start_ptr == nullptr) {
        std::cerr << "For FlashMask Overlap, lt_start_ptr and ut_end_ptr can't be null.\n";
        throw std::runtime_error("nullptr found for mask pointers.");
    }
    constexpr int num_reduce_warp = RDMA_ROW_PER_WARP == 16 ? 4 : num_warps;
#define CallBlockSparsityKernel(S_chunk, grid, skip_local)                                             \
    BlockSparsityCheckSpecializedKernel<S_chunk, num_reduce_warp, RDMA_ROW_PER_WARP, skip_local>       \
                <<< grid, num_reduce_warp * 32, 0, stream >>>(lt_start_ptr, ut_end_ptr, copy_chunk_mask, H_mask, head_stride)

#define ComputeChunkMaskBody(_S_chunk)                                                                 \
    do {                                                                                               \
        constexpr int S_chunk = _S_chunk;                                                              \
        const int head_stride = S_local * _total_n_pes;                                                \
        const int valid_seqlen_k = head_stride - S_chunk;                                              \
        if (fwd) {                                                                                     \
            dim3 grids = dim3(valid_seqlen_k / (num_reduce_warp * 32 * 4), B, 1);                      \
            CallBlockSparsityKernel(S_chunk, grids, false);                                            \
        } else {                                                                                       \
            WARN_PRINT("(%d) Before compute_chunk_mask (bwd).\n", _my_pe);                             \
            if (dkv_buffer) {                                                                          \
                dim3 grids = dim3((valid_seqlen_k + S_chunk) / (num_reduce_warp * 32 * 4), B, 1);      \
                CallBlockSparsityKernel(S_chunk, grids, false);                                        \
            } else {                                                                                   \
                dim3 grids = dim3(valid_seqlen_k / (num_reduce_warp * 32 * 4), B, 1);                  \
                CallBlockSparsityKernel(S_chunk, grids, true);                                         \
            }                                                                                          \
            WARN_PRINT_SYNC(stream, "(%d) After compute_chunk_mask (bwd).\n", _my_pe);                 \
        }                                                                                              \
    } while(0)

    SChunkDispatch(ComputeChunkMaskBody, S_local);
#undef ComputeChunkMaskBody
#undef CallBlockSparsityKernel
}

template <typename KVType>
int OverlapCommunicator<KVType>::chunk_per_seg() const {
    return num_chunks;
}

/**
 * run the overlap kernel asynchronously
 * @param S the seqlen of local K (for example, 32K full length, CP=4, local S=8K)
*/
template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_ag_kernel(
    int* const write_ptr,
    int& S,
    const bool fwd
) {
    S = S_local * _total_n_pes;
    // BHSD layout: treat data as (B*H, S, D) — each (b,h) pair is an independent batch with S_stride=D.
    const int effective_batch = _flags.use_bhsd_layout ? B * H : B;
    const int effective_s_stride = _flags.use_bhsd_layout ? D : H * D;
    const int effective_num_heads = _flags.use_bhsd_layout ? H : 1;
    // set 0 every time we start the comm kernel --- meaning that we don't have any available attn-blocks
    cudaMemsetAsync(block_work_ids, 0, sizeof(int) * num_blocks, comm_stream);
    if constexpr (USE_SEMAPHORES) {
        // Hierarchical: per-batch Phase 1 counters
        // BHSD: effective_batch = B*H (each (b,h) pair treated as independent batch), BSHD: effective_batch = B
        if (_flags.use_hierarchical) {
            cudaMemsetAsync(rank_empty_counters, 0, sizeof(int) * effective_batch * _total_n_pes, comm_stream);
        }
    }
    WARN_PRINT_SYNC(comm_stream, "Before remote get kernel: %d\n", _my_pe);

#define AgKernelBody(_S_chunk)                                                       \
    do {                                                                             \
        constexpr int S_chunk = _S_chunk;                                            \
        const int _S_full = S_local * _total_n_pes;                                  \
        if (fwd) {                                                                   \
            SeqlenDispatch(SparseLargeChunkKernel, _S_full, false);                  \
        } else {                                                                     \
            SeqlenDispatch(SparseLargeChunkKernel, _S_full, true);                   \
        }                                                                            \
    } while(0)

    SChunkDispatch(AgKernelBody, S_local);
#undef AgKernelBody

    // Post-AG cleanup: notify all remote producers that we finished consuming their data.
    // This replaces the old inline try_release_rank in the get kernel.
    if constexpr (USE_SEMAPHORES) {
        if (_flags.use_hierarchical) {
            sema::ag::notify_all_empty_hier(
                sema_inter(), sema_intra(),
                _my_pe, _total_n_pes,
                _gpus_per_node, _num_nodes,
                effective_batch,
                comm_stream
            );
        } else {
            sema::ag::notify_all_empty(
                kv_buffer->semaphores(),
                _my_pe, _total_n_pes,
                comm_stream
            );
        }
    }
    cudaEventRecord(ag_done, comm_stream);
    WARN_PRINT_SYNC(comm_stream, "After remote_get kernel\n");
}
#undef SeqlenDispatch
#undef SparseLargeChunkKernel

#define SparseLargeChunkSplittedKernel(num_chunk, _has_local, _use_hier, start_rank, seg_idx, num_segs, smem_bytes) \
    SparseLargeKVChunkSplittedRemoteGetKernel<KVType, S_chunk,                                  \
        num_warps, RDMA_ROW_PER_WARP, num_chunk, _has_local, USE_STREAM_COORD, USE_SEMAPHORES,  \
        _use_hier>                                                                              \
        <<<num_blocks, num_warps * 32, smem_bytes, comm_stream>>>(                              \
                        _k_sr_ptr,                                                              \
                        _v_sr_ptr,                                                              \
                        _local_k_ptr,                                                           \
                        _local_v_ptr,                                                           \
                        block_work_ids,                                                         \
                        block_cnt_semaphore,                                                    \
                        stream_coordinator,                                                     \
                        copy_chunk_mask,                                                        \
                        _my_pe,                                                                 \
                        start_rank,                                                             \
                        seg_idx,                                                                \
                        _total_n_pes,                                                           \
                        effective_batch,                                                        \
                        effective_s_stride,                                                     \
                        num_segs,                                                               \
                        kv_buffer->semaphores(),                                                \
                        rank_empty_counters,                                                    \
                        _gpus_per_node,                                                         \
                        _sema_inter_size,                                                       \
                        effective_num_heads                                                     \
                    )

#define NumChunkDispatchSplitted(MACRO_FUNC, num_chunk, _has_local, ...)  \
    switch (num_chunk) {                                                  \
        case 4: { MACRO_FUNC(4, _has_local, __VA_ARGS__); break; }        \
        case 2: { MACRO_FUNC(2, _has_local, __VA_ARGS__); break; }        \
        case 8: { MACRO_FUNC(8, _has_local, __VA_ARGS__); break; }        \
        case 1: { MACRO_FUNC(1, _has_local, __VA_ARGS__); break; }        \
    default:                                                              \
        throw std::invalid_argument(                                      \
            "[FlashMask Overlap] Num chunk per segment must be one of "   \
            "{1, 2, 4, 8}, got: " + std::to_string(num_chunk));           \
    }

template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_splitted_ag_kernel(
    int* const write_ptr,
    int segment_idx
) {
    WARN_PRINT("(%d) Before run_overlap_splitted_ag_kernel, segment: %d\n", _my_pe, segment_idx);
    const int start_pe = (_my_pe + segment_idx * num_chunks) % _total_n_pes;
    const int num_segs = _total_n_pes / num_chunks;
    // BHSD: effective batch = B*H, stride between S rows = D
    const int effective_batch = _flags.use_bhsd_layout ? B * H : B;
    const int effective_s_stride = _flags.use_bhsd_layout ? D : H * D;
    const int effective_num_heads = _flags.use_bhsd_layout ? H : 1;

#define SplittedAgBody(_S_chunk)                                                                              \
    do {                                                                                                      \
        constexpr int S_chunk = _S_chunk;                                                                     \
        const int mask_smem_bytes = effective_batch * sizeof(int) * num_chunks * S_chunk / (num_warps * RDMA_ROW_PER_WARP); \
        if constexpr (USE_SEMAPHORES) {                                                                       \
            /* Hierarchical: per-(effective_batch) Phase 1 counters */                                        \
            /* Non-hierarchical: no counters needed (release by post-AG kernel) */                          \
            if (_flags.use_hierarchical) {                                                                        \
                cudaMemsetAsync(rank_empty_counters, 0, sizeof(int) * effective_batch * num_chunks, comm_stream); \
            }                                                                                               \
        }                                                                                                     \
        cudaMemsetAsync(block_work_ids, 0, sizeof(int) * _bitmap_region_size, comm_stream);                   \
        /* Compute stage-advanced dst (k_sr) and base src (local_k) pointers. */                              \
        /* Hierarchical: SR buffer is num_segments sub-tensors; each stage uses its own region. */            \
        /* Non-hierarchical: circular-reuse; k_sr stays at base, local_k at tail backup. */                   \
        const size_t _stage_elems = (size_t)num_chunks * S_chunk * H * D;                                     \
        KVType* const _k_sr_ptr = _flags.use_hierarchical                                                           \
            ? kv_buffer->k_data() + (size_t)segment_idx * B * _stage_elems                                    \
            : kv_buffer->k_data();                                                                            \
        KVType* const _v_sr_ptr = _flags.use_hierarchical                                                           \
            ? kv_buffer->v_data() + (size_t)segment_idx * B * _stage_elems                                    \
            : kv_buffer->v_data();                                                                            \
        const KVType* const _local_k_ptr = _flags.use_hierarchical ? kv_buffer->k_data() : local_k_data();          \
        const KVType* const _local_v_ptr = _flags.use_hierarchical ? kv_buffer->v_data() : local_v_data();          \
        if (segment_idx) {                                                                                    \
            if (_flags.use_hierarchical) {                                                                          \
                NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks, false, true,             \
                    start_pe, segment_idx, num_segs, mask_smem_bytes);                                        \
            } else {                                                                                          \
                NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks, false, false,            \
                    start_pe, segment_idx, num_segs, mask_smem_bytes);                                        \
            }                                                                                                 \
        } else {                                                                                              \
            constexpr int work_to_skip = S_chunk / (num_warps * RDMA_ROW_PER_WARP);                           \
            const int work_per_seg = num_chunks * work_to_skip;                                             \
            const int total_fill = effective_batch * work_to_skip;                                          \
            InitBitmapForLocalSkip<<<1, std::min(std::max(total_fill, 1), 256), 0, comm_stream>>>(         \
                block_work_ids, work_to_skip, work_per_seg, effective_batch);                                            \
            if (_flags.use_hierarchical) {                                                                          \
                NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks, true, true,              \
                    start_pe, segment_idx, num_segs, mask_smem_bytes);                                        \
            } else {                                                                                          \
                NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks, true, false,             \
                    start_pe, segment_idx, num_segs, mask_smem_bytes);                                        \
            }                                                                                                 \
        }                                                                                                     \
    } while(0)

    SChunkDispatch(SplittedAgBody, S_local);
#undef SplittedAgBody

    // Post-AG cleanup: notify producers we finished consuming their data in this segment.
    if constexpr (USE_SEMAPHORES) {
        if (_flags.use_hierarchical) {
            sema::ag::notify_segment_empty_hier(
                sema_inter(), sema_intra(),
                _my_pe, segment_idx, num_chunks,
                _total_n_pes, _gpus_per_node,
                effective_batch,
                comm_stream
            );
        } else {
            sema::ag::notify_segment_empty(
                kv_buffer->semaphores(),
                _my_pe, start_pe, num_chunks,
                _total_n_pes,
                comm_stream
            );
        }
    }
    cudaEventRecord(ag_done, comm_stream);
    WARN_PRINT_SYNC(comm_stream, "(%d) After run_overlap_splitted_ag_kernel, segment: %d\n", _my_pe, segment_idx);
}
#undef SparseLargeChunkSplittedKernel
#undef SeqlenCase

#define SegmentIdxPutKernelDispatch(_num_chunks, _has_local, _use_hier, seg_idx)     \
SparseLargeKVChunkRemotePutKernel<KVType, S_chunk, num_warps,                        \
            RDMA_ROW_PER_WARP, _num_chunks, _has_local, _use_hier>                   \
            <<<num_blocks, num_warps * 32, dynamic_smem, p_stream>>>(               \
        dkv_buffer->k_send(seg_idx),                                                \
        dkv_buffer->v_send(seg_idx),                                                \
        dkv_buffer->k_recv(seg_idx),                                                \
        dkv_buffer->v_recv(seg_idx),                                                \
        rs_cnt,                                                                     \
        commit_counters,                                                            \
        copy_chunk_mask,                                                            \
        _my_pe,                                                                     \
        remote_consumer_start_rank,                                                 \
        _total_n_pes,                                                               \
        seg_idx,                                                                    \
        B,                                                                          \
        H * D,                                                                      \
        dkv_buffer->semaphores(seg_idx),                                            \
        num_segments(),                                                             \
        _gpus_per_node,                                                             \
        _flags.per_stage_buffer                                                     \
    )

template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_rs_kernel(
    KVType* const dk_accum,
    KVType* const dv_accum,
    int segment_idx,
    cudaStream_t comp_stream
) {
    WARN_PRINT("(%d) Before run_overlap_rs_kernel, seg: %d\n", _my_pe, segment_idx);
    // Per-segment resource selection: multi-stream when available, else single-stream fallback
    cudaStream_t p_stream = _flags.per_stage_buffer ? put_streams[segment_idx] : aux_p_stream;
    int* const rs_cnt = _flags.per_stage_buffer ? (rs_block_cnt + segment_idx) : (block_cnt_semaphore + 1);
    int* const commit_counters = _flags.per_stage_buffer
        ? (rank_commit_counters + segment_idx * num_chunks) : rank_commit_counters;
    cudaEvent_t done_event = _flags.per_stage_buffer ? bwd_done_events[segment_idx] : bwd_done;

    // step 1 (pre-process and prepare) comp_stream should notify p_stream, post-process is done.
    // also, reset the block_cnt_semaphore for remote_put dynamic scheduling
    cudaMemsetAsync(rs_cnt, 0, sizeof(int), p_stream);
    cudaMemsetAsync(commit_counters, 0, sizeof(int) * num_chunks, p_stream);
    cudaEventRecord(done_event, comp_stream);
    cudaStreamWaitEvent(p_stream, done_event);
    cudaStreamWaitEvent(aux_c_stream, done_event);

    const int remote_consumer_start_rank = (_my_pe + num_chunks * segment_idx) % _total_n_pes;
    const int remote_producer_end_rank = (_my_pe - num_chunks * segment_idx + _total_n_pes) % _total_n_pes;

    // When capacity >= num_segments, each segment owns a dedicated buffer slot.
    // notify_consumer_empty is deferred to post-reduce, so that producer_wait_empty latency
    // is hidden by the full inter-BWD interval. constructor handles the first init.

    // When capacity < num_segments, slots are reused within a BWD pass.
    // Post-reduce notify would let a fast remote rank overwrite a reused slot's live data.
    // Must use the original pre-notify path: notify at stage start, before the put kernel.
    const bool post_reduce_notify = dkv_buffer_stage() == num_segments();

    if (!post_reduce_notify) {
        if (segment_idx) dkv_buffer->zero_recv_buf(segment_idx, aux_c_stream);
        if (_flags.use_hierarchical) {
            sema::rs::notify_consumer_empty_hier(
                dkv_buffer->semaphores(segment_idx),
                segment_idx, num_chunks,
                _total_n_pes, _gpus_per_node, _my_pe,
                aux_c_stream
            );
        } else {
            sema::rs::notify_consumer_empty(
                dkv_buffer->semaphores(segment_idx),
                remote_producer_end_rank,
                num_chunks, _total_n_pes, _my_pe,
                aux_c_stream
            );
        }
    }

    // step 2. local producer (put) wait empty (in the kernel) and start remote_put
#define RsKernelBody(_S_chunk)                                                                              \
    do {                                                                                                    \
        constexpr int S_chunk = _S_chunk;                                                                   \
        const int dynamic_smem = B * sizeof(int) * S_chunk * num_chunks / (num_warps * RDMA_ROW_PER_WARP);  \
        if (segment_idx) {                                                                                  \
            if (_flags.use_hierarchical) {                                                                  \
                NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, false, true, segment_idx);\
            } else {                                                                                        \
                NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, false, false, segment_idx); \
            }                                                                                           \
        } else {                                                                                        \
            const int S_stride = H * D;                                                                 \
            const int batch_stride = num_chunks * S_chunk * S_stride;                                   \
            KVType* const dk_dst = dkv_buffer->k_recv(0), *const dv_dst = dkv_buffer->v_recv(0);        \
            const KVType* const dk_src = dkv_buffer->k_send(0), *const dv_src = dkv_buffer->v_send(0);  \
            for (int batch_offset = 0, bid = 0; bid < B; bid ++, batch_offset += batch_stride) {        \
                cudaMemcpyAsync(dk_dst + batch_offset, dk_src + batch_offset,                           \
                    sizeof(KVType) * S_chunk * S_stride, cudaMemcpyDeviceToDevice, p_stream);           \
                cudaMemcpyAsync(dv_dst + batch_offset, dv_src + batch_offset,                           \
                    sizeof(KVType) * S_chunk * S_stride, cudaMemcpyDeviceToDevice, p_stream);           \
            }                                                                                           \
            cudaEventRecord(local_moved, p_stream);                                                     \
            cudaStreamWaitEvent(aux_c_stream, local_moved);                                             \
            if (_flags.use_hierarchical) {                                                                   \
                NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, true, true, segment_idx);  \
            } else {                                                                                         \
                NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, true, false, segment_idx); \
            }                                                                                           \
        }                                                                                               \
        dkv_buffer->release_buffer(segment_idx, p_stream);                                              \
        /* consumer wait full + reduce */                                                               \
        sema::rs::consumer_wait_full(                                                                   \
            dkv_buffer->semaphores(segment_idx),                                                        \
            _my_pe, segment_idx == 0 ? (num_chunks - 1) : num_chunks, aux_c_stream,                     \
            _flags.per_stage_buffer                                                                     \
        );                                                                                              \
        launch_dk_dv_reduce(                                                                            \
            dkv_buffer->k_recv(segment_idx),                                                            \
            dkv_buffer->v_recv(segment_idx),                                                            \
            dk_accum, dv_accum,                                                                         \
            B, S_chunk, H, D, num_chunks,                                                               \
            segment_idx == 0,                                                                           \
            aux_c_stream                                                                                \
        );                                                                                              \
        /* post-reduce notify: only when each segment has its own slot (capacity >= num_segments). */   \
        /* Zero the recv buffer first, then signal remote producers — safe because reduce already */    \
        /* consumed the data, and no other segment will touch this slot within this BWD pass. */        \
        if (post_reduce_notify) {                                                                       \
            dkv_buffer->zero_recv_buf(segment_idx, aux_c_stream);                                       \
            if (_flags.use_hierarchical) {                                                              \
                sema::rs::notify_consumer_empty_hier(                                                   \
                    dkv_buffer->semaphores(segment_idx),                                                \
                    segment_idx, num_chunks,                                                            \
                    _total_n_pes, _gpus_per_node, _my_pe,                                               \
                    aux_c_stream, true                                                                  \
                );                                                                                      \
            } else {                                                                                    \
                sema::rs::notify_consumer_empty(                                                        \
                    dkv_buffer->semaphores(segment_idx),                                                \
                    remote_producer_end_rank, num_chunks, _total_n_pes, _my_pe,                         \
                    aux_c_stream, true                                                                  \
                );                                                                                      \
            }                                                                                           \
        }                                                                                               \
    } while(0)

    SChunkDispatch(RsKernelBody, S_local);
#undef RsKernelBody

    WARN_PRINT_SYNC(aux_c_stream, "(%d) After run_overlap_rs_kernel, segment: %d\n", _my_pe, segment_idx);
}

#undef SegmentIdxPutKernelDispatch
#undef NumChunkDispatchSplitted

template <typename KVType>
int OverlapCommunicator<KVType>::dkv_buffer_stage() const {
    return _flags.per_stage_buffer ? num_segments() : RS_BUFFER_CAPACITY;
}

template <typename KVType>
int OverlapCommunicator<KVType>::seqlen_scale() const {
    if (dkv_buffer) {
        return num_chunks;
    } else {
        return _total_n_pes;
    }
}

template <typename KVType>
int OverlapCommunicator<KVType>::num_segments() const {
    if (dkv_buffer) {
        return _total_n_pes / num_chunks;
    } else {
        return 1;
    }
}

template <typename KVType>
int OverlapCommunicator<KVType>::overlap_sm_margin() const {
    return dkv_buffer ? OVERLAP_SM_MARGIN : 0;
}

template <typename KVType>
int OverlapCommunicator<KVType>::get_comm_rpb() const {
    return num_warps * RDMA_ROW_PER_WARP;
}

template <typename KVType>
void OverlapCommunicator<KVType>::prepare_dkv_buffer(cudaStream_t stream) {
    if (!dkv_buffer) return;

    const bool pre_reduce_notify = dkv_buffer_stage() < num_segments();
    if (pre_reduce_notify) {
        // capacity < num_segments: slots are reused within a BWD pass.
        // Only zero seg 0's recv buffer on comp_stream (original behavior).
        // notify_consumer_empty is done at stage start in run_overlap_rs_kernel.
        dkv_buffer->zero_recv_buf(0, stream);
    }
}

// ====== Dynamic Reconfiguration ======

template <typename KVType>
void OverlapCommunicator<KVType>::reallocate_block_work_ids() {
    // Free old allocation
    CUDA_DEBUG_CHECK(cudaFreeAsync(block_work_ids, comm_stream));
    CUDA_DEBUG_CHECK(cudaStreamSynchronize(comm_stream));

    _num_copy_chunks = B * S_local * _total_n_pes / (RDMA_ROW_PER_WARP * num_warps);
    // BHSD: bitmap indexed by effective_batch (B*H), needs more entries than num_copy_chunks
    const int bitmap_entries = _flags.use_bhsd_layout
        ? (B * H * S_local * _total_n_pes / (RDMA_ROW_PER_WARP * num_warps))
        : _num_copy_chunks;
    _bitmap_region_size = ((bitmap_entries + 1) + 3) & ~3;
    const int head_region_size = std::max(num_blocks, _bitmap_region_size);
    const int effective_batch_reconf = _flags.use_bhsd_layout ? B * H : B;
    const int rank_counters_size = _total_n_pes + effective_batch_reconf * _total_n_pes;
    CUDA_DEBUG_CHECK(cudaMallocAsync(&block_work_ids, sizeof(int) * (head_region_size + _num_copy_chunks + 4 + rank_counters_size), comm_stream));
    copy_chunk_mask = block_work_ids + head_region_size;
    block_cnt_semaphore = copy_chunk_mask + _num_copy_chunks;
    stream_coordinator = block_cnt_semaphore + STREAM_COORD_OFFSET;
    rank_commit_counters = block_cnt_semaphore + 4;
    rank_empty_counters = rank_commit_counters + _total_n_pes;
    if constexpr (USE_STREAM_COORD) {
        cudaMemsetAsync(stream_coordinator, 0, sizeof(int), comm_stream);
    }
}

template <typename KVType>
void OverlapCommunicator<KVType>::setup_dkv_buffer(bool need_rs, nvshmem_team_t cp_team) {
    size_t new_single_k_numel = _local_batch_stride * B;

    // Apply 1.5x headroom + 32-alignment (consistent with SRBuffer strategy)
    auto alloc_with_headroom = [](size_t needed) -> size_t {
        size_t alloc = static_cast<size_t>(needed * 1.5);
        return (alloc + 31) & ~static_cast<size_t>(31);
    };

    if (need_rs && !dkv_buffer) {
        // RS-overlap newly enabled: create aux streams, events, and dkv_buffer
        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        cudaStreamCreateWithPriority(&aux_p_stream, cudaStreamNonBlocking,
            std::min(greatest_priority + 1, least_priority));
        cudaStreamCreateWithPriority(&aux_c_stream, cudaStreamNonBlocking,
            std::min(greatest_priority + 1, least_priority));
        cudaEventCreateWithFlags(&bwd_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&reduce_done, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&local_moved, cudaEventDisableTiming);

        size_t alloc_numel = alloc_with_headroom(new_single_k_numel);
        dkv_buffer = std::make_unique<SepSRBuffer<KVType>>(
            alloc_numel, _total_n_pes, num_chunks, _flags.per_stage_buffer ? _total_n_pes / num_chunks : RS_BUFFER_CAPACITY, cp_team
        );
        dkv_buffer->initialize_buffer(_my_pe, _flags.per_stage_buffer);
        _dkv_single_k_numel_capacity = alloc_numel;
        _dkv_num_chunks = num_chunks;

        // Per-segment producer streams (per-stage buffering only)
        if (_flags.per_stage_buffer) {
            const int ns = _total_n_pes / num_chunks;
            _num_put_streams = ns;
            put_streams = new cudaStream_t[ns];
            bwd_done_events = new cudaEvent_t[ns];
            for (int i = 0; i < ns; i++) {
                cudaStreamCreateWithPriority(&put_streams[i], cudaStreamNonBlocking,
                    std::min(greatest_priority + 1, least_priority));
                cudaEventCreateWithFlags(&bwd_done_events[i], cudaEventDisableTiming);
            }
            CUDA_DEBUG_CHECK(cudaMalloc(&rs_block_cnt, sizeof(int) * ns));
            kv_buffer->team_bar();
            CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
            prime_rs_semaphores();
        }

        WARN_PRINT("[FlashMask Overlap] Reconfigure: created dkv_buffer, chunks: %d\n", num_chunks);

    } else if (need_rs && dkv_buffer) {
        // RS-overlap already exists: check if reallocation needed
        bool need_recreate = (new_single_k_numel > _dkv_single_k_numel_capacity)
                          || (num_chunks != _dkv_num_chunks);
        if (need_recreate) {
            size_t alloc_numel = alloc_with_headroom(new_single_k_numel);
            dkv_buffer->release_for_realloc();
            dkv_buffer = std::make_unique<SepSRBuffer<KVType>>(
                alloc_numel, _total_n_pes, num_chunks, _flags.per_stage_buffer ? _total_n_pes / num_chunks : RS_BUFFER_CAPACITY, cp_team
            );
            dkv_buffer->initialize_buffer(_my_pe, _flags.per_stage_buffer);
            _dkv_single_k_numel_capacity = alloc_numel;
            _dkv_num_chunks = num_chunks;

            // Recreate multi-stream resources if num_segments changed
            if (_flags.per_stage_buffer) {
                const int ns = _total_n_pes / num_chunks;
                if (ns != _num_put_streams) {
                    // Destroy old
                    for (int i = 0; i < _num_put_streams; i++) {
                        cudaStreamDestroy(put_streams[i]);
                        cudaEventDestroy(bwd_done_events[i]);
                    }
                    delete[] put_streams;
                    delete[] bwd_done_events;
                    CUDA_DEBUG_CHECK(cudaFree(rs_block_cnt));
                    // Create new
                    _num_put_streams = ns;
                    put_streams = new cudaStream_t[ns];
                    bwd_done_events = new cudaEvent_t[ns];
                    int lp, gp;
                    cudaDeviceGetStreamPriorityRange(&lp, &gp);
                    for (int i = 0; i < ns; i++) {
                        cudaStreamCreateWithPriority(&put_streams[i], cudaStreamNonBlocking,
                            std::min(gp + 1, lp));
                        cudaEventCreateWithFlags(&bwd_done_events[i], cudaEventDisableTiming);
                    }
                    CUDA_DEBUG_CHECK(cudaMalloc(&rs_block_cnt, sizeof(int) * ns));
                }
                // Prime per-stage semaphores after buffer recreation
                kv_buffer->team_bar();
                CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
                prime_rs_semaphores();
            }

            WARN_PRINT("[FlashMask Overlap] Reconfigure: recreated dkv_buffer, chunks: %d\n", num_chunks);
        }

    } else if (!need_rs && dkv_buffer) {
        // RS-overlap disabled: release dkv_buffer and aux resources
        if (_num_put_streams > 0) {
            for (int i = 0; i < _num_put_streams; i++) {
                cudaStreamDestroy(put_streams[i]);
                cudaEventDestroy(bwd_done_events[i]);
            }
            delete[] put_streams;
            delete[] bwd_done_events;
            CUDA_DEBUG_CHECK(cudaFree(rs_block_cnt));
            put_streams = nullptr;
            bwd_done_events = nullptr;
            rs_block_cnt = nullptr;
            _num_put_streams = 0;
        }
        CUDA_DEBUG_CHECK(cudaEventDestroy(bwd_done));
        CUDA_DEBUG_CHECK(cudaEventDestroy(reduce_done));
        CUDA_DEBUG_CHECK(cudaEventDestroy(local_moved));
        CUDA_DEBUG_CHECK(cudaStreamDestroy(aux_p_stream));
        CUDA_DEBUG_CHECK(cudaStreamDestroy(aux_c_stream));
        dkv_buffer->release_for_realloc();
        dkv_buffer.reset();
        _dkv_single_k_numel_capacity = 0;
        _dkv_num_chunks = 0;
        WARN_PRINT("[FlashMask Overlap] Reconfigure: destroyed dkv_buffer (RS-overlap disabled)\n");
    }
}

template <typename KVType>
void OverlapCommunicator<KVType>::prime_rs_semaphores() {
    const int num_stages = _total_n_pes / num_chunks;
    for (int seg = 0; seg < num_stages; seg++) {
        if (_flags.use_hierarchical) {
            sema::rs::notify_consumer_empty_hier(
                dkv_buffer->semaphores(seg), seg, num_chunks,
                _total_n_pes, _gpus_per_node, _my_pe, aux_c_stream, true);
        } else {
            const int rpe = (_my_pe - num_chunks * seg + _total_n_pes) % _total_n_pes;
            sema::rs::notify_consumer_empty(
                dkv_buffer->semaphores(seg), rpe, num_chunks, _total_n_pes, _my_pe, aux_c_stream, true);
        }
    }
}

template <typename KVType>
bool OverlapCommunicator<KVType>::reconfigure_if_needed(
    int new_b, int new_s_local, int new_h, int new_d,
    int rank, int nranks, int new_mask_head, 
    bool new_overlap_rs, const uint8_t* unique_id_ptr
) {
    OverlapConfig new_config = {new_b, new_s_local, new_h, new_mask_head, new_d, nranks, new_overlap_rs, _flags.use_bhsd_layout};

    if (new_config == _config) return false;  // No change needed

    // NVSHMEM bootstrap persists after finalize — re-init with different nranks is silently ignored.
    if (nranks != _total_n_pes && unique_id_ptr == nullptr) {
        std::cerr << "[FlashMask Overlap] FATAL: nranks changed from " + std::to_string(_total_n_pes) +
            " to " + std::to_string(nranks) +
            ". For this case, we need to finalize the NVSHMEM env and re-init" +
            ". So unique_id_ptr is required but currently not given.\n";
        throw std::invalid_argument(
            "[FlashMask Overlap] CP size change but unable to reinit correctly."
        );
    }

    // S_local change: validate the new value is in the supported dispatch set
    if (new_s_local != S_local) {
        if (new_s_local != 4096 && new_s_local != 8192 && new_s_local != 16384
            && new_s_local != 32768 && new_s_local != 65536 && new_s_local != 131072) {
            throw std::invalid_argument(
                "[FlashMask Overlap] S_local must be one of {4096, 8192, 16384, 32768, 65536, 131072}, got: "
                + std::to_string(new_s_local)
            );
        }
    }

    WARN_PRINT("[FlashMask Overlap] Reconfiguring: old=[%s], new=[%s]\n",
        _config.to_string().c_str(), new_config.to_string().c_str());

    // Synchronize all work before reconfiguring — no kernel should be using old buffers
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    if (nranks != _total_n_pes) {
        // NVSHMEM finalize and re-init
        nvshmem_finalize();
        WARN_PRINT("[FlashMask Overlap] Finalizing and re-init: old-CP: %d, new-CP: %d\n", _total_n_pes, nranks);
        init_distributed_environment(rank, nranks, _my_pe, _total_n_pes, unique_id_ptr);
    }

    // BHSD fallback check: if new B*H exceeds bitmask limit, disable BHSD
    if (_flags.use_bhsd_layout && _flags.use_hierarchical && new_b * new_h > 64) {
        printf("[FlashMask Overlap] Reconfigure: BHSD + hierarchical requires B*H <= 64, "
               "got B*H=%d. Falling back to BSHD layout.\n", new_b * new_h);
        _flags.use_bhsd_layout = false;
        new_config.use_bhsd = false;
    }

    // Update member variables
    B = new_b;
    S_local = new_s_local;
    H = new_h;
    H_mask = new_mask_head;
    D = new_d;
    num_chunks = get_num_chunk_per_segment(S_local, _total_n_pes, H);
    _local_batch_stride = new_config.cp_chunk_size();
    _total_numel = new_config.sr_buffer_numel();

    // SRBuffer: check if reallocation needed
    nvshmem_team_t cp_team = NVSHMEM_TEAM_WORLD;
    if (_total_numel > _sr_buffer_capacity) {
        // Allocate with headroom to reduce reallocation frequency
        size_t new_capacity = static_cast<size_t>(_total_numel * 1.5);
        new_capacity = (new_capacity + 31) & ~static_cast<size_t>(31);  // align to 32

        fprintf(stderr, "[BHSD DEBUG] Reconfigure: SR buffer realloc needed: %zu -> %zu (new B=%d, H=%d, S_local=%d)\n",
                _sr_buffer_capacity, new_capacity, B, H, S_local);
        kv_buffer->release_for_realloc();
        const int reconf_sema_count = USE_SEMAPHORES ? (_flags.use_hierarchical ? _num_nodes + _total_n_pes : _total_n_pes) : 0;
        kv_buffer = std::make_unique<SRBuffer<KVType>>(new_capacity, cp_team, reconf_sema_count);
        if constexpr (USE_SEMAPHORES) {
            cudaMemset(kv_buffer->semaphores(), 0, sizeof(int64_t) * reconf_sema_count);
        }
        WARN_PRINT("[FlashMask Overlap] Reconfigure: reallocated SRBuffer, capacity: %zu -> %zu\n",
            _sr_buffer_capacity, new_capacity);
        _sr_buffer_capacity = new_capacity;
        kv_buffer->team_bar();
    }

    // block_work_ids: depends on B (and H in BHSD mode for bitmap/rank_counters), reallocate when needed
    int new_num_copy_chunks = B * S_local * _total_n_pes / (RDMA_ROW_PER_WARP * num_warps);
    // BHSD mode: bitmap and rank_counters depend on B*H, so also reallocate when H changes
    bool need_realloc_block_work = (new_num_copy_chunks != _num_copy_chunks);
    if (_flags.use_bhsd_layout && !need_realloc_block_work) {
        // Check if effective_batch (B*H) changed — bitmap and rank_counters depend on it
        const int old_effective_batch = _config.B * _config.H;
        const int new_effective_batch = B * H;
        if (new_effective_batch != old_effective_batch) {
            need_realloc_block_work = true;
        }
    }
    if (need_realloc_block_work) {
        reallocate_block_work_ids();
    }

    // SepSRBuffer (dkv_buffer): handle RS-overlap state transitions
    setup_dkv_buffer(new_overlap_rs, cp_team);

    _config = new_config;

    WARN_PRINT("[FlashMask Overlap] Reconfigured: B=%d, S_local=%d, H=%d, D=%d, H_mask=%d, num_chunks=%d, overlap_rs=%d\n",
        B, S_local, H, D, H_mask, num_chunks, int(new_overlap_rs));

    return true;
}

// explicit instantiation and singleton management

template class OverlapCommunicator<cutlass::bfloat16_t>;
static std::unique_ptr<flashmask::OverlapCommunicator<cutlass::bfloat16_t>> overlap_comm = nullptr;

namespace comm {

OverlapCommunicator<cutlass::bfloat16_t>& init_singleton_instance(
    const cutlass::bfloat16_t* const k_data,
    const cutlass::bfloat16_t* const v_data,
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    const uint8_t* unique_id_ptr,
    int mask_head
) {
    // RS-overlap requires H_k >= rs_overlap_min_h_k AND nranks > 1.
    // num_chunks=1 (CP2 or S_local >= 32768) is now supported by the splitted kernels:
    // segment 0's local-only chunk triggers early return, and segment 1 processes the single remote chunk.
    int num_chunks_preview = get_num_chunk_per_segment(s_kv, nranks, h_kv);
    bool new_overlap_rs = (h_kv >= rs_overlap_min_h_k && nranks > 1);

    // Validate S_local is in the supported dispatch set (applies to both first-time and reconfigure)
    if (s_kv != 4096 && s_kv != 8192 && s_kv != 16384
        && s_kv != 32768 && s_kv != 65536 && s_kv != 131072) {
        throw std::invalid_argument(
            "[FlashMask Overlap] S_local must be one of {4096, 8192, 16384, 32768, 65536, 131072}, got: "
            + std::to_string(s_kv)
        );
    }

    if (!overlap_comm) {
        // First-time creation
        overlap_comm = std::make_unique<OverlapCommunicator<cutlass::bfloat16_t>>(
            k_data, v_data, b_kv, s_kv, h_kv, d_kv, rank, nranks,
            unique_id_ptr, mask_head, new_overlap_rs
        );
    } else {
        // Check if reconfiguration is needed (handles param change detection internally)
        overlap_comm->reconfigure_if_needed(
            b_kv, s_kv, h_kv, d_kv, rank, nranks, mask_head, new_overlap_rs, unique_id_ptr
        );
    }
    return *overlap_comm;
}

OverlapCommunicator<cutlass::bfloat16_t>& singleton() {
    return *overlap_comm;
}

bool is_singleton_null() {
    return overlap_comm == nullptr;
}

void destroy_singleton() {
    if (overlap_comm) {
        // Ensure all async work is complete before destroying
        CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        overlap_comm.reset();   // triggers destructor (which handles nvshmem_finalize if SHOULD_MANAGE_NVSHMEM)
        WARN_PRINT("[FlashMask Overlap] Singleton destroyed for topology refresh.\n");
    }
}

}   // namespace comm

}   // namespace flashmask

#undef SChunkCase
#undef SChunkDispatch
