#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "overlap_comm.cuh"
#include "remote_get_kernel.cuh"
#include "remote_put_kernel.cuh"
#include "dk_dv_reduce_bf16.cuh"
#include "cp_heuristic.cuh"

namespace flashmask {

// whether should we manually manage nvshmem related environment setups
// deprecation warning: will be removed in the future
static constexpr bool SHOULD_MANAGE_NVSHMEM = true;
// no team_bar but fine-grained signaling, good for single node CUDA IPC
static constexpr bool USE_SEMAPHORES = false;
// If true, we will scale up the transfering chunk for one CTA (16 rows per chunk ---> 256-512 rows per chunk)
// and calculate per-chunk sparsity to skip some comm (avg 50% of the chunks are not needed)
static constexpr bool USE_SPARSE_LARGE_CHUNK = true;
// whether to use stream coordinator to make sure the scheduling order of comm & comp kernels
// Note(heqianyue): If we can make sure CUDA_DEVICE_MAX_CONNECTION=1, stream_coord is actually not required
static constexpr bool USE_STREAM_COORD = true;
// whether to use double buffer for dK, dV SepSRBuffer, either 1 (single) or 2 (double buffer)
// Note(heqianyue): double buffering might be deprecated in the future
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
// WARNING: For very large S_local (131072) with high cp_size (16), S_full * H * D may approach
//          INT_MAX in kernel address calculations. Known limitation of the existing kernel code.

#define SChunkCase(MACRO_FUNC, _S_chunk, ...) \
    case _S_chunk: { MACRO_FUNC(_S_chunk, ##__VA_ARGS__); break; }

#define SChunkDispatch(MACRO_FUNC, _S_local, ...)                               \
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

// Direct NVSHMEM plugin .so files to FlashMask's own renamed copies.
// This ensures that when both FlashMask and DeepEP statically link libnvshmem.a,
// each side dlopen()s a distinct plugin file (different realpath = independent
// loaded instance), so the plugin's internal static variables are not shared.
// See: https://github.com/NVIDIA/nvshmem/issues/68
static void set_flashmask_nvshmem_plugin_env() {
    setenv("NVSHMEM_BOOTSTRAP_UID_PLUGIN", "flashmask_nvshmem_bootstrap_uid.so.3", 0);
    setenv("NVSHMEM_TRANSPORT_IBRC_PLUGIN", "flashmask_nvshmem_transport_ibrc.so.3", 0);
    setenv("NVSHMEM_TRANSPORT_IBGDA_PLUGIN", "flashmask_nvshmem_transport_ibgda.so.3", 0);
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
    set_flashmask_nvshmem_plugin_env();
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
    int cp_size,
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
   _cp_size(cp_size),
   num_chunks(get_num_chunk_per_segment(s_kv, nranks, h_kv)),
   block_work_ids(nullptr),
   block_cnt_semaphore(nullptr),
   copy_chunk_mask(nullptr)
{
    if constexpr (SHOULD_MANAGE_NVSHMEM) {
        init_distributed_environment(rank, nranks, _my_pe, _total_n_pes, unique_id_ptr);
    } else {
        get_nvshmem_info(_my_pe, _total_n_pes);     // get info if nvshmem is already avaliable
    }
    _cp_stride = _total_n_pes / cp_size;

    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    cudaStreamCreateWithPriority(&comm_stream, 
                                  cudaStreamNonBlocking,
                                  greatest_priority);
    cudaEventCreateWithFlags(&wptr_init, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&sr_usable, cudaEventDisableTiming);
    cudaEventRecord(sr_usable, comm_stream);                    // set initial status for SR buffer
    _cp_chunk_size = s_kv * h_kv * d_kv;
    _total_numel = _cp_chunk_size * b_kv * cp_size;             // won't overflow, but should be careful

    // This variable is simply a int32_t, so can be passed by value
    nvshmem_team_t cp_team = NVSHMEM_TEAM_WORLD;
    kv_buffer = std::make_unique<SRBuffer<KVType>>(_total_numel, cp_team, USE_SEMAPHORES ? _total_n_pes : 0);
    if constexpr (USE_SEMAPHORES) {
        cudaMemset(kv_buffer->semaphores(), 0, sizeof(int64_t) * _total_n_pes);
    }
    if constexpr (USE_SPARSE_LARGE_CHUNK) {
        const int num_copy_chunks = b_kv * s_kv * cp_size / (RDMA_ROW_PER_WARP * num_warps);
        // block_cnt_semaphore (AG overlap only requires only 1 extra int, 2 is for RS overlap, 2 more for padding)
        CUDA_DEBUG_CHECK(cudaMallocAsync(&block_work_ids, sizeof(int) * (num_blocks + num_copy_chunks + 4), comm_stream));
        copy_chunk_mask = block_work_ids + num_blocks;
        block_cnt_semaphore = copy_chunk_mask + num_copy_chunks;
        stream_coordinator = block_cnt_semaphore + STREAM_COORD_OFFSET;
    } else {
        CUDA_DEBUG_CHECK(cudaMallocAsync(&block_work_ids, sizeof(int) * (num_blocks + 4), comm_stream));    // +4 for stream coordinator and padding
        stream_coordinator = block_work_ids + num_blocks + STREAM_COORD_OFFSET;
    }
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
        dkv_buffer = std::make_unique<SepSRBuffer<KVType>>(
            _cp_chunk_size * b_kv,      // single chunk K numel (B * S_local * H * D)
            _total_n_pes,
            num_chunks,
            RS_BUFFER_CAPACITY,
            cp_team
        );
        // zero initialize the dkv semaphores
        dkv_buffer->reset_semaphores();
        printf("[FlashMask Overlap] Using RS-Overlap, buffer capacity: %d, num_chunks: %d\n", RS_BUFFER_CAPACITY, num_chunks);
    }
    kv_buffer->team_bar();
    // Initialize configuration tracking
    _config = {b_kv, s_kv, h_kv, mask_head, d_kv, cp_size, overlap_rs};
    _sr_buffer_capacity = _total_numel;
    _dkv_single_k_numel_capacity = overlap_rs ? (_cp_chunk_size * b_kv) : 0;
    _dkv_num_chunks = overlap_rs ? num_chunks : 0;
    if constexpr (USE_SPARSE_LARGE_CHUNK) {
        _num_copy_chunks = b_kv * s_kv * cp_size / (RDMA_ROW_PER_WARP * num_warps);
    } else {
        _num_copy_chunks = 0;
    }
    // copy to the last position of the SR buffer
    WARN_PRINT("SR buffer valid: %d, B, S, H, D: %d, %d, %d, %d, cp_size: %d, stride: %d\n", int(kv_buffer->is_valid()), B, S_local, H, D, cp_size, _cp_stride);
    printf("[FlashMask Overlap] constructor rank: %d, nranks: %d\n", rank, nranks);
}

template <typename KVType>
OverlapCommunicator<KVType>::~OverlapCommunicator() {
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    CUDA_DEBUG_CHECK(cudaFreeAsync(block_work_ids, comm_stream));
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    CUDA_DEBUG_CHECK(cudaEventDestroy(wptr_init));
    CUDA_DEBUG_CHECK(cudaEventDestroy(sr_usable));
    CUDA_DEBUG_CHECK(cudaStreamDestroy(comm_stream));
    kv_buffer->release();           // do not depend on auto-release
    if (dkv_buffer) {
        CUDA_DEBUG_CHECK(cudaEventDestroy(bwd_done));
        CUDA_DEBUG_CHECK(cudaEventDestroy(reduce_done));
        CUDA_DEBUG_CHECK(cudaEventDestroy(local_moved));
        CUDA_DEBUG_CHECK(cudaStreamDestroy(aux_p_stream));
        CUDA_DEBUG_CHECK(cudaStreamDestroy(aux_c_stream));
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
void OverlapCommunicator<KVType>::wait_sr_buffer_empty() {
    // TBH, since there is enough time between two attention call,
    // for an unsafe impl, we actually don't need to wait.
    // This would save some time, but not entirely safe if
    // the attention's workload is too too small. Yet currently
    // we haven't trigger unsafe problems even once
    cudaStreamWaitEvent(comm_stream, sr_usable);
    if constexpr (USE_SEMAPHORES) {
        WARN_PRINT("Before wait_self_empty\n");
        // block if computation that uses the local chunk is not finished
        // so that we won't corrupt the local buffer being used
        sema::ag::wait_self_empty(
            kv_buffer->semaphores(),
            _my_pe,
            comm_stream
        );
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
    const int local_offset = fwd ? (_cp_chunk_size * (_cp_size - 1)) : 0;
    // for bwd RS-overlap splitted AG, the batch stride is num_chunks * S_local * S_stride
    int batch_stride = _cp_chunk_size * _cp_size;
    // FIXME(heqianyue): Note that for RS-overlap, batch size > 1 is currently not correct!
    // there is a mismatch between RS-overlap stride and SR buffer stride.
    if (fwd == false && dkv_buffer) {
        batch_stride = _cp_chunk_size * num_chunks;
        const size_t copy_bytes = B * _cp_chunk_size * sizeof(KVType);
        // save local KV chunks at the end of SR buffer
        CUDA_DEBUG_CHECK(cudaMemcpyAsync(local_k_data(), new_k_data, copy_bytes, 
                                cudaMemcpyDeviceToDevice, comm_stream));
        CUDA_DEBUG_CHECK(cudaMemcpyAsync(local_v_data(), new_v_data, copy_bytes,
                                cudaMemcpyDeviceToDevice, comm_stream));
    }
    for (int bid = 0; bid < B; bid++) {
        CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->k_data() + local_offset + bid * batch_stride,
                                new_k_data + bid * _cp_chunk_size, 
                                _cp_chunk_size * sizeof(KVType), 
                                cudaMemcpyDeviceToDevice, 
                                comm_stream));
        CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->v_data() + local_offset + bid * batch_stride,
                                new_v_data + bid * _cp_chunk_size, 
                                _cp_chunk_size * sizeof(KVType),
                                cudaMemcpyDeviceToDevice, 
                                comm_stream));
    }
    if constexpr (USE_SEMAPHORES) {
        // notify all other PEs that the local data is ready (1. set self to be `total_pes - 1`. 2. broadcast to other PEs)
        sema::ag::notify_full(
            kv_buffer->semaphores(),
            _my_pe, _total_n_pes, 
            kv_buffer->team(), comm_stream
        );
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
// S==S_chunk (cp_size=1) combinations generated by SChunkDispatch × SeqlenDispatch.
// These are unreachable at runtime (overlap requires cp_size >= 2, so S_full >= 2*S_chunk).
// Eliminating them saves ~40% of kernel binary from combinatorial expansion.

#define DenseFewHeadKernel(_S, self_rank)                                           \
    if constexpr ((_S) > S_chunk) {                                                 \
        DenseKVFewHeadRemoteGetKernel<KVType, _S, S_chunk, num_blocks,              \
            num_warps, USE_SEMAPHORES><<<num_blocks, num_warps * 32, 0, comm_stream>>>( \
                            kv_buffer->k_data(),                                    \
                            kv_buffer->v_data(),                                    \
                            write_ptr,                                              \
                            block_work_ids,                                         \
                            self_rank,                                              \
                            _cp_stride,                                             \
                            _total_n_pes,                                           \
                            B,                                                      \
                            H * D,                                                  \
                            kv_buffer->semaphores()                                 \
                        );                                                          \
    }


#define SparseLargeChunkKernel(_S, bwd)                                             \
    if constexpr ((_S) > S_chunk) {                                                 \
        SparseLargeKVChunkRemoteGetKernel<KVType, _S, S_chunk,                      \
            num_warps, RDMA_ROW_PER_WARP, USE_STREAM_COORD, USE_SEMAPHORES, bwd>    \
            <<<num_blocks, num_warps * 32, 0, comm_stream>>>(                       \
                            kv_buffer->k_data(),                                    \
                            kv_buffer->v_data(),                                    \
                            write_ptr,                                              \
                            block_work_ids,                                         \
                            block_cnt_semaphore,                                    \
                            stream_coordinator,                                     \
                            copy_chunk_mask,                                        \
                            _my_pe,                                                 \
                            _total_n_pes,                                           \
                            B,                                                      \
                            H * D,                                                  \
                            kv_buffer->semaphores()                                 \
                        );                                                          \
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
            "[FlashMask Overlap] Full seqlen (S_local * cp_size) must be one of " \
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
    if constexpr (USE_SPARSE_LARGE_CHUNK) {
        constexpr int num_reduce_warp = RDMA_ROW_PER_WARP == 16 ? 4 : num_warps;
#define CallBlockSparsityKernel(S_chunk, grid, skip_local)                                            \
    BlockSparsityCheckSpecializedKernel<S_chunk, num_reduce_warp, RDMA_ROW_PER_WARP, skip_local>     \
                <<< grid, num_reduce_warp * 32, 0, stream >>>(lt_start_ptr, ut_end_ptr, copy_chunk_mask, H_mask, head_stride)

#define ComputeChunkMaskBody(_S_chunk)                                                                \
    do {                                                                                               \
        constexpr int S_chunk = _S_chunk;                                                              \
        const int head_stride = S_local * _cp_size;                                                    \
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
}

template <typename KVType>
int OverlapCommunicator<KVType>::chunk_per_seg() const {
    // TODO(heqianyue): this is fixed for now, we can make this
    // dynamic in the future (for example, for 32K CP4, chunk might be 2)
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
    S = S_local * _cp_size;
    // set 0 every time we start the comm kernel --- meaning that we don't have any available attn-blocks
    cudaMemsetAsync(block_work_ids, 0, sizeof(int) * num_blocks, comm_stream);

    WARN_PRINT_SYNC(comm_stream, "Before remote get kernel: %d\n", _my_pe);

#define AgKernelBody(_S_chunk)                                                      \
    do {                                                                             \
        constexpr int S_chunk = _S_chunk;                                            \
        const int _S_full = S_local * _cp_size;                                      \
        if constexpr (USE_SPARSE_LARGE_CHUNK) {                                      \
            if (fwd) {                                                               \
                SeqlenDispatch(SparseLargeChunkKernel, _S_full, false);              \
            } else {                                                                 \
                SeqlenDispatch(SparseLargeChunkKernel, _S_full, true);               \
            }                                                                        \
        } else {                                                                     \
            if (fwd) {                                                               \
                SeqlenDispatch(DenseFewHeadKernel, _S_full, _my_pe);                 \
            } else {                                                                 \
                SeqlenDispatch(DenseFewHeadKernel, _S_full, _my_pe);                 \
            }                                                                        \
        }                                                                            \
    } while(0)

    SChunkDispatch(AgKernelBody, S_local);
#undef AgKernelBody

    WARN_PRINT_SYNC(comm_stream, "After remote_get kernel\n");
    if constexpr (USE_SEMAPHORES) {
        WARN_PRINT("Before notify_all_empty kernel\n");
        sema::ag::notify_all_empty(
            kv_buffer->semaphores(),
            _my_pe,
            _total_n_pes,
            comm_stream
        );
        WARN_PRINT_SYNC(comm_stream, "After notify_all_empty kernel\n");
    }
}
#undef SeqlenDispatch
#undef DenseFewHeadKernel
#undef SparseLargeChunkKernel

#define SparseLargeChunkSplittedKernel(num_chunk, start_rank, seg_idx, num_segs, smem_bytes)\
    SparseLargeKVChunkSplittedRemoteGetKernel<KVType, S_chunk,                              \
        num_warps, RDMA_ROW_PER_WARP, num_chunk, USE_STREAM_COORD, USE_SEMAPHORES>          \
        <<<num_blocks, num_warps * 32, smem_bytes, comm_stream>>>(                          \
                        kv_buffer->k_data(),                                                \
                        kv_buffer->v_data(),                                                \
                        local_k_data(),                                                     \
                        local_v_data(),                                                     \
                        write_ptr,                                                          \
                        block_work_ids,                                                     \
                        block_cnt_semaphore,                                                \
                        stream_coordinator,                                                 \
                        copy_chunk_mask,                                                    \
                        start_rank,                                                         \
                        seg_idx,                                                            \
                        _total_n_pes,                                                       \
                        B,                                                                  \
                        H * D,                                                              \
                        num_segs,                                                           \
                        kv_buffer->semaphores()                                             \
                    )

#define NumChunkDispatchSplitted(MACRO_FUNC, num_chunk, ...)    \
    switch (num_chunk) {                                        \
        case 4: { MACRO_FUNC(4, __VA_ARGS__); break; }          \
        case 2: { MACRO_FUNC(2, __VA_ARGS__); break; }          \
        case 3: { MACRO_FUNC(3, __VA_ARGS__); break; }          \
        case 1: { MACRO_FUNC(1, __VA_ARGS__); break; }          \
        case 8: { MACRO_FUNC(8, __VA_ARGS__); break; }          \
        case 7: { MACRO_FUNC(7, __VA_ARGS__); break; }          \
    default:                                                    \
        throw std::invalid_argument("Num chunk per segment must be in [2, 4, 8]. [1, 3, 7] cases considers local chunk skipping. Other segment size is not supported."); \
    }

template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_splitted_ag_kernel(
    int* const write_ptr,
    int& S,
    int segment_idx
) {
    if constexpr (!USE_SPARSE_LARGE_CHUNK) {
        throw std::runtime_error("Overlap splitted AG kernel must use sparse large chunk.");
    }
    WARN_PRINT("(%d) Before run_overlap_splitted_ag_kernel, segment: %d\n", _my_pe, segment_idx);
    const int start_pe = (_my_pe + segment_idx * num_chunks) % _total_n_pes;
    const int num_segs = _cp_size / num_chunks;

#define SplittedAgBody(_S_chunk)                                                                       \
    do {                                                                                                \
        constexpr int S_chunk = _S_chunk;                                                               \
        const int mask_smem_bytes = B * sizeof(int) * num_chunks * S_chunk / (num_warps * RDMA_ROW_PER_WARP); \
        if (segment_idx) {                                                                              \
            cudaMemsetAsync(block_work_ids, 0, sizeof(int) * num_blocks, comm_stream);                  \
            NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks,                         \
                start_pe, segment_idx, num_segs, mask_smem_bytes);                                      \
        } else {                                                                                        \
            constexpr int work_to_skip = S_chunk / (num_warps * RDMA_ROW_PER_WARP);                     \
            sema::rs::SetValueKernel<<<1, num_blocks, 0, comm_stream>>>(block_work_ids, work_to_skip);  \
            NumChunkDispatchSplitted(SparseLargeChunkSplittedKernel, num_chunks - 1,                     \
                start_pe, segment_idx, num_segs, mask_smem_bytes);                                      \
        }                                                                                               \
    } while(0)

    SChunkDispatch(SplittedAgBody, S_local);
#undef SplittedAgBody

    if constexpr (USE_SEMAPHORES) {
        WARN_PRINT("Before notify_all_empty kernel (split AG)\n");
        sema::ag::notify_segment_empty(
            kv_buffer->semaphores(),
            _my_pe,
            start_pe,
            num_chunks,
            _total_n_pes,
            comm_stream
        );
        WARN_PRINT_SYNC(comm_stream, "After notify_all_empty kernel (split AG)\n");
    }
    WARN_PRINT_SYNC(comm_stream, "(%d) After run_overlap_splitted_ag_kernel, segment: %d\n", _my_pe, segment_idx);
}
#undef SparseLargeChunkSplittedKernel
#undef SeqlenCase

#define SegmentIdxPutKernelDispatch(_num_chunks, seg_idx)                   \
SparseLargeKVChunkRemotePutKernel<KVType, S_chunk, num_warps,               \
            RDMA_ROW_PER_WARP, _num_chunks>                                 \
            <<<num_blocks, num_warps * 32, dynamic_smem, aux_p_stream>>>(   \
        dkv_buffer->k_send(seg_idx),                                        \
        dkv_buffer->v_send(seg_idx),                                        \
        dkv_buffer->k_recv(seg_idx),                                        \
        dkv_buffer->v_recv(seg_idx),                                        \
        block_cnt_semaphore + 1,                                            \
        copy_chunk_mask,                                                    \
        _my_pe,                                                             \
        remote_consumer_start_rank,                                         \
        _total_n_pes,                                                       \
        seg_idx,                                                            \
        B,                                                                  \
        H * D,                                                              \
        dkv_buffer->semaphores(seg_idx),                                    \
        num_segments()                                                      \
    )

template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_rs_kernel(
    KVType* const dk_accum,
    KVType* const dv_accum,
    int segment_idx,
    cudaStream_t comp_stream
) {
    WARN_PRINT("(%d) Before run_overlap_rs_kernel, seg: %d\n", _my_pe, segment_idx);
    // step 1 (pre-process and prepare) comp_stream should notify aux_p_stream, post-process is done.
    // also, reset the block_cnt_semaphore for remote_put dynamic scheduling
    cudaMemsetAsync(block_cnt_semaphore + 1, 0, sizeof(int), aux_p_stream);
    cudaEventRecord(bwd_done, comp_stream);
    cudaStreamWaitEvent(aux_p_stream, bwd_done);
    cudaStreamWaitEvent(aux_c_stream, bwd_done);

    // Note(heqianyue): remote_producer: who is putting data to us. remote_consumer: who should we put data to.
    // for remote producer, it is easier to compute the last rank.
    const int remote_producer_end_rank = (_my_pe - num_chunks * segment_idx) % _total_n_pes;
    const int remote_consumer_start_rank = (_my_pe + num_chunks * segment_idx) % _total_n_pes;

    // local consumer clears recv buffer so that places where remote_put kernel does not put data to are clean
    // set local state: local consumer needs data (this prevents dk_dv reducer starts before other remote ranks
    // put their data). Also, notify remote producer after setting local state, so that they won't change our local
    // state before we've set it, which might cause deadlock hang). After the following function call:
    // local consumer (recv_buffer and reduce kernel) is ready to wait for the buffer to be full
    if (segment_idx) dkv_buffer->zero_recv_buf(segment_idx, aux_c_stream);
    sema::rs::notify_consumer_empty(
        dkv_buffer->semaphores(segment_idx),
        remote_producer_end_rank,
        num_chunks, _cp_size, _my_pe,
        aux_c_stream
    );                          // local consumer

    // step 2. local producer (put) wait empty (in the kernel) and start remote_put
#define RsKernelBody(_S_chunk)                                                                         \
    do {                                                                                                \
        constexpr int S_chunk = _S_chunk;                                                               \
        const int dynamic_smem = B * sizeof(int) * S_chunk * num_chunks / (num_warps * RDMA_ROW_PER_WARP); \
        if (segment_idx) {                                                                              \
            NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks, segment_idx);              \
        } else {                                                                                        \
            const int S_stride = H * D;                                                                 \
            const int batch_stride = num_chunks * S_chunk * S_stride;                                   \
            KVType* const dk_dst = dkv_buffer->k_recv(0), *const dv_dst = dkv_buffer->v_recv(0);       \
            const KVType* const dk_src = dkv_buffer->k_send(0), *const dv_src = dkv_buffer->v_send(0); \
            for (int batch_offset = 0, bid = 0; bid < B; bid ++, batch_offset += batch_stride) {        \
                cudaMemcpyAsync(dk_dst + batch_offset, dk_src + batch_offset,                           \
                    sizeof(KVType) * S_chunk * S_stride, cudaMemcpyDeviceToDevice, aux_p_stream);       \
                cudaMemcpyAsync(dv_dst + batch_offset, dv_src + batch_offset,                           \
                    sizeof(KVType) * S_chunk * S_stride, cudaMemcpyDeviceToDevice, aux_p_stream);       \
            }                                                                                           \
            cudaEventRecord(local_moved, aux_p_stream);                                                 \
            cudaStreamWaitEvent(aux_c_stream, local_moved);                                             \
            NumChunkDispatchSplitted(SegmentIdxPutKernelDispatch, num_chunks - 1, segment_idx);          \
        }                                                                                               \
        dkv_buffer->release_buffer(segment_idx, aux_p_stream);                                          \
        nvshmemx_quiet_on_stream(aux_p_stream);                                                         \
        /* step 3. local producer notifies remote consumer: put done. Can start reduce. */               \
        sema::rs::producer_commit_all(                                                                  \
            dkv_buffer->semaphores(segment_idx),                                                        \
            remote_consumer_start_rank,                                                                 \
            _cp_size, _my_pe,                                                                           \
            num_chunks,                                                                                 \
            aux_p_stream                                                                                \
        );                                                                                              \
        /* step 4. the local rank consumer (reduce) wait full */                                         \
        sema::rs::consumer_wait_full(                                                                   \
            dkv_buffer->semaphores(segment_idx),                                                        \
            _my_pe, aux_c_stream                                                                        \
        );                                                                                              \
        /* step 5. zero-copy reduce */                                                                   \
        launch_dk_dv_reduce(                                                                            \
            dkv_buffer->k_recv(segment_idx),                                                            \
            dkv_buffer->v_recv(segment_idx),                                                            \
            dk_accum, dv_accum,                                                                         \
            B, S_chunk, H, D, num_chunks,                                                               \
            segment_idx == 0,                                                                           \
            aux_c_stream                                                                                \
        );                                                                                              \
    } while(0)

    SChunkDispatch(RsKernelBody, S_local);
#undef RsKernelBody

    // the end state should be: semaphore is all zero
    WARN_PRINT_SYNC(aux_c_stream, "(%d) After run_overlap_rs_kernel.\n", _my_pe);
    // the next post process should wait for the current reduce to be done
    // otherwise, the local dk/v_send will risk being overwritten by the next post-process
}

#undef SegmentIdxPutKernelDispatch
#undef NumChunkDispatchSplitted

template <typename KVType>
int OverlapCommunicator<KVType>::dkv_buffer_stage() const {
    return RS_BUFFER_CAPACITY;
}

template <typename KVType>
int OverlapCommunicator<KVType>::seqlen_scale() const {
    if (dkv_buffer) {
        return num_chunks;
    } else {
        return _cp_size;
    }
}

template <typename KVType>
int OverlapCommunicator<KVType>::num_segments() const {
    if (dkv_buffer) {
        return _cp_size / num_chunks;
    } else {
        return 1;
    }
}

template <typename KVType>
int OverlapCommunicator<KVType>::overlap_sm_margin() const {
    return dkv_buffer ? OVERLAP_SM_MARGIN : 0;
}

template <typename KVType>
void OverlapCommunicator<KVType>::prepare_dkv_buffer(cudaStream_t stream) {
    if (!dkv_buffer) return;
    // we need to use compute stream to zero the recv buffer
    // to make sure the buffer is zero-ed before producer/consumer starts
    dkv_buffer->zero_recv_buf(0, stream);
}

// ====== Dynamic Reconfiguration ======

template <typename KVType>
void OverlapCommunicator<KVType>::reallocate_block_work_ids() {
    // Free old allocation
    CUDA_DEBUG_CHECK(cudaFreeAsync(block_work_ids, comm_stream));
    CUDA_DEBUG_CHECK(cudaStreamSynchronize(comm_stream));

    if constexpr (USE_SPARSE_LARGE_CHUNK) {
        _num_copy_chunks = B * S_local * _cp_size / (RDMA_ROW_PER_WARP * num_warps);
        CUDA_DEBUG_CHECK(cudaMallocAsync(&block_work_ids, sizeof(int) * (num_blocks + _num_copy_chunks + 4), comm_stream));
        copy_chunk_mask = block_work_ids + num_blocks;
        block_cnt_semaphore = copy_chunk_mask + _num_copy_chunks;
        stream_coordinator = block_cnt_semaphore + STREAM_COORD_OFFSET;
    } else {
        _num_copy_chunks = 0;
        CUDA_DEBUG_CHECK(cudaMallocAsync(&block_work_ids, sizeof(int) * (num_blocks + 4), comm_stream));
        stream_coordinator = block_work_ids + num_blocks + STREAM_COORD_OFFSET;
    }
    if constexpr (USE_STREAM_COORD) {
        cudaMemsetAsync(stream_coordinator, 0, sizeof(int), comm_stream);
    }
}

template <typename KVType>
void OverlapCommunicator<KVType>::setup_dkv_buffer(bool need_rs, nvshmem_team_t cp_team) {
    size_t new_single_k_numel = _cp_chunk_size * B;

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
            alloc_numel, _total_n_pes, num_chunks, RS_BUFFER_CAPACITY, cp_team
        );
        dkv_buffer->reset_semaphores();
        _dkv_single_k_numel_capacity = alloc_numel;
        _dkv_num_chunks = num_chunks;
        WARN_PRINT("[FlashMask Overlap] Reconfigure: created dkv_buffer, chunks: %d\n", num_chunks);

    } else if (need_rs && dkv_buffer) {
        // RS-overlap already exists: check if reallocation needed
        bool need_recreate = (new_single_k_numel > _dkv_single_k_numel_capacity)
                          || (num_chunks != _dkv_num_chunks);
        if (need_recreate) {
            size_t alloc_numel = alloc_with_headroom(new_single_k_numel);
            dkv_buffer->release_for_realloc();
            dkv_buffer = std::make_unique<SepSRBuffer<KVType>>(
                alloc_numel, _total_n_pes, num_chunks, RS_BUFFER_CAPACITY, cp_team
            );
            dkv_buffer->reset_semaphores();
            _dkv_single_k_numel_capacity = alloc_numel;
            _dkv_num_chunks = num_chunks;
            WARN_PRINT("[FlashMask Overlap] Reconfigure: recreated dkv_buffer, chunks: %d\n", num_chunks);
        }

    } else if (!need_rs && dkv_buffer) {
        // RS-overlap disabled: release dkv_buffer and aux resources
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
bool OverlapCommunicator<KVType>::reconfigure_if_needed(
    int new_b, int new_s_local, int new_h, int new_d,
    int rank, int nranks, int new_mask_head, 
    bool new_overlap_rs, const uint8_t* unique_id_ptr
) {
    OverlapConfig new_config = {new_b, new_s_local, new_h, new_mask_head, new_d, nranks, new_overlap_rs};

    if (new_config == _config) return false;  // No change needed

    // FATAL: cp_size change is NVSHMEM-unsafe.
    // NVSHMEM bootstrap persists after finalize — re-init with different nranks is silently ignored.
    if (nranks != _cp_size && unique_id_ptr == nullptr) {
        std::cerr << "[FlashMask Overlap] FATAL: cp_size changed from " + std::to_string(_cp_size) +
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

    if (nranks != _cp_size) {
        // NVSHMEM finalize and re-init
        nvshmem_finalize();
        WARN_PRINT("[FlashMask Overlap] Finalizing and re-init: old-CP: %d, new-CP: %d\n", _total_n_pes, nranks);
        init_distributed_environment(rank, nranks, _my_pe, _total_n_pes, unique_id_ptr);
    }

    // Update member variables
    B = new_b;
    S_local = new_s_local;
    H = new_h;
    H_mask = new_mask_head;
    D = new_d;
    num_chunks = get_num_chunk_per_segment(S_local, _cp_size, H);
    _cp_chunk_size = new_config.cp_chunk_size();
    _total_numel = new_config.sr_buffer_numel();

    // SRBuffer: check if reallocation needed
    nvshmem_team_t cp_team = NVSHMEM_TEAM_WORLD;
    if (_total_numel > _sr_buffer_capacity) {
        // Allocate with headroom to reduce reallocation frequency
        size_t new_capacity = static_cast<size_t>(_total_numel * 1.5);
        new_capacity = (new_capacity + 31) & ~static_cast<size_t>(31);  // align to 32

        kv_buffer->release_for_realloc();
        kv_buffer = std::make_unique<SRBuffer<KVType>>(new_capacity, cp_team, USE_SEMAPHORES ? _total_n_pes : 0);
        if constexpr (USE_SEMAPHORES) {
            cudaMemset(kv_buffer->semaphores(), 0, sizeof(int64_t) * _total_n_pes);
        }
        WARN_PRINT("[FlashMask Overlap] Reconfigure: reallocated SRBuffer, capacity: %zu -> %zu\n",
            _sr_buffer_capacity, new_capacity);
        _sr_buffer_capacity = new_capacity;
        kv_buffer->team_bar();
    }

    // block_work_ids: depends on B, so always reallocate when B changes
    int new_num_copy_chunks = USE_SPARSE_LARGE_CHUNK
        ? (B * S_local * _cp_size / (RDMA_ROW_PER_WARP * num_warps)) : 0;
    if (new_num_copy_chunks != _num_copy_chunks) {
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
    int cp_size,
    const uint8_t* unique_id_ptr,
    int mask_head
) {
    // RS-overlap requires H_k >= 4, nranks > 1, AND num_chunks > 1.
    // When num_chunks=1 (S_local >= 32768, or cp_size=2), the splitted AG/RS dispatch
    // has no valid code path for the degenerate single-chunk case. Fall back to AG-only overlap.
    int num_chunks_preview = get_num_chunk_per_segment(s_kv, nranks, h_kv);
    bool new_overlap_rs = (h_kv >= rs_overlap_min_h_k && nranks > 1 && num_chunks_preview > 1);

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
            cp_size, unique_id_ptr, mask_head, new_overlap_rs
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
