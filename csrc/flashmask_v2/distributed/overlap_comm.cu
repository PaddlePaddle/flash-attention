#include <iostream>
#include "overlap_comm.cuh"
#include "nvshmem_handle.h"
#include "remote_get_kernel_specialized.cuh"

namespace flashmask {

// whether should we manually manage nvshmem related environment setups
// deprecation warning: will be removed in the future
static constexpr bool SHOULD_MANAGE_NVSHMEM = true;
static constexpr bool USE_DENSE_COPY = true;      // no sparse mask KV skipping
static constexpr bool USE_SEMAPHORES = true;      // no team_bar but fine-grained signaling

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
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
    // TODO(heqianyue): Do we need to bar here?
    nvshmem_barrier_all();
}

void init_distributed_environment(
    int rank,
    int nranks,
    int& my_pe, 
    int& n_pes,
    const uint8_t* unique_id_ptr
) {
    DEBUG_PRINT("[FlashMask Overlap] Initializing NVSHMEM... Rank: %d / %d, PE ID: %d / %d\n", rank, nranks, my_pe, n_pes);
    std::vector<uint8_t> unique_id_val;
    if (unique_id_ptr == nullptr) {         
        // TODO(heqianyue): deprecate in the future, we do not allow local shared file
        if (rank == 0) {
            unique_id_val = UniqueIdFileSync::generate_and_write_unique_id(rank);
        } else {
            unique_id_val = UniqueIdFileSync::wait_and_read_unique_id(rank);
        }
    } else {
        unique_id_val.resize(sizeof(nvshmemx_uniqueid_t));
        std::memcpy(unique_id_val.data(), unique_id_ptr, sizeof(nvshmemx_uniqueid_t));
    }
    init_with_unique_id(std::move(unique_id_val), rank, nranks);
    get_nvshmem_info(my_pe, n_pes);
    DEBUG_PRINT("[FlashMask Overlap] NVSHMEM initialized. Rank: %d / %d, PE ID: %d / %d\n", rank, nranks, my_pe, n_pes);
    UniqueIdFileSync::clean_up_file();
}

void finalize_distributed_environment() {
    DEBUG_PRINT("[FlashMask Overlap] Finalizing...\n");
    nvshmem_finalize();
    DEBUG_PRINT("[FlashMask Overlap] NVSHMEM env finalized.\n");
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
    const uint8_t* unique_id_ptr
    // Maybe we should manage the following by ourselves? Do not pass as parameters
): kv_buffer(nullptr),
   B(b_kv),
   S_local(s_kv),
   H(h_kv),
   D(d_kv),
   _cp_size(cp_size)
{
    if constexpr (SHOULD_MANAGE_NVSHMEM) {
        init_distributed_environment(rank, nranks, _my_pe, _total_n_pes, unique_id_ptr);
    } else {
        get_nvshmem_info(_my_pe, _total_n_pes);     // get info if nvshmem is already avaliable
    }
    _cp_stride = _total_n_pes / cp_size;

    // TODO(heqianyue): cudaStreamCreateWithPriority
    cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&wptr_init, cudaEventDisableTiming);
    _cp_chunk_size = b_kv * s_kv * h_kv * d_kv;
    _total_numel = _cp_chunk_size * cp_size;             // won't overflow, but should be careful

    // This variable is simply a int32_t, so can be passed by value
    nvshmem_team_t cp_team = simple_collective_topology_setter(_my_pe, _cp_stride, _total_n_pes);
    kv_buffer = std::make_unique<SRBuffer<KVType>>(_total_numel, cp_team, USE_SEMAPHORES ? _total_n_pes : 0);
    if constexpr (USE_SEMAPHORES) {
        cudaMemset(kv_buffer->semaphores(), 0, sizeof(int) * _total_n_pes);
    }
    CUDA_DEBUG_CHECK(cudaMallocAsync(&block_work_ids, sizeof(int) * num_blocks, comm_stream));
    kv_buffer->team_bar();
    // copy to the last position of the SR buffer
    WARN_PRINT("SR buffer valid: %d, B, S, H, D: %d, %d, %d, %d, cp_size: %d, stride: %d\n", int(kv_buffer->is_valid()), B, S_local, H, D, cp_size, _cp_stride);
    update_kv_buffer(k_data, v_data);
    WARN_PRINT("[FlashMask Overlap] constructor rank: %d, nranks: %d\n", rank, nranks);
}

template <typename KVType>
OverlapCommunicator<KVType>::~OverlapCommunicator() {
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    CUDA_DEBUG_CHECK(cudaFreeAsync(block_work_ids, comm_stream));
    CUDA_DEBUG_CHECK(cudaDeviceSynchronize());
    CUDA_DEBUG_CHECK(cudaEventDestroy(wptr_init));
    CUDA_DEBUG_CHECK(cudaStreamDestroy(comm_stream));
    kv_buffer->release();           // do not depend on auto-release
    if constexpr (SHOULD_MANAGE_NVSHMEM && MANUAL_CLEANUP) {
        finalize_distributed_environment();
    }
}

template <typename KVType>
void OverlapCommunicator<KVType>::wait_wptr_init() {
    cudaStreamWaitEvent(comm_stream, wptr_init);
}

template <typename KVType>
void OverlapCommunicator<KVType>::wait_sr_buffer_empty() {
    // TBH, since there is enough time between two attention call,
    // for an unsafe impl, we actually don't need to wait.
    // This would save some time, but not entirely safe if
    // the attention's workload is too too small. Yet currently
    // we haven't trigger unsafe problems even once
    if constexpr (USE_SEMAPHORES) {
        WARN_PRINT("Before wait_self_empty\n");
        sema::wait_self_empty(
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
    WARN_PRINT("Before cudaMemcpyAsync...\n");
    // bwd copies the data to the start chunk of the SR, while fwd copies to the last chunk
    const int local_offset = fwd ? _total_numel - _cp_chunk_size : 0;
    CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->k_data() + local_offset, new_k_data, 
                            _cp_chunk_size * sizeof(KVType), cudaMemcpyDeviceToDevice, comm_stream));
    CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->v_data() + local_offset, new_v_data,
                            _cp_chunk_size * sizeof(KVType), cudaMemcpyDeviceToDevice, comm_stream));
    if constexpr (USE_SEMAPHORES) {
        // notify all other PEs that the local data is ready (1. set self to be `total_pes - 1`. 2. broadcast to other PEs)
        sema::notify_full(
            kv_buffer->semaphores(),
            _my_pe, _total_n_pes, 
            kv_buffer->team(), comm_stream
        );
    } else {
        // bar, so that comm_stream will finish transfering data before starting to get data from remote
        // this can be slow if the pace difference of PEs is large  
        kv_buffer->team_bar_on_stream(comm_stream);
    }
    // sync, otherwise it might hang unexceptedly
    CUDA_DEBUG_CHECK(cudaStreamSynchronize(comm_stream));
    WARN_PRINT(comm_stream, "After cudaMemcpyAsync and notify full\n");
}

#define DenseFewHeadKernel(_S, KernelTraits)                                        \
    DenseKVFewHeadRemoteGet##KernelTraits##Kernel<KVType, _S, S_chunk, num_blocks,  \
        num_warps, USE_SEMAPHORES><<<num_blocks, num_warps * 32, 0, comm_stream>>>( \
                        kv_buffer->k_data(),                                        \
                        kv_buffer->v_data(),                                        \
                        write_ptr,                                                  \
                        block_work_ids,                                             \
                        _my_pe,                                                     \
                        _cp_stride,                                                 \
                        _total_n_pes,                                               \
                        B,                                                          \
                        H * D,                                                      \
                        kv_buffer->semaphores()                                     \
                    )

#define FewHeadKernel(_S, KernelTraits)                                             \
    SparseKVFewHeadRemoteGet##KernelTraits##Kernel<KVType, _S, S_chunk, num_blocks, \
        num_warps, USE_SEMAPHORES><<<num_blocks, num_warps * 32, 0, comm_stream>>>( \
                        kv_buffer->k_data(),                                        \
                        kv_buffer->v_data(),                                        \
                        write_ptr,                                                  \
                        block_work_ids,                                             \
                        lt_start_ptr,                                               \
                        ut_end_ptr,                                                 \
                        _my_pe,                                                     \
                        _cp_stride,                                                 \
                        _total_n_pes,                                               \
                        B,                                                          \
                        H * D,                                                      \
                        kv_buffer->semaphores()                                     \
                    )

#define SeqlenCase(MACRO_FUNC, _S, KernelTraits, ...)       \
    case _S: {                                              \
        MACRO_FUNC(_S, KernelTraits, ##__VA_ARGS__); break; \
    }

#define SeqlenDispatch(MACRO_FUNC, _S, ...)                 \
    switch (_S) {                                           \
        SeqlenCase(MACRO_FUNC, 32768, ##__VA_ARGS__)        \
        SeqlenCase(MACRO_FUNC, 131072, ##__VA_ARGS__)       \
    default:                                                \
        throw std::invalid_argument("Full seqlen must be 32K or 128K"); \
    }

// can be adjust to use non-specialized version
#define FwdKernelTrait Specialized
#define BwdKernelTrait SpecializedBwd

/**
 * run the overlap kernel asynchronously
 * @param S the seqlen of local K (for example, 32K full length, CP=4, local S=8K)
*/
template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_kernel(
    const int* const lt_start_ptr,
    const int* const ut_end_ptr,
    int* const write_ptr,
    int& S,
    const bool fwd
) {
    constexpr int S_chunk = 8192;
    if (S_chunk != S_local) {
        throw std::runtime_error("Local KV seqlen should be equal to S-chunk size (8192).");
    }
    S = S_local * _cp_size;
    // set 0 every time we start the comm kernel --- meanning that we don't have any available attn-blocks
    cudaMemsetAsync(block_work_ids, 0, sizeof(int) * num_blocks, comm_stream);

    WARN_PRINT_SYNC(comm_stream, "Before remote get kernel\n");
    // Note(heqianyue): input `S` for the following macros are full length, be careful
    
    // TODO(heqianyue): in the previous versions (before commit 7fe9e3f), we support multi-head case (head > 4)
    // but since we don't need it **yet**, the support is removed. So, in the current state,
    // make sure the head of KV is no more than 4. Multi-head kernels are removed temporarily.
    if constexpr (USE_DENSE_COPY) {
        if (fwd) {
            SeqlenDispatch(DenseFewHeadKernel, S, FwdKernelTrait);
        } else {
            SeqlenDispatch(DenseFewHeadKernel, S, BwdKernelTrait);
        }
    } else {
        if (fwd) {
            SeqlenDispatch(FewHeadKernel, S, FwdKernelTrait);
        } else {
            SeqlenDispatch(FewHeadKernel, S, BwdKernelTrait);
        }
    }

    WARN_PRINT_SYNC(comm_stream, "After remote_get kernel\n");
    if constexpr (USE_SEMAPHORES) {
        // after this kernel, other PEs will know we have finished using their data.
        // also, the local value will be reset to 0 before we notify other PEs.
        // **comm_stream** itself does the reset (to not-ready state)
        WARN_PRINT("Before notify_all_empty kernel\n");
        sema::notify_all_empty(
            kv_buffer->semaphores(),
            _my_pe,
            _total_n_pes,
            comm_stream
        );
        WARN_PRINT_SYNC(comm_stream, "After notify_all_empty kernel\n");
    }
}
#undef SeqlenDispatch
#undef SeqlenCase
#undef FewHeadKernel
#undef MultiHeadKernel

// returns the team and stride between teams
template <typename KVType>
nvshmem_team_t OverlapCommunicator<KVType>::simple_collective_topology_setter(int my_global_pe, int stride, int n_pes) {
    if (stride == 1) {       
        // the launch world only has 4 GPUs, we consider this a single node use case
        // the world group is the CP communication group
        return NVSHMEM_TEAM_WORLD; 
    } else {
        // for example: 2 node (4, 4, 16, 4) ---> comm group [0, 4, 8, 12], stride 4
        // 4 node （8, 4, 32, 4） ---> comm group [0, 8, 16, 24], stride 8
        int my_group_start_pe = my_global_pe % stride;

        nvshmem_team_t cp_team;
        nvshmem_team_config_t config; // Default config
        long config_mask = 0;

        int status = nvshmem_team_split_strided(
            NVSHMEM_TEAM_WORLD,  // Parent team
            my_group_start_pe,   // Start PE (rank in parent)
            stride,              // Stride
            n_pes / stride,      // Size (num PEs in new team)
            &config,
            config_mask,
            &cp_team
        );
        return cp_team;
    }
}

// explicit instantiation and singleton management

template class OverlapCommunicator<cutlass::bfloat16_t>;
static std::unique_ptr<flashmask::OverlapCommunicator<cutlass::bfloat16_t>> overlap_comm = nullptr;

namespace comm {

void init_singleton_instance(
    const cutlass::bfloat16_t* const k_data,
    const cutlass::bfloat16_t* const v_data,
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    int cp_size,
    const uint8_t* unique_id_ptr
) {
    if (!overlap_comm) {
        overlap_comm = std::make_unique<OverlapCommunicator<cutlass::bfloat16_t>>(
            k_data, v_data, b_kv, s_kv, h_kv, d_kv, rank, nranks, cp_size, unique_id_ptr
        );
    }
}

OverlapCommunicator<cutlass::bfloat16_t>& singleton() {
    return *overlap_comm;
}

bool is_singleton_null() {
    return overlap_comm == nullptr;
}

}   // namespace comm

}   // namespace flashmask
