#include <mpi.h>            // deprecate note
#include <iostream>
#include "overlap_comm.cuh"
#include "nvshmem_handle.h"
#include "remote_get_kernel.cuh"
#include "cutlass/bfloat16.h"

namespace flashmask {

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
    int& n_pes
) {
    printf("[FlashMask Overlap] Initializing NVSHMEM... Rank: %d / %d, PE ID: %d / %d\n", rank, nranks, my_pe, n_pes);
    std::vector<uint8_t> unique_id_val;
    if (rank == 0) {
        unique_id_val = UniqueIdFileSync::generate_and_write_unique_id(rank);
    } else {
        unique_id_val = UniqueIdFileSync::wait_and_read_unique_id(rank);
    }
    init_with_unique_id(std::move(unique_id_val), rank, nranks);
    get_nvshmem_info(my_pe, n_pes);
    printf("[FlashMask Overlap] NVSHMEM initialized. Rank: %d / %d, PE ID: %d / %d\n", rank, nranks, my_pe, n_pes);
}

void finalize_distributed_environment() {
    UniqueIdFileSync::clean_up_file();
    nvshmem_finalize();
    printf("[FlashMask Overlap] NVSHMEM env finalized.\n");
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
    int cp_size
    // Maybe we should manage the following by ourselves? Do not pass as parameters
): write_ptr(nullptr),
   kv_buffer(nullptr),
   B(b_kv),
   S_local(s_kv),
   H(h_kv),
   D(d_kv),
   _cp_size(cp_size)
{
    if constexpr (SHOULD_MANAGE_NVSHMEM) {
        init_distributed_environment(rank, nranks, _my_pe, _total_n_pes);
    } else {
        get_nvshmem_info(_my_pe, _total_n_pes);     // get info if nvshmem is already avaliable
    }
    _cp_stride = _total_n_pes / cp_size;

    cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&wptr_init, cudaEventDisableTiming);
    _cp_chunk_size = b_kv * s_kv * h_kv * d_kv;
    _total_numel = _cp_chunk_size * cp_size;             // won't overflow, but should be careful
    
    // This variable is simply a int32_t, so can be passed by value
    nvshmem_team_t cp_team = simple_collective_topology_setter(_my_pe, _cp_stride, _total_n_pes);
    auto kv_buffer = std::make_unique<SRBuffer<KVType>>(_total_numel, cp_team);

    // copy to the last position of the SR buffer
    update_kv_buffer(k_data, v_data);
    CUDA_DEBUG_CHECK(cudaMallocAsync(&block_work_ids, sizeof(int) * num_blocks, comm_stream));
    printf("[FlashMask Overlap] constructor rank: %d, nranks: %d\n", rank, nranks);
}

template <typename KVType>
OverlapCommunicator<KVType>::~OverlapCommunicator() {
    CUDA_DEBUG_CHECK(cudaStreamSynchronize(comm_stream));
    CUDA_DEBUG_CHECK(cudaFreeAsync(block_work_ids, comm_stream));
    cudaEventDestroy(wptr_init);
    cudaStreamDestroy(comm_stream);
    if constexpr (SHOULD_MANAGE_NVSHMEM) {
        finalize_distributed_environment();
    }
}

template <typename KVType>
void OverlapCommunicator<KVType>::update_kv_buffer(
    const KVType* const new_k_data,
    const KVType* const new_v_data
) {
    CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->k_data() + (_total_numel - _cp_chunk_size), new_k_data, 
                        _cp_chunk_size * sizeof(KVType), cudaMemcpyDeviceToDevice, comm_stream));
    CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer->v_data() + (_total_numel - _cp_chunk_size), new_v_data,
                        _cp_chunk_size * sizeof(KVType), cudaMemcpyDeviceToDevice, comm_stream));
}

/**
 * run the overlap kernel asynchronously
 * @param S the seqlen of local K (for example, 32K full length, CP=4, local S=8K)
*/
template <typename KVType>
void OverlapCommunicator<KVType>::run_overlap_kernel(
    const int* const lt_start_ptr,
    const int* const ut_end_ptr,
    int& S
) {
    constexpr int S_chunk = 8192;
    if (S_chunk != S_local) {
        throw std::runtime_error("Local KV seqlen should be equal to S-chunk size (8192).");
    }
    S = S_local * _cp_size;
    // set 0 every time we start the comm kernel --- meanning that we don't have any available attn-blocks
    cudaMemsetAsync(block_work_ids, 0, sizeof(int) * num_blocks, comm_stream);

#define MultiHeadKernel(_S, _D)                                                 \
    SparseKVMultiHeadRemoteGetKernel<KVType, _S, S_chunk, _D, num_blocks,       \
                    num_warps><<<num_blocks, num_warps * 32, 0, comm_stream>>>( \
                        kv_buffer->k_data(),                                    \
                        kv_buffer->v_data(),                                    \
                        write_ptr,                                              \
                        block_work_ids,                                         \
                        lt_start_ptr,                                           \
                        ut_end_ptr,                                             \
                        flashmask::FastDivmod(H),                               \
                        _my_pe,                                                 \
                        _cp_stride,                                             \
                        _total_n_pes,                                           \
                        H,                                                      \
                        B * (_S - S_chunk) * H / 2                              \
                    )

#define FewHeadKernel(_S)                                                   \
    SparseKVFewHeadRemoteGetKernel<KVType, _S, S_chunk, num_blocks,         \
                num_warps><<<num_blocks, num_warps * 32, 0, comm_stream>>>( \
                        kv_buffer->k_data(),                                \
                        kv_buffer->v_data(),                                \
                        write_ptr,                                          \
                        block_work_ids,                                     \
                        lt_start_ptr,                                       \
                        ut_end_ptr,                                         \
                        _my_pe,                                             \
                        _cp_stride,                                         \
                        _total_n_pes,                                       \
                        B,                                                  \
                        H * D                                               \
                    )

#define SeqlenCase(MACRO_FUNC, _S, ...)          \
    case _S: {                                   \
        MACRO_FUNC(_S, ##__VA_ARGS__); break;    \
    }

#define SeqlenDispatch(MACRO_FUNC, _S, ...)             \
    switch (_S) {                                       \
        SeqlenCase(MACRO_FUNC, 131072, ##__VA_ARGS__)   \
        SeqlenCase(MACRO_FUNC, 32768, ##__VA_ARGS__)    \
    default:                                            \
        throw std::invalid_argument("Full seqlen must be 32K or 128K"); \
    }

    // Note(heqianyue): input `S` for the following macros are full length, be careful
    if (H <= 4) {
        SeqlenDispatch(FewHeadKernel, S);
    } else {
        if (D == 128) {
            SeqlenDispatch(MultiHeadKernel, S, 128);
        } else if (D == 80) {
            SeqlenDispatch(MultiHeadKernel, S, 80);
        } else if (D == 64) {
            SeqlenDispatch(MultiHeadKernel, S, 64);
        } else {
            throw std::invalid_argument("Supported HeadDim is [64, 80, 128]");
        }
    }   
#undef SeqlenDispatch
#undef SeqlenCase
#undef FewHeadKernel
#undef MultiHeadKernel
}

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

// explicit instantiation
template class OverlapCommunicator<cutlass::bfloat16_t>;

}   // namespace flashmask