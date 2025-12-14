#pragma once
#include <cuda_runtime.h>
#include "sr_buffer.cuh"
#include "cutlass/fast_math.h"

#define NVSHMEM_DEBUG

inline void CheckCudaErrorAux(const char *file, unsigned line,
                                       const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err),
           err, file, line);
    exit(1);
}

#if defined(NVSHMEM_DEBUG)
#define CUDA_DEBUG_CHECK(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
#else
#define CUDA_DEBUG_CHECK(value) do {} while(0)
#endif

/**
 * This kernel assumes num head is small, like <= 3, so that we directly copy
 * the entire head dimension, without the need to calculate index
 * 
 * @param column_mask tensor shape (B, 1, S, 2) for now, in the future when
 *      more masks need support, this can be easily extended
 * @param S_stride stride seqluence length: each warp is responsible for transfering
 *      one row. Passing D means we only transfer one row, passing H * D means we are
 *      transfering the same rows on different heads.
 * @param seqlen_offset Offset of the seqlen id. We start from the end of a CP chunk, for example:
 *      | ----------------CP chunk 2--------------- | ----------------CP chunk 3---------------- |
 *                                          <---- moves leftwards   ... W0B1 W3B0 W2B0 W1B0 W0B0
 *                                                 (seqlen_offset of PE id=1 in this CP group)  ^
 * @param work_per_warp Number of KV copies for each warp: H * (S - S / cp_size) / num_blocks / num_warps
 *      Since we don't need to remote get the locally available chunk (S / cp_size) 
 * 
 * gridDim.x: total number of rows to be copied / num_warps per block (4)
 * blockDim.x: number of warps (4) * 32
 * 
 * num_blocks & num_warps & S are made compile-time known to alleviate integer op pressure
 * 
 * Note that: using this kernel, we can either fetch the entire row on different heads (H * D)
 * or single head, depending on the `S_stride` and `work_per_warp`
*/
template <typename T, int S, int S_chunk, int num_blocks=64, int num_warps=4>
__global__ __launch_bounds__(128, 16) void SparseKVFewHeadRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    const int* const __restrict__ column_mask,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int num_batch,                // B
    const int S_stride                  // H * D
) {
    // assume num_head is even, so that we can transfer two rows per warp
    // and only need to read mask once (since heads reuse the same mask, 
    // even head num won't result in two rows in a warp mapped to different seq)
    constexpr int seqlen_stride = num_warps * num_blocks;
    const int warp_id = threadIdx.x >> 5;
    const int work_per_warp = num_batch * seqlen_offset / (num_blocks * num_warps);
    constexpr int seqlen_offset = S - S_chunk;

    for (int i = 0, seqlen_id = seqlen_offset - 1 - (blockIdx.x * num_warps + warp_id), batch_offset = 0;
        i < work_per_warp; 
        i++, seqlen_id -= seqlen_stride        // reverse traversal
    ) {
        // TODO(heqianyue): We only support LTS + UTE and multi-head shared mask currently, this should be extended
        if (seqlen_id < 0) {
            seqlen_id = seqlen_offset - 1 - (blockIdx.x * num_warps + warp_id);
            batch_offset += S;
        }
        // TODO(heqianyue): We need a simple way to set batch
        const int2 lts_ute = *(reinterpret_cast<const int2*>(&column_mask[batch_offset * 2 + seqlen_id * 2]));
        if (lts_ute.x > lts_ute.y) {         // mask does not cover the whole row of KV
            // actually, two calls can be merged
            int cp_chunk_id = S - 1 - seqlen_id / S_chunk;
            int remote_pe = my_pe - cp_chunk_id * cp_stride;
            remote_pe = remote_pe >= 0 ? remote_pe : remote_pe + total_n_pes; 
            const int dst_addr = (seqlen_id + batch_offset) * S_stride;
            // TODO(heqianyue): Check the following
            const int src_addr = (S - S_chunk + (seqlen_id % S_chunk) + batch_offset) * S_stride;
            // TODO(heqianyue): Check whether we should use blocking version -- we do not call get multiple times before sync
            nvshmemx_getmem_nbi_warp(
                        &k_sr[dst_addr],
                        &k_sr[src_addr],
                        S_stride * sizeof(T), remote_pe
            );
            nvshmemx_getmem_nbi_warp(
                    &v_sr[dst_addr],
                    &v_sr[src_addr],
                    S_stride * sizeof(T), remote_pe
            );
        }
        // ensures all transfer is completed. If we use `nvshmemx_getmem_warp` (blocking ver), the following line should be commented
        nvshmem_quiet();
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(wptr, num_warps);     // 4 rows are done, move the write ptr
    }
}


/**
 * This kernel does not copy an entire chunk of (B, S, H, D)
 * instead, it progresses firstly along `S`, then `H`, then `D`
 * with each warp responsible of copying two rows of D, for example:
 * 
 * warp 0 of block i copys (i, j, 2)-th row and (i, j, 3)-th row
 * S - S / cp_size must be a multiple of 256
*/
template <typename T, int S, int S_chunk, int D, int num_blocks=64, int num_warps=4>
__global__ __launch_bounds__(128, 16) void SparseKVMultiHeadRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    const int* const __restrict__ column_mask,
    const cutlass::FastDivmod head_divmod,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int num_head,                     // head / 2
    const int work_per_row,                 // S - S / cp_size
    const int total_work                    // B * (S - S / cp_size) * (H / 2)
) {
    const int warp_id = threadIdx.x >> 5;
    constexpr int work_stride = num_warps * num_blocks;
    // Use static scheduling (this might lead to workload imbalance)
    int head_id = 0, batch_id = 0;
    for (int work_id = 0, seqlen_id = 0; work_id < total_work; work_id ++, seqlen_id -= work_stride) {
        if ((work_id % work_per_row) == 0) {
            seqlen_id = work_per_row - 1 - (blockIdx.x * num_warps + warp_id);
            // TODO(heqianyue): If we don't have enough regs, we will forsake FastDivmod, 
            // since this divmod computation frequency is relatively low but reg-consuming
            batch_id = head_divmod.divmod(head_id, head_id + 2);
        }
        const int mask_id = (batch_id * S + seqlen_id) * 2;
        const int2 lts_ute = *(reinterpret_cast<const int2*>(&column_mask[mask_id]));
        if (lts_ute.x > lts_ute.y) {         // mask does not cover the whole row of KV
            // get the target PE to get from
            int cp_chunk_id = S - 1 - seqlen_id / S_chunk;
            int remote_pe = my_pe - cp_chunk_id * cp_stride;
            remote_pe = remote_pe >= 0 ? remote_pe : remote_pe + total_n_pes; 
            // TODO (heqianyue): can we simplify this? Integer ops are too heavy
            const int dst_addr = ((batch_id * S + seqlen_id) * num_head + head_id) * D;
            const int src_addr = ((batch_id * S_chunk + seqlen_id % S_chunk) * num_head + head_id) * D;
            // copy data on two different heads (K and V)
            nvshmemx_getmem_nbi_warp(
                        &k_sr[dst_addr],
                        &k_sr[src_addr],
                        D * sizeof(T) * 2, remote_pe
            );
            nvshmemx_getmem_nbi_warp(
                        &v_sr[dst_addr],
                        &v_sr[src_addr],
                        D * sizeof(T) * 2, remote_pe
            );
        }
        nvshmem_quiet();
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(wptr, num_warps);     // move the write ptr
    }
}


/**
 * SM-level overlapping communicator
 * 
 * An RAII object managing CP-groups / comm stream / buffer lifetimes automatically
 * 
 * Call the constructor of this communicator, and call `run_overlap_kernel` before
 * the main attention kernel, make sure the lifetime of the instance outlast the main kernel
 * Then you should be able to get async remote get, costing only 4 SMs
*/
template <typename KVType>
class OverlapCommunicator {
public:
    OverlapCommunicator(
        const KVType* const k_data,
        const KVType* const v_data,
        int b_kv,
        int s_kv,
        int h_kv,
        int d_kv,
        int cp_size,
        // Maybe we should manage the following by ourselves? Do not pass as parameters
        int nvshmem_my_pe,
        int num_pes
    ): kv_buffer(nullptr),
       _my_pe(nvshmem_my_pe),
       _total_n_pes(num_pes),
       _cp_size(cp_size),
       _cp_stride(num_pes / cp_size) {
        cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
        _cp_chunk_size = b_kv * s_kv * h_kv * d_kv;
        _total_numel = _cp_chunk_size * cp_size;             // won't overflow, but should be careful
        
        // This variable is simply a int32_t, so can be passed by value
        nvshmem_team_t cp_team = simple_collective_topology_setter(nvshmem_my_pe, _cp_stride);
        auto kv_buffer = std::make_unique<SRBuffer<KVType>>(_total_numel, cp_team);

        // copy to the last position of the SR buffer
        CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer.k_data() + (_total_numel - _cp_chunk_size), k_data, 
                                _cp_chunk_size * sizeof(KVType), cudaMemcpyDeviceToDevice, comm_stream));
        CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer.v_data() + (_total_numel - _cp_chunk_size), v_data, 
                                _cp_chunk_size * sizeof(KVType), cudaMemcpyDeviceToDevice, comm_stream));
    }

    ~OverlapCommunicator() {
        CUDA_DEBUG_CHECK(cudaStreamSynchronize(comm_stream));
        cudaStreamDestroy(comm_stream);
    }

    /**
     * run the overlap kernel asynchronously
     * @param S the seqlen of local K (for example, 32K full length, CP=4, local S=8K)
    */
    void run_overlap_kernel(const int* const column_mask, int B, int S, int H, int D) {
        const int work_per_row = S * (_cp_size - 1);
        constexpr int S_chunk = 8192;

#define MultiHeadKernel(_S, _D)                                                 \
    SparseKVMultiHeadRemoteGetKernel<KVType, _S, S_chunk, _D, num_blocks,       \
                    num_warps><<<num_blocks, num_warps * 4, 0, comm_stream>>>(  \
                        kv_buffer->k_data(),                                    \
                        kv_buffer->v_data(),                                    \
                        write_ptr,                                              \
                        cutlass::FastDivmod(H),                                 \
                        _my_pe,                                                 \
                        _cp_stride,                                             \
                        _total_n_pes,                                           \
                        H,                                                      \
                        work_per_row,                                           \
                        B * work_per_row * H / 2                                \
                    )

#define FewHeadKernel(_S)                                                   \
    SparseKVFewHeadRemoteGetKernel<KVType, _S, S_chunk, num_blocks,         \
                num_warps><<<num_blocks, num_warps * 4, 0, comm_stream>>>(  \
                        kv_buffer->k_data(),                                \
                        kv_buffer->v_data(),                                \
                        write_ptr,                                          \
                        _my_pe,                                             \
                        _cp_stride,                                         \
                        _total_n_pes,                                       \
                        B,                                                  \
                        H * D                                               \
                    )

#define SeqlenCase(MACRO_FUNC, _S, ...)         \
    case _S {                                   \
        MACRO_FUNC(_S, ##__VA_ARGS__); break;   \
    }

#define SeqlenDispatch(MACRO_FUNC, _S, ...)             \
    switch (_S * _cp_size) {                            \
        SeqlenCase(MACRO_FUNC, 131072, ##__VA_ARGS__)   \
        SeqlenCase(MACRO_FUNC, 32768, ##__VA_ARGS__)    \
    default:                                            \
        throw std::invalid_argument("Full seqlen must be 32K or 128K"); \
    }

        if (H > 4) {
            // `SparseKVMultiHeadRemoteGetKernel`
            if (d_kv == 128) {
                SeqlenDispatch(MultiHeadKernel, S, 128);
            } else if (d_kv == 80) {
                SeqlenDispatch(MultiHeadKernel, S, 80);
            } else if (d_kv == 64) {
                SeqlenDispatch(MultiHeadKernel, S, 64);
            } else {
                throw std::invalid_argument("Supported HeadDim is [64, 80, 128]");
            }
        } else {
            SeqlenDispatch(FewHeadKernel, S);
        }   
#undef SeqlenDispatch
#undef SeqlenCase
#undef FewHeadKernel
#undef MultiHeadKernel
    }

    int* write_ptr;
private:
    cudaStream_t comm_stream;
    const int _total_n_pes;
    const int _my_pe;
    const int _cp_size;
    const int _cp_stride;
    size_t _cp_chunk_size;
    size_t _total_numel;
    std::unique_ptr<SRBuffer<KVType>> kv_buffer;

    static constexpr int num_warp = 4;
    static constexpr int num_blocks = 64;   // 32 reg, 128 thread, one SM of H800 can hold 16 blocks, we use 4 SM

    static std::pair<int, int> get_id_and_stride(int cp_size) {
        return std::make_pair<int, int>(nvshmem_my_pe(), nvshmem_n_pes() / cp_size);
    }

    // returns the team and stride between teams
    static nvshmem_team_t simple_collective_topology_setter(int my_global_pe, int stride) {
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
                cp_group_size,       // Size (num PEs in new team)
                &config,
                config_mask,
                &cp_team
            );
            return cp_team;
        }
    }
};
