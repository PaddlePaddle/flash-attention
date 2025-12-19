#pragma once
#include <cuda_runtime.h>
#include "sr_buffer.cuh"
#include "fast_divmod.cuh"

namespace flashmask {

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
#define CUDA_DEBUG_CHECK(value) (value)
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
 * gridDim.x: total number of rows to be copied / num_warps per block (8)
 * blockDim.x: number of warps (8) * 32
 * 
 * num_blocks & num_warps & S are made compile-time known to alleviate integer op pressure
 * 
 * Note that: using this kernel, we can either fetch the entire row on different heads (H * D)
 * or single head, depending on the `S_stride` and `work_per_warp`
*/
template <typename T, int S, int S_chunk, int num_blocks=32, int num_warps=8>
__global__ __launch_bounds__(256, 8) void SparseKVFewHeadRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    const int* const __restrict__ lt_start_ptr,
    const int* const __restrict__ ut_end_ptr,
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
    constexpr int seqlen_offset = S - S_chunk;
    const int warp_id = threadIdx.x >> 5;
    const int work_per_warp = num_batch * seqlen_offset / (num_blocks * num_warps);

    for (int work_id = 1, seqlen_id = seqlen_offset - 1 - (blockIdx.x * num_warps + warp_id), batch_offset = 0;
        work_id <= work_per_warp; 
        work_id++, seqlen_id -= seqlen_stride        // reverse traversal
    ) {
        if (seqlen_id < 0) {
            seqlen_id = seqlen_offset - 1 - (blockIdx.x * num_warps + warp_id);
            batch_offset += S;
        }
        // TODO(heqianyue): We only support LTS + UTE and multi-head shared mask currently, this should be extended
        // the mask indices are rolled, so we can use the following simple load
        const int lts = lt_start_ptr[batch_offset + seqlen_id];
        const int ute = ut_end_ptr[batch_offset + seqlen_id];
        if (lts > ute) {         // mask does not cover the whole row of KV
            int cp_chunk_id = (S - 1 - seqlen_id) / S_chunk;
            int remote_pe = my_pe - cp_chunk_id * cp_stride;
            remote_pe = remote_pe >= 0 ? remote_pe : remote_pe + total_n_pes; 
            // batch_offset * S_stride (batch-address) + (S-S_chunk) * S_stride (stored in the last chunk) + (seqlen_id % S_chunk) * S_stride (within chunk address)
            const int src_addr = (S - S_chunk + (seqlen_id % S_chunk) + batch_offset) * S_stride;
            const int dst_addr = (seqlen_id + batch_offset) * S_stride;
            // actually, two calls can be merged
            // TODO(heqianyue): Double check whether we should use NBI API or not
            nvshmemx_getmem_warp(
                        &k_sr[dst_addr],
                        &k_sr[src_addr],
                        S_stride * sizeof(T), remote_pe
            );
            nvshmemx_getmem_warp(
                        &v_sr[dst_addr],
                        &v_sr[src_addr],
                        S_stride * sizeof(T), remote_pe
            );
        }
        // ensures all transfer is completed. If we use `nvshmemx_getmem_warp` (blocking ver), the following line should be commented
        // TODO(heqianyue): I really hope to change this API again in the future. For within node transfer, there seems to be no need to
        // use nbi API and call this `quite`, since the memory transfer is done via CUDA IPC using CUDA-core (dst[..]=src[..]). Also,
        // two `get` ops can be merged into one by 
        __syncthreads();
        // the following code ensures ordered write_ptr update without the need to sync_grid
        // using cooperative_group::this_grid().sync() is definitely correct, but can be costly
        // I don't want blocks to wait for each other
        if (threadIdx.x < 32) {
            // TODO(heqianyue): make this better by not using __syncthreads
            // make sure the work_idx update is visible to other blocks (in terms of cache state). Otherwise, use a __threadfence
            if (threadIdx.x == 0) atomicExch(&block_work_idx[blockIdx.x], work_id);
            int work_idx = threadIdx.x == blockIdx.x ? work_id : block_work_idx[threadIdx.x];
            work_idx = __reduce_min_sync(0xffffffff, work_idx);
            // if the status of all 32 blocks are 'work_idx = work_per_warp', this will mean we
            // finished transfering, just set the write ptr INT_MAX so that mainloop.load will
            // never need to check it again
            work_idx = work_idx == work_per_warp ? INT_MAX : (work_idx * num_warps * num_blocks);
            // reduce the slowest block (for example, 8 blocks, 1, 2, 1, 1, 2, 1, 1, 0) --->
            // there are two blocks still not finishing work_id = 0, wptr can not move
            if (threadIdx.x == 0)
                atomicMax(wptr, work_idx);     // 256 * work_idx, or INT_MAX
        }
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
template <typename T, int S, int S_chunk, int D, int num_blocks=32, int num_warps=8>
__global__ __launch_bounds__(256, 8) void SparseKVMultiHeadRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    const int* const __restrict__ lt_start_ptr,
    const int* const __restrict__ ut_end_ptr,
    const FastDivmod head_divmod,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int num_head,                     // head / 2
    const int total_work                    // B * (S - S / cp_size) * (H / 2)
) {
    const int warp_id = threadIdx.x >> 5;
    constexpr int work_per_row = S - S_chunk;
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
        const int mask_id = batch_id * S + seqlen_id;
        const int lts = lt_start_ptr[mask_id];
        const int ute = ut_end_ptr[mask_id];
        if (lts > ute) {         // mask does not cover the whole row of KV
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
        // check `FewHeadKernel` for explanation
        if (threadIdx.x < 32) {
            // make sure the work_idx update is visible to other blocks. Otherwise, use a __threadfence
            if (threadIdx.x == 0) atomicExch(&block_work_idx[blockIdx.x], work_id + 1);
            int work_idx = threadIdx.x == blockIdx.x ? work_id + 1 : block_work_idx[threadIdx.x];
            work_idx = __reduce_min_sync(0xFFFFFFFF, work_idx);
            // reduce the slowest block (for example, 8 blocks, -1, 0, 0, 1, 2, 0, -1, 0) --->
            // there are two blocks still not finishing work_id = 0, wptr can not move
            if (threadIdx.x == 0)
                atomicMax(wptr, work_idx * num_warps * num_blocks);     // 256 * work_idx
        }
    }
}

}   // namespace flashmask