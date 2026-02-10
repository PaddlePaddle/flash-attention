#pragma once
#include <cuda_runtime.h>
#include "sr_buffer.cuh"
#include "nvshmem_copy_utils.cuh"
#include "ag_semaphore_ops.cuh"
#include "debug_logger.cuh"

namespace flashmask {

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
template <typename T, int S, int S_chunk, int num_blocks=32, int num_warps=8, bool use_semaphore=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseKVFewHeadRemoteGetKernel(
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
    const int S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    // assume num_head is even, so that we can transfer two rows per warp
    // and only need to read mask once (since heads reuse the same mask, 
    // even head num won't result in two rows in a warp mapped to different seq)
    constexpr int seqlen_stride = num_warps * num_blocks;
    constexpr int seqlen_offset = S - S_chunk;
    const int warp_id = threadIdx.x >> 5;
    const int work_per_warp = num_batch * seqlen_offset / seqlen_stride;

    // When `use_semaphore` is true, there is no choice but to cache the GMEM semaphore onto SMEM
    // for faster loading. There is no way to cache in the register file, since we are using
    // dynamic indexing, local array will inevitablely lead to local memory (GMEM) reads
    __shared__ int cached_semaphores[16];

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

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
            if constexpr (use_semaphore) {
                if ((threadIdx.x & 31) == 0 && cached_semaphores[remote_pe] == 0) {
                    sema::ag::wait_full(semaphores, remote_pe);
                    cached_semaphores[remote_pe] = 1;
                    // no need to sync, since `two_buffers_getmem_<scope>` will sync 
                }
            }
            // batch_offset * S_stride (batch-address) + (S-S_chunk) * S_stride (stored in the last chunk) + (seqlen_id % S_chunk) * S_stride (within chunk address)
            const int src_addr = (S - S_chunk + (seqlen_id % S_chunk) + batch_offset) * S_stride;
            const int dst_addr = (seqlen_id + batch_offset) * S_stride;
            shmem::two_buffers_getmem_warp(
                k_sr + dst_addr,
                v_sr + dst_addr,
                k_sr + src_addr,
                v_sr + src_addr,
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


template <typename T, int S, int S_chunk, int num_blocks=32, int num_warps=8, bool use_semaphore=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) DenseKVFewHeadRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    // assume num_head is even, so that we can transfer two rows per warp
    // and only need to read mask once (since heads reuse the same mask, 
    // even head num won't result in two rows in a warp mapped to different seq)
    constexpr int row_per_block = num_warps * 2;     // 16-line per-block (* 2), if (* 1) 8-line per-block
    constexpr int seqlen_stride = row_per_block * num_blocks;
    constexpr int seqlen_offset = S - S_chunk;
    const int work_per_block = num_batch * seqlen_offset / seqlen_stride;
    __shared__ int cached_semaphores[16];

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    for (int work_id = 1, seqlen_id = seqlen_offset - (blockIdx.x + 1) * row_per_block, batch_offset = 0;
        work_id <= work_per_block; 
        work_id++, seqlen_id -= seqlen_stride        // reverse traversal
    ) {
        if (seqlen_id < 0) {
            seqlen_id = seqlen_offset - (blockIdx.x + 1) * row_per_block;
            batch_offset += S;
        }
        int cp_chunk_id = (S - 1 - seqlen_id) / S_chunk;
        int remote_pe = my_pe - cp_chunk_id * cp_stride;
        remote_pe = remote_pe >= 0 ? remote_pe : remote_pe + total_n_pes; 
        // batch_offset * S_stride (batch-address) + (S-S_chunk) * S_stride (stored in the last chunk) + (seqlen_id % S_chunk) * S_stride (within chunk address)
        if constexpr (use_semaphore) {
            if ((threadIdx.x & 31) == 0 && cached_semaphores[remote_pe] == 0) {
                sema::ag::wait_full(semaphores, remote_pe);
                cached_semaphores[remote_pe] = 1;
                // no need to sync, since `two_buffers_getmem_<scope>` will sync 
            }
        }
        const int src_addr = (S - S_chunk + (seqlen_id % S_chunk) + batch_offset) * S_stride;
        const int dst_addr = (seqlen_id + batch_offset) * S_stride;
        shmem::two_buffers_getmem_block(
            k_sr + dst_addr,
            v_sr + dst_addr,
            k_sr + src_addr,
            v_sr + src_addr,
            row_per_block * S_stride * sizeof(T), remote_pe
        );
        // No need to __syncthreads again, since in `getmem_block` we will be calling __syncthreads at the end
        if (threadIdx.x < 32) {
            if (threadIdx.x == 0) atomicExch(&block_work_idx[blockIdx.x], work_id);
            int work_idx = threadIdx.x == blockIdx.x ? work_id : block_work_idx[threadIdx.x];
            work_idx = __reduce_min_sync(0xffffffff, work_idx);
            work_idx = work_idx == work_per_block ? INT_MAX : (work_idx * seqlen_stride);
            if (threadIdx.x == 0)
                atomicMax(wptr, work_idx);     // 128 * work_idx, or INT_MAX
        }
    }
}

// ================================== comm kernels used in bwd =======================================
// the fwd kernels get data in reverse, while the backward does not

template <typename T, int S, int S_chunk, int num_blocks=32, int num_warps=8, bool use_semaphore=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseKVFewHeadRemoteGetBwdKernel(
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
    const int S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    constexpr int seqlen_stride = num_warps * num_blocks;
    const int warp_id = threadIdx.x >> 5;
    const int work_per_warp = num_batch * (S - S_chunk) / seqlen_stride;

    __shared__ int cached_semaphores[16];

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    for (int work_id = 1, seqlen_id = blockIdx.x * num_warps + S_chunk + warp_id, batch_offset = 0;
        work_id <= work_per_warp; 
        work_id++, seqlen_id += seqlen_stride        // reverse traversal
    ) {
        if (seqlen_id >= S) {
            seqlen_id = blockIdx.x * num_warps + S_chunk + warp_id;
            batch_offset += S;
        }
        const int lts = lt_start_ptr[batch_offset + seqlen_id];
        const int ute = ut_end_ptr[batch_offset + seqlen_id];
        if (lts > ute) {         // mask does not cover the whole row of KV
            int cp_chunk_id = seqlen_id / S_chunk;
            int remote_pe = my_pe + cp_chunk_id * cp_stride;
            remote_pe = remote_pe < total_n_pes ? remote_pe : remote_pe - total_n_pes; 
            if constexpr (use_semaphore) {
                if ((threadIdx.x & 31) == 0 && cached_semaphores[remote_pe] == 0) {
                    sema::ag::wait_full(semaphores, remote_pe);
                    cached_semaphores[remote_pe] = 1;
                }
            }
            const int src_addr = ((seqlen_id % S_chunk) + batch_offset) * S_stride;
            const int dst_addr = (seqlen_id + batch_offset) * S_stride;
            shmem::two_buffers_getmem_warp(
                k_sr + dst_addr,
                v_sr + dst_addr,
                k_sr + src_addr,
                v_sr + src_addr,
                S_stride * sizeof(T), remote_pe
            );
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            if (threadIdx.x == 0) atomicExch(&block_work_idx[blockIdx.x], work_id);
            int work_idx = threadIdx.x == blockIdx.x ? work_id : block_work_idx[threadIdx.x];
            work_idx = __reduce_min_sync(0xffffffff, work_idx);
            work_idx = work_idx == work_per_warp ? INT_MAX : (work_idx * num_warps * num_blocks);
            if (threadIdx.x == 0)
                atomicMax(wptr, work_idx);     // 256 * work_idx, or INT_MAX
        }
    }
}


template <typename T, int S, int S_chunk, int num_blocks=32, int num_warps=8, bool use_semaphore=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) DenseKVFewHeadRemoteGetBwdKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    constexpr int row_per_block = num_warps * 2;     // 16-line per-block (* 2), if (* 1) 8-line per-block
    constexpr int seqlen_stride = row_per_block * num_blocks;
    const int work_per_block = num_batch * (S - S_chunk) / seqlen_stride;
    __shared__ int cached_semaphores[16];

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    for (int work_id = 1, seqlen_id = blockIdx.x * row_per_block + S_chunk, batch_offset = 0;
        work_id <= work_per_block; 
        work_id++, seqlen_id += seqlen_stride        // reverse traversal
    ) {
        if (seqlen_id >= S) {
            seqlen_id = blockIdx.x * row_per_block + S_chunk;
            batch_offset += S;
        }
        int cp_chunk_id = seqlen_id / S_chunk;
        int remote_pe = my_pe + cp_chunk_id * cp_stride;
        remote_pe = remote_pe < total_n_pes ? remote_pe : remote_pe - total_n_pes; 
        if constexpr (use_semaphore) {
            if ((threadIdx.x & 31) == 0 && cached_semaphores[remote_pe] == 0) {
                sema::ag::wait_full(semaphores, remote_pe);
                cached_semaphores[remote_pe] = 1;
            }
        }
        const int src_addr = ((seqlen_id % S_chunk) + batch_offset) * S_stride;
        const int dst_addr = (seqlen_id + batch_offset) * S_stride;
        shmem::two_buffers_getmem_block(
            k_sr + dst_addr,
            v_sr + dst_addr,
            k_sr + src_addr,
            v_sr + src_addr,
            row_per_block * S_stride * sizeof(T), remote_pe
        );
        if (threadIdx.x < 32) {
            if (threadIdx.x == 0) atomicExch(&block_work_idx[blockIdx.x], work_id);
            int work_idx = threadIdx.x == blockIdx.x ? work_id : block_work_idx[threadIdx.x];
            work_idx = __reduce_min_sync(0xffffffff, work_idx);
            work_idx = work_idx == work_per_block ? INT_MAX : (work_idx * seqlen_stride);
            if (threadIdx.x == 0)
                atomicMax(wptr, work_idx);     // 128 * work_idx, or INT_MAX
        }
    }
}

template <int S_chunk, int num_warps = 8, int row_per_warp = 32, bool bwd=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) BlockSparsityCheckKernel(
    const int* const __restrict__ lt_start_ptr,
    const int* const __restrict__ ut_end_ptr,
    int* const __restrict__ copy_chunk_mask,
    const int num_head,                         // note that this is not KV head, but mask head
    const int head_stride
) {
    // each thread load 1 int4 from lt_start_ptr and 1 int4 from ut_end_ptr
    static_assert(num_warps == 16 || num_warps == 8 || num_warps == 4);
    static_assert(row_per_warp == 32 || row_per_warp == 64 || row_per_warp == 16);
    // num_warp = 4 and row_per_warp = 16 can't be different
    static_assert(row_per_warp != 16 || (row_per_warp == 16 && num_warps == 4));
    static constexpr int rows_per_cta = row_per_warp * num_warps;
    __shared__ int warps_masked[num_warps];
    __shared__ int temp_result[128 / row_per_warp];
    // bwd valid mask starts from mask[S_chunk], while fwd starts from mask[0] due to different roll strategy
    constexpr int mask_offset = bwd ? S_chunk : 0;
    const int batch_offset = blockIdx.y * num_head * head_stride;
    const int load_index = blockIdx.x * num_warps * 32 + threadIdx.x;
    const int* mask_lts = lt_start_ptr + mask_offset + batch_offset + load_index * 4;
    const int* mask_ute = ut_end_ptr + mask_offset + batch_offset + load_index * 4;
    for (int head_id = 0, head_offset = 0; head_id < num_head; head_id ++, head_offset += head_stride) {
        const int4 lts = *(reinterpret_cast<const int4*>(mask_lts + head_offset));
        const int4 ute = *(reinterpret_cast<const int4*>(mask_ute + head_offset));
        int is_masked = (lts.x <= ute.x) && (lts.y <= ute.y) && (lts.z <= ute.z) && (lts.w <= ute.w);
        // warp reduce to check whether all the 128 rows (32 * 4 = 128 rows) are masked 
        int current_warp_masked = __all_sync(0xffffffff, is_masked);
        if ((threadIdx.x % 32) == 0) {      // first lane
            warps_masked[threadIdx.x / 32] = current_warp_masked;
        }
        __syncthreads();
        // all of the LTS <= UTE, then the 4 rows are fully masked
        if (threadIdx.x == 0) {
            // two warps will produce the masking result of one 256/512-row block
            const int4* smem_int4 = reinterpret_cast<const int4*>(warps_masked);
            if constexpr (rows_per_cta >= 512) {
                int2 result;
                int4 src = *smem_int4;
                result.x = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 1);
                if constexpr (rows_per_cta == 1024) {
                    result.x &= src.x & src.y & src.z & src.w;
                    // reduce the second 2-int4s
                    src = *(smem_int4 + 2);
                    result.y = src.x & src.y & src.z & src.w;
                    src = *(smem_int4 + 3);
                    result.y &= src.x & src.y & src.z & src.w;
                } else {
                    result.y = src.x & src.y & src.z & src.w;
                }
                int2* const reduction_addr = reinterpret_cast<int2*>(temp_result);
                if (head_id == 0) {
                    *reduction_addr = result;
                } else {
                    int2 old_result = *reduction_addr;
                    // reduce over heads (the chunk can be skipped only when all-heads can be skipped)
                    old_result.x &= result.x;
                    old_result.y &= result.y;
                    *reduction_addr = old_result;
                }
            }
            if constexpr (rows_per_cta == 256) {        // reduce 2 int
                int4 result;
                int4 src = *smem_int4;
                result.x = src.x & src.y;
                result.y = src.z & src.w;
                src = *(smem_int4 + 1);
                result.z = src.x & src.y;
                result.w = src.z & src.w;
                if (head_id == 0) {
                    *(reinterpret_cast<int4*>(temp_result)) = result;
                } else {
                    int4 old_result = *(reinterpret_cast<int4*>(temp_result));
                    // reduce over heads (the chunk can be skipped only when all-heads can be skipped)
                    old_result.x &= result.x;
                    old_result.y &= result.y;
                    old_result.z &= result.z;
                    old_result.w &= result.w;
                    *(reinterpret_cast<int4*>(temp_result)) = old_result;
                }
            }
            if constexpr (rows_per_cta == 64) {         // no reduction
                int4 result = *smem_int4;
                if (head_id == 0) {
                    *(reinterpret_cast<int4*>(temp_result)) = result;
                } else {
                    int4 old_result = *(reinterpret_cast<int4*>(temp_result));
                    // reduce over heads (the chunk can be skipped only when all-heads can be skipped)
                    old_result.x &= result.x;
                    old_result.y &= result.y;
                    old_result.z &= result.z;
                    old_result.w &= result.w;
                    *(reinterpret_cast<int4*>(temp_result)) = old_result;
                }
            }
        }
    }
    if (threadIdx.x == 0) {
        int block_offset = blockIdx.y * gridDim.x + blockIdx.x;     // grid is: (num-chunks (int4), num_batch, 1)
        if constexpr (rows_per_cta >= 512) {
            *(reinterpret_cast<int2*>(copy_chunk_mask) + block_offset) = *(reinterpret_cast<int2*>(temp_result));
        } else {
            *(reinterpret_cast<int4*>(copy_chunk_mask) + block_offset) = *(reinterpret_cast<int4*>(temp_result));
        }
    }
}

/**
 * 'Sparse' means that we will check whether the 256/512-row chunk can be skipped or not.
 * @param copy_chunk_mask Pre-computed chunk mask. If all 256/512 rows in a chunk are masked, mask[chunk_id] = 1
 *  copy_chunk_mask is generated by `BlockSparsityCheckKernel`, 1D buffer (size: (S - S_chunk) / 256 or 512)
 * @param  block_cnt_semaphore Used in dynamic scheduling. Since each CTA is responsible for remote-getting
 *  one entire chunk (256/512 rows), if the chunk is masked then the computation power will be wasted. To avoid
 *  load-imbalance for this communication kernel (which is crucial to overlap performance), a chunk-cnt semaphore
 *  is used so every CTA use atomic op to get a chunk ID to process.
*/
template <typename T, int S, int S_chunk, int num_warps=8, int row_per_warp=32, bool use_stream_coord=false, bool use_semaphore=false, bool bwd=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseLargeKVChunkRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    int* const __restrict__ block_cnt_semaphore,      // for dynamic scheduling
    int* const __restrict__ stream_coordinator,
    const int* const __restrict__ copy_chunk_mask,
    const int my_pe,
    const int total_n_pes,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    if constexpr (use_stream_coord) {
        // notify computation stream that one of the CTAs for communication kernel is running
        if (threadIdx.x == 0) atomicOr(stream_coordinator, 1 << blockIdx.x);
    }
    constexpr int row_per_block = num_warps * row_per_warp;     // 256 or 512 row per block (32 or 64 per warp)
    constexpr int seqlen_offset = bwd ? 0 : (S - S_chunk);
    constexpr int chunk_per_batch = (S - S_chunk) / row_per_block;
    const int total_chunks = num_batch * chunk_per_batch;     // 8192 chunk: 32 blocks --> 96 blocks
    const int batch_stride = S * S_stride;
    __shared__ int cached_semaphores[16];
    __shared__ int next_work_id;
    // note that block_cnt_semaphore starts from 1. dyn-scheduling from the beginning.

    // this lambda ensures correct visibility to the change of block_work_idx and 
    // the last CTA will correctly set the wptr to be INT_MAX to notify completion 
    auto update_wptr_and_work_id_sync = [&](int wid) {
        if (threadIdx.x < 32) {
            if (threadIdx.x == blockIdx.x) {
                int next_wid = atomicAdd(block_cnt_semaphore, 1);       // fetch and check the next work ID
                // if there is no more work to do: set the wid for the current block to be INT_MAX
                wid = next_wid <= total_chunks ? wid : INT_MAX;
                next_work_id = next_wid;
                atomicExch(&block_work_idx[blockIdx.x], wid);
            }
            // make sure atomicExch happen before load from block_work_idx[threadIdx.x],
            // so that if one block finished atomicExch first, other blocks will know
            __syncwarp();
            wid = threadIdx.x != blockIdx.x ? block_work_idx[threadIdx.x] : wid;
            wid = __reduce_min_sync(0xffffffff, wid);
            if (threadIdx.x == blockIdx.x) {
                atomicMax(wptr, wid == INT_MAX ? INT_MAX : (wid * row_per_block));     // 256 or 512 * wid, or INT_MAX
            }
        }
        __syncthreads();
        return next_work_id;
    };

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
    }

    // // Uncomment this line to check what happens when communication is not stalling the computation
    // if (threadIdx.x == 0) {
    //     atomicMax(wptr, INT_MAX);
    // }

    for (int work_id = update_wptr_and_work_id_sync(0); work_id <= total_chunks;) {
        const int work_id_m1 = work_id - 1;
        const int batch_id = work_id_m1 / chunk_per_batch;
        const int seq_work_id = (work_id_m1 % chunk_per_batch) + 1;     // this is in range [1, chunk_per_batch]
        int mask_index = bwd ? (work_id - 1) : (chunk_per_batch * (batch_id + 1) - seq_work_id);
        if (copy_chunk_mask[mask_index]) {
            // does not need copying since the current KV block is masked. skip directly
            // __syncthreads() here is necessary: in case some of the warp haven't updated
            // the work_id and warp 0 overwrites next_work_id first, which will be bad.
            __syncthreads();
            work_id = update_wptr_and_work_id_sync(work_id);
            continue;
        }
        // calculate seqlen offset via work_id
        int seqlen_id = 0, remote_pe = 0;
        if constexpr (bwd) {        // bwd is forward traversal
            seqlen_id = S_chunk + (seq_work_id - 1) * row_per_block;
            remote_pe = my_pe + seqlen_id / S_chunk;
            remote_pe = remote_pe < total_n_pes ? remote_pe : remote_pe - total_n_pes; 
        } else {                    // fwd is reversed traversal
            seqlen_id = seqlen_offset - seq_work_id * row_per_block;
            int cp_chunk_id = (S - 1 - seqlen_id) / S_chunk;
            remote_pe = my_pe - cp_chunk_id;
            remote_pe = remote_pe >= 0 ? remote_pe : remote_pe + total_n_pes; 
        }
        if constexpr (use_semaphore) {
            if (threadIdx.x == 0 && cached_semaphores[remote_pe] == 0) {
                sema::ag::wait_full(semaphores, remote_pe);
                cached_semaphores[remote_pe] = 1;
                // no need to sync, since `two_buffers_getmem_<scope>` will sync 
            }
        }
        // copy upto 4 heads (S_stride = H * D), and row_per_block rows (seqlen axis) per CTA
        const int src_addr = batch_id * batch_stride + (seqlen_offset + (seqlen_id % S_chunk)) * S_stride;
        const int dst_addr = batch_id * batch_stride + seqlen_id * S_stride;
        shmem::two_buffers_getmem_block(
            k_sr + dst_addr,
            v_sr + dst_addr,
            k_sr + src_addr,
            v_sr + src_addr,
            row_per_block * S_stride * sizeof(T), remote_pe
        );
        // buffer getmem_block will call syncthreads(), so next_work_id will not be updated
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        work_id = update_wptr_and_work_id_sync(work_id);
    }
}

// remote get kernel (multi-stage remote_get overlapped gather), can only be called in the BWD when RS-overlap is ON.
// Current BWD write_ptr wait logic: absolute offset (not relative offset), meaning that the bwd won't skip
// the first chunk itself and make no assumption on the validity of the data. All it does now is to check whether
// the local read ptr is exceeded by the write ptr and load KV can procede if true. Therefore, we can choose to
// start the AG-overlap kernel (non-splitted version) just once at the beginning of the BWD kernel. We only need to
// inform the bwd kernel one more thing: what is the current segment ID? For now, use the splitted version.
template <typename T, int S_chunk, int num_warps=8, int row_per_warp=32, int num_chunks=4, bool use_stream_coord=false, bool use_semaphore=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseLargeKVChunkSplittedRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    int* const __restrict__ block_cnt_semaphore,      // for dynamic scheduling
    int* const __restrict__ stream_coordinator,
    const int* const __restrict__ copy_chunk_mask,
    const int start_rank,               // first call is my_pe + 1
    const int segment_idx,
    const int total_n_pes,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int num_segments = 4,
    const int64_t* const __restrict__ semaphores = nullptr
) {
    if constexpr (use_stream_coord) {
        // notify computation stream that one of the CTAs for communication kernel is running
        if (threadIdx.x == 0) atomicOr(stream_coordinator, 1 << blockIdx.x);
    }
    constexpr bool has_local = (num_chunks & 1) > 0;
    constexpr int row_per_block = num_warps * row_per_warp;
    // round the num_chunks to include the local chunk
    constexpr int rounded_num_chunks = has_local ? (num_chunks + 1) : num_chunks;
    constexpr int S = S_chunk * rounded_num_chunks;         // seqlen of this segment
    constexpr int work_per_seg = S / row_per_block;
    // for each segment, get the number of work we can skip (due to being local). Note that
    // if work_to_skip is not 0, there will be some skippable works for **each batch**
    constexpr int work_to_skip = has_local ? (S_chunk / row_per_block) : 0;
    constexpr int seqlen_offset = has_local ? 0 : S_chunk;
    // though the actual skippable work is `num_batch * (work_per_seg - work_to_skip)`
    // for batch_idx > 0, those skippable local works cannot be skipped **directly**.
    const int total_works = num_batch * work_per_seg;
    // this batch_stride is the segment batch stride, not full (num_segs * segment_size) batch stride
    const int batch_stride = S * S_stride;

    extern __shared__ int smem_chunk_mask[];
    __shared__ int cached_semaphores[16];
    __shared__ int next_work_id;

    if (threadIdx.x < total_works) {
        // global address to local segment address. total_works are usually small (<128), no need for vectorization
        const int batch_id = threadIdx.x / work_per_seg;
        auto* src_ptr = copy_chunk_mask + (segment_idx + batch_id * num_segments) * work_per_seg + threadIdx.x;
        smem_chunk_mask[batch_id * work_per_seg + threadIdx.x] = *src_ptr;
    }

    if constexpr (use_semaphore) {
        if (threadIdx.x < total_n_pes) {
            cached_semaphores[threadIdx.x] = 0;
        }
    }

    // block_cnt_semaphore starts from 1, and we can offset it
    auto update_wptr_and_work_id_sync = [&](int wid) {
        if (threadIdx.x < 32) {
            if (threadIdx.x == blockIdx.x) {
                int next_wid = atomicAdd(block_cnt_semaphore, 1) + work_to_skip;       // fetch and check the next work ID
                // if there is no more work to do: set the wid for the current block to be INT_MAX
                wid = next_wid <= total_works ? wid : INT_MAX;
                next_work_id = next_wid;
                atomicExch(&block_work_idx[blockIdx.x], wid);
            }
            // make sure atomicExch happen before load from block_work_idx[threadIdx.x],
            // so that if one block finished atomicExch first, other blocks will know
            __syncwarp();
            wid = threadIdx.x != blockIdx.x ? block_work_idx[threadIdx.x] : wid;
            wid = __reduce_min_sync(0xffffffff, wid);
            if (threadIdx.x == blockIdx.x) {
                atomicMax(wptr, wid == INT_MAX ? INT_MAX : (wid * row_per_block));     // 256 or 512 * wid, or INT_MAX
            }
        }
        __syncthreads();
        return next_work_id;
    };

    // // Uncomment this line to check what happens when communication is not stalling the computation
    // if (threadIdx.x == 0) {
    //     atomicMax(wptr, INT_MAX);
    // }

    // Note(heqianyue): the simple way to skip the first chunk: offset the initial work_cnt
    // and the atomicAdd result (semaphore fetch result)
    for (int work_id = update_wptr_and_work_id_sync(work_to_skip); work_id <= total_works;) {
        const int work_id_m1 = work_id - 1;
        const int batch_id = work_id_m1 / work_per_seg;
        const int seq_work_id = work_id_m1 % work_per_seg;
        // if there is local_chunk and B > 1, we might have some of the 
        // work to be skipped (due to being local) directly.
        if (seq_work_id < work_to_skip || smem_chunk_mask[seq_work_id + work_per_seg * batch_id]) {
            __syncthreads();
            work_id = update_wptr_and_work_id_sync(work_id);
            continue;
        }
        // calculate seqlen offset via work_id
        int seqlen_id = seq_work_id * row_per_block;
        int remote_pe = start_rank + seqlen_id / S_chunk;
        remote_pe = remote_pe < total_n_pes ? remote_pe : remote_pe - total_n_pes;

        if constexpr (use_semaphore) {
            if (threadIdx.x == 0 && cached_semaphores[remote_pe] == 0) {
                sema::ag::wait_full(semaphores, remote_pe);
                cached_semaphores[remote_pe] = 1;
                // no need to sync, since `two_buffers_getmem_<scope>` will sync 
            }
        }

        // copy upto 4 heads (S_stride = H * D), and row_per_block rows (seqlen axis) per CTA
        const int src_addr = batch_id * batch_stride + (seqlen_id % S_chunk) * S_stride;
        // if there is no local chunk, we offset the dst by one chunk (seqlen_offset = S_chunk)
        const int dst_addr = batch_id * batch_stride + (seqlen_id + seqlen_offset) * S_stride;
        shmem::two_buffers_getmem_block(
            k_sr + dst_addr,
            v_sr + dst_addr,
            k_sr + src_addr,
            v_sr + src_addr,
            row_per_block * S_stride * sizeof(T), remote_pe
        );
        // buffer getmem_block will call syncthreads(), so next_work_id will not be updated.
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        work_id = update_wptr_and_work_id_sync(work_id);
    }
}

}   // namespace flashmask