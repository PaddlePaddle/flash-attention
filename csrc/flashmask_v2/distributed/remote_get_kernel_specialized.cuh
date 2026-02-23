#pragma once
#include "remote_get_kernel.cuh"

// specialized version for the kernels defined in `remote_get_kernel.cuh`
// targeted for single batch, D = 128, H(KV) = 1 case.

namespace flashmask {

#define UNUSED __attribute__((unused))

template <typename T, int S, int S_chunk, int num_blocks=32, int num_warps=8, bool use_semaphore=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseKVFewHeadRemoteGetSpecializedKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    const int* const __restrict__ lt_start_ptr,
    const int* const __restrict__ ut_end_ptr,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int UNUSED _num_batch,                // B
    const int UNUSED _S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    constexpr int seqlen_stride = num_warps * num_blocks;
    constexpr int seqlen_offset = S - S_chunk;
    constexpr int S_stride = 128;           // single KV head, hd128
    const int warp_id = threadIdx.x >> 5;
    const int work_per_warp = seqlen_offset / seqlen_stride;
    __shared__ int cached_semaphores[16];

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    for (int work_id = 1, seqlen_id = seqlen_offset - 1 - (blockIdx.x * num_warps + warp_id);
        work_id <= work_per_warp; 
        work_id++, seqlen_id -= seqlen_stride        // reverse traversal
    ) {
        const int lts = lt_start_ptr[seqlen_id];
        const int ute = ut_end_ptr[seqlen_id];
        if (lts > ute) {         // mask does not cover the whole row of KV
            int cp_chunk_id = (S - 1 - seqlen_id) / S_chunk;
            int remote_pe = my_pe - cp_chunk_id * cp_stride;
            remote_pe = remote_pe >= 0 ? remote_pe : remote_pe + total_n_pes; 
            if constexpr (use_semaphore) {
                if ((threadIdx.x & 31) == 0 && cached_semaphores[remote_pe] == 0) {
                    sema::ag::wait_full(semaphores, remote_pe);
                    cached_semaphores[remote_pe] = 1;
                }
            }
            const int src_addr = (S - S_chunk + (seqlen_id % S_chunk)) * S_stride;
            const int dst_addr = seqlen_id * S_stride;
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
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) DenseKVFewHeadRemoteGetSpecializedKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int UNUSED _num_batch,                // B
    const int UNUSED _S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    constexpr int row_per_block = num_warps * 32;     // 16-line per-block (* 2), if (* 1) 8-line per-block
    constexpr int seqlen_stride = row_per_block * num_blocks;
    constexpr int seqlen_offset = S - S_chunk;
    constexpr int S_stride = 128;           // single KV head, hd128
    const int work_per_block = seqlen_offset / seqlen_stride;
    __shared__ int cached_semaphores[16];

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    for (int work_id = 1, seqlen_id = seqlen_offset - (blockIdx.x + 1) * row_per_block;
        work_id <= work_per_block; 
        work_id++, seqlen_id -= seqlen_stride        // reverse traversal
    ) {
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
        const int src_addr = (S - S_chunk + (seqlen_id % S_chunk)) * S_stride;
        const int dst_addr = seqlen_id * S_stride;
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
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseKVFewHeadRemoteGetSpecializedBwdKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    const int* const __restrict__ lt_start_ptr,
    const int* const __restrict__ ut_end_ptr,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int UNUSED _num_batch,                // B
    const int UNUSED _S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    constexpr int seqlen_stride = num_warps * num_blocks;
    constexpr int S_stride = 128;           // single KV head, hd128
    const int warp_id = threadIdx.x >> 5;
    const int work_per_warp = (S - S_chunk) / seqlen_stride;

    __shared__ int cached_semaphores[16];

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    for (int work_id = 1, seqlen_id = blockIdx.x * num_warps + S_chunk + warp_id;
        work_id <= work_per_warp; 
        work_id++, seqlen_id += seqlen_stride        // reverse traversal
    ) {
        const int lts = lt_start_ptr[seqlen_id];
        const int ute = ut_end_ptr[seqlen_id];
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
            const int src_addr = (seqlen_id % S_chunk) * S_stride;
            const int dst_addr = seqlen_id * S_stride;
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
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) DenseKVFewHeadRemoteGetSpecializedBwdKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    const int my_pe,
    const int cp_stride,
    const int total_n_pes,
    const int UNUSED _num_batch,                // B
    const int UNUSED _S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores = nullptr
) {
    constexpr int row_per_block = num_warps * 32;     // 16-line per-block (* 2), if (* 1) 8-line per-block
    constexpr int seqlen_stride = row_per_block * num_blocks;
    constexpr int S_stride = 128;           // single KV head, hd128
    const int work_per_block = (S - S_chunk) / seqlen_stride;
    __shared__ int cached_semaphores[16];

    if constexpr (use_semaphore) {
        if (threadIdx.x * 4 < total_n_pes) {
            *(reinterpret_cast<int4*>(cached_semaphores) + threadIdx.x) = make_int4(0, 0, 0, 0);
        }
        __syncthreads();
    }

    for (int work_id = 1, seqlen_id = blockIdx.x * row_per_block + S_chunk;
        work_id <= work_per_block; 
        work_id++, seqlen_id += seqlen_stride        // reverse traversal
    ) {
        int remote_pe = my_pe + (seqlen_id / S_chunk) * cp_stride;
        remote_pe = remote_pe < total_n_pes ? remote_pe : remote_pe - total_n_pes; 
        if constexpr (use_semaphore) {
            if ((threadIdx.x & 31) == 0 && cached_semaphores[remote_pe] == 0) {
                sema::ag::wait_full(semaphores, remote_pe);
                cached_semaphores[remote_pe] = 1;
            }
        }
        const int src_addr = (seqlen_id % S_chunk) * S_stride;
        const int dst_addr = seqlen_id * S_stride;
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


// =========================== bigger patch transfer kernel ====================================
/**
 * The latest experiment shows that for multi-node overlap, communication is the culprit for degraded performance
 * Splitting the KV tensor into smaller chunks performs worse than using larger chunks.
 * Currently, 256 or 512 rows per CTA (32 CTAs in total, so each wave transfers 8192 rows, which is one CP chunk)
 * yields the best performance, but the copy is in terms of dense chunks. After dual chunk reordering and mask processing,
 * the 256/512-row block sparsity is actually not low: avg 50% of the chunks are masked, which need no transfering.
 * I suppose this can save us a lot of time in the multi-node context, and is potentially beneficial for single-node
 * 
 * The follow kernels implements specialized sparse large chunk transfer kernels. These kernels:
 * - Check with the given mask: whether a KV block (256/512 rows) is masked (so that we can skip transfering)
 * - Transfer KV data sparsely. The sparisty granularity is 256/512-row block (either to copy entire block, or discard it)
 * - Dynamically schedule the CTAs. Since there will be skipped blocks (50%), we want the workload for
 *  different CTAs to be more balanced. 
*/

// Specialized kernel used only when (Mask Head = 1, meaning that multiple KV heads share the same mask)
// When num_warp is 4. this kernel only has 128 threads per CTA and each CTA reduces 512 mask pos to an int4
// Can be used when KV head > 1 to reduce comm workload per CTA (so that computation won't stall that long)
template <int S_chunk, int num_warps = 8, int row_per_warp = 32, bool skip_local=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) BlockSparsityCheckSpecializedKernel(
    const int* const __restrict__ lt_start_ptr,
    const int* const __restrict__ ut_end_ptr,
    int* const __restrict__ copy_chunk_mask,
    const int UNUSED num_head,
    const int head_stride
) {
    // each thread load 1 int4 from lt_start_ptr and 1 int4 from ut_end_ptr
    static_assert(num_warps == 16 || num_warps == 8 || num_warps == 4);
    static_assert(row_per_warp == 32 || row_per_warp == 64 || row_per_warp == 16);
    // num_warp = 4 and row_per_warp = 16 can't be different
    static_assert(row_per_warp != 16 || (row_per_warp == 16 && num_warps == 4));
    static constexpr int rows_per_cta = row_per_warp * num_warps;
    __shared__ int warps_masked[num_warps];
    // bwd valid mask starts from mask[S_chunk], while fwd starts from mask[0] due to different roll strategy
    const int batch_offset = blockIdx.y * head_stride;
    const int mask_offset = (skip_local ? S_chunk : 0) + batch_offset;
    const int load_index = blockIdx.x * num_warps * 32 + threadIdx.x;
    const int4 lts = *(reinterpret_cast<const int4*>(lt_start_ptr + mask_offset) + load_index);
    const int4 ute = *(reinterpret_cast<const int4*>(ut_end_ptr + mask_offset) + load_index);
    // all of the LTS <= UTE, then the 4 rows are fully masked
    int is_masked = (lts.x <= ute.x) && (lts.y <= ute.y) && (lts.z <= ute.z) && (lts.w <= ute.w);
    // warp reduce to check whether all the 128 rows (32 * 4 = 128 rows) are masked 
    int current_warp_masked = __all_sync(0xffffffff, is_masked);
    if ((threadIdx.x % 32) == 0) {      // first lane
        warps_masked[threadIdx.x / 32] = current_warp_masked;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // grid is: (num-chunks (# 1024-row chunks), num_batch, 1)
        const int block_offset = blockIdx.y * gridDim.x + blockIdx.x;
        // two warps will produce the masking result of one 256/512-row block
        const int4* smem_int4 = reinterpret_cast<const int4*>(warps_masked);
        if constexpr (rows_per_cta == 1024) {       // reduce 8 int
            // happens only when num_warps == 16 and rows_per_warp = 64
            // we reduce to 16ints, each int represents 128 rows. So
            // we need to further reduce two int4 into a single int value
            int2 result;
            int4 src = *smem_int4;
            result.x = src.x & src.y & src.z & src.w;
            src = *(smem_int4 + 1);
            result.x &= src.x & src.y & src.z & src.w;
            // reduce the second 2-int4s
            src = *(smem_int4 + 2);
            result.y = src.x & src.y & src.z & src.w;
            src = *(smem_int4 + 3);
            result.y &= src.x & src.y & src.z & src.w;
            *(reinterpret_cast<int2*>(copy_chunk_mask) + block_offset) = result;
        }
        if constexpr (rows_per_cta == 512) {        // reduce 4 int
            int4 src = *smem_int4;
            if constexpr (num_warps == 16) {
                int4 result;
                result.x = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 1);
                result.y = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 2);
                result.z = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 3);
                result.w = src.x & src.y & src.z & src.w;
                *(reinterpret_cast<int4*>(copy_chunk_mask) + block_offset) = result;
            } else {
                int2 result;
                result.x = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 1);
                result.y = src.x & src.y & src.z & src.w;
                *(reinterpret_cast<int2*>(copy_chunk_mask) + block_offset) = result;
            }
        }
        if constexpr (rows_per_cta == 256) {        // reduce 2 int
            int4 result;
            int4 src = *(smem_int4);
            result.x = src.x & src.y;
            result.y = src.z & src.w;
            src = *(smem_int4 + 1);
            result.z = src.x & src.y;
            result.w = src.z & src.w;
            *(reinterpret_cast<int4*>(copy_chunk_mask) + block_offset) = result;
        }
        if constexpr (rows_per_cta == 64) {        // no reduction
            *(reinterpret_cast<int4*>(copy_chunk_mask) + block_offset) = *smem_int4;
        }
    }
}

#undef UNUSED

}   // namespace flashmask