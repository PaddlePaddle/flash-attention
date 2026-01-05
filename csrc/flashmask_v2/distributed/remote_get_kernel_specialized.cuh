#pragma once
#include "remote_get_kernel.cuh"

// specialized version for the kernels defined in `remote_get_kernel.cuh`
// targeted for single batch, D = 128, H(KV) = 1 case.

namespace flashmask {

#define UNUSED __attribute__((unused))

template <typename T, int S, int S_chunk, int num_blocks=32, int num_warps=8, bool use_semaphore=false>
__global__ void __launch_bounds__(256, 8) SparseKVFewHeadRemoteGetSpecializedKernel(
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
                    sema::wait_full(semaphores, remote_pe);
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
__global__ void __launch_bounds__(256, 8) DenseKVFewHeadRemoteGetSpecializedKernel(
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
    constexpr int row_per_block = num_warps * 2;     // 16-line per-block (* 2), if (* 1) 8-line per-block
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
                sema::wait_full(semaphores, remote_pe);
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
__global__ void __launch_bounds__(256, 8) SparseKVFewHeadRemoteGetSpecializedBwdKernel(
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
                    sema::wait_full(semaphores, remote_pe);
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
__global__ void __launch_bounds__(256, 8) DenseKVFewHeadRemoteGetSpecializedBwdKernel(
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
    constexpr int row_per_block = num_warps * 2;     // 16-line per-block (* 2), if (* 1) 8-line per-block
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
                sema::wait_full(semaphores, remote_pe);
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

#undef UNUSED

}   // namespace flashmask