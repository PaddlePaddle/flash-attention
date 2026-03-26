#pragma once
#include <cuda_runtime.h>
#include "sr_buffer.cuh"
#include "nvshmem_copy_utils.cuh"
#include "rs_semaphore_ops.cuh"
#include "debug_logger.cuh"

namespace flashmask {

/**
 * @brief Splitted remote put kernel for RS-overlap.
 *
 * @param num_chunk  total chunks per segment (always the real count, e.g. 4 for CP16)
 * @param has_local_chunk  true iff segment 0 (the first chunk is local and should be skipped)
 *
 * @param x_send & x_recv: SepSRBuffer separates send and recv into two buffers, since this is
 * actually an all-gather op implemented by an A2A op, the buffer cannot be reused the same way as the
 * all-gather overlap.
*/
template <typename T, int S_chunk, int num_warps=8, int row_per_warp=32, int num_chunk=4, bool has_local_chunk=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseLargeKVChunkRemotePutKernel(
    const T* const __restrict__ k_send,                 // K src addr (local)
    const T* const __restrict__ v_send,                 // V src addr (local)
    T* const __restrict__ k_recv,                       // K dst addr (remote)
    T* const __restrict__ v_recv,                       // V dst addr (remote)
    int* const __restrict__ block_cnt_semaphore,        // for dynamic scheduling
    const int* const __restrict__ copy_chunk_mask,
    const int my_pe,
    const int start_rank,               // start rank is chunk 0
    const int cp_size,
    const int segment_idx,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int64_t* const __restrict__ semaphores,
    const int num_segments = 4
) {
#ifdef NVSHMEM_DEBUG
    if (threadIdx.x == 0) {
       DEBUG_PRINT("Remote put starts, blockIdx: %d, self rank: %d, segment_idx: %d / %d\n", blockIdx.x, my_pe, segment_idx, num_segments);
    }
#endif  // NVSHMEM_DEBUG
    // segment has only a local chunk, nothing to remote-put. Return immediately.
    if constexpr (has_local_chunk && num_chunk == 1) {
        return;
    }
    static constexpr bool has_local = has_local_chunk;
    // skip the local chunk by using 1 as offset
    static constexpr int chunk_offset = has_local ? 1 : 0;
    static constexpr int remote_chunks = num_chunk - chunk_offset > 0 ? num_chunk - chunk_offset : 1;  // clamp to 1 to suppress div-by-zero warning; total_works=0 guards runtime
    static constexpr int row_per_block = num_warps * row_per_warp;     // 256 or 512 row per block (32 or 64 per warp)
    static constexpr int work_per_chunk = S_chunk / row_per_block;
    static constexpr int work_per_seg = work_per_chunk * remote_chunks;

    const int total_works = num_batch * work_per_seg;
    const int batch_stride = S_chunk * num_chunk * S_stride;         // num_chunk is the real total

    extern __shared__ int smem_chunk_mask[];
    __shared__ int cached_empty[num_chunk];
    __shared__ int next_work_id;

    if (threadIdx.x < total_works) {
        const int batch_id = threadIdx.x / work_per_seg;
        const int seqlen_id = threadIdx.x % work_per_seg;
        constexpr int start_offset = chunk_offset * work_per_chunk;
        auto* src_ptr = copy_chunk_mask + (segment_idx + batch_id * num_segments) * num_chunk * work_per_chunk + seqlen_id;
        smem_chunk_mask[start_offset + threadIdx.x] = *(src_ptr + start_offset);
        if (threadIdx.x < num_chunk) {
            cached_empty[threadIdx.x] = 0;
        }
    }
    __syncthreads();

    // this lambda ensures correct visibility to the change of block_work_idx and 
    // the last CTA will correctly set the wptr to be INT_MAX to notify completion 
    auto update_work_id_sync = [&]() {
        if (threadIdx.x == blockIdx.x) {
            // Note: block_cnt_semaphore for remote_put starts from 0 (for remote get, starts from 1)
            int _work_id = atomicAdd(block_cnt_semaphore, 1);
            next_work_id = _work_id;
            // chunk ID changes first: chunk [0, 1, 2, 3, 0, 1, 2, 3, ...]
            int chunk_id = (_work_id % remote_chunks) + chunk_offset;
            int target_pe = (start_rank + chunk_id) % cp_size;
            // the current CTA might be scheduled to send things to different ranks
            // so each CTA should keep track of whether the target rank is empty
            if (cached_empty[chunk_id] == 0) {
                sema::rs::producer_wait_empty(semaphores, target_pe);
                cached_empty[chunk_id] = 1;
            }
        }
        __syncthreads();
        return next_work_id;
    };

    for (int work_id = update_work_id_sync(); work_id < total_works;) {
        const int chunk_work_id = work_id / remote_chunks;         // the work_id within the chunk
        // seg_chunk_id is also the offset to the start_rank
        const int seg_chunk_id = (work_id % remote_chunks) + chunk_offset;         // which chunk the work_id falls into
        const int target_rank = (seg_chunk_id + start_rank) % cp_size;

        // within segment seqlen work_id
        const int batch_id = chunk_work_id / work_per_chunk;
        const int seq_work_id = chunk_work_id % work_per_chunk;     // this is in range [0, work_per_chunk)

        // Note(heqianyue): this copy_chunk_mask is computed without sharding. Therefore we need to compute
        // the correct index, considering the stride of the full seqlen_k
        int mask_index = seq_work_id + work_per_chunk * (seg_chunk_id + remote_chunks * batch_id);

        if (smem_chunk_mask[mask_index]) {
            // __syncthreads() here is necessary: in case some of the warp haven't updated
            // the work_id and warp 0 overwrites next_work_id first, which will be bad.
            __syncthreads();
            work_id = update_work_id_sync();
            continue;
        }
        
        // batch_offset + chunk_offset + work_offset, the src and dst in the buffer is the same
        const int addr = batch_id * batch_stride + (seg_chunk_id * S_chunk + seq_work_id * row_per_block) * S_stride;

        shmem::two_buffers_putmem_block(
            k_recv + addr,
            v_recv + addr,
            k_send + addr,
            v_send + addr,
            row_per_block * S_stride * sizeof(T), target_rank
        );
        // buffer putmem_block will call syncthreads(), so next_work_id will not be updated
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        work_id = update_work_id_sync();
    }
#ifdef NVSHMEM_DEBUG
    if (threadIdx.x == 0) {
       DEBUG_PRINT("Remote put quits, blockIdx: %d, self rank: %d, segment_idx: %d\n", blockIdx.x, my_pe, segment_idx);
    }
#endif  // NVSHMEM_DEBUG
}

}   // namespace flashmask