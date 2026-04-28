#pragma once
#include <cuda_runtime.h>
#include "sr_buffer.cuh"
#include "nvshmem_copy_utils.cuh"
#include "ag_semaphore_ops.cuh"
#include "hierarchical_rank_map.cuh"
#include "debug_logger.cuh"

namespace flashmask {

__device__ __forceinline__ int load_global_cg(const int* __restrict__ ptr) {
    int value;
    asm volatile ("ld.global.cg.s32 %0, [%1];" : "=r"(value) : "l"(ptr) : "memory");
    return value;
}

__global__ void InitBitmapForLocalSkip(int* const __restrict__ work_done) {
    int tid = threadIdx.x + 1;  // work_done is 1-based
    work_done[tid] = 1;
}

// Supports mask_head >= 1. When mask_head > 1, AND-reduces across all mask heads
// so that a chunk is skippable only when ALL heads agree it is fully masked.
// When mask_head == 1, the head loop runs once with negligible overhead.
template <int S_chunk, int num_warps = 8, int row_per_warp = 32, bool skip_local=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) BlockSparsityCheckSpecializedKernel(
    const int* const __restrict__ lt_start_ptr,
    const int* const __restrict__ ut_end_ptr,
    int* const __restrict__ copy_chunk_mask,
    const int num_head,                         // mask head count (H_mask)
    const int head_stride
) {
    static_assert(num_warps == 16 || num_warps == 8 || num_warps == 4);
    static_assert(row_per_warp == 32 || row_per_warp == 64 || row_per_warp == 16);
    static_assert(row_per_warp != 16 || (row_per_warp == 16 && num_warps == 4));
    static constexpr int rows_per_cta = row_per_warp * num_warps;
    __shared__ int warps_masked[num_warps];
    __shared__ int temp_result[4];              // cross-head AND-reduction buffer
    // mask layout: (B, H_mask, S_k), batch stride = num_head * head_stride
    const int batch_offset = blockIdx.y * num_head * head_stride;
    const int seq_offset = skip_local ? S_chunk : 0;
    const int load_index = blockIdx.x * num_warps * 32 + threadIdx.x;
    const int* mask_lts = lt_start_ptr + batch_offset + seq_offset + load_index * 4;
    const int* mask_ute = ut_end_ptr + batch_offset + seq_offset + load_index * 4;
    for (int head_id = 0, head_offset = 0; head_id < num_head; head_id++, head_offset += head_stride) {
        const int4 lts = *(reinterpret_cast<const int4*>(mask_lts + head_offset));
        const int4 ute = *(reinterpret_cast<const int4*>(mask_ute + head_offset));
        int is_masked = (lts.x <= ute.x) && (lts.y <= ute.y) && (lts.z <= ute.z) && (lts.w <= ute.w);
        int current_warp_masked = __all_sync(0xffffffff, is_masked);
        if ((threadIdx.x % 32) == 0) {
            warps_masked[threadIdx.x / 32] = current_warp_masked;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int4* smem_int4 = reinterpret_cast<const int4*>(warps_masked);
            if constexpr (rows_per_cta == 1024) {
                int2 result;
                int4 src = *smem_int4;
                result.x = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 1);
                result.x &= src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 2);
                result.y = src.x & src.y & src.z & src.w;
                src = *(smem_int4 + 3);
                result.y &= src.x & src.y & src.z & src.w;
                int2* const addr = reinterpret_cast<int2*>(temp_result);
                if (head_id == 0) { *addr = result; }
                else { int2 old = *addr; old.x &= result.x; old.y &= result.y; *addr = old; }
            }
            if constexpr (rows_per_cta == 512) {
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
                    int4* const addr = reinterpret_cast<int4*>(temp_result);
                    if (head_id == 0) { *addr = result; }
                    else { int4 old = *addr; old.x &= result.x; old.y &= result.y; old.z &= result.z; old.w &= result.w; *addr = old; }
                } else {
                    int2 result;
                    result.x = src.x & src.y & src.z & src.w;
                    src = *(smem_int4 + 1);
                    result.y = src.x & src.y & src.z & src.w;
                    int2* const addr = reinterpret_cast<int2*>(temp_result);
                    if (head_id == 0) { *addr = result; }
                    else { int2 old = *addr; old.x &= result.x; old.y &= result.y; *addr = old; }
                }
            }
            if constexpr (rows_per_cta == 256) {
                int4 result;
                int4 src = *smem_int4;
                result.x = src.x & src.y;
                result.y = src.z & src.w;
                src = *(smem_int4 + 1);
                result.z = src.x & src.y;
                result.w = src.z & src.w;
                int4* const addr = reinterpret_cast<int4*>(temp_result);
                if (head_id == 0) { *addr = result; }
                else { int4 old = *addr; old.x &= result.x; old.y &= result.y; old.z &= result.z; old.w &= result.w; *addr = old; }
            }
            if constexpr (rows_per_cta == 64) {
                int4 result = *smem_int4;
                int4* const addr = reinterpret_cast<int4*>(temp_result);
                if (head_id == 0) { *addr = result; }
                else { 
                    int4 old = *addr;
                    old.x &= result.x;
                    old.y &= result.y;
                    old.z &= result.z;
                    old.w &= result.w;
                    *addr = old; 
                }
            }
        }
        // barrier before next head iteration so warps_masked is not overwritten prematurely
        if (head_id < num_head - 1) __syncthreads();
    }
    // write the AND-reduced result to global memory
    if (threadIdx.x == 0) {
        const int block_offset = blockIdx.y * gridDim.x + blockIdx.x;
        if constexpr (rows_per_cta == 1024 || (rows_per_cta == 512 && num_warps != 16)) {
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
template <typename T, int S, int S_chunk, int num_warps=8, int row_per_warp=32, bool use_stream_coord=false, bool use_semaphore=false, bool bwd=false, bool use_hierarchical=false>
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
    int64_t* const __restrict__ semaphores = nullptr,
    int* const __restrict__ rank_empty_counters = nullptr,
    const int gpus_per_node = 1
) {
    // Degenerate case: S == S_chunk means nranks=1, nothing to remote-get.
    // This template instantiation is unreachable at runtime (overlap requires nranks > 1)
    // but is generated by SChunkDispatch × SeqlenDispatch combinatorial expansion.
    // Early return with proper wptr signaling to prevent hang if ever launched by mistake.
    if constexpr (S <= S_chunk) {
        if constexpr (use_stream_coord) {
            if (threadIdx.x == 0) atomicOr(stream_coordinator, 1 << blockIdx.x);
        }
        if (threadIdx.x == 0) atomicMax(wptr, INT_MAX);
        return;
    }

    if constexpr (use_stream_coord) {
        // notify computation stream that one of the CTAs for communication kernel is running
        if (threadIdx.x == 0) atomicOr(stream_coordinator, 1 << blockIdx.x);
    }
    constexpr int row_per_block = num_warps * row_per_warp;     // 256 or 512 row per block (32 or 64 per warp)
    constexpr int seqlen_offset = bwd ? 0 : (S - S_chunk);
    constexpr int chunk_per_batch = (S - S_chunk) / row_per_block;
    constexpr int work_per_chunk = S_chunk / row_per_block;
    const int total_chunks = num_batch * chunk_per_batch;
    const int works_per_rank = num_batch * work_per_chunk;
    const int batch_stride = S * S_stride;
    __shared__ int cached_semaphores[64];
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
            wid = threadIdx.x != blockIdx.x ? load_global_cg(block_work_idx + threadIdx.x) : wid;
            wid = __reduce_min_sync(0xffffffff, wid);
            if (threadIdx.x == blockIdx.x) {
                __threadfence();                // prevent the change on L2 visibility order (make sure previous stores are done) 
                atomicMax(wptr, wid == INT_MAX ? INT_MAX : (wid * row_per_block));     // 256 or 512 * wid, or INT_MAX
            }
        }
        __syncthreads();
        return next_work_id;
    };

    // P2P empty notify: when all works for a target rank are done, notify it immediately.
    // wait_full ensures the producer has set its semaphore before we send the atomic_add.
    // In hierarchical mode:
    //   Phase 1 release: decrement on target_rank (data owner) + congruence_notify same-node ranks
    //   Phase 2 release: decrement on src_pe (same-node rank holding target_rank's data)
    auto try_release_rank = [&](int target_rank_val, int src_pe_val, bool is_phase1_val) {
        if constexpr (use_semaphore) {
            if (threadIdx.x == 0) {
                int prev = atomicAdd(&rank_empty_counters[target_rank_val], 1);
                if (prev + 1 == works_per_rank) {
                    if (cached_semaphores[target_rank_val] == 0) {
                        sema::ag::wait_full(semaphores, target_rank_val);
                    }
                    if constexpr (use_hierarchical) {
                        if (is_phase1_val) {
                            // Phase 1: cross-node congruence group release
                            // 1. Decrement refcount on target_rank (data owner)
                            nvshmem_long_atomic_add(semaphores + target_rank_val, -1, target_rank_val);
                            // 2. Congruence notify: set local refcount for Phase 2 consumers + broadcast
                            //    No need to clear local semaphore first — this store overwrites,
                            //    and the remote AMO operates on target_rank's copy (not ours)
                            semaphores[target_rank_val] = hier::hier_congruence_refcount(gpus_per_node);
                            __threadfence();
                            const int my_pe_node = my_pe % gpus_per_node;
                            const int my_node_id = my_pe / gpus_per_node;
                            for (int slot = 1; slot < gpus_per_node; slot++) {
                                int base = (my_pe_node + slot) % gpus_per_node;
                                int sn_rank = base + my_node_id * gpus_per_node;
                                nvshmem_int64_p(semaphores + target_rank_val, 1, sn_rank);
                            }
                        } else {
                            // Phase 2: intra-node redistribution release
                            // Clear local semaphore + decrement on src_pe (same-node holder)
                            semaphores[target_rank_val] = 0;
                            nvshmem_long_atomic_add(semaphores + target_rank_val, -1, src_pe_val);
                        }
                    } else {
                        // Non-hierarchical: bitmask — clear our bit
                        semaphores[target_rank_val] = 0;
                        nvshmem_long_atomic_add(
                            semaphores + target_rank_val, -(1LL << my_pe), target_rank_val);
                    }
                }
            }
        }
    };

    if constexpr (use_semaphore) {
        if (threadIdx.x < total_n_pes) {
            cached_semaphores[threadIdx.x] = 0;
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
        int seqlen_id = 0, remote_pe = 0;
        int src_addr = 0;
        bool is_phase1 = true;
        int target_rank = 0;  // For hierarchical: data owner rank (semaphore wait target)

        if constexpr (use_hierarchical) {
            // Hierarchical traversal: congruence group first, intra-node second
            const int logical_pos = (seq_work_id - 1) / work_per_chunk + 1;  // 1-based remote chunk: skip local
            const int row_within_chunk = (seq_work_id - 1) % work_per_chunk;
            auto info = hier::hier_map_chunk(logical_pos, my_pe, total_n_pes, gpus_per_node);

            target_rank = info.target_rank;  // Data owner (semaphore wait target)
            remote_pe = info.src_pe;         // PE to fetch from (Phase 1: = target_rank; Phase 2: = same-node rank)
            is_phase1 = info.is_phase1;

            seqlen_id = hier::hier_seqlen_id(logical_pos, row_within_chunk, row_per_block, total_n_pes, S_chunk, bwd);

            // Source address in remote_pe's SR buffer
            const int src_chunk_seqlen = hier::hier_src_chunk_offset(
                info, total_n_pes, S_chunk, total_n_pes, gpus_per_node, bwd);
            src_addr = batch_id * batch_stride + (src_chunk_seqlen + row_within_chunk * row_per_block) * S_stride;
        } else {
            // Original circular shift traversal
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
            target_rank = remote_pe;  // Non-hierarchical: target_rank = remote_pe
            src_addr = batch_id * batch_stride + (seqlen_offset + (seqlen_id % S_chunk)) * S_stride;
        }

        if (copy_chunk_mask[mask_index]) {
            // Masked: no get needed, but still counts toward rank release
            try_release_rank(target_rank, remote_pe, is_phase1);
            __syncthreads();
            work_id = update_wptr_and_work_id_sync(work_id);
            continue;
        }
        if constexpr (use_semaphore) {
            // Phase 1: wait for semaphores[target_rank] > 0 (notify_full_hier from target_rank)
            // Phase 2: wait for semaphores[target_rank] > 0 (congruence_notify from src_pe)
            // In both cases, the semaphore index is target_rank, not remote_pe.
            // Phase 1: target_rank == remote_pe (fetching directly from data owner)
            // Phase 2: target_rank != remote_pe (fetching from src_pe who holds target_rank's data)
            if (threadIdx.x == 0 && cached_semaphores[target_rank] == 0) {
                sema::ag::wait_full(semaphores, target_rank);
                cached_semaphores[target_rank] = 1;
            }
        }
        // copy upto 4 heads (S_stride = H * D), and row_per_block rows (seqlen axis) per CTA
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
        try_release_rank(target_rank, remote_pe, is_phase1);
        work_id = update_wptr_and_work_id_sync(work_id);
    }
}

// remote get kernel (multi-stage remote_get overlapped gather), can only be called in the BWD when RS-overlap is ON.
// Current BWD write_ptr wait logic: absolute offset (not relative offset), meaning that the bwd won't skip
// the first chunk itself and make no assumption on the validity of the data. All it does now is to check whether
// the local read ptr is exceeded by the write ptr and load KV can procede if true. Therefore, we can choose to
// start the AG-overlap kernel (non-splitted version) just once at the beginning of the BWD kernel. We only need to
// inform the bwd kernel one more thing: what is the current segment ID? For now, use the splitted version.
//
// @param num_chunks  total chunks per segment (always the real count, e.g. 4 for CP16)
// @param has_local_chunk  true iff segment 0 (the first chunk is local and should be skipped)
template <typename T, int S_chunk, int num_warps=8, int row_per_warp=32, int num_chunks=4, bool has_local_chunk=false, bool use_stream_coord=false, bool use_semaphore=false>
__global__ void __launch_bounds__(num_warps * 32, 64 / num_warps) SparseLargeKVChunkSplittedRemoteGetKernel(
    T* const __restrict__ k_sr,
    T* const __restrict__ v_sr,
    const T* const __restrict__ local_k,
    const T* const __restrict__ local_v,
    int* const __restrict__ wptr,
    int* const __restrict__ block_work_idx,
    int* const __restrict__ block_cnt_semaphore,      // for dynamic scheduling
    int* const __restrict__ stream_coordinator,
    const int* const __restrict__ copy_chunk_mask,
    const int my_pe,
    const int start_rank,               // first call is my_pe + 1
    const int segment_idx,
    const int total_n_pes,
    const int num_batch,                // B
    const int S_stride,                 // H * D
    const int num_segments = 4,
    int64_t* const __restrict__ semaphores = nullptr,
    int* const __restrict__ rank_empty_counters = nullptr
) {
    if constexpr (use_stream_coord) {
        // notify computation stream that one of the CTAs for communication kernel is running
        if (threadIdx.x == 0) atomicOr(stream_coordinator, 1 << blockIdx.x);
    }
    // segment has only a local chunk, nothing to remote-get.
    if constexpr (has_local_chunk && num_chunks == 1) {
        if (threadIdx.x == 0) atomicMax(wptr, INT_MAX);
        return;
    }
    constexpr bool has_local = has_local_chunk;
    constexpr int row_per_block = num_warps * row_per_warp;
    constexpr int S = S_chunk * num_chunks;                  // seqlen of this segment (num_chunks is already the real total)
    constexpr int work_per_seg = S / row_per_block;
    constexpr int work_per_chunk = S_chunk / row_per_block;
    // for each segment, get the number of work we can skip (due to being local). Note that
    // if work_to_skip is not 0, there will be some skippable works for **each batch**
    constexpr int work_to_skip = has_local ? (S_chunk / row_per_block) : 0;
    constexpr int chunk_offset = has_local ? 1 : 0;
    // though the actual skippable work is `num_batch * (work_per_seg - work_to_skip)`
    // for batch_idx > 0, those skippable local works cannot be skipped **directly**.
    const int total_works = num_batch * work_per_seg;
    const int works_per_rank = num_batch * work_per_chunk;
    // this batch_stride is the segment batch stride, not full (num_segs * segment_size) batch stride
    const int batch_stride = S * S_stride;
    const int local_batch_stride = S_chunk * S_stride;

    extern __shared__ int smem_chunk_mask[];
    __shared__ int cached_semaphores[64];
    __shared__ int next_work_id;

    if (threadIdx.x < total_works) {
        // global address to local segment address. total_works are usually small (<128), no need for vectorization
        const int batch_id = threadIdx.x / work_per_seg;
        auto* src_ptr = copy_chunk_mask + (segment_idx + batch_id * num_segments) * work_per_seg 
            + (threadIdx.x % work_per_seg);
        smem_chunk_mask[threadIdx.x] = *src_ptr;
    }

    if constexpr (use_semaphore) {
        if (threadIdx.x < total_n_pes) {
            cached_semaphores[threadIdx.x] = 0;
        }
    }

    // bitmap pointers (compute kernel reads work_done[work_id] directly — per-work ready flag)
    int* const work_done = block_work_idx;

    // Per-work ready flag update (thread-0 only):
    //   Mark:  work_done[wid]=1 + __threadfence()  (makes flag visible to compute kernel)
    //   Next:  atomicAdd to grab the next work item
    // No scan or wptr update needed — compute kernel checks work_done[] directly.
    auto update_wptr_and_work_id_sync = [&](int wid) {
        if (threadIdx.x == 0) {
            if (wid > work_to_skip) {
                __threadfence();
                atomicExch(&work_done[wid], 1);
            }
            next_work_id = atomicAdd(block_cnt_semaphore, 1) + work_to_skip;
        }
        __syncthreads();
        return next_work_id;
    };

    // // Uncomment this line to check what happens when communication is not stalling the computation
    // if (threadIdx.x == 0) {
    //     atomicMax(wptr, INT_MAX);
    // }

    // P2P empty notify: when all works for a source PE (chunk) are done, notify it immediately.
    // Counter indexed by chunk_id - chunk_offset (same scheme as remote_put's try_commit_rank).
    // wait_full ensures the producer has set its bitmask before we send the atomic_add.
    auto try_release_rank = [&](int chunk_id) {
        if constexpr (use_semaphore) {
            if (threadIdx.x == 0) {
                int counter_idx = chunk_id - chunk_offset;
                int prev = atomicAdd(&rank_empty_counters[counter_idx], 1);
                if (prev + 1 == works_per_rank) {
                    int target_rank = (start_rank + chunk_id) % total_n_pes;
                    if (cached_semaphores[target_rank] == 0) {
                        sema::ag::wait_full(semaphores, target_rank);
                    }
                    semaphores[target_rank] = 0;
                    nvshmem_long_atomic_add(
                        semaphores + target_rank,
                        -(1LL << my_pe), target_rank);
                }
            }
        }
    };

    // Note(heqianyue): the simple way to skip the first chunk: offset the initial work_cnt
    // and the atomicAdd result (semaphore fetch result)
    for (int work_id = update_wptr_and_work_id_sync(work_to_skip); work_id <= total_works;) {
        const int work_id_m1 = work_id - 1;
        const int batch_id = work_id_m1 / work_per_seg;
        const int seq_work_id = work_id_m1 % work_per_seg;

        // Local chunk works: skip, no release needed
        if (seq_work_id < work_to_skip) {
            __syncthreads();
            work_id = update_wptr_and_work_id_sync(work_id);
            continue;
        }

        int chunk_id = seq_work_id / work_per_chunk;

        if (smem_chunk_mask[work_id_m1]) {
            // Masked remote: no get needed, but still counts toward rank release
            try_release_rank(chunk_id);
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
            }
        }

        // copy upto 4 heads (S_stride = H * D), and row_per_block rows (seqlen axis) per CTA
        const int src_addr = batch_id * local_batch_stride + (seqlen_id % S_chunk) * S_stride;
        const int dst_addr = batch_id * batch_stride + seqlen_id * S_stride;
        shmem::two_buffers_getmem_block(
            k_sr + dst_addr,
            v_sr + dst_addr,
            local_k + src_addr,
            local_v + src_addr,
            row_per_block * S_stride * sizeof(T), remote_pe
        );
        // buffer getmem_block will call syncthreads(), so next_work_id will not be updated.
        // therefore, next_work_id's update is visible to all threads and won't be overwritten
        // before some threads reading it. Safe!
        try_release_rank(chunk_id);
        work_id = update_wptr_and_work_id_sync(work_id);
    }
}

}   // namespace flashmask