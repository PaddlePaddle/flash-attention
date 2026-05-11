#pragma once
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "hierarchical_rank_map.cuh"

namespace flashmask {
namespace sema {
namespace ag {

/**
 * @brief Compute bitmask with bits [0, num_batch) set, safe for num_batch in [0, 64].
 */
__host__ __device__ __forceinline__ int64_t make_all_batch_bits(int num_batch) {
    // Avoid UB: shift by 64 on a 64-bit type is undefined.
    return num_batch >= 64 ? static_cast<int64_t>(~0ULL)
                           : static_cast<int64_t>((1ULL << num_batch) - 1);
}

/**
 * @brief (Device Function) Wait until semaphores[target_pe] > 0 (data ready).
 */
__device__ __forceinline__ void wait_full(
    const int64_t* const __restrict__ semaphores,
    const int target_pe
) {
    nvshmem_int64_wait_until(const_cast<int64_t*>(semaphores) + target_pe, NVSHMEM_CMP_GT, 0);
}


/**
 * @brief (Device Function) Wait until semaphores[target_pe] has batch_idx bit set.
 *   Used in hierarchical Phase 2 to wait for per-batch relay data readiness.
 */
__device__ __forceinline__ void wait_full_one_batch(
    const int64_t* const __restrict__ semaphores,
    const int batch_idx,
    const int target_pe
) {
    const int64_t batch_bit = static_cast<int64_t>(1ULL << batch_idx);
    int64_t current_val;
    do {
        asm volatile("ld.volatile.global.s64 %0, [%1];"
            : "=l"(current_val) : "l"(semaphores + target_pe) : "memory");
    } while (!(current_val & batch_bit));
}

/**
 * @brief (Device Function) Wait until all batch bits are set in semaphores[target_pe].
 *   Used in HierNotifyEmpty to drain per-batch relay signals before clearing.
 */
__device__ __forceinline__ void wait_full_all_batch(
    const int64_t* const __restrict__ semaphores,
    const int64_t all_batch_bits,
    const int target_pe
) {
    int64_t current_val;
    do {
        asm volatile("ld.volatile.global.s64 %0, [%1];"
            : "=l"(current_val) : "l"(semaphores + target_pe) : "memory");
    } while (current_val != all_batch_bits);
}

// Note(heqianyue): single node AMO can use int (4B) as semaphore types, but when in multi-node
// env, IBRC does not allow 4B AMO. Check NVSHMEM 3.2.5 src/modules/transport/ibrc/ibrc.cpp:1265
// So we need to use int64_t semaphores. If we know for sure that our CP distributed overlap
// utilizes only 1 node, change the dtype of SR buffer, remote_get kernels and current file.

// num thread: total_pes
__global__ void NotifySemaphoreEmptyKernel(
    int64_t* const __restrict__ semaphores,
    const int my_pe
) {
    if (threadIdx.x != my_pe) {
        // the other PE will not notify us before we reset
        wait_full(semaphores, threadIdx.x);
        semaphores[threadIdx.x] = 0;
        // Note(heqianyue): bitwise op is generally safer than add, if we are using only 1 node
        // we can opt for the following atomic_and approach
        // clear bit representing the current PE on the all other target PE
        nvshmem_long_atomic_add(semaphores + threadIdx.x, -(1LL << my_pe), threadIdx.x);
    }
}

// notify some of the remote kernels: local rank has finished 
// using the data of yours. Used in RS-overlap splitted AG
__global__ void NotifySegmentSemaphoreEmptyKernel(
    int64_t* const __restrict__ semaphores,
    const int my_pe,
    const int start_rank,
    const int total_pes
) {
    const int target_rank = (start_rank + threadIdx.x) % total_pes;
    if (target_rank != my_pe) {
        // the other PE will not notify us before we reset
        wait_full(semaphores, target_rank);
        semaphores[target_rank] = 0;
        nvshmem_long_atomic_add(semaphores + target_rank, -(1LL << my_pe), target_rank);
    }
}

// A debug kernel for `wait_self_empty`. Spins until the max-cycles or predicate is true.
// If max-cycles is reached, skip this kernel and report status with print
__global__ void DebugWaitOnStreamLocalKernel(
    int64_t* const __restrict__ semaphore,
    const int64_t target_val
) {
    static constexpr int64_t max_allowed_wait_cycles = 100000000000; 
    int64_t start_cycles = clock64();
    int64_t current_val = 0;

    while (true) {
        asm volatile("ld.volatile.global.s64 %0, [%1];" 
                     : "=l"(current_val) : "l"(semaphore) : "memory");

        if (current_val == target_val) {
            printf("Semaphore is already empty, quit waiting\n");
            return;
        }

        if (clock64() - start_cycles > max_allowed_wait_cycles) {
            printf("[WaitOnStreamKernel TimeOut] Wait for %ld, but still got: %ld\n", 
                target_val, current_val);
            start_cycles = clock64();
        } 
    }
}

__global__ void SetFullKernel(
    int64_t* const __restrict__ semaphores,
    int64_t value,
    int self_rank
) {
    if (threadIdx.x == 0) {
        semaphores[self_rank] = value;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == self_rank) return;
    // set the semaphores[self_rank] = 1 for all remote ranks
    nvshmem_int64_p(semaphores + self_rank, 1, threadIdx.x);
}

/**
 * @brief CPU wait until the semaphores[my_pe] reached 0
 * @param semaphores int semaphores allocated by nvshmem: size is total_n_pes
 * @param my_pe the id of semaphore to wait for
 * @param stream waiting stream. This API is therefore async on stream (if non-blocking)
*/
void wait_self_empty(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    cudaStream_t stream
) {
    static constexpr bool IS_DEBUG = false;
    if constexpr (IS_DEBUG) {
        DebugWaitOnStreamLocalKernel<<<1, 1, 0, stream>>>(
            semaphores + my_pe,
            0
        );
    } else {
        nvshmemx_int64_wait_until_on_stream(
            semaphores + my_pe,
            NVSHMEM_CMP_EQ,
            0,
            stream
        );
    }
}

/**
 * @brief Tell all other PEs that the local PE has finished using their data
    so that the semaphore value on the specific PE is decreased by 1. 

    The behavior is simple: set all semaphores[i] except i = my_pe, to 0, locally.
    So that the next remote_get kernel on comm_stream will know that there is no
    data available (before we do copy on aux_stream). Also, decrease all semaphores[i]
    (i != my_pe) by 1, so other PEs will know that their local data has one few
    dependent PE. If 0 is reached, they can start clean up. 

 * @param semaphores int semaphores allocated by nvshmem: size is total_n_pes
 * @param my_pe except for semaphores[my_pe], for all other local semaphores: set zero
    , and for remote semaphores: decrease (data ref_cnt) by 1
 * @param stream waiting stream. This API is therefore async on stream (if non-blocking)
*/
void notify_all_empty(
    int64_t* const __restrict__ semaphore,
    int my_pe,
    int total_pes,
    cudaStream_t stream
) {
    NotifySemaphoreEmptyKernel<<<1, total_pes, 0, stream>>>(semaphore, my_pe);
}

void notify_segment_empty(
    int64_t* const __restrict__ semaphore,
    int my_pe,
    int start_rank,
    int chunk_per_seg,
    int total_pes,
    cudaStream_t stream
) {
    NotifySegmentSemaphoreEmptyKernel<<<1, chunk_per_seg, 0, stream>>>(semaphore, my_pe, start_rank, total_pes);
}

/**
 * Hierarchical notify_full: sets refcounts and broadcasts to congruence partners + same-node ranks.
 *
 * Dual-refcount protocol (local writes by thread 0):
 *   sema_intra[my_pe] = producer refcount = (num_nodes-1) + (gpus_per_node-1)
 *   sema_intra[partner] = relay refcount = (gpus_per_node-1), for each congruence partner
 *     (partner's data will be relayed through us; same-node ranks will read and decrement)
 *
 * Remote signals (threads 1..N):
 *   sema_inter[my_node_id] = 1 on each congruence partner  (cross-node data ready)
 *   sema_intra[my_pe] = all_batch_bits on each same-node rank  (local chunk ready)
 *
 * Precondition: wait_self_empty_hier has ensured all local sema_intra entries being
 *   written here are already 0 (previous round fully consumed).
 */
__global__ void HierSetFullKernel(
    int64_t* const __restrict__ sema_inter,   // [num_nodes] cross-node flags
    int64_t* const __restrict__ sema_intra,   // [total_n_pes] refcount + intra-node signals
    int64_t refcount,
    int self_rank,
    int my_pe_node,
    int my_node_id,
    int num_nodes,
    int gpus_per_node,
    int num_batch
) {
    if (threadIdx.x == 0) {
        // Producer refcount: decremented by congruence partners + same-node ranks
        sema_intra[self_rank] = refcount;
        // Relay refcounts: each congruence partner's data will be relayed through us,
        // and (gpus_per_node-1) same-node ranks will read the relay then decrement.
        for (int i = 1; i < num_nodes; i++) {
            int partner = my_pe_node + ((my_node_id + i) % num_nodes) * gpus_per_node;
            sema_intra[partner] = static_cast<int64_t>(gpus_per_node - 1);
        }
    }
    __threadfence();
    __syncthreads();

    int tid = threadIdx.x;
    if (tid == 0) return;

    const int congruence_count = num_nodes - 1;
    if (tid <= congruence_count) {
        // Congruence partner: same node-local index, different node
        int node_offset = tid;  // 1..num_nodes-1
        int target_rank = my_pe_node + ((my_node_id + node_offset) % num_nodes) * gpus_per_node;
        // Write sema_inter[my_node_id] = 1 on target (tells them "node my_node_id's data is ready")
        nvshmem_int64_p(sema_inter + my_node_id, 1, target_rank);
    } else if (tid <= congruence_count + gpus_per_node - 1) {
        // Same-node rank: different node-local index, same node
        int slot = tid - congruence_count;  // 1..gpus_per_node-1
        int base = (my_pe_node + slot) % gpus_per_node;
        int target_rank = base + my_node_id * gpus_per_node;
        const int64_t all_batch_bits = make_all_batch_bits(num_batch);
        // Write all-batch-bits on target (all batches ready for my local KV)
        nvshmem_int64_p(sema_intra + self_rank, all_batch_bits, target_rank);
    }
}

void notify_full(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    int total_pes,
    nvshmem_team_t team,
    cudaStream_t stream
) {
    int64_t bit_val = (1LL << total_pes) - (1LL << my_pe) - 1;
    // make sure local store is visible to other ranks and notify other PE that data is ready (full)
    SetFullKernel<<<1, total_pes, 0, stream>>>(semaphores, bit_val, my_pe);
}

/**
 * Hierarchical notify_full: sets reference count = (num_nodes - 1) + (gpus_per_node - 1).
 * Broadcasts to:
 *   - sema_inter[my_node_id] = 1 on congruence partners (Phase 1 cross-node signal)
 *   - sema_intra[my_pe] = 1 on same-node ranks (Phase 2 local chunk signal)
 */
void notify_full_hier(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    int my_pe,
    int total_pes,
    int gpus_per_node,
    int num_nodes,
    int num_batch,
    cudaStream_t stream
) {
    const int my_pe_node = my_pe % gpus_per_node;
    const int my_node_id = my_pe / gpus_per_node;
    // Phase 1 consumers (num_nodes - 1) + Phase 2 consumers reading local chunk (gpus_per_node - 1)
    const int64_t refcount = hier::hier_local_refcount(total_pes, gpus_per_node);
    // Thread count: 1 (self) + congruence partners + same-node ranks
    const int num_notify = 1 + refcount;
    HierSetFullKernel<<<1, num_notify, 0, stream>>>(
        sema_inter, sema_intra, refcount, my_pe, my_pe_node, my_node_id, num_nodes, gpus_per_node, num_batch);
}

/**
 * Hierarchical post-AG cleanup kernel.
 * Runs after AG remote_get kernel completes on comm_stream.
 *
 * Dual-refcount protocol:
 *   - Producer refcount: sema_intra[X] on producer PE X = (num_nodes-1) + (gpus_per_node-1)
 *     Decremented by: congruent ranks (Phase 1 consumers) + same-node ranks (Phase 2 direct readers)
 *   - Relay refcount: sema_intra[X] on relay holder PE = (gpus_per_node-1)
 *     Set by AG kernel's congruence_notify after Phase 1 fetch.
 *     Decremented by: same-node non-congruent ranks that read the relay copy.
 *
 * Thread layout: threadIdx.x = 0..total_n_pes-1
 *
 * Three cases per target:
 *   1. Congruent (same slot, different node): Phase 1 target.
 *      Clear sema_inter, decrement producer's refcount on target PE.
 *   2. Non-congruent, same node: Phase 2 direct read from producer.
 *      Wait for signal, clear local sema_intra, decrement producer's refcount on target PE.
 *   3. Non-congruent, different node: Phase 2 relay read from same-node relay holder.
 *      Wait for signal, clear local sema_intra, decrement relay holder's relay refcount.
 */
__global__ void HierNotifyEmptyKernel(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    const int my_pe,
    const int total_n_pes,
    const int num_nodes,
    const int gpus_per_node,
    const int num_batch
) {
    const int target = threadIdx.x;
    const int my_slot = my_pe % gpus_per_node;
    const int my_node = my_pe / gpus_per_node;

    if (target == my_pe) return;

    const int target_slot = target % gpus_per_node;
    const int target_node = target / gpus_per_node;

    if (target_slot == my_slot) {
        // Case 1: Congruent (Phase 1 target, cross-node).
        sema_inter[target_node] = 0;
        nvshmem_long_atomic_add(sema_intra + target, -1, target);
    } else {
        const int64_t all_batch_bits = make_all_batch_bits(num_batch);
        wait_full_all_batch(sema_intra, all_batch_bits, target);
        sema_intra[target] = 0;
        // same node: no remote_pe relay. Othewise we should calculate the relay rank
        int remote_rank = target_node == my_node ? target : target_slot + my_node * gpus_per_node;
        nvshmem_long_atomic_add(sema_intra + target, -1, remote_rank);
    }
}

/**
 * Hierarchical post-AG cleanup for splitted (BWD RS-overlap) AG.
 * Same logic as HierNotifyEmptyKernel but only processes ranks in the current segment.
 *
 * Thread layout: threadIdx.x = 0..chunk_per_seg-1, maps to segment's logical positions.
 */
__global__ void HierNotifySegmentEmptyKernel(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    const int my_pe,
    const int segment_idx,
    const int num_chunks,
    const int total_n_pes,
    const int gpus_per_node,
    const int num_batch
) {
    const int my_slot = my_pe % gpus_per_node;

    // Map threadIdx.x to the global logical position in this segment
    const int local_chunk_id = threadIdx.x;
    const int global_logical_pos = segment_idx * num_chunks + local_chunk_id;
    if (global_logical_pos == 0) return;  // local chunk, nothing to do

    const int target = hier::hier_target_rank(global_logical_pos, my_pe, total_n_pes, gpus_per_node);
    if (target == my_pe) return;

    const int target_slot = target % gpus_per_node;
    const int target_node = target / gpus_per_node;

    const int my_node = my_pe / gpus_per_node;

    if (target_slot == my_slot) {
        // Case 1: Congruent (Phase 1 target, cross-node).
        sema_inter[target_node] = 0;
        // Do NOT zero sema_intra[target] — relay refcount managed by Phase 2 consumers.
        nvshmem_long_atomic_add(sema_intra + target, -1, target);
    } else {
        const int64_t all_batch_bits = make_all_batch_bits(num_batch);
        wait_full_all_batch(sema_intra, all_batch_bits, target);
        sema_intra[target] = 0;
        // same node: no remote_pe relay. Othewise we should calculate the relay rank
        int remote_rank = target_node == my_node ? target : target_slot + my_node * gpus_per_node;
        nvshmem_long_atomic_add(sema_intra + target, -1, remote_rank);
    }
}

void notify_all_empty_hier(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    int my_pe,
    int total_pes,
    int gpus_per_node,
    int num_nodes,
    int num_batch,
    cudaStream_t stream
) {
    HierNotifyEmptyKernel<<<1, total_pes, 0, stream>>>(
        sema_inter, sema_intra, my_pe, total_pes, num_nodes, gpus_per_node, num_batch);
}

void notify_segment_empty_hier(
    int64_t* const __restrict__ sema_inter,
    int64_t* const __restrict__ sema_intra,
    int my_pe,
    int segment_idx,
    int num_chunks,
    int total_pes,
    int gpus_per_node,
    int num_batch,
    cudaStream_t stream
) {
    HierNotifySegmentEmptyKernel<<<1, num_chunks, 0, stream>>>(
        sema_inter, sema_intra, my_pe, segment_idx, num_chunks, total_pes, gpus_per_node, num_batch);
}

__global__ void WaitSelfEmptyHierarchicalKernel(
    int64_t* const __restrict__ sema_intra,
    int my_pe_node,
    int my_node_id,
    int gpus_per_node
) {
    int partner = my_pe_node + ((my_node_id + threadIdx.x) % blockDim.x) * gpus_per_node;
    int64_t current_val;
    do {
        asm volatile("ld.volatile.global.s64 %0, [%1];"
            : "=l"(current_val) : "l"(sema_intra + partner) : "memory");
    } while (current_val);
}

/**
 * @brief CPU wait until both producer refcount and all relay refcounts reach 0.
 *   Waits for sema_intra[my_pe] == 0 (producer refcount) and
 *   sema_intra[each_congruence_partner] == 0 (relay refcounts).
 *   Must be called before HierSetFullKernel to guarantee no data overwrite.
 */
void wait_self_empty_hier(
    int64_t* const __restrict__ sema_intra,
    int my_pe,
    int gpus_per_node,
    int num_nodes,
    cudaStream_t stream
) {
    const int my_pe_node = my_pe % gpus_per_node;
    const int my_node_id = my_pe / gpus_per_node;
    WaitSelfEmptyHierarchicalKernel<<<1, num_nodes, 0, stream>>>(
        sema_intra, my_pe_node, my_node_id, gpus_per_node
    );
}

}   // namespace ag
}   // namespace sema
}   // namespace flashmask