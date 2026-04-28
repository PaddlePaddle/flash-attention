#pragma once
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "hierarchical_rank_map.cuh"

namespace flashmask {
namespace sema {
namespace ag {

/**
 * @brief (Device Function) Used in `remote_get` kernels. The remote get kernels need to
   wait for the readiness of data. For example: PE0 will first get data from PE3, but PE3
   need to finish cudaMemcpyAsync (local KV) to the SR buffer so that the data is not dirty.
   We only need to wait for non-zero status, since if one PE is ready, it will broadcast its
   status (`total_pes - 1`) to other ranks.

 * @param semaphores int semaphores allocated by nvshmem: size is total_n_pes
 * @param target_pe the rank to get data from, 
*/
__device__ __forceinline__ void wait_full(
    const int64_t* const __restrict__ semaphores,
    const int target_pe
) {
    nvshmem_int64_wait_until(const_cast<int64_t*>(semaphores) + target_pe, NVSHMEM_CMP_GT, 0);   // wait until not 0
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
 * @brief Tell all other PEs that the local PE has prepared the data.

    The behavior is simple: First: set local buffer to `total_pes - 1`. Then broadcast
    to all the `semaphores[my_pe]` position, so that other PEs will know we are ready and
    can start getting data from the local PE.

 * @param semaphores int semaphores allocated by nvshmem: size is total_n_pes
 * @param my_pe local rank of the semaphore
 * @param stream waiting stream. This API is therefore async on stream (if non-blocking)
*/
/**
 * Hierarchical notify_full: sets reference count and broadcasts to congruence partners + same-node ranks.
 *
 * Semaphore protocol:
 *   semaphores[my_pe] on producer = (num_nodes - 1) + (gpus_per_node - 1)
 *     = Phase 1 consumers (congruence partners) + Phase 2 consumers (same-node ranks reading local chunk)
 *   semaphores[my_pe] on congruence partner = 1  (signal: local data ready for Phase 1)
 *   semaphores[my_pe] on same-node rank = 1      (signal: local chunk ready for Phase 2 direct read)
 *
 * Note: same-node ranks reading congruence partner data from src_pe's buffer
 *       wait on semaphores[partner_rank], which is set by congruence_notify, NOT by this function.
 *
 * Thread layout: thread 0 sets local semaphore; threads 1..num_notify-1 broadcast to targets.
 *   Targets: congruence partners (num_nodes - 1) then same-node ranks (gpus_per_node - 1).
 */
__global__ void HierSetFullKernel(
    int64_t* const __restrict__ semaphores,
    int64_t refcount,
    int self_rank,
    int my_pe_node,
    int my_node_id,
    int num_nodes,
    int gpus_per_node,
    int total_n_pes
) {
    if (threadIdx.x == 0) {
        semaphores[self_rank] = refcount;
    }
    __threadfence();
    __syncthreads();

    int tid = threadIdx.x;
    if (tid == 0) return;

    const int congruence_count = num_nodes - 1;
    int target_rank = -1;

    if (tid <= congruence_count) {
        // Congruence partner: same node-local index, different node
        int node_offset = tid;  // 1..num_nodes-1
        target_rank = my_pe_node + ((my_node_id + node_offset) % num_nodes) * gpus_per_node;
    } else if (tid <= congruence_count + gpus_per_node - 1) {
        // Same-node rank: different node-local index, same node
        int slot = tid - congruence_count;  // 1..gpus_per_node-1
        int base = (my_pe_node + slot) % gpus_per_node;
        target_rank = base + my_node_id * gpus_per_node;
    } else {
        return;  // excess thread
    }

    if (target_rank != self_rank) {
        nvshmem_int64_p(semaphores + self_rank, 1, target_rank);
    }
}

/**
 * Congruence notify: after src_pe completes Phase 1 fetch for a congruence partner,
 * it signals same-node ranks that the partner's data is now available in src_pe's SR buffer.
 *
 * Sets semaphores[partner] = congruence_refcount on src_pe's local copy,
 * then broadcasts 1 to each same-node rank's semaphores[partner].
 *
 * Thread layout: thread 0 sets local semaphore; threads 1..gpus_per_node-1 broadcast to same-node ranks.
 */
__global__ void CongruenceNotifyKernel(
    int64_t* const __restrict__ semaphores,
    int64_t refcount,
    int partner_rank,
    int my_pe,
    int my_pe_node,
    int gpus_per_node
) {
    if (threadIdx.x == 0) {
        semaphores[partner_rank] = refcount;
    }
    __threadfence();
    __syncthreads();

    // Threads 1..gpus_per_node-1 notify same-node ranks
    int tid = threadIdx.x;
    if (tid == 0 || tid >= gpus_per_node) return;

    // Same-node rank: different node-local index, same node
    int base = (my_pe_node + tid) % gpus_per_node;
    int target_rank = base + (my_pe / gpus_per_node) * gpus_per_node;
    if (target_rank != my_pe) {
        nvshmem_int64_p(semaphores + partner_rank, 1, target_rank);
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
 * Broadcasts to congruence partners (Phase 1 consumers) AND same-node ranks (Phase 2 local chunk consumers).
 * Same-node ranks reading congruence partner data get signaled later via congruence_notify.
 */
void notify_full_hier(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    int total_pes,
    int gpus_per_node,
    int num_nodes,
    cudaStream_t stream
) {
    const int my_pe_node = my_pe % gpus_per_node;
    const int my_node_id = my_pe / gpus_per_node;
    // Phase 1 consumers (num_nodes - 1) + Phase 2 consumers reading local chunk (gpus_per_node - 1)
    const int64_t refcount = hier::hier_local_refcount(total_pes, gpus_per_node);
    // Thread count: 1 (self) + congruence partners + same-node ranks
    const int num_notify = 1 + refcount;
    HierSetFullKernel<<<1, num_notify, 0, stream>>>(
        semaphores, refcount, my_pe, my_pe_node, my_node_id, num_nodes, gpus_per_node, total_pes);
}

/**
 * Congruence notify: src_pe signals same-node ranks that partner's data is available.
 * Called after src_pe completes all Phase 1 works for a congruence partner.
 *
 * @param partner_rank  The congruence partner whose data src_pe has fetched
 * @param gpus_per_node Number of GPUs per node
 */
void congruence_notify(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    int partner_rank,
    int gpus_per_node,
    cudaStream_t stream
) {
    const int my_pe_node = my_pe % gpus_per_node;
    // Phase 2 consumers: gpus_per_node - 1 same-node ranks
    const int64_t refcount = hier::hier_congruence_refcount(gpus_per_node);
    // Thread count: 1 (self, sets local value) + gpus_per_node - 1 (same-node ranks)
    CongruenceNotifyKernel<<<1, gpus_per_node, 0, stream>>>(
        semaphores, refcount, partner_rank, my_pe, my_pe_node, gpus_per_node);
}

}   // namespace ag
}   // namespace sema
}   // namespace flashmask