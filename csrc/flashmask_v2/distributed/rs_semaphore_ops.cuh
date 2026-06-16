#pragma once
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "debug_logger.cuh"
#include "hierarchical_rank_map.cuh"

namespace flashmask {
namespace sema {
namespace rs {

// num thread: 1
template <typename SemaphoreT>
__global__ void SetValueKernel(
    SemaphoreT* const __restrict__ semaphore,
    const int value
) {
    *(semaphore + threadIdx.x) = static_cast<SemaphoreT>(value);
#ifdef NVSHMEM_DEBUG
    if (gridDim.x == 1) {
        DEBUG_PRINT("Consumer sets self empty value: %d\n", value);
    }
#endif  // NVSHMEM_DEBUG
}

__global__ void ProducerNotifyFull(
    int64_t* const __restrict__ semaphores,
    int remote_consumer_start_rank,
    int nranks,
    int self_rank
) {
    const int target_rank = (remote_consumer_start_rank + threadIdx.x) % nranks;
    // quiet make sure the previous put/get on this stream is done, then we can clear bit
    if (self_rank == target_rank) return;
    semaphores[target_rank] = 0;        // clear the local status (set by the remote target)
    const int64_t clear_mask = -(1LL << self_rank);
#ifdef NVSHMEM_DEBUG
    auto fetched = nvshmem_long_atomic_add(semaphores + target_rank, clear_mask, target_rank);
    DEBUG_PRINT("Producer %d notifies remote %d full, fetched: %ld\n", self_rank, target_rank, fetched);
#else
    nvshmem_long_atomic_add(semaphores + target_rank, clear_mask, target_rank);
#endif  // NVSHMEM_DEBUG
}

__global__ void FusedConsumerNotifyEmpty(
    int64_t* const __restrict__ semaphores,
    int remote_producer_end_rank,
    int nranks,
    int64_t value,
    int self_rank
) {
    // for example: rank 3 local consumer needs the data from [12, 15] (remote producer) for seg 1
    // remote_producer_end_rank will be 15 (computed by mod_nranks(3 - 4 * seg_idx) --> (-1 % 16) --> 15)
    if (threadIdx.x == 0) {
        semaphores[self_rank] = value;
        DEBUG_PRINT("Consumer %d fused, sets self empty value: %lx, nranks: %d, end_rank: %d\n", 
            self_rank, value, nranks, remote_producer_end_rank);
    }
    // the following fence makes sure semaphore setting is visible across all CP ranks 
    __threadfence();
    __syncthreads();
    int target_rank = remote_producer_end_rank - threadIdx.x;
    target_rank = target_rank >= 0 ? target_rank : target_rank + nranks;
    if (target_rank == self_rank) return;
    nvshmem_int64_p(semaphores + self_rank, 1, target_rank);
    DEBUG_PRINT("Consumer %d notifies remote %d empty, end_rank: %d\n", self_rank, target_rank, remote_producer_end_rank);
}

__global__ void DebugWaitAndResetKernel(
    int64_t* const volatile semaphores,
    const int64_t target_value
) {
    while (true) {
        int64_t cur_val;
        asm volatile("ld.volatile.global.s64 %0, [%1];" 
             : "=l"(cur_val) 
             : "l"(semaphores));
        if (cur_val == target_value) break;
        // printf("Current: %lx, while target: %lx\n", cur_val, target_value);
        // __nanosleep(1000000);
    }
    *semaphores = 0;
}

// [local consumer (dk dv reducer and recv buffer)] sends out an empty
// notifcation for [remote producer (put kernel)] to fill the buffer
//
// per_stage_buffer: when true, semaphores[self_rank] is a pure refcount (starts at 0,
// producers atomicAdd +1, consumer waits for == K then CAS to 0). The local value
// must remain 0 — only remote notifications are needed.
// When false, semaphores[self_rank] is a bitmask (producers clear their bit via
// atomicAdd(-(1<<rank)), consumer waits for == 0).
void notify_consumer_empty(
    int64_t* const semaphores,
    int remote_producer_end_rank,
    int seg_size,
    int nranks,
    int self_rank,
    cudaStream_t comm_stream,
    bool per_stage_buffer = false
) {
    int64_t local_flag = 0;
    if (!per_stage_buffer) {
        for (int i = 0; i < seg_size; i++) {
            int target_rank = remote_producer_end_rank - i;
            target_rank = target_rank >= 0 ? target_rank : target_rank + nranks;
            local_flag |= target_rank == self_rank ? 0 : (1LL << target_rank);
        }
    }
    FusedConsumerNotifyEmpty<<<1, seg_size, 0, comm_stream>>>(semaphores,
                    remote_producer_end_rank, nranks, local_flag, self_rank);
}

// self rank notifies all remote consumers that needs dK, dV data 
// from the local rank that the data is ready (sent).
// Also, clear all the local status
void producer_commit_all(
    int64_t* const semaphores,
    int remote_consumer_start_rank,
    int nranks,
    int self_rank,
    int chunks_per_seg,
    cudaStream_t comm_stream
) {
    ProducerNotifyFull<<<1, chunks_per_seg, 0, comm_stream>>>(
        semaphores, remote_consumer_start_rank, nranks, self_rank);
}

static __global__ void SpinWaitAndReplaceKernel(int64_t* ptr, int64_t target) {
    if (threadIdx.x > 0) return;
    while (true) {
        int64_t val;
        asm volatile("ld.global.cg.s64 %0, [%1];" 
            : "=l"(val) 
            : "l"(ptr) 
            : "memory"
        );

        if (val == target) {
            if (atomicCAS(reinterpret_cast<unsigned long long int*>(ptr), 
                          static_cast<unsigned long long int>(target), 
                          0ULL) == static_cast<unsigned long long int>(target)) {
                break;
            }
        }
        __nanosleep(16); 
    }
}

/**
 * @brief CPU wait until the semaphores[my_pe] reached 0
 * @param semaphores int semaphores allocated by nvshmem: size is total_n_pes
 * @param my_pe the id of semaphore to wait for
 * @param stream waiting stream. This API is therefore async on stream (if non-blocking)
*/
void consumer_wait_full(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    int wait_value,
    cudaStream_t comm_stream,
    bool use_per_stage_buffer = false
) {
    WARN_PRINT("Consumer wait full ...\n");
    if (use_per_stage_buffer) {
        SpinWaitAndReplaceKernel<<<1, 1, 0, comm_stream>>>(
            semaphores + my_pe, wait_value
        );
    } else {
        nvshmemx_int64_wait_until_on_stream(
            semaphores + my_pe,
            NVSHMEM_CMP_EQ,
            0,
            comm_stream
        );
    }
    WARN_PRINT_SYNC(comm_stream, "Consumer wait full succeeded.\n");
}

__device__ void producer_wait_empty(
    const int64_t* const __restrict__ semaphores,
    const int target_pe
) {
    WARN_PRINT("Producer block %d waits remote %d empty ...\n", blockIdx.x, target_pe);
    nvshmem_int64_wait_until(const_cast<int64_t*>(semaphores) + target_pe, NVSHMEM_CMP_NE, 0);   // wait until not 0
    WARN_PRINT("Producer block %d waits remote %d empty succeeded.\n", blockIdx.x, target_pe);
}

// ---------------------------------------------------------------------------
// Hierarchical variants for RS-overlap
// ---------------------------------------------------------------------------

/**
 * Hierarchical FusedConsumerNotifyEmpty.
 *
 * In hierarchical mode, the producers for my_pe are NOT a contiguous rank range.
 * For segment [logical_pos_base, logical_pos_base+num_chunks), PE X sends chunk c
 * to my_pe iff X == hier_sender_at_pos(logical_pos_base+c, my_pe, ...).
 * This kernel sets semaphores[self_rank]=local_flag (pre-computed bitmask of senders),
 * then notifies each sender that the buffer slot is empty and ready to receive.
 *
 * Launch: <<<1, num_chunks>>>
 */
__global__ void HierFusedConsumerNotifyEmpty(
    int64_t* const __restrict__ semaphores,
    int logical_pos_base,
    int num_chunks,
    int total_n_pes,
    int gpus_per_node,
    int64_t local_flag,
    int self_rank
) {
    if (threadIdx.x == 0) {
        semaphores[self_rank] = local_flag;
        DEBUG_PRINT("HierConsumer %d sets self flag: %lx\n", self_rank, local_flag);
    }
    __threadfence();
    __syncthreads();

    const int logical_pos = logical_pos_base + threadIdx.x;
    const int sender = hier::hier_sender_at_pos(logical_pos, self_rank, total_n_pes, gpus_per_node);
    if (sender == self_rank) return;  // local chunk, no notification needed
    nvshmem_int64_p(semaphores + self_rank, 1, sender);
    DEBUG_PRINT("HierConsumer %d notifies sender %d (logical_pos %d) empty\n", self_rank, sender, logical_pos);
}

/**
 * Host wrapper: compute sender bitmask and launch HierFusedConsumerNotifyEmpty.
 * per_stage_buffer: same semantics as notify_consumer_empty — when true, local_flag = 0.
 */
void notify_consumer_empty_hier(
    int64_t* const semaphores,
    int segment_idx,
    int num_chunks,
    int total_n_pes,
    int gpus_per_node,
    int self_rank,
    cudaStream_t stream,
    bool per_stage_buffer = false
) {
    const int logical_pos_base = segment_idx * num_chunks;
    int64_t local_flag = 0;
    if (!per_stage_buffer) {
        for (int c = 0; c < num_chunks; c++) {
            const int sender = hier::hier_sender_at_pos(logical_pos_base + c, self_rank, total_n_pes, gpus_per_node);
            if (sender != self_rank) local_flag |= (1LL << sender);
        }
    }
    HierFusedConsumerNotifyEmpty<<<1, num_chunks, 0, stream>>>(
        semaphores, logical_pos_base, num_chunks, total_n_pes, gpus_per_node, local_flag, self_rank
    );
}

}   // namespace rs
}   // namespace sema
}   // namespace flashmask