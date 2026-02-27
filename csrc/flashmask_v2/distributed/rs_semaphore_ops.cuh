#pragma once
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "debug_logger.cuh"

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
    int cp_size,
    int self_rank
) {
    const int target_rank = (remote_consumer_start_rank + threadIdx.x) % cp_size;
    // quiet make sure the previous put/get on this stream is done, then we can clear bit
    if (self_rank == target_rank) return;
    semaphores[target_rank] = 0;        // clear the local status (set by the remote target)
#ifdef NVSHMEM_DEBUG
    auto fetched = nvshmem_long_atomic_fetch_add(semaphores + target_rank, -(1 << self_rank), target_rank);
    DEBUG_PRINT("Producer %d notifies remote %d full, fetched: %ld\n", self_rank, target_rank, fetched);
#else
    nvshmem_long_atomic_add(semaphores + target_rank, -(1 << self_rank), target_rank);
#endif  // NVSHMEM_DEBUG
}

__global__ void FusedConsumerNotifyEmpty(
    int64_t* const __restrict__ semaphores,
    int remote_producer_end_rank,
    int cp_size,
    int value,
    int self_rank
) {
    // for example: rank 3 local consumer needs the data from [12, 15] (remote producer) for seg 1
    // remote_producer_end_rank will be 15 (computed by mod_cp_size(3 - 4 * seg_idx) --> (-1 % 16) --> 15)
    if (threadIdx.x == 0) {
        semaphores[self_rank] = value;
        DEBUG_PRINT("Consumer %d fused, sets self empty value: %d, \
            cp_size: %d, end_rank: %d\n", self_rank, value, cp_size, remote_producer_end_rank);
    }
    // the following fence makes sure semaphore setting is visible across all CP ranks 
    __threadfence();
    __syncwarp();
    int target_rank = remote_producer_end_rank - threadIdx.x;
    target_rank = target_rank >= 0 ? target_rank : target_rank + cp_size;
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
void notify_consumer_empty(
    int64_t* const semaphores,
    int remote_producer_end_rank,
    int seg_size,
    int cp_size,
    int self_rank,
    cudaStream_t comm_stream
) {
    int local_flag = 0;
    for (int i = 0; i < seg_size; i++) {
        int target_rank = remote_producer_end_rank - i;
        target_rank = target_rank >= 0 ? target_rank : target_rank + cp_size;
        local_flag |= target_rank == self_rank ? 0 : (1 << target_rank);
    }
    // Fused step 1 & 2 in one kernel. Should the following gets buggy, you can revert to 1540b3438fb8 for testing
    // step 1. set self (inform reduce kernel that we haven't got data from other ranks, so we wait)
    // step 2. notify all other src ranks: you can start putting data to this rank
    // for example: local_rank is 7, we notify rank 0,1,2,3 to put data by setting sema[7] to 1
    // set remote empty state can not start before we set the local state
    // otherwise there will be corrupted read-write
    FusedConsumerNotifyEmpty<<<1, seg_size, 0, comm_stream>>>(semaphores,
                    remote_producer_end_rank, cp_size, local_flag, self_rank);
}

// self rank notifies all remote consumers that needs dK, dV data 
// from the local rank that the data is ready (sent).
// Also, clear all the local status
void producer_commit_all(
    int64_t* const semaphores,
    int remote_consumer_start_rank,
    int cp_size,
    int self_rank,
    int chunks_per_seg,
    cudaStream_t comm_stream
) {
    ProducerNotifyFull<<<1, chunks_per_seg, 0, comm_stream>>>(
        semaphores, remote_consumer_start_rank, cp_size, self_rank);
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
    cudaStream_t comm_stream
) {
    nvshmemx_int64_wait_until_on_stream(
        semaphores + my_pe,
        NVSHMEM_CMP_EQ,
        0,
        comm_stream
    );
}

__device__ void producer_wait_empty(
    const int64_t* const __restrict__ semaphores,
    const int target_pe
) {
    WARN_PRINT("Producer block %d waits remote %d empty ...\n", blockIdx.x, target_pe);
    nvshmem_int64_wait_until(const_cast<int64_t*>(semaphores) + target_pe, NVSHMEM_CMP_NE, 0);   // wait until not 0
    WARN_PRINT("Producer block %d waits remote %d empty succeeded.\n", blockIdx.x, target_pe);
}

}   // namespace rs
}   // namespace sema
}   // namespace flashmask