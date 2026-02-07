#pragma once
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

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
    nvshmem_long_atomic_add(semaphores + target_rank, -(1 << self_rank), target_rank);
}

__global__ void ConsumerNotifyEmpty(
    int64_t* const __restrict__ semaphores,
    int remote_producer_end_rank,
    int cp_size,
    int self_rank
) {
    // for example: rank 3 local consumer needs the data from [12, 15] (remote producer) for seg 1
    // remote_producer_end_rank will be 15 (computed by mod_cp_size(3 - 4 * seg_idx) --> (-1 % 16) --> 15)
    int target_rank = remote_producer_end_rank - threadIdx.x;
    target_rank = target_rank >= 0 ? target_rank : target_rank + cp_size;
    if (target_rank == self_rank) return;
    nvshmem_int64_p(semaphores + self_rank, 1, target_rank);
}

__global__ void Print16Semaphores(
    int64_t* const semaphores,
    int self_rank
) {
    printf("rank %d: [%lx, %lx, %lx, %lx], [%lx, %lx, %lx, %lx], [%lx, %lx, %lx, %lx], [%lx, %lx, %lx, %lx]\n", 
        self_rank,
        semaphores[0], semaphores[1], semaphores[2], semaphores[3],
        semaphores[4], semaphores[5], semaphores[6], semaphores[7],
        semaphores[8], semaphores[9], semaphores[10], semaphores[11],
        semaphores[12], semaphores[13], semaphores[14], semaphores[15]
    );
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

__global__ void PrintWritePtrKernel(
    int* const wptr,
    int self_rank
) {
    printf("rank %d: debug print wptr: %d, int max is: %d\n", self_rank, wptr[0], INT_MAX);
}

void debug_print_semaphore(
    int64_t* const semaphores,
    int self_rank,
    cudaStream_t comm_stream
) {
    Print16Semaphores<<<1, 1, 0, comm_stream>>>(semaphores, self_rank);
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
    int64_t local_flag = 0;
    for (int i = 0; i < seg_size; i++) {
        int target_rank = remote_producer_end_rank - i;
        target_rank = target_rank >= 0 ? target_rank : target_rank + cp_size;
        local_flag |= target_rank == self_rank ? 0 : (1 << target_rank);
    }
    // step 1. set self (inform reduce kernel that we haven't got data from other ranks, so we wait)
    SetValueKernel<<<1, 1, 0, comm_stream>>>(semaphores + self_rank, local_flag);
    // step 2. notify all other src ranks: you can start putting data to this rank
    // for example: local_rank is 7, we notify rank 0,1,2,3 to put data by setting sema[7] to 1
    // set remote empty state can not start before we set the local state
    // otherwise there will be corrupted read-write
    ConsumerNotifyEmpty<<<1, seg_size, 0, comm_stream>>>(semaphores, remote_producer_end_rank, cp_size, self_rank);
    nvshmemx_quiet_on_stream(comm_stream);
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

// self rank notifies one specified consumer that needs dK, dV data 
// from the local rank that the data is ready (sent)
void producer_commit_one(
    int64_t* const semaphores,
    int target_rank,
    int cp_size,
    int self_rank,
    cudaStream_t comm_stream
) {
    ProducerNotifyFull<<<1, 1, 0, comm_stream>>>(
        semaphores, target_rank, cp_size, self_rank);
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
    nvshmem_int64_wait_until(const_cast<int64_t*>(semaphores) + target_pe, NVSHMEM_CMP_NE, 0);   // wait until not 0
}

}   // namespace rs
}   // namespace sema
}   // namespace flashmask