#pragma once
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

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
        // nvshmem_int64_atomic_and(semaphores + threadIdx.x, ~(1 << my_pe), threadIdx.x);
        nvshmem_long_atomic_add(semaphores + threadIdx.x, -(1 << my_pe), threadIdx.x);
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
        nvshmem_long_atomic_add(semaphores + target_rank, -(1 << my_pe), target_rank);
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
    int value,
    int self_rank
) {
    if (threadIdx.x == 0) {
        semaphores[self_rank] = int64_t(value);
    }
    __threadfence();
    __syncwarp();
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
void notify_full(
    int64_t* const __restrict__ semaphores,
    int my_pe,
    int total_pes,
    nvshmem_team_t team,
    cudaStream_t stream
) {
    int bit_val = (1 << total_pes) - (1 << my_pe) - 1;
    // make sure local store is visible to other ranks and notify other PE that data is ready (full) 
    SetFullKernel<<<1, total_pes, 0, stream>>>(semaphores, bit_val, my_pe);
}

}   // namespace ag
}   // namespace sema
}   // namespace flashmask