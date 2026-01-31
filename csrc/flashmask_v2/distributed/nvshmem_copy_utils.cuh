#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>

namespace flashmask {
namespace shmem {

template <threadgroup_t scope>
__device__ inline void threadgroup_sync() {
    switch (scope) {
        case NVSHMEMI_THREADGROUP_THREAD:
            return;
        case NVSHMEMI_THREADGROUP_WARP:
            __syncwarp();
            break;
        case NVSHMEMI_THREADGROUP_BLOCK:
            __syncthreads();
            break;
        default:
            printf("unrecognized threadscope passed\n");
            assert(0);
            break;
    }
}

template <threadgroup_t scope, typename CopyT = int4>
__device__ __forceinline__ void memcpy_two_buffer(
    void* const __restrict__ dst1, 
    void* const __restrict__ dst2, 
    const void* const __restrict__ src1, 
    const void* const __restrict__ src2, 
    int len
) {
    // we know that len is within the int32 range, so don't use 64bit integer ops
    int my_idx;
    int stride = 0;
    if constexpr (scope == NVSHMEMI_THREADGROUP_WARP) {
        asm volatile("mov.u32  %0,  %%laneid;" : "=r"(my_idx));
        stride = 32;
    } else {
        my_idx = threadIdx.x;       // make sure the block is 1D
        stride = blockDim.x;        // make sure the grid is 1D
    }
    const int nelems = len / sizeof(CopyT);

    // make sure len is the multiple of 16 (usually holds)
    // using int4, for D = 128, H = 1, bf16. A warp can only use the first 16 threads. int2 however can use the whole warp
    // the same is for block, using int4, block of 256 threads can only use 128 threads to copy. int2 can use the whole block
    CopyT* const __restrict__ dst_p1 = reinterpret_cast<CopyT*>(dst1);
    CopyT* const __restrict__ dst_p2 = reinterpret_cast<CopyT*>(dst2);
    const CopyT* const __restrict__ src_p1 = reinterpret_cast<const CopyT*>(src1);
    const CopyT* const __restrict__ src_p2 = reinterpret_cast<const CopyT*>(src2);
    for (int i = my_idx; i < nelems; i += stride) {
        dst_p1[i] = src_p1[i];
        dst_p2[i] = src_p2[i];
    }
}

/**
 * Note(heqianyue): the following getmem primitive is modified from nvshmem nvshmemi_get_threadgroup
 * For the buffers on the exact same PE (like, KV) to be fetched from, instead of calling
 * `nvshmemi_get_threadgroup` twice (where we need to sync 4 times and __ldg twice), we can merge
 * the call into one, so we only need two syncs and one __ldg.
*/
template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void __internal_two_buffer_get_threadgroup(
    char* const __restrict__ dest1,
    char* const __restrict__ dest2,
    const char* const __restrict__ source1,
    const char* const __restrict__ source2,
    const int nelems, const int pe
) {
    // If we know that the peer_base_addr (if any) is the same for two src/dst buffers
    // So we can save one load + two syncs.
    threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        const char* const source_actual1 = (char *)(peer_base_addr) +
                              (source1 - (char *)(nvshmemi_device_state_d.heap_base));
        const char* const source_actual2 = (char *)(peer_base_addr) +
                              (source2 - (char *)(nvshmemi_device_state_d.heap_base));
        memcpy_two_buffer<SCOPE, int4>(
            (void *)dest1, (void *)dest2,
            (const void *)source_actual1, (const void *)source_actual2,
            nelems              // number of bytes to copy
        );
    } else {
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_GET>((void *)source1, (void *)dest1, nelems, pe);
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_GET>((void *)source2, (void *)dest2, nelems, pe);
    }
    threadgroup_sync<SCOPE>();
}

template <threadgroup_t SCOPE>
__device__ NVSHMEMI_DEVICE_ALWAYS_INLINE void __internal_two_buffer_put_threadgroup(
    char* const __restrict__ dest1,
    char* const __restrict__ dest2,
    const char* const __restrict__ source1,
    const char* const __restrict__ source2,
    const int nelems, const int pe                                                                       
) {
    threadgroup_sync<SCOPE>();
    void *peer_base_addr =
        (void *)__ldg((const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    if (peer_base_addr) {
        char *dest_actual1 =
            (char *)(peer_base_addr) + ((char *)dest1 - (char *)(nvshmemi_device_state_d.heap_base));
        char *dest_actual2 =
            (char *)(peer_base_addr) + ((char *)dest2 - (char *)(nvshmemi_device_state_d.heap_base));
        memcpy_two_buffer<SCOPE, int4>(
            (void *)dest_actual1, (void *)dest_actual2,
            (const void *)source1, (const void *)source2,
            nelems              // number of bytes to copy
        );
    } else {
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_PUT>((void *)dest1, (void *)source1, nelems, pe);
        nvshmemi_transfer_rma<SCOPE, NVSHMEMI_OP_PUT>((void *)dest2, (void *)source2, nelems, pe);
    }
    threadgroup_sync<SCOPE>();
}

#define DEFINE_TWO_BUFFERS_MEMOP_THREADGROUP(Group, Op)                                             \
    NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void two_buffers_##Op##mem_##Group(        \
        void* __restrict__ dest1, void* __restrict__ dest2,                                         \
        const void* const __restrict__ source1,  const void* const __restrict__ source2,            \
        size_t bytes, int pe) {                                                                     \
        __internal_two_buffer_##Op##_threadgroup<nvshmemi_threadgroup_##Group>(                     \
            (char *)dest1, (char *)dest2, (const char *)source1, (const char *)source2, bytes, pe); \
    }

DEFINE_TWO_BUFFERS_MEMOP_THREADGROUP(warp, get)
DEFINE_TWO_BUFFERS_MEMOP_THREADGROUP(block, get)
DEFINE_TWO_BUFFERS_MEMOP_THREADGROUP(warp, put)
DEFINE_TWO_BUFFERS_MEMOP_THREADGROUP(block, put)

}   // namespace shmem
}   // namespace flashmask