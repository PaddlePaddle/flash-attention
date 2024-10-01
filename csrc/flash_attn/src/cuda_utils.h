#pragma once

#include <cstdio>
#include "cuda.h"
#include "cuda_runtime.h"

#if !FLASH_ATTN_WITH_TORCH
////////////////////////////////////////////////////////////////////////////////////////////////////

#define C10_CUDA_CHECK( call )                                                                    \
    do {                                                                                           \
        cudaError_t status_ = call;                                                                \
        if( status_ != cudaSuccess ) {                                                             \
            fprintf( stderr,                                                                       \
                     "CUDA error (%s:%d): %s\n",                                                   \
                     __FILE__,                                                                     \
                     __LINE__,                                                                     \
                     cudaGetErrorString( status_ ) );                                              \
            exit( 1 );                                                                             \
        }                                                                                          \
    } while( 0 )

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace at {
namespace cuda {
static int GetCurrentDeviceId();

static int GetCudaDeviceCount();

cudaDeviceProp* getCurrentDeviceProperties(int id = -1);
} // namespace cuda
} // namespace at
#endif
