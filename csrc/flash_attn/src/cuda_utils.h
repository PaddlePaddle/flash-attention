#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

#define FMHA_CHECK_CUDA( call )                                                                    \
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

#define FMHA_CUDA_KERNEL_LAUNCH_CHECK() FMHA_CHECK_CUDA(cudaGetLastError())

////////////////////////////////////////////////////////////////////////////////////////////////////

static int GetCurrentDeviceId();

static int GetCudaDeviceCount();

cudaDeviceProp* GetDeviceProperties(int id);
