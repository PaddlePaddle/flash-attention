#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define LOG_LV_NONE 0
#define LOG_LV_DEBUG 1
#define LOG_LV_WARN 2
#define LOG_LV_ERROR 3

#define DEBUG_LOGGING LOG_LV_NONE

// #define NVSHMEM_DEBUG

inline void CheckCudaErrorAux(const char *file, unsigned line,
                                       const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err),
           err, file, line);
    exit(1);
}

#if defined(NVSHMEM_DEBUG)
#define CUDA_DEBUG_CHECK(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
#else
#define CUDA_DEBUG_CHECK(value) (value)
#endif

#if DEBUG_LOGGING > 0
    // print directly
    #define DEBUG_PRINT(fmt, ...) printf("[Debug] " fmt, ##__VA_ARGS__)

    // print with stream sync
    #define DEBUG_PRINT_SYNC(stream, fmt, ...) CUDA_DEBUG_CHECK(cudaStreamSynchronize(stream)); printf("[Debug] " fmt, ##__VA_ARGS__)
    #if DEBUG_LOGGING > 1
        // print directly
        #define WARN_PRINT(fmt, ...) printf("[Warn] " fmt, ##__VA_ARGS__)

        // print with stream sync
        #define WARN_PRINT_SYNC(stream, fmt, ...) CUDA_DEBUG_CHECK(cudaStreamSynchronize(stream)); printf("[Warn] " fmt, ##__VA_ARGS__) 
        #if DEBUG_LOGGING > 2
            // print directly
            #define ERROR_PRINT(fmt, ...) printf("[Error] " fmt, ##__VA_ARGS__)

            // print with stream sync
            #define ERROR_PRINT_SYNC(stream, fmt, ...) CUDA_DEBUG_CHECK(cudaStreamSynchronize(stream)); printf("[Error] " fmt, ##__VA_ARGS__) 
        #else
            // Do nothing since log level is Warn
            #define ERROR_PRINT(fmt, ...) ((void)0)

            // Do nothing since log level is Warn
            #define ERROR_PRINT_SYNC(fmt, ...) ((void)0)
        #endif // ERROR
    #else
        // Do nothing since log level is Debug
        #define ERROR_PRINT(fmt, ...) ((void)0)

        // Do nothing since log level is Debug
        #define ERROR_PRINT_SYNC(fmt, ...) ((void)0)

        // Do nothing since log level is Debug
        #define WARN_PRINT(fmt, ...) ((void)0)

        // Do nothing since log level is Debug
        #define WARN_PRINT_SYNC(fmt, ...) ((void)0)
    #endif // WARN
#else
    // Do nothing since log level is None
    #define ERROR_PRINT(fmt, ...) ((void)0)

    // Do nothing since log level is None
    #define ERROR_PRINT_SYNC(fmt, ...) ((void)0)

    // Do nothing since log level is None
    #define WARN_PRINT(fmt, ...) ((void)0)

    // Do nothing since log level is None
    #define WARN_PRINT_SYNC(fmt, ...) ((void)0)

    // Do nothing since log level is None
    #define DEBUG_PRINT(fmt, ...) ((void)0)

    // Do nothing since log level is None
    #define DEBUG_PRINT_SYNC(fmt, ...) ((void)0)
#endif // DEBUG_LOGGING