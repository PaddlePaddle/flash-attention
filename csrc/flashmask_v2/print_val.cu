#include <cstdio>
#include "utils.h"

namespace flash{
      __global__ void print_addr_value(int* base, size_t offset_bytes) {
    int* ptr = (int*)((char*)base + offset_bytes);
    printf("Value at address %p: %d\n", ptr, *ptr);
    }

    __global__ void print_addr_value_ordered(int* base, size_t start_offset_bytes, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // 按线程ID顺序打印，避免输出混乱
    for (int current_thread = 0; current_thread < total_threads; current_thread++) {
        if (tid == current_thread && tid < count) {
            size_t offset_bytes = start_offset_bytes + tid * sizeof(int);
            int* ptr = (int*)((char*)base + offset_bytes);
            printf("Thread %d - Value at address %p (offset %zu): %d\n", 
                   tid, ptr, offset_bytes, *ptr);
        }
        __syncthreads(); // 同步保证顺序
    }
}
}