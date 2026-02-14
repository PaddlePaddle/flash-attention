/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 *
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

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