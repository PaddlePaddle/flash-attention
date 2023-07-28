#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <initializer_list>

#include "fmha_utils.h"

void SetZero(void *ptr, size_t sizeof_type, std::initializer_list<int> shapes, cudaStream_t stream);

template <typename T>
void SetConstValue(void *ptr, T value, size_t n, cudaStream_t stream);

void Float2Half(void *float_ptr, void *half_ptr, size_t n, cudaStream_t stream);
void Float2BF16(void *float_ptr, void *bf16_ptr, size_t n, cudaStream_t stream);
