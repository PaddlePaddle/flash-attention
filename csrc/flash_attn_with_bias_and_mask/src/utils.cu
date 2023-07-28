#include "utils.h"

void SetZero(void *ptr, size_t sizeof_type, std::initializer_list<int> shapes, cudaStream_t stream) {
    size_t n = sizeof_type;
    for (int s : shapes) n *= s;
    FMHA_CHECK_CUDA(cudaMemsetAsync(ptr, 0, n, stream));
}

template <typename T>
static __global__ void FillConstantKernel(T *ptr, T value, size_t n) {
  auto idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  if (idx < n) {
    ptr[idx] = value;
  }
} 

template <typename T>
void SetConstValue(void *ptr, T value, size_t n, cudaStream_t stream) {
  constexpr auto kNumThreads = 1024;
  auto block = (n + kNumThreads - 1) / kNumThreads; 
  FillConstantKernel<T><<<block, kNumThreads, 0, stream>>>(static_cast<T *>(ptr), value, n);
} 

template
void SetConstValue(void *ptr, float value, size_t n, cudaStream_t stream);

static __global__ void _float2half(float *float_ptr, __half *half_ptr, size_t n) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    half_ptr[idx] = __float2half(float_ptr[idx]);
  }
}

void Float2Half(void *float_ptr, void *half_ptr, size_t n, cudaStream_t stream) {
  constexpr auto kNumThreads = 1024;
  auto block = (n + kNumThreads - 1) / kNumThreads; 
  _float2half<<<block, kNumThreads, 0, stream>>>(static_cast<float *>(float_ptr), static_cast<__half *>(half_ptr), n);
} 

static __global__ void _float2bfloat16(float *float_ptr, __nv_bfloat16 *bf16_ptr, size_t n) {
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    bf16_ptr[idx] = __float2bfloat16(float_ptr[idx]);
  }
}

void Float2BF16(void *float_ptr, void *bf16_ptr, size_t n, cudaStream_t stream) {
  constexpr auto kNumThreads = 1024;
  auto block = (n + kNumThreads - 1) / kNumThreads; 
  _float2bfloat16<<<block, kNumThreads, 0, stream>>>(static_cast<float *>(float_ptr), static_cast<__nv_bfloat16 *>(bf16_ptr), n);
} 
