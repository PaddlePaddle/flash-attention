#pragma once

#if FLASH_ATTN_WITH_TORCH
#include <ATen/cuda/CUDAContext.h>
#endif

#include "../static_switch.h"
#include "../calc_reduced_attn_scores_kernel.h"
#include "../cuda_utils.h"

namespace reduced_scores {
  template<typename T, int Headdim> void run_(Params &params, cudaStream_t stream);

  template<typename Kernel_traits>
  void run(Params &params, cudaStream_t stream) {
      const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
      dim3 grid_m(num_m_block, params.b, params.h);
      const int num_n_block = params.num_splits == 1 ? params.num_splits : (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
      dim3 grid_n(num_n_block, params.b, params.h);
  
      // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
      // a multiple of kBlockN, we'll need to apply mask in the loop.
      const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
      const bool is_even_K = params.d == Kernel_traits::kHeadDim;
      constexpr int smem_size = Kernel_traits::kSmemSize;
      const bool return_softmax = params.p_ptr != nullptr;
      BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
          BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
              BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                      auto kernel = &run_seqk_parallel<Kernel_traits, IsEvenMNConst, IsEvenKConst, ReturnSoftmaxConst>;
                      if (smem_size >= 48 * 1024)  {
                          C10_CUDA_CHECK(cudaFuncSetAttribute(
                              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                      }
                      kernel<<<grid_n, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                      C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
          });
      });
  }
  
  template<typename T>
  void run_hdim32(Params &params, cudaStream_t stream) {
      constexpr int Headdim = 32;
      int device;
      cudaGetDevice(&device);
      int max_smem_per_block;
      cudaError status_ = cudaDeviceGetAttribute(
          &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  
      if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
          run<Kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>>(params, stream);
      } else {  // 96 KB
          run<Kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>>(params, stream);
      }
  }
  
  template<typename T>
  void run_hdim64(Params &params, cudaStream_t stream) {
      constexpr int Headdim = 64;
      int device;
      cudaGetDevice(&device);
      int max_smem_per_block;
      cudaError status_ = cudaDeviceGetAttribute(
          &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      // This is slightly faster. We want to split M more so we need fewer registers to store LSE.
      if (max_smem_per_block >= 144 * 1024) {
          run<Kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>>(params, stream);
      // This has a lot of register spilling
      } else {
          run<Kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>>(params, stream);
      }
  }
  
  template<typename T>
  void run_hdim96(Params &params, cudaStream_t stream) {
      constexpr int Headdim = 96;
      int device;
      cudaGetDevice(&device);
      int max_smem_per_block;
      cudaError status_ = cudaDeviceGetAttribute(
          &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      if (max_smem_per_block >= 116 * 1024) {
          run<Kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>>(params, stream);
      } else {
          run<Kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>>(params, stream);
      }
  }
  
  template<typename T>
  void run_hdim128(Params &params, cudaStream_t stream) {
      constexpr int Headdim = 128;
      int device;
      cudaGetDevice(&device);
      int max_smem_per_block;
      cudaError status_ = cudaDeviceGetAttribute(
          &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      // This is faster, in the case of sequence-parallel bwd (where we need fewer registers).
      // Out of these three, the 2nd one is slightly faster (2% faster than the first). Idk why.
      // run<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 2, 2, false, false, T>>(params, stream);
      if (max_smem_per_block >= 144 * 1024) {
          run<Kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>>(params, stream);
      } else {
          run<Kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>>(params, stream);
      }
  }
  
  template<typename T>
  void run_hdim160(Params &params, cudaStream_t stream) {
      constexpr int Headdim = 160;
      int device;
      cudaGetDevice(&device);
      int max_smem_per_block;
      cudaError status_ = cudaDeviceGetAttribute(
          &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      if (max_smem_per_block >= 116 * 1024) {
          run<Kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>>(params, stream);
      } else {
          run<Kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, true, T>>(params, stream);
      }
  }
  
  template<typename T>
  void run_hdim192(Params &params, cudaStream_t stream) {
      constexpr int Headdim = 192;
      int device;
      cudaGetDevice(&device);
      int max_smem_per_block;
      cudaError status_ = cudaDeviceGetAttribute(
          &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      if (max_smem_per_block >= 136 * 1024) {
          run<Kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>>(params, stream);
      } else {
          run<Kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>>(params, stream);
      }
  }
  
  template<typename T>
  void run_hdim224(Params &params, cudaStream_t stream) {
      constexpr int Headdim = 224;
      run<Kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>>(params, stream);
  }
  
  template<typename T>
  void run_hdim256(Params &params, cudaStream_t stream) {
      constexpr int Headdim = 256;
      int device;
      cudaGetDevice(&device);
      int max_smem_per_block;
      cudaError status_ = cudaDeviceGetAttribute(
          &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
      if (max_smem_per_block >= 176 * 1024) {  // H100
          run<Kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>>(params, stream);
      } else {  // A100, we don't do double buffering to save smem
          run<Kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>>(params, stream);
      }
  }
} // namespace reduced_scores
