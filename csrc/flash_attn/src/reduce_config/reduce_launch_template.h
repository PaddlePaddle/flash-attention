#pragma once

#if FLASH_ATTN_WITH_TORCH
#include <ATen/cuda/CUDAContext.h>
#endif

#include "../static_switch.h"
#include "../flash.h"
#include "../reduce_attn_scores.h"
#include "../cuda_utils.h"

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_attn_mask, bool Is_deterministic>
__global__ void reduce_attn_scores_kernel(Reduce_attn_scores_params params) {
    flash::reduce_attn_scores_seqk_parallel<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_attn_mask, Is_deterministic>(params);
}

template<typename Kernel_traits>
void run_reduce_seqk_parallel(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);
    const int num_n_block = params.num_splits == 1 ? params.num_splits : (params.seqlen_k + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    dim3 grid_n(num_n_block, params.b, params.h);

    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_q % Kernel_traits::kBlockM == 0 && params.seqlen_k % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    const bool is_attn_mask = params.attn_mask_ptr != nullptr;
    const bool is_deterministic = params.num_splits == 1;
    params.attn_mask_start_row = (int)(params.attn_mask_start_row / Kernel_traits::kBlockM) * Kernel_traits::kBlockM;
    BOOL_SWITCH(params.is_causal, IsCausalConst, [&] {
        BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
            BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
                BOOL_SWITCH(is_attn_mask, Is_attn_mask, [&] {
                    BOOL_SWITCH(is_deterministic, Is_deterministic, [&] {
                        // auto kernel = &reduce_attn_scores_kernel<Kernel_traits, IsCausalConst, IsEvenMNConst, IsEvenKConst, Is_attn_mask && !IsCausalConst, Is_deterministic>;
                        auto kernel = &flash::reduce_attn_scores_seqk_parallel<Kernel_traits, IsCausalConst, IsEvenMNConst, IsEvenKConst, Is_attn_mask && !IsCausalConst, Is_deterministic>;
                        if (smem_size_dq_dk_dv >= 48 * 1024)  {
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                        }
                        kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
}

template<typename Kernel_traits>
void run_reduce(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    if (configure) return;
        run_reduce_seqk_parallel<Kernel_traits>(params, stream, configure);
}

template<typename T>
void run_reduce_hdim32(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 32;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
        run_reduce<Reduce_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>>(params, stream, configure);
    } else {  // 96 KB
        run_reduce<Reduce_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>>(params, stream, configure);
    }
}

template<typename T>
void run_reduce_hdim64(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 64;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // This is slightly faster. We want to split M more so we need fewer registers to store LSE.
    if (max_smem_per_block >= 144 * 1024) {
        run_reduce<Reduce_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>>(params, stream, configure);
    // This has a lot of register spilling
    } else {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>>(params, stream, configure);
    }
}

template<typename T>
void run_reduce_hdim96(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 96;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_smem_per_block >= 116 * 1024) {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>>(params, stream, configure);
    } else {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>>(params, stream, configure);
    }
}

template<typename T>
void run_reduce_hdim128(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 128;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // This is faster, in the case of sequence-parallel bwd (where we need fewer registers).
    // Out of these three, the 2nd one is slightly faster (2% faster than the first). Idk why.
    // run_reduce<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 2, 2, false, false, T>>(params, stream, configure);
    if (max_smem_per_block >= 144 * 1024) {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>>(params, stream, configure);
    } else {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>>(params, stream, configure);
    }
}

template<typename T>
void run_reduce_hdim160(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 160;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_smem_per_block >= 116 * 1024) {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>>(params, stream, configure);
    } else {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, true, T>>(params, stream, configure);
    }
}

template<typename T>
void run_reduce_hdim192(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 192;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_smem_per_block >= 136 * 1024) {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>>(params, stream, configure);
    } else {
        run_reduce<Reduce_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>>(params, stream, configure);
    }
}

template<typename T>
void run_reduce_hdim224(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 224;
    run_reduce<Reduce_kernel_traits<Headdim, 64, 64, 8, 4, 4, 4, false, false, T>>(params, stream, configure);
}

template<typename T>
void run_reduce_hdim256(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    constexpr int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (max_smem_per_block >= 176 * 1024) {  // H100
        run_reduce<Reduce_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>>(params, stream, configure);
    } else {  // A100, we don't do double buffering to save smem
        run_reduce<Reduce_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>>(params, stream, configure);
    }
}
