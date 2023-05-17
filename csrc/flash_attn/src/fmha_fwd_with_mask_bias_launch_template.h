// Copyright (c) 2022, Tri Dao.

#pragma once

#include <vector>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "static_switch.h"
#include "fmha.h"
#include "fmha_fprop_kernel_1xN.h"
#include "cuda_utils.h"

template<typename Kernel_traits, bool Is_dropout>
__global__ void fmha_fprop_fp16_sm80_loop_kernel(FMHA_fprop_params params,
                                                 const bool need_attn_mask,
                                                 const bool need_attn_bias) {
    fmha::device_1xN_loop_with_mask_bias<Kernel_traits, Is_dropout, false, false>(
        params, need_attn_mask, need_attn_bias);
}

template<typename Kernel_traits>
bool run_fmha_fp16_sm80_loop_(Launch_params<FMHA_fprop_params> &launch_params,
                              const bool configure) {
    if (launch_params.params.is_causal || launch_params.return_softmax) {
        // Only support the implementation for is_causal = false and return_softmax = false.
        return false;
    }

    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    const int loop_steps = (launch_params.params.seqlen_k + blocksize_c - 1) / blocksize_c;
    if (configure) {
        using Mma_tile_p = fmha::Hmma_tile<typename Kernel_traits::Cta_tile_p>;
        constexpr int M = Kernel_traits::Cta_tile_p::M;
        size_t STEPS = (launch_params.params.seqlen_q + M - 1) / M;
        constexpr size_t MMAS_M = Mma_tile_p::MMAS_M;
        constexpr size_t MMAS_N = Mma_tile_p::MMAS_N;
        size_t elts_per_head = STEPS * MMAS_M * MMAS_N * 8 * loop_steps;
        launch_params.elts_per_thread = elts_per_head;
        return true;
    }

    constexpr int smem_size_softmax_lse = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;
    // Don't need smem_size_softmax_lse if we're not looping
    const int smem_size = fmha::get_dynamic_smem_size<Kernel_traits>()
        + (loop_steps > 1 ? smem_size_softmax_lse : 0);

    bool has_attn_mask = !(launch_params.params.attn_mask_ptr == nullptr);
    bool has_attn_bias = !(launch_params.params.attn_bias_ptr == nullptr);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH_FUNC.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21
    BOOL_SWITCH_FUNC(launch_params.is_dropout, IsDropoutConst, [&] {
        auto kernel = &fmha_fprop_fp16_sm80_loop_kernel<Kernel_traits, IsDropoutConst>;
        if (smem_size >= 48 * 1024) {
            FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 grid(launch_params.params.b, launch_params.params.h);

        kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
            launch_params.params, has_attn_mask, has_attn_bias);
        FMHA_CHECK_CUDA(cudaPeekAtLastError());
    });
    return true;
}
