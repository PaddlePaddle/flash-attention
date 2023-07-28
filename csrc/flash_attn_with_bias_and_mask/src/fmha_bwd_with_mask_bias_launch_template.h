// Copyright (c) 2022, Tri Dao.

#pragma once

#include "static_switch.h"
#include "fmha.h"
#include "fmha_dgrad_kernel_1xN_loop.h"
#include "cuda_utils.h"

template<typename Kernel_traits, bool Is_dropout, int loop_steps=-1>
__global__ void fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel(FMHA_dgrad_params params,
                                                          const bool need_attn_mask,
                                                          const bool need_attn_bias) {
    fmha::compute_dq_dk_dv_1xN_with_bias_mask<Kernel_traits, Is_dropout, false, loop_steps>(
        params, need_attn_mask, need_attn_bias);
}

template<typename Kernel_traits>
bool run_fmha_dgrad_fp16_sm80_loop_(const FMHA_dgrad_params &params, cudaStream_t stream) {
    if (params.is_causal) {
        // Only support the implementation for is_causal = false.
        return false;
    }

    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_dq = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;

    using Smem_tile_s = fmha::Smem_tile_mma_transposed<typename Kernel_traits::Cta_tile_p>;
    constexpr int smem_size_s = Smem_tile_s::BYTES_PER_TILE;
    static_assert(smem_size_s == 16 * Kernel_traits::Cta_tile_p::N * 2);
    static_assert(smem_size_dq == 16 * Kernel_traits::Cta_tile_p::K * 4 * Kernel_traits::Cta_tile_p::WARPS_N);

    constexpr int smem_size_dq_dk_dv = smem_size_q * 2 + smem_size_v * (Kernel_traits::V_IN_REGS ? 1 : 2) + smem_size_dq + smem_size_s * 2;
    constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
    // printf("blocksize_c = %d, WARPS_N = %d, Smem size = %d\n", blocksize_c, Kernel_traits::Cta_tile_p::WARPS_N, smem_size_dq_dk_dv);

    bool is_dropout = params.p_dropout < 1.f;  // params.p_dropout is the probability of "keeping"

    bool has_attn_mask = !(params.attn_mask_ptr == nullptr);
    bool has_attn_bias = !(params.attn_bias_ptr == nullptr);

    BOOL_SWITCH_FUNC(is_dropout, IsDropoutConst, [&] {
        auto kernel = &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst>;
        if (params.seqlen_k == blocksize_c) {
            kernel = &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, /*loop_steps=*/1>;
        } else if (params.seqlen_k == blocksize_c * 2) {
            kernel = &fmha_dgrad_fp16_sm80_dq_dk_dv_loop_kernel<Kernel_traits, IsDropoutConst, /*loop_steps=*/2>;
        }
        if( smem_size_dq_dk_dv >= 48 * 1024 ) {
            FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
        }
        dim3 grid(params.b, params.h);
        kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params, has_attn_mask, has_attn_bias);
        FMHA_CHECK_CUDA(cudaPeekAtLastError());
    });

    return true;
}
