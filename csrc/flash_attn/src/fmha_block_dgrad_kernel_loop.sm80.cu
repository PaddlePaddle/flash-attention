/* Copyright (c) 2022, Tri Dao.
 */
#include "fmha.h"
#include "static_switch.h"
#include "fmha_block_dgrad_kernel_1xN_loop.h"
#include "cuda_utils.h"

template<typename Kernel_traits>
__global__ void fmha_bwd_dot_do_o_kernel(FMHA_dgrad_params params) {
    fmha::compute_dot_do_o<Kernel_traits>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, int loop_steps=-1>
__global__ void fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel(FMHA_dgrad_params params) {
    fmha::compute_block_dq_dk_dv_1xN<Kernel_traits, Is_dropout, Is_causal, loop_steps>(params);
}
template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
__global__ void fmha_block_dgrad_sm80_dq_dk_dv_loop_seqparallel_kernel(FMHA_dgrad_params params) {
    fmha::compute_block_dq_dk_dv_1xN_seqparallel<Kernel_traits, Is_dropout, Is_causal>(params);
}
inline int num_splits_heuristic_bwd(int batch_nheads, int num_SMs, int ctas_per_sm, int seqlen,
        int blocksize, bool is_causal) {
    float n_waves_1 = float(batch_nheads) / (num_SMs * ctas_per_sm);
    float eff_1 = n_waves_1 / ceil(n_waves_1);
    int num_splits_parallel = seqlen / blocksize;
    float n_waves_parallel = float(batch_nheads * num_splits_parallel) / (num_SMs * ctas_per_sm);
    float eff_parallel_raw = n_waves_parallel / ceil(n_waves_parallel);
    float discount_factor;
    if (!is_causal) {
        discount_factor = 1.f + float(blocksize) / seqlen;
    } else {  // For causal, parallelizing seems to help with load-balancing as well
        // For example, if headdim=128, seqlen >= 1280 always prefers parallel
        if (seqlen / blocksize >= 10) return num_splits_parallel;
        discount_factor = 1.f + 0.5 * float(blocksize) / seqlen;
    }
    float eff_parallel = eff_parallel_raw / discount_factor;
    return eff_1 >= eff_parallel ? 1 : num_splits_parallel;
}

template<typename Kernel_traits>
void run_fmha_block_dgrad_sm80_loop_(FMHA_dgrad_params &params, cudaStream_t stream) {
    constexpr int smem_size_softmax = Kernel_traits::Cta_tile_p::M * Kernel_traits::Cta_tile_p::WARPS_N * sizeof(float);
    constexpr int smem_size_q = Kernel_traits::Smem_tile_q::BYTES_PER_TILE;
    constexpr int smem_size_v = Kernel_traits::Smem_tile_v::BYTES_PER_TILE;
    constexpr int smem_size_dq = Kernel_traits::Smem_tile_o::BYTES_PER_TILE;
    constexpr int smem_size_dp_sum = Kernel_traits::Smem_dp_sum::BYTES_PER_TILE;

    using Smem_tile_s = fmha::Smem_tile_mma_transposed<typename Kernel_traits::Cta_tile_p>;
    constexpr int smem_size_s = Smem_tile_s::BYTES_PER_TILE;
    static_assert(smem_size_s == 16 * Kernel_traits::Cta_tile_p::N * 2);
    static_assert(smem_size_dq == 16 * Kernel_traits::Cta_tile_p::K * 4 * Kernel_traits::Cta_tile_p::WARPS_N);
    static_assert(smem_size_dp_sum == 16 * 4 * 2);

    constexpr int smem_size_dq_dk_dv = smem_size_q * 2 + smem_size_v * (Kernel_traits::V_IN_REGS ? 1 : 2) + smem_size_dq + smem_size_s * 2 + smem_size_dp_sum;

    bool is_dropout = params.p_dropout < 1.f;  // params.p_dropout is the probability of "keeping"
    bool is_causal = params.is_causal;
    BOOL_SWITCH(is_dropout, IsDropoutConst, ([&] {
        auto kernel = is_dropout
            ? (is_causal ? &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, true> : &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, false>)
            : (is_causal ? &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, true> : &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, false>);
        constexpr int blocksize_c = Kernel_traits::Cta_tile_p::N;
        if (params.seqlen_k == blocksize_c) {
            kernel = is_dropout
                ? (is_causal ? &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, true, /*loop_steps=*/1> : &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, false, /*loop_steps=*/1>)
                : (is_causal ? &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, true, /*loop_steps=*/1> : &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, false, /*loop_steps=*/1>);
        } else if (params.seqlen_k == blocksize_c * 2) {
            kernel = is_dropout
            ? (is_causal ? &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, true, /*loop_steps=*/2> : &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, true, false, /*loop_steps=*/2>)
            : (is_causal ? &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, true, /*loop_steps=*/2> : &fmha_block_dgrad_sm80_dq_dk_dv_loop_kernel<Kernel_traits, false, false, /*loop_steps=*/2>);
        }

        auto kernel_seqparallel = params.is_causal
            ? &fmha_block_dgrad_sm80_dq_dk_dv_loop_seqparallel_kernel<Kernel_traits, IsDropoutConst, true>
            : &fmha_block_dgrad_sm80_dq_dk_dv_loop_seqparallel_kernel<Kernel_traits, IsDropoutConst, false>;

        if( smem_size_dq_dk_dv >= 48 * 1024 ) {
            FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
            FMHA_CHECK_CUDA(cudaFuncSetAttribute(
                    kernel_seqparallel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
        }
    //dim3 grid(params.b, params.h);
    //kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
    //FMHA_CHECK_CUDA(cudaPeekAtLastError());
        // Automatically set num_splits to maximize occupancy
        if (params.num_splits <= 0) {
            int ctas_per_sm;
            cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &ctas_per_sm, kernel, Kernel_traits::THREADS, smem_size_dq_dk_dv);
            auto dprops = GetDeviceProperties(-1);
            // printf("CTAS_PER_SM = %d, nSMs = %d\n", ctas_per_sm, dprops->multiProcessorCount);
            constexpr int M = Kernel_traits::Cta_tile_p::M;
            // We don't want more than 10 splits due to numerical error.
            // Numerical error on dk/dv scales as sqrt(num_splits).
            params.num_splits = num_splits_heuristic_bwd(
                params.b * params.h, dprops->multiProcessorCount,
                ctas_per_sm, params.seqlen_k, blocksize_c, params.is_causal
            );
        }
        //if (configure) return;
        if (params.num_splits == 1) {
            dim3 grid(params.b, params.h, params.num_splits);
            kernel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
        } else {
            dim3 grid_dot(params.b, params.h, (params.seqlen_q + 128 - 1) / 128);
            fmha_bwd_dot_do_o_kernel<Kernel_traits><<<grid_dot, Kernel_traits::THREADS, 0, stream>>>(params);
            int num_splits = params.seqlen_k / blocksize_c;  // seqlen_k is divisible by blocksize_c
            dim3 grid(params.b, params.h, num_splits);
            kernel_seqparallel<<<grid, Kernel_traits::THREADS, smem_size_dq_dk_dv, stream>>>(params);
        }
        FMHA_CHECK_CUDA(cudaPeekAtLastError());
    }));
}

void run_fmha_block_dgrad_sm80(FMHA_dgrad_params &params, cudaStream_t stream) {
    FP16_SWITCH(params.is_bf16, ([&] {
        if (params.d == 16) {
            using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_block_dgrad_sm80_loop_<Kernel_traits>(params, stream);
        } else if (params.d == 32) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_block_dgrad_sm80_loop_<Kernel_traits>(params, stream);
        } else if (params.d == 64) {
            using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 8, 0x100u, elem_type>;
            run_fmha_block_dgrad_sm80_loop_<Kernel_traits>(params, stream);
        } else if (params.d == 128) {
            using Kernel_traits = FMHA_kernel_traits<256, 128, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_block_dgrad_sm80_loop_<Kernel_traits>(params, stream);
        }
    }));
}
