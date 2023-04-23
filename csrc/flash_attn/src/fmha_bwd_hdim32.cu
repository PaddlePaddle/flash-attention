// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "fmha_bwd_launch_template.h"

void run_fmha_bwd_hdim32(FMHA_dgrad_params &params, cudaStream_t stream, const bool configure) {
    FP16_SWITCH(params.is_bf16, ([&] {
        if (params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
        } else if (params.seqlen_k >= 256) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 8, 0x08u, elem_type>;
            run_fmha_bwd_loop<Kernel_traits>(params, stream, configure);
        }
    }));
}

void run_fmha_bwd_with_bias_mask(const FMHA_dgrad_params &params, cudaStream_t stream) {
    // work around for MSVC issue
    FP16_SWITCH(params.is_bf16, [&] {
        auto dprops = at::cuda::getCurrentDeviceProperties();
        if (params.d == 16) {
            if( params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 16, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            } else if( params.seqlen_k == 256 ) {
                using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            } else {
                // TD [2022-05-15] 512 gives wrong results rn
                // using Kernel_traits = FMHA_kernel_traits<512, 16, 16, 1, 8, 0x08u, elem_type>;
                using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            }
        } else if (params.d == 32) {
            if( params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            } else if( params.seqlen_k >= 256 ) {
                using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            }
        } else if (params.d == 64) {
            if( params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 8, 0x08u, elem_type>;
                run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
            } else if( params.seqlen_k >= 256 ) {
                if (dprops->major == 8 && dprops->minor == 0) {
                    // Don't share smem for K & V, and don't keep V in registers
                    // This speeds things up by 2-3% by avoiding register spills, but it
                    // uses more shared memory, which is fine on A100 but not other GPUs.
                    // For other GPUs, we keep V in registers.
                    using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 8, 0x100u, elem_type>;
                    run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
                } else if (dprops->major == 8 && dprops->minor > 0) {
                    using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 8, 0x08u, elem_type>;
                    run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
                } else if (dprops->major == 7 && dprops->minor == 5) {
                    using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 8, 0x08u, elem_type>;
                    run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
                }
            }
        } else if (params.d == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 8, 0x100u, elem_type>;
            run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        }
    });
}