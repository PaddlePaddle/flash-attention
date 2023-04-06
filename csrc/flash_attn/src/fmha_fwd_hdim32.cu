// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "fmha_fwd_launch_template.h"

void run_fmha_fwd_hdim32(Launch_params<FMHA_fprop_params> &launch_params) {
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
        if (launch_params.params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_fwd_loop<Kernel_traits>(launch_params);
        } else if (launch_params.params.seqlen_k >= 256) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
            run_fmha_fwd_loop<Kernel_traits>(launch_params);
        }
    }));
}

void run_fmha_fwd_with_bias_mask(Launch_params<FMHA_fprop_params> &launch_params,
                        const bool configure) {
    FP16_SWITCH_FUNC(launch_params.params.is_bf16, [&] {
        auto dprops = GetDeviceProperties(-1);
        if (launch_params.params.d == 16) {
            if( launch_params.params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 16, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } 
            else if( launch_params.params.seqlen_k == 256 ) {
                using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else {
                // TD [2022-05-15] 512 gives wrong results rn
                // using Kernel_traits = FMHA_kernel_traits<512, 16, 16, 1, 4, 0x08u, elem_type>;
                using Kernel_traits = FMHA_kernel_traits<256, 16, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            }
        }
        else if (launch_params.params.d == 32) {
            if( launch_params.params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else if( launch_params.params.seqlen_k == 256 ) {
                using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else {
                using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            }
        } 
        else if (launch_params.params.d == 64) {
            if( launch_params.params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else if( launch_params.params.seqlen_k >= 256 ) {
                if (dprops->major == 8 && dprops->minor >= 0) {
                    using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;
                    run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                } else if (dprops->major == 7 && dprops->minor == 5) {
                    if (launch_params.is_dropout) { // Need to use the same block size as backward
                        using Kernel_traits = FMHA_kernel_traits<128, 64, 16, 1, 4, 0x08u, elem_type>;
                        run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                    } else {
                        using Kernel_traits = FMHA_kernel_traits<256, 64, 16, 1, 4, 0x08u, elem_type>;
                        run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                    }
                }
            }
        } else if (launch_params.params.d == 128) {
            if( launch_params.params.seqlen_k == 128 ) {
                using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, elem_type>;
                run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
            } else {
                if (dprops->major == 8 && dprops->minor == 0 && !launch_params.is_dropout) {
                    // TD [2022-06-05] Keep K in registers to reduce register spilling
                    // Gives about 6% speedup compared to using block size 128.
                    using Kernel_traits = FMHA_kernel_traits<256, 128, 16, 1, 4, 0x18u, elem_type>;
                    run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                } else {  // Need to use the same block size as backward
                    using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 4, 0x08u, elem_type>;
                    run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
                }
            }
        }
    });
}