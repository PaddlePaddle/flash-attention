// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "fmha_fwd_launch_template.h"

void run_fmha_bwd_with_mask_bias_hdim16(Launch_params<FMHA_fprop_params> &launch_params) {
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
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
    }));
}
