// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "fmha_fwd_launch_template.h"

bool run_fmha_fwd_with_mask_bias_hdim32(Launch_params<FMHA_fprop_params> &launch_params,
                                        const bool configure) {
    bool status = false;
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
        if (launch_params.params.seqlen_k == 128) {
            using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 4, 0x08u, elem_type>;
            status = run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        } else if (launch_params.params.seqlen_k == 256) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
            status = run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        } else {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 4, 0x08u, elem_type>;
            status = run_fmha_fp16_sm80_loop_<Kernel_traits>(launch_params, configure);
        }
    }));
    return status;
}
