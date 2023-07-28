// Copyright (c) 2022, Tri Dao.

#include "fmha_bwd_with_mask_bias_launch_template.h"

bool run_fmha_bwd_with_mask_bias_hdim32(FMHA_dgrad_params &params, cudaStream_t stream) {
    bool status = true;
    FP16_SWITCH(params.is_bf16, ([&] {
        if( params.seqlen_k == 128 ) {
            using Kernel_traits = FMHA_kernel_traits<128, 32, 16, 1, 8, 0x08u, elem_type>;
            status = run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        } else if( params.seqlen_k >= 256 ) {
            using Kernel_traits = FMHA_kernel_traits<256, 32, 16, 1, 8, 0x08u, elem_type>;
            status = run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
        }
    }));
    return status;
}
