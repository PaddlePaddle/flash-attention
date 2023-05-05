// Copyright (c) 2022, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "fmha_bwd_launch_template.h"

void run_fmha_bwd_with_mask_bias_hdim128(const FMHA_dgrad_params &launch_params, 
                                         cudaStream_t stream) {
    FP16_SWITCH(launch_params.params.is_bf16, ([&] {
        using Kernel_traits = FMHA_kernel_traits<128, 128, 16, 1, 8, 0x100u, elem_type>;
        run_fmha_dgrad_fp16_sm80_loop_<Kernel_traits>(params, stream);
    }));
}
