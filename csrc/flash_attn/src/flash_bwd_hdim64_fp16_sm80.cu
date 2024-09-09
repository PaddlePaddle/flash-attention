// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<cutlass::half_t, 64, false, false, false>(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    run_mha_bwd_hdim64<cutlass::half_t, false, false, false>(params, stream, configure);
}
