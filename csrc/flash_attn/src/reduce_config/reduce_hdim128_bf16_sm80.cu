#include "reduce_launch_template.h"

template<>
void run_reduce_<cutlass::bfloat16_t, 128>(Reduce_attn_scores_params &params, cudaStream_t stream) {
    run_reduce_hdim128<cutlass::bfloat16_t>(params, stream);
}
