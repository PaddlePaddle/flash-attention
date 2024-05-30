#include "reduce_launch_template.h"

template<>
void run_reduce_<cutlass::bfloat16_t, 224>(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    run_reduce_hdim224<cutlass::bfloat16_t>(params, stream, configure);
}
