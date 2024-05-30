#include "reduce_launch_template.h"

template<>
void run_reduce_<cutlass::half_t, 64>(Reduce_attn_scores_params &params, cudaStream_t stream, const bool configure) {
    run_reduce_hdim64<cutlass::half_t>(params, stream, configure);
}