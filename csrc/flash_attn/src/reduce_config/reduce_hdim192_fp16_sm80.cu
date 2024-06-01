#include "reducelaunch_template.h"

template<>
void run_reduce_<cutlass::half_t, 192>(Reduce_attn_scores_params &params, cudaStream_t stream) {
    run_reduce_hdim192<cutlass::half_t>(params, stream);
}
