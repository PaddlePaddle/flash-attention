#include "reduce_launch_template.h"

template<>
void run_reduce_<cutlass::half_t, 96>(Reduce_params &params, cudaStream_t stream) {
    run_reduce_hdim96<cutlass::half_t>(params, stream);
}
