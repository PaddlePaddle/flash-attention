#include "launch_template.h"

namespace reduced_scores {
template<>
void run_<cutlass::bfloat16_t, 96>(Params &params, cudaStream_t stream) {
    run_hdim96<cutlass::bfloat16_t>(params, stream);
}
} // namespace reduced_scores