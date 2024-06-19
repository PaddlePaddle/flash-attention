#include "launch_template.h"

namespace reduced_scores {
template<>
void run_<cutlass::half_t, 96>(Params &params, cudaStream_t stream) {
    run_hdim96<cutlass::half_t>(params, stream);
}
} // namespace reduced_scores