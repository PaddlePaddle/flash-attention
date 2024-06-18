#include "launch_template.h"

namespace reduced_scores {
template<>
void run_<cutlass::half_t, 160>(Params &params, cudaStream_t stream) {
    run_hdim160<cutlass::half_t>(params, stream);
}
} // namespace reduced_scores