#pragma once
#include <cmath>
#include "debug_logger.cuh"

namespace flashmask {

// Note(heqianyue): using 2 chunks per segement is actually faster for CP16
inline int get_num_chunk_per_segment(int local_seqlen_k, int nranks, int kv_head) {
    // 32K+ seqlen does not need chunk grouping
    if (local_seqlen_k >= 32768) return 1;
    // logarithm heuristic
    int power = int(std::floor(std::log2(nranks) * 0.5));
    int num_chunks = std::pow(2, power);
    DEBUG_PRINT("BWD RS-overlap chunk per segment: %d\n", num_chunks);
    return num_chunks;
}

}   // namespace flashmask