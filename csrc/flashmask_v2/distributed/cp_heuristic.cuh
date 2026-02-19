#pragma once
#include <cmath>
#include "debug_logger.cuh"

namespace flashmask {

inline int get_num_chunk_per_segment(int local_seqlen_k, int cp_size, int kv_head) {
    // 32K+ seqlen does not need chunk grouping
    if (local_seqlen_k >= 32768) return 1;
    // logarithm heuristic
    int power = int(std::floor(std::log2(cp_size) * 0.5));
    int chunk_size = std::pow(2, power);
    DEBUG_PRINT("BWD RS-overlap chunk per segment: %d\n", chunk_size);
    return chunk_size;
}

}   // namespace flashmask