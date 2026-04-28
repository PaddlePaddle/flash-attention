#pragma once

/**
 * Hierarchical Rank Mapping for AG Overlap Communication
 *
 * Provides mapping from logical chunk position to target rank, following the
 * "congruence group first, intra-node redistribution second" traversal order.
 *
 * Traversal order (BWD, left-to-right):
 *   [local, congruence_partners..., same_node_rank_1, its_congruence_partners..., ...]
 *
 * FWD is the physical reverse of BWD (right-to-left traversal).
 *
 * When num_nodes == 1 (single node), degenerates to the current circular shift order.
 */

namespace flashmask::hier {

struct HierRankInfo {
    int target_rank;    // The rank whose KV data is at this position
    int src_pe;         // The PE to fetch from (Phase 1: target_rank; Phase 2: same-node rank holding the data)
    int logical_pos;    // Logical chunk position (0 = local, 1 = first remote, ...)
    bool is_phase1;     // True = cross-node congruence group fetch (IB RDMA)
    bool is_local;      // True = local chunk (no fetch needed)
};

/**
 * Map a logical chunk position to rank info in the hierarchical traversal order.
 *
 * @param logical_pos  0-based chunk position (0 = local, 1..nranks-1 = remote)
 * @param my_pe        This PE's global rank
 * @param total_n_pes  Total number of PEs (= nranks)
 * @param gpus_per_node  Number of GPUs per node
 * @return HierRankInfo with target_rank, src_pe, phase info
 */
__device__ __host__ inline HierRankInfo hier_map_chunk(
    int logical_pos,
    int my_pe,
    int total_n_pes,
    int gpus_per_node
) {
    HierRankInfo info;
    info.logical_pos = logical_pos;

    const int my_pe_node = my_pe % gpus_per_node;  // my_pe's index within its node
    const int my_node_id = my_pe / gpus_per_node;  // my_pe's node index (NOT the target's)
    const int num_nodes = total_n_pes / gpus_per_node;

    // Slot 0: local rank's congruence group (positions 0..num_nodes-1)
    //   Position 0: local chunk
    //   Position 1..num_nodes-1: congruence partners (Phase 1, cross-node)
    // Slot s (s >= 1): same-node rank's congruence group (Phase 2, intra-node)
    //   Positions s*num_nodes .. (s+1)*num_nodes-1

    if (logical_pos < num_nodes) {
        // Slot 0: local congruence group
        // sub = logical_pos = node offset from my_node (0=my node, 1=next node, ...)
        // The congruence partner at node (my_node_id + sub) % num_nodes
        // target_rank = my_pe_node + ((my_node_id + sub) % num_nodes) * gpus_per_node
        //   Example: rank 0 (my_pe_node=0, my_node_id=0), sub=1 → 0 + (1%4)*8 = 8
        int sub = logical_pos;
        info.target_rank = my_pe_node + ((my_node_id + sub) % num_nodes) * gpus_per_node;
        info.is_local = (logical_pos == 0);
        info.is_phase1 = (logical_pos > 0);  // Congruence partners are Phase 1
        info.src_pe = info.target_rank;       // Phase 1: fetch directly from target
        return info;
    }

    // Slot s >= 1: same-node rank's congruence group
    int adj_pos = logical_pos - num_nodes;  // 0-based within non-local slots
    int slot = adj_pos / num_nodes + 1;     // slot index (1-based)
    int sub = adj_pos % num_nodes;           // sub index within the slot

    // Base node-local rank for this slot (wraps around gpus_per_node)
    int base = (my_pe_node + slot) % gpus_per_node;

    // Target rank: base rank's congruence partner at node (my_node_id + sub) % num_nodes
    info.target_rank = base + ((my_node_id + sub) % num_nodes) * gpus_per_node;
    info.is_local = false;
    info.is_phase1 = false;  // Phase 2: intra-node redistribution

    // Source PE: the same-node rank that holds target_rank's data
    // This is the PE with node-local index = base, on our node
    info.src_pe = base + my_node_id * gpus_per_node;

    return info;
}

/**
 * Compute the chunk position of target_rank's data in src_pe's SR buffer.
 * Used to calculate src_addr for Phase 2 (intra-node) fetches.
 *
 * Precondition: target_rank is in src_pe's congruence group
 *   (target_rank % gpus_per_node == src_pe % gpus_per_node)
 *
 * @return Chunk position (0-based) in src_pe's hierarchical order
 */
__device__ __host__ inline int hier_position_in_src_pe(
    int target_rank,
    int src_pe,
    int total_n_pes,
    int gpus_per_node
) {
    const int src_node_id = src_pe / gpus_per_node;
    const int target_node_id = target_rank / gpus_per_node;
    const int num_nodes = total_n_pes / gpus_per_node;

    // target_rank is in src_pe's slot 0 (congruence group)
    // Position j: the congruence partner at node (src_node_id + j) % num_nodes
    // We need j such that (src_node_id + j) % num_nodes == target_node_id
    int j = (target_node_id - src_node_id + num_nodes) % num_nodes;
    return j;
}

/**
 * Compute seqlen_id for a given logical chunk position and direction.
 *
 * @param logical_pos  0-based chunk position
 * @param row_within_chunk  Row offset within the chunk (in units of row_per_block)
 * @param row_per_block  Granularity of work items
 * @param nranks  Total number of ranks
 * @param S_chunk  Size of each chunk (= S_local)
 * @param bwd  True for backward (left-to-right), false for forward (right-to-left)
 */
__device__ __host__ inline int hier_seqlen_id(
    int logical_pos,
    int row_within_chunk,
    int row_per_block,
    int nranks,
    int S_chunk,
    bool bwd
) {
    if (bwd) {
        // BWD: left-to-right, position 0 at seqlen 0
        return logical_pos * S_chunk + row_within_chunk * row_per_block;
    } else {
        // FWD: right-to-left, position 0 at seqlen (nranks-1)*S_chunk
        return (nranks - 1 - logical_pos) * S_chunk + row_within_chunk * row_per_block;
    }
}

/**
 * Compute the source chunk's seqlen offset in src_pe's SR buffer.
 * For Phase 1, this is the local chunk offset (same as current seqlen_offset).
 * For Phase 2, this is the position of target_rank's data in src_pe's SR buffer.
 *
 * @param info  The HierRankInfo for this chunk
 * @param nranks  Total number of ranks
 * @param S_chunk  Size of each chunk
 * @param total_n_pes  Total number of PEs
 * @param gpus_per_node  GPUs per node
 * @param bwd  Direction flag
 */
__device__ __host__ inline int hier_src_chunk_offset(
    const HierRankInfo& info,
    int nranks,
    int S_chunk,
    int total_n_pes,
    int gpus_per_node,
    bool bwd
) {
    if (info.is_phase1 || info.is_local) {
        // Phase 1: fetch from remote PE's local chunk
        // Same as current seqlen_offset
        if (bwd) {
            return 0;  // BWD: local chunk at position 0
        } else {
            return (nranks - 1) * S_chunk;  // FWD: local chunk at last position
        }
    }

    // Phase 2: fetch from src_pe's SR buffer at the position of target_rank's data
    int src_chunk_pos = hier_position_in_src_pe(info.target_rank, info.src_pe, total_n_pes, gpus_per_node);

    if (bwd) {
        return src_chunk_pos * S_chunk;
    } else {
        return (nranks - 1 - src_chunk_pos) * S_chunk;
    }
}

/**
 * Check if hierarchical overlap should be used.
 * When num_nodes == 1 (single node), hierarchical degenerates to circular shift.
 */
__device__ __host__ inline bool hier_is_effective(int total_n_pes, int gpus_per_node) {
    return total_n_pes > gpus_per_node;
}

/**
 * Compute the initial reference count for the local chunk's semaphore.
 * This is the number of consumers that will come read our local KV:
 *   - (num_nodes - 1) cross-node congruence partners (Phase 1)
 *   - (gpus_per_node - 1) same-node ranks (Phase 2)
 */
__device__ __host__ inline int hier_local_refcount(int total_n_pes, int gpus_per_node) {
    int num_nodes = total_n_pes / gpus_per_node;
    return (num_nodes - 1) + (gpus_per_node - 1);
}

/**
 * Compute the initial reference count for a congruence partner's data.
 * After we fetch a congruence partner's KV, same-node ranks will come read it.
 * Count = (gpus_per_node - 1) (all same-node ranks except us, since we already have it)
 */
__device__ __host__ inline int hier_congruence_refcount(int gpus_per_node) {
    return gpus_per_node - 1;
}

/**
 * Compute number of chunks in Phase 1 (cross-node congruence group, excluding local).
 * = num_nodes - 1
 */
__device__ __host__ inline int hier_phase1_chunk_count(int total_n_pes, int gpus_per_node) {
    return total_n_pes / gpus_per_node - 1;
}

/**
 * Compute number of chunks in Phase 2 (intra-node redistribution).
 * = nranks - num_nodes (all chunks not in local's congruence group)
 */
__device__ __host__ inline int hier_phase2_chunk_count(int total_n_pes, int gpus_per_node) {
    int num_nodes = total_n_pes / gpus_per_node;
    return total_n_pes - num_nodes;
}

} // namespace flashmask::comm::hier
