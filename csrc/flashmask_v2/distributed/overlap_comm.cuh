#pragma once
#include <memory>
#include <string>
#include "sr_buffer.cuh"
#include "sep_sr_buffer.cuh"
#include "cutlass/bfloat16.h"

namespace flashmask {

// Configuration snapshot for detecting parameter changes between calls.
// All fields that affect buffer sizes or kernel behavior are tracked here.
struct OverlapConfig {
    int B = 0;
    int S_local = 0;       // dispatched at runtime: {4096, 8192, 16384, 32768, 65536, 131072}
    int H = 0;
    int H_mask = 0;
    int D = 0;
    int nranks = 0;        // NVSHMEM-unsafe to change; kept for validation
    bool overlap_rs = false;

    bool operator==(const OverlapConfig& other) const {
        return B == other.B && S_local == other.S_local && H == other.H
            && H_mask == other.H_mask && D == other.D && nranks == other.nranks
            && overlap_rs == other.overlap_rs;
    }
    bool operator!=(const OverlapConfig& other) const { return !(*this == other); }

    // Compute the SRBuffer numel for one of K or V (= B * S_local * H * D * nranks)
    size_t sr_buffer_numel() const {
        return static_cast<size_t>(S_local) * H * D * B * nranks;
    }

    // Compute the per-chunk size (= S_local * H * D)
    size_t cp_chunk_size() const {
        return static_cast<size_t>(S_local) * H * D;
    }

    std::string to_string() const {
        return "B=" + std::to_string(B) + ", S_local=" + std::to_string(S_local)
            + ", H=" + std::to_string(H) + ", H_mask=" + std::to_string(H_mask)
            + ", D=" + std::to_string(D) + ", nranks=" + std::to_string(nranks)
            + ", overlap_rs=" + std::to_string(int(overlap_rs));
    }
};

/**
 * SM-level overlapping communicator
 *
 * An RAII object managing CP-groups / comm stream / buffer lifetimes automatically
 *
 * Call the constructor of this communicator, and call `run_overlap_ag_kernel` before
 * the main attention kernel, make sure the lifetime of the instance outlast the main kernel
 * Then you should be able to get async remote get, costing only 4 SMs.
 *
 * Dynamic reconfiguration: when B/H/D/mask_head/S_local change between calls,
 * `reconfigure_if_needed()` will detect the change and reallocate buffers as needed.
 * S_local is dispatched at compile time: supported values are {4096, 8192, 16384, 32768, 65536, 131072}.
 * RS-overlap is automatically enabled when H_k >= rs_overlap_min_h_k and nranks > 1.
*/
template <typename KVType>
class OverlapCommunicator {
public:
    OverlapCommunicator(
        const KVType* const k_data,
        const KVType* const v_data,
        int b_kv,
        int s_kv,
        int h_kv,
        int d_kv,
        int rank,
        int nranks,
        const uint8_t* unique_id_ptr = nullptr,
        int mask_head = 0,
        bool overlap_rs = false
    );

    ~OverlapCommunicator();

    /**
     * Check if reconfiguration is needed given new params, and perform it if so.
     * Returns true if reconfiguration happened.
     * S_local must be one of {4096, 8192, 16384, 32768, 65536, 131072}.
    */
    bool reconfigure_if_needed(
        int new_b, int new_s_local, int new_h, int new_d,
        int rank, int nranks, int new_mask_head, bool new_overlap_rs,
        const uint8_t* unique_id_ptr = nullptr
    );

    OverlapConfig current_config() const { return _config; }

    /**
     * run the overlap kernel asynchronously
     * @param S the seqlen of local K (for example, 32K full length, CP=4, local S=8K)
     *      After the execution of this function, S will be set to `S * nranks`
     * TODO(heqianyue): extend to more mask types!
    */
    void run_overlap_ag_kernel(
        int* const write_ptr,
        int& S,
        const bool fwd = true
    );

    // only used when use_rs_overlap and in the bwd
    void run_overlap_splitted_ag_kernel(
        int* const write_ptr,
        int segment_idx
    );

    void run_overlap_rs_kernel(
        KVType* const dk_accum,
        KVType* const dv_accum,
        int segment_idx,
        cudaStream_t comp_stream    // compute_stream (stream of Tensor and bwd kernel)
    );

    void wait_wptr_init();

    void* k_data() const { return kv_buffer->k_data(); }
    void* v_data() const { return kv_buffer->v_data(); }

    // we need to reroute the bwd dx_accum output buffer to dk_send and dv_send
    // so that the output of post-proc kernel can be directly sent
    // DO NOT call the following methods, if overlap_rs = false
    void* dk_send(int seg_idx) const { return dkv_buffer->k_send(seg_idx); }
    void* dv_send(int seg_idx) const { return dkv_buffer->v_send(seg_idx); }

    // computation stream wait the comm_stream kernel to be scheduled with SMs
    void wait_reset_stream_coordinator(cudaStream_t stream);

    void update_kv_buffer(
        const KVType* const new_k_data,
        const KVType* const new_v_data,
        const bool fwd = true
    );

    int dkv_buffer_stage() const;

    void wait_reduce_done(cudaStream_t stream) const {
        cudaEventRecord(reduce_done, aux_c_stream);
        cudaStreamWaitEvent(stream, reduce_done);
    }

    void compute_chunk_mask(
        const int* const lt_start_ptr,
        const int* const lt_end_ptr,
        const int* const ut_start_ptr,
        const int* const ut_end_ptr,
        cudaStream_t stream,
        const bool fwd = true
    );

    // in `USE_SEMAPHORES` mode, call this function before calling `updayte_kv_buffer`
    // to make sure other PEs have finished reading the local KV data in our SR buffer
    // also, barriers the remote_get `comm_kernel`
    void wait_sr_buffer_empty(cudaStream_t stream);

    int* get_block_cnt_semaphore() const {
        return block_cnt_semaphore;
    }

    int nranks() const {
        return _total_n_pes;
    }

    int s_local() const {
        return S_local;
    }

    // this function is only called in the bwd
    int seqlen_scale() const;
    int num_segments() const;
    // for RS overlap, returns number of chunks per segment
    int chunk_per_seg() const;
    int overlap_sm_margin() const;
    void prepare_dkv_buffer(cudaStream_t stream);

    // wptr_init: comp_stream notifies comm_stream, write_ptr is usable
    // sr_usable: comp_stream notifies comm_stream, KV SR buffer local chunk can be reused (since computation is done)
    // bwd_done (only when RS-overlap): comp_stream notifies aux_streams, bwd post-proc done
    // reduce_done (only when RS-overlap): aux_c_stream notifies comp_stream, dk/v recv buffer are released and ready
    // local_moved (only when RS-overlap): for segment 0, aux_p_stream notifies aux_c_stream whether the memcpy d2d is completed.
    cudaEvent_t wptr_init, sr_usable, bwd_done, reduce_done, local_moved;
    /**
     * If overlap_rs is set, dkv_buffer will be populated.
     * and since the fwd AG buffer is always bigger than bwd AG
     * (due to the fact that bwd AG is splitted), fwd kv_buffer
     * can be reused (carefully).
    */
    std::unique_ptr<SepSRBuffer<KVType>> dkv_buffer;
private:
    /**
     * Note(heqianyue): for B > 1, RS-overlap, we need a place to store the local KV chunk
     * so that each split AG remote_get call can correctly send the data to other ranks.
     * We choose to store one more copy of the local KV chunk data at the end of the SR buffer.
     * Note that this makes it two copies of the local KV chunks: the first copy is ordered
     * with a batch stride of num_chunks * S_local * H * D, so that for the first segement,
     * attention kernel can directly use SR buffer for bwd recompute. This copy of local KV chunks
     * will be overwritten by the upcoming segments, so we need the second copy (gauranteed:
     * will never be overwritten) for remote ranks to get from.
     *
     * return the last B * S_local * H * D elems in the respective SR buffer
     *
    */
    inline KVType* local_k_data() const {
        return kv_buffer->k_data() + _total_numel - B * _local_batch_stride;
    }
    inline KVType* local_v_data() const {
        return kv_buffer->v_data() + _total_numel - B * _local_batch_stride;
    }

    // Helper to (re)allocate block_work_ids and derived pointers
    void reallocate_block_work_ids();
    // Helper to create or recreate the dkv_buffer for RS-overlap
    void setup_dkv_buffer(bool need_rs, nvshmem_team_t cp_team);

private:
    std::unique_ptr<SRBuffer<KVType>> kv_buffer;
    cudaStream_t comm_stream;
    cudaStream_t aux_p_stream;      // RS-overlap: producer (put) stream
    cudaStream_t aux_c_stream;      // RS-overlap: consumer (reduce) stream

    // Shape parameters (non-const to allow dynamic reconfiguration)
    int B;
    int S_local;            // dispatched at runtime: {4096, 8192, 16384, 32768, 65536, 131072}
    int H;
    int H_mask;             // mask head
    int D;
    int num_chunks;

    int _my_pe;
    int _total_n_pes;
    size_t _local_batch_stride;
    size_t _total_numel;

    // Configuration tracking for dynamic reconfiguration
    OverlapConfig _config;
    size_t _sr_buffer_capacity;             // allocated SRBuffer numel capacity
    size_t _dkv_single_k_numel_capacity;    // allocated SepSRBuffer single_k_numel capacity
    int _dkv_num_chunks;                    // SepSRBuffer's chunks_per_seg (layout-defining)
    int _num_copy_chunks;                   // block_work_ids array size tracking

    int* block_work_ids;
    int* block_cnt_semaphore;
    int* copy_chunk_mask;
    int* stream_coordinator;        // make sure comm kernel is scheduled to GPU before computation kernel
};

namespace comm {

// OverlapCommunicator instance is managed via this singleton, therefore
// the instance is accessible to both fwd and bwd passes

// init or reconfigure instance (mutable ref), used in fwd.
// On first call: creates the singleton.
// On subsequent calls: checks if params changed and reconfigures if needed.
OverlapCommunicator<cutlass::bfloat16_t>& init_singleton_instance(
    const cutlass::bfloat16_t* const k_data,
    const cutlass::bfloat16_t* const v_data,
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    const uint8_t* unique_id_ptr,
    int mask_head = 1
);

// get instance (mutable ref), make sure the instance is initialized, used in both fwd and bwd
OverlapCommunicator<cutlass::bfloat16_t>& singleton();

// check whether the singleton unique_ptr is nullptr
bool is_singleton_null();

// Destroy the singleton for topology refresh.
// After calling this, the next init_singleton_instance() will re-create from scratch.
// IMPORTANT: per NVSHMEM bootstrap persistence, finalize + re-init is only safe when
// rank/nranks remain the same. This function is intended for refreshing
// transport-level resources (e.g., after node migration), not for changing topology.
void destroy_singleton();

}   // namespace comm

}   // namespace flashmask
