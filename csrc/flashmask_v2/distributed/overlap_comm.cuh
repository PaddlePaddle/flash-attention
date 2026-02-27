#pragma once
#include <memory>
#include "sr_buffer.cuh"
#include "sep_sr_buffer.cuh"
#include "cutlass/bfloat16.h"

namespace flashmask {

/**
 * SM-level overlapping communicator
 * 
 * An RAII object managing CP-groups / comm stream / buffer lifetimes automatically
 * 
 * Call the constructor of this communicator, and call `run_overlap_ag_kernel` before
 * the main attention kernel, make sure the lifetime of the instance outlast the main kernel
 * Then you should be able to get async remote get, costing only 4 SMs.
 * 
 * TODO(heqianyue): Code-style --- make this class move-only explicitly. Currently this class is only implicitly move-only.
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
        int cp_size,
        const uint8_t* unique_id_ptr = nullptr,
        int mask_head = 0,
        bool overlap_rs = false
    );

    ~OverlapCommunicator();

    /**
     * run the overlap kernel asynchronously
     * @param S the seqlen of local K (for example, 32K full length, CP=4, local S=8K)
     *      After the execution of this function, S will be set to `S * cp_size`
     * TODO(heqianyue): extend to more mask types!
    */
    void run_overlap_ag_kernel(
        const int* const lt_start_ptr,
        const int* const ut_end_ptr,
        int* const write_ptr,
        int& S,
        const bool fwd = true
    );

    // only used when use_rs_overlap and in the bwd
    void run_overlap_splitted_ag_kernel(
        const int* const lt_start_ptr,
        const int* const ut_end_ptr,
        int* const write_ptr,
        int& S,
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

    void set_sr_usable(cudaStream_t stream) {
        cudaEventRecord(sr_usable, stream);
    }

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
        const int* const ut_end_ptr,
        cudaStream_t stream,
        const bool fwd = true
    );

    // in `USE_SEMAPHORES` mode, call this function before calling `updayte_kv_buffer`
    // to make sure other PEs have finished reading the local KV data in our SR buffer
    // also, barriers the remote_get `comm_kernel`
    void wait_sr_buffer_empty();

    int* get_block_cnt_semaphore() const {
        return block_cnt_semaphore;
    }

    int cp_size() const {
        return _cp_size;
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
    cudaEvent_t wptr_init, sr_usable, bwd_done, reduce_done;
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
        return kv_buffer->k_data() + _total_numel - B * _cp_chunk_size;
    }
    inline KVType* local_v_data() const {
        return kv_buffer->v_data() + _total_numel - B * _cp_chunk_size;
    }
private:
    std::unique_ptr<SRBuffer<KVType>> kv_buffer;
    cudaStream_t comm_stream;
    cudaStream_t aux_p_stream;      // RS-overlap: producer (put) stream
    cudaStream_t aux_c_stream;      // RS-overlap: consumer (reduce) stream
    const int B;
    const int S_local;
    const int H;
    const int H_mask;       // mask head
    const int D;
    const int _cp_size;
    const int num_chunks;
    int _my_pe;
    int _total_n_pes;
    int _cp_stride;
    size_t _cp_chunk_size;
    size_t _total_numel;

    int* block_work_ids;
    int* block_cnt_semaphore;
    int* copy_chunk_mask;
    int* stream_coordinator;        // make sure comm kernel is scheduled to GPU before computation kernel

    // returns the team and stride between teams
    static nvshmem_team_t simple_collective_topology_setter(int my_global_pe, int stride, int n_pes);
};

namespace comm {

// OverlapCommunicator instance is managed via this singleton, therefore
// the instance is accessible to both fwd and bwd passes

// init and get instance (mutable ref), used in fwd
OverlapCommunicator<cutlass::bfloat16_t>& init_singleton_instance(
    const cutlass::bfloat16_t* const k_data,
    const cutlass::bfloat16_t* const v_data,
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int rank,
    int nranks,
    int cp_size,
    const uint8_t* unique_id_ptr,
    int mask_head = 1
);

// get instance (mutable ref), make sure the instance is initialized, used in both fwd and bwd
OverlapCommunicator<cutlass::bfloat16_t>& singleton();

// check whether the singleton unique_ptr is nullptr
bool is_singleton_null();

}   // namespace comm

}   // namespace flashmask
