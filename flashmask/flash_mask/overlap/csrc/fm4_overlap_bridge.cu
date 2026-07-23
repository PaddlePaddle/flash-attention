// FM-4 Overlap bridge: a thin extern "C" wrapper around the FM-3
// flashmask::comm NVSHMEM overlap singleton, so Python (ctypes) can drive the
// proven FM-3 comm runtime and pull the SRBuffer raw pointer back out to wrap
// as a cute tensor. Every symbol is prefixed fm4_overlap_ so one version-script
// glob exports exactly this surface; all pointers cross as uint64_t.

#include "overlap_comm.cuh"
#include "cutlass/bfloat16.h"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

using bf16 = cutlass::bfloat16_t;

namespace {

// uint64 handle from Python -> cudaStream_t (0 == default stream).
inline cudaStream_t as_stream(uint64_t handle) {
    return reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(handle));
}

// FM-4 does not run prepare_flashmask_kernel, so reset the persistent AG
// scheduler counter here before every forward or backward gather.
__global__ void fm4_reset_ag_counter_kernel(int* const block_cnt_semaphore) {
    if (threadIdx.x == 0) { *block_cnt_semaphore = 1; }
}

}  // namespace

extern "C" {

// rank 0 fills `out` with the NVSHMEM unique id; caller broadcasts it and passes
// the same bytes to fm4_overlap_init. Returns 1.
int fm4_overlap_get_unique_id(uint8_t* out) {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::memcpy(out, &unique_id, sizeof(nvshmemx_uniqueid_t));
    return 1;
}

// Create or reconfigure the singleton. Only shape (b/s/h/d) + topology
// (rank/nranks/uid) are needed; the ctor stores but never derefs k/v, so we pass
// nullptr (the local-KV copy happens later in fm4_overlap_update_kv). Returns 1.
int fm4_overlap_init(int b_kv, int s_kv, int h_kv, int d_kv,
                     int rank, int nranks,
                     const uint8_t* unique_id, int mask_head) {
    flashmask::comm::init_singleton_instance(
        static_cast<const bf16*>(nullptr), static_cast<const bf16*>(nullptr),
        b_kv, s_kv, h_kv, d_kv, rank, nranks, unique_id, mask_head);
    return 1;
}

// SRBuffer K/V base pointers as raw uint64 (wrapped into cute tensors in Python).
uint64_t fm4_overlap_k_data() {
    return reinterpret_cast<uint64_t>(flashmask::comm::singleton().k_data());
}
uint64_t fm4_overlap_v_data() {
    return reinterpret_cast<uint64_t>(flashmask::comm::singleton().v_data());
}
uint64_t fm4_overlap_work_done() {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().get_work_done_ptr());
}

// Local seqlen chunk (S_local); after AG, S_total = s_local * nranks.
int fm4_overlap_s_local() { return flashmask::comm::singleton().s_local(); }
int fm4_overlap_nranks() { return flashmask::comm::singleton().nranks(); }
int fm4_overlap_use_bhsd_layout() {
    return flashmask::comm::singleton().use_bhsd_layout() ? 1 : 0;
}
int fm4_overlap_use_hierarchical() {
    return flashmask::comm::singleton().use_hierarchical() ? 1 : 0;
}
int fm4_overlap_comm_rpb() { return flashmask::comm::singleton().get_comm_rpb(); }
int fm4_overlap_num_segments() { return flashmask::comm::singleton().num_segments(); }
int fm4_overlap_segment_seqlen() {
    auto& c = flashmask::comm::singleton();
    return c.s_local() * c.chunk_per_seg();
}
uint64_t fm4_overlap_segment_k_data(int segment_idx) {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().segment_k_data(segment_idx));
}
uint64_t fm4_overlap_segment_v_data(int segment_idx) {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().segment_v_data(segment_idx));
}
uint64_t fm4_overlap_dk_send(int segment_idx) {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().dk_send(segment_idx));
}
uint64_t fm4_overlap_dv_send(int segment_idx) {
    return reinterpret_cast<uint64_t>(
        flashmask::comm::singleton().dv_send(segment_idx));
}

// Copy local K/V into the SRBuffer on the internal comm_stream (cudaMemcpyAsync).
void fm4_overlap_update_kv(uint64_t k_ptr, uint64_t v_ptr, int fwd) {
    flashmask::comm::singleton().update_kv_buffer(
        reinterpret_cast<const bf16*>(static_cast<uintptr_t>(k_ptr)),
        reinterpret_cast<const bf16*>(static_cast<uintptr_t>(v_ptr)), fwd != 0);
}

// Compute the per-chunk sparsity mask (copy_chunk_mask) so the AG kernel can skip
// fully-masked chunks. MUST precede fm4_overlap_run_ag (the kernel reads the mask
// at remote_get_kernel.cuh:383; it is uninitialized otherwise). lt_start/ut_end
// are device int32 (B, H_mask, S_total); lt_end/ut_start are null (non-causal).
// Pass the compute stream so the write is ordered before the AG kernel picks it
// up via the wait_sr_buffer_empty handshake.
void fm4_overlap_compute_chunk_mask(uint64_t lt_start_ptr, uint64_t ut_end_ptr,
                                    uint64_t stream, int fwd) {
    flashmask::comm::singleton().compute_chunk_mask(
        reinterpret_cast<const int*>(static_cast<uintptr_t>(lt_start_ptr)),
        nullptr, nullptr,
        reinterpret_cast<const int*>(static_cast<uintptr_t>(ut_end_ptr)),
        as_stream(stream), fwd != 0);
}

// Launch the all-gather remote-get kernel on the internal comm_stream. write_ptr
// is the caller's device int the gate spins on. The communicator rewrites S to
// S_local * nranks; returned via s_total_out.
void fm4_overlap_run_ag(uint64_t write_ptr_dev, int* s_total_out, int fwd) {
    int S = 0;
    flashmask::comm::singleton().run_overlap_ag_kernel(
        reinterpret_cast<int*>(static_cast<uintptr_t>(write_ptr_dev)), S, fwd != 0);
    if (s_total_out) { *s_total_out = S; }
}

// compute_stream notifies comm_stream that the local SRBuffer chunk may be reused.
void fm4_overlap_wait_sr_buffer_empty(uint64_t compute_stream) {
    flashmask::comm::singleton().wait_sr_buffer_empty(as_stream(compute_stream));
}

void fm4_overlap_prepare_dkv_buffer(uint64_t compute_stream) {
    flashmask::comm::singleton().prepare_dkv_buffer(as_stream(compute_stream));
}

void fm4_overlap_start_bwd_segment(int segment_idx, uint64_t compute_stream) {
    auto& c = flashmask::comm::singleton();
    cudaStream_t stream = as_stream(compute_stream);
    c.ensure_ag_done(stream);
    fm4_reset_ag_counter_kernel<<<1, 32, 0, stream>>>(c.get_block_cnt_semaphore());
    cudaEventRecord(c.wptr_init, stream);
    c.wait_wptr_init();
    c.run_overlap_splitted_ag_kernel(nullptr, segment_idx);
    c.wait_reset_stream_coordinator(stream);
}

void fm4_overlap_wait_dkv_buffer(int segment_idx, uint64_t compute_stream) {
    flashmask::comm::singleton().wait_dkv_buffer(segment_idx, as_stream(compute_stream));
}

void fm4_overlap_run_rs(uint64_t dk_ptr, uint64_t dv_ptr, int segment_idx,
                        uint64_t compute_stream) {
    flashmask::comm::singleton().run_overlap_rs_kernel(
        reinterpret_cast<bf16*>(static_cast<uintptr_t>(dk_ptr)),
        reinterpret_cast<bf16*>(static_cast<uintptr_t>(dv_ptr)),
        segment_idx, as_stream(compute_stream));
}

void fm4_overlap_wait_reduce_done(uint64_t compute_stream) {
    flashmask::comm::singleton().wait_reduce_done(as_stream(compute_stream));
}

// Reset the AG scheduler on the compute stream; pairs with
// fm4_overlap_wait_wptr_init on the comm stream.
void fm4_overlap_reset_ag_counter(uint64_t compute_stream) {
    auto& c = flashmask::comm::singleton();
    cudaStream_t s = as_stream(compute_stream);
    c.ensure_ag_done(s);
    fm4_reset_ag_counter_kernel<<<1, 32, 0, s>>>(c.get_block_cnt_semaphore());
    cudaEventRecord(c.wptr_init, s);
}

// comm_stream waits on wptr_init so the remote-get kernel reads the counter only
// after the reset lands. Pairs with fm4_overlap_reset_ag_counter.
void fm4_overlap_wait_wptr_init() {
    flashmask::comm::singleton().wait_wptr_init();
}

// compute stream waits until the comm kernel is actually scheduled onto SMs.
void fm4_overlap_wait_reset_stream_coordinator(uint64_t stream) {
    flashmask::comm::singleton().wait_reset_stream_coordinator(as_stream(stream));
}

// Legacy host-sync entry point retained for ABI compatibility. Active FM-4
// forward and split backward paths use asynchronous readiness handshakes instead.
void fm4_overlap_sync_comm_stream() {
    flashmask::comm::singleton().sync_comm_stream();
}

}  // extern "C"
