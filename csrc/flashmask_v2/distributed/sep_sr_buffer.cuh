/**
 * Separated S-R buffer for a2a based all-gather 
*/
#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdexcept>
#include <cstring>
#include <array>

namespace flashmask {

// RAII object of separate send/recv buffer

#define CLAMP_IDX(idx) (idx & _idx_mask)
template <typename KVType>
class SepSRBuffer {
using SemaphoreType = int64_t;

private:
    KVType* _k_data;
    KVType* _v_data;
    SemaphoreType* _semaphores;
    bool _allocated;
    nvshmem_team_t _team;

    // offset to the recv buffer (2 * chunks_per_seg * k_numel)
    size_t _buf_offset;
    int _semaphore_size;

    const int _idx_mask;
    std::array<cudaEvent_t, 2> _empty_states; 

    // this object cannot be moved or copied
    SepSRBuffer(const SepSRBuffer&) = delete;
    SepSRBuffer(SepSRBuffer&&) = delete;
    SepSRBuffer& operator=(const SepSRBuffer&) = delete;
    SepSRBuffer& operator=(SepSRBuffer&&) = delete;
public:
    explicit SepSRBuffer(
        size_t single_k_numel, 
        int semaphore_size,
        int chunks_per_seg,
        int buffer_capacity = 1,
        nvshmem_team_t team = NVSHMEM_TEAM_WORLD
    );

    void team_bar() const {
        nvshmem_team_sync(_team);
    }

    void team_bar_on_stream(cudaStream_t stream) const {
        nvshmemx_team_sync_on_stream(_team, stream);
    }

    void release();

    ~SepSRBuffer();

    // [K_send, V_send] --> buf_offset size, therefore 2 * buf_offset is the double buffer offset
    inline KVType* k_send(int seg_idx) const { return _k_data + CLAMP_IDX(seg_idx) * 2 * _buf_offset; }
    inline KVType* v_send(int seg_idx) const { return _v_data + CLAMP_IDX(seg_idx) * 2 * _buf_offset; }
    inline KVType* k_recv(int seg_idx) const { return _k_data + (CLAMP_IDX(seg_idx) * 2 + 1) * _buf_offset; }
    inline KVType* v_recv(int seg_idx) const { return _v_data + (CLAMP_IDX(seg_idx) * 2 + 1) * _buf_offset; }
    inline SemaphoreType* semaphores(int seg_idx) const { return _semaphores + CLAMP_IDX(seg_idx) * _semaphore_size; }

    void wait_buffer(int seg_idx, cudaStream_t stream) {
        cudaStreamWaitEvent(stream, _empty_states[CLAMP_IDX(seg_idx)]);
    }

    void release_buffer(int seg_idx, cudaStream_t stream) {
        cudaEventRecord(_empty_states[CLAMP_IDX(seg_idx)], stream);
    }

    void reset(cudaStream_t comm_stream);
    // clear recv buffer (so that reduce won't op on dirty data)
    void zero_recv_buf(int seg_idx, cudaStream_t comm_stream);

    inline bool is_valid() const noexcept {
        return _allocated && _k_data && _v_data && _semaphores && _team != NVSHMEM_TEAM_INVALID;
    }

    nvshmem_team_t team() const noexcept {
        return _team;
    }

    void swap(SepSRBuffer& other);
};

#undef CLAMP_IDX

}   // namespace flashmask