/**
 * Separated S-R buffer for a2a based all-gather 
*/
#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdexcept>
#include <cstring>

namespace flashmask {

// RAII object of separate send/recv buffer
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

    // no copy, move-only object
    SepSRBuffer(const SepSRBuffer&) = delete;
    SepSRBuffer& operator=(const SepSRBuffer&) = delete;
public:
    explicit SepSRBuffer(
        size_t single_k_numel, 
        int semaphore_size,
        int chunks_per_seg,
        nvshmem_team_t team = NVSHMEM_TEAM_WORLD
    );

    void team_bar() const {
        nvshmem_team_sync(_team);
    }

    void team_bar_on_stream(cudaStream_t stream) const {
        nvshmemx_team_sync_on_stream(_team, stream);
    }

    SepSRBuffer(SepSRBuffer&& other);

    SepSRBuffer& operator=(SepSRBuffer&& other);

    void release();

    ~SepSRBuffer();

    inline KVType* k_send() const { return _k_data; }
    inline KVType* v_send() const { return _v_data; }
    inline KVType* k_recv() const { return _k_data + _buf_offset; }
    inline KVType* v_recv() const { return _v_data + _buf_offset; }
    inline SemaphoreType* semaphores() const { return _semaphores; }

    void reset(cudaStream_t comm_stream);

    inline bool is_valid() const noexcept {
        return _allocated && _k_data && _v_data && _semaphores && _team != NVSHMEM_TEAM_INVALID;
    }

    nvshmem_team_t team() const noexcept {
        return _team;
    }

    void swap(SepSRBuffer& other);
};

}   // namespace flashmask