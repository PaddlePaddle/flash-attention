#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdexcept>
#include <cstring>

namespace flashmask {

// RAII object of send/recv buffer
template <typename KVType>
class SRBuffer {
private:
    KVType* _k_sr;
    KVType* _v_sr;
    int* _semaphores;
    bool _allocated;
    nvshmem_team_t _team;

    // no copy, move-only object
    SRBuffer(const SRBuffer&) = delete;
    SRBuffer& operator=(const SRBuffer&) = delete;
public:
    explicit SRBuffer(size_t numel, nvshmem_team_t team = NVSHMEM_TEAM_WORLD, int semaphore_size = 0) : 
        _k_sr(nullptr), _v_sr(nullptr), _semaphores(nullptr), _allocated(false), _team(team) 
    {
        if (numel == 0) {
            throw std::invalid_argument("SRBuffer: numel must be positive");
        }
        if (numel & 31) {
            throw std::invalid_argument("SRBuffer: numel should be a multiple of 32");
        }

        size_t total_elements = 2 * numel + semaphore_size * sizeof(int) / sizeof(KVType);
        size_t total_bytes = total_elements * sizeof(KVType);
        
        _k_sr = static_cast<KVType*>(nvshmem_malloc(total_bytes));
        if (!_k_sr) {
            throw std::bad_alloc();
        }
        _v_sr = _k_sr + numel;
        _semaphores = reinterpret_cast<int*>(_v_sr + numel);
        _allocated = true;
    }

    void team_bar() const {
        nvshmem_team_sync(_team);
    }

    void team_bar_on_stream(cudaStream_t stream) const {
        nvshmemx_team_sync_on_stream(_team, stream);
    }

    SRBuffer(SRBuffer&& other) noexcept
        : _k_sr(other._k_sr), 
          _v_sr(other._v_sr), 
          _allocated(other._allocated),
          _team(other._team)
    {
        other._k_sr = nullptr;
        other._v_sr = nullptr;
        other._semaphores = nullptr;
        other._team = NVSHMEM_TEAM_INVALID;
        other._allocated = false;
    }

    SRBuffer& operator=(SRBuffer&& other) noexcept {
        if (this != &other) {
            if (_allocated && _k_sr) {
                nvshmem_free(_k_sr);
            }
            
            _k_sr = other._k_sr;
            _v_sr = other._v_sr;
            _semaphores = other._semaphores;
            _allocated = other._allocated;
            _team = other._team;
            
            other._k_sr = nullptr;
            other._v_sr = nullptr;
            other._semaphores = nullptr;
            other._allocated = false;
            other._team = NVSHMEM_TEAM_INVALID;
        }
        return *this;
    }

    void release() {
        if (_allocated && _k_sr) {
            nvshmem_free(_k_sr);
            _k_sr = nullptr;
            _v_sr = nullptr;
            _semaphores = nullptr;
            _allocated = false;
            _team = NVSHMEM_TEAM_INVALID;
        }
    }

    ~SRBuffer() noexcept {
        release();
    }

    inline KVType* k_data() const noexcept {
        return _k_sr;
    }

    inline KVType* v_data() const noexcept {
        return _v_sr;
    }

    inline int* semaphores() const noexcept {
        return _semaphores;
    }

    inline bool is_valid() const noexcept {
        return _allocated && _k_sr && _v_sr && _semaphores && _team != NVSHMEM_TEAM_INVALID;
    }

    nvshmem_team_t team() const noexcept {
        return _team;
    }

    void swap(SRBuffer& other) noexcept {
        std::swap(_k_sr, other._k_sr);
        std::swap(_v_sr, other._v_sr);
        std::swap(_semaphores, other._semaphores);
        std::swap(_allocated, other._allocated);
        std::swap(_team, other._team);
    }
};

}   // namespace flashmask