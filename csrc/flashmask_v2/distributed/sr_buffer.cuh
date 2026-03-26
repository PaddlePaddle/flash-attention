/**
 * S-R buffer for a2a-based gather
*/
#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdexcept>
#include <cstring>

namespace flashmask {

// Note(heqianyue): This is a strange impasse, calling nvshmem_free / nvshmem_finalize
// for statically-managed instance might be deadly, there can be mutex-failure / failed
// cuda resources handle, etc.. skipping free-ing and finalizing can get rid of this error
// at the risk of being seemingly unsafe. I've tested it though, there is no leaking
// or potential bugs found. I suppose the OS and nv-driver is doing the resource management for us.
// To avoid blowing-up the process during exitting, we can choose to leave the cleanup to the driver and OS.
static constexpr bool MANUAL_CLEANUP = false;
using SemaphoreType = int64_t;

// RAII object of send/recv buffer
template <typename KVType>
class SRBuffer {
private:
    KVType* _k_sr;
    KVType* _v_sr;
    SemaphoreType* _semaphores;
    bool _allocated;
    nvshmem_team_t _team;
    size_t _numel;  // allocated capacity in numel (for one of K or V)

    // no copy, move-only object
    SRBuffer(const SRBuffer&) = delete;
    SRBuffer& operator=(const SRBuffer&) = delete;
public:
    explicit SRBuffer(size_t numel, nvshmem_team_t team = NVSHMEM_TEAM_WORLD, int semaphore_size = 0) :
        _k_sr(nullptr), _v_sr(nullptr), _semaphores(nullptr), _allocated(false), _team(team), _numel(0)
    {
        if (numel == 0) {
            throw std::invalid_argument("SRBuffer: numel must be positive");
        }
        if (numel & 31) {
            throw std::invalid_argument("SRBuffer: numel should be a multiple of 32");
        }

        size_t total_elements = 2 * numel + semaphore_size * sizeof(SemaphoreType) / sizeof(KVType);
        size_t total_bytes = total_elements * sizeof(KVType);
        
        _k_sr = static_cast<KVType*>(nvshmem_malloc(total_bytes));
        if (!_k_sr) {
            throw std::bad_alloc();
        }
        _v_sr = _k_sr + numel;
        _semaphores = reinterpret_cast<SemaphoreType*>(_v_sr + numel);
        _allocated = true;
        _numel = numel;
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
          _team(other._team),
          _numel(other._numel)
    {
        other._k_sr = nullptr;
        other._v_sr = nullptr;
        other._semaphores = nullptr;
        other._team = NVSHMEM_TEAM_INVALID;
        other._allocated = false;
        other._numel = 0;
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
            
            _numel = other._numel;

            other._k_sr = nullptr;
            other._v_sr = nullptr;
            other._semaphores = nullptr;
            other._allocated = false;
            other._team = NVSHMEM_TEAM_INVALID;
            other._numel = 0;
        }
        return *this;
    }

    void release() {
        if (_allocated && _k_sr) {
            if constexpr (MANUAL_CLEANUP) {
                nvshmem_free(_k_sr);
            }
            _k_sr = nullptr;
            _v_sr = nullptr;
            _semaphores = nullptr;
            _allocated = false;
            _team = NVSHMEM_TEAM_INVALID;
            _numel = 0;
        }
    }

    // Unconditionally free NVSHMEM memory for runtime reallocation.
    // Unlike release() which is gated by MANUAL_CLEANUP (for safe process-exit),
    // this method always calls nvshmem_free. Must be called with all PEs synchronized.
    void release_for_realloc() {
        if (_allocated && _k_sr) {
            nvshmem_free(_k_sr);
            _k_sr = nullptr;
            _v_sr = nullptr;
            _semaphores = nullptr;
            _allocated = false;
            _team = NVSHMEM_TEAM_INVALID;
            _numel = 0;
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

    inline SemaphoreType* semaphores() const noexcept {
        return _semaphores;
    }

    inline bool is_valid() const noexcept {
        return _allocated && _k_sr && _v_sr && _semaphores && _team != NVSHMEM_TEAM_INVALID;
    }

    size_t capacity() const noexcept {
        return _numel;
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
        std::swap(_numel, other._numel);
    }
};

}   // namespace flashmask