#pragma once

#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdexcept>
#include <cstring>

namespace flashmask {

__global__ void ResetSemaphoreKernel(
    int* const __restrict__ semaphores, 
    int my_pe, int semaphore_size
) {
    if (threadIdx.x < semaphore_size) {
        semaphores[threadIdx.x] = threadIdx.x == my_pe ? 1 : 0;
    }
}

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
        _semaphores = static_cast<int*>(_v_sr + numel);
        _allocated = true;
        // TODO(heqianyue): is this optional? Will this barrier introduce significant runtime overhead?
        team_bar();
    }

    void team_bar() const {
        nvshmem_team_sync(_team);
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

    // always reset the semaphore before prepare_sender and remote get kernel
    void reset_semaphores(int my_pe, int n_pes, cudaStream_t stream) {
        ResetSemaphoreKernel<<<1, 32, stream>>>(_semaphores, my_pe, n_pes);
    }

    // copy data from the local non-nvshmem allocated buffer
    // and signal with broadcast
    void prepare_sender(
        const KVType* const k_local,
        const KVType* const v_local,
        size_t sr_buffer_offset,
        size_t numel,
        int my_pe,
        cudaStream_t stream
    ) {
        // clear the local semaphore
        cudaMemcpyAsync(_k_sr + sr_buffer_offset, k_local, 
                        numel * sizeof(KVType), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(_v_sr + sr_buffer_offset, v_local,
                        numel * sizeof(KVType), cudaMemcpyDeviceToDevice, stream);
        // broadcast signal on aux_stream, tell comm_stream remote_get kernel KV data is ready
        nvshmemx_int_broadcast_on_stream(_team,
                            &_semaphores[my_pe],            // dst (remote)
                            &_semaphores[my_pe],            // src (local)
                            1,                              // n_elems
                            my_pe,                          // root
                            stream);
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