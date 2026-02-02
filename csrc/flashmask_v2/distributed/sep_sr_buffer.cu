/**
 * Statistics: for extreme use case, CP16 N4
 *  K shape (1, 128K, 8, 128) -> CP -> (1, 8192, 8, 128)
 * 
 *  We need (4 or 2, fp32 or bf16) * 2 (KV) * 2 (SR) * 4 (num_chunks) * 
 *  numel = 256M (bf16) or 512M (bf16). If double buffering, x2. 
 * So I suppose we should not use double buffering to alleviate GMEM consumption
*/

#include "sep_sr_buffer.cuh"
#include <cutlass/bfloat16.h>

namespace flashmask {

static constexpr bool MANUAL_CLEANUP = false;

template <typename KVType>
SepSRBuffer<KVType>::SepSRBuffer(
    size_t single_k_numel, 
    int semaphore_size,
    int chunks_per_seg,
    nvshmem_team_t team
) : 
    _k_data(nullptr), _v_data(nullptr), _semaphores(nullptr), 
    _allocated(false), _team(team), 
    _buf_offset(2 * chunks_per_seg * single_k_numel), 
    _semaphore_size(semaphore_size)
{
    if (single_k_numel & 31) {
        throw std::invalid_argument("SepSRBuffer: numel should be the positive multiple of 32");
    }

    // 2 = (K & V -->) 2 * (send recv -->) 2
    size_t total_elements = 4 * chunks_per_seg * single_k_numel + 
            semaphore_size * sizeof(SemaphoreType) / sizeof(KVType);
    size_t total_bytes = total_elements * sizeof(KVType);
    
    _k_data = static_cast<KVType*>(nvshmem_malloc(total_bytes));
    if (!_k_data) {
        throw std::bad_alloc();
    }
    _v_data = _k_data + chunks_per_seg * single_k_numel;
    _semaphores = reinterpret_cast<SemaphoreType*>(_k_data + 4 * chunks_per_seg * single_k_numel);
    _allocated = true;
}

template <typename KVType>
SepSRBuffer<KVType>::SepSRBuffer(SepSRBuffer<KVType>&& other)
    : _k_data(other._k_data), 
        _v_data(other._v_data), 
        _allocated(other._allocated),
        _team(other._team),
        _buf_offset(other._buf_offset),
        _semaphore_size(other._semaphore_size)
{
    other._k_data = nullptr;
    other._v_data = nullptr;
    other._semaphores = nullptr;
    other._allocated = false;
    other._team = NVSHMEM_TEAM_INVALID;
    other._buf_offset = 0;
    other._semaphore_size = 0;
}

template <typename KVType>
SepSRBuffer<KVType>& 
SepSRBuffer<KVType>::operator=(SepSRBuffer<KVType>&& other) {
    if (this != &other) {
        if (_allocated && _k_data) {
            nvshmem_free(_k_data);
        }
        
        _k_data = other._k_data;
        _v_data = other._v_data;
        _semaphores = other._semaphores;
        _allocated = other._allocated;
        _team = other._team;
        _buf_offset = other._buf_offset;
        _semaphore_size = other._semaphore_size;
        
        other._k_data = nullptr;
        other._v_data = nullptr;
        other._semaphores = nullptr;
        other._allocated = false;
        other._team = NVSHMEM_TEAM_INVALID;
        other._buf_offset = 0;
        other._semaphore_size = 0;
    }
    return *this;
}

template <typename KVType>
void SepSRBuffer<KVType>::release() {
    if (_allocated && _k_data) {
        if constexpr (MANUAL_CLEANUP) {
            nvshmem_free(_k_data);
        }
        _k_data = nullptr;
        _v_data = nullptr;
        _semaphores = nullptr;
        _allocated = false;
        _team = NVSHMEM_TEAM_INVALID;
        _buf_offset = 0;
        _semaphore_size = 0;
    }
}

template <typename KVType>
SepSRBuffer<KVType>::~SepSRBuffer() {
    release();
}

template <typename KVType>
void SepSRBuffer<KVType>::reset(cudaStream_t comm_stream) {
    cudaMemsetAsync(_semaphores, 0, sizeof(KVType) * _semaphore_size, comm_stream);
}

template <typename KVType>
void SepSRBuffer<KVType>::zero_recv_buf(cudaStream_t comm_stream) {
    cudaMemsetAsync(_k_data + _buf_offset, 0, sizeof(KVType) * _buf_offset, comm_stream);
}

template <typename KVType>
void SepSRBuffer<KVType>::swap(SepSRBuffer& other) {
    std::swap(_k_data, other._k_data);
    std::swap(_v_data, other._v_data);
    std::swap(_semaphores, other._semaphores);
    std::swap(_allocated, other._allocated);
    std::swap(_team, other._team);
    std::swap(_buf_offset, other._buf_offset);
    std::swap(_semaphore_size, other._semaphore_size);
}

template class SepSRBuffer<cutlass::bfloat16_t>;

}   // namespace flashmask

