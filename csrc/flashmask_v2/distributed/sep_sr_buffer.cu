/**
 * Statistics: for extreme use case, CP16 N4
 *  K shape (1, 128K, 8, 128) -> CP -> (1, 8192, 8, 128)
 * 
 *  We need (4 or 2, fp32 or bf16) * 2 (KV) * 2 (SR) * 4 (num_chunks) * 
 *  numel = 256M (bf16) or 512M (bf16). If double buffering, x2. 
 * So I suppose we should not use double buffering to alleviate GMEM consumption
*/

#include "sep_sr_buffer.cuh"
#include "debug_logger.cuh"
#include <cutlass/bfloat16.h>

namespace flashmask {

static constexpr bool MANUAL_CLEANUP = false;

template <typename KVType>
SepSRBuffer<KVType>::SepSRBuffer(
    size_t single_k_numel, 
    int semaphore_size,
    int chunks_per_seg,
    int buffer_capacity,
    nvshmem_team_t team
) : 
    _k_data(nullptr), _v_data(nullptr), _semaphores(nullptr), 
    _allocated(false), _idx_mask(buffer_capacity - 1), _team(team), 
    _buf_offset(2 * chunks_per_seg * single_k_numel), 
    _semaphore_size(semaphore_size)
{
    if (single_k_numel & 31) {
        throw std::invalid_argument("SepSRBuffer: numel should be the positive multiple of 32");
    }
    if (buffer_capacity > 2 || buffer_capacity <= 0) {
        throw std::invalid_argument("SepSRBuffer: only support upto double buffer. Buffer capacity should be 1 or 2.");
    }

    // 2 = (K & V -->) 2 * (send recv -->) 2
    size_t total_elements = 4 * chunks_per_seg * single_k_numel + 
            semaphore_size * sizeof(SemaphoreType) / sizeof(KVType);

    total_elements *= buffer_capacity;
    for (int i = 0; i < buffer_capacity; i++)
        cudaEventCreateWithFlags(&_empty_states[i], cudaEventDisableTiming);

    size_t total_bytes = total_elements * sizeof(KVType);
    
    _k_data = static_cast<KVType*>(nvshmem_malloc(total_bytes));
    if (!_k_data) {
        throw std::bad_alloc();
    }
    _v_data = _k_data + chunks_per_seg * single_k_numel;
    _semaphores = reinterpret_cast<SemaphoreType*>(_k_data + 2 * buffer_capacity * _buf_offset);
    _allocated = true;
}

template <typename KVType>
void SepSRBuffer<KVType>::release() {
    if (_allocated && _k_data) {
        if constexpr (MANUAL_CLEANUP) {
            nvshmem_free(_k_data);
        }
        for (int i = 0; i < _idx_mask + 1; i++) {
            CUDA_DEBUG_CHECK(cudaEventDestroy(_empty_states[i]));
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
void SepSRBuffer<KVType>::reset_semaphores() {
    size_t semaphore_bytes = _semaphore_size * sizeof(SemaphoreType);
    // set recv buffer and semaphores to be 0 all at once
    cudaMemset(_semaphores, 0, semaphore_bytes * (1 + _idx_mask));
}

template <typename KVType>
void SepSRBuffer<KVType>::zero_recv_buf(int seg_idx, cudaStream_t comm_stream) {
    cudaMemsetAsync(_k_data + _buf_offset * (1 + 2 * (seg_idx & _idx_mask)), 0, sizeof(KVType) * _buf_offset, comm_stream);
}

template class SepSRBuffer<cutlass::bfloat16_t>;

}   // namespace flashmask

