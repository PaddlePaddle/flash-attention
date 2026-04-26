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
    _dk_data(nullptr), _dv_data(nullptr), _semaphores(nullptr),
    _allocated(false), _capacity(buffer_capacity), _team(team),
    _buf_offset(2 * chunks_per_seg * single_k_numel),
    _semaphore_size(semaphore_size),
    _single_k_numel(single_k_numel),
    _chunks_per_seg(chunks_per_seg)
{
    if (single_k_numel & 31) {
        throw std::invalid_argument("SepSRBuffer: numel should be the positive multiple of 32");
    }
    if (buffer_capacity <= 0) {
        throw std::invalid_argument("SepSRBuffer: buffer_capacity must be > 0, got: " + std::to_string(buffer_capacity));
    }

    // 2 = (K & V -->) 2 * (send recv -->) 2
    size_t total_elements = 4 * chunks_per_seg * single_k_numel + 
            semaphore_size * sizeof(SemaphoreType) / sizeof(KVType);

    total_elements *= buffer_capacity;
    _empty_states.resize(buffer_capacity);
    for (int i = 0; i < buffer_capacity; i++)
        cudaEventCreateWithFlags(&_empty_states[i], cudaEventDisableTiming);

    size_t total_bytes = total_elements * sizeof(KVType);
    
    _dk_data = static_cast<KVType*>(nvshmem_malloc(total_bytes));
    if (!_dk_data) {
        throw std::bad_alloc();
    }
    _dv_data = _dk_data + chunks_per_seg * single_k_numel;
    _semaphores = reinterpret_cast<SemaphoreType*>(_dk_data + 2 * buffer_capacity * _buf_offset);
    _allocated = true;
}

template <typename KVType>
void SepSRBuffer<KVType>::release() {
    if (_allocated && _dk_data) {
        if constexpr (MANUAL_CLEANUP) {
            nvshmem_free(_dk_data);
        }
        for (int i = 0; i < _capacity; i++) {
            CUDA_DEBUG_CHECK(cudaEventDestroy(_empty_states[i]));
        }
        _dk_data = nullptr;
        _dv_data = nullptr;
        _semaphores = nullptr;
        _allocated = false;
        _team = NVSHMEM_TEAM_INVALID;
        _buf_offset = 0;
        _semaphore_size = 0;
        _single_k_numel = 0;
        _chunks_per_seg = 0;
    }
}

template <typename KVType>
void SepSRBuffer<KVType>::release_for_realloc() {
    if (_allocated && _dk_data) {
        nvshmem_free(_dk_data);
        for (int i = 0; i < _capacity; i++) {
            CUDA_DEBUG_CHECK(cudaEventDestroy(_empty_states[i]));
        }
        _dk_data = nullptr;
        _dv_data = nullptr;
        _semaphores = nullptr;
        _allocated = false;
        _team = NVSHMEM_TEAM_INVALID;
        _buf_offset = 0;
        _semaphore_size = 0;
        _single_k_numel = 0;
        _chunks_per_seg = 0;
    }
}

template <typename KVType>
SepSRBuffer<KVType>::~SepSRBuffer() {
    release();
}

template <typename KVType>
void SepSRBuffer<KVType>::zero_recv_buf(int seg_idx, cudaStream_t comm_stream) {
    cudaMemsetAsync(_dk_data + _buf_offset * (1 + 2 * (seg_idx % _capacity)), 0, sizeof(KVType) * _buf_offset, comm_stream);
}

template <typename KVType>
void SepSRBuffer<KVType>::initialize_buffer(int self_rank, bool per_stage_buffer) {
    if (per_stage_buffer) {
        // all the GPU ops uses the default blocking stream, since this is only called during initialization (one-off)
        size_t recv_buffer_sz = sizeof(KVType) * _buf_offset;
        for (int stage = 0; stage < _capacity; stage++) {
            cudaMemset(_dk_data + _buf_offset * (1 + 2 * (stage % _capacity)), 0, recv_buffer_sz);
        }
    }
    size_t semaphore_bytes = _semaphore_size * sizeof(SemaphoreType);
    // set recv buffer and semaphores to be 0 all at once
    cudaMemset(_semaphores, 0, semaphore_bytes * _capacity);
}

template class SepSRBuffer<cutlass::bfloat16_t>;

}   // namespace flashmask

