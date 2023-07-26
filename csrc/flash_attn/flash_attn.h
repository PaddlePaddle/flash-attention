#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

bool flash_attn_fwd(
        const void *q,
	const void *k,
	const void *v,
	void *out,
	const int batch_size,
	const int seqlen_q,
	const int seqlen_k,
	const int num_heads,
	const int num_heads_k,
	const int head_size,
	const float p_dropout,
	const float softmax_scale,
	const bool is_causal,
	const bool return_softmax,
	const bool is_bf16,
	void *softmax_ptr,
	void *softmax_lse_ptr,
	cudaStream_t stream,
	uint64_t seed,
	uint64_t offset);

bool flash_attn_varlen_fwd(
        void * const q,
	void * const k,
	void * const v,
	void * const out,
	void * const cu_seqlens_q,
	void * const cu_seqlens_k,
	const int batch_size,
	const int max_seqlen_q,
	const int max_seqlen_k,
	const int seqlen_q_rounded,
	const int seqlen_k_rounded,
	const int num_heads,
	const int num_heads_k,
	const int head_size,
	const int head_size_rounded,
	const float p_dropout,
	const float softmax_scale,
	const bool zero_tensors,
	const bool is_causal,
	const bool return_softmax,
	const bool is_bf16,
	void * const softmax_ptr,
        void * const softmax_lse_ptr,
	cudaStream_t stream,
	uint64_t seed,
	uint64_t offset);

void flash_attn_set_error(const char *msg);

const char *flash_attn_error();

#ifdef __cplusplus
}
#endif
