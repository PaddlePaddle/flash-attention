#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

bool flash_attn_fwd(const void * const q,         // batch_size x seqlen_q x num_heads x head_size
                    const void * const k,         // batch_size x seqlen_k x num_heads_k x head_size
                    const void * const v,         // batch_size x seqlen_k x num_heads_k x head_size
                    void * const rng_state,
                    void * const out,
                    void * const softmax_ptr,
                    void * const softmax_lse_ptr,
                    const int batch_size,
                    const int seqlen_q,
                    const int seqlen_k,
                    const int seqlen_q_rounded,
                    const int seqlen_k_rounded,
                    const int num_heads,
                    const int num_heads_k,
                    const int head_size,
                    const int head_size_rounded,
                    const float p_dropout,
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool return_softmax,
                    const bool is_bf16,
                    hipStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims,
                    const bool is_infer);

bool flash_attn_varlen_fwd(const void * const q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                           const void * const k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                           const void * const v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                           const int32_t * const cu_seqlens_q,  // b+1
                           const int32_t * const cu_seqlens_k,  // b+1
                           void * const rng_state,
                           void * const out, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                           void * const softmax_ptr,
                           void * const softmax_lse_ptr,
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
                           const float softmax_unscale,
                           const bool is_causal,
                           const bool return_softmax,
                           const bool is_bf16,
                           hipStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * const attn_mask,
                           const void * const mask_dims);

bool flash_attn_bwd(const void * const dout,  // batch_size x seqlen_q x num_heads, x head_size_og
                    const void * const q,   // batch_size x seqlen_q x num_heads x head_size
                    const void * const k,   // batch_size x seqlen_k x num_heads_k x head_size
                    const void * const v,   // batch_size x seqlen_k x num_heads_k x head_size
                    const void * const out,   // batch_size x seqlen_q x num_heads x head_size
                    const void * const softmax_d,
                    const void * const softmax_lse,     // b x h x seqlen_q
                    void * const rng_state,
                    void * const dq,   // batch_size x seqlen_q x num_heads x head_size
                    void * const dk,   // batch_size x seqlen_k x num_heads_k x head_size
                    void * const dv,   // batch_size x seqlen_k x num_heads_k x head_size
                    void * const dq_accum,
                    const int batch_size,
                    const int seqlen_q,
                    const int seqlen_k,
                    const int seqlen_q_rounded,
                    const int seqlen_k_rounded,
                    const int num_heads,
                    const int num_heads_k,
                    const int head_size,
                    const int head_size_rounded,
                    const float p_dropout,         // probability to drop
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool is_bf16,
                    const int num_splits,
                    hipStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims);

bool flash_attn_varlen_bwd(const void * const dout,  // total_q x num_heads, x head_size
                           const void * const q,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                           const void * const k,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                           const void * const v,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                           const void * const out,   // total_q x num_heads x head_size
                           const void * const softmax_d,
                           const void * const softmax_lse,     // b x h x s   softmax logsumexp
                           const int32_t * const cu_seqlens_q,  // b+1
                           const int32_t * const cu_seqlens_k,  // b+1
                           void * const rng_state,
                           void * const dq,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                           void * const dk,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                           void * const dv,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                           void * const dq_accum,
                           const int batch_size,
                           const int max_seqlen_q,
                           const int max_seqlen_k,          // max sequence length to choose the kernel
                           const int seqlen_q_rounded,
                           const int seqlen_k_rounded,
                           const int num_heads,
                           const int num_heads_k,
                           const int head_size,
                           const int head_size_rounded,
                           const float p_dropout,         // probability to drop
                           const float softmax_scale,
                           const float softmax_unscale,
                           const bool is_causal,
                           const bool is_bf16,
                           const int num_splits,
                           hipStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * attn_mask,
                           const int64_t * const mask_dims);

void flash_attn_set_error(const char *msg);

const char *flash_attn_error();

#ifdef __cplusplus
}
#endif
