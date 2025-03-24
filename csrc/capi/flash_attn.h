#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
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
                    cudaStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims,
                    const void * const flashmask_downstart_ptr,
                    const int64_t * const flashmask_dims,
                    const void * const flashmask_upend_ptr,
                    const void * const flashmask_downend_ptr,
                    const void * const flashmask_upstart_ptr,
                    const void * const flashmask_maxmin_ptr,
                    const int q_row_stride,
                    const int k_row_stride,
                    const int v_row_stride,
                    const int q_head_stride,
                    const int k_head_stride,
                    const int v_head_stride,
                    const int o_row_stride,
                    const int o_head_stride,
                    const int q_batch_stride,
                    const int k_batch_stride,
                    const int v_batch_stride,
                    const int o_batch_stride);

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
                           cudaStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * const attn_mask,
                           const void * const mask_dims,
                           const int q_row_stride,
                           const int k_row_stride,
                           const int v_row_stride,
                           const int q_head_stride,
                           const int k_head_stride,
                           const int v_head_stride,
                           const int o_row_stride,
                           const int o_head_stride,
                           const int q_batch_stride,
                           const int k_batch_stride,
                           const int v_batch_stride,
                           const int o_batch_stride,
                           bool varlen_padded_input,
                           const bool unpadded_lse,
                           const int total_q);

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
                    cudaStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims,
                    const void * const flashmask_downstart_ptr,
                    const int64_t * const flashmask_dims,
                    const void * const flashmask_upend_ptr,
                    const void * const flashmask_downend_ptr,
                    const void * const flashmask_upstart_ptr,
                    const void * const flashmask_maxmin_ptr,
                    const int q_row_stride,
                    const int k_row_stride,
                    const int v_row_stride,
                    const int q_head_stride,
                    const int k_head_stride,
                    const int v_head_stride,
                    const int o_row_stride,
                    const int o_head_stride,
                    const int q_batch_stride,
                    const int k_batch_stride,
                    const int v_batch_stride,
                    const int o_batch_stride,
                    const int dq_row_stride,
                    const int dk_row_stride,
                    const int dv_row_stride,
                    const int dq_head_stride,
                    const int dk_head_stride,
                    const int dv_head_stride,
                    const int do_row_stride,
                    const int do_head_stride,
                    const int dq_batch_stride,
                    const int dk_batch_stride,
                    const int dv_batch_stride,
                    const int do_batch_stride);

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
                           cudaStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * attn_mask,
                           const int64_t * const mask_dims,
                           const int q_row_stride,
                           const int k_row_stride,
                           const int v_row_stride,
                           const int q_head_stride,
                           const int k_head_stride,
                           const int v_head_stride,
                           const int o_row_stride,
                           const int o_head_stride,
                           const int q_batch_stride,
                           const int k_batch_stride,
                           const int v_batch_stride,
                           const int o_batch_stride,
                           const int dq_row_stride,
                           const int dk_row_stride,
                           const int dv_row_stride,
                           const int dq_head_stride,
                           const int dk_head_stride,
                           const int dv_head_stride,
                           const int do_row_stride,
                           const int do_head_stride,
                           const int dq_batch_stride,
                           const int dk_batch_stride,
                           const int dv_batch_stride,
                           const int do_batch_stride,
                           const bool varlen_padded_input,
                           const bool unpadded_lse,
                           const int total_q);

bool calc_reduced_attn_scores(const void * const q,
                              const void * const k,
                              const void * const softmax_lse,
                              void * const reduced_scores,
                              void * const softmax_ptr,
                              const int batch_size,
                              const int seqlen_q,
                              const int seqlen_k,
                              const int num_heads,
                              const int num_heads_k,
                              const int head_size,
                              const float softmax_scale,
                              const bool return_softmax,
                              const bool is_bf16,
                              const int num_splits,
                              cudaStream_t stream,
                              const int q_row_stride,
                              const int k_row_stride,
                              const int o_row_stride,
                              const int q_head_stride,
                              const int k_head_stride,
                              const int o_head_stride,
                              const int q_batch_stride,
                              const int k_batch_stride,
                              const int o_batch_stride);

bool flash_attn_fwd_with_bias_and_mask(const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       void *out,                  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const int32_t *cu_seqlens_q,   // int32, batch_size+1, starting offset of each sequence
                                       const int32_t *cu_seqlens_k,   // int32, batch_size+1, starting offset of each sequence
                                       const int total_q,
                                       const int total_k,
                                       const int batch_size,
                                       const int num_heads,
                                       const int head_size,
                                       const int max_seqlen_q_,
                                       const int max_seqlen_k_,
                                       const float p_dropout,
                                       const float softmax_scale,
                                       const bool zero_tensors,
                                       const bool is_bf16,
                                       const int num_splits,        // SMs per attention matrix, can be 1
                                       void *softmax_lse_ptr,       // softmax log_sum_exp
                                       void *workspace_ptr,
                                       uint64_t *workspace_size,
                                       cudaStream_t stream,
                                       uint64_t seed,
                                       uint64_t offset,
                                       const void *attn_mask,
                                       const void *attn_bias,
                                       const int64_t* mask_dims,
                                       const int64_t* bias_dims);

bool flash_attn_bwd_with_bias_and_mask(const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       void *dq,                   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       void *dk,                   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       void *dv,                   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const void *out,            // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const void *dout,           // total_q x num_heads, x head_size
                                       const int32_t *cu_seqlens_q,   // int32, batch_size+1
                                       const int32_t *cu_seqlens_k,   // int32, batch_size+1
                                       const int total_q,
                                       const int total_k,
                                       const int batch_size,
                                       const int num_heads,
                                       const int head_size,
                                       const int max_seqlen_q_,
                                       const int max_seqlen_k_,
                                       const float p_dropout,
                                       const float softmax_scale,
                                       const bool zero_tensors,
                                       const bool is_bf16,
                                       const int num_splits,
                                       const void *softmax_lse_ptr,
                                       void *dsoftmax_ptr,
                                       void *dbias_ptr,
                                       void *workspace_ptr,
                                       uint64_t *workspace_size,
                                       cudaStream_t stream,
                                       uint64_t seed,
                                       uint64_t offset,
                                       const void* attn_mask,
                                       const void* attn_bias,
                                       const int64_t* mask_dims,
                                       const int64_t* bias_dims);

void flash_attn_set_error(const char *msg);

const char *flash_attn_error();

#ifdef __cplusplus
}
#endif
