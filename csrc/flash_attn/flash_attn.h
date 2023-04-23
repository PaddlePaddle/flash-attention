#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

bool flash_attn_fwd(
        const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        void *out,                  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *cu_seqlens_q,   // int32, batch_size+1, starting offset of each sequence
        const void *cu_seqlens_k,   // int32, batch_size+1, starting offset of each sequence
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
        const bool is_causal,
        const bool is_bf16,
        const int num_splits,        // SMs per attention matrix, can be 1
        void *softmax_lse_ptr,       // softmax log_sum_exp
        void *softmax_ptr,
        void *workspace_ptr,
        uint64_t *workspace_size,
        cudaStream_t stream,
        uint64_t seed,
        uint64_t offset
);

bool flash_attn_bwd(
        const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        void *dq,                   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void *dk,                   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        void *dv,                   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *out,            // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *dout,           // total_q x num_heads, x head_size
        const void *cu_seqlens_q,   // int32, batch_size+1
        const void *cu_seqlens_k,   // int32, batch_size+1
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
        const bool is_causal,
        const bool is_bf16,
        const int num_splits,
        void *softmax_lse_ptr,
        void *dsoftmax_ptr,
        void *workspace_ptr,
        uint64_t *workspace_size,
        cudaStream_t stream,
        uint64_t seed,
        uint64_t offset
);

bool flash_attn_fwd_with_bias_and_mask(
        const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        void *out,                  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *cu_seqlens_q,   // int32, batch_size+1, starting offset of each sequence
        const void *cu_seqlens_k,   // int32, batch_size+1, starting offset of each sequence
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
        const bool is_causal,
        const bool is_bf16,
        const int num_splits,        // SMs per attention matrix, can be 1
        void *softmax_lse_ptr,       // softmax log_sum_exp
        void *softmax_ptr,
        void *workspace_ptr,
        uint64_t *workspace_size,
        cudaStream_t stream,
        uint64_t seed,
        uint64_t offset,
        const void *attn_mask,
        const void *attn_bias,
        const int64_t* mask_dims,
        const int64_t* bias_dims
);
,

bool flash_attn_bwd_with_bias_and_mask(
        const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        void *dq,                   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
        void *dk,                   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        void *dv,                   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *out,            // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
        const void *dout,           // total_q x num_heads, x head_size
        int32_t *cu_seqlens_q,   // int32, batch_size+1
        int32_t *cu_seqlens_k,   // int32, batch_size+1
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
        const bool is_causal,
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
        void* attn_mask,
        void* attn_bias,
        const int64_t* mask_dims,
        const int64_t* bias_dims
);

const char *flash_attn_error();

#ifdef __cplusplus
}
#endif
