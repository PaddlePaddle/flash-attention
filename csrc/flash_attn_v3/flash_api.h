#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

bool flash_attn_v3_fwd(const void * const q,         // batch_size x seqlen_q x num_heads x head_size
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
                    const void * const flashmask_downend_ptr,
                    const void * const flashmask_upend_ptr,
                    const void * const flashmask_upstart_ptr,
                    const void * const flashmask_maxmin_ptr,
                    const int64_t * const flashmask_dims,
                    const int q_batch_stride,
                    const int k_batch_stride,
                    const int v_batch_stride,
                    const int q_row_stride,
                    const int k_row_stride,
                    const int v_row_stride,
                    const int q_head_stride,
                    const int k_head_stride,
                    const int v_head_stride,
                    const int o_batch_stride,
                    const int o_row_stride,
                    const int o_head_stride,
                    const bool is_e4m3,
                    void * tile_count_semaphore,
                    const float * const descale_q_ptr,
                    const float * const descale_k_ptr,
                    const float * const descale_v_ptr,
                    const bool use_gqa_packing);

bool flash_attn_v3_bwd(const void * const dout,  // batch_size x seqlen_q x num_heads, x head_size_og
                    const void * const q,   // batch_size x seqlen_q x num_heads x head_size
                    const void * const k,   // batch_size x seqlen_k x num_heads_k x head_size
                    const void * const v,   // batch_size x seqlen_k x num_heads_k x head_size
                    const void * const out,   // batch_size x seqlen_q x num_heads x head_size
                    const void * const softmax_d,
                    const void * const softmax_lse,     // b x h x seqlen_q
                    const void * const softmax_lse_log2,
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
                    const bool deterministic,
                    cudaStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims,
                    const void * const flashmask_downstart_ptr,
                    const void * const flashmask_downend_ptr,
                    const void * const flashmask_upend_ptr,
                    const void * const flashmask_upstart_ptr,
                    const void * const flashmask_maxmin_ptr,
                    const int64_t * const flashmask_dims,
                    const int q_batch_stride,
                    const int k_batch_stride,
                    const int v_batch_stride,
                    const int q_row_stride,
                    const int k_row_stride,
                    const int v_row_stride,
                    const int q_head_stride,
                    const int k_head_stride,
                    const int v_head_stride,
                    const int o_batch_stride,
                    const int o_row_stride,
                    const int o_head_stride,
                    const int dq_batch_stride,
                    const int dk_batch_stride,
                    const int dv_batch_stride,
                    const int dq_row_stride,
                    const int dk_row_stride,
                    const int dv_row_stride,
                    const int dq_head_stride,
                    const int dk_head_stride,
                    const int dv_head_stride,
                    const int do_batch_stride,
                    const int do_row_stride,
                    const int do_head_stride,
                    void * dq_semaphore);

void flash_attn_set_error(const char *msg);

const char *flash_attn_error();

#ifdef __cplusplus
}
#endif
