/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
 /******************************************************************************
 * Copyright (c) 2024, PaddlePaddle.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod
#include "random_utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = uint32_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;
    float unscale_softmax; // for dense mask unscale not inf value
    uint32_t scale_softmax_log2_half2;

    ////////// start 在 mha_varlen_fwd 中使用 ////////// 
    bool varlen_padded_input = false;
    // array of length b+1 holding starting offset of each sequence.
    int total_q, total_k;
    
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    // If provided, the actual length of each q / o sequence.
    int * __restrict__ seqused_q;
    // If provided, the actual length of each k / v sequence.
    int * __restrict__ seqused_k;
    
    bool unpadded_lse; // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    ////////// end 在 mha_varlen_fwd 中使用 ////////// 

    // 暂无使用
    int *__restrict__ blockmask;


    ////////// start 在 mha_fwd_kvcache 中才会使用 //////////
    void * __restrict__ oaccum_ptr;
    // The stride between rows of Oaccum.
    index_t oaccum_batch_stride;
    index_t oaccum_row_stride;
    index_t oaccum_head_stride;
    index_t oaccum_split_stride;
    
    void * __restrict__ softmax_lseaccum_ptr;
    
    int b_k, seqlen_knew, rotary_dim;
    
    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    
    int num_splits;  // For split-KV version
    
    bool is_rotary_interleaved;
    
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).
    int * __restrict__ tile_count_semaphore;
    ////////// end 在 mha_fwd_kvcache 中才会使用 //////////

    ////////// start 在 dropout 中才会使用 //////////
    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;
    ////////// end 在 dropout 中才会使用 //////////

    // Local window size
    int window_size_left, window_size_right;

    // Random state.
    at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_e4m3;
    bool is_causal;
    bool is_local;
    bool is_kv_cache;
    bool use_gqa_packing;

    ////////// start 支持 alibi 才会使用 //////////
    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;
    ////////// end  支持 alibi //////////
    
    ////////// start FP8 类型才会使用 ////////// 
    float * __restrict__ descale_q_ptr;
    float * __restrict__ descale_k_ptr;
    float * __restrict__ descale_v_ptr;
    ////////// end FP8 类型才会使用 ////////// 
    
    
    ////////// start dense mask 才会使用 ////////// 
    void * __restrict__ attn_mask_ptr;
    int mask_head_mod_size;
    int mask_seq_q_mod_size;
    ////////// end dense mask 才会使用 ////////// 

    ////////// start flash mask 才会使用 ////////// 
    void * __restrict__ flashmask_downstart_ptr = nullptr;
    void * __restrict__ flashmask_upend_ptr = nullptr;
    void * __restrict__ flashmask_downend_ptr = nullptr;
    void * __restrict__ flashmask_upstart_ptr = nullptr;
    int *__restrict__ flashmask_maxmin_ptr = nullptr;
    int *__restrict__ flashmask_upend_nblockmax = nullptr;
    int *__restrict__ flashmask_upend_nblockmin = nullptr;
    int *__restrict__ flashmask_downstart_nblockmax = nullptr;
    int *__restrict__ flashmask_downstart_nblockmin = nullptr;
    int *__restrict__ flashmask_downend_nblockmax = nullptr;
    int *__restrict__ flashmask_downend_nblockmin = nullptr;
    int *__restrict__ flashmask_upstart_nblockmax = nullptr;
    int *__restrict__ flashmask_upstart_nblockmin = nullptr;
    int h_flashmask;
    int h_h_flashmask_ratio;
    bool enable_mask_bypass;
    ////////// end flash mask 才会使用 ////////// 
};

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;
    void *__restrict__ softmax_lse_log2_ptr;

    int *__restrict__ dq_semaphore;
    
    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, int kBlockH> void run_mha_fwd_gqa_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
