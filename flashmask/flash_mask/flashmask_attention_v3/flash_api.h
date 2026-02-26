/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 *
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Flash_fwd_params FlashMask_fwd_params;
typedef struct Flash_bwd_params FlashMask_bwd_params;

Flash_fwd_params* flashmaskv3_create_fwd_params_handle();
Flash_bwd_params* flashmaskv3_create_bwd_params_handle();
void flashmaskv3_clear_fwd_params_handle(Flash_fwd_params* params_handle);
void flashmaskv3_clear_bwd_params_handle(Flash_bwd_params* params_handle);
Flash_fwd_params* flashmaskv3_cast_to_fwd_params_handle(Flash_bwd_params* params_handle);
void flashmaskv3_destroy_fwd_params_handle(Flash_fwd_params* params_handle);
void flashmaskv3_destroy_bwd_params_handle(Flash_bwd_params* params_handle);
void flashmaskv3_run_mha_fwd_combine(Flash_fwd_params* params_handle, cudaStream_t stream, bool enable_pdl=false);
void flashmaskv3_run_mha_fwd(Flash_fwd_params* params_handle, cudaStream_t stream);
bool flashmaskv3_get_pagedkv_tma(Flash_fwd_params* params_handle);
bool flashmaskv3_get_pack_gqa(Flash_fwd_params* params_handle);
int flashmaskv3_get_num_splits(Flash_fwd_params* params_handle);
void flashmaskv3_run_mha_bwd(Flash_bwd_params* params_handle, cudaStream_t stream);

#define DECLARE_GETTER_SETTER(type, member) \
type flashmaskv3_fwd_params_get_##member(const Flash_fwd_params* params_handle); \
void flashmaskv3_fwd_params_set_##member(Flash_fwd_params* params_handle, const type value); \
type flashmaskv3_bwd_params_get_##member(const Flash_bwd_params* params_handle); \
void flashmaskv3_bwd_params_set_##member(Flash_bwd_params* params_handle, type value);

// The QKV matrices.
DECLARE_GETTER_SETTER(void *, q_ptr)
DECLARE_GETTER_SETTER(void *, k_ptr)
DECLARE_GETTER_SETTER(void *, v_ptr)

// The stride between rows of the Q, K and V matrices.
DECLARE_GETTER_SETTER(int64_t, q_batch_stride)
DECLARE_GETTER_SETTER(int64_t, k_batch_stride)
DECLARE_GETTER_SETTER(int64_t, v_batch_stride)
DECLARE_GETTER_SETTER(int64_t, q_row_stride)
DECLARE_GETTER_SETTER(int64_t, k_row_stride)
DECLARE_GETTER_SETTER(int64_t, v_row_stride)
DECLARE_GETTER_SETTER(int64_t, q_head_stride)
DECLARE_GETTER_SETTER(int64_t, k_head_stride)
DECLARE_GETTER_SETTER(int64_t, v_head_stride)
DECLARE_GETTER_SETTER(int64_t, v_dim_stride)

// The number of heads.
DECLARE_GETTER_SETTER(int, h)
DECLARE_GETTER_SETTER(int, h_k)

// The O matrix (output).
DECLARE_GETTER_SETTER(void *, o_ptr)
DECLARE_GETTER_SETTER(void *, oaccum_ptr)

// The stride between rows of O.
DECLARE_GETTER_SETTER(int64_t, o_batch_stride)
DECLARE_GETTER_SETTER(int64_t, o_row_stride)
DECLARE_GETTER_SETTER(int64_t, o_head_stride)

// The pointer to the softmax sum.
DECLARE_GETTER_SETTER(void *, softmax_lse_ptr)
DECLARE_GETTER_SETTER(void *, softmax_lseaccum_ptr)

// For FP8 scaling
DECLARE_GETTER_SETTER(float *, q_descale_ptr)
DECLARE_GETTER_SETTER(float *, k_descale_ptr)
DECLARE_GETTER_SETTER(float *, v_descale_ptr)
DECLARE_GETTER_SETTER(int64_t, q_descale_batch_stride)
DECLARE_GETTER_SETTER(int64_t, q_descale_head_stride)
DECLARE_GETTER_SETTER(int64_t, k_descale_batch_stride)
DECLARE_GETTER_SETTER(int64_t, k_descale_head_stride)
DECLARE_GETTER_SETTER(int64_t, v_descale_batch_stride)
DECLARE_GETTER_SETTER(int64_t, v_descale_head_stride)

// The dimensions.
DECLARE_GETTER_SETTER(int, b)
DECLARE_GETTER_SETTER(int, seqlen_q)
DECLARE_GETTER_SETTER(int, seqlen_k)
DECLARE_GETTER_SETTER(int, seqlen_knew)
DECLARE_GETTER_SETTER(int, d)
DECLARE_GETTER_SETTER(int, seqlen_q_rounded)
DECLARE_GETTER_SETTER(int, seqlen_k_rounded)
DECLARE_GETTER_SETTER(int, d_rounded)
DECLARE_GETTER_SETTER(int, rotary_dim)
DECLARE_GETTER_SETTER(int, total_q)
DECLARE_GETTER_SETTER(int, total_k)
DECLARE_GETTER_SETTER(int, total_knew)
DECLARE_GETTER_SETTER(int, b_k)
DECLARE_GETTER_SETTER(int, dv)
DECLARE_GETTER_SETTER(int, dv_rounded)

// The scaling factors for the kernel.
DECLARE_GETTER_SETTER(float, scale_softmax)
DECLARE_GETTER_SETTER(float, softcap)

// array of length b+1 holding starting offset of each sequence.
DECLARE_GETTER_SETTER(int *, cu_seqlens_q)
DECLARE_GETTER_SETTER(int *, cu_seqlens_k)
DECLARE_GETTER_SETTER(int *, cu_seqlens_knew)
DECLARE_GETTER_SETTER(int *, leftpad_k)

// If provided, the actual length of each q/k sequence.
DECLARE_GETTER_SETTER(int *, seqused_q)
DECLARE_GETTER_SETTER(int *, seqused_k)

// The stride between rows of Oaccum.
DECLARE_GETTER_SETTER(int64_t, oaccum_split_stride)
DECLARE_GETTER_SETTER(int64_t, oaccum_batch_stride)
DECLARE_GETTER_SETTER(int64_t, oaccum_row_stride)
DECLARE_GETTER_SETTER(int64_t, oaccum_head_stride)

// The stride between rows of LSEaccum.
DECLARE_GETTER_SETTER(int64_t, lseaccum_split_stride)
DECLARE_GETTER_SETTER(int64_t, lseaccum_batch_stride)
DECLARE_GETTER_SETTER(int64_t, lseaccum_head_stride)

// The K_new and V_new matrices.
DECLARE_GETTER_SETTER(void *, knew_ptr)
DECLARE_GETTER_SETTER(void *, vnew_ptr)

// The stride between rows of the Q, K and V matrices.
DECLARE_GETTER_SETTER(int64_t, knew_batch_stride)
DECLARE_GETTER_SETTER(int64_t, vnew_batch_stride)
DECLARE_GETTER_SETTER(int64_t, knew_row_stride)
DECLARE_GETTER_SETTER(int64_t, vnew_row_stride)
DECLARE_GETTER_SETTER(int64_t, knew_head_stride)
DECLARE_GETTER_SETTER(int64_t, vnew_head_stride)

DECLARE_GETTER_SETTER(void *, qv_ptr)
DECLARE_GETTER_SETTER(int64_t, qv_batch_stride)
DECLARE_GETTER_SETTER(int64_t, qv_row_stride)
DECLARE_GETTER_SETTER(int64_t, qv_head_stride)

// The cos and sin matrices for rotary embedding.
DECLARE_GETTER_SETTER(void *, rotary_cos_ptr)
DECLARE_GETTER_SETTER(void *, rotary_sin_ptr)

// The indices to index into the KV cache.
DECLARE_GETTER_SETTER(int *, kv_batch_idx)

// Paged KV cache
DECLARE_GETTER_SETTER(int *, page_table)
DECLARE_GETTER_SETTER(int64_t, page_table_batch_stride)
DECLARE_GETTER_SETTER(int, page_size)
DECLARE_GETTER_SETTER(int, num_pages)
DECLARE_GETTER_SETTER(bool, pagedkv_tma)

// The dropout probability (probability of keeping an activation).
DECLARE_GETTER_SETTER(float, p_dropout)
// uint32_t p_dropout_in_uint;
// uint16_t p_dropout_in_uint16_t;
DECLARE_GETTER_SETTER(uint8_t, p_dropout_in_uint8_t)

// Scale factor of 1 / (1 - p_dropout).
DECLARE_GETTER_SETTER(float, rp_dropout)

// Local window size
DECLARE_GETTER_SETTER(int, window_size_left)
DECLARE_GETTER_SETTER(int, window_size_right)

// Pointer to the RNG seed (idx 0) and offset (idx 1).
DECLARE_GETTER_SETTER(uint64_t *, rng_state)

DECLARE_GETTER_SETTER(bool, is_bf16)
DECLARE_GETTER_SETTER(bool, is_fp32)
DECLARE_GETTER_SETTER(bool, is_e4m3)
DECLARE_GETTER_SETTER(bool, is_causal)
DECLARE_GETTER_SETTER(bool, is_local)

DECLARE_GETTER_SETTER(bool, is_rotary_interleaved)

DECLARE_GETTER_SETTER(int, num_splits)  // For split-KV version
DECLARE_GETTER_SETTER(bool, pack_gqa)

DECLARE_GETTER_SETTER(int, num_splits)  // For split-KV version
DECLARE_GETTER_SETTER(bool, pack_gqa)

DECLARE_GETTER_SETTER(int *, tile_count_semaphore)
// int * __restrict__ num_m_blocks_ptr;
// int * __restrict__ num_n_blocks_ptr;
DECLARE_GETTER_SETTER(int *, num_splits_dynamic_ptr)
DECLARE_GETTER_SETTER(bool, skip_scheduler_metadata_computation)

DECLARE_GETTER_SETTER(int, arch)
DECLARE_GETTER_SETTER(int, num_sm)

DECLARE_GETTER_SETTER(int, h_flashmask)
DECLARE_GETTER_SETTER(int, h_h_flashmask_ratio)

DECLARE_GETTER_SETTER(int32_t *, lt_start_ptr)
DECLARE_GETTER_SETTER(int32_t *, lt_end_ptr)

DECLARE_GETTER_SETTER(int32_t *, ut_start_ptr)
DECLARE_GETTER_SETTER(int32_t *, ut_end_ptr)

DECLARE_GETTER_SETTER(int32_t *, flashmask_maxmin_ptr)

DECLARE_GETTER_SETTER(int, m_block_dim)
DECLARE_GETTER_SETTER(int, n_block_dim)
DECLARE_GETTER_SETTER(int32_t *, block_mask_ptr)

#define DECLARE_BWD_GETTER_SETTER(type, member) \
type flashmaskv3_bwd_params_get_##member(const Flash_bwd_params* params_handle); \
void flashmaskv3_bwd_params_set_##member(Flash_bwd_params* params_handle, type value);

// The dO and dQKV matrices.
DECLARE_BWD_GETTER_SETTER(void *, do_ptr)
DECLARE_BWD_GETTER_SETTER(void *, dq_ptr)
DECLARE_BWD_GETTER_SETTER(void *, dk_ptr)
DECLARE_BWD_GETTER_SETTER(void *, dv_ptr)

// To accumulate dQ
DECLARE_BWD_GETTER_SETTER(void *, dq_accum_ptr)
DECLARE_BWD_GETTER_SETTER(void *, dk_accum_ptr)
DECLARE_BWD_GETTER_SETTER(void *, dv_accum_ptr)

// // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
// dimension void *__restrict__ dk_accum_ptr; void *__restrict__
// dv_accum_ptr;

// The stride between rows of the dO, dQ, dK and dV matrices.
DECLARE_BWD_GETTER_SETTER(int64_t, do_batch_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, do_row_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, do_head_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dq_batch_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dk_batch_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dv_batch_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dq_row_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dk_row_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dv_row_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dq_head_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dk_head_stride)
DECLARE_BWD_GETTER_SETTER(int64_t, dv_head_stride)

// The pointer to the softmax d sum.
DECLARE_BWD_GETTER_SETTER(void *, dsoftmax_sum)
DECLARE_BWD_GETTER_SETTER(void *, softmax_lse_log2_ptr)

DECLARE_BWD_GETTER_SETTER(int *, dq_semaphore)
DECLARE_BWD_GETTER_SETTER(int *, dk_semaphore)
DECLARE_BWD_GETTER_SETTER(int *, dv_semaphore)

DECLARE_BWD_GETTER_SETTER(bool, deterministic)
DECLARE_BWD_GETTER_SETTER(int64_t, dq_accum_split_stride)
DECLARE_BWD_GETTER_SETTER(int, h_flashmask)
DECLARE_BWD_GETTER_SETTER(int, h_h_flashmask_ratio)

DECLARE_BWD_GETTER_SETTER(int32_t *, lt_start_ptr)
DECLARE_BWD_GETTER_SETTER(int32_t *, lt_end_ptr)

DECLARE_BWD_GETTER_SETTER(int32_t *, ut_start_ptr)
DECLARE_BWD_GETTER_SETTER(int32_t *, ut_end_ptr)

DECLARE_BWD_GETTER_SETTER(int32_t *, flashmask_maxmin_ptr)

DECLARE_BWD_GETTER_SETTER(int32_t *, lt_start_nblockmax)
DECLARE_BWD_GETTER_SETTER(int32_t *, lt_start_nblockmin)

DECLARE_BWD_GETTER_SETTER(int32_t *, lt_end_nblockmax)
DECLARE_BWD_GETTER_SETTER(int32_t *, lt_end_nblockmin)

DECLARE_BWD_GETTER_SETTER(int32_t *, ut_start_nblockmax)
DECLARE_BWD_GETTER_SETTER(int32_t *, ut_start_nblockmin)

DECLARE_BWD_GETTER_SETTER(int32_t *, ut_end_nblockmax)
DECLARE_BWD_GETTER_SETTER(int32_t *, ut_end_nblockmin)
#ifdef __cplusplus
}
#endif
