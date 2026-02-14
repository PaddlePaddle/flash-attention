/******************************************************************************
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

#include "paddle/extension.h"

#ifdef PADDLE_WITH_FLASHATTN_V3

#define CHECK_DEVICE(x) PD_CHECK(x.is_gpu(), #x " must be on CUDA Device")

#define CHECK_SHAPE(x, ...)                                                    \
  PADDLE_ENFORCE_EQ(x.dims(), common::make_ddim({__VA_ARGS__}),                \
                    common::errors::InvalidArgument(                           \
                        #x " must have shape (" #__VA_ARGS__ ")"))

#define CHECK_CONTIGUOUS(x)                                                    \
  PADDLE_ENFORCE_EQ(x.is_contiguous(), true,                                   \
                    common::errors::InvalidArgument(#x " must be contiguous"))
#endif

template <typename T>
void FlashMaskV3BaseKernel(
    const paddle::Tensor &q, const paddle::Tensor &k, const paddle::Tensor &v,
    const paddle::optional<paddle::Tensor>
        &k_new_, // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is
                 // cu_seqlens_k_new
    const paddle::optional<paddle::Tensor>
        &v_new_, // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is
                 // cu_seqlens_k_new
    const paddle::optional<paddle::Tensor>
        &q_v_, // (b, s_q, h, dv) or (total_q_new, h,
               // dv) if there is cu_seqlens_q
    const paddle::optional<paddle::Tensor>
        &out_, // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    const paddle::optional<paddle::Tensor> &cu_seqlens_q_,     // b+1
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_,     // b+1
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_new_, // b+1
    const paddle::optional<paddle::Tensor>
        &seqused_q_, // b. If given, only this many elements of each batch
                     // element's queries and outputs are used.
    const paddle::optional<paddle::Tensor>
        &seqused_k_, // b. If given, only this many elements of each batch
                     // element's keys are used.
    const paddle::optional<paddle::Tensor>
        &page_table_, // (b_k, max_num_pages_per_seq)
    const paddle::optional<paddle::Tensor>
        &kv_batch_idx_, // b. indices to index into the KV cache
    const paddle::optional<paddle::Tensor> &leftpad_k_, // b
    const paddle::optional<paddle::Tensor>
        &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
    const paddle::optional<paddle::Tensor>
        &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
    const paddle::optional<paddle::Tensor> &q_descale_, // (b, h_k), not (b, h)
    const paddle::optional<paddle::Tensor> &k_descale_, // (b, h_k)
    const paddle::optional<paddle::Tensor> &v_descale_, // (b, h_k)
    const paddle::optional<paddle::Tensor> &scheduler_metadata_, // (b + 1)
    const paddle::optional<paddle::Tensor>
        &startend_row_indices_, // （b,h,s_1,[1,2,4])
    const paddle::optional<paddle::Tensor>
        &block_mask_,        // （(b,h,s// 128,s // 128)
    const int max_seqlen_q_, // if max_seqlen_q_ is set to 0, it indicates that
                             // it is uninitialized and should not be referenced
    // TODO(tridao): check if we need max_seqlen_k
    const int max_seqlen_k_, // if max_seqlen_q_ is set to 0, it indicates that
                             // it is uninitialized and should not be referenced
    const float softmax_scale, bool is_causal, int window_size_left,
    int window_size_right, const float softcap,
    const bool is_rotary_interleaved, // if true, rotary combines indices 0 &
                                      // 1, else indices 0 & rotary_dim / 2
    int num_splits, const bool manual_set_pack_gqa,
    const bool
        pack_gqa_, // the pack_gqa_ will be used only if manual_set_pack_gqa is
                   // set to True; otherwise, the internal heuristic
                   // get_pack_gqa() from fa3 will decide whether to pack gqa
    const int sm_margin, paddle::Tensor *out, paddle::Tensor *softmax_lse,
    paddle::Tensor *out_accum, paddle::Tensor *softmax_lse_accum);

template <typename T>
void FlashMaskV3GradBaseKernel(
    const paddle::Tensor
        &dout, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const paddle::Tensor
        &q, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const paddle::Tensor
        &k, // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const paddle::Tensor
        &v, // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const paddle::Tensor
        &out, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const paddle::Tensor
        &softmax_lse, // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
    const paddle::optional<paddle::Tensor>
        &dq_, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const paddle::optional<paddle::Tensor>
        &dk_, // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const paddle::optional<paddle::Tensor>
        &dv_, // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
    const paddle::optional<paddle::Tensor> &cu_seqlens_q_, // b+1
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_, // b+1
    const paddle::optional<paddle::Tensor>
        &seqused_q_, // b. If given, only this many elements of each batch
                     // element's queries and outputs are used.
    const paddle::optional<paddle::Tensor>
        &seqused_k_, // b. If given, only this many elements of each batch
                     // element's keys are used.
    const paddle::optional<paddle::Tensor> &startend_row_indices_,
    const paddle::optional<paddle::Tensor>
        &block_mask_, // （(b,h,s//128,s//128)
    int max_seqlen_q_, int max_seqlen_k_, float const softmax_scale,
    bool is_causal, int window_size_left, int window_size_right,
    float const softcap, bool const deterministic, int const sm_margin,
    paddle::Tensor *dq, paddle::Tensor *dk, paddle::Tensor *dv,
    paddle::Tensor *softmax_d, paddle::Tensor *softmax_lse_log2,
    paddle::Tensor *dq_accum, paddle::Tensor *dk_accum,
    paddle::Tensor *dv_accum);
