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

#include "flash_attn_v3_utils.h"

#ifdef PADDLE_WITH_FLASHATTN_V3

void set_flashmaskv3_params_fprop(
    Flash_fwd_params *params,
    // sizes
    const size_t b, const size_t seqlen_q, const size_t seqlen_k,
    const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
    const size_t h, const size_t h_k, const size_t d, const size_t d_rounded,
    // device pointers
    const paddle::Tensor &q, const paddle::Tensor &k, const paddle::Tensor &v,
    const paddle::Tensor *out, void *cu_seqlens_q_d, void *cu_seqlens_k_d,
    void *seqused_q, void *seqused_k, void *softmax_lse_d, float p_dropout,
    float softmax_scale, int window_size_left, int window_size_right,
    const cudaDeviceProp &dprops, const float softcap, const int sm_margin) {
  params->is_bf16 = (q.dtype() == paddle::DataType::BFLOAT16);
  params->is_e4m3 = (q.dtype() == paddle::DataType::FLOAT8_E4M3FN);

  // Set the pointers and strides.
  params->q_ptr = const_cast<void *>(q.data());
  params->k_ptr = const_cast<void *>(k.data());
  params->v_ptr = const_cast<void *>(v.data());
  // All stride are in elements, not bytes.
  params->q_row_stride = q.strides()[q.strides().size() - 3];
  params->k_row_stride = k.strides()[k.strides().size() - 3];
  params->v_row_stride = v.strides()[v.strides().size() - 3];
  params->q_head_stride = q.strides()[q.strides().size() - 2];
  params->k_head_stride = k.strides()[k.strides().size() - 2];
  params->v_head_stride = v.strides()[v.strides().size() - 2];
  params->v_dim_stride = v.strides()[v.strides().size() - 1];
  params->o_ptr = const_cast<void *>(out->data());
  params->o_row_stride = out->strides()[out->strides().size() - 3];
  params->o_head_stride = out->strides()[out->strides().size() - 2];

  if (cu_seqlens_q_d == nullptr) {
    params->q_batch_stride = q.strides()[0];
    params->o_batch_stride = out->strides()[0];
  }
  if (cu_seqlens_k_d == nullptr) {
    params->k_batch_stride = k.strides()[0];
    params->v_batch_stride = v.strides()[0];
  }

  params->cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params->cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
  params->seqused_q = static_cast<int *>(seqused_q);
  params->seqused_k = static_cast<int *>(seqused_k);

  // Softmax sum
  params->softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params->b = b;
  params->h = h;
  params->h_k = h_k;
  params->seqlen_q = seqlen_q;
  params->seqlen_k = seqlen_k;
  params->seqlen_q_rounded = seqlen_q_rounded;
  params->seqlen_k_rounded = seqlen_k_rounded;
  params->d = d;
  params->d_rounded = d_rounded;

  // Set the different scale values.
  params->scale_softmax = softmax_scale;
  params->softcap = softcap;

  // Set this to probability of keeping an element to simplify things.
  params->p_dropout = 1.f - p_dropout;
  params->p_dropout_in_uint8_t =
      uint8_t(std::floor(params->p_dropout * 255.0));
  params->rp_dropout = 1.f / params->p_dropout;
  PADDLE_ENFORCE_LT(
      p_dropout, 1.f,
      common::errors::InvalidArgument("p_dropout must less than 1"));

  PADDLE_ENFORCE_EQ(
      p_dropout, 0.0f,
      common::errors::InvalidArgument(
          "This flash attention build does not support dropout."));

  // Causal is the special case where window_size_right == 0 and
  // window_size_left < 0. Local is the more general case where
  // window_size_right >= 0 or window_size_left >= 0.
  params->is_causal = (window_size_left < 0 && window_size_right == 0);
  params->is_local = ((window_size_left >= 0 || window_size_right >= 0) &&
                      !params->is_causal);

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  params->window_size_left = window_size_left;
  params->window_size_right = window_size_right;

  int arch = dprops.major * 10 + dprops.minor;
  int num_sm = dprops.multiProcessorCount - sm_margin;

  params->arch = arch;
  params->num_sm = num_sm;
}

void set_flashmaskv3_params_dgrad(
    Flash_bwd_params *params,
    // sizes
    const size_t b, const size_t seqlen_q, const size_t seqlen_k,
    const size_t seqlen_q_rounded, const size_t seqlen_k_rounded,
    const size_t h, const size_t h_k, const size_t d, const size_t d_rounded,
    // device pointers
    const paddle::Tensor &q, const paddle::Tensor &k, const paddle::Tensor &v,
    const paddle::Tensor &out, const paddle::Tensor &dout, paddle::Tensor *dq,
    paddle::Tensor *dk, paddle::Tensor *dv, void *cu_seqlens_q_d,
    void *cu_seqlens_k_d, void *seqused_q, void *seqused_k, void *dq_accum_d,
    void *dk_accum_d, void *dv_accum_d, void *softmax_lse_d,
    void *dsoftmax_sum_d, float p_dropout, float softmax_scale,
    int window_size_left, int window_size_right, const cudaDeviceProp &dprops,
    const float softcap, bool deterministic, int const sm_margin) {
  // Reuse fprop setup for the base Flash_fwd_params fields
  set_flashmaskv3_params_fprop(
      static_cast<Flash_fwd_params *>(params), b, seqlen_q,
      seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded, q, k,
      v, &out, cu_seqlens_q_d, cu_seqlens_k_d, seqused_q, seqused_k,
      softmax_lse_d, p_dropout, softmax_scale, window_size_left,
      window_size_right, dprops, softcap, sm_margin);

  // Set the pointers and strides.
  params->do_ptr = const_cast<void *>(dout.data());
  params->do_row_stride = dout.strides()[dout.strides().size() - 3];
  params->do_head_stride = dout.strides()[dout.strides().size() - 2];
  params->dq_ptr = dq->data();
  params->dk_ptr = dk->data();
  params->dv_ptr = dv->data();
  params->dq_row_stride = dq->strides()[dq->strides().size() - 3];
  params->dk_row_stride = dk->strides()[dk->strides().size() - 3];
  params->dv_row_stride = dv->strides()[dv->strides().size() - 3];
  params->dq_head_stride = dq->strides()[dq->strides().size() - 2];
  params->dk_head_stride = dk->strides()[dk->strides().size() - 2];
  params->dv_head_stride = dv->strides()[dv->strides().size() - 2];

  if (cu_seqlens_q_d == nullptr) {
    params->do_batch_stride = dout.strides()[0];
    params->dq_batch_stride = dq->strides()[0];
    params->dk_batch_stride = dk->strides()[0];
    params->dv_batch_stride = dv->strides()[0];
  }

  params->dq_accum_ptr = dq_accum_d;
  params->dk_accum_ptr = dk_accum_d;
  params->dv_accum_ptr = dv_accum_d;

  // Softmax sum
  params->dsoftmax_sum = dsoftmax_sum_d;

  params->deterministic = deterministic;
}
#endif
