// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flash_attn_v3_utils.h"
#include <memory>

// [BQW_CHANGE] 移除了 namespace phi { 包裹 (外迁后不在 phi 命名空间)
// [BQW_CHANGE] 移除了所有 fa3_* (FlashAttention V3 非flashmask) 相关函数
// [BQW_CHANGE] 所有 dynload::flashmaskv2_* 和 phi::dynload::flashmaskv2_*
//              替换为直接调用 flashmaskv2_* (因为直接链接了 libflashmaskv2.so)
// [BQW_CHANGE] 所有 phi::DataType:: 替换为 paddle::DataType::

#ifdef PADDLE_WITH_FLASHATTN_V3

static void destroy_flashmask_fwd_params_handle(Flash_fwd_params *params_handle) {
  flashmaskv2_destroy_fwd_params_handle(params_handle);
}

static void destroy_flashmask_bwd_params_handle(Flash_bwd_params *params_handle) {
  flashmaskv2_destroy_bwd_params_handle(params_handle);
}

FlashMask_fwd_params *get_flashmask_fwd_params_handle() {
  static std::unique_ptr<Flash_fwd_params,
                         decltype(&destroy_flashmask_fwd_params_handle)>
      params_handle(flashmaskv2_create_fwd_params_handle(),
                    &destroy_flashmask_fwd_params_handle);
  return params_handle.get();
}

FlashMask_bwd_params *get_flashmask_bwd_params_handle() {
  static std::unique_ptr<Flash_bwd_params,
                         decltype(&destroy_flashmask_bwd_params_handle)>
      params_handle(flashmaskv2_create_bwd_params_handle(),
                    &destroy_flashmask_bwd_params_handle);
  return params_handle.get();
}

void set_flashmaskv2_params_fprop(Flash_fwd_params *params_handle,
                                  // sizes
                                  const size_t b,
                                  const size_t seqlen_q,
                                  const size_t seqlen_k,
                                  const size_t seqlen_q_rounded,
                                  const size_t seqlen_k_rounded,
                                  const size_t h,
                                  const size_t h_k,
                                  const size_t d,
                                  const size_t d_rounded,
                                  // device pointers
                                  const paddle::Tensor &q,
                                  const paddle::Tensor &k,
                                  const paddle::Tensor &v,
                                  const paddle::Tensor *out,
                                  void *cu_seqlens_q_d,
                                  void *cu_seqlens_k_d,
                                  void *seqused_q,
                                  void *seqused_k,
                                  void *softmax_lse_d,
                                  float p_dropout,
                                  float softmax_scale,
                                  int window_size_left,
                                  int window_size_right,
                                  const cudaDeviceProp &dprops,
                                  const float softcap,
                                  const int sm_margin) {
  flashmaskv2_fwd_params_set_is_bf16(
      params_handle, q.dtype() == paddle::DataType::BFLOAT16);
  flashmaskv2_fwd_params_set_is_e4m3(
      params_handle, q.dtype() == paddle::DataType::FLOAT8_E4M3FN);

  // Set the pointers and strides.
  flashmaskv2_fwd_params_set_q_ptr(params_handle,
                                   const_cast<void *>(q.data()));
  flashmaskv2_fwd_params_set_k_ptr(params_handle,
                                   const_cast<void *>(k.data()));
  flashmaskv2_fwd_params_set_v_ptr(params_handle,
                                   const_cast<void *>(v.data()));
  // All stride are in elements, not bytes.
  flashmaskv2_fwd_params_set_q_row_stride(
      params_handle, q.strides()[q.strides().size() - 3]);
  flashmaskv2_fwd_params_set_k_row_stride(
      params_handle, k.strides()[k.strides().size() - 3]);
  flashmaskv2_fwd_params_set_v_row_stride(
      params_handle, v.strides()[v.strides().size() - 3]);
  flashmaskv2_fwd_params_set_q_head_stride(
      params_handle, q.strides()[q.strides().size() - 2]);
  flashmaskv2_fwd_params_set_k_head_stride(
      params_handle, k.strides()[k.strides().size() - 2]);
  flashmaskv2_fwd_params_set_v_head_stride(
      params_handle, v.strides()[v.strides().size() - 2]);
  flashmaskv2_fwd_params_set_v_dim_stride(
      params_handle, v.strides()[v.strides().size() - 1]);
  flashmaskv2_fwd_params_set_o_ptr(params_handle,
                                   const_cast<void *>(out->data()));
  flashmaskv2_fwd_params_set_o_row_stride(
      params_handle, out->strides()[out->strides().size() - 3]);
  flashmaskv2_fwd_params_set_o_head_stride(
      params_handle, out->strides()[out->strides().size() - 2]);

  if (cu_seqlens_q_d == nullptr) {
    flashmaskv2_fwd_params_set_q_batch_stride(params_handle,
                                              q.strides()[0]);
    flashmaskv2_fwd_params_set_o_batch_stride(params_handle,
                                              out->strides()[0]);
  }
  if (cu_seqlens_k_d == nullptr) {
    flashmaskv2_fwd_params_set_k_batch_stride(params_handle,
                                              k.strides()[0]);
    flashmaskv2_fwd_params_set_v_batch_stride(params_handle,
                                              v.strides()[0]);
  }

  flashmaskv2_fwd_params_set_cu_seqlens_q(
      params_handle, static_cast<int *>(cu_seqlens_q_d));
  flashmaskv2_fwd_params_set_cu_seqlens_k(
      params_handle, static_cast<int *>(cu_seqlens_k_d));
  flashmaskv2_fwd_params_set_seqused_q(params_handle,
                                       static_cast<int *>(seqused_q));
  flashmaskv2_fwd_params_set_seqused_k(params_handle,
                                       static_cast<int *>(seqused_k));

  // Softmax sum
  flashmaskv2_fwd_params_set_softmax_lse_ptr(params_handle,
                                             softmax_lse_d);

  // Set the dimensions.
  flashmaskv2_fwd_params_set_b(params_handle, b);
  flashmaskv2_fwd_params_set_h(params_handle, h);
  flashmaskv2_fwd_params_set_h_k(params_handle, h_k);
  flashmaskv2_fwd_params_set_seqlen_q(params_handle, seqlen_q);
  flashmaskv2_fwd_params_set_seqlen_k(params_handle, seqlen_k);
  flashmaskv2_fwd_params_set_seqlen_q_rounded(params_handle,
                                              seqlen_q_rounded);
  flashmaskv2_fwd_params_set_seqlen_k_rounded(params_handle,
                                              seqlen_k_rounded);
  flashmaskv2_fwd_params_set_d(params_handle, d);
  flashmaskv2_fwd_params_set_d_rounded(params_handle, d_rounded);

  // Set the different scale values.
  flashmaskv2_fwd_params_set_scale_softmax(params_handle,
                                           softmax_scale);
  flashmaskv2_fwd_params_set_softcap(params_handle, softcap);

  // Set this to probability of keeping an element to simplify things.
  flashmaskv2_fwd_params_set_p_dropout(params_handle, 1.f - p_dropout);
  flashmaskv2_fwd_params_set_p_dropout_in_uint8_t(
      params_handle,
      uint8_t(std::floor(
          flashmaskv2_fwd_params_get_p_dropout(params_handle) *
          255.0)));
  flashmaskv2_fwd_params_set_rp_dropout(
      params_handle,
      1.f / flashmaskv2_fwd_params_get_p_dropout(params_handle));
  PADDLE_ENFORCE_LT(
      p_dropout,
      1.f,
      common::errors::InvalidArgument("p_dropout must less than 1"));

  PADDLE_ENFORCE_EQ(
      p_dropout,
      0.0f,
      common::errors::InvalidArgument(
          "This flash attention build does not support dropout."));

  // Causal is the special case where window_size_right == 0 and
  // window_size_left < 0. Local is the more general case where
  // window_size_right >= 0 or window_size_left >= 0.
  flashmaskv2_fwd_params_set_is_causal(
      params_handle, window_size_left < 0 && window_size_right == 0);
  flashmaskv2_fwd_params_set_is_local(
      params_handle,
      (window_size_left >= 0 || window_size_right >= 0) &&
          !flashmaskv2_fwd_params_get_is_causal(params_handle));

  if (window_size_left < 0 && window_size_right >= 0) {
    window_size_left = seqlen_k - 1;
  }
  if (window_size_left >= 0 && window_size_right < 0) {
    window_size_right = seqlen_q - 1;
  }
  flashmaskv2_fwd_params_set_window_size_left(params_handle,
                                              window_size_left);
  flashmaskv2_fwd_params_set_window_size_right(params_handle,
                                               window_size_right);

  int arch = dprops.major * 10 + dprops.minor;
  int num_sm = dprops.multiProcessorCount - sm_margin;

  flashmaskv2_fwd_params_set_arch(params_handle, arch);
  flashmaskv2_fwd_params_set_num_sm(params_handle, num_sm);
}

void set_flashmaskv2_params_dgrad(Flash_bwd_params *params_handle,
                                  // sizes
                                  const size_t b,
                                  const size_t seqlen_q,
                                  const size_t seqlen_k,
                                  const size_t seqlen_q_rounded,
                                  const size_t seqlen_k_rounded,
                                  const size_t h,
                                  const size_t h_k,
                                  const size_t d,
                                  const size_t d_rounded,
                                  // device pointers
                                  const paddle::Tensor &q,
                                  const paddle::Tensor &k,
                                  const paddle::Tensor &v,
                                  const paddle::Tensor &out,
                                  const paddle::Tensor &dout,
                                  paddle::Tensor *dq,
                                  paddle::Tensor *dk,
                                  paddle::Tensor *dv,
                                  void *cu_seqlens_q_d,
                                  void *cu_seqlens_k_d,
                                  void *seqused_q,
                                  void *seqused_k,
                                  void *dq_accum_d,
                                  void *dk_accum_d,
                                  void *dv_accum_d,
                                  void *softmax_lse_d,
                                  void *dsoftmax_sum_d,
                                  float p_dropout,
                                  float softmax_scale,
                                  int window_size_left,
                                  int window_size_right,
                                  const cudaDeviceProp &dprops,
                                  const float softcap,
                                  bool deterministic,
                                  int const sm_margin) {
  set_flashmaskv2_params_fprop(
      flashmaskv2_cast_to_fwd_params_handle(params_handle),
      b,
      seqlen_q,
      seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      h,
      h_k,
      d,
      d_rounded,
      q,
      k,
      v,
      &out,
      cu_seqlens_q_d,
      cu_seqlens_k_d,
      seqused_q,
      seqused_k,
      softmax_lse_d,
      p_dropout,
      softmax_scale,
      window_size_left,
      window_size_right,
      dprops,
      softcap,
      sm_margin);

  // Set the pointers and strides.
  flashmaskv2_bwd_params_set_do_ptr(params_handle,
                                    const_cast<void *>(dout.data()));
  flashmaskv2_bwd_params_set_do_row_stride(
      params_handle, dout.strides()[dout.strides().size() - 3]);
  flashmaskv2_bwd_params_set_do_head_stride(
      params_handle, dout.strides()[dout.strides().size() - 2]);
  flashmaskv2_bwd_params_set_dq_ptr(params_handle, dq->data());
  flashmaskv2_bwd_params_set_dk_ptr(params_handle, dk->data());
  flashmaskv2_bwd_params_set_dv_ptr(params_handle, dv->data());
  flashmaskv2_bwd_params_set_dq_row_stride(
      params_handle, dq->strides()[dq->strides().size() - 3]);
  flashmaskv2_bwd_params_set_dk_row_stride(
      params_handle, dk->strides()[dk->strides().size() - 3]);
  flashmaskv2_bwd_params_set_dv_row_stride(
      params_handle, dv->strides()[dv->strides().size() - 3]);
  flashmaskv2_bwd_params_set_dq_head_stride(
      params_handle, dq->strides()[dq->strides().size() - 2]);
  flashmaskv2_bwd_params_set_dk_head_stride(
      params_handle, dk->strides()[dk->strides().size() - 2]);
  flashmaskv2_bwd_params_set_dv_head_stride(
      params_handle, dv->strides()[dv->strides().size() - 2]);

  if (cu_seqlens_q_d == nullptr) {
    flashmaskv2_bwd_params_set_do_batch_stride(params_handle,
                                               dout.strides()[0]);
    flashmaskv2_bwd_params_set_dq_batch_stride(params_handle,
                                               dq->strides()[0]);
    flashmaskv2_bwd_params_set_dk_batch_stride(params_handle,
                                               dk->strides()[0]);
    flashmaskv2_bwd_params_set_dv_batch_stride(params_handle,
                                               dv->strides()[0]);
  }

  flashmaskv2_bwd_params_set_dq_accum_ptr(params_handle, dq_accum_d);
  flashmaskv2_bwd_params_set_dk_accum_ptr(params_handle, dk_accum_d);
  flashmaskv2_bwd_params_set_dv_accum_ptr(params_handle, dv_accum_d);

  // Softmax sum
  flashmaskv2_bwd_params_set_dsoftmax_sum(params_handle,
                                          dsoftmax_sum_d);

  flashmaskv2_bwd_params_set_deterministic(params_handle,
                                           deterministic);
}
#endif
