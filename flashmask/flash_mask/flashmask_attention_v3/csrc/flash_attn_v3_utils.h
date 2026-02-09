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

#pragma once

#define PADDLE_WITH_FLASHATTN_V3
#ifdef PADDLE_WITH_FLASHATTN_V3

// [BQW_CHANGE] 直接包含 flash_api.h，不使用 Paddle 内部 dynload 机制
// 因为我们直接链接 libflashmaskv2.so，所有 flashmaskv2_* 函数符号直接可用
// 移除了: #include "flashmaskv2.h"  (Paddle内部dynload封装)
// 移除了: #include "paddle/phi/core/dense_tensor.h"
// 移除了: #include "paddle/phi/core/platform/device_context.h"
#include "../flash_api.h"
#include "paddle/extension.h"
#include <cuda_runtime.h>

// [BQW_CHANGE] 移除了 fa3 (FlashAttention V3 非flashmask) 相关的全部函数声明
// 只保留 flashmask 相关的函数

FlashMask_fwd_params *get_flashmask_fwd_params_handle();

FlashMask_bwd_params *get_flashmask_bwd_params_handle();

inline int flashmaskv2_get_max_headdim() { return 256; }

inline int flashmaskv2_round_up_headdim(int head_size) {
#ifndef FLASHATTENTION_DISABLE_HDIM64
  if (head_size <= 64) {
    return 64;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  if (head_size <= 128) {
    return 128;
  }
#endif
  return 256;
}

void set_flashmaskv2_params_fprop(FlashMask_fwd_params *params_handle,
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
                                  const float softcap = 0.f,
                                  const int sm_margin = 0);

void set_flashmaskv2_params_dgrad(FlashMask_bwd_params *params_handle,
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
                                  const float softcap = 0.f,
                                  bool deterministic = false,
                                  int const sm_margin = 0);
#endif
