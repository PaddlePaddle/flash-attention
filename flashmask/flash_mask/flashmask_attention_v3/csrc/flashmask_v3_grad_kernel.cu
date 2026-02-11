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
#include "flashmask_v3.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template <typename T>
void FlashMaskV3GradBaseKernel(
    const paddle::Tensor &dout, const paddle::Tensor &q,
    const paddle::Tensor &k, const paddle::Tensor &v, const paddle::Tensor &out,
    const paddle::Tensor &softmax_lse,
    const paddle::optional<paddle::Tensor> &dq_,
    const paddle::optional<paddle::Tensor> &dk_,
    const paddle::optional<paddle::Tensor> &dv_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_q_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_,
    const paddle::optional<paddle::Tensor> &seqused_q_,
    const paddle::optional<paddle::Tensor> &seqused_k_,
    const paddle::optional<paddle::Tensor> &startend_row_indices_,
    const paddle::optional<paddle::Tensor> &block_mask_, int max_seqlen_q_,
    int max_seqlen_k_, float const softmax_scale, bool is_causal,
    int window_size_left, int window_size_right, float const softcap,
    bool const deterministic, int const sm_margin, paddle::Tensor *dq,
    paddle::Tensor *dk, paddle::Tensor *dv, paddle::Tensor *softmax_d,
    paddle::Tensor *softmax_lse_log2, paddle::Tensor *dq_accum,
    paddle::Tensor *dk_accum, paddle::Tensor *dv_accum) {
#ifdef PADDLE_WITH_FLASHATTN_V3
  // TODO(umiswing): support ampere
  cudaStream_t stream = static_cast<cudaStream_t>(q.stream());
  auto place = q.place();

  int device_id = place.GetDeviceId();
  cudaDeviceProp dprops;
  cudaGetDeviceProperties(&dprops, device_id);

  const bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  PADDLE_ENFORCE_EQ(is_sm90, true,
                    common::errors::Unavailable(
                        "FlashAttention-3 only supports Hopper GPUs."));

  auto q_type = q.dtype();

  PADDLE_ENFORCE_EQ(
      (q_type == paddle::DataType::FLOAT16 ||
       q_type == paddle::DataType::BFLOAT16),
      true,
      common::errors::InvalidArgument(
          "FlashAttention-3 bwd only support fp16 and bf16 data type"));
  PADDLE_ENFORCE_EQ(k.dtype(), q_type,
                    common::errors::InvalidArgument(
                        "query and key must have the same dtype"));
  PADDLE_ENFORCE_EQ(v.dtype(), q_type,
                    common::errors::InvalidArgument(
                        "query and value must have the same dtype"));
  PADDLE_ENFORCE_EQ(out.dtype(), q_type,
                    common::errors::InvalidArgument(
                        "query and out must have the same dtype"));
  PADDLE_ENFORCE_EQ(dout.dtype(), q_type,
                    common::errors::InvalidArgument(
                        "query and dout must have the same dtype"));

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(softmax_lse);

  PADDLE_ENFORCE_EQ(q.strides()[q.strides().size() - 1], 1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(k.strides()[k.strides().size() - 1], 1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(v.strides()[v.strides().size() - 1], 1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(out.strides()[out.strides().size() - 1], 1,
                    common::errors::InvalidArgument(
                        "out tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(dout.strides()[dout.strides().size() - 1], 1,
                    common::errors::InvalidArgument(
                        "dout tensor must have contiguous last dimension"));

  paddle::Tensor cu_seqlens_q;
  bool const is_varlen_q = cu_seqlens_q_.is_initialized();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.get();
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    PADDLE_ENFORCE_EQ(cu_seqlens_q.dtype(), paddle::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_q must have dtype paddle.int32"));
    PADDLE_ENFORCE_GT(
        max_seqlen_q_, 0,
        common::errors::InvalidArgument(
            "max_seqlen_q must be provided if cu_seqlens_q is provided"));
  }
  paddle::Tensor cu_seqlens_k;
  bool const is_varlen_k = cu_seqlens_k_.is_initialized();
  if (is_varlen_k) {
    cu_seqlens_k = cu_seqlens_k_.get();
    CHECK_DEVICE(cu_seqlens_k);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    PADDLE_ENFORCE_EQ(cu_seqlens_k.dtype(), paddle::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_k must have dtype paddle.int32"));
    PADDLE_ENFORCE_GT(
        max_seqlen_k_, 0,
        common::errors::InvalidArgument(
            "max_seqlen_k must be provided if cu_seqlens_k is provided"));
  }
  // This is what we will template on
  bool const is_varlen = is_varlen_q || is_varlen_k ||
                         seqused_q_.is_initialized() ||
                         seqused_k_.is_initialized();
#ifdef FLASHATTENTION_DISABLE_VARLEN
  PADDLE_ENFORCE_EQ(!is_varlen, true,
                    common::errors::Unavailable(
                        "This flash attention build does not support varlen."));
#endif

  auto const sizes = q.dims();
  int const batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.dims()[0] - 1;
  int const seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_;
  int const total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
  int const num_heads = q.dims()[q.dims().size() - 2];
  int const head_size = q.dims()[q.dims().size() - 1];
  int const seqlen_k = !is_varlen_k ? k.dims()[1] : max_seqlen_k_;
  int const total_k = !is_varlen_k ? batch_size * k.dims()[1] : k.dims()[0];
  int const num_heads_k = k.dims()[k.dims().size() - 2];
  PADDLE_ENFORCE_EQ(
      head_size % 8, 0,
      common::errors::InvalidArgument("head_size should be a multiple of 8"));
  int const max_headdim = flashmaskv3_get_max_headdim();
  PADDLE_ENFORCE_LE(
      head_size, max_headdim,
      common::errors::InvalidArgument(
          "FlashAttention forward only supports head dimension at most %d",
          max_headdim));
  PADDLE_ENFORCE_EQ(
      num_heads % num_heads_k, 0,
      common::errors::InvalidArgument(
          "Number of heads in key/value must divide number of heads in query"));

  // This needs to go before kBlockM & kBlockN since we rely on the correct
  // window_size and is_causal to set kBlockM
  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_q - 1) {
    window_size_right = -1;
  }
  if (is_causal) {
    window_size_right = 0;
  }
  // There's a case where is_causal=false, window_size=(-1, 0). Then
  // set_params_bprop will set params.is_causal=true. If we don't have is_causal
  // here matching params.is_causal, we might get the wrong kBlockM (and cause
  // IMA).
  is_causal = window_size_left < 0 && window_size_right == 0;

  int const arch = dprops.major * 10 + dprops.minor;
  int const head_size_rounded = flashmaskv3_round_up_headdim(head_size);
  // Very important that these match the kernel configs
  bool const is_local =
      (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
  bool const is_flashmask = startend_row_indices_.is_initialized();
  paddle::Tensor startend_row_indices;
  if (is_flashmask)
    startend_row_indices = startend_row_indices_.get();
  bool const has_softcap = softcap > 0.0;

  paddle::Tensor flashmask_maxmin;
  paddle::Tensor lt_start_slice, lt_end_slice, ut_start_slice, ut_end_slice;
  const int32_t *lt_start_ptr;
  const int32_t *lt_end_ptr;
  const int32_t *ut_start_ptr;
  const int32_t *ut_end_ptr;

  if (is_flashmask) {
    PADDLE_ENFORCE_EQ(
        startend_row_indices.dtype(), paddle::DataType::INT32,
        common::errors::InvalidArgument(
            "flashmask_attention startend_row_indices must be INT32 type"));
    PADDLE_ENFORCE_EQ(
        startend_row_indices.dims().size(), 4,
        common::errors::InvalidArgument(
            "flashmask_attention receive startend_row_indices with dim "
            "[batch_size, num_heads,seq_len, mask_bounds]"));
    PADDLE_ENFORCE_EQ(startend_row_indices.dims()[3] == 1 ||
                          startend_row_indices.dims()[3] == 2 ||
                          startend_row_indices.dims()[3] == 4,
                      true,
                      common::errors::InvalidArgument(
                          "flashmask_attention startend_row_indices "
                          "mask_bounds must in [1,2,4]"));

    auto flashmask_maxmin_shape = startend_row_indices.dims();
    // TODO(umiswing): refine this block constraint (kBlockN % 32), since some
    // of kBlockN is not divisible by 32 flashmask_maxmin_shape[2] =
    // (flashmask_maxmin_shape[2] + 31) / 32 * 8;
    flashmask_maxmin_shape[2] =
        ((flashmask_maxmin_shape[2] + 31) / 32 + 3) / 4 * 4;
    flashmask_maxmin_shape[3] = 8;

    flashmask_maxmin =
        paddle::empty({flashmask_maxmin_shape[0], flashmask_maxmin_shape[1],
                       flashmask_maxmin_shape[2], flashmask_maxmin_shape[3]},
                      paddle::DataType::INT32, place);

    const int32_t *mask_base_ptr = startend_row_indices.data<int32_t>();
    auto mask_dims = startend_row_indices.dims();
    int B_mask = mask_dims[0];
    int H_mask = mask_dims[1];
    int S_mask = mask_dims[2];
    int C = mask_dims[3];
    int total_elements = B_mask * H_mask * S_mask;

    lt_start_ptr = nullptr;
    lt_end_ptr = nullptr;
    ut_start_ptr = nullptr;
    ut_end_ptr = nullptr;

    auto extract_channel = [&](int channel_idx) -> paddle::Tensor {
      auto slice = paddle::empty({B_mask, H_mask, S_mask},
                                 paddle::DataType::INT32, place);
      cudaMemcpy2DAsync(slice.data<int32_t>(), sizeof(int32_t),
                        mask_base_ptr + channel_idx, C * sizeof(int32_t),
                        sizeof(int32_t), total_elements,
                        cudaMemcpyDeviceToDevice, stream);
      return slice;
    };

    if (C == 1) {
      lt_start_ptr = mask_base_ptr;
    } else if (C == 2) {
      lt_start_slice = extract_channel(0);
      lt_start_ptr = lt_start_slice.data<int32_t>();
      if (!is_causal) {
        ut_end_slice = extract_channel(1);
        ut_end_ptr = ut_end_slice.data<int32_t>();
      } else {
        lt_end_slice = extract_channel(1);
        lt_end_ptr = lt_end_slice.data<int32_t>();
      }
    } else if (C == 4) {
      lt_start_slice = extract_channel(0);
      lt_start_ptr = lt_start_slice.data<int32_t>();
      lt_end_slice = extract_channel(1);
      lt_end_ptr = lt_end_slice.data<int32_t>();
      ut_start_slice = extract_channel(2);
      ut_start_ptr = ut_start_slice.data<int32_t>();
      ut_end_slice = extract_channel(3);
      ut_end_ptr = ut_end_slice.data<int32_t>();
    }
  }

  bool const is_blockmask = block_mask_.is_initialized();
  paddle::Tensor block_mask;
  if (is_blockmask)
    block_mask = block_mask_.get();

  if (is_blockmask) {
    PADDLE_ENFORCE_EQ(
        is_flashmask, true,
        common::errors::InvalidArgument(
            "blockmask should be used with flashmask at the same time "));

    PADDLE_ENFORCE_EQ(block_mask.dims().size(), 4,
                      common::errors::InvalidArgument(
                          "blockmask receive blockmask_indices with dim "
                          "[batch_size, num_heads, blocklen_q, blocklen_k]"));

    PADDLE_ENFORCE_EQ(block_mask.dims()[2], (seqlen_q + 127) / 128,
                      common::errors::InvalidArgument(
                          "blockmask only supports blockdim_q = 128 now"));

    PADDLE_ENFORCE_EQ(block_mask.dims()[3], (seqlen_k + 127) / 128,
                      common::errors::InvalidArgument(
                          "blockmask only supports blockdim_k = 128 now"));

    PADDLE_ENFORCE_EQ(
        block_mask.dims()[1], startend_row_indices.dims()[1],
        common::errors::InvalidArgument(
            "blockmask only supports same dim num_heads with flashmask now"));

    PADDLE_ENFORCE_LE(seqlen_k, 1024 * 128,
                      common::errors::InvalidArgument(
                          "blockmask only supports seqlen <= 128k in bwd now"));

    PADDLE_ENFORCE_LE(seqlen_q, 1024 * 128,
                      common::errors::InvalidArgument(
                          "blockmask only supports seqlen <= 128k in bwd now"));
  }

  // const bool has_lt_start = lt_start_row_indices.initialized();
  // const bool has_lt_end = lt_end_row_indices.initialized();
  // const bool has_ut_start = ut_start_row_indices.initialized();
  // const bool has_ut_end = ut_end_row_indices.initialized();

  const bool has_lt_start = (lt_start_ptr != nullptr);
  const bool has_lt_end = (lt_end_ptr != nullptr);
  const bool has_ut_start = (ut_start_ptr != nullptr);
  const bool has_ut_end = (ut_end_ptr != nullptr);

  // umiswing: The tile dispatch for flashmask is now different from fa3.
  // Replacing the original ternary operator with lambda makes the code
  // easier to reason about and less error-prone.
  const auto [kBlockM_sm90, kBlockN_sm90] = [&]() -> std::pair<int, int> {
    if (head_size_rounded <= 64) {
      if (is_flashmask && !is_causal) {
        return {64, 96};
      } else if (is_causal && has_softcap || is_flashmask) {
        return {96, 128};
      } else {
        return {128, 128};
      }
    } else if (head_size_rounded <= 128) {
      // umiswing: by now, we reuse template instantiation of head dim 128 for
      // head dim in range (64, 128], and therefore no separate dispatch for
      // head dim in range (64, 96]
      if (is_causal || is_local || has_softcap) {
        return {64, 128};
      } else {
        if ((seqlen_q >= 1024 || seqlen_k >= 1024) &&
            !(has_lt_end && has_ut_start)) {
          return {64, 128};
        } else {
          return {64, 64};
        }
      }
    } else if (head_size_rounded <= 256) {
      // umiswing: by now, we reuse template instantiation of head dim 256 for
      // head dim in range (128, 256], and therefore no separate dispatch for
      // head dim in range (128, 192]
      if (has_lt_end && has_ut_start) {
        return {64, 32};
      } else {
        return {64, 64};
      }
    } else {
      PADDLE_THROW(
          common::errors::Unimplemented("head dim is rounded to %d, which is "
                                        "not supported in FlashMask V3 now.",
                                        head_size_rounded));
      return {0, 0};
    }
  }();

  int const kBlockM_sm80 = head_size_rounded <= 64 ? 128 : 64;
  int const kBlockM_sm86 = head_size_rounded <= 192 ? 64 : 32;
  int const kBlockM =
      arch >= 90 ? kBlockM_sm90
                 : (arch == 86 || arch == 89 ? kBlockM_sm86 : kBlockM_sm80);
  int const kBlockN_sm80 =
      head_size_rounded <= 128 ? 128 : (head_size_rounded <= 192 ? 80 : 64);
  int const kBlockN_sm86 =
      head_size_rounded <= 64
          ? 128
          : (head_size_rounded <= 96
                 ? 128
                 : (head_size_rounded <= 128
                        ? 96
                        : (head_size_rounded <= 192 ? 64 : 64)));
  int const kBlockN =
      arch >= 90 ? kBlockN_sm90
                 : (arch == 86 || arch == 89 ? kBlockN_sm86 : kBlockN_sm80);
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
  int const seqlen_k_rounded = round_multiple(seqlen_k, kBlockN);
  int const total_q_padded_rounded =
      round_multiple(total_q + batch_size * kBlockM, kBlockM);
  int const total_k_padded_rounded =
      round_multiple(total_k + batch_size * kBlockN, kBlockN);

  if (!is_varlen_q) {
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);
  } else {
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  }
  if (!is_varlen_k) {
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
  } else {
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  }

  if (seqused_q_.is_initialized()) {
    auto seqused_q = seqused_q_.get();
    PADDLE_ENFORCE_EQ(
        seqused_q.dtype(), paddle::DataType::INT32,
        common::errors::InvalidArgument("seqused_q must have dtype int32"));
    CHECK_DEVICE(seqused_q);
    CHECK_CONTIGUOUS(seqused_q);
    CHECK_SHAPE(seqused_q, batch_size);
  }
  if (seqused_k_.is_initialized()) {
    auto seqused_k = seqused_k_.get();
    PADDLE_ENFORCE_EQ(
        seqused_k.dtype(), paddle::DataType::INT32,
        common::errors::InvalidArgument("seqused_k must have dtype int32"));
    CHECK_DEVICE(seqused_k);
    CHECK_CONTIGUOUS(seqused_k);
    CHECK_SHAPE(seqused_k, batch_size);
  }

  if (dq_.is_initialized()) {
    *dq = dq_.get();
    PADDLE_ENFORCE_EQ(
        dq->dtype(), q_type,
        common::errors::InvalidArgument("dq must have the same dtype as q"));
    CHECK_DEVICE((*dq));
    PADDLE_ENFORCE_EQ(dq->strides()[dq->strides().size() - 1], 1,
                      common::errors::InvalidArgument(
                          "dq must have contiguous last dimension"));
    if (!is_varlen_q) {
      CHECK_SHAPE((*dq), batch_size, seqlen_q, num_heads, head_size);
    } else {
      CHECK_SHAPE((*dq), total_q, num_heads, head_size);
    }
  } else {
    *dq = paddle::empty_like(q);
  }
  if (dk_.is_initialized()) {
    *dk = dk_.get();
    PADDLE_ENFORCE_EQ(
        dk->dtype(), q_type,
        common::errors::InvalidArgument("dk must have the same dtype as q"));
    CHECK_DEVICE((*dk));
    PADDLE_ENFORCE_EQ(dk->strides()[dk->strides().size() - 1], 1,
                      common::errors::InvalidArgument(
                          "dk must have contiguous last dimension"));
    if (!is_varlen_k) {
      CHECK_SHAPE((*dk), batch_size, seqlen_k, num_heads_k, head_size);
    } else {
      CHECK_SHAPE((*dk), total_k, num_heads_k, head_size);
    }
  } else {
    *dk = paddle::empty_like(k);
  }
  if (dv_.is_initialized()) {
    *dv = dv_.get();
    PADDLE_ENFORCE_EQ(
        dv->dtype(), q_type,
        common::errors::InvalidArgument("dv must have the same dtype as q"));
    CHECK_DEVICE((*dv));
    PADDLE_ENFORCE_EQ(dv->strides()[dv->strides().size() - 1], 1,
                      common::errors::InvalidArgument(
                          "dv must have contiguous last dimension"));
    if (!is_varlen_k) {
      CHECK_SHAPE((*dv), batch_size, seqlen_k, num_heads_k, head_size);
    } else {
      CHECK_SHAPE((*dv), total_k, num_heads_k, head_size);
    }
  } else {
    *dv = paddle::empty_like(v);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing

  // Need softmax_d to have total_q_padded_rounded since we want its address to
  // be aligned by 16/8 bytes for TMA / LDG.64
  if (!is_varlen) {
    if (softmax_d) {
      // Need softmax_d to have seqlen_q_rounded since we want its address to be
      // aligned by 16/8 bytes for TMA / LDG.64
      *softmax_d = paddle::empty({batch_size, num_heads, seqlen_q_rounded},
                                 paddle::DataType::FLOAT32, place);
    }
    if (softmax_lse_log2) {
      *softmax_lse_log2 =
          paddle::empty({batch_size, num_heads, seqlen_q_rounded},
                        paddle::DataType::FLOAT32, place);
    }
  } else {
    if (softmax_d) {
      *softmax_d = paddle::empty({num_heads, total_q_padded_rounded},
                                 paddle::DataType::FLOAT32, place);
    }
    if (softmax_lse_log2) {
      *softmax_lse_log2 = paddle::empty({num_heads, total_q_padded_rounded},
                                        paddle::DataType::FLOAT32, place);
    }
  }

  if (dq_accum) {
    if (!is_varlen) {
      *dq_accum = paddle::empty(
          {batch_size, num_heads, seqlen_q_rounded * head_size_rounded},
          paddle::DataType::FLOAT32, place);

    } else {
      *dq_accum =
          paddle::empty({num_heads, total_q_padded_rounded * head_size_rounded},
                        paddle::DataType::FLOAT32, place);
    }
  }

  if (num_heads_k != num_heads) { // MQA / GQA
    if (!is_varlen) {
      if (dk_accum) {
        *dk_accum = paddle::empty(
            {batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded},
            paddle::DataType::FLOAT32, place);
      }
      if (dv_accum) {
        *dv_accum = paddle::empty(
            {batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded},
            paddle::DataType::FLOAT32, place);
      }
    } else {
      if (dk_accum) {
        *dk_accum = paddle::empty(
            {num_heads_k, total_k_padded_rounded, head_size_rounded},
            paddle::DataType::FLOAT32, place);
      }
      if (dv_accum) {
        *dv_accum = paddle::empty(
            {num_heads_k, total_k_padded_rounded, head_size_rounded},
            paddle::DataType::FLOAT32, place);
      }
    }

    if (dk_accum) {
      *dk_accum = paddle::full(dk_accum->shape(), float{0},
                               paddle::DataType::FLOAT32, place);
    }
    if (dv_accum) {
      *dv_accum = paddle::full(dv_accum->shape(), float{0},
                               paddle::DataType::FLOAT32, place);
    }
  }

  FlashMask_bwd_params *params_handle = get_flashmask_bwd_params_handle();
  flashmaskv3_clear_bwd_params_handle(params_handle);
  set_flashmaskv3_params_dgrad(
      params_handle, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded,
      seqlen_k_rounded, num_heads, num_heads_k, head_size, head_size_rounded, q,
      k, v, out, dout, dq, dk, dv, !is_varlen_q ? nullptr : cu_seqlens_q.data(),
      !is_varlen_k ? nullptr : cu_seqlens_k.data(),
      seqused_q_.is_initialized() ? const_cast<void *>(seqused_q_.get().data())
                                  : nullptr,
      seqused_k_.is_initialized() ? const_cast<void *>(seqused_k_.get().data())
                                  : nullptr,
      dq_accum ? dq_accum->data() : nullptr,
      num_heads_k != num_heads && dk_accum ? dk_accum->data() : nullptr,
      num_heads_k != num_heads && dv_accum ? dv_accum->data() : nullptr,
      const_cast<void *>(softmax_lse.data()),
      softmax_d ? (softmax_d->data()) : nullptr,
      /*p_dropout=*/0.f, softmax_scale, window_size_left, window_size_right,
      dprops, softcap, deterministic, sm_margin);
  flashmaskv3_bwd_params_set_total_q(params_handle, total_q);
  flashmaskv3_bwd_params_set_total_k(params_handle, total_k);
  flashmaskv3_bwd_params_set_softmax_lse_log2_ptr(
      params_handle, softmax_lse_log2 ? softmax_lse_log2->data() : nullptr);
  flashmaskv3_bwd_params_set_dv(params_handle,
                                head_size); // We don't support hdim_v being
                                            // different from hdim_qk for now
  paddle::Tensor tile_count_semaphore;
  if (arch >= 90) {
    tile_count_semaphore = paddle::full({1}, 0, paddle::DataType::INT32, place);

    flashmaskv3_bwd_params_set_tile_count_semaphore(
        params_handle, tile_count_semaphore.data<int>());
  } else {
    flashmaskv3_bwd_params_set_tile_count_semaphore(params_handle, nullptr);
  }

  paddle::Tensor dq_semaphore =
      paddle::empty({(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads},
                    paddle::DataType::INT32, place);
  flashmaskv3_bwd_params_set_dq_semaphore(params_handle,
                                          dq_semaphore.data<int>());

  paddle::Tensor dk_semaphore;
  paddle::Tensor dv_semaphore;
  if (num_heads_k != num_heads &&
      flashmaskv3_bwd_params_get_deterministic(params_handle)) {
    // xiangrui: we need to zero them out
    dk_semaphore = paddle::full(
        {(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, 0,
        paddle::DataType::INT32, place);

    dv_semaphore = paddle::full(
        {(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, 0,
        paddle::DataType::INT32, place);

    flashmaskv3_bwd_params_set_dk_semaphore(params_handle,
                                            dk_semaphore.data<int>());
    flashmaskv3_bwd_params_set_dv_semaphore(params_handle,
                                            dv_semaphore.data<int>());
  }

  if (is_flashmask) {
    flashmaskv3_bwd_params_set_lt_start_ptr(
        params_handle, const_cast<int32_t *>(lt_start_ptr));
    flashmaskv3_bwd_params_set_lt_end_ptr(params_handle,
                                          const_cast<int32_t *>(lt_end_ptr));
    flashmaskv3_bwd_params_set_ut_start_ptr(
        params_handle, const_cast<int32_t *>(ut_start_ptr));
    flashmaskv3_bwd_params_set_ut_end_ptr(params_handle,
                                          const_cast<int32_t *>(ut_end_ptr));

    if (flashmask_maxmin.initialized())
      flashmaskv3_bwd_params_set_flashmask_maxmin_ptr(
          params_handle, (flashmask_maxmin.data<int32_t>()));
    else
      flashmaskv3_bwd_params_set_flashmask_maxmin_ptr(params_handle, nullptr);

    flashmaskv3_bwd_params_set_h_flashmask(params_handle,
                                           startend_row_indices.dims()[1]);
    flashmaskv3_bwd_params_set_h_h_flashmask_ratio(
        params_handle, num_heads / startend_row_indices.dims()[1]);
  } else {
    flashmaskv3_bwd_params_set_lt_start_ptr(params_handle, nullptr);
    flashmaskv3_bwd_params_set_lt_end_ptr(params_handle, nullptr);
    flashmaskv3_bwd_params_set_ut_start_ptr(params_handle, nullptr);
    flashmaskv3_bwd_params_set_ut_end_ptr(params_handle, nullptr);
    flashmaskv3_bwd_params_set_flashmask_maxmin_ptr(params_handle, nullptr);
    flashmaskv3_bwd_params_set_h_flashmask(params_handle, 0);
    flashmaskv3_bwd_params_set_h_h_flashmask_ratio(params_handle, 0);
  }

  if (is_blockmask) {
    // xhy: blockmask is now only support blockdim_q k = 128
    flashmaskv3_bwd_params_set_m_block_dim(params_handle, 128);
    flashmaskv3_bwd_params_set_n_block_dim(params_handle, 128);
    flashmaskv3_bwd_params_set_block_mask_ptr(params_handle,
                                              (block_mask.data<int32_t>()));
  }
#ifdef FLASHATTENTION_DISABLE_LOCAL
  PADDLE_ENABLE_EQ(
      !flashmaskv3_bwd_params_get_is_local(params_handle), true,
      "This flash attention build does not support local attention.");
#endif
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  PADDLE_ENABLE_EQ(
      flashmaskv3_bwd_params_get_softcap(params_handle), 0.0,
      "This flash attention build does not support tanh softcapping.");
#endif

  if (total_q > 0 && total_k > 0 && num_heads_k > 0) {
    flashmaskv3_run_mha_bwd(params_handle, stream);
  } else if (total_k > 0 && num_heads_k > 0) {
    *dk = paddle::full(dk->shape(), T{0}, q_type, place);
    *dv = paddle::full(dv->shape(), T{0}, q_type, place);
    if (softmax_d) {
      *softmax_d = paddle::full(softmax_d->shape(), float{0},
                                paddle::DataType::FLOAT32, place);
    }
  } else if (total_q > 0 && num_heads_k > 0) {
    *dq = paddle::full(dq->shape(), T{0}, q_type, place);
    if (softmax_d) {
      *softmax_d = paddle::full(softmax_d->shape(), float{0},
                                paddle::DataType::FLOAT32, place);
    }
  }
#else
  RaiseNotSupportedError();
#endif
}

#define FLASHMASK_V3_GRAD_BASE_KERNEL_IMPL(DType)                              \
  template void FlashMaskV3GradBaseKernel<DType>(                              \
      const paddle::Tensor &dout, const paddle::Tensor &q,                     \
      const paddle::Tensor &k, const paddle::Tensor &v,                        \
      const paddle::Tensor &out, const paddle::Tensor &softmax_lse,            \
      const paddle::optional<paddle::Tensor> &dq_,                             \
      const paddle::optional<paddle::Tensor> &dk_,                             \
      const paddle::optional<paddle::Tensor> &dv_,                             \
      const paddle::optional<paddle::Tensor> &cu_seqlens_q_,                   \
      const paddle::optional<paddle::Tensor> &cu_seqlens_k_,                   \
      const paddle::optional<paddle::Tensor> &seqused_q_,                      \
      const paddle::optional<paddle::Tensor> &seqused_k_,                      \
      const paddle::optional<paddle::Tensor> &startend_row_indices_,           \
      const paddle::optional<paddle::Tensor> &block_mask_, int max_seqlen_q_,  \
      int max_seqlen_k_, float const softmax_scale, bool is_causal,            \
      int window_size_left, int window_size_right, float const softcap,        \
      bool const deterministic, int const sm_margin, paddle::Tensor *dq,       \
      paddle::Tensor *dk, paddle::Tensor *dv, paddle::Tensor *softmax_d,       \
      paddle::Tensor *softmax_lse_log2, paddle::Tensor *dq_accum,              \
      paddle::Tensor *dk_accum, paddle::Tensor *dv_accum)

FLASHMASK_V3_GRAD_BASE_KERNEL_IMPL(paddle::float16);
FLASHMASK_V3_GRAD_BASE_KERNEL_IMPL(paddle::bfloat16);
