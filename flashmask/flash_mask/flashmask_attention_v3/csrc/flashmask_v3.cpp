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

#include "flashmask_v3.h"
#include "paddle/extension.h"
#include "paddle/common/flags.h"

COMMON_DECLARE_bool(cudnn_deterministic);

static void RaiseNotSupportedError(int version = 2) {
  PADDLE_THROW(common::errors::Unimplemented(
      "FlashAttention %d is unsupported, please check "
      "the GPU compatibility and CUDA Version.",
      version));
}

std::vector<paddle::Tensor>
FlashMaskV3Forward(const paddle::Tensor &query, const paddle::Tensor &key,
                   const paddle::Tensor &value,
                   const paddle::Tensor &startend_row_indices,
                   const paddle::optional<paddle::Tensor> &block_mask,
                   float softmax_scale, bool is_causal) {
#ifdef PADDLE_WITH_FLASHATTN_V3

  paddle::Tensor out;
  paddle::Tensor softmax_lse;
  paddle::Tensor out_accum;
  paddle::Tensor softmax_lse_accum;

#define CALL_FLASHMASK_V3_BASE_KERNEL(DType)                                   \
  FlashMaskV3BaseKernel<DType>(                                                \
      query, key, value, paddle::none, paddle::none, paddle::none,             \
      paddle::none, paddle::none, paddle::none, paddle::none, paddle::none,    \
      paddle::none, paddle::none, paddle::none, paddle::none, paddle::none,    \
      paddle::none, paddle::none, paddle::none, paddle::none, paddle::none,    \
      startend_row_indices, block_mask, 0, 0, softmax_scale, is_causal, -1,    \
      -1, float{0}, true, 1, false, false, 0, &out, &softmax_lse, &out_accum,  \
      &softmax_lse_accum)

  switch (query.dtype()) {
  case paddle::DataType::FLOAT16: {
    CALL_FLASHMASK_V3_BASE_KERNEL(paddle::float16);
    break;
  }
  case paddle::DataType::BFLOAT16: {
    CALL_FLASHMASK_V3_BASE_KERNEL(paddle::bfloat16);
    break;
  }
  default: {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "FlashMaskV3BaseKernel only support bfloat16 and float16, but got %d",
        static_cast<int>(query.dtype())));
  }
  }

  return {out, softmax_lse};
#else
  RaiseNotSupportedError();
#endif
}

std::vector<paddle::Tensor> FlashMaskV3GradKernel(
    const paddle::Tensor &query, const paddle::Tensor &key,
    const paddle::Tensor &value, const paddle::Tensor &out,
    const paddle::Tensor &softmax_lse,
    const paddle::Tensor &startend_row_indices, // TODO(xiehaoyang): remove this
    const paddle::optional<paddle::Tensor> &block_mask,
    const paddle::Tensor &out_grad, float const softmax_scale, bool is_causal) {
#ifdef PADDLE_WITH_FLASHATTN_V3

  PADDLE_ENFORCE_EQ(
      query.dims()[query.dims().size() - 1],
      value.dims()[value.dims().size() - 1],
      common::errors::InvalidArgument("head_dim_q != head_dim_v (%d != %d)",
                                      query.dims()[query.dims().size() - 1],
                                      value.dims()[value.dims().size() - 1]));

  // umiswing: fake grad tensor for FlashAttnV3GradBaseKernel
  paddle::Tensor softmax_d;
  paddle::Tensor softmax_lse_log2;
  paddle::Tensor dq_accum;
  paddle::Tensor dk_accum;
  paddle::Tensor dv_accum;

  paddle::Tensor dq;
  paddle::Tensor dk;
  paddle::Tensor dv;

#define CALL_FLASHMASK_V3_BASE_GRAD_KERNEL(DType)                              \
  FlashMaskV3GradBaseKernel<DType>(                                            \
      out_grad, query, key, value, out, softmax_lse, paddle::none,             \
      paddle::none, paddle::none, paddle::none, paddle::none, paddle::none,    \
      paddle::none, startend_row_indices, block_mask, 0, 0, softmax_scale,     \
      is_causal, -1, -1, 0, FLAGS_cudnn_deterministic, 0, &dq, &dk, &dv,       \
      &softmax_d, &softmax_lse_log2, &dq_accum, &dk_accum, &dv_accum);

  switch (query.dtype()) {
  case paddle::DataType::FLOAT16: {
    CALL_FLASHMASK_V3_BASE_GRAD_KERNEL(paddle::float16);
    break;
  }
  case paddle::DataType::BFLOAT16: {
    CALL_FLASHMASK_V3_BASE_GRAD_KERNEL(paddle::bfloat16);
    break;
  }
  default: {
    PADDLE_THROW(
        phi::errors::InvalidArgument("FlashMaskV3GradBaseKernel only support "
                                     "bfloat16 and float16, but got %d",
                                     static_cast<int>(query.dtype())));
  }
  }

  // umiswing: some branch in upstream fa3 could have padded the head dimension
  PADDLE_ENFORCE_EQ(
      dq.dims()[dq.dims().size() - 1],
      out_grad.dims()[out_grad.dims().size() - 1],
      common::errors::InvalidArgument(
          "head dimension of dq != head dimension of out_grad (%d != %d)",
          dq.dims()[dq.dims().size() - 1],
          out_grad.dims()[out_grad.dims().size() - 1]));

  PADDLE_ENFORCE_EQ(
      dk.dims()[dk.dims().size() - 1],
      out_grad.dims()[out_grad.dims().size() - 1],
      common::errors::InvalidArgument(
          "head dimension of dk != head dimension of out_grad (%d != %d)",
          dk.dims()[dk.dims().size() - 1],
          out_grad.dims()[out_grad.dims().size() - 1]));

  PADDLE_ENFORCE_EQ(
      dv.dims()[dv.dims().size() - 1],
      out_grad.dims()[out_grad.dims().size() - 1],
      common::errors::InvalidArgument(
          "head dimension of dv != head dimension of out_grad (%d != %d)",
          dv.dims()[dv.dims().size() - 1],
          out_grad.dims()[out_grad.dims().size() - 1]));
  return {dq, dk, dv};

#else
  RaiseNotSupportedError();
#endif
}

std::vector<std::vector<int64_t>> FlashMaskV3FwdInferShape(
    const std::vector<int64_t> &query_shape,
    const std::vector<int64_t> &key_shape,
    const std::vector<int64_t> &value_shape,
    const std::vector<int64_t> &startend_row_indices_shape,
    const paddle::optional<std::vector<int64_t>> &block_mask_shape,
    float softmax_scale, bool is_causal) {
  int64_t batch_size = query_shape[0];
  int64_t seqlen_q = query_shape[1];
  int64_t num_heads = query_shape[query_shape.size() - 2];
  int64_t head_size_v = value_shape[value_shape.size() - 1];

  return {{batch_size, seqlen_q, num_heads, head_size_v},
          {batch_size, num_heads, seqlen_q}};
}

std::vector<std::vector<int64_t>> FlashMaskV3GradInferShape(
    const std::vector<int64_t> &query_shape,
    const std::vector<int64_t> &key_shape,
    const std::vector<int64_t> &value_shape,
    const std::vector<int64_t> &out_shape,
    const std::vector<int64_t> &softmax_lse_shape,
    const std::vector<int64_t> &startend_row_indices_shape,
    const paddle::optional<std::vector<int64_t>> &block_mask_shape,
    const std::vector<int64_t> &dout_shape, float softmax_scale,
    bool is_causal) {

  return {query_shape, key_shape, value_shape};
}

std::vector<paddle::DataType> FlashMaskV3FwdInferDtype(
    paddle::DataType query_dtype, paddle::DataType key_dtype,
    paddle::DataType value_dtype, paddle::DataType startend_row_indices_dtype,
    const paddle::optional<paddle::DataType> &block_mask_dtype,
    float softmax_scale, bool is_causal) {
  auto out_type = (query_dtype == paddle::DataType::FLOAT8_E4M3FN)
                      ? paddle::DataType::BFLOAT16
                      : query_dtype;
  return {out_type, paddle::DataType::FLOAT32};
}

std::vector<paddle::DataType> FlashMaskV3GradInferDtype(
    paddle::DataType query_dtype, paddle::DataType key_dtype,
    paddle::DataType value_dtype, paddle::DataType out_dtype,
    paddle::DataType softmax_lse_dtype,
    paddle::DataType startend_row_indices_dtype,
    const paddle::optional<paddle::DataType> &block_mask_dtype,
    paddle::DataType dout_dtype, float softmax_scale, bool is_causal) {
  return {query_dtype, key_dtype, value_dtype};
}

PD_BUILD_OP(flashmask_attention_v3)
    .Inputs({"query", "key", "value", "startend_row_indices",
             paddle::Optional("block_mask")})
    .Outputs({"out", "softmax_lse"})
    .Attrs({"softmax_scale: float", "is_causal: bool"})
    .SetKernelFn(PD_KERNEL(FlashMaskV3Forward))
    .SetInferShapeFn(PD_INFER_SHAPE(FlashMaskV3FwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FlashMaskV3FwdInferDtype));

PD_BUILD_GRAD_OP(flashmask_attention_v3)
    .Inputs({
        "query", "key", "value", "out", "softmax_lse", "startend_row_indices",
        paddle::Optional("block_mask"),
        paddle::Grad("out") // dout
    })
    .Outputs({paddle::Grad("query"), paddle::Grad("key"),
              paddle::Grad("value")})
    .Attrs({"softmax_scale: float", "is_causal: bool"})
    .SetKernelFn(PD_KERNEL(FlashMaskV3GradKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(FlashMaskV3GradInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FlashMaskV3GradInferDtype));
