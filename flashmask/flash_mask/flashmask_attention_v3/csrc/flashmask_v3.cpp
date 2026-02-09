// [BQW_CHANGE] 使用 PD_BUILD_OP 注册自定义算子，替代原来的 pybind11 绑定
// 这样 Paddle 的 CUDAExtension 会自动处理 Python 绑定，避免模块名不匹配问题

#include "paddle/extension.h"
#include "flashmask_v3.h"

std::vector<paddle::Tensor> FlashMaskV2Forward(
    const paddle::Tensor &query,
    const paddle::Tensor &key,
    const paddle::Tensor &value,
    const paddle::Tensor &startend_row_indices,
    const paddle::optional<paddle::Tensor> &block_mask,
    float softmax_scale,
    bool is_causal
) {
    paddle::Tensor out;
    paddle::Tensor softmax_lse;
    paddle::Tensor out_accum;
    paddle::Tensor softmax_lse_accum;

    FlashMaskV2BaseKernel<paddle::bfloat16>(
        query, key, value,
        paddle::none,  // k_new_
        paddle::none,  // v_new_
        paddle::none,  // q_v_
        paddle::none,  // out_
        paddle::none,  // cu_seqlens_q_
        paddle::none,  // cu_seqlens_k_
        paddle::none,  // cu_seqlens_k_new_
        paddle::none,  // seqused_q_
        paddle::none,  // seqused_k_
        paddle::none,  // page_table_
        paddle::none,  // kv_batch_idx_
        paddle::none,  // leftpad_k_
        paddle::none,  // rotary_cos_
        paddle::none,  // rotary_sin_
        paddle::none,  // q_descale_
        paddle::none,  // k_descale_
        paddle::none,  // v_descale_
        paddle::none,  // scheduler_metadata_
        startend_row_indices,
        block_mask,
        0,  // max_seqlen_q_
        0,  // max_seqlen_k_
        softmax_scale,
        is_causal,
        -1,        // window_size_left
        -1,        // window_size_right
        float{0},  // softcap
        true,      // is_rotary_interleaved
        1,         // num_splits
        false,     // manual_set_pack_gqa
        false,     // pack_gqa_
        0,         // sm_margin
        &out,
        &softmax_lse,
        &out_accum,
        &softmax_lse_accum);

    return {out, softmax_lse};
}

std::vector<std::vector<int64_t>> FlashMaskV2FwdInferShape(
    const std::vector<int64_t>& query_shape,
    const std::vector<int64_t>& key_shape,
    const std::vector<int64_t>& value_shape,
    const std::vector<int64_t>& startend_row_indices_shape,
    const paddle::optional<std::vector<int64_t>>& block_mask_shape,
    float softmax_scale,
    bool is_causal
) {
    int64_t batch_size = query_shape[0];
    int64_t seqlen_q = query_shape[1];
    int64_t num_heads = query_shape[2];
    int64_t head_size_v = value_shape[3];

    return {{batch_size, seqlen_q, num_heads, head_size_v},
            {batch_size, num_heads, seqlen_q}};
}

std::vector<paddle::DataType> FlashMaskV2FwdInferDtype(
    paddle::DataType query_dtype,
    paddle::DataType key_dtype,
    paddle::DataType value_dtype,
    paddle::DataType startend_row_indices_dtype,
    const paddle::optional<paddle::DataType>& block_mask_dtype,
    float softmax_scale,
    bool is_causal
) {
    auto out_type = (query_dtype == paddle::DataType::FLOAT8_E4M3FN)
                    ? paddle::DataType::BFLOAT16
                    : query_dtype;
    return {out_type, paddle::DataType::FLOAT32};
}

PD_BUILD_OP(flashmask_attention_v2)
    .Inputs({"query", "key", "value", "startend_row_indices",
             paddle::Optional("block_mask")})
    .Outputs({"out", "softmax_lse"})
    .Attrs({"softmax_scale: float", "is_causal: bool"})
    .SetKernelFn(PD_KERNEL(FlashMaskV2Forward))
    .SetInferShapeFn(PD_INFER_SHAPE(FlashMaskV2FwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FlashMaskV2FwdInferDtype));
