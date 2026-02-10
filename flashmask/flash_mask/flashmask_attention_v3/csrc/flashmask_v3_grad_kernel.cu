// flash_mask/flashmask_attention_v3/csrc/flashmask_v3_kernel.cu
#include "flash_attn_v3_utils.h"
// [BQW_CHANGE] 移除了重复的 #include "paddle/phi/core/dense_tensor.h" (外迁后不需要)
#include "flashmask_v3.h"
// 这里可以安全地包含 CUDA 头文件，因为 nvcc 会处理
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define PADDLE_WITH_FLASHATTN_V3
#ifdef PADDLE_WITH_FLASHATTN_V3

template <typename T>
void FlashMaskV2GradBaseKernel(
    const paddle::Tensor &dout,
    const paddle::Tensor &q,
    const paddle::Tensor &k,
    const paddle::Tensor &v,
    const paddle::Tensor &out,
    const paddle::Tensor &softmax_lse,
    const paddle::optional<paddle::Tensor> &dq_,
    const paddle::optional<paddle::Tensor> &dk_,
    const paddle::optional<paddle::Tensor> &dv_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_q_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_,
    const paddle::optional<paddle::Tensor> &seqused_q_,
    const paddle::optional<paddle::Tensor> &seqused_k_,
    const paddle::optional<paddle::Tensor> &startend_row_indices_,
    const paddle::optional<paddle::Tensor> &block_mask_,
    int max_seqlen_q_,
    int max_seqlen_k_,
    float const softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const deterministic,
    int const sm_margin,
    paddle::Tensor *dq,
    paddle::Tensor *dk,
    paddle::Tensor *dv,
    paddle::Tensor *softmax_d,
    paddle::Tensor *softmax_lse_log2,
    paddle::Tensor *dq_accum,
    paddle::Tensor *dk_accum,
    paddle::Tensor *dv_accum)
#if 0
  {
    printf("now in FlashMaskV2GradBaseKernel \n");
  }
#else
 {
#ifdef PADDLE_WITH_FLASHATTN_V3

  printf("now in FlashMaskV2GradBaseKernel \n");

  // TODO(umiswing): support ampere
  // int device_id = dev_ctx.GetPlace().GetDeviceId();
  // auto dprops = paddle::platform::GetDeviceProperties(device_id);

  // [BQW_CHANGE] 获取 stream
  cudaStream_t stream = static_cast<cudaStream_t>(q.stream());
  int device_id = q.place().GetDeviceId();
  cudaDeviceProp dprops;
  cudaGetDeviceProperties(&dprops, device_id);

  const bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  PADDLE_ENFORCE_EQ(is_sm90,
                    true,
                    common::errors::Unavailable(
                        "FlashAttention-3 only supports Hopper GPUs."));

  auto q_type = q.dtype();
  PADDLE_ENFORCE_EQ(
      (q_type == paddle::DataType::FLOAT16 || q_type == paddle::DataType::BFLOAT16),
      true,
      common::errors::InvalidArgument(
          "FlashAttention-3 bwd only support fp16 and bf16 data type"));
  PADDLE_ENFORCE_EQ(k.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and key must have the same dtype"));
  PADDLE_ENFORCE_EQ(v.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and value must have the same dtype"));
  PADDLE_ENFORCE_EQ(out.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and out must have the same dtype"));
  PADDLE_ENFORCE_EQ(dout.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and dout must have the same dtype"));

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(dout);
  CHECK_DEVICE(softmax_lse);

  PADDLE_ENFORCE_EQ(q.strides()[q.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(k.strides()[k.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(v.strides()[v.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "Input tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(out.strides()[out.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "out tensor must have contiguous last dimension"));
  PADDLE_ENFORCE_EQ(dout.strides()[dout.strides().size() - 1],
                    1,
                    common::errors::InvalidArgument(
                        "dout tensor must have contiguous last dimension"));

  paddle::Tensor cu_seqlens_q;
  bool const is_varlen_q = cu_seqlens_q_.is_initialized();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.get();
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    PADDLE_ENFORCE_EQ(cu_seqlens_q.dtype(),
                      paddle::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_q must have dtype paddle.int32"));
    PADDLE_ENFORCE_GT(
        max_seqlen_q_,
        0,
        common::errors::InvalidArgument(
            "max_seqlen_q must be provided if cu_seqlens_q is provided"));
  }
  paddle::Tensor cu_seqlens_k;
  bool const is_varlen_k = cu_seqlens_k_.is_initialized();
  if (is_varlen_k) {
    cu_seqlens_k = cu_seqlens_k_.get();
    CHECK_DEVICE(cu_seqlens_k);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    PADDLE_ENFORCE_EQ(cu_seqlens_k.dtype(),
                      paddle::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_k must have dtype paddle.int32"));
    PADDLE_ENFORCE_GT(
        max_seqlen_k_,
        0,
        common::errors::InvalidArgument(
            "max_seqlen_k must be provided if cu_seqlens_k is provided"));
  }
  // This is what we will template on
  bool const is_varlen = is_varlen_q || is_varlen_k ||
                         seqused_q_.is_initialized() ||
                         seqused_k_.is_initialized();
#ifdef FLASHATTENTION_DISABLE_VARLEN
  PADDLE_ENFORCE_EQ(!is_varlen,
                    true,
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
      head_size % 8,
      0,
      common::errors::InvalidArgument("head_size should be a multiple of 8"));
  int const max_headdim = flashmaskv2_get_max_headdim();
  PADDLE_ENFORCE_LE(
      head_size,
      max_headdim,
      common::errors::InvalidArgument(
          "FlashAttention forward only supports head dimension at most %d",
          max_headdim));
  PADDLE_ENFORCE_EQ(
      num_heads % num_heads_k,
      0,
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
  int const head_size_rounded = flashmaskv2_round_up_headdim(head_size);
  // Very important that these match the kernel configs
  bool const is_local =
      (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
  bool const is_flashmask = startend_row_indices_.is_initialized();
  paddle::Tensor startend_row_indices;
  if (is_flashmask) startend_row_indices = startend_row_indices_.get();
  bool const has_softcap = softcap > 0.0;

  // flashmask
  // paddle::Tensor flashmask_maxmin, lt_start_row_indices, lt_end_row_indices,
  //     ut_start_row_indices, ut_end_row_indices;

  paddle::Tensor flashmask_maxmin;
  // [BQW_CHANGE] 声明 slice tensor 在外层作用域，确保生命周期覆盖整个 kernel 执行
  // phi::Slice 原本会创建连续内存拷贝(非 view)，这里用 paddle::Tensor 持有拷贝结果
  paddle::Tensor lt_start_slice, lt_end_slice, ut_start_slice, ut_end_slice;
  const int32_t* lt_start_ptr;
  const int32_t* lt_end_ptr;
  const int32_t* ut_start_ptr;
  const int32_t* ut_end_ptr;

  if (is_flashmask) {
    PADDLE_ENFORCE_EQ(
        startend_row_indices.dtype(),
        paddle::DataType::INT32,
        common::errors::InvalidArgument(
            "flashmask_attention startend_row_indices must be INT32 type"));
    PADDLE_ENFORCE_EQ(
        startend_row_indices.dims().size(),
        4,
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

    // flashmask_maxmin.set_type(paddle::DataType::INT32);
    // flashmask_maxmin.Resize(flashmask_maxmin_shape);
    // dev_ctx.template Alloc<int32_t>(&flashmask_maxmin);
    flashmask_maxmin = paddle::empty(
      {flashmask_maxmin_shape[0], flashmask_maxmin_shape[1], 
       flashmask_maxmin_shape[2], flashmask_maxmin_shape[3]},
      paddle::DataType::INT32,
      q.place()
    );

    const int32_t* mask_base_ptr = startend_row_indices.data<int32_t>();
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
      auto slice = paddle::empty({B_mask, H_mask, S_mask}, paddle::DataType::INT32, q.place());
      cudaMemcpy2DAsync(
          slice.data<int32_t>(),                    // dst (连续)
          sizeof(int32_t),                           // dpitch
          mask_base_ptr + channel_idx,               // src (交错起始位置)
          C * sizeof(int32_t),                       // spitch (源 stride)
          sizeof(int32_t),                           // width (单个元素)
          total_elements,                            // height (元素总数)
          cudaMemcpyDeviceToDevice,
          stream
      );
      return slice;
    };

    if (C == 1) {
      // C=1 时数据本身就是连续的，无需拷贝
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

    // lt_start_row_indices =
    //     phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {0}, {1});
    // if (startend_row_indices.dims()[3] == 2) {
    //   if (!is_causal) {
    //     ut_end_row_indices =
    //         phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {1}, {2});
    //   } else {
    //     lt_end_row_indices =
    //         phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {1}, {2});
    //   }
    // } else if (startend_row_indices.dims()[3] == 4) {
    //   ut_end_row_indices =
    //       phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {3}, {4});
    //   lt_end_row_indices =
    //       phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {1}, {2});
    //   ut_start_row_indices =
    //       phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {2}, {3});
    // }
  }

  bool const is_blockmask = block_mask_.is_initialized();
  paddle::Tensor block_mask;
  if (is_blockmask) block_mask = block_mask_.get();

  if (is_blockmask) {
    PADDLE_ENFORCE_EQ(
        is_flashmask,
        true,
        common::errors::InvalidArgument(
            "blockmask should be used with flashmask at the same time "));

    PADDLE_ENFORCE_EQ(block_mask.dims().size(),
                      4,
                      common::errors::InvalidArgument(
                          "blockmask receive blockmask_indices with dim "
                          "[batch_size, num_heads, blocklen_q, blocklen_k]"));

    PADDLE_ENFORCE_EQ(block_mask.dims()[2],
                      (seqlen_q + 127) / 128,
                      common::errors::InvalidArgument(
                          "blockmask only supports blockdim_q = 128 now"));

    PADDLE_ENFORCE_EQ(block_mask.dims()[3],
                      (seqlen_k + 127) / 128,
                      common::errors::InvalidArgument(
                          "blockmask only supports blockdim_k = 128 now"));

    PADDLE_ENFORCE_EQ(
        block_mask.dims()[1],
        startend_row_indices.dims()[1],
        common::errors::InvalidArgument(
            "blockmask only supports same dim num_heads with flashmask now"));

    PADDLE_ENFORCE_LE(seqlen_k,
                      1024 * 128,
                      common::errors::InvalidArgument(
                          "blockmask only supports seqlen <= 128k in bwd now"));

    PADDLE_ENFORCE_LE(seqlen_q,
                      1024 * 128,
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
        seqused_q.dtype(),
        paddle::DataType::INT32,
        common::errors::InvalidArgument("seqused_q must have dtype int32"));
    CHECK_DEVICE(seqused_q);
    CHECK_CONTIGUOUS(seqused_q);
    CHECK_SHAPE(seqused_q, batch_size);
  }
  if (seqused_k_.is_initialized()) {
    auto seqused_k = seqused_k_.get();
    PADDLE_ENFORCE_EQ(
        seqused_k.dtype(),
        paddle::DataType::INT32,
        common::errors::InvalidArgument("seqused_k must have dtype int32"));
    CHECK_DEVICE(seqused_k);
    CHECK_CONTIGUOUS(seqused_k);
    CHECK_SHAPE(seqused_k, batch_size);
  }

  if (dq_.is_initialized()) {
    *dq = dq_.get();
    PADDLE_ENFORCE_EQ(
        dq->dtype(),
        q_type,
        common::errors::InvalidArgument("dq must have the same dtype as q"));
    CHECK_DEVICE((*dq));
    PADDLE_ENFORCE_EQ(dq->strides()[dq->strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "dq must have contiguous last dimension"));
    if (!is_varlen_q) {
      CHECK_SHAPE((*dq), batch_size, seqlen_q, num_heads, head_size);
    } else {
      CHECK_SHAPE((*dq), total_q, num_heads, head_size);
    }
  } else {
    // *dq = EmptyLike<T, Context>(dev_ctx, q);
    *dq = paddle::empty_like(q);
  }
  if (dk_.is_initialized()) {
    *dk = dk_.get();
    PADDLE_ENFORCE_EQ(
        dk->dtype(),
        q_type,
        common::errors::InvalidArgument("dk must have the same dtype as q"));
    CHECK_DEVICE((*dk));
    PADDLE_ENFORCE_EQ(dk->strides()[dk->strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "dk must have contiguous last dimension"));
    if (!is_varlen_k) {
      CHECK_SHAPE((*dk), batch_size, seqlen_k, num_heads_k, head_size);
    } else {
      CHECK_SHAPE((*dk), total_k, num_heads_k, head_size);
    }
  } else {
    // *dk = EmptyLike<T, Context>(dev_ctx, k);
    *dk = paddle::empty_like(k);
  }
  if (dv_.is_initialized()) {
    *dv = dv_.get();
    PADDLE_ENFORCE_EQ(
        dv->dtype(),
        q_type,
        common::errors::InvalidArgument("dv must have the same dtype as q"));
    CHECK_DEVICE((*dv));
    PADDLE_ENFORCE_EQ(dv->strides()[dv->strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "dv must have contiguous last dimension"));
    if (!is_varlen_k) {
      CHECK_SHAPE((*dv), batch_size, seqlen_k, num_heads_k, head_size);
    } else {
      CHECK_SHAPE((*dv), total_k, num_heads_k, head_size);
    }
  } else {
    // *dv = EmptyLike<T, Context>(dev_ctx, v);
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
      // softmax_d->Resize(
      //     common::make_ddim({batch_size, num_heads, seqlen_q_rounded}));
      *softmax_d = paddle::empty(
        {batch_size, num_heads, seqlen_q_rounded},
        paddle::DataType::FLOAT32,
        q.place());
    }
    if (softmax_lse_log2) {
      // softmax_lse_log2->Resize(
      //     common::make_ddim({batch_size, num_heads, seqlen_q_rounded}));
      *softmax_lse_log2 = paddle::empty(
        {batch_size, num_heads, seqlen_q_rounded},
        paddle::DataType::FLOAT32,
        q.place());
    }
  } else {
    if (softmax_d) {
      // softmax_d->Resize(common::make_ddim({num_heads, total_q_padded_rounded}));
      *softmax_d = paddle::empty(
        {num_heads, total_q_padded_rounded},
        paddle::DataType::FLOAT32,
        q.place());

    }
    if (softmax_lse_log2) {
      // softmax_lse_log2->Resize(
      //     common::make_ddim({num_heads, total_q_padded_rounded}));
      *softmax_lse_log2 = paddle::empty(
        {num_heads, total_q_padded_rounded},
        paddle::DataType::FLOAT32,
        q.place());
    }
  }
  // if (softmax_d) {
  //   dev_ctx.template Alloc<float>(softmax_d);
  // }
  // if (softmax_lse_log2) {
  //   dev_ctx.template Alloc<float>(softmax_lse_log2);
  // }

  if (dq_accum) {
    if (!is_varlen) {
      // dq_accum->Resize(common::make_ddim(
      //     {batch_size, num_heads, seqlen_q_rounded * head_size_rounded}));
      *dq_accum = paddle::empty(
        {batch_size, num_heads, seqlen_q_rounded * head_size_rounded},
        paddle::DataType::FLOAT32,
        q.place());

    } else {
      // dq_accum->Resize(common::make_ddim(
      //     {num_heads, total_q_padded_rounded * head_size_rounded}));
      *dq_accum = paddle::empty(
        {num_heads, total_q_padded_rounded * head_size_rounded},
        paddle::DataType::FLOAT32,
        q.place());

    }
    // dev_ctx.template Alloc<float>(dq_accum);
  }

  if (num_heads_k != num_heads) {  // MQA / GQA
    if (!is_varlen) {
      if (dk_accum) {
        // dk_accum->Resize(common::make_ddim(
        //     {batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}));
        *dk_accum = paddle::empty(
          {batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded},
          paddle::DataType::FLOAT32,
          q.place());

      }
      if (dv_accum) {
        // dv_accum->Resize(common::make_ddim(
        //     {batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}));
        *dv_accum = paddle::empty(
          {batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded},
          paddle::DataType::FLOAT32,
          q.place());
      }
    } else {
      if (dk_accum) {
        // dk_accum->Resize(common::make_ddim(
        //     {num_heads_k, total_k_padded_rounded, head_size_rounded}));
        *dk_accum = paddle::empty(
          {num_heads_k, total_k_padded_rounded, head_size_rounded},
          paddle::DataType::FLOAT32,
          q.place());

      }
      if (dv_accum) {
        // dv_accum->Resize(common::make_ddim(
        //     {num_heads_k, total_k_padded_rounded, head_size_rounded}));
        *dv_accum = paddle::empty(
          {num_heads_k, total_k_padded_rounded, head_size_rounded},
          paddle::DataType::FLOAT32,
          q.place());

      }
    }
    // if (dk_accum) {
    //   dev_ctx.template Alloc<float>(dk_accum);
    // }
    // if (dv_accum) {
    //   dev_ctx.template Alloc<float>(dv_accum);
    // }


    // funcs::SetConstant<Context, float> set_zero;
    if (dk_accum) {
      // set_zero(dev_ctx, dk_accum, float{0});
      paddle::experimental::fill(*dk_accum, float{0});
    }
    if (dv_accum) {
      // set_zero(dev_ctx, dv_accum, float{0});
      paddle::experimental::fill(*dv_accum, float{0});
    }
  }

  FlashMask_bwd_params *params_handle = get_flashmask_bwd_params_handle();
  flashmaskv2_clear_bwd_params_handle(params_handle);
  set_flashmaskv2_params_dgrad(
      params_handle,
      batch_size,
      seqlen_q,
      seqlen_k,
      seqlen_q_rounded,
      seqlen_k_rounded,
      num_heads,
      num_heads_k,
      head_size,
      head_size_rounded,
      q,
      k,
      v,
      out,
      dout,
      dq,
      dk,
      dv,
      !is_varlen_q ? nullptr : cu_seqlens_q.data(),
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
      /*p_dropout=*/0.f,
      softmax_scale,
      window_size_left,
      window_size_right,
      dprops,
      softcap,
      deterministic,
      sm_margin);
  flashmaskv2_bwd_params_set_total_q(params_handle, total_q);
  flashmaskv2_bwd_params_set_total_k(params_handle, total_k);
  flashmaskv2_bwd_params_set_softmax_lse_log2_ptr(
      params_handle, softmax_lse_log2 ? softmax_lse_log2->data() : nullptr);
  flashmaskv2_bwd_params_set_dv(
      params_handle,
      head_size);  // We don't support hdim_v being
                   // different from hdim_qk for now
  paddle::Tensor tile_count_semaphore;
  if (arch >= 90) {
    // tile_count_semaphore = phi::Full<int32_t, Context>(dev_ctx, {1}, 0);
    tile_count_semaphore = paddle::full(
        {1}, 0, paddle::DataType::INT32, q.place());

    flashmaskv2_bwd_params_set_tile_count_semaphore(
        params_handle, tile_count_semaphore.data<int>());
  } else {
    flashmaskv2_bwd_params_set_tile_count_semaphore(params_handle,
                                                                  nullptr);
  }


  paddle::Tensor dq_semaphore = paddle::empty(
    {(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads},
    paddle::DataType::INT32,
    q.place());
  flashmaskv2_bwd_params_set_dq_semaphore(params_handle,
                                                  dq_semaphore.data<int>());

  paddle::Tensor dk_semaphore;
  paddle::Tensor dv_semaphore;
  if (num_heads_k != num_heads &&
      flashmaskv2_bwd_params_get_deterministic(params_handle)) {
    // xiangrui: we need to zero them out
    // funcs::SetConstant<Context, int32_t> set_zero_dk;
    // set_zero_dk(dev_ctx, &dk_semaphore, static_cast<int32_t>(0));
    // funcs::SetConstant<Context, int32_t> set_zero_dv;
    // set_zero_dv(dev_ctx, &dv_semaphore, static_cast<int32_t>(0));
    dk_semaphore = paddle::zeros(
        {(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k},
        paddle::DataType::INT32,
        q.place());

    dv_semaphore = paddle::zeros(
        {(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k},
        paddle::DataType::INT32,
        q.place());
    
    flashmaskv2_bwd_params_set_dk_semaphore(params_handle,
                                                     dk_semaphore.data<int>());
    flashmaskv2_bwd_params_set_dv_semaphore(params_handle,
                                                     dv_semaphore.data<int>());
  }

  if (is_flashmask) {
    // [BQW_CHANGE] 添加 const_cast，setter 函数接受 int32_t* (非 const)
    flashmaskv2_bwd_params_set_lt_start_ptr(params_handle, const_cast<int32_t*>(lt_start_ptr));
    flashmaskv2_bwd_params_set_lt_end_ptr(params_handle, const_cast<int32_t*>(lt_end_ptr));
    flashmaskv2_bwd_params_set_ut_start_ptr(params_handle, const_cast<int32_t*>(ut_start_ptr));
    flashmaskv2_bwd_params_set_ut_end_ptr(params_handle, const_cast<int32_t*>(ut_end_ptr));

    if (flashmask_maxmin.initialized())
      flashmaskv2_bwd_params_set_flashmask_maxmin_ptr(
          params_handle, (flashmask_maxmin.data<int32_t>()));
    else
      flashmaskv2_bwd_params_set_flashmask_maxmin_ptr(params_handle,
                                                               nullptr);

    flashmaskv2_bwd_params_set_h_flashmask(
        params_handle, startend_row_indices.dims()[1]);
    flashmaskv2_bwd_params_set_h_h_flashmask_ratio(
        params_handle, num_heads / startend_row_indices.dims()[1]);
  } else {
    flashmaskv2_bwd_params_set_lt_start_ptr(params_handle, nullptr);
    flashmaskv2_bwd_params_set_lt_end_ptr(params_handle, nullptr);
    flashmaskv2_bwd_params_set_ut_start_ptr(params_handle, nullptr);
    flashmaskv2_bwd_params_set_ut_end_ptr(params_handle, nullptr);
    flashmaskv2_bwd_params_set_flashmask_maxmin_ptr(params_handle,
                                                             nullptr);
    flashmaskv2_bwd_params_set_h_flashmask(params_handle, 0);
    flashmaskv2_bwd_params_set_h_h_flashmask_ratio(params_handle, 0);
  }

  if (is_blockmask) {
    // xhy: blockmask is now only support blockdim_q k = 128
    flashmaskv2_bwd_params_set_m_block_dim(params_handle, 128);
    flashmaskv2_bwd_params_set_n_block_dim(params_handle, 128);
    flashmaskv2_bwd_params_set_block_mask_ptr(
        params_handle, (block_mask.data<int32_t>()));
  }
#ifdef FLASHATTENTION_DISABLE_LOCAL
  PADDLE_ENABLE_EQ(
      !flashmaskv2_bwd_params_get_is_local(params_handle),
      true,
      "This flash attention build does not support local attention.");
#endif
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  PADDLE_ENABLE_EQ(
      flashmaskv2_bwd_params_get_softcap(params_handle),
      0.0,
      "This flash attention build does not support tanh softcapping.");
#endif

  if (total_q > 0 && total_k > 0 && num_heads_k > 0) {
    flashmaskv2_run_mha_bwd(params_handle, stream);
  } else if (total_k > 0 && num_heads_k > 0) {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the output
    // to 0.
    // funcs::SetConstant<Context, T> set_zero;
    // set_zero(dev_ctx, dk, T{0});
    // set_zero(dev_ctx, dv, T{0});
    paddle::experimental::fill(*dk, T{0});
    paddle::experimental::fill(*dv, T{0});
    if (softmax_d) {
      // funcs::SetConstant<Context, float> set_zero_fp32;
      // set_zero_fp32(dev_ctx, softmax_d, float{0});
      paddle::experimental::fill(*softmax_d, float{0});
    }
  } else if (total_q > 0 && num_heads_k > 0) {
    // funcs::SetConstant<Context, T> set_zero;
    // set_zero(dev_ctx, dq, T{0});
    paddle::experimental::fill(*dq, T{0});
    if (softmax_d) {
      // funcs::SetConstant<Context, float> set_zero_fp32;
      // set_zero_fp32(dev_ctx, softmax_d, float{0});
      paddle::experimental::fill(*softmax_d, float{0});
    }
  }
#else
  RaiseNotSupportedError();
#endif
}

#endif






// [BQW_CHANGE] phi::dtype::bfloat16 → paddle::bfloat16 (外迁后不直接使用 phi 命名空间)
template void FlashMaskV2GradBaseKernel<paddle::bfloat16>(
    const paddle::Tensor &dout,
    const paddle::Tensor &q,
    const paddle::Tensor &k,
    const paddle::Tensor &v,
    const paddle::Tensor &out,
    const paddle::Tensor &softmax_lse,
    const paddle::optional<paddle::Tensor> &dq_,
    const paddle::optional<paddle::Tensor> &dk_,
    const paddle::optional<paddle::Tensor> &dv_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_q_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_,
    const paddle::optional<paddle::Tensor> &seqused_q_,
    const paddle::optional<paddle::Tensor> &seqused_k_,
    const paddle::optional<paddle::Tensor> &startend_row_indices_,
    const paddle::optional<paddle::Tensor> &block_mask_,
    int max_seqlen_q_,
    int max_seqlen_k_,
    float const softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float const softcap,
    bool const deterministic,
    int const sm_margin,
    paddle::Tensor *dq,
    paddle::Tensor *dk,
    paddle::Tensor *dv,
    paddle::Tensor *softmax_d,
    paddle::Tensor *softmax_lse_log2,
    paddle::Tensor *dq_accum,
    paddle::Tensor *dk_accum,
    paddle::Tensor *dv_accum);

#endif
