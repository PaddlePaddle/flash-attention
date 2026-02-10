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
void FlashMaskV2BaseKernel(
    const paddle::Tensor &q,
    const paddle::Tensor &k,
    const paddle::Tensor &v,
    const paddle::optional<paddle::Tensor>
        &k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is
                  // cu_seqlens_k_new
    const paddle::optional<paddle::Tensor>
        &v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is
                  // cu_seqlens_k_new
    const paddle::optional<paddle::Tensor> &q_v_,  // (b, s_q, h, dv) or (total_q_new, h,
                                        // dv) if there is cu_seqlens_q
    const paddle::optional<paddle::Tensor>
        &out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    const paddle::optional<paddle::Tensor> &cu_seqlens_q_,      // b+1
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_,      // b+1
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_new_,  // b+1
    const paddle::optional<paddle::Tensor>
        &seqused_q_,  // b. If given, only this many elements of each batch
                      // element's queries and outputs are used.
    const paddle::optional<paddle::Tensor>
        &seqused_k_,  // b. If given, only this many elements of each batch
                      // element's keys are used.
    const paddle::optional<paddle::Tensor> &page_table_,  // (b_k, max_num_pages_per_seq)
    const paddle::optional<paddle::Tensor>
        &kv_batch_idx_,  // b. indices to index into the KV cache
    const paddle::optional<paddle::Tensor> &leftpad_k_,   // b
    const paddle::optional<paddle::Tensor> &rotary_cos_,  // seqlen_ro x (rotary_dim / 2)
    const paddle::optional<paddle::Tensor> &rotary_sin_,  // seqlen_ro x (rotary_dim / 2)
    const paddle::optional<paddle::Tensor> &q_descale_,   // (b, h_k), not (b, h)
    const paddle::optional<paddle::Tensor> &k_descale_,   // (b, h_k)
    const paddle::optional<paddle::Tensor> &v_descale_,   // (b, h_k)
    const paddle::optional<paddle::Tensor> &scheduler_metadata_,    // (b + 1)
    const paddle::optional<paddle::Tensor> &startend_row_indices_,  // （b,h,s_1,[1,2,4])
    const paddle::optional<paddle::Tensor> &block_mask_,  // （(b,h,s// 128,s // 128)
    const int
        max_seqlen_q_,  // if max_seqlen_q_ is set to 0, it indicates that it is
                        // uninitialized and should not be referenced
    // TODO(tridao): check if we need max_seqlen_k
    const int
        max_seqlen_k_,  // if max_seqlen_q_ is set to 0, it indicates that it is
                        // uninitialized and should not be referenced
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool is_rotary_interleaved,  // if true, rotary combines indices 0 &
                                       // 1, else indices 0 & rotary_dim / 2
    int num_splits,
    const bool manual_set_pack_gqa,
    const bool
        pack_gqa_,  // the pack_gqa_ will be used only if manual_set_pack_gqa is
                    // set to True; otherwise, the internal heuristic
                    // get_pack_gqa() from fa3 will decide whether to pack gqa
    const int sm_margin,
    paddle::Tensor *out,
    paddle::Tensor *softmax_lse,
    paddle::Tensor *out_accum,
    paddle::Tensor *softmax_lse_accum)

#if 0
  {
    printf("now in FlashMaskV2BaseKernel\n");
  } 
#else   
{
#ifdef PADDLE_WITH_FLASHATTN_V3
  printf("now in FlashMaskV2BaseKernel\n");

  // [BQW_CHANGE] 使用 tensor 关联的 CUDA stream，而非创建新的独立 stream
  // 创建独立 stream 会导致与 Paddle 执行 stream 不同步的问题
  cudaStream_t stream = static_cast<cudaStream_t>(q.stream());

  // TODO(umiswing): support ampere
  // int device_id = dev_ctx.GetPlace().GetDeviceId();
  int device_id = q.place().GetDeviceId();
  // [BQW_CHANGE] 使用 CUDA API 获取设备属性，替代 Paddle 内部 API
  // paddle::platform::GetDeviceProperties 是 Paddle 内部函数
  cudaDeviceProp dprops;
  cudaGetDeviceProperties(&dprops, device_id);
  const bool is_sm90 = dprops.major == 9 && dprops.minor == 0;
  PADDLE_ENFORCE_EQ(is_sm90,
                    true,
                    common::errors::Unavailable(
                        "FlashAttention-3 only supports Hopper GPUs."));

  auto q_type = q.dtype();
  PADDLE_ENFORCE_EQ(
      (q_type == paddle::DataType::FLOAT16 || q_type == paddle::DataType::BFLOAT16 ||
       q_type == paddle::DataType::FLOAT8_E4M3FN),
      true,
      common::errors::InvalidArgument(
          "FlashAttention-3 only supports fp16, bf16, and fp8_e4m3 data type"));

  PADDLE_ENFORCE_EQ(k.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and key must have the same dtype"));
  PADDLE_ENFORCE_EQ(v.dtype(),
                    q_type,
                    common::errors::InvalidArgument(
                        "query and value must have the same dtype"));

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

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

  paddle::Tensor page_table;
  // const bool paged_KV = page_table_.has_value();
  // umiswing: this is stupid but idk how to use optional
  const bool paged_KV = page_table_.is_initialized();
  if (paged_KV) {
    page_table = page_table_.get();
    CHECK_DEVICE(page_table);
    PADDLE_ENFORCE_EQ(page_table.dtype(),
                      paddle::DataType::INT32,
                      common::errors::InvalidArgument(
                          "page_table must have dtype paddle.int32"));
    PADDLE_ENFORCE_EQ(page_table.strides()[page_table.strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "page_table must have contiguous last dimension"));
  }

  // TODO(umiswing): support cusum

  paddle::Tensor cu_seqlens_q;
  // bool const is_varlen_q = cu_seqlens_q_.has_value();
  // TODO(umiswing): this is stupid, must fix it (after understand
  // optional)
  const bool is_varlen_q = cu_seqlens_q_.is_initialized();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.get();
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_q);
    PADDLE_ENFORCE_EQ(cu_seqlens_q.dtype(),
                      paddle::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_q must have dtype paddle.int32"));
    PADDLE_ENFORCE_NE(
        max_seqlen_q_,
        0,
        common::errors::InvalidArgument(
            "max_seqlen_q must be provided if cu_seqlens_q is provided"));
  }

  paddle::Tensor cu_seqlens_k;
  const bool is_varlen_k = cu_seqlens_k_.is_initialized();
  if (is_varlen_k) {
    cu_seqlens_k = cu_seqlens_k_.get();
    CHECK_DEVICE(cu_seqlens_k);
    CHECK_CONTIGUOUS(cu_seqlens_k);
    PADDLE_ENFORCE_EQ(cu_seqlens_k.dtype(),
                      paddle::DataType::INT32,
                      common::errors::InvalidArgument(
                          "cu_seqlens_k must have dtype paddle.int32"));
    PADDLE_ENFORCE_NE(
        max_seqlen_k_,
        0,
        common::errors::InvalidArgument(
            "max_seqlen_k must be provided if cu_seqlens_k is provided"));
    PADDLE_ENFORCE_EQ(
        !paged_KV,
        true,
        common::errors::InvalidArgument(
            "If cu_seqlens_k is passed in, then page table is not supported"));
    PADDLE_ENFORCE_EQ(
        !kv_batch_idx_,
        true,
        common::errors::InvalidArgument(
            "If cu_seqlens_k is passed in, then page table is not supported"));
  }

  auto const sizes = q.dims();
  const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.dims()[0] - 1;
  int seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_;
  int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];
  int64_t num_heads = q.dims()[q.dims().size() - 2];
  int64_t const head_size = q.dims()[q.dims().size() - 1];
  int const head_size_v = v.dims()[v.dims().size() - 1];
  int const max_num_pages_per_seq = !paged_KV ? 0 : page_table.dims()[1];
  int const num_pages = !paged_KV ? 0 : k.dims()[0];
  int const page_size = !paged_KV ? 1 : k.dims()[1];
  int const seqlen_k =
      !is_varlen_k
          ? (!paged_KV ? k.dims()[1] : max_num_pages_per_seq * page_size)
          : max_seqlen_k_;
  int const total_k = !is_varlen_k ? batch_size * k.dims()[1] : k.dims()[0];
  int const num_heads_k = k.dims()[k.dims().size() - 2];
  int const batch_size_k =
      !paged_KV ? (!is_varlen_k ? k.dims()[0] : cu_seqlens_k.dims()[0] - 1)
                : page_table.dims()[0];
  if (!kv_batch_idx_.is_initialized()) {
    PADDLE_ENFORCE_EQ(batch_size,
                      batch_size_k,
                      common::errors::InvalidArgument(
                          "batch_size must be equal to batch_size_k"));
  }
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
  if (head_size_v != head_size) {
    PADDLE_ENFORCE_EQ(
        ((head_size > 128 && head_size <= 192 && head_size_v > 96 &&
          head_size_v <= 128) ||
         (head_size <= 64 && head_size_v <= 512)),
        true,
        common::errors::InvalidArgument(
            "If V headdim is different from Q/K dim, we only support "
            "Q/K headdim in (128, 192] and V headdim in (96, 128], "
            "or (Q/K <= 64 and V <= 512)."));
    PADDLE_ENFORCE_EQ(dprops.major,
                      9,
                      common::errors::InvalidArgument(
                          "Only Hopper supports different V headdim"));
    if (head_size_v > 256) {
      PADDLE_ENFORCE_EQ((q_type == paddle::DataType::FLOAT16 ||
                         q_type == paddle::DataType::BFLOAT16),
                        true,
                        common::errors::InvalidArgument(
                            "HeaddimV > 256 requires fp16 and bf16 data type"));
    }
  }

  bool const is_flashmask = startend_row_indices_.is_initialized();
  bool const is_blockmask = block_mask_.is_initialized();

  // This needs to go before kBlockM & kBlockN since we rely on the correct
  // window_size and is_causal to set kBlockM
  // TODO(tridao): check this
  if (window_size_left >= seqlen_k - 1) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_q - 1) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1) {
    // Special case of hdim 128 where we want causal to have kBlockN=128, better
    // for pagedKV and TMA
    if (((head_size <= 64 || head_size > 128) || !paged_KV) && !is_flashmask) {
      is_causal = false;
    }
  }
  if (is_causal) {
    window_size_right = 0;
  }
  // There's a case where is_causal=false, window_size=(-1, 0). Then
  // set_params_fprop will set params.is_causal=true. If we don't have is_causal
  // here matching params.is_causal, we might get the wrong kBlockM.
  is_causal = window_size_left < 0 && window_size_right == 0;

  if (!is_varlen_q) {
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  } else {
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  }
  if (!paged_KV) {
    if (!is_varlen_k) {
      CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
      CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
    } else {
      CHECK_SHAPE(k, total_k, num_heads_k, head_size);
      CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    }
  } else {
    CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
    CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
    CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
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

  if (leftpad_k_.is_initialized()) {
    auto leftpad_k = leftpad_k_.get();
    PADDLE_ENFORCE_EQ(
        leftpad_k.dtype(),
        paddle::DataType::INT32,
        common::errors::InvalidArgument("leftpad_k must have dtype int32"));
    CHECK_DEVICE(leftpad_k);
    CHECK_CONTIGUOUS(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  // This is what we will template on
  bool const is_varlen =
      is_varlen_q || is_varlen_k || seqused_q_.is_initialized() ||
      seqused_k_.is_initialized() || leftpad_k_.is_initialized();
#ifdef FLASHATTENTION_DISABLE_VARLEN
  PADDLE_ENFORCE_EQ(!is_varlen,
                    true,
                    common::errors::Unavailable(
                        "This flash attention build does not support varlen."));
#endif

  int const alignment = q_type == paddle::DataType::FLOAT8_E4M3FN ? 16 : 8;
  PADDLE_ENFORCE_EQ(head_size % alignment,
                    0,
                    common::errors::InvalidArgument(
                        "head_size should be a multiple of %d", alignment));
  PADDLE_ENFORCE_EQ(head_size_v % alignment,
                    0,
                    common::errors::InvalidArgument(
                        "head_size_v should be a multiple of %d", alignment));

  auto out_type =
      q_type == paddle::DataType::FLOAT8_E4M3FN ? paddle::DataType::BFLOAT16 : q_type;
  if (out_.is_initialized()) {
    *out = out_.get();
    PADDLE_ENFORCE_EQ(
        out->dtype(),
        out_type,
        common::errors::InvalidArgument(
            "For FP16/BF16 input, output must have the same dtype as "
            "inputs. For FP8 input, output must have dtype BF16"));
    CHECK_DEVICE((*out));
    PADDLE_ENFORCE_EQ(out->strides()[out->strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "Output tensor must have contiguous last dimension"));
    if (!is_varlen_q) {
      CHECK_SHAPE((*out), batch_size, seqlen_q, num_heads, head_size_v);
    } else {
      CHECK_SHAPE((*out), total_q, num_heads, head_size_v);
    }
  } else {
    // TODO 明确一下，q type是不是和 t 是一样的
    auto out_type = q_type == paddle::DataType::FLOAT8_E4M3FN ? paddle::DataType::BFLOAT16 : q_type;

    if (!is_varlen_q) {
      // out->Resize(
      //     common::make_ddim({batch_size, seqlen_q, num_heads, head_size_v}));
      *out = paddle::empty(
        {batch_size, seqlen_q, num_heads, head_size_v},
        out_type,
        q.place());

    } else {
      // out->Resize(common::make_ddim({total_q, num_heads, head_size_v}));
      *out = paddle::empty(
        {total_q, num_heads, head_size_v},
        out_type,
        q.place());
    }
    // if (q_type == paddle::DataType::FLOAT8_E4M3FN) {
    //   dev_ctx.template Alloc<phi::bfloat16>(out);
    // } else {
    //   // umiswing: assuming T is Input Type
    //   dev_ctx.template Alloc<T>(out);
    // }
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const head_size_rounded = flashmaskv2_round_up_headdim(head_size);
  int const head_size_v_rounded = flashmaskv2_round_up_headdim(head_size_v);
  int const seqlen_q_rounded = round_multiple(seqlen_q, 128);
  int const seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // if (!is_varlen_q) {
  //   softmax_lse->Resize(common::make_ddim({batch_size, num_heads, seqlen_q}));
  // } else {
  //   softmax_lse->Resize(common::make_ddim({num_heads, total_q}));
  // }
  // dev_ctx.template Alloc<float>(softmax_lse);

  *softmax_lse = paddle::empty(
    {batch_size, num_heads, seqlen_q},
    paddle::DataType::FLOAT32,
    q.place());

  FlashMask_fwd_params *params_handle = get_flashmask_fwd_params_handle();
  flashmaskv2_clear_fwd_params_handle(params_handle);
  set_flashmaskv2_params_fprop(
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
      !is_varlen_q ? nullptr : cu_seqlens_q.data(),
      !is_varlen_k ? nullptr : cu_seqlens_k.data(),
      seqused_q_.is_initialized() ? const_cast<void *>(seqused_q_.get().data())
                                  : nullptr,
      seqused_k_.is_initialized() ? const_cast<void *>(seqused_k_.get().data())
                                  : nullptr,
      softmax_lse->data(),
      /*p_dropout=*/0.f,
      softmax_scale,
      window_size_left,
      window_size_right,
      dprops,
      softcap,
      sm_margin);
  flashmaskv2_fwd_params_set_total_q(params_handle, total_q);
  flashmaskv2_fwd_params_set_total_k(params_handle, total_k);
  flashmaskv2_fwd_params_set_b_k(params_handle, batch_size_k);
  flashmaskv2_fwd_params_set_dv(params_handle, head_size_v);
  flashmaskv2_fwd_params_set_dv_rounded(params_handle,
                                                      head_size_v_rounded);

  if (leftpad_k_
          .is_initialized()) {  // This needs to be set before get_pagedkv_tma
    flashmaskv2_fwd_params_set_leftpad_k(
        params_handle, leftpad_k_.get().data<int>());
  }
  if (paged_KV) {
    flashmaskv2_fwd_params_set_page_table(params_handle,
                                                        page_table.data<int>());
    flashmaskv2_fwd_params_set_page_table_batch_stride(
        params_handle, page_table.strides()[0]);
  }
  flashmaskv2_fwd_params_set_page_size(params_handle, page_size);
  flashmaskv2_fwd_params_set_num_pages(params_handle, num_pages);

  if (k_new_.is_initialized()) {  // This needs to be set before get_pagedkv_tma
    paddle::Tensor k_new, v_new;
    PADDLE_ENFORCE_EQ(
        v_new_.is_initialized(),
        true,
        common::errors::InvalidArgument(
            "If k_new is supplied, v_new must also be passed in"));
    PADDLE_ENFORCE_EQ(
        seqused_k_.is_initialized(),
        true,
        common::errors::InvalidArgument(
            "If k_new is supplied, seqlens_k must also be passed in"));
    PADDLE_ENFORCE_LE(
        seqlen_q,
        seqlen_k,
        common::errors::InvalidArgument(
            "If k_new is supplied, it must have seqlen <= the seqlen "
            "of the KV cache"));
    paddle::Tensor cu_seqlens_k_new;
    bool const is_varlen_k_new = cu_seqlens_k_new_.is_initialized();
    if (is_varlen_k_new) {
      cu_seqlens_k_new = cu_seqlens_k_new_.get();
      CHECK_DEVICE(cu_seqlens_k_new);
      CHECK_CONTIGUOUS(cu_seqlens_k_new);
      PADDLE_ENFORCE_EQ(cu_seqlens_k_new.dtype(),
                        paddle::DataType::INT32,
                        common::errors::InvalidArgument(
                            "cu_seqlens_k_new must have dtype paddle.int32"));
    }
    k_new = k_new_.get();
    v_new = v_new_.get();
    PADDLE_ENFORCE_EQ(k_new.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "k_new must have the same dtype as query"));
    PADDLE_ENFORCE_EQ(v_new.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "v_new must have the same dtype as query"));
    CHECK_DEVICE(k_new);
    CHECK_DEVICE(v_new);
    PADDLE_ENFORCE_EQ(k_new.strides()[k_new.strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "k_new tensor must have contiguous last dimension"));
    PADDLE_ENFORCE_EQ(v_new.strides()[v_new.strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "v_new tensor must have contiguous last dimension"));
    // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when
    // is_varlen_k_new
    int seqlen_k_new = !is_varlen_k_new ? k_new.dims()[1] : 0;
    int total_k_new =
        !is_varlen_k_new ? batch_size * k_new.dims()[1] : k_new.dims()[0];
    if (!is_varlen_k_new) {
      CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
      CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size_v);
    } else {
      CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
      CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
    }
    // umiswing: dump this to shared library
    flashmaskv2_fwd_params_set_seqlen_knew(params_handle,
                                                         seqlen_k_new);
    flashmaskv2_fwd_params_set_total_knew(params_handle,
                                                        total_k_new);
    flashmaskv2_fwd_params_set_knew_ptr(params_handle,
                                                      (k_new.data()));
    flashmaskv2_fwd_params_set_vnew_ptr(params_handle,
                                                      (v_new.data()));
    // All stride are in elements, not bytes.
    flashmaskv2_fwd_params_set_knew_row_stride(
        params_handle, k_new.strides()[k_new.strides().size() - 3]);
    flashmaskv2_fwd_params_set_vnew_row_stride(
        params_handle, v_new.strides()[v_new.strides().size() - 3]);
    flashmaskv2_fwd_params_set_knew_head_stride(
        params_handle, k_new.strides()[k_new.strides().size() - 2]);
    flashmaskv2_fwd_params_set_vnew_head_stride(
        params_handle, v_new.strides()[v_new.strides().size() - 2]);
    if (!is_varlen_k_new) {
      flashmaskv2_fwd_params_set_knew_batch_stride(
          params_handle, k_new.strides()[0]);
      flashmaskv2_fwd_params_set_vnew_batch_stride(
          params_handle, v_new.strides()[0]);
    }
    if (is_varlen_k_new) {
      flashmaskv2_fwd_params_set_cu_seqlens_knew(
          params_handle, cu_seqlens_k_new.data<int>());
    }
  }

  // 992 = 32 * 31 is the max supported batch in prepare_varlen_num_blocks
  // kernel
  bool const use_dynamic_split =
      is_varlen &&
      flashmaskv2_fwd_params_get_b(params_handle) <= 992;
  // Temporarily set num_splits_dynamic_ptr to 1 since get_num_splits checks it
  flashmaskv2_fwd_params_set_num_splits_dynamic_ptr(
      params_handle, !use_dynamic_split ? nullptr : reinterpret_cast<int *>(1));

  flashmaskv2_fwd_params_set_pagedkv_tma(
      params_handle, flashmaskv2_get_pagedkv_tma(params_handle));
  if (num_splits <= 0) {
    num_splits = flashmaskv2_get_num_splits(params_handle);
  }
  flashmaskv2_fwd_params_set_num_splits(params_handle,
                                                      num_splits);

  // Always enable PackGQA for Split, and get_pack_gqa requires
  // params.num_splits to decide
  const bool pack_gqa =
      manual_set_pack_gqa
          ? pack_gqa_
          : flashmaskv2_get_pack_gqa(params_handle);
  flashmaskv2_fwd_params_set_pack_gqa(params_handle, pack_gqa);

  // This needs to be set after get_num_splits
  paddle::Tensor tile_count_semaphore;  // Contains the semaphore and optionally
                                     // num_splits_dynamic
  // We don't use the persistent scheduler if Split and not Varlen
  const bool params_is_causal =
      flashmaskv2_fwd_params_get_is_causal(params_handle);
  const bool params_is_local =
      flashmaskv2_fwd_params_get_is_local(params_handle);
  const int params_num_splits =
      flashmaskv2_fwd_params_get_num_splits(params_handle);
  const int params_b =
      flashmaskv2_fwd_params_get_b(params_handle);
  const int params_arch =
      flashmaskv2_fwd_params_get_arch(params_handle);
  bool const scheduler_needs_semaphore =
      params_arch >= 90 ? true
                        : ((params_is_causal && !is_varlen) ||
                           (is_varlen && params_num_splits > 1));
  int metadata_size = 0;
  if (scheduler_needs_semaphore || use_dynamic_split) {
    // int metadata_size = static_cast<int>(scheduler_needs_semaphore) +
    //                     static_cast<int>(use_dynamic_split) * params_b;
    metadata_size = static_cast<int>(scheduler_needs_semaphore) +
                        static_cast<int>(use_dynamic_split) * params_b;

    // phi::dynload::
    //     flashmaskv2_fwd_params_set_skip_scheduler_metadata_computation(
    //         params_handle, scheduler_metadata_.is_initialized());

    flashmaskv2_fwd_params_set_skip_scheduler_metadata_computation(
            params_handle, scheduler_metadata_.is_initialized());

    if (scheduler_metadata_.is_initialized()) {
      paddle::Tensor scheduler_metadata = scheduler_metadata_.get();
      CHECK_DEVICE(scheduler_metadata);
      CHECK_SHAPE(scheduler_metadata, metadata_size);
      CHECK_CONTIGUOUS(scheduler_metadata);
      PADDLE_ENFORCE_EQ(scheduler_metadata.dtype(),
                        paddle::DataType::INT32,
                        common::errors::InvalidArgument(
                            "scheduler_metadata must have dtype int32"));
      tile_count_semaphore = scheduler_metadata;
    } else {
      // tile_count_semaphore = Empty<int32_t>(dev_ctx, {metadata_size});
      tile_count_semaphore = paddle::empty({metadata_size}, paddle::DataType::INT32, q.place());

    }
    if (scheduler_needs_semaphore && !use_dynamic_split) {
      // funcs::SetConstant<Context, int32_t> set_zero;
      // set_zero(dev_ctx,
      //          &tile_count_semaphore,
      //          int32_t{0});  // If varlen we'll manually do the zero-ing

      cudaMemsetAsync(const_cast<int32_t*>(tile_count_semaphore.data<int>()),
                      0,
                      metadata_size * sizeof(int32_t),
                      stream);
    }
    flashmaskv2_fwd_params_set_tile_count_semaphore(
        params_handle,
        scheduler_needs_semaphore ? (tile_count_semaphore.data<int>())
                                  : nullptr);
    flashmaskv2_fwd_params_set_num_splits_dynamic_ptr(
        params_handle,
        use_dynamic_split ? (tile_count_semaphore.data<int>()) + 1 : nullptr);
  }

  if (q_v_.is_initialized()) {
    PADDLE_ENFORCE_LT(head_size,
                      64,
                      common::errors::InvalidArgument(
                          "q_v is only supported for head_size <= 64"));
    PADDLE_ENFORCE_EQ(
        (q_type == paddle::DataType::FLOAT16 || q_type == paddle::DataType::FLOAT16),
        true,
        common::errors::InvalidArgument(
            "q_v is only supported for fp16 and bf16 data type"));
    PADDLE_ENFORCE_EQ(params_arch,
                      90,
                      common::errors::InvalidArgument(
                          "q_v is only supported for Hopper GPUs"));
    paddle::Tensor q_v = q_v_.get();
    PADDLE_ENFORCE_EQ(q_v.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "q_v must have the same dtype as query"));
    CHECK_DEVICE(q_v);
    PADDLE_ENFORCE_EQ(q_v.strides()[q_v.strides().size() - 1],
                      1,
                      common::errors::InvalidArgument(
                          "q_v tensor must have contiguous last dimension"));
    if (!is_varlen_q) {
      CHECK_SHAPE(q_v, batch_size, seqlen_q, num_heads, head_size_v);
    } else {
      CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
    }
    flashmaskv2_fwd_params_set_qv_ptr(params_handle,
                                                    (q_v.data()));
    // All stride are in elements, not bytes.
    flashmaskv2_fwd_params_set_qv_row_stride(
        params_handle, q_v.strides()[q_v.strides().size() - 3]);
    flashmaskv2_fwd_params_set_qv_head_stride(
        params_handle, q_v.strides()[q_v.strides().size() - 2]);
    if (!is_varlen_q) {
      flashmaskv2_fwd_params_set_qv_batch_stride(
          params_handle, q_v.strides()[0]);
    }
  }

  if (rotary_cos_.is_initialized()) {
    PADDLE_ENFORCE_EQ(
        k_new_.is_initialized(),
        true,
        common::errors::InvalidArgument(
            "If rotary cos/sin are provided, new key / value to be "
            "appended to KV cache must also be provided"));
    paddle::Tensor rotary_cos = rotary_cos_.get();
    CHECK_DEVICE(rotary_cos);
    CHECK_CONTIGUOUS(rotary_cos);
    int params_rotary_dim = rotary_cos.dims()[1] * 2;
    flashmaskv2_fwd_params_set_rotary_dim(params_handle,
                                                        params_rotary_dim);
    PADDLE_ENFORCE_LE(
        params_rotary_dim,
        head_size,
        common::errors::InvalidArgument("rotary_dim must be <= headdim"));
    PADDLE_ENFORCE_EQ(
        params_rotary_dim % 16,
        0,
        common::errors::InvalidArgument(
            "Only rotary dimensions divisible by 16 are currently supported"));
    // TODO(large-tensor): downstream functors may still use int; guard until
    // upgraded.
    int64_t seqlen_ro = rotary_cos.dims()[0];

    if (paged_KV) {
      PADDLE_ENFORCE_GE(
          seqlen_ro,
          seqlen_k,
          common::errors::InvalidArgument(
              "cos/sin seqlen must be at least the seqlen of KV cache"));
    }
    CHECK_SHAPE(rotary_cos, seqlen_ro, params_rotary_dim / 2);
    PADDLE_ENFORCE_EQ(rotary_cos.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "rotary_cos must have the same dtype as query"));

    PADDLE_ENFORCE_EQ(
        rotary_sin_.is_initialized(),
        true,
        common::errors::InvalidArgument(
            "If rotary cos is provided, rotary sin must also be provided"));
    auto rotary_sin = rotary_sin_.get();
    CHECK_DEVICE(rotary_sin);
    CHECK_CONTIGUOUS(rotary_sin);
    CHECK_SHAPE(rotary_sin, seqlen_ro, params_rotary_dim / 2);
    PADDLE_ENFORCE_EQ(rotary_sin.dtype(),
                      q_type,
                      common::errors::InvalidArgument(
                          "rotary_cos must have the same dtype as query"));

    flashmaskv2_fwd_params_set_rotary_cos_ptr(
        params_handle, (rotary_cos.data()));
    flashmaskv2_fwd_params_set_rotary_sin_ptr(
        params_handle, (rotary_sin.data()));
    flashmaskv2_fwd_params_set_is_rotary_interleaved(
        params_handle, is_rotary_interleaved);
  } else {
    flashmaskv2_fwd_params_set_rotary_dim(params_handle, 0);
  }

  if (kv_batch_idx_.is_initialized()) {
    paddle::Tensor kv_batch_idx = kv_batch_idx_.get();
    CHECK_DEVICE(kv_batch_idx);
    CHECK_CONTIGUOUS(kv_batch_idx);
    PADDLE_ENFORCE_EQ(
        kv_batch_idx.dtype(),
        paddle::DataType::INT32,
        common::errors::InvalidArgument("kv_batch_idx must have dtype int32"));
    flashmaskv2_fwd_params_set_kv_batch_idx(
        params_handle, reinterpret_cast<int *>(kv_batch_idx.data()));
  }

  if (flashmaskv2_fwd_params_get_num_splits(params_handle) > 1) {
    PADDLE_ENFORCE_LE(
        flashmaskv2_fwd_params_get_num_splits(params_handle),
        256,
        common::errors::InvalidArgument("num_splits > 256 not supported"));
    if (!is_varlen_q) {
      // out_accum->Resize(common::make_ddim(
      //     {flashmaskv2_fwd_params_get_num_splits(params_handle),
      //      batch_size,
      //      num_heads,
      //      seqlen_q,
      //      head_size_v}));
      // softmax_lse_accum->Resize(common::make_ddim(
      //     {flashmaskv2_fwd_params_get_num_splits(params_handle),
      //      batch_size,
      //      num_heads,
      //      seqlen_q}));
      // dev_ctx.template Alloc<float>(out_accum);
      // dev_ctx.template Alloc<float>(softmax_lse_accum);

      *out_accum = paddle::empty(
        {flashmaskv2_fwd_params_get_num_splits(params_handle), batch_size, num_heads, seqlen_q, head_size_v},
        paddle::DataType::FLOAT32,
        q.place());

      *softmax_lse_accum = paddle::empty(
        {flashmaskv2_fwd_params_get_num_splits(params_handle), batch_size, num_heads, seqlen_q},
        paddle::DataType::FLOAT32,
        q.place());



      flashmaskv2_fwd_params_set_oaccum_batch_stride(
          params_handle, out_accum->strides()[1]);
      flashmaskv2_fwd_params_set_lseaccum_batch_stride(
          params_handle, softmax_lse_accum->strides()[1]);
    } else {
      // out_accum->Resize(common::make_ddim(
      //     {flashmaskv2_fwd_params_get_num_splits(params_handle),
      //      num_heads,
      //      total_q,
      //      head_size_v}));
      // softmax_lse_accum->Resize(common::make_ddim(
      //     {flashmaskv2_fwd_params_get_num_splits(params_handle),
      //      num_heads,
      //      total_q}));
      // dev_ctx.template Alloc<float>(out_accum);
      // dev_ctx.template Alloc<float>(softmax_lse_accum);
      *out_accum = paddle::empty(
        {flashmaskv2_fwd_params_get_num_splits(params_handle), num_heads, total_q, head_size_v},
        paddle::DataType::FLOAT32,
        q.place());

      *softmax_lse_accum = paddle::empty(
        {flashmaskv2_fwd_params_get_num_splits(params_handle), num_heads, total_q},
        paddle::DataType::FLOAT32,
        q.place());

    }
    flashmaskv2_fwd_params_set_is_fp32(params_handle, false);
    flashmaskv2_fwd_params_set_oaccum_ptr(params_handle,
                                                        (out_accum->data()));
    flashmaskv2_fwd_params_set_softmax_lseaccum_ptr(
        params_handle, (softmax_lse_accum->data()));
    flashmaskv2_fwd_params_set_oaccum_split_stride(
        params_handle, out_accum->strides()[0]);
    flashmaskv2_fwd_params_set_oaccum_row_stride(
        params_handle, out_accum->strides()[out_accum->strides().size() - 2]);
    flashmaskv2_fwd_params_set_oaccum_head_stride(
        params_handle, out_accum->strides()[out_accum->strides().size() - 3]);
    flashmaskv2_fwd_params_set_lseaccum_split_stride(
        params_handle, softmax_lse_accum->strides()[0]);
    flashmaskv2_fwd_params_set_lseaccum_head_stride(
        params_handle,
        softmax_lse_accum->strides()[softmax_lse_accum->strides().size() - 2]);
  }

  if (q_type == paddle::DataType::FLOAT8_E4M3FN) {
    if (q_descale_.is_initialized()) {
      paddle::Tensor q_descale = q_descale_.get();
      CHECK_DEVICE(q_descale);
      CHECK_SHAPE(q_descale, batch_size, num_heads_k);
      flashmaskv2_fwd_params_set_q_descale_ptr(
          params_handle, (q_descale.data<float>()));
      flashmaskv2_fwd_params_set_q_descale_batch_stride(
          params_handle, q_descale.strides()[0]);
      flashmaskv2_fwd_params_set_q_descale_head_stride(
          params_handle, q_descale.strides()[1]);
    } else {
      flashmaskv2_fwd_params_set_q_descale_ptr(params_handle,
                                                             nullptr);
    }
    if (k_descale_.is_initialized()) {
      paddle::Tensor k_descale = k_descale_.get();
      CHECK_DEVICE(k_descale);
      CHECK_SHAPE(k_descale, batch_size, num_heads_k);
      flashmaskv2_fwd_params_set_k_descale_ptr(
          params_handle, (k_descale.data<float>()));
      flashmaskv2_fwd_params_set_k_descale_batch_stride(
          params_handle, k_descale.strides()[0]);
      flashmaskv2_fwd_params_set_k_descale_head_stride(
          params_handle, k_descale.strides()[1]);
    } else {
      flashmaskv2_fwd_params_set_k_descale_ptr(params_handle,
                                                             nullptr);
    }
    if (v_descale_.is_initialized()) {
      paddle::Tensor v_descale = v_descale_.get();
      CHECK_DEVICE(v_descale);
      CHECK_SHAPE(v_descale, batch_size, num_heads_k);
      flashmaskv2_fwd_params_set_v_descale_ptr(
          params_handle, (v_descale.data<float>()));
      flashmaskv2_fwd_params_set_v_descale_batch_stride(
          params_handle, v_descale.strides()[0]);
      flashmaskv2_fwd_params_set_v_descale_head_stride(
          params_handle, v_descale.strides()[1]);
    } else {
      flashmaskv2_fwd_params_set_v_descale_ptr(params_handle,
                                                             nullptr);
    }
  }

#ifdef FLASHATTENTION_DISABLE_LOCAL
  PADDLE_ENFORCE_EQ(
      !flashmaskv2_fwd_params_get_is_local(params_handle),
      true,
      common::errors::InvalidArgument(
          "This flash attention build does not support local attention."));
#endif
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  PADDLE_ENFORCE_EQ(
      flashmaskv2_fwd_params_get_softcap(params_handle),
      0.0,
      common::errors::InvalidArgument(
          "This flash attention build does not support tanh softcapping."));
#endif
#ifdef FLASHATTENTION_DISABLE_SPLIT
  PADDLE_ENFORCE_EQ(
      flashmaskv2_fwd_params_get_num_splits(params_handle),
      1,
      common::errors::InvalidArgument(
          "This flash attention build does not support splits."));
#endif
#ifdef FLASHATTENTION_DISABLE_PACKGQA
  PADDLE_ENFORCE_EQ(
      (!flashmaskv2_fwd_params_get_pack_gqa(params_handle) ||
       flashmaskv2_fwd_params_get_arch(params_handle) < 90 ||
       (flashmaskv2_fwd_params_get_page_table(params_handle) &&
        !flashmaskv2_fwd_params_get_pagedkv_tma(params_handle)) ||
       flashmaskv2_fwd_params_get_num_splits(params_handle) > 1),
      true,
      common::errors::InvalidArgument(
          "This flash attention build does not support pack_gqa."));
#endif
#ifdef FLASHATTENTION_DISABLE_PAGEDKV
  PADDLE_ENFORCE_EQ(
      (!(flashmaskv2_fwd_params_get_page_table(params_handle) &&
         !flashmaskv2_fwd_params_get_pagedkv_tma(params_handle))),
      true,
      common::errors::InvalidArgument(
          "This flash attention build does not support paged KV."));
#endif
#ifdef FLASHATTENTION_DISABLE_APPENDKV
  PADDLE_ENFORCE_EQ(
      !k_new_.is_initialized(),
      true,
      common::errors::InvalidArgument(
          "This flash attention build does not support appending KV."));
#endif

  // flashmask
  paddle::Tensor startend_row_indices;
  if (is_flashmask) startend_row_indices = startend_row_indices_.get();
  paddle::Tensor block_mask;
  if (is_blockmask) block_mask = block_mask_.get();
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

    // int device_id = dev_ctx.GetPlace().GetDeviceId();
    // auto dprops = paddle::platform::GetDeviceProperties(device_id);
    const bool is_sm90 = dprops.major == 9 && dprops.minor == 0;

    if (is_sm90) {
      // seqlen_k to nblock_seqlen, here we use kBlockN = 64
      // as a conservative estimation (reduce allocation size)
      flashmask_maxmin_shape[2] =
          ((flashmask_maxmin_shape[2] + 63) / 64 + 3) / 4 * 4;
      // make sure this is the same with FlashMaskV3 fwd main loop
      static constexpr int flashmask_buffer_length = 16 * 1024;
      // estimate the upper bound of the possible chunk size
      static constexpr int chunk_padded_length =
          ((flashmask_buffer_length + 63) / 64 + 31) & 0xffffffe0;
      static constexpr int chunk_valid_length =
          ((flashmask_buffer_length + 63) / 64 + 3) & 0xfffffffc;
      const int num_chunk =
          (flashmask_maxmin_shape[2] + chunk_valid_length - 1) /
          chunk_valid_length;
      flashmask_maxmin_shape[2] = num_chunk * chunk_padded_length;
    } else {
      // seqlen_k to nblock_seqlen
      flashmask_maxmin_shape[2] =
          ((flashmask_maxmin_shape[2] + 31) / 32 + 3) / 4 * 4;
    }
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

    // [BQW_CHANGE] 修复 slice 逻辑：phi::Slice 创建的是**连续内存拷贝**，不是 view
    // 对于 shape [B, H, S, C] 的 contiguous tensor，strides = [H*S*C, S*C, C, 1]
    // 数据交错存储：[lt_s0, lt_e0, ut_s0, ut_e0, lt_s1, lt_e1, ut_s1, ut_e1, ...]
    // 内核通过裸指针线性访问，假设 [B, H, S] 连续布局
    // 因此必须将每个 channel 拷贝到独立的连续内存中
    //
    // 原始 Paddle 代码 (phi::Slice) 等价于：
    //   lt_start_row_indices = phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {0}, {1});
    //   lt_end_row_indices   = phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {1}, {2});
    //   ut_start_row_indices = phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {2}, {3});
    //   ut_end_row_indices   = phi::Slice<int32_t>(dev_ctx, startend_row_indices, {3}, {3}, {4});
    // 每次 Slice 都会分配新内存并拷贝，确保结果连续

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

    // 辅助 lambda：将 channel_idx 对应的数据从交错布局拷贝到连续 tensor 中
    // 使用 cudaMemcpy2DAsync 做 strided copy:
    //   src 起始位置 = mask_base_ptr + channel_idx
    //   src pitch    = C * sizeof(int32_t)  (源数据中同一 channel 相邻元素间距)
    //   dst pitch    = sizeof(int32_t)       (目标连续存储)
    //   width        = sizeof(int32_t)       (每次拷贝一个 int32)
    //   height       = total_elements        (共 B*H*S 个元素)
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

  }

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
                          "blockmask is now only support blockdim_q = 128 "));

    PADDLE_ENFORCE_EQ(block_mask.dims()[3],
                      (seqlen_k + 127) / 128,
                      common::errors::InvalidArgument(
                          "blockmask is now only support blockdim_k = 128 "));

    PADDLE_ENFORCE_EQ(
        block_mask.dims()[1],
        startend_row_indices.dims()[1],
        common::errors::InvalidArgument("blockmask is now only support same "
                                        "dim num_heads with flashmask "));
  }

  if (is_blockmask) {
    // xhy: blockmask is now only support blockdim_q k = 128
    flashmaskv2_fwd_params_set_m_block_dim(params_handle, 128);
    flashmaskv2_fwd_params_set_n_block_dim(params_handle, 128);
    flashmaskv2_fwd_params_set_block_mask_ptr(
        params_handle, (block_mask.data<int32_t>()));
  }

  if (is_flashmask) {
    // [BQW_CHANGE] 添加 const_cast，setter 函数接受 int32_t* (非 const)
    flashmaskv2_fwd_params_set_lt_start_ptr(params_handle, const_cast<int32_t*>(lt_start_ptr));
    flashmaskv2_fwd_params_set_lt_end_ptr(params_handle, const_cast<int32_t*>(lt_end_ptr));
    flashmaskv2_fwd_params_set_ut_start_ptr(params_handle, const_cast<int32_t*>(ut_start_ptr));
    flashmaskv2_fwd_params_set_ut_end_ptr(params_handle, const_cast<int32_t*>(ut_end_ptr));

    if (flashmask_maxmin.initialized())
      flashmaskv2_fwd_params_set_flashmask_maxmin_ptr(
          params_handle, (flashmask_maxmin.data<int32_t>()));
    else
      flashmaskv2_fwd_params_set_flashmask_maxmin_ptr(params_handle,
                                                               nullptr);

    flashmaskv2_fwd_params_set_h_flashmask(
        params_handle, startend_row_indices.dims()[1]);
    flashmaskv2_fwd_params_set_h_h_flashmask_ratio(
        params_handle, num_heads / startend_row_indices.dims()[1]);
  } else {
    flashmaskv2_fwd_params_set_lt_start_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_lt_end_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_ut_start_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_ut_end_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_flashmask_maxmin_ptr(params_handle,
                                                             nullptr);
    flashmaskv2_fwd_params_set_h_flashmask(params_handle, 0);
    flashmaskv2_fwd_params_set_h_h_flashmask_ratio(params_handle, 0);
  }

  if (total_q > 0 &&
      (total_k +
       flashmaskv2_fwd_params_get_total_knew(params_handle)) > 0 &&
      num_heads_k > 0) {
    // flashmaskv2_run_mha_fwd(params_handle, dev_ctx.stream());
    flashmaskv2_run_mha_fwd(params_handle, stream);
    if (flashmaskv2_fwd_params_get_num_splits(params_handle) > 1) {
      if (out_type == paddle::DataType::BFLOAT16) {
        // Since we want output in BF16. Otherwise fwd_combine will output to
        // FP16
        flashmaskv2_fwd_params_set_is_bf16(params_handle, true);
      }
      // Unless there's seqused_q, for the purpose of attn_combine, we can just
      // treat it as batch=1 and seqlen = total_q, and don't need to dispatch to
      // Varlen there. However, with dynamic split, each row needs to know which
      // batch it belongs to to read the number of splits, so we just use the
      // varlen version of combine kernel. if (is_varlen_q &&
      // !seqused_q_.has_value()) { if (is_varlen_q) {
      //     params.b = 1;
      //     params.seqlen_q = total_q;
      // }
      // }
      // flashmaskv2_run_mha_fwd_combine(
      //     params_handle, dev_ctx.stream(), true /*enable_pdl*/);

      flashmaskv2_run_mha_fwd_combine(
          params_handle, stream, true /*enable_pdl*/);

    }
  } else if (total_q > 0 && num_heads_k > 0) {
    PADDLE_ENFORCE_EQ(
        (out->dtype() == paddle::DataType::BFLOAT16 ||
         out->dtype() == paddle::DataType::FLOAT16 ||
         out->dtype() == paddle::DataType::FLOAT8_E4M3FN),
        true,
        common::errors::InvalidArgument("flash attention 3 supports bfloat16, "
                                        "float16 and float8_e4m3fn only."));
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output
    // to 0.
    // [BQW_CHANGE] 修复 memset 大小 BUG：使用 out tensor 实际大小，而非 metadata_size
    int64_t out_numel = batch_size * seqlen_q * num_heads * head_size_v;
    if (out->dtype() == paddle::DataType::BFLOAT16) {
      cudaMemsetAsync(out->data(), 0, out_numel * 2, stream);  // bf16 = 2 bytes
    } else if (out->dtype() == paddle::DataType::FLOAT16) {
      cudaMemsetAsync(out->data(), 0, out_numel * 2, stream);  // fp16 = 2 bytes
    } else if (out->dtype() == paddle::DataType::FLOAT8_E4M3FN) {
      cudaMemsetAsync(out->data(), 0, out_numel * 1, stream);  // fp8 = 1 byte
    }
    // [BQW_CHANGE] 修复 infinity 设置 BUG：cudaMemsetAsync 逐字节设置，
    // 不能设置 float infinity 值 (infinity 的字节表示不全相同)
    // 使用 paddle::full 创建填充了 infinity 的 tensor 替代
    *softmax_lse = paddle::full(
        {batch_size, num_heads, seqlen_q},
        std::numeric_limits<float>::infinity(),
        paddle::DataType::FLOAT32,
        q.place());

  }

#else
  RaiseNotSupportedError();
#endif
}
#endif


// [BQW_CHANGE] phi::dtype::bfloat16 → paddle::bfloat16 (外迁后不直接使用 phi 命名空间)
template void FlashMaskV2BaseKernel<paddle::bfloat16>(
    const paddle::Tensor &q,
    const paddle::Tensor &k,
    const paddle::Tensor &v,
    const paddle::optional<paddle::Tensor> &k_new_,
    const paddle::optional<paddle::Tensor> &v_new_,
    const paddle::optional<paddle::Tensor> &q_v_,
    const paddle::optional<paddle::Tensor> &out_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_q_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_,
    const paddle::optional<paddle::Tensor> &cu_seqlens_k_new_,
    const paddle::optional<paddle::Tensor> &seqused_q_,
    const paddle::optional<paddle::Tensor> &seqused_k_,
    const paddle::optional<paddle::Tensor> &page_table_,
    const paddle::optional<paddle::Tensor> &kv_batch_idx_,
    const paddle::optional<paddle::Tensor> &leftpad_k_,
    const paddle::optional<paddle::Tensor> &rotary_cos_,
    const paddle::optional<paddle::Tensor> &rotary_sin_,
    const paddle::optional<paddle::Tensor> &q_descale_,
    const paddle::optional<paddle::Tensor> &k_descale_,
    const paddle::optional<paddle::Tensor> &v_descale_,
    const paddle::optional<paddle::Tensor> &scheduler_metadata_,
    const paddle::optional<paddle::Tensor> &startend_row_indices_,
    const paddle::optional<paddle::Tensor> &block_mask_,
    const int max_seqlen_q_,
    const int max_seqlen_k_,
    const float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool is_rotary_interleaved,
    int num_splits,
    const bool manual_set_pack_gqa,
    const bool pack_gqa_,
    const int sm_margin,
    paddle::Tensor *out,
    paddle::Tensor *softmax_lse,
    paddle::Tensor *out_accum,
    paddle::Tensor *softmax_lse_accum);


#endif
