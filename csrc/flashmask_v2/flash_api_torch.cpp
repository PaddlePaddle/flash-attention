#include <Python.h>
#include <torch/nn/functional/padding.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>
#include <stdio.h>
#include <unistd.h>
#include "flash.h"
#include "flash_api.h"

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the TORCH_LIBRARY static initializers
    below are run. 
*/
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",   /* name of module */
        NULL,   /* module documentation, may be NULL */
        -1,     /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
        NULL,   /* methods */
    };
    return PyModule_Create(&module_def);
}
}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


#define PRINT_FIELD(name) \
    std::cout << #name << " = " << params.name << std::endl;

#define PRINT_PTR(name) \
    std::cout << #name << " = " << params.name << " (ptr)" << std::endl;

void print_qkv_params(const Qkv_params& params) {
    std::cout << "===== Qkv_params =====" << std::endl;

    PRINT_PTR(q_ptr);
    PRINT_PTR(k_ptr);
    PRINT_PTR(v_ptr);

    PRINT_FIELD(q_batch_stride);
    PRINT_FIELD(k_batch_stride);
    PRINT_FIELD(v_batch_stride);

    PRINT_FIELD(q_row_stride);
    PRINT_FIELD(k_row_stride);
    PRINT_FIELD(v_row_stride);

    PRINT_FIELD(q_head_stride);
    PRINT_FIELD(k_head_stride);
    PRINT_FIELD(v_head_stride);
    PRINT_FIELD(v_dim_stride);

    PRINT_FIELD(h);
    PRINT_FIELD(h_k);
}

void print_flash_fwd_params(const Flash_fwd_params& params) {
    print_qkv_params(params);

    std::cout << "===== Flash_fwd_params =====" << std::endl;

    PRINT_PTR(o_ptr);
    PRINT_PTR(oaccum_ptr);

    PRINT_FIELD(o_batch_stride);
    PRINT_FIELD(o_row_stride);
    PRINT_FIELD(o_head_stride);

    PRINT_PTR(softmax_lse_ptr);
    PRINT_PTR(softmax_lseaccum_ptr);

    PRINT_PTR(q_descale_ptr);
    PRINT_PTR(k_descale_ptr);
    PRINT_PTR(v_descale_ptr);

    PRINT_FIELD(q_descale_batch_stride);
    PRINT_FIELD(q_descale_head_stride);
    PRINT_FIELD(k_descale_batch_stride);
    PRINT_FIELD(k_descale_head_stride);
    PRINT_FIELD(v_descale_batch_stride);
    PRINT_FIELD(v_descale_head_stride);

    PRINT_FIELD(b);
    PRINT_FIELD(seqlen_q);
    PRINT_FIELD(seqlen_k);
    PRINT_FIELD(seqlen_knew);
    PRINT_FIELD(d);
    PRINT_FIELD(dv);
    PRINT_FIELD(rotary_dim);

    PRINT_FIELD(total_q);
    PRINT_FIELD(total_k);
    PRINT_FIELD(total_knew);
    PRINT_FIELD(b_k);

    PRINT_FIELD(scale_softmax);
    PRINT_FIELD(softcap);

    PRINT_PTR(cu_seqlens_q);
    PRINT_PTR(cu_seqlens_k);
    PRINT_PTR(cu_seqlens_knew);
    PRINT_PTR(leftpad_k);

    PRINT_PTR(seqused_q);
    PRINT_PTR(seqused_k);

    PRINT_FIELD(oaccum_split_stride);
    PRINT_FIELD(oaccum_batch_stride);
    PRINT_FIELD(oaccum_row_stride);
    PRINT_FIELD(oaccum_head_stride);

    PRINT_FIELD(lseaccum_split_stride);
    PRINT_FIELD(lseaccum_batch_stride);
    PRINT_FIELD(lseaccum_head_stride);

    PRINT_PTR(knew_ptr);
    PRINT_PTR(vnew_ptr);

    PRINT_FIELD(knew_batch_stride);
    PRINT_FIELD(vnew_batch_stride);
    PRINT_FIELD(knew_row_stride);
    PRINT_FIELD(vnew_row_stride);
    PRINT_FIELD(knew_head_stride);
    PRINT_FIELD(vnew_head_stride);

    PRINT_PTR(qv_ptr);
    PRINT_FIELD(qv_batch_stride);
    PRINT_FIELD(qv_row_stride);
    PRINT_FIELD(qv_head_stride);

    PRINT_PTR(rotary_cos_ptr);
    PRINT_PTR(rotary_sin_ptr);

    PRINT_PTR(kv_batch_idx);

    PRINT_PTR(page_table);
    PRINT_FIELD(page_table_batch_stride);
    PRINT_FIELD(page_size);
    PRINT_FIELD(num_pages);
    PRINT_FIELD(pagedkv_tma);

    PRINT_FIELD(p_dropout);
    PRINT_FIELD(p_dropout_in_uint8_t);
    PRINT_FIELD(rp_dropout);

    PRINT_FIELD(window_size_left);
    PRINT_FIELD(window_size_right);

    PRINT_PTR(rng_state);

    PRINT_FIELD(is_bf16);
    PRINT_FIELD(is_fp32);
    PRINT_FIELD(is_e4m3);
    PRINT_FIELD(is_causal);
    PRINT_FIELD(is_local);
    PRINT_FIELD(is_rotary_interleaved);

    PRINT_FIELD(num_splits);
    PRINT_FIELD(pack_gqa);

    PRINT_PTR(tile_count_semaphore);
    PRINT_PTR(num_splits_dynamic_ptr);
    PRINT_FIELD(skip_scheduler_metadata_computation);

    PRINT_FIELD(arch);
    PRINT_FIELD(num_sm);
}

namespace {
inline at::cuda::CUDAGuard make_cuda_guard_from_tensor(const at::Tensor& t) {
  return at::cuda::CUDAGuard(static_cast<c10::DeviceIndex>(t.get_device()));
}
} // namespace

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


void destroy_flashmask_fwd_params_handle(Flash_fwd_params *params_handle) {
  flashmaskv2_destroy_fwd_params_handle(params_handle);
}

void destroy_flashmask_bwd_params_handle(Flash_bwd_params *params_handle) {
  flashmaskv2_destroy_bwd_params_handle(params_handle);
}

FlashMask_fwd_params *get_flashmask_fwd_params_handle() {
  static std::unique_ptr<Flash_fwd_params,
                         decltype(&destroy_flashmask_fwd_params_handle)>
      params_handle(flashmaskv2_create_fwd_params_handle(), &destroy_flashmask_fwd_params_handle);

  return params_handle.get();
}

FlashMask_bwd_params *get_flashmask_bwd_params_handle() {
  static std::unique_ptr<Flash_bwd_params,
                         decltype(&destroy_flashmask_bwd_params_handle)>
      params_handle(flashmaskv2_create_bwd_params_handle(), &destroy_flashmask_bwd_params_handle);

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
                                  const at::Tensor &q,
                                  const at::Tensor &k,
                                  const at::Tensor &v,
                                  const at::Tensor out,
                                  void *cu_seqlens_q_d,
                                  void *cu_seqlens_k_d,
                                  void *seqused_q,
                                  void *seqused_k,
                                  void *softmax_lse_d,
                                  float p_dropout,
                                  float softmax_scale,
                                  int window_size_left,
                                  int window_size_right,
                                  const cudaDeviceProp *dprops,
                                  const float softcap=0.f,
                                  const int sm_margin=0) {
  flashmaskv2_fwd_params_set_is_bf16(params_handle, q.dtype() == torch::kBFloat16);
  flashmaskv2_fwd_params_set_is_e4m3(params_handle, q.dtype() == torch::kFloat8_e4m3fn);

  // Set the pointers and strides.
  flashmaskv2_fwd_params_set_q_ptr(params_handle, const_cast<void *>(q.data_ptr()));
  flashmaskv2_fwd_params_set_k_ptr(params_handle, const_cast<void *>(k.data_ptr()));
  flashmaskv2_fwd_params_set_v_ptr(params_handle, const_cast<void *>(v.data_ptr()));
  // All stride are in elements, not bytes.
  flashmaskv2_fwd_params_set_q_row_stride(params_handle, q.stride(-3));
  flashmaskv2_fwd_params_set_k_row_stride(params_handle, k.stride(-3));
  flashmaskv2_fwd_params_set_v_row_stride(params_handle, v.stride(-3));
  flashmaskv2_fwd_params_set_q_head_stride(params_handle, q.stride(-2));
  flashmaskv2_fwd_params_set_k_head_stride(params_handle, k.stride(-2));
  flashmaskv2_fwd_params_set_v_head_stride(params_handle, v.stride(-2));
  flashmaskv2_fwd_params_set_v_dim_stride(params_handle, v.stride(-1));
  flashmaskv2_fwd_params_set_o_ptr(params_handle, out.data_ptr());
  flashmaskv2_fwd_params_set_o_row_stride(params_handle, out.stride(-3));
  flashmaskv2_fwd_params_set_o_head_stride(params_handle, out.stride(-2));

  if (cu_seqlens_q_d == nullptr) {
    flashmaskv2_fwd_params_set_q_batch_stride(params_handle, q.stride(0));
    flashmaskv2_fwd_params_set_o_batch_stride(params_handle, out.stride(0));
  }
  if (cu_seqlens_k_d == nullptr) {
    flashmaskv2_fwd_params_set_k_batch_stride(params_handle, k.stride(0));
    flashmaskv2_fwd_params_set_v_batch_stride(params_handle, v.stride(0));
  }

  flashmaskv2_fwd_params_set_cu_seqlens_q(params_handle, static_cast<int *>(cu_seqlens_q_d));
  flashmaskv2_fwd_params_set_cu_seqlens_k(params_handle, static_cast<int *>(cu_seqlens_k_d));
  flashmaskv2_fwd_params_set_seqused_q(params_handle, static_cast<int *>(seqused_q));
  flashmaskv2_fwd_params_set_seqused_k(params_handle, static_cast<int *>(seqused_k));

  // Softmax sum
  flashmaskv2_fwd_params_set_softmax_lse_ptr(params_handle, softmax_lse_d);

  // Set the dimensions.
  flashmaskv2_fwd_params_set_b(params_handle, b);
  flashmaskv2_fwd_params_set_h(params_handle, h);
  flashmaskv2_fwd_params_set_h_k(params_handle, h_k);
  flashmaskv2_fwd_params_set_seqlen_q(params_handle, seqlen_q);
  flashmaskv2_fwd_params_set_seqlen_k(params_handle, seqlen_k);
  flashmaskv2_fwd_params_set_seqlen_q_rounded(params_handle, seqlen_q_rounded);
  flashmaskv2_fwd_params_set_seqlen_k_rounded(params_handle, seqlen_k_rounded);
  flashmaskv2_fwd_params_set_d(params_handle, d);
  flashmaskv2_fwd_params_set_d_rounded(params_handle, d_rounded);

  // Set the different scale values.
  flashmaskv2_fwd_params_set_scale_softmax(params_handle, softmax_scale);
  flashmaskv2_fwd_params_set_softcap(params_handle, softcap);

  // Set this to probability of keeping an element to simplify things.
  flashmaskv2_fwd_params_set_p_dropout(params_handle, 1.f - p_dropout);
  // Convert p from float to int so we don't have to convert the random uint to float to compare. 
  // [Minor] We want to round down since when we do the comparison we use <= instead of < params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));

  flashmaskv2_fwd_params_set_p_dropout_in_uint8_t(params_handle, uint8_t(std::floor(flashmaskv2_fwd_params_get_p_dropout(params_handle) * 255.0)));
  flashmaskv2_fwd_params_set_rp_dropout(params_handle, 1.f / flashmaskv2_fwd_params_get_p_dropout(params_handle));
  TORCH_CHECK(p_dropout < 1.f, "p_dropout must less than 1");
  TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
  // Causal is the special case where window_size_right == 0 and window_size_left < 0. 
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  flashmaskv2_fwd_params_set_is_causal(params_handle, window_size_left < 0 && window_size_right == 0);
  flashmaskv2_fwd_params_set_is_local(params_handle, (window_size_left >= 0 || window_size_right >= 0) && !flashmaskv2_fwd_params_get_is_causal(params_handle));

  if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k - 1; }
  if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_q - 1; }
  flashmaskv2_fwd_params_set_window_size_left(params_handle, window_size_left);
  flashmaskv2_fwd_params_set_window_size_right(params_handle, window_size_right);

  int arch = dprops->major * 10 + dprops->minor;
  int num_sm = dprops->multiProcessorCount - sm_margin;

  flashmaskv2_fwd_params_set_arch(params_handle, arch);
  flashmaskv2_fwd_params_set_num_sm(params_handle, num_sm);

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(!flashmaskv2_fwd_params_get_is_local(params_handle), "This flash attention build does not support local attention.");
#endif
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
                                  const at::Tensor &q,
                                  const at::Tensor &k,
                                  const at::Tensor &v,
                                  const at::Tensor &out,
                                  const at::Tensor &dout,
                                  at::Tensor dq,
                                  at::Tensor dk,
                                  at::Tensor dv,
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
                                  const cudaDeviceProp *dprops,
                                  const float softcap=0.f,
                                  bool deterministic=false,
                                  const int sm_margin=0) {
  set_flashmaskv2_params_fprop(flashmaskv2_cast_to_fwd_params_handle(params_handle),
                               b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded,
                               h, h_k, d, d_rounded,
                               q, k, v, out,
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
  flashmaskv2_bwd_params_set_do_ptr(params_handle, dout.data_ptr());
  flashmaskv2_bwd_params_set_do_row_stride(params_handle, dout.stride(-3));
  flashmaskv2_bwd_params_set_do_head_stride(params_handle, dout.stride(-2));
  flashmaskv2_bwd_params_set_dq_ptr(params_handle, dq.data_ptr());
  flashmaskv2_bwd_params_set_dk_ptr(params_handle, dk.data_ptr());
  flashmaskv2_bwd_params_set_dv_ptr(params_handle, dv.data_ptr());
  flashmaskv2_bwd_params_set_dq_row_stride(params_handle, dq.stride(-3));
  flashmaskv2_bwd_params_set_dk_row_stride(params_handle, dk.stride(-3));
  flashmaskv2_bwd_params_set_dv_row_stride(params_handle, dv.stride(-3));
  flashmaskv2_bwd_params_set_dq_head_stride(params_handle, dq.stride(-2));
  flashmaskv2_bwd_params_set_dk_head_stride(params_handle, dk.stride(-2));
  flashmaskv2_bwd_params_set_dv_head_stride(params_handle, dv.stride(-2));

  if (cu_seqlens_q_d == nullptr) {
    flashmaskv2_bwd_params_set_do_batch_stride(params_handle, dout.stride(0));
    flashmaskv2_bwd_params_set_dq_batch_stride(params_handle, dq.stride(0));
    flashmaskv2_bwd_params_set_dk_batch_stride(params_handle, dk.stride(0));
    flashmaskv2_bwd_params_set_dv_batch_stride(params_handle, dv.stride(0));
  }

  flashmaskv2_bwd_params_set_dq_accum_ptr(params_handle, dq_accum_d);
  flashmaskv2_bwd_params_set_dk_accum_ptr(params_handle, dk_accum_d);
  flashmaskv2_bwd_params_set_dv_accum_ptr(params_handle, dv_accum_d);

  // Softmax sum
  flashmaskv2_bwd_params_set_dsoftmax_sum(params_handle, dsoftmax_sum_d);

  flashmaskv2_bwd_params_set_deterministic(params_handle, deterministic);
}


// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FlashMaskV2BaseKernel(const at::Tensor &q,
                      const at::Tensor &k,
                      const at::Tensor &v,
                      const c10::optional<at::Tensor> &k_new_,  // (b, s_k_new, h_k, d ) or (total_k_new, h_k, d ) if there is cu_seqlens_k_new
                      const c10::optional<at::Tensor> &v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
                      const c10::optional<at::Tensor> &q_v_,    // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
                      const c10::optional<at::Tensor> &out_,    // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
                      const c10::optional<at::Tensor> &cu_seqlens_q_,      // (b + 1)
                      const c10::optional<at::Tensor> &cu_seqlens_k_,      // (b + 1)
                      const c10::optional<at::Tensor> &cu_seqlens_k_new_,  // (b + 1)
                      const c10::optional<at::Tensor> &seqused_q_,   // b. If given, only this many elements of each batch element's queries and outputs are used.
                      const c10::optional<at::Tensor> &seqused_k_,   // b. If given, only this many elements of each batch element's keys are used.
                      c10::optional<int64_t> max_seqlen_q_,  // if max_seqlen_q_ is set to 0, it indicates that it is uninitialized and should not be referenced
                      c10::optional<int64_t> max_seqlen_k_,  // if max_seqlen_q_ is set to 0, it indicates that it is uninitialized and should not be referenced
                      const c10::optional<at::Tensor> &page_table_,  // (b_k, max_num_pages_per_seq)
                      const c10::optional<at::Tensor> &kv_batch_idx_,// b. indices to index into the KV cache
                      const c10::optional<at::Tensor> &leftpad_k_,   // b
                      const c10::optional<at::Tensor> &rotary_cos_,  // seqlen_ro x (rotary_dim / 2)
                      const c10::optional<at::Tensor> &rotary_sin_,  // seqlen_ro x (rotary_dim / 2)
                      const c10::optional<at::Tensor> &q_descale_,   // (b, h_k), not (b, h)
                      const c10::optional<at::Tensor> &k_descale_,   // (b, h_k)
                      const c10::optional<at::Tensor> &v_descale_,   // (b, h_k)
                      const c10::optional<at::Tensor> &scheduler_metadata_,    // (b + 1)
                      const c10::optional<at::Tensor> &startend_row_indices_,  //（b, h, s_1, [1, 2, 4])
                      const c10::optional<at::Tensor> &block_mask_,            // (b, h, s // 128, s // 128)
                      c10::optional<double> softmax_scale_,
                      bool is_causal,
                      int64_t window_size_left,
                      int64_t window_size_right,
                      const double softcap,
                      const bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                      int64_t num_splits,
                      const bool manual_set_pack_gqa,
                      const bool pack_gqa_,               // the pack_gqa_ will be used only if manual_set_pack_gqa is set to True; otherwise, the internal heuristic get_pack_gqa() from fa3 will decide whether to pack gqa
                      const int64_t sm_margin) {

  auto dprops = at::cuda::getCurrentDeviceProperties();
  const bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90, "FlashMask only supports Hopper GPUs currently.");

  auto q_type = q.scalar_type();
  TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 || q_type == at::ScalarType::Float8_e4m3fn,
              "FlashMask only supports fp16, bf16, and fp8_e4m3 data type.");

  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  at::Tensor page_table;
  const bool paged_KV = page_table_.has_value();
  if (paged_KV) {
    page_table = page_table_.value();
    CHECK_DEVICE(page_table);
    TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype torch.int32");
    TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
  }

  at::Tensor cu_seqlens_q;
  const bool is_varlen_q = cu_seqlens_q_.has_value();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.value();
    CHECK_DEVICE(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_q);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
    TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
  }

  at::Tensor cu_seqlens_k;
  const bool is_varlen_k = cu_seqlens_k_.has_value();
  if (is_varlen_k) {
    cu_seqlens_k = cu_seqlens_k_.value();
    CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "page_table must have dtype torch.int32")
    TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
    TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then page table is not supported");
    TORCH_CHECK(!kv_batch_idx_, "If cu_seqlens_k is passed in, then page table is not supported");
  }

  const auto sizes = q.sizes();
  const int batch_size = is_varlen_q ? cu_seqlens_q.size(0) - 1 : sizes[0];
  int seqlen_q = is_varlen_q ? max_seqlen_q_.value() : sizes[1];
  int total_q = is_varlen_q ? sizes[0] : batch_size * sizes[1];
  int64_t num_heads = q.size(-2);
  const int64_t head_size = q.size(-1);
  const int head_size_v = v.size(-1);
  const int max_num_pages_per_seq = paged_KV ? page_table.size(1) : 0;
  const int num_pages = paged_KV ? k.size(0) : 0;
  const int page_size = paged_KV ? k.size(1) : 1;
  const int seqlen_k = is_varlen_k ? max_seqlen_k_.value() : (paged_KV ? max_num_pages_per_seq * page_size : k.size(1));
  const int total_k = is_varlen_k ? k.size(0): batch_size * k.size(1);
  const int num_heads_k = k.size(-2);
  const int batch_size_k = paged_KV ? page_table.size(0) : (is_varlen_k ? cu_seqlens_k.size(0) - 1 : k.size(0));
  double softmax_scale = 1.0 / sqrt(double(head_size));
  if (softmax_scale_.has_value()) {
      softmax_scale = softmax_scale_.value();
  }
  if (!kv_batch_idx_.has_value()) {
    TORCH_CHECK(batch_size == batch_size_k, "batch_size must be equal to batch_size_k")
  }

  const int max_headdim = flashmaskv2_get_max_headdim();
  TORCH_CHECK(head_size <= max_headdim, "FlashMask forward only supports head dimension at most " + std::to_string(max_headdim));
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  if (head_size_v != head_size) {
    TORCH_CHECK(((head_size > 128 && head_size <= 192 && head_size_v > 96 && head_size_v <= 128) || 
                 (head_size <= 64 && head_size_v <= 512)),
                 "If V headdim is different from Q/K dim, we only support Q/K headdim in (128, 192] and V headdim in (96, 128], "
                 "or (Q/K <= 64 and V <= 512).");
    TORCH_CHECK(dprops->major == 9, "Only Hopper supports different V headdim");
    if (head_size_v > 256) {
      TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
                   "HeaddimV > 256 requires fp16 and bf16 data type");
    }
  }

  const bool is_flashmask = startend_row_indices_.has_value();
  const bool is_blockmask = block_mask_.has_value();

  // This needs to go before kBlockM & kBlockN since we rely on the correct
  // window_size and is_causal to set kBlockM
  window_size_left = (window_size_left >= seqlen_k - 1) ? -1 : window_size_left;
  window_size_right = (window_size_right >= seqlen_q - 1) ? -1 : window_size_right;

  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && window_size_left == -1 && window_size_right == -1) {
    // Special case of hdim 128 where we want causal to have kBlockN=128, better for pagedKV and TMA
      is_causal = ((head_size <= 64 || head_size > 128) || !paged_KV) && !is_flashmask ? false : is_causal;
  }
  window_size_right = is_causal ? 0 : window_size_right;

  // There's one case where is_causal=false, window_size=(-1, 0). Then
  // set_params_fprop will set params.is_causal=true. If we don't have is_causal
  // here matching params.is_causal, we might get the wrong kBlockM.
  is_causal = window_size_left < 0 && window_size_right == 0;

  if (is_varlen_q) {
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  } else {
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  }

  if (paged_KV) {
    CHECK_SHAPE(k, num_pages, page_size, num_heads_k, head_size);
    CHECK_SHAPE(v, num_pages, page_size, num_heads_k, head_size_v);
    CHECK_SHAPE(page_table, batch_size_k, max_num_pages_per_seq);
  } else {
    if (is_varlen_k) {
      CHECK_SHAPE(k, total_k, num_heads_k, head_size);
      CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    } else {
      CHECK_SHAPE(k, batch_size_k, seqlen_k, num_heads_k, head_size);
      CHECK_SHAPE(v, batch_size_k, seqlen_k, num_heads_k, head_size_v);
    }
  }

  if (seqused_q_.has_value()) {
    auto seqused_q = seqused_q_.value();
    TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
    CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
    CHECK_SHAPE(seqused_q, batch_size);
  }

  if (seqused_k_.has_value()) {
    auto seqused_k = seqused_k_.value();
    TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
    CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
    CHECK_SHAPE(seqused_k, batch_size);
  }

  if (leftpad_k_.has_value()) {
    auto leftpad_k = leftpad_k_.value();
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    CHECK_DEVICE(leftpad_k); CHECK_CONTIGUOUS(leftpad_k); 
    CHECK_SHAPE(leftpad_k, batch_size);
  }

  // This is what we will template on
  const bool is_varlen = is_varlen_q || is_varlen_k || seqused_q_.has_value() || seqused_k_.has_value() || leftpad_k_.has_value();
#ifdef FLASHATTENTION_DISABLE_VARLEN
  TORCH_CHECK(!is_varlen, "This flash attention build does not support varlen.");
#endif

  const int alignment = q_type == torch::kFloat8_e4m3fn ? 16 : 8;
  TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
  TORCH_CHECK(head_size_v % alignment == 0, "head_size_v should be a multiple of " + std::to_string(alignment));

  auto opts = q.options();
  auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;

  at::Tensor out;
  if (out_.has_value()) {
      out = out_.value();
      TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
      CHECK_DEVICE(out);
      TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
      if (is_varlen_q) {
          CHECK_SHAPE(out, total_q, num_heads, head_size_v);
      } else {
          CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
      }
  } else {
      out = is_varlen_q
          ? torch::empty({total_q, num_heads, head_size_v}, opts.dtype(out_type))
          : torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts.dtype(out_type));
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = flashmaskv2_round_up_headdim(head_size);
  const int head_size_v_rounded = flashmaskv2_round_up_headdim(head_size_v);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  auto device_guard = make_cuda_guard_from_tensor(q);
  at::Tensor softmax_lse;
  if (is_varlen_q) {
      softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
  } else {
      softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
  }

  FlashMask_fwd_params *params_handle = get_flashmask_fwd_params_handle();
  flashmaskv2_clear_fwd_params_handle(params_handle);
  // REF https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp#L895
  set_flashmaskv2_params_fprop(params_handle,
                               batch_size,
                               seqlen_q, seqlen_k,
                               seqlen_q_rounded, seqlen_k_rounded,
                               num_heads, num_heads_k,
                               head_size, head_size_rounded,
                               q, k, v, out,
                               is_varlen_q ? cu_seqlens_q.data_ptr() : nullptr,
                               is_varlen_k ? cu_seqlens_k.data_ptr() : nullptr,
                               seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
                               seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                               softmax_lse.data_ptr(),
                               /*p_dropout = */ 0.f,
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
  flashmaskv2_fwd_params_set_dv_rounded(params_handle, head_size_v_rounded);

  if (leftpad_k_.has_value()) {  // This needs to be set before get_pagedkv_tma
    flashmaskv2_fwd_params_set_leftpad_k(params_handle, leftpad_k_.value().data_ptr<int>());
  }
  if (paged_KV) {
    flashmaskv2_fwd_params_set_page_table(params_handle, page_table.data_ptr<int>());
    flashmaskv2_fwd_params_set_page_table_batch_stride(params_handle, page_table.stride(0));
  }
  flashmaskv2_fwd_params_set_page_size(params_handle, page_size);
  flashmaskv2_fwd_params_set_num_pages(params_handle, num_pages);

  if (k_new_.has_value()) {  // This needs to be set before get_pagedkv_tma
    at::Tensor k_new, v_new;
    TORCH_CHECK(v_new_.has_value(), "If k_new is supplied, v_new must also be passed in");
    TORCH_CHECK(seqused_k_.has_value(), "If k_new is supplied, seqlens_k must also be passed in");
    TORCH_CHECK(seqlen_q <= seqlen_k, "If k_new is supplied, it must have seqlen <= the seqlen of the KV cache");
    at::Tensor cu_seqlens_k_new;
    const bool is_varlen_k_new = cu_seqlens_k_new_.has_value();
    if (is_varlen_k_new) {
      cu_seqlens_k_new = cu_seqlens_k_new_.value();
      CHECK_DEVICE(cu_seqlens_k_new); CHECK_CONTIGUOUS(cu_seqlens_k_new);
      TORCH_CHECK(cu_seqlens_k_new.dtype() == torch::kInt32, "cu_seqlens_k_new must have dtype torch.int32");
    }
    k_new = k_new_.value();
    v_new = v_new_.value();
    TORCH_CHECK(k_new.dtype() == q_type, "k_new must have the same dtype as query");
    TORCH_CHECK(v_new.dtype() == q_type, "v_new must have the same dtype as query");
    CHECK_DEVICE(k_new); CHECK_DEVICE(v_new);
    TORCH_CHECK(k_new.stride(-1) == 1, "k_new tensor must have contiguous last dimension");
    TORCH_CHECK(v_new.stride(-1) == 1, "v_new tensor must have contiguous last dimension");
    // We don't need max_seqlen_k_new, so seqlen_k_new can be whatever when
    // is_varlen_k_new
    int seqlen_k_new = is_varlen_k_new ? 0 : k_new.size(1);
    int total_k_new = is_varlen_k_new ? k_new.size(0) : batch_size * k_new.size(1);
    if (is_varlen_k_new) {
      CHECK_SHAPE(k_new, total_k_new, num_heads_k, head_size);
      CHECK_SHAPE(v_new, total_k_new, num_heads_k, head_size_v);
      CHECK_SHAPE(cu_seqlens_k_new, batch_size + 1);
    } else {
      CHECK_SHAPE(k_new, batch_size, seqlen_k_new, num_heads_k, head_size);
      CHECK_SHAPE(v_new, batch_size, seqlen_k_new, num_heads_k, head_size_v);
    }

    flashmaskv2_fwd_params_set_seqlen_knew(params_handle, seqlen_k_new);
    flashmaskv2_fwd_params_set_total_knew(params_handle, total_k_new);
    flashmaskv2_fwd_params_set_knew_ptr(params_handle, k_new.data_ptr());
    flashmaskv2_fwd_params_set_vnew_ptr(params_handle, v_new.data_ptr());
    // All stride are in elements, not bytes.
    flashmaskv2_fwd_params_set_knew_row_stride(params_handle, k_new.stride(-3));
    flashmaskv2_fwd_params_set_vnew_row_stride(params_handle, v_new.stride(-3));
    flashmaskv2_fwd_params_set_knew_head_stride(params_handle, k_new.stride(-2));
    flashmaskv2_fwd_params_set_vnew_head_stride(params_handle, v_new.stride(-2));
    if (is_varlen_k_new) {
      flashmaskv2_fwd_params_set_cu_seqlens_knew(params_handle, cu_seqlens_k_new.data_ptr<int>());
    } else {
      flashmaskv2_fwd_params_set_knew_batch_stride(params_handle, k_new.stride(0));
      flashmaskv2_fwd_params_set_vnew_batch_stride(params_handle, v_new.stride(0));
    }
  }

  // 992 = 32 * 31 is the max supported batch in prepare_varlen_num_blocks
  // kernel
  const bool use_dynamic_split = is_varlen && flashmaskv2_fwd_params_get_b(params_handle) <= 992;
  // Temporarily set num_splits_dynamic_ptr to 1 since get_num_splits checks it
  flashmaskv2_fwd_params_set_num_splits_dynamic_ptr(params_handle, use_dynamic_split ? reinterpret_cast<int*>(1) : nullptr);

  flashmaskv2_fwd_params_set_pagedkv_tma(params_handle, flashmaskv2_get_pagedkv_tma(params_handle));
  num_splits = num_splits <= 0 ? flashmaskv2_get_num_splits(params_handle) : num_splits;
  flashmaskv2_fwd_params_set_num_splits(params_handle, num_splits);

  // Always enable PackGQA for Split, and get_pack_gqa requires
  // params.num_splits to decide
  const bool pack_gqa = manual_set_pack_gqa ? pack_gqa_ : flashmaskv2_get_pack_gqa(params_handle);
  flashmaskv2_fwd_params_set_pack_gqa(params_handle, pack_gqa);

  // This needs to be set after get_num_splits
  at::Tensor tile_count_semaphore;  // Contains the semaphore and optionally num_splits_dynamic
  // We don't use the persistent scheduler if Split and not Varlen
  const bool params_is_causal = flashmaskv2_fwd_params_get_is_causal(params_handle);
  const int params_num_splits = flashmaskv2_fwd_params_get_num_splits(params_handle);
  const int params_b = flashmaskv2_fwd_params_get_b(params_handle);
  const int params_arch = flashmaskv2_fwd_params_get_arch(params_handle);
  bool const scheduler_needs_semaphore =
      params_arch >= 90 ? true
                        : ((params_is_causal && !is_varlen) ||
                           (is_varlen && params_num_splits > 1));
  if (scheduler_needs_semaphore || use_dynamic_split) {
    int metadata_size = static_cast<int>(scheduler_needs_semaphore) + static_cast<int>(use_dynamic_split) * params_b;
    flashmaskv2_fwd_params_set_skip_scheduler_metadata_computation(params_handle, scheduler_metadata_.has_value());
    if (scheduler_metadata_.has_value()) {
      at::Tensor scheduler_metadata = scheduler_metadata_.value();
      CHECK_DEVICE(scheduler_metadata); CHECK_SHAPE(scheduler_metadata, metadata_size); CHECK_CONTIGUOUS(scheduler_metadata);
      TORCH_CHECK(scheduler_metadata.dtype() == torch::kInt32, "scheduler_metadata must have dtype int32");
      tile_count_semaphore = scheduler_metadata;
    } else {
      tile_count_semaphore = torch::empty({metadata_size}, opts.dtype(torch::kInt32));
    }
    if (scheduler_needs_semaphore && !use_dynamic_split) {
      tile_count_semaphore.zero_();
    }
    flashmaskv2_fwd_params_set_tile_count_semaphore(params_handle, scheduler_needs_semaphore ? (tile_count_semaphore.data_ptr<int>()) : nullptr);
    flashmaskv2_fwd_params_set_num_splits_dynamic_ptr(params_handle, use_dynamic_split ? (tile_count_semaphore.data_ptr<int>()) + 1 : nullptr);
  }

  if (q_v_.has_value()) {
    TORCH_CHECK(head_size <= 64, "q_v is only supported for head_size <= 64");
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16, "q_v is only supported for fp16 and bf16 data type");
    TORCH_CHECK(params_arch == 90, "q_v is only supported for Hopper GPUs");
    at::Tensor q_v = q_v_.value();
    TORCH_CHECK(q_v.dtype() == q_type, "q_v must have the same dtype as query");
    CHECK_DEVICE(q_v);
    TORCH_CHECK(q_v.stride(-1) == 1, "q_v tensor must have contiguous last dimension");
    if (is_varlen_q) {
      CHECK_SHAPE(q_v, total_q, num_heads, head_size_v);
    } else {
      CHECK_SHAPE(q_v, batch_size, seqlen_q, num_heads, head_size_v);
    }
    flashmaskv2_fwd_params_set_qv_ptr(params_handle,q_v.data_ptr());
    // All stride are in elements, not bytes.
    flashmaskv2_fwd_params_set_qv_row_stride(params_handle, q_v.stride(-3));
    flashmaskv2_fwd_params_set_qv_head_stride(params_handle, q_v.stride(-2));
    if (!is_varlen_q) {
      flashmaskv2_fwd_params_set_qv_batch_stride(params_handle, q_v.stride(0));
    }
  }

  if (rotary_cos_.has_value()) {
    TORCH_CHECK(k_new_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
    at::Tensor rotary_cos = rotary_cos_.value();
    CHECK_DEVICE(rotary_cos); CHECK_CONTIGUOUS(rotary_cos);
    int params_rotary_dim = rotary_cos.size(1) * 2;
    flashmaskv2_fwd_params_set_rotary_dim(params_handle, params_rotary_dim);
    TORCH_CHECK(params_rotary_dim <= head_size, "rotary_dim must be <= headdim");
    TORCH_CHECK(params_rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
    int64_t seqlen_ro = rotary_cos.size(0);
    if (paged_KV) {
      TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
    }
    CHECK_SHAPE(rotary_cos, seqlen_ro, params_rotary_dim / 2);
    TORCH_CHECK(rotary_cos.scalar_type() == q_type, "rotary_cos must have the same dtype as query");
    TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
    auto rotary_sin = rotary_sin_.value();
    CHECK_DEVICE(rotary_sin); CHECK_CONTIGUOUS(rotary_sin);
    CHECK_SHAPE(rotary_sin, seqlen_ro, params_rotary_dim / 2);
    TORCH_CHECK(rotary_sin.scalar_type() == q_type, "rotary_sin must have the same dtype as query");

    flashmaskv2_fwd_params_set_rotary_cos_ptr(params_handle, (rotary_cos.data_ptr()));
    flashmaskv2_fwd_params_set_rotary_sin_ptr(params_handle, (rotary_sin.data_ptr()));
    flashmaskv2_fwd_params_set_is_rotary_interleaved(params_handle, is_rotary_interleaved);
  } else {
    flashmaskv2_fwd_params_set_rotary_dim(params_handle, 0);
  }

  if (kv_batch_idx_.has_value()) {
    at::Tensor kv_batch_idx = kv_batch_idx_.value();
    CHECK_DEVICE(kv_batch_idx); CHECK_CONTIGUOUS(kv_batch_idx);
    TORCH_CHECK(kv_batch_idx.scalar_type() == torch::kInt32, "kv_batch_idx must have dtype int32");
    flashmaskv2_fwd_params_set_kv_batch_idx(params_handle, reinterpret_cast<int*>(kv_batch_idx.data_ptr()));
  }

  at::Tensor out_accum, softmax_lse_accum;
  auto outaccum_type = at::ScalarType::Float;
  if (flashmaskv2_fwd_params_get_num_splits(params_handle) > 1) {
    TORCH_CHECK(flashmaskv2_fwd_params_get_num_splits(params_handle) <=256, "num_splits > 256 not supported");
    if (is_varlen_q) {
      out_accum = torch::empty({flashmaskv2_fwd_params_get_num_splits(params_handle), num_heads, total_q, head_size_v}, opts.dtype(outaccum_type));
      softmax_lse_accum = torch::empty({flashmaskv2_fwd_params_get_num_splits(params_handle), num_heads, total_q}, opts.dtype(at::kFloat));
    } else {
      out_accum = torch::empty({flashmaskv2_fwd_params_get_num_splits(params_handle), batch_size, num_heads, seqlen_q, head_size_v}, opts.dtype(outaccum_type));
      softmax_lse_accum = torch::empty({flashmaskv2_fwd_params_get_num_splits(params_handle), batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
      flashmaskv2_fwd_params_set_oaccum_batch_stride(params_handle, out_accum.stride(1));
      flashmaskv2_fwd_params_set_lseaccum_batch_stride(params_handle, softmax_lse_accum.stride(1));
    }
    flashmaskv2_fwd_params_set_is_fp32(params_handle, false);
    flashmaskv2_fwd_params_set_oaccum_ptr(params_handle, out_accum.data_ptr());
    flashmaskv2_fwd_params_set_softmax_lseaccum_ptr(params_handle, softmax_lse_accum.data_ptr());
    flashmaskv2_fwd_params_set_oaccum_split_stride(params_handle, out_accum.stride(0));
    flashmaskv2_fwd_params_set_oaccum_row_stride(params_handle, out_accum.stride(-2));
    flashmaskv2_fwd_params_set_oaccum_head_stride(params_handle, out_accum.stride(-3));
    flashmaskv2_fwd_params_set_lseaccum_split_stride(params_handle, softmax_lse_accum.stride(0));
    flashmaskv2_fwd_params_set_lseaccum_head_stride(params_handle, softmax_lse_accum.stride(-2));
  }

  if (q_type == at::ScalarType::Float8_e4m3fn) {
    if (q_descale_.has_value()) {
      auto q_descale = q_descale_.value();
      CHECK_DEVICE(q_descale); CHECK_SHAPE(q_descale, batch_size, num_heads_k);
      flashmaskv2_fwd_params_set_q_descale_ptr(params_handle, (q_descale.data_ptr<float>()));
      flashmaskv2_fwd_params_set_q_descale_batch_stride(params_handle, q_descale.stride(0));
      flashmaskv2_fwd_params_set_q_descale_head_stride(params_handle, q_descale.stride(1));
    } else {
      flashmaskv2_fwd_params_set_q_descale_ptr(params_handle, nullptr);
    }
    if (k_descale_.has_value()) {
      auto k_descale = k_descale_.value();
      CHECK_DEVICE(k_descale); CHECK_SHAPE(k_descale, batch_size, num_heads_k);
      flashmaskv2_fwd_params_set_k_descale_ptr(params_handle, (k_descale.data_ptr<float>()));
      flashmaskv2_fwd_params_set_k_descale_batch_stride(params_handle, k_descale.stride(0));
      flashmaskv2_fwd_params_set_k_descale_head_stride(params_handle, k_descale.stride(1));
    } else {
      flashmaskv2_fwd_params_set_k_descale_ptr(params_handle, nullptr);
    }
    if (v_descale_.has_value()) {
      auto v_descale = v_descale_.value();
      CHECK_DEVICE(v_descale); CHECK_SHAPE(v_descale, batch_size, num_heads_k);
      flashmaskv2_fwd_params_set_v_descale_ptr(params_handle, (v_descale.data_ptr<float>()));
      flashmaskv2_fwd_params_set_v_descale_batch_stride(params_handle, v_descale.stride(0));
      flashmaskv2_fwd_params_set_v_descale_head_stride(params_handle, v_descale.stride(1));
    } else {
      flashmaskv2_fwd_params_set_v_descale_ptr(params_handle, nullptr);
    }
  }

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(!flashmaskv2_fwd_params_get_is_local(params_handle), "This flash attention build does not support local attention.");
#endif
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  TORCH_CHECK(flashmaskv2_fwd_params_get_softcap(params_handle) == 0.0, "This flash attention build does not support tanh softcapping.");
#endif
#ifdef FLASHATTENTION_DISABLE_SPLIT
  TORCH_CHECK(flashmaskv2_fwd_params_get_num_splits(params_handle) == 1, "This flash attention build does not support splits.");
#endif
#ifdef FLASHATTENTION_DISABLE_PACKGQA
  TORCH_CHECK(!flashmaskv2_fwd_params_get_pack_gqa(params_handle) || flashmaskv2_fwd_params_get_arch(params_handle) < 90 || (flashmaskv2_fwd_params_get_page_table(params_handle) && !flashmaskv2_fwd_params_get_pagedkv_tma(params_handle) || flashmaskv2_fwd_params_get_num_splits(params_handle) > 1),
               "This flash attention build does not support pack_gqa.");
#endif
#ifdef FLASHATTENTION_DISABLE_PAGEDKV
  TORCH_CHECK(!(flashmaskv2_fwd_params_get_page_table(params_handle) && !flashmaskv2_fwd_params_get_pagedkv_tma(params_handle)),"This flash attention build does not support paged KV.");
#endif
#ifdef FLASHATTENTION_DISABLE_APPENDKV
  TORCH_CHECK(!k_new_.has_value(), "This flash attention build does not support appending KV.");
#endif

  // flashmask
  at::Tensor startend_row_indices;
  if (is_flashmask) startend_row_indices = startend_row_indices_.value();
  at::Tensor block_mask;
  if (is_blockmask) block_mask = block_mask_.value();
  at::Tensor flashmask_maxmin, lt_start_row_indices, lt_end_row_indices, ut_start_row_indices, ut_end_row_indices;
  if (is_flashmask) {
    TORCH_CHECK(startend_row_indices.dim() == 4, "flashmask_attention receive startend_row_indices with dim [batch_size, num_heads,seq_len, mask_bounds]");
    TORCH_CHECK(startend_row_indices.size(3) == 1 || startend_row_indices.size(3) == 2 || startend_row_indices.size(3) == 4, "flashmask_attention startend_row_indices mask_bounds must in [1, 2, 4]");

    std::vector<int64_t> flashmask_maxmin_shape = startend_row_indices.sizes().vec();
    auto dprops = at::cuda::getCurrentDeviceProperties();
    const bool is_sm90 = (dprops->major == 9 && dprops->minor == 0);

    if (is_sm90) {
      // seqlen_k to nblock_seqlen, here we use kBlockN = 64
      // as a conservative estimation (reduce allocation size)
      flashmask_maxmin_shape[2] = ((flashmask_maxmin_shape[2] + 63) / 64 + 3) / 4 * 4;
      // make sure this is the same with FlashMaskV3 fwd main loop
      static constexpr int flashmask_buffer_length = 16 * 1024;
      // estimate the upper bound of the possible chunk size
      static constexpr int chunk_padded_length = ((flashmask_buffer_length + 63) / 64 + 31) & 0xffffffe0;
      static constexpr int chunk_valid_length = ((flashmask_buffer_length + 63) / 64 + 3) & 0xfffffffc;
      const int num_chunk = (flashmask_maxmin_shape[2] + chunk_valid_length - 1) / chunk_valid_length;
      flashmask_maxmin_shape[2] = num_chunk * chunk_padded_length;
    } else {
      // seqlen_k to nblock_seqlen
      flashmask_maxmin_shape[2] = ((flashmask_maxmin_shape[2] + 31) / 32 + 3) / 4 * 4;
    }
    flashmask_maxmin_shape[3] = 8;

    flashmask_maxmin = torch::empty(flashmask_maxmin_shape, opts.dtype(torch::kInt32));

    auto lt_start_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).contiguous();
    if (startend_row_indices.size(3) == 2) {  
      if (!is_causal) {  
        ut_end_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, 2)}).contiguous();
      } else {  
        lt_end_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, 2)}).contiguous();
      }  
    } else if (startend_row_indices.size(3) == 4) {  
      ut_end_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(3, 4)}).contiguous();
      lt_end_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, 2)}).contiguous();
      ut_start_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2, 3)}).contiguous(); 
    }
  }

  if (is_blockmask) {
    TORCH_CHECK(is_flashmask, "blockmask should be used with flashmask at the same time ");
    TORCH_CHECK(block_mask.dim() == 4, "blockmask receive blockmask_indices with dim [batch_size, num_heads, blocklen_q, blocklen_k]");
    TORCH_CHECK(block_mask.size(2)== (seqlen_q + 127) / 128, "blockmask is now only support blockdim_q = 128 ");
    TORCH_CHECK(block_mask.size(3) == (seqlen_k + 127) / 128, "blockmask is now only support blockdim_k = 128 ");
    TORCH_CHECK(block_mask.size(1) == startend_row_indices.size(1), "blockmask is now only support same dim num_heads with flashmask");
  }

  if (is_blockmask) {
    // xhy: blockmask is now only support blockdim_q k = 128
    flashmaskv2_fwd_params_set_m_block_dim(params_handle, 128);
    flashmaskv2_fwd_params_set_n_block_dim(params_handle, 128);
    flashmaskv2_fwd_params_set_block_mask_ptr(params_handle, (block_mask.data_ptr<int32_t>()));
  }

  if (is_flashmask) {
    if (lt_start_row_indices.defined() && lt_start_row_indices.data_ptr())
      flashmaskv2_fwd_params_set_lt_start_ptr(params_handle, (lt_start_row_indices.data_ptr<int32_t>()));
    else
      flashmaskv2_fwd_params_set_lt_start_ptr(params_handle, nullptr);

    if (lt_end_row_indices.defined() && lt_end_row_indices.data_ptr())
      flashmaskv2_fwd_params_set_lt_end_ptr(params_handle, (lt_end_row_indices.data_ptr<int32_t>()));
    else
      flashmaskv2_fwd_params_set_lt_end_ptr(params_handle, nullptr);

    if (ut_start_row_indices.defined() && ut_start_row_indices.data_ptr())
      flashmaskv2_fwd_params_set_ut_start_ptr(params_handle, (ut_start_row_indices.data_ptr<int32_t>()));
    else
      flashmaskv2_fwd_params_set_ut_start_ptr(params_handle, nullptr);

    if (ut_end_row_indices.defined() && ut_end_row_indices.data_ptr())
      flashmaskv2_fwd_params_set_ut_end_ptr(params_handle, (ut_end_row_indices.data_ptr<int32_t>()));
    else
      flashmaskv2_fwd_params_set_ut_end_ptr(params_handle, nullptr);

    if (flashmask_maxmin.defined() && flashmask_maxmin.data_ptr())
      flashmaskv2_fwd_params_set_flashmask_maxmin_ptr(params_handle, (flashmask_maxmin.data_ptr<int32_t>()));
    else
      flashmaskv2_fwd_params_set_flashmask_maxmin_ptr(params_handle, nullptr);

    flashmaskv2_fwd_params_set_h_flashmask(params_handle, startend_row_indices.size(1));
    flashmaskv2_fwd_params_set_h_h_flashmask_ratio(params_handle, num_heads / startend_row_indices.size(1));
  } else {
    flashmaskv2_fwd_params_set_lt_start_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_lt_end_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_ut_start_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_ut_end_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_flashmask_maxmin_ptr(params_handle, nullptr);
    flashmaskv2_fwd_params_set_h_flashmask(params_handle, 0);
    flashmaskv2_fwd_params_set_h_h_flashmask_ratio(params_handle, 0);
  }

  if (total_q > 0 && (total_k + flashmaskv2_fwd_params_get_total_knew(params_handle)) > 0 && num_heads_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // print_flash_fwd_params(*params_handle);
    flashmaskv2_run_mha_fwd(params_handle, stream);
    if (flashmaskv2_fwd_params_get_num_splits(params_handle) > 1) {
      if (out_type == at::ScalarType::BFloat16) {
        // Since we want output in BF16. Otherwise fwd_combine will output to FP16
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
      flashmaskv2_run_mha_fwd_combine(params_handle, stream, true /*enable_pdl*/);
    } // no elif, diff from https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp#L1186
  } else if (total_q > 0 && num_heads_k > 0) {
    TORCH_CHECK(out.dtype() == torch::kBFloat16 || out.dtype() == torch::kFloat16 || out.dtype() == torch::kFloat8_e4m3fn,
                "flash attention 3 supports bfloat16, float16 and float8_e4m3fn only.");
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
  }
  return {out, softmax_lse, out_accum, softmax_lse_accum};
}


// b: batch_size
// s_q: seqlen_q
// s_k: seqlen_k
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
FlashMaskV2GradBaseKernel(const at::Tensor &dout,        // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
                          const at::Tensor &q,           // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
                          const at::Tensor &k,           // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
                          const at::Tensor &v,           // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
                          const at::Tensor &out,         // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
                          const at::Tensor &softmax_lse, // (b, h, s_q) or (h, total_q) if there is cu_seqlens_q
                          const c10::optional<at::Tensor> &dq_,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
                          const c10::optional<at::Tensor> &dk_,  // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
                          const c10::optional<at::Tensor> &dv_,  // (b, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k
                          const c10::optional<at::Tensor> &cu_seqlens_q_,  // b + 1
                          const c10::optional<at::Tensor> &cu_seqlens_k_,  // b + 1
                          const c10::optional<at::Tensor> &seqused_q_,     // b. If given, only this many elements of each batch element's queries and outputs are used.
                          const c10::optional<at::Tensor> &seqused_k_,     // b. If given, only this many elements of each batch element's keys are used.
                          const c10::optional<at::Tensor> &startend_row_indices_,
                          const c10::optional<at::Tensor> &block_mask_,    // （(b,h,s//128,s//128)
                          std::optional<int64_t> max_seqlen_q_,
                          std::optional<int64_t> max_seqlen_k_,
                          std::optional<double> softmax_scale_,
                          bool is_causal,
                          int64_t window_size_left,
                          int64_t window_size_right,
                          double const softcap,
                          const bool deterministic,
                          const int64_t sm_margin) {

  auto dprops = at::cuda::getCurrentDeviceProperties();
  const bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
  TORCH_CHECK(is_sm90, "FlashAttention-3 only supports Hopper GPUs.");

  auto q_type = q.dtype();
  TORCH_CHECK(q_type == torch::kFloat16 || q_type == torch::kBFloat16, "FlashAttention-3 bwd only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_type, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_type, "query and out must have the same dtype");
  TORCH_CHECK(dout.dtype() == q_type, "query and dout must have the same dtype");

  CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
  CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

  TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(out.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(dout.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  at::Tensor cu_seqlens_q;
  const bool is_varlen_q = cu_seqlens_q_.has_value();
  if (is_varlen_q) {
    cu_seqlens_q = cu_seqlens_q_.value();
    CHECK_DEVICE(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_q);
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
    TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
  }
  at::Tensor cu_seqlens_k;
  const bool is_varlen_k = cu_seqlens_k_.has_value();
  if (is_varlen_k) {
    cu_seqlens_k = cu_seqlens_k_.value();
    CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
    TORCH_CHECK(cu_seqlens_k.dtype() ==torch::kInt32, "cu_seqlens_k must have dtype torch.int32");
    TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
  }
  // This is what we will template on
  bool const is_varlen = is_varlen_q || is_varlen_k || seqused_q_.has_value() || seqused_k_.has_value();
#ifdef FLASHATTENTION_DISABLE_VARLEN
  TORCH_CHECK(!is_varlen, "This flash attention build does not support varlen.");
#endif

  const auto sizes = q.sizes();
  const int batch_size = is_varlen_q ? cu_seqlens_q.size(0) - 1 : sizes[0];
  const int seqlen_q = is_varlen_q ? max_seqlen_q_.value() : sizes[1];
  const int total_q = is_varlen_q ? sizes[0] : batch_size * sizes[1];
  const int num_heads = q.size(-2);
  const int head_size = q.size(-1);
  const int seqlen_k = is_varlen_k ? max_seqlen_k_.value() : k.size(1);
  const int total_k = is_varlen_k ? k.size(0) : batch_size * k.size(1);
  const int num_heads_k = k.size(-2);
  TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
  const int max_headdim = flashmaskv2_get_max_headdim();
  TORCH_CHECK(head_size <= max_headdim, "FlashMask forward only supports head dimension at most " + std::to_string(max_headdim));
  TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
  double softmax_scale = 1.0 / sqrt(double(head_size));
  if (softmax_scale_.has_value()) {
      softmax_scale = softmax_scale_.value();
  }
  // This needs to go before kBlockM & kBlockN since we rely on the correct
  // window_size and is_causal to set kBlockM
  window_size_left = window_size_left >= seqlen_k - 1 ? -1 : window_size_left;
  window_size_right = window_size_right >= seqlen_q - 1 ? -1 : window_size_right;
  window_size_right = is_causal ? 0 : window_size_right;
  // There's a case where is_causal=false, window_size=(-1, 0). Then set_params_bprop will set params.is_causal=true.
  // If we don't have is_causal here matching params.is_causal, we might get the wrong kBlockM (and cause IMA).
  is_causal = window_size_left < 0 && window_size_right == 0;

  const int arch = dprops->major * 10 + dprops->minor;
  const int head_size_rounded = flashmaskv2_round_up_headdim(head_size);
  // Very important that these match the kernel configs
  const bool is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
  const bool is_flashmask = startend_row_indices_.has_value();
  at::Tensor startend_row_indices;
  if (is_flashmask) startend_row_indices = startend_row_indices_.value();
  const bool has_softcap = softcap > 0.0;
  auto opts = q.options();

  // flashmask
  at::Tensor flashmask_maxmin, lt_start_row_indices, lt_end_row_indices, ut_start_row_indices, ut_end_row_indices;
  if (is_flashmask) {
    TORCH_CHECK(startend_row_indices.dtype() == torch::kInt32, "flashmask_attention startend_row_indices must be INT32 type");
    TORCH_CHECK(startend_row_indices.dim() == 4, "flashmask_attention receive startend_row_indices with dim [batch_size, num_heads,seq_len, mask_bounds]");
    TORCH_CHECK(startend_row_indices.size(3) == 1 || startend_row_indices.size(3) == 2 || startend_row_indices.size(3) == 4, "flashmask_attention startend_row_indices mask_bounds must in [1, 2, 4]");

    std::vector<int64_t> flashmask_maxmin_shape = startend_row_indices.sizes().vec();
    flashmask_maxmin_shape[2] = ((flashmask_maxmin_shape[2] + 31) / 32 + 3) / 4 * 4;
    flashmask_maxmin_shape[3] = 8;
    flashmask_maxmin = torch::empty(flashmask_maxmin_shape, opts.dtype(torch::kInt32));

    auto lt_start_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).contiguous();
    if (startend_row_indices.size(3) == 2) {  
      if (is_causal) {  
        lt_end_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, 2)}).contiguous();
      } else {  
        ut_end_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, 2)}).contiguous();
      }  
    } else if (startend_row_indices.size(3) == 4) {  
      ut_end_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(3, 4)}).contiguous();
      lt_end_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, 2)}).contiguous();
      ut_start_row_indices = startend_row_indices.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2, 3)}).contiguous();
    }
  }

  const bool is_blockmask = block_mask_.has_value();
  at::Tensor block_mask;
  if (is_blockmask) block_mask = block_mask_.value();

  if (is_blockmask) {
    TORCH_CHECK(is_flashmask, "blockmask should be used with flashmask at the same time");
    TORCH_CHECK(block_mask.dim() == 4, "blockmask receive blockmask_indices with dim [batch_size, num_heads, blocklen_q, blocklen_k]");
    TORCH_CHECK(block_mask.size(2) == (seqlen_q + 127) / 128, "blockmask only supports blockdim_q = 128 now");
    TORCH_CHECK(block_mask.size(3) == (seqlen_k + 127) / 128, "blockmask only supports blockdim_k = 128 now");

    TORCH_CHECK(block_mask.size(1) == startend_row_indices.size(1), "blockmask only supports same dim num_heads with flashmask now");
    TORCH_CHECK(seqlen_k <= 1024 * 128, "blockmask only supports seqlen <= 128k in bwd now");
    TORCH_CHECK(seqlen_q <=1024 * 128, "blockmask only supports seqlen <= 128k in bwd now");
  }

  const bool has_lt_end = lt_end_row_indices.defined() && lt_end_row_indices.data_ptr();
  const bool has_ut_start = ut_start_row_indices.defined() && ut_start_row_indices.data_ptr();

  // The tile dispatch for flashmask is now different from fa3. Replacing the original ternary operator with lambda makes the code
  // easier to reason about and less error-prone.
  const auto [kBlockM_sm90, kBlockN_sm90] = [&]() -> std::pair<int, int> {
    if (head_size_rounded <= 64) {
      if (is_flashmask && !is_causal) {
        return std::pair<int,int>{64, 96};
      } else {
        return ((is_causal && has_softcap) || is_flashmask) ? std::pair<int,int>{96, 128} : std::pair<int,int>{128, 128};
      }
    } else if (head_size_rounded <= 128) {
      // by now, we reuse template instantiation of head dim 128 for head dim in range (64, 128],
      // and therefore no separate dispatch for head dim in range (64, 96]
      if (is_causal || is_local || has_softcap) {
        return std::pair<int,int>{64, 128};
      } else {
        return ((seqlen_q >= 1024 || seqlen_k >= 1024) && !(has_lt_end && has_ut_start)) ? std::pair<int,int>{64, 128} : std::pair<int,int>{64, 64};
      }
    } else if (head_size_rounded <= 256) {
      // by now, we reuse template instantiation of head dim 256 for head dim in range (128, 256],
      // and therefore no separate dispatch for head dim in range (128, 192]
      return has_lt_end && has_ut_start ? std::pair<int,int>{64, 32} : std::pair<int,int>{64, 64};
    } else {
      TORCH_CHECK(false, "head dim is rounded to " + std::to_string(head_size_rounded) + ", which is not supported in FlashMask V3 now.");
      return std::pair<int,int>{0, 0};
    }
  }();

  const int kBlockM_sm80 = head_size_rounded <= 64 ? 128 : 64;
  const int kBlockM_sm86 = head_size_rounded <= 192 ? 64 : 32;
  const int kBlockM = arch >= 90 ? kBlockM_sm90 : (arch == 86 || arch == 89 ? kBlockM_sm86 : kBlockM_sm80);
  const int kBlockN_sm80 = head_size_rounded <= 128 ? 128 : (head_size_rounded <= 192 ? 80 : 64);
  const int kBlockN_sm86 = head_size_rounded <= 64 ? 128 : (head_size_rounded <= 96 ? 128 : (head_size_rounded <= 128 ? 96 : (head_size_rounded <= 192 ? 64 : 64)));
  const int kBlockN = arch >= 90 ? kBlockN_sm90 : (arch == 86 || arch == 89 ? kBlockN_sm86 : kBlockN_sm80);
  
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
  const int seqlen_k_rounded = round_multiple(seqlen_k, kBlockN);
  const int total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockM, kBlockM);
  const int total_k_padded_rounded = round_multiple(total_k + batch_size * kBlockN, kBlockN);

  if (is_varlen_q) {
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  } else {
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);
  }
  if (is_varlen_k) {
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
  } else {
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
  }


  if (seqused_q_.has_value()) {
    auto seqused_q = seqused_q_.value();
    TORCH_CHECK(seqused_q.dtype() == torch::kInt32, "seqused_q must have dtype int32");
    CHECK_DEVICE(seqused_q); CHECK_CONTIGUOUS(seqused_q);
    CHECK_SHAPE(seqused_q, batch_size);
  }
  if (seqused_k_.has_value()) {
    auto seqused_k = seqused_k_.value();
    TORCH_CHECK(seqused_k.dtype() == torch::kInt32, "seqused_k must have dtype int32");
       CHECK_DEVICE(seqused_k); CHECK_CONTIGUOUS(seqused_k);
    CHECK_SHAPE(seqused_k, batch_size);
  }

  at::Tensor dq, dk, dv;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.dtype() == q_type,"dq must have the same dtype as q");
    CHECK_DEVICE(dq);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
    if (!is_varlen_q) {
      CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
      CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    }
  } else {
    dq = torch::empty_like(q);
  }

  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == q_type, "dk must have the same dtype as q");
    CHECK_DEVICE(dk);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
    if (is_varlen_k) {
      CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
      CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    }
  } else {
    dk = torch::empty_like(k);
  }

  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == q_type, "dv must have the same dtype as q");
    CHECK_DEVICE(dv);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
    if (is_varlen_k) {
      CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
      CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    }
  } else {
    dv = torch::empty_like(v);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  auto device_guard = make_cuda_guard_from_tensor(q);
  // Need softmax_d to have total_q_padded_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
  at::Tensor softmax_d, softmax_lse_log2;
  if (is_varlen) {
      softmax_d = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
      softmax_lse_log2 = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
  } else {
    // Need softmax_d to have total_q_padded_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
      softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
      softmax_lse_log2 = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
  }
  
  at::Tensor dq_accum, dk_accum, dv_accum;
  if (is_varlen) {
      dq_accum = torch::empty({num_heads, total_q_padded_rounded * head_size_rounded}, opts.dtype(at::kFloat));
  } else {
      dq_accum = torch::empty({batch_size, num_heads, seqlen_q_rounded * head_size_rounded}, opts.dtype(at::kFloat));
  }
  if (num_heads_k != num_heads) {  // MQA / GQA
      if (is_varlen) {
          dk_accum = torch::zeros({num_heads_k, total_k_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
          dv_accum = torch::zeros({num_heads_k, total_k_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
      } else {
          dk_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, opts.dtype(at::kFloat));
          dv_accum = torch::zeros({batch_size, num_heads_k, seqlen_k_rounded * head_size_rounded}, opts.dtype(at::kFloat));
      }   
  }

  FlashMask_bwd_params *params_handle = get_flashmask_bwd_params_handle();
  flashmaskv2_clear_bwd_params_handle(params_handle);
  set_flashmaskv2_params_dgrad(params_handle,
                               batch_size,
                               seqlen_q, seqlen_k,
                               seqlen_q_rounded, seqlen_k_rounded,
                               num_heads, num_heads_k,
                               head_size, head_size_rounded,
                               q, k, v, out,
                               dout, dq, dk, dv,
                               is_varlen_q ? cu_seqlens_q.data_ptr() : nullptr,
                               is_varlen_k ? cu_seqlens_k.data_ptr() : nullptr,
                               seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
                               seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                               dq_accum.data_ptr(),
                               num_heads_k != num_heads ? dk_accum.data_ptr() : nullptr,
                               num_heads_k != num_heads ? dv_accum.data_ptr() : nullptr,
                               softmax_lse.data_ptr(),
                               softmax_d.data_ptr(),
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
  flashmaskv2_bwd_params_set_softmax_lse_log2_ptr(params_handle, softmax_lse_log2.data_ptr());
  flashmaskv2_bwd_params_set_dv(params_handle, head_size);  // We don't support hdim_v being different from hdim_qk for now

  at::Tensor tile_count_semaphore;
  if (arch >= 90) {
    tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
    flashmaskv2_bwd_params_set_tile_count_semaphore(params_handle, tile_count_semaphore.data_ptr<int>());
  } else {
    flashmaskv2_bwd_params_set_tile_count_semaphore(params_handle, nullptr);
  }

  at::Tensor dq_semaphore = torch::empty({(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
  flashmaskv2_bwd_params_set_dq_semaphore(params_handle, dq_semaphore.data_ptr<int>());

  at::Tensor dk_semaphore, dv_semaphore;
  if (num_heads_k != num_heads && flashmaskv2_bwd_params_get_deterministic(params_handle)) {
    dk_semaphore = torch::zeros({(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
    dv_semaphore = torch::zeros({(seqlen_k + kBlockN - 1) / kBlockN, batch_size, num_heads_k}, opts.dtype(torch::kInt32));
    flashmaskv2_bwd_params_set_dk_semaphore(params_handle, dk_semaphore.data_ptr<int>());
    flashmaskv2_bwd_params_set_dv_semaphore(params_handle, dv_semaphore.data_ptr<int>());
  }

  if (is_flashmask) {
    if (lt_start_row_indices.defined() && lt_start_row_indices.data_ptr())
      flashmaskv2_bwd_params_set_lt_start_ptr(params_handle, (lt_start_row_indices.data_ptr<int32_t>()));
    else
      flashmaskv2_bwd_params_set_lt_start_ptr(params_handle, nullptr);

    if (lt_end_row_indices.defined() && lt_end_row_indices.data_ptr())
      flashmaskv2_bwd_params_set_lt_end_ptr(params_handle, (lt_end_row_indices.data_ptr<int32_t>()));
    else
      flashmaskv2_bwd_params_set_lt_end_ptr(params_handle, nullptr);

    if (ut_start_row_indices.defined() && ut_start_row_indices.data_ptr())
      flashmaskv2_bwd_params_set_ut_start_ptr(params_handle, (ut_start_row_indices.data_ptr<int32_t>()));
    else
      flashmaskv2_bwd_params_set_ut_start_ptr(params_handle, nullptr);

    if (ut_end_row_indices.defined() && ut_end_row_indices.data_ptr())
      flashmaskv2_bwd_params_set_ut_end_ptr(params_handle, (ut_end_row_indices.data_ptr<int32_t>()));
    else
      flashmaskv2_bwd_params_set_ut_end_ptr(params_handle, nullptr);

    if (flashmask_maxmin.defined() && flashmask_maxmin.data_ptr())
      flashmaskv2_bwd_params_set_flashmask_maxmin_ptr(params_handle, (flashmask_maxmin.data_ptr<int32_t>()));
    else
      flashmaskv2_bwd_params_set_flashmask_maxmin_ptr(params_handle, nullptr);

    flashmaskv2_bwd_params_set_h_flashmask(params_handle, startend_row_indices.size(1));
    flashmaskv2_bwd_params_set_h_h_flashmask_ratio(params_handle, num_heads / startend_row_indices.size(1));
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
    flashmaskv2_bwd_params_set_block_mask_ptr(params_handle, (block_mask.data_ptr<int32_t>()));
  }
#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(!flashmaskv2_bwd_params_get_is_local(params_handle), "This flash attention build does not support local attention.");
#endif
#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  TORCH_CHECK(flashmaskv2_bwd_params_get_softcap(params_handle) == 0.0, "This flash attention build does not support tanh softcapping.");
#endif

  if (total_q > 0 && total_k > 0 && num_heads_k > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    flashmaskv2_run_mha_bwd(params_handle, stream);
  } else if (total_k > 0 && num_heads_k > 0) {
    // If seqlen_q == 0, then we have an empty tensor. We need to set the output
    // to 0.
    dk.zero_();
    dv.zero_();
    softmax_d.zero_();
  } else if (total_q > 0 && num_heads_k > 0) {
    dq.zero_();
    softmax_d.zero_();
  }
  return { softmax_d, softmax_lse_log2, dq_accum, dk_accum, dv_accum };
}

TORCH_LIBRARY(flash_mask, m) {
    m.def("fwd("
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor(k_new!)? k_new = None,"
        "Tensor(v_new!)? v_new = None,"
        "Tensor? q_v = None,"
        "Tensor(out!)? out = None,"
        "Tensor? cu_seqlens_q = None,"
        "Tensor? cu_seqlens_k = None,"
        "Tensor? cu_seqlens_k_new = None,"
        "Tensor? seqused_q = None,"
        "Tensor? seqused_k = None,"
        "int? max_seqlen_q = 0,"
        "int? max_seqlen_k = 0,"
        "Tensor? page_table = None,"
        "Tensor? kv_batch_idx = None,"
        "Tensor? leftpad_k = None,"
        "Tensor? rotary_cos = None,"
        "Tensor? rotary_sin = None,"
        "Tensor? q_descale = None,"
        "Tensor? k_descale = None,"
        "Tensor? v_descale = None,"
        "Tensor? scheduler_metadata = None,"
        "Tensor? startend_row_indices = None,"
        "Tensor? block_mask = None,"
        "float? softmax_scale = None,"
        "bool is_causal = False,"
        "int window_size_left = -1,"
        "int window_size_right = -1,"
        "float softcap = 0.0,"
        "bool is_rotary_interleaved = True,"
        "int num_splits = 1,"
        "bool manual_set_pack_gqa = False,"
        "bool pack_gqa = False,"
        "int sm_margin = 0) -> (Tensor(out!), Tensor, Tensor, Tensor)");
    m.def("bwd("
        "Tensor dout,"
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor out,"
        "Tensor softmax_lse,"
        "Tensor(dq!)? dq = None,"
        "Tensor(dk!)? dk = None,"
        "Tensor(dv!)? dv = None,"
        "Tensor? cu_seqlens_q = None,"
        "Tensor? cu_seqlens_k = None,"
        "Tensor? seqused_q = None,"
        "Tensor? seqused_k = None,"
        "Tensor? startend_row_indices = None,"
        "Tensor? block_mask = None,"
        "int? max_seqlen_q = None,"
        "int? max_seqlen_k = None,"
        "float? softmax_scale = None,"
        "bool is_causal = False,"
        "int window_size_left = -1,"
        "int window_size_right = -1,"
        "float softcap = 0.0,"
        "bool deterministic = False,"
        "int sm_margin = 0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_mask, CUDA, m) {
    m.impl("fwd", &FlashMaskV2BaseKernel);
    m.impl("bwd", &FlashMaskV2GradBaseKernel);
}
