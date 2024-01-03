/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include "flash_attn_with_bias_mask.h"
#include "src/flash.h"
#include "src/static_switch.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "dlfcn.h"
#include "math.h"
#include "src/cuda_utils.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

#include <cmath>
#include <limits>

#include <memory>
#include <mutex>
#include <stdexcept>

#include <cstring>
#include <exception>
#include <string>

#define ASSERT_CHECK(__cond)                             \
      do {                                               \
        const bool __cond_var = (__cond);                \
        if (!__cond_var) {                               \
          ::std::string __err_msg = ::std::string("`") + \
                #__cond + "` check failed at " +         \
                __FILE__ + ":" +                         \
                ::std::to_string(__LINE__);              \
          throw std::runtime_error(__err_msg);           \
        }                                                \
      } while (0)

#ifdef __cplusplus
extern "C" {
#endif

static thread_local std::unique_ptr<char[]> flash_attn_err_msg;

void flash_attn_set_error(const char *msg) {
  if (msg == nullptr || *msg == '\0') {
    msg = "unknown error";
  }

  auto n = strlen(msg);
  std::unique_ptr<char[]> new_err_msg(new char[n+1]);
  std::strcpy(new_err_msg.get(), msg);
  flash_attn_err_msg = std::move(new_err_msg);
}

const char *flash_attn_error() {
  return flash_attn_err_msg.get();
}

#ifdef __cplusplus
}
#endif

#define FLASHATTNLIB_BEGIN_FUNC try {
#define FLASHATTNLIB_END_FUNC } catch (::std::exception &__e) { flash_attn_set_error(__e.what()); return false; } catch (...) { flash_attn_set_error(nullptr); return false; }

#define CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                     \
      auto dprops = at::cuda::getCurrentDeviceProperties();              \
      const bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;     \
      const bool is_sm90 = dprops->major == 9 && dprops->minor == 0;     \
      ASSERT_CHECK(is_sm8x || is_sm90);                                  \
      ASSERT_CHECK(batch_size > 0);                                      \
      ASSERT_CHECK(head_size % 8 == 0);                                  \
      ASSERT_CHECK(head_size <= 256);                                    \
      ASSERT_CHECK(num_heads % num_heads_k == 0);                        \
      if (attn_mask) {                                                   \
          ASSERT_CHECK(mask_dims[0] == batch_size);                      \
          ASSERT_CHECK(mask_dims[1] == 1 || mask_dims[1] == num_heads);  \
          ASSERT_CHECK(mask_dims[2] == 1 || mask_dims[2] == __seqlen_q); \
          ASSERT_CHECK(mask_dims[3] == __seqlen_k);                      \
      }

#define CHECK_BWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                       \
      CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                         \
      const bool is_sm80 = dprops->major == 8 && dprops->minor == 0;                       \
      if (head_size > 192) {                                                               \
          /* FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800 */ \
          ASSERT_CHECK(is_sm80 || is_sm90);                                                \
      }

void set_params_fprop(Flash_fwd_params &params,
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
                      void * const q,
                      void * const k,
                      void * const v,
                      void * const out,
                      void * const cu_seqlens_q_d,
                      void * const cu_seqlens_k_d,
                      void * const p_d,
                      void * const softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      float softmax_unscale,
                      bool is_causal,
                      bool is_bf16,
                      void * attn_mask = nullptr,
                      int mask_head_mod_size = 0,
                      int mask_seq_q_mod_size = 0) {
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = is_bf16;
    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    // All stride are in elements, not bytes.
    params.q_row_stride = h * d;
    params.k_row_stride = h_k * d;
    params.v_row_stride = h_k * d;
    params.q_head_stride = d;
    params.k_head_stride = d;
    params.v_head_stride = d;
    params.o_ptr = out;
    params.o_row_stride = h * d;
    params.o_head_stride = d;

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = seqlen_q * h * d;
        params.k_batch_stride = seqlen_k * h_k * d;
        params.v_batch_stride = seqlen_k * h_k * d;
        params.o_batch_stride = seqlen_q * h * d;
    }


    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // attn mask
    params.attn_mask_ptr = attn_mask;
    params.mask_head_mod_size = mask_head_mod_size;
    params.mask_seq_q_mod_size = mask_seq_q_mod_size;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.unscale_softmax = softmax_unscale;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    ASSERT_CHECK(p_dropout < 1.f);

    params.is_causal = is_causal;
}

void set_params_dgrad(Flash_bwd_params &params,
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
                      void * const q,
                      void * const k,
                      void * const v,
                      void * const out,
                      void * const dout,
                      void * const dq,
                      void * const dk,
                      void * const dv,
                      void * const cu_seqlens_q_d,
                      void * const cu_seqlens_k_d,
                      void * const dq_accum_d,
                      void * const dk_accum_d,
                      void * const dv_accum_d,
                      void * const softmax_lse_d,
                      void * const dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      float softmax_unscale,
                      bool is_causal,
                      bool is_bf16,
                      const int num_splits = 0,
                      void * attn_mask = nullptr,
                      int mask_head_mod_size = 0,
                      int mask_seq_q_mod_size = 0) {

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     is_causal,
                     is_bf16,
                     attn_mask,
                     mask_head_mod_size,
                     mask_seq_q_mod_size);

    // Set the pointers and strides.
    params.do_ptr = dout;
    params.do_row_stride = h * d;
    params.do_head_stride = d;
    params.dq_ptr = dq;
    params.dk_ptr = dk;
    params.dv_ptr = dv;
    params.dq_row_stride = h * d;
    params.dk_row_stride = h * d;
    params.dv_row_stride = h * d;
    params.dq_head_stride = d;
    params.dk_head_stride = d;
    params.dv_head_stride = d;

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = seqlen_q * h * d;
        params.dq_batch_stride = seqlen_q * h * d;
        params.dk_batch_stride = seqlen_k * h * d;
        params.dv_batch_stride = seqlen_k * h * d;
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
    params.num_splits = num_splits;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        FWD_HEADDIM_SWITCH(params.d, [&] {
            run_mha_fwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

#ifdef __cplusplus
extern "C" {
#endif

bool flash_attn_fwd(const void * const q,
                    const void * const k,
                    const void * const v,
                    void * const rng_state,
                    void * const out,
                    void * const softmax_ptr,
                    void * const softmax_lse_ptr,
                    const int batch_size,
                    const int seqlen_q,
                    const int seqlen_k,
                    const int seqlen_q_rounded,
                    const int seqlen_k_rounded,
                    const int num_heads,
                    const int num_heads_k,
                    const int head_size,
                    const int head_size_rounded,
                    const float p_dropout,
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool return_softmax,
                    const bool is_bf16,
                    cudaStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    CHECK_FWD_EXECTUABLE(seqlen_q, seqlen_k)

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     const_cast<void *>(q),
                     const_cast<void *>(k),
                     const_cast<void *>(v),
                     out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     return_softmax ? softmax_ptr : nullptr,
                     softmax_lse_ptr,
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     is_causal,
                     is_bf16,
                     const_cast<void *>(attn_mask),
                     mask_head_mod_size,
                     mask_seq_q_mod_size);

    params.rng_state = static_cast<uint64_t*>(rng_state);

    if (is_dropout) {
        // number of times random will be generated per thread, to offset philox counter in thc random
        // state
        // We use a custom RNG that increases the offset by batch_size * nheads * 32.
        params.philox_args = at::PhiloxCudaState(seed, offset);
    }

    run_mha_fwd(params, stream);
    
    return true;

    FLASHATTNLIB_END_FUNC
}

bool flash_attn_varlen_fwd(const void * const q,
                           const void * const k,
                           const void * const v,
                           const int32_t * const cu_seqlens_q,
                           const int32_t * const cu_seqlens_k,
                           void * const rng_state,
                           void * const out,
                           void * const softmax_ptr,
                           void * const softmax_lse_ptr,
                           const int batch_size,
                           const int max_seqlen_q,
                           const int max_seqlen_k,
                           const int seqlen_q_rounded,
                           const int seqlen_k_rounded,
                           const int num_heads,
                           const int num_heads_k,
                           const int head_size,
                           const int head_size_rounded,
                           const float p_dropout,
                           const float softmax_scale,
                           const float softmax_unscale,
                           const bool is_causal,
                           const bool return_softmax,
                           const bool is_bf16,
                           cudaStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * const attn_mask,
                           const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    CHECK_FWD_EXECTUABLE(max_seqlen_q, max_seqlen_k)
    
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     const_cast<void *>(q),
                     const_cast<void *>(k),
                     const_cast<void *>(v),
                     out,
                     const_cast<int32_t *>(cu_seqlens_q),
                     const_cast<int32_t *>(cu_seqlens_k),
                     return_softmax ? softmax_ptr : nullptr,
                     softmax_lse_ptr,
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     is_causal,
                     is_bf16,
                     const_cast<void *>(attn_mask),
                     mask_head_mod_size,
                     mask_seq_q_mod_size);
    
    params.rng_state = static_cast<uint64_t*>(rng_state);

    if (is_dropout) {
        params.philox_args = at::PhiloxCudaState(seed, offset);
    }

    run_mha_fwd(params, stream);

    return true;
    
    FLASHATTNLIB_END_FUNC
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
    FP16_SWITCH(!params.is_bf16, [&] {
        if (params.d <= 32) {
            run_mha_bwd_<elem_type, 32>(params, stream, configure);
        } else if (params.d <= 64) {
            run_mha_bwd_<elem_type, 64>(params, stream, configure);
        } else if (params.d <= 96) {
            run_mha_bwd_<elem_type, 96>(params, stream, configure);
        } else if (params.d <= 128) {
            run_mha_bwd_<elem_type, 128>(params, stream, configure);
        } else if (params.d <= 160) {
            run_mha_bwd_<elem_type, 160>(params, stream, configure);
        } else if (params.d <= 192) {
            run_mha_bwd_<elem_type, 192>(params, stream, configure);
        } else if (params.d <= 224) {
          run_mha_bwd_<elem_type, 224>(params, stream, configure);
        } else if (params.d <= 256) {
          run_mha_bwd_<elem_type, 256>(params, stream, configure);
        }
    });
}

bool flash_attn_bwd(const void * const dout,
                    const void * const q,
                    const void * const k,
                    const void * const v,
                    const void * const out,
                    const void * const softmax_d,
                    const void * const softmax_lse,
                    void * const rng_state,
                    void * const dq,
                    void * const dk,
                    void * const dv,
                    void * const dq_accum,
                    const int batch_size,
                    const int seqlen_q,
                    const int seqlen_k,
                    const int seqlen_q_rounded,
                    const int seqlen_k_rounded,
                    const int num_heads,
                    const int num_heads_k,
                    const int head_size,
                    const int head_size_rounded,
                    const float p_dropout,
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool is_bf16,
                    const int num_splits,
                    cudaStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    CHECK_BWD_EXECTUABLE(seqlen_q, seqlen_k)

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    const bool loop = true;

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     const_cast<void *>(q),
                     const_cast<void *>(k),
                     const_cast<void *>(v),
                     const_cast<void *>(out),
                     const_cast<void *>(dout),
                     dq,
                     dk,
                     dv,
                     nullptr,
                     nullptr,
                     loop ? dq_accum : nullptr,
                     nullptr,
                     nullptr,
                     const_cast<void *>(softmax_lse),
                     const_cast<void *>(softmax_d),
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     is_causal,
                     is_bf16,
                     num_splits,
                     const_cast<void *>(attn_mask),
                     mask_head_mod_size,
                     mask_seq_q_mod_size);

    auto launch = &run_mha_bwd;
    
    if (is_dropout) {
        params.philox_args = at::PhiloxCudaState(seed, offset);
        // seems a wild pointer at fa2: https://github.com/PaddlePaddle/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp#L690-L691
        params.rng_state = static_cast<uint64_t*>(rng_state);
        uint64_t rng_state_data[2] = {seed, offset};
        cudaMemcpyAsync(params.rng_state, rng_state_data, 2*sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    }

    launch(params, stream, /*configure=*/false);
    
    return true;
    
    FLASHATTNLIB_END_FUNC

}

bool flash_attn_varlen_bwd(const void * const dout,
                           const void * const q,
                           const void * const k,
                           const void * const v,
                           const void * const out,
                           const void * const softmax_d,
                           const void * const softmax_lse,
                           const int32_t * const cu_seqlens_q,
                           const int32_t * const cu_seqlens_k,
                           void * const rng_state,
                           void * const dq,
                           void * const dk,
                           void * const dv,
                           void * const dq_accum,
                           const int batch_size,
                           const int max_seqlen_q,
                           const int max_seqlen_k,
                           const int seqlen_q_rounded,
                           const int seqlen_k_rounded,
                           const int num_heads,
                           const int num_heads_k,
                           const int head_size,
                           const int head_size_rounded,
                           const float p_dropout,
                           const float softmax_scale,
                           const float softmax_unscale,
                           const bool is_causal,
                           const bool is_bf16,
                           const int num_splits,
                           cudaStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * const attn_mask,
                           const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    const bool loop = true;

    CHECK_BWD_EXECTUABLE(max_seqlen_q, max_seqlen_k)

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     const_cast<void*>(q),
                     const_cast<void*>(k),
                     const_cast<void*>(v),
                     const_cast<void*>(out),
                     const_cast<void*>(dout),
                     dq,
                     dk,
                     dv,
                     const_cast<int32_t*>(cu_seqlens_q),
                     const_cast<int32_t*>(cu_seqlens_k),
                     loop ? dq_accum : nullptr,
                     nullptr,
                     nullptr,
                     const_cast<void*>(softmax_lse),
                     const_cast<void*>(softmax_d),
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     is_causal,
                     is_bf16,
                     num_splits,
                     const_cast<void *>(attn_mask),
                     mask_head_mod_size,
                     mask_seq_q_mod_size);

    auto launch = &run_mha_bwd;

    if (is_dropout) {
        params.philox_args = at::PhiloxCudaState(seed, offset);
        // seems a wild pointer at fa2: https://github.com/PaddlePaddle/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp#L908-L909
        params.rng_state = static_cast<uint64_t*>(rng_state);
        uint64_t rng_state_data[2] = {seed, offset};
        cudaMemcpyAsync(params.rng_state, rng_state_data, 2*sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    }

    launch(params, stream, /*configure=*/false);
    
    return true;
    
    FLASHATTNLIB_END_FUNC

}

bool flash_attn_fwd_with_bias_and_mask(const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       void *out,                  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const int32_t *cu_seqlens_q,   // int32, batch_size+1, starting offset of each sequence
                                       const int32_t *cu_seqlens_k,   // int32, batch_size+1, starting offset of each sequence
                                       const int total_q,
                                       const int total_k,
                                       const int batch_size,
                                       const int num_heads,
                                       const int head_size,
                                       const int max_seqlen_q_,
                                       const int max_seqlen_k_,
                                       const float p_dropout,
                                       const float softmax_scale,
                                       const bool zero_tensors,
                                       const bool is_bf16,
                                       const int  num_splits,        // SMs per attention matrix, can be 1
                                       void *softmax_lse_ptr,       // softmax log_sum_exp
                                       void *workspace_ptr,
                                       uint64_t *workspace_size,
                                       cudaStream_t stream,
                                       uint64_t seed,
                                       uint64_t offset,
                                       const void *attn_mask = nullptr,
                                       const void *attn_bias = nullptr,
                                       const int64_t* mask_dims = nullptr,
                                       const int64_t* bias_dims = nullptr) {
    return flash_attn_fwd_with_bias_and_mask_(q,
                                              k,
                                              v,
                                              out,
                                              cu_seqlens_q,
                                              cu_seqlens_k,
                                              total_q,
                                              total_k,
                                              batch_size,
                                              num_heads,
                                              head_size,
                                              max_seqlen_q_,
                                              max_seqlen_k_,
                                              p_dropout,
                                              softmax_scale,
                                              zero_tensors,
                                              is_bf16,
                                              num_splits,
                                              softmax_lse_ptr,
                                              workspace_ptr,
                                              workspace_size,
                                              stream,
                                              seed,
                                              offset,
                                              attn_mask,
                                              attn_bias,
                                              mask_dims,
                                              bias_dims);
}

bool flash_attn_bwd_with_bias_and_mask(const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       const void *k,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const void *v,              // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       void *dq,                   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       void *dk,                   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       void *dv,                   // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const void *out,            // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const void *dout,           // total_q x num_heads, x head_size
                                       const int32_t *cu_seqlens_q,   // int32, batch_size+1
                                       const int32_t *cu_seqlens_k,   // int32, batch_size+1
                                       const int total_q,
                                       const int total_k,
                                       const int batch_size,
                                       const int num_heads,
                                       const int head_size,
                                       const int max_seqlen_q_,
                                       const int max_seqlen_k_,
                                       const float p_dropout,
                                       const float softmax_scale,
                                       const bool zero_tensors,
                                       const bool is_bf16,
                                       const int num_splits,
                                       const void *softmax_lse_ptr,
                                       void *dsoftmax_ptr,
                                       void *dbias_ptr,
                                       void *workspace_ptr,
                                       uint64_t *workspace_size,
                                       cudaStream_t stream,
                                       uint64_t seed,
                                       uint64_t offset,
                                       const void* attn_mask = nullptr,
                                       const void* attn_bias = nullptr,
                                       const int64_t* mask_dims = nullptr,
                                       const int64_t* bias_dims = nullptr) {
    return flash_attn_bwd_with_bias_and_mask_(q,
                                              k,
                                              v,
                                              dq,
                                              dk,
                                              dv,
                                              out,
                                              dout,
                                              cu_seqlens_q,
                                              cu_seqlens_k,
                                              total_q,
                                              total_k,
                                              batch_size,
                                              num_heads,
                                              head_size,
                                              max_seqlen_q_,
                                              max_seqlen_k_,
                                              p_dropout,
                                              softmax_scale,
                                              zero_tensors,
                                              is_bf16,
                                              num_splits,
                                              softmax_lse_ptr,
                                              dsoftmax_ptr,
                                              dbias_ptr,
                                              workspace_ptr,
                                              workspace_size,
                                              stream,
                                              seed,
                                              offset,
                                              attn_mask,
                                              attn_bias,
                                              mask_dims,
                                              bias_dims);
}
#ifdef __cplusplus
}
#endif

