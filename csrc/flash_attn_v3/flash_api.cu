/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <math.h>
#include <cuda_utils.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <limits>

#include <memory>
#include <mutex>
#include <stdexcept>

#include <cstring>
#include <exception>
#include <string>

#include "flash.h"
#include "static_switch.h"

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

#define CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                       \
      auto dprops = at::cuda::getCurrentDeviceProperties();                                \
      const bool is_sm90 = dprops->major == 9 && dprops->minor == 0;                       \
      ASSERT_CHECK(is_sm90);                                                               \
      ASSERT_CHECK(batch_size > 0);                                                        \
      ASSERT_CHECK(head_size % 8 == 0);                                                    \
      ASSERT_CHECK(head_size <= 256);                                                      \
      ASSERT_CHECK(num_heads % num_heads_k == 0);                                          \
      ASSERT_CHECK(head_size == 64 || head_size == 128 || head_size == 256);               \

#define CHECK_BWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                       \
      CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                         \
      /* FlashAttention backward only supports head dimension at most 128 */               \
      ASSERT_CHECK(head_size <= 128);                                                      \

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t b_k,
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
                      void * seqused_q,
                      void * seqused_k,
                      void * const p_d,
                      void * const softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      float softmax_unscale,
                      int window_size_left,
                      int window_size_right,
                      bool is_causal,
                      bool is_bf16,
                      const int q_row_stride,
                      const int k_row_stride,
                      const int v_row_stride,
                      const int q_head_stride,
                      const int k_head_stride,
                      const int v_head_stride,
                      const int o_row_stride,
                      const int o_head_stride,
                      const int q_batch_stride,
                      const int k_batch_stride,
                      const int v_batch_stride,
                      const int o_batch_stride,
                      bool varlen_padded_input = false,
                      void * attn_mask = nullptr,
                      void * flashmask_downstart_ptr = nullptr,
                      void * flashmask_upend_ptr = nullptr,
                      void * flashmask_downend_ptr = nullptr,
                      void * flashmask_upstart_ptr = nullptr,
                      void * flashmask_maxmin_ptr = nullptr,
                      int mask_head_mod_size = 0,
                      int mask_seq_q_mod_size = 0,
                      bool is_e4m3 = false,
                      int * tile_count_semaphore = nullptr,
                      float * descale_q_ptr = nullptr,
                      float * descale_k_ptr = nullptr,
                      float * descale_v_ptr = nullptr,
                      bool use_gqa_packing = false,
                      bool seqlenq_ngroups_swapped = false,
                      bool unpadded_lse=false) {
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = is_bf16;
    params.is_e4m3 = is_e4m3;
    params.is_kv_cache = false;
    params.tile_count_semaphore = tile_count_semaphore;

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    // All stride are in elements, not bytes.
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_ptr = out;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;
    params.varlen_padded_input = varlen_padded_input;

    if (cu_seqlens_q_d == nullptr ||  params.varlen_padded_input) {
        params.q_batch_stride = q_batch_stride;
        params.k_batch_stride = k_batch_stride;
        params.v_batch_stride = v_batch_stride;
        params.o_batch_stride = o_batch_stride;
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.b_k = b_k;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Guard against mistaken setting of gqa flag
    if (h == h_k) { use_gqa_packing = false; }

    // flashmask row index
    params.attn_mask_ptr = attn_mask;
    params.mask_head_mod_size = mask_head_mod_size;
    params.mask_seq_q_mod_size = mask_seq_q_mod_size;
    params.flashmask_downstart_ptr = flashmask_downstart_ptr;
    params.flashmask_upend_ptr = flashmask_upend_ptr;
    params.flashmask_downend_ptr = flashmask_downend_ptr;
    params.flashmask_upstart_ptr = flashmask_upstart_ptr;
    params.flashmask_maxmin_ptr = static_cast<int*>(flashmask_maxmin_ptr);
    params.enable_mask_bypass = true;
    if(flashmask_downstart_ptr != nullptr || flashmask_upend_ptr != nullptr) {
        params.h_flashmask = mask_head_mod_size;
        params.h_h_flashmask_ratio = h / mask_head_mod_size;
        if (params.enable_mask_bypass){
            ASSERT_CHECK(params.flashmask_maxmin_ptr != nullptr);
        }
    }

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);
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

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    window_size_left = std::min(int(seqlen_k), window_size_left);
    window_size_right = std::min(int(seqlen_k), window_size_right);
    if (window_size_left < 0) { window_size_left = seqlen_k; }
    if (window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    // params.is_causal = is_causal;
    params.is_causal = window_size_left == int(seqlen_k) && window_size_right == 0;
    if ((window_size_left < int(seqlen_k) || window_size_right < int(seqlen_k)) && !params.is_causal) {
        params.is_local = true;
    }

    params.descale_q_ptr = descale_q_ptr;
    params.descale_k_ptr = descale_k_ptr;
    params.descale_v_ptr = descale_v_ptr;
    params.use_gqa_packing = use_gqa_packing;

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(Flash_bwd_params &params,
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
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
                      void * seqused_q,
                      void * seqused_k,
                      void * const dq_accum_d,
                      void * const dk_accum_d,
                      void * const dv_accum_d,
                      void * const softmax_lse_d,
                      void * const dsoftmax_sum_d,
                      void * const softmax_lse_log2_d,
                      float p_dropout,
                      float softmax_scale,
                      float softmax_unscale,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic,
                      bool is_causal,
                      bool is_bf16,
                      const int q_batch_stride,
                      const int k_batch_stride,
                      const int v_batch_stride,
                      const int q_row_stride,
                      const int k_row_stride,
                      const int v_row_stride,
                      const int q_head_stride,
                      const int k_head_stride,
                      const int v_head_stride,
                      const int o_batch_stride,
                      const int o_row_stride,
                      const int o_head_stride,
                      const int dq_batch_stride,
                      const int dk_batch_stride,
                      const int dv_batch_stride,
                      const int dq_row_stride,
                      const int dk_row_stride,
                      const int dv_row_stride,
                      const int dq_head_stride,
                      const int dk_head_stride,
                      const int dv_head_stride,
                      const int do_batch_stride,
                      const int do_row_stride,
                      const int do_head_stride,
                      const bool varlen_padded_input = false,
                      const int num_splits = 0,
                      void * attn_mask = nullptr,
                      void * flashmask_downstart_ptr = nullptr,
                      void * flashmask_upend_ptr = nullptr,
                      void * flashmask_downend_ptr = nullptr,
                      void * flashmask_upstart_ptr = nullptr,
                      void * flashmask_maxmin_ptr = nullptr,
                      int mask_head_mod_size = 0,
                      int mask_seq_q_mod_size = 0,
                      int * dq_semaphore = nullptr) {

    set_params_fprop(params,
                     b, b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     seqused_q,
                     seqused_k,
                     /*p_d=*/nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     window_size_left,
                     window_size_right,
                     is_causal,
                     is_bf16,
                     q_row_stride,
                     k_row_stride,
                     v_row_stride,
                     q_head_stride,
                     k_head_stride,
                     v_head_stride,
                     o_row_stride,
                     o_head_stride,
                     q_batch_stride,
                     k_batch_stride,
                     v_batch_stride,
                     o_batch_stride,
                     varlen_padded_input,
                     attn_mask,
                     flashmask_downstart_ptr,
                     flashmask_upend_ptr,
                     flashmask_downend_ptr,
                     flashmask_upstart_ptr,
                     flashmask_maxmin_ptr,
                     mask_head_mod_size,
                     mask_seq_q_mod_size);

    // Set the pointers and strides.
    params.do_ptr = dout;
    params.do_row_stride = do_row_stride;
    params.do_head_stride = do_head_stride;
    params.dq_ptr = dq;
    params.dk_ptr = dk;
    params.dv_ptr = dv;
    params.dq_row_stride = dq_row_stride;
    params.dk_row_stride = dk_row_stride;
    params.dv_row_stride = dv_row_stride;
    params.dq_head_stride = dq_head_stride;
    params.dk_head_stride = dk_head_stride;
    params.dv_head_stride = dv_head_stride;

    if (cu_seqlens_q_d == nullptr || varlen_padded_input) {
        params.do_batch_stride = do_batch_stride;
        params.dq_batch_stride = dq_batch_stride;
        params.dk_batch_stride = dk_batch_stride;
        params.dv_batch_stride = dv_batch_stride;
    }
    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
    params.num_splits = num_splits;
    params.deterministic = deterministic;
    params.dq_semaphore = dq_semaphore;
    params.softmax_lse_log2_ptr = softmax_lse_log2_d;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    int dtype = 1;
    if (params.is_bf16) { dtype = 2; }
    else if (params.is_e4m3) { dtype = 3; }
    PREC_SWITCH(dtype, Element, [&] {
      HEADDIM_SWITCH(params.d, kHeadSize, [&] {
        if(!params.use_gqa_packing) {
          run_mha_fwd_<Element, kHeadSize>(params, stream);
        } else {
          QUERYHEAD_SWITCH(params.h_h_k_ratio, kBlockH, [&] {
            run_mha_fwd_gqa_<Element, kHeadSize, kBlockH>(params, stream);
          });
        }
      });
    });
}


#ifdef __cplusplus
extern "C" {
#endif

bool flash_attn_v3_fwd(const void * const q,         // batch_size x seqlen_q x num_heads x head_size
                    const void * const k,         // batch_size x seqlen_k x num_heads_k x head_size
                    const void * const v,         // batch_size x seqlen_k x num_heads_k x head_size
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
                    const int64_t * const mask_dims,
                    const void * const flashmask_downstart_ptr,
                    const void * const flashmask_downend_ptr,
                    const void * const flashmask_upend_ptr,
                    const void * const flashmask_upstart_ptr,
                    const void * const flashmask_maxmin_ptr,
                    const int64_t * const flashmask_dims,
                    const int q_batch_stride,
                    const int k_batch_stride,
                    const int v_batch_stride,
                    const int q_row_stride,
                    const int k_row_stride,
                    const int v_row_stride,
                    const int q_head_stride,
                    const int k_head_stride,
                    const int v_head_stride,
                    const int o_batch_stride,
                    const int o_row_stride,
                    const int o_head_stride,
                    const bool is_e4m3,
                    void * tile_count_semaphore,
                    const float * const descale_q_ptr,
                    const float * const descale_k_ptr,
                    const float * const descale_v_ptr,
                    const bool use_gqa_packing) {

    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : flashmask_dims ? flashmask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    CHECK_FWD_EXECTUABLE(seqlen_q, seqlen_k)

    int window_size_left = -1;
    int window_size_right = -1;
    if (is_causal) { window_size_right = 0; }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, batch_size,
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
                     /*seqused_q=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? softmax_ptr : nullptr,
                     softmax_lse_ptr,
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     window_size_left,
                     window_size_right,
                     is_causal,
                     is_bf16,
                     q_row_stride,
                     k_row_stride,
                     v_row_stride,
                     q_head_stride,
                     k_head_stride,
                     v_head_stride,
                     o_row_stride,
                     o_head_stride,
                     q_batch_stride,
                     k_batch_stride,
                     v_batch_stride,
                     o_batch_stride,
                     /*varlen_padded_input=*/false,
                     const_cast<void *>(attn_mask),
                     const_cast<void *>(flashmask_downstart_ptr),
                     const_cast<void *>(flashmask_upend_ptr),
                     const_cast<void *>(flashmask_downend_ptr),
                     const_cast<void *>(flashmask_upstart_ptr),
                     const_cast<void *>(flashmask_maxmin_ptr),
                     mask_head_mod_size,
                     mask_seq_q_mod_size,
                     is_e4m3,
                     reinterpret_cast<int *>(tile_count_semaphore),
                     const_cast<float *>(descale_q_ptr),
                     const_cast<float *>(descale_k_ptr),
                     const_cast<float *>(descale_v_ptr),
                     use_gqa_packing);

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

//void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {
//    FP16_SWITCH(!params.is_bf16, [&] {
//        HEADDIM_SWITCH(params.d, [&] {
//            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
//                BOOL_SWITCH(params.attn_mask_ptr != nullptr, Is_attn_mask, [&] {
//                    BOOL_SWITCH(params.flashmask_downstart_ptr != nullptr, Is_flashmask, [&] {
//                        run_mha_bwd_<elem_type, kHeadDim, Is_causal, Is_attn_mask, Is_flashmask>(params, stream, configure);
//                    });
//                });
//            });
//        });
//    });
//}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  // FP16_SWITCH(!params.is_bf16, [&] {
  //     HEADDIM_SWITCH(params.d, [&] {
  //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
  //     });
  // });
  if (!params.is_bf16) {
    if (params.d <= 64) {
      run_mha_bwd_<cutlass::half_t, 64>(params, stream);
    } else if (params.d <= 96) {
      run_mha_bwd_<cutlass::half_t, 96>(params, stream);
    } else {
      run_mha_bwd_<cutlass::half_t, 128>(params, stream);
    }
  } else {
    if (params.d <= 64) {
      run_mha_bwd_<cutlass::bfloat16_t, 64>(params, stream);
    } else if (params.d <= 96) {
      run_mha_bwd_<cutlass::bfloat16_t, 96>(params, stream);
    } else {
      run_mha_bwd_<cutlass::bfloat16_t, 128>(params, stream);
    }
  }
}

bool flash_attn_v3_bwd(const void * const dout,  // batch_size x seqlen_q x num_heads, x head_size_og
                    const void * const q,   // batch_size x seqlen_q x num_heads x head_size
                    const void * const k,   // batch_size x seqlen_k x num_heads_k x head_size
                    const void * const v,   // batch_size x seqlen_k x num_heads_k x head_size
                    const void * const out,   // batch_size x seqlen_q x num_heads x head_size
                    const void * const softmax_d,
                    const void * const softmax_lse,     // b x h x seqlen_q
                    const void * const softmax_lse_log2,
                    void * const rng_state,
                    void * const dq,   // batch_size x seqlen_q x num_heads x head_size
                    void * const dk,   // batch_size x seqlen_k x num_heads_k x head_size
                    void * const dv,   // batch_size x seqlen_k x num_heads_k x head_size
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
                    const float p_dropout,         // probability to drop
                    const float softmax_scale,
                    const float softmax_unscale,
                    const bool is_causal,
                    const bool is_bf16,
                    const int num_splits,
                    const bool deterministic,
                    cudaStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims,
                    const void * const flashmask_downstart_ptr,
                    const void * const flashmask_downend_ptr,
                    const void * const flashmask_upend_ptr,
                    const void * const flashmask_upstart_ptr,
                    const void * const flashmask_maxmin_ptr,
                    const int64_t * const flashmask_dims,
                    const int q_batch_stride,
                    const int k_batch_stride,
                    const int v_batch_stride,
                    const int q_row_stride,
                    const int k_row_stride,
                    const int v_row_stride,
                    const int q_head_stride,
                    const int k_head_stride,
                    const int v_head_stride,
                    const int o_batch_stride,
                    const int o_row_stride,
                    const int o_head_stride,
                    const int dq_batch_stride,
                    const int dk_batch_stride,
                    const int dv_batch_stride,
                    const int dq_row_stride,
                    const int dk_row_stride,
                    const int dv_row_stride,
                    const int dq_head_stride,
                    const int dk_head_stride,
                    const int dv_head_stride,
                    const int do_batch_stride,
                    const int do_row_stride,
                    const int do_head_stride,
                    void * dq_semaphore) {

    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    const int mask_head_mod_size = attn_mask ? mask_dims[1] : flashmask_dims ? flashmask_dims[1] : 0;
    const int mask_seq_q_mod_size = attn_mask ? mask_dims[2] : 0;

    CHECK_BWD_EXECTUABLE(seqlen_q, seqlen_k)

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    const bool loop = true;

    int window_size_left = -1;
    int window_size_right = -1;
    if (is_causal) { window_size_right = 0; }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     seqlen_q_rounded,
                     seqlen_k_rounded,
                     num_heads,
                     num_heads_k,
                     head_size,
                     head_size_rounded,
                     const_cast<void *>(q),
                     const_cast<void *>(k),
                     const_cast<void *>(v),
                     const_cast<void *>(out),
                     const_cast<void *>(dout),
                     dq,
                     dk,
                     dv,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q=*/nullptr,
                     /*seqused_k=*/nullptr,
                     dq_accum,
                     /*dk_accum_d=*/nullptr,
                     /*dv_accum_d=*/nullptr,
                     const_cast<void *>(softmax_lse),
                     const_cast<void *>(softmax_d),
                     const_cast<void *>(softmax_lse_log2),
                     p_dropout,
                     softmax_scale,
                     softmax_unscale,
                     window_size_left,
                     window_size_right,
                     deterministic,
                     is_causal,
                     is_bf16,
                     q_batch_stride,
                     k_batch_stride,
                     v_batch_stride,
                     q_row_stride,
                     k_row_stride,
                     v_row_stride,
                     q_head_stride,
                     k_head_stride,
                     v_head_stride,
                     o_batch_stride,
                     o_row_stride,
                     o_head_stride,
                     dq_batch_stride,
                     dk_batch_stride,
                     dv_batch_stride,
                     dq_row_stride,
                     dk_row_stride,
                     dv_row_stride,
                     dq_head_stride,
                     dk_head_stride,
                     dv_head_stride,
                     do_batch_stride,
                     do_row_stride,
                     do_head_stride,
                     /*varlen_padded_input=*/false,
                     num_splits,
                     const_cast<void *>(attn_mask),
                     const_cast<void *>(flashmask_downstart_ptr),
                     const_cast<void *>(flashmask_upend_ptr),
                     const_cast<void *>(flashmask_downend_ptr),
                     const_cast<void *>(flashmask_upstart_ptr),
                     const_cast<void *>(flashmask_maxmin_ptr),
                     mask_head_mod_size,
                     mask_seq_q_mod_size,
                     reinterpret_cast<int *>(dq_semaphore));

    auto launch = &run_mha_bwd;
    
    if (is_dropout) {
        params.philox_args = at::PhiloxCudaState(seed, offset);
        // seems a wild pointer at fa2: https://github.com/PaddlePaddle/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp#L690-L691
        params.rng_state = static_cast<uint64_t*>(rng_state);
        uint64_t rng_state_data[2] = {seed, offset};
        cudaMemcpyAsync(params.rng_state, rng_state_data, 2*sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    }

    launch(params, stream);
    
    return true;
    
    FLASHATTNLIB_END_FUNC

}

#ifdef __cplusplus
}
#endif

