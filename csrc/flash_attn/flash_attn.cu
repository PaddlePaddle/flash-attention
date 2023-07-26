/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include "src/flash.h"
#include "src/static_switch.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "dlfcn.h"
#include "math.h"
//#include "src/utils.h"
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
                      bool is_causal,
		      bool is_bf16) {
    // TODO: assert dq.shape == do.shape, dk.shape == dv.shape
    // TODO: should we use d_rounded and seqlen_rounded?

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = is_bf16;

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    // All stride are in elements, not bytes.
    params.q_row_stride = h * d;
    params.k_row_stride = h * d;
    params.v_row_stride = h * d;
    params.q_head_stride = d;
    params.k_head_stride = d;
    params.v_head_stride = d;
    params.o_ptr = out;
    params.o_row_stride = h * d;
    params.o_head_stride = d;

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = seqlen_q * h * d;
        params.k_batch_stride = seqlen_k * h * d;
        params.v_batch_stride = seqlen_k * h * d;
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

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

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
                      bool is_causal,
		      bool is_bf16) {
    // TODO: assert dq.shape == do.shape, dk.shape == dv.shape
    // TODO: should we use d_rounded and seqlen_rounded?

    set_params_fprop(params,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     is_causal, is_bf16);

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
        params.dk_batch_stride = seqlen_k * h_k * d;
        params.dv_batch_stride = seqlen_k * h_k * d;
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
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

bool flash_attn_fwd(
        void * const q,
	void * const k,
	void * const v,
	void * const out,
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
	const bool is_causal,
	const bool return_softmax,
	const bool is_bf16,
	void * const softmax_ptr,
	void * const softmax_lse_ptr,
	cudaStream_t stream,
	uint64_t seed,
	uint64_t offset) {
    printf("\nwe are here\n");

    FLASHATTNLIB_BEGIN_FUNC

    auto dprops = GetDeviceProperties(-1);

    const bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    const bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    
    ASSERT_CHECK(is_sm8x || is_sm90);
    ASSERT_CHECK(batch_size > 0);
    // We should use pad to remove this assert
    // But maybe we don't have to pad this shit.
    ASSERT_CHECK(head_size % 8 == 0) ;
    ASSERT_CHECK(head_size <= 256);
    ASSERT_CHECK(num_heads % num_heads_k == 0);

    // where should I add check_shape?
    // CHECK_SHAPE(...)
    
    // where should I add head padded?
    
    // where should I deal with out?

    // maybe we don't have to pad it now...

    // where should I round head_size and seqlen_size?
    // auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m;};
    // const int head_size = round_multiple(head_size_og, 8);
    // const int h

    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        ASSERT_CHECK(p_dropout > 0.0f);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
		     seqlen_q, seqlen_k,
		     seqlen_q_rounded, seqlen_k_rounded,
		     num_heads, num_heads_k,
		     head_size, head_size_rounded,
		     q, k, v, out,
		     /*cu_seqlens_q_d=*/nullptr,
		     /*cu_seqlens_k_d=*/nullptr,
		     return_softmax ? softmax_ptr : nullptr,
		     softmax_lse_ptr,
		     p_dropout,
		     softmax_scale,
		     is_causal,
		     is_bf16);

    if (p_dropout > 0.0) {
        // number of times random will be generated per thread, to offset philox counter in thc random
        // state
        // We use a custom RNG that increases the offset by batch_size * nheads * 32.
	params.philox_args = PhiloxCudaState(seed, offset);
    }
	
    run_mha_fwd(params, stream);
    
    // we don't pad this shit yet.
    // TODO
    // if (head_size_og % 8 != 0)
    // ...
    
    return true;

    FLASHATTNLIB_END_FUNC
}

bool flash_attn_varlen_fwd(
        void * const q,
	void * const k,
	void * const v,
	void * const out,
	void * const cu_seqlens_q,
	void * const cu_seqlens_k,
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
	const bool zero_tensors,
	const bool is_causal,
	const bool return_softmax,
	const bool is_bf16,
	void * const softmax_ptr,
        void * const softmax_lse_ptr,
	cudaStream_t stream,
	uint64_t seed,
	uint64_t offset) {
    FLASHATTNLIB_BEGIN_FUNC
    auto dprops = GetDeviceProperties(-1);

    const bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    const bool is_sm90 = dprops->major == 9 && dprops->minor == 0;

    ASSERT_CHECK(is_sm8x || is_sm90);
    ASSERT_CHECK(batch_size > 0);
    
    // Add checking shit in paddle
    
    // This shit seems to be so easy...
    
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
		     max_seqlen_q, max_seqlen_k,
		     seqlen_q_rounded, seqlen_k_rounded,
		     num_heads, num_heads_k,
		     head_size, head_size_rounded,
		     q, k, v, out,
		     cu_seqlens_q,
		     cu_seqlens_k,
		     return_softmax ? softmax_ptr : nullptr,
		     softmax_lse_ptr,
		     p_dropout,
		     softmax_scale,
		     is_causal,
		     is_bf16);
    
    if (p_dropout > 0.0) {
        params.philox_args = PhiloxCudaState(seed, offset);
    }

    run_mha_fwd(params, stream);

    // we don't pad this shit yet.
    // TODO
    // if (head_size_og % 8 != 0)
    // ...
    
    return true;
    
    FLASHATTNLIB_END_FUNC
}

#ifdef __cplusplus
}
#endif

