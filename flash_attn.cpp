/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#include "flash_runner.hpp"
#include "random_utils.h"

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <cstring>
#include <exception>
#include <string>

template<typename T>
void printTensor(const std::string& str, const T* devTensor, size_t b, size_t s, size_t n, size_t h) {
  T* hostTensor;
  hostTensor = (T*)malloc(b * s * n * h * sizeof(T));
  hipMemcpy(hostTensor, devTensor, b * s * n * h * sizeof(T), hipMemcpyDeviceToHost);
  std::cout << str << ": ";
  for(int i = 0; i < h; i++) {
    if (i % 32 == 0) {
      std::cout << std::endl;
    }
    std::cout << static_cast<float>(hostTensor[i]) << ", ";
  }
  std::cout << " ...... ";
  for(int i = (b * s * n - 1) * h; i < b * s * n * h; i++) {
    if (i % 32 == 0) {
      std::cout << std::endl;
    }
    std::cout << static_cast<float>(hostTensor[i]) << ", ";
  }
  std::cout << str << " finish" << std::endl;
  free(hostTensor);
}

template<>
void printTensor<float>(const std::string& str, const float* devTensor, size_t b, size_t s, size_t n, size_t h) {
  float* hostTensor;
  hostTensor = (float*)malloc(b * s * n * h * sizeof(float));
  hipMemcpy(hostTensor, devTensor, b * s * n * h * sizeof(float), hipMemcpyDeviceToHost);
  std::cout << str << ": ";
  for(int i = 0; i < s / 8; i++) {
    if (i % 32 == 0) {
      std::cout << std::endl;
    }
    std::cout << static_cast<float>(hostTensor[i]) << ", ";
  }
  std::cout << " ...... " << std::endl;
  std::cout << str << " finish" << std::endl;
  free(hostTensor);
}

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
      ASSERT_CHECK(batch_size > 0);                                      \
      ASSERT_CHECK(head_size % 8 == 0);                                  \
      ASSERT_CHECK(head_size <= 128);                                    \
      ASSERT_CHECK(num_heads % num_heads_k == 0);                        \
      if (attn_mask) {                                                   \
          ASSERT_CHECK(mask_dims[0] == batch_size);                      \
          ASSERT_CHECK(mask_dims[1] == 1 || mask_dims[1] == num_heads);  \
          ASSERT_CHECK(mask_dims[2] == 1 || mask_dims[2] == __seqlen_q); \
          ASSERT_CHECK(mask_dims[3] == __seqlen_k);                      \
      }

#define CHECK_BWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                       \
      CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                                         

void run_mha_fwd(FlashFwdBatchedParams &params, hipStream_t stream) {
    FlashRunner flash_runner;
    flash_runner.Run(params, stream);
}

void run_mha_fwd_infer(FlashInferBatchedParams &params, hipStream_t stream) {
    FlashRunner flash_runner;
    flash_runner.Infer(params, stream);
}

void run_mha_varlen_fwd(FlashFwdGroupedParams &params, hipStream_t stream) {
    FlashRunner flash_runner;
    flash_runner.Run(params, stream);
}

void run_mha_bwd(FlashBwdBatchedParams &params, hipStream_t stream) {
  FlashRunner flash_runner;
  flash_runner.Run(params, stream);
}

void run_mha_varlen_bwd(FlashBwdGroupedParams &params, hipStream_t stream) {
  FlashRunner flash_runner;
  flash_runner.Run(params, stream);
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
                    hipStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims,
                    const bool is_infer) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    CHECK_FWD_EXECTUABLE(seqlen_q, seqlen_k)

    if (is_infer) {
        ASSERT_CHECK(head_size == 128);
        FlashInferBatchedParams params(batch_size, seqlen_q, seqlen_k, num_heads,
                                      num_heads_k, head_size, const_cast<void *>(q), const_cast<void *>(k),
                                      const_cast<void *>(v), out, softmax_scale, is_causal, is_bf16, softmax_lse_ptr);

        run_mha_fwd_infer(params, stream);
    } else {
        FlashFwdBatchedParams params(batch_size, seqlen_q, seqlen_k, num_heads,
                                    num_heads_k, head_size, const_cast<void *>(q), const_cast<void *>(k),
                                    const_cast<void *>(v), out, softmax_ptr, softmax_lse_ptr, 
                                    p_dropout, softmax_scale, is_causal, return_softmax, is_bf16);
    
        if (is_dropout) {
            auto philox_args = at::PhiloxCudaState(seed, offset);
            params.seeds = at::cuda::philox::unpack(philox_args);
            auto rng_state_ptr = reinterpret_cast<uint64_t *>(rng_state);
            std::tie(rng_state_ptr[0], rng_state_ptr[1]) = params.seeds;
        }

        run_mha_fwd(params, stream);
    }
    
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
                           hipStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * const attn_mask,
                           const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    CHECK_FWD_EXECTUABLE(max_seqlen_q, max_seqlen_k)
    ASSERT_CHECK((return_softmax == false) || (is_bf16 == false));
    
    FlashFwdGroupedParams params(batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k,
                                head_size, const_cast<void *>(q), const_cast<void *>(k), const_cast<void *>(v), out, cu_seqlens_q,
                                cu_seqlens_k, softmax_ptr, softmax_lse_ptr, p_dropout, softmax_scale,
                                is_causal, return_softmax, is_bf16);

    if (is_dropout) {
        auto philox_args = at::PhiloxCudaState(seed, offset);
        params.seeds = at::cuda::philox::unpack(philox_args);
        auto rng_state_ptr = reinterpret_cast<uint64_t *>(rng_state);
        std::tie(rng_state_ptr[0], rng_state_ptr[1]) = params.seeds;
    }

    run_mha_varlen_fwd(params, stream);

    return true;
    
    FLASHATTNLIB_END_FUNC
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
                    hipStream_t stream,
                    uint64_t seed,
                    uint64_t offset,
                    const void * const attn_mask,
                    const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    CHECK_BWD_EXECTUABLE(seqlen_q, seqlen_k)

    hipMemset(dq, 0, sizeof(half) * batch_size * seqlen_q * num_heads * head_size);
    // hipMemset(dk, 0, sizeof(half) * batch_size * seqlen_k * num_heads * head_size);
    // hipMemset(dv, 0, sizeof(half) * batch_size * seqlen_k * num_heads * head_size);

    FlashBwdBatchedParams params(
      batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k, head_size,
      const_cast<void *>(q), // q is padded
      const_cast<void *>(k), // k is padded
      const_cast<void *>(v), // v is padded
      const_cast<void *>(out), // out is padded
      const_cast<void *>(dout),
      dq, // dq is padded
      dk, // dk is padded
      dv, // dv is padded
      const_cast<void *>(softmax_d), const_cast<void *>(softmax_lse), p_dropout, softmax_scale, is_causal, is_bf16);

    if (is_dropout) {
      auto philox_args = at::PhiloxCudaState(seed, offset);
      params.seeds = at::cuda::philox::unpack(philox_args);
    }

    // printTensor<half>("******* Q: ", static_cast<half*>(params.q_ptr), batch_size, seqlen_q, num_heads, head_size);
    // printTensor<half>("******* K: ", static_cast<half*>(params.k_ptr), batch_size, seqlen_k, num_heads_k, head_size);
    // printTensor<half>("******* V: ", static_cast<half*>(params.v_ptr), batch_size, seqlen_k, num_heads_k, head_size);
    // printTensor<half>("******* out: ", static_cast<half*>(params.out_ptr), batch_size, seqlen_q, num_heads, head_size);
    // printTensor<half>("******* dout: ", static_cast<half*>(params.dout_ptr), batch_size, seqlen_q, num_heads, head_size);
    // printTensor<float>("******* softmax_d: ", static_cast<float*>(params.dsoftmax_ptr), batch_size, seqlen_q, num_heads, 1);
    // printTensor<half>("******* dQ: ", static_cast<half*>(params.dq_ptr), batch_size, seqlen_q, num_heads, head_size);
    // printTensor<half>("******* dK: ", static_cast<half*>(params.dk_ptr), batch_size, seqlen_k, num_heads, head_size);
    // printTensor<half>("******* dV: ", static_cast<half*>(params.dv_ptr), batch_size, seqlen_k, num_heads, head_size);
    // printTensor<float>("******* softmax_lse: ", static_cast<float*>(params.softmax_lse_ptr), batch_size, seqlen_q, num_heads, 1);
    run_mha_bwd(params, stream);
    // printTensor<half>("******* dQ after fa: ", static_cast<half*>(params.dq_ptr), batch_size, seqlen_q, num_heads, head_size);
    // printTensor<half>("******* dK after fa: ", static_cast<half*>(params.dk_ptr), batch_size, seqlen_k, num_heads, head_size);
    // printTensor<half>("******* dV after fa: ", static_cast<half*>(params.dv_ptr), batch_size, seqlen_k, num_heads, head_size);
    // printTensor<float>("******* softmax_d after fa: ", static_cast<float*>(params.dsoftmax_ptr), batch_size, seqlen_q, num_heads, 1);
    
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
                           hipStream_t stream,
                           uint64_t seed,
                           uint64_t offset,
                           const void * const attn_mask,
                           const int64_t * const mask_dims) {
    FLASHATTNLIB_BEGIN_FUNC
    const bool is_dropout = p_dropout > 0.0;
    CHECK_BWD_EXECTUABLE(max_seqlen_q, max_seqlen_k)

    hipMemset(dq, 0, sizeof(half) * batch_size * max_seqlen_q * num_heads * head_size);
    hipMemset(dk, 0, sizeof(half) * batch_size * max_seqlen_k * num_heads * head_size);
    hipMemset(dv, 0, sizeof(half) * batch_size * max_seqlen_k * num_heads * head_size);

    FlashBwdGroupedParams params(
      batch_size, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k,
      head_size,
      const_cast<void *>(q), // q is padded
      const_cast<void *>(k), // k is padded
      const_cast<void *>(v), // v is padded
      const_cast<void *>(out), // out is padded
      const_cast<void *>(dout),
      dq, // dq is padded
      dk, // dk is padded
      dv, // dv is padded
      cu_seqlens_q, cu_seqlens_k, const_cast<void *>(softmax_d),
      const_cast<void *>(softmax_lse), p_dropout, softmax_scale, is_causal, is_bf16);

    if (is_dropout) {
      auto philox_args = at::PhiloxCudaState(seed, offset);
      params.seeds = at::cuda::philox::unpack(philox_args);
    }

    // printTensor<half>("******* out: ", static_cast<half*>(const_cast<void*>(params.bwd_out_ptrs[0])), batch_size, max_seqlen_q, num_heads, head_size);
    // printTensor<half>("******* dout: ", static_cast<half*>(const_cast<void*>(params.dout_ptrs[0])), batch_size, max_seqlen_q, num_heads, head_size);
    // printTensor<float>("******* softmax_d: ", static_cast<float*>(params.dsoftmax_ptrs[0]), batch_size, max_seqlen_q, num_heads, 1);
    // printTensor<half>("******* dQ: ", static_cast<half*>(params.dq_ptrs[0]), batch_size, max_seqlen_q, num_heads, head_size);
    // printTensor<half>("******* dK: ", static_cast<half*>(params.dk_ptrs[0]), batch_size, max_seqlen_k, num_heads, head_size);
    // printTensor<half>("******* dV: ", static_cast<half*>(params.dv_ptrs[0]), batch_size, max_seqlen_k, num_heads, head_size);
    // printTensor<float>("******* softmax_lse: ", static_cast<float*>(const_cast<void*>(params.bwd_softmax_lse_ptrs[0])), batch_size, max_seqlen_q, num_heads, 1);
    run_mha_varlen_bwd(params, stream);
    // printTensor<half>("******* dQ after fa: ", static_cast<half*>(params.dq_ptrs[0]), batch_size, max_seqlen_q, num_heads, head_size);
    // printTensor<half>("******* dK after fa: ", static_cast<half*>(params.dk_ptrs[0]), batch_size, max_seqlen_k, num_heads, head_size);
    // printTensor<half>("******* dV after fa: ", static_cast<half*>(params.dv_ptrs[0]), batch_size, max_seqlen_k, num_heads, head_size);
    // printTensor<float>("******* softmax_d after fa: ", static_cast<float*>(params.dsoftmax_ptrs[0]), batch_size, max_seqlen_q, num_heads, 1);
    
    return true;
    
    FLASHATTNLIB_END_FUNC

}

#ifdef __cplusplus
}
#endif

