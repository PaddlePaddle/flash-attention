/******************************************************************************
 * Copyright (c) 2022, Tri Dao.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "flash_attn_with_bias_mask.h"
#include "fmha.h"
#include "utils.h"
#include "cuda_utils.h"
#include <cmath>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"
#include "dlfcn.h"
#include "math.h"
#include <memory>
#include <mutex>
#include <stdexcept>

#include <cstring>
#include <exception>
#include <string>

#define FLASH_ATTN_ASSERT_CHECK(__cond)                                         \
    do {                                                                        \
        const bool __cond_var = (__cond);                                       \
        if (!__cond_var) {                                                      \
            ::std::string __err_msg = ::std::string("`") + #__cond +            \
            "` check failed at " + __FILE__ + ":" + ::std::to_string(__LINE__); \
          throw std::runtime_error(__err_msg);                                  \
        }                                                                       \
      } while (0)

#define FLASHATTNLIB_BEGIN_FUNC try {
#define FLASHATTNLIB_END_FUNC } catch (::std::exception &__e) { \
        flash_attn_set_error(__e.what());                       \
        return false;                                           \
    } catch (...) {                                             \
        flash_attn_set_error(nullptr);                          \
        return false;                                           \
    }

void set_params_fprop_with_bias_mask(FMHA_fprop_params &params,
                                     // sizes
                                     const size_t b,
                                     const size_t seqlen_q,
                                     const size_t seqlen_k,
                                     const size_t h,
                                     const size_t d,
                                     // device pointers
                                     void *q,
                                     void *k,
                                     void *v,
                                     void *out,
                                     int32_t *cu_seqlens_q_d,
                                     int32_t *cu_seqlens_k_d,
                                     void *o_tmp_d, // temprary output
                                     void *s_d,     // softmax output
                                     void *softmax_lse_d,
                                     float p_dropout,
                                     float softmax_scale,
                                     bool is_causal,
                                     bool is_bf16,
                                     int num_splits,
                                     void *attn_mask = nullptr,
                                     void *attn_bias = nullptr,
                                     int bias_mod_size = 0,
                                     int mask_head_mod_size = 0,
                                     int mask_seq_mod_size = 0) {
    Data_type data_type = is_bf16 ? DATA_TYPE_BF16 : DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = is_bf16;

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    params.q_row_stride_in_elts = h * d;
    params.k_row_stride_in_elts = h * d;
    params.v_row_stride_in_elts = h * d;
    params.q_head_stride_in_elts = d;
    params.k_head_stride_in_elts = d;
    params.v_head_stride_in_elts = d;
    params.o_ptr = out;
    params.o_row_stride_in_elts = h * d;
    params.o_head_stride_in_elts = d;
    params.o_tmp_ptr = o_tmp_d;
    params.o_tmp_row_stride_in_elts = h * d;
    params.o_tmp_head_stride_in_elts = d;

    params.cu_seqlens_q = cu_seqlens_q_d;
    params.cu_seqlens_k = cu_seqlens_k_d;

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * seqlen_k, data_type);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;

    // attn mask & bias
    params.attn_mask_ptr = attn_mask;
    params.attn_bias_ptr = attn_bias;
    params.bias_mod_size = bias_mod_size;
    params.mask_head_mod_size = mask_head_mod_size;
    params.mask_seq_mod_size = mask_seq_mod_size;

    // Set the different scale values.
    // const float scale_bmm1 = 1.f / sqrtf(d);
    const float scale_bmm1 = softmax_scale;

    params.scale_bmm1f = scale_bmm1;
    set_alpha(params.scale_bmm1, scale_bmm1, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_bmm1_rp_dropout = params.rp_dropout * params.scale_bmm1f;
    FLASH_ATTN_ASSERT_CHECK(p_dropout < 1.f);
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);

    params.is_causal = is_causal;
    params.num_splits = num_splits;
}

void set_params_dgrad_with_bias_mask(FMHA_dgrad_params &params,
                                     const size_t b, // sizes
                                     const size_t seqlen_q,
                                     const size_t seqlen_k,
                                     const size_t h,
                                     const size_t d,
                                     void *q, // device pointers
                                     void *k,
                                     void *v,
                                     void *out,
                                     void *dq,
                                     void *dk,
                                     void *dv,
                                     int32_t *cu_seqlens_q_d,
                                     int32_t *cu_seqlens_k_d,
                                     void *dq_tmp_d,
                                     void *do_packed_d,
                                     void *softmax_lse_d,
                                     void *dsoftmax_sum_d,
                                     float p_dropout,
                                     float softmax_scale,
                                     bool is_causal,
                                     bool is_bf16,
                                     int num_splits,
                                     void *attn_mask = nullptr,
                                     void *attn_bias = nullptr,
                                     void *attn_ds = nullptr,
                                     int bias_mod_size = 0,
                                     int mask_head_mod_size = 0,
                                     int mask_seq_mod_size = 0) {
    set_params_fprop_with_bias_mask(params,
                                    b, 
                                    seqlen_q,
                                    seqlen_k,
                                    h,
                                    d,
                                    q,
                                    k,
                                    v,
                                    out,
                                    cu_seqlens_q_d,
                                    cu_seqlens_k_d,
                                    dq_tmp_d,  // Reusing the o_tmp_ptr variable to store dq_tmp
                                    nullptr,
                                    softmax_lse_d,
                                    p_dropout,
                                    softmax_scale,
                                    is_causal,
                                    is_bf16,
                                    num_splits,
                                    attn_mask,
                                    attn_bias,
                                    bias_mod_size,
                                    mask_head_mod_size,
                                    mask_seq_mod_size);

    // Set the pointers and strides.
    params.dq_ptr = dq;
    params.dk_ptr = dk;
    params.dv_ptr = dv;
    params.dq_row_stride_in_elts = h * d;
    params.dk_row_stride_in_elts = h * d;
    params.dv_row_stride_in_elts = h * d;
    params.dq_head_stride_in_elts = d;
    params.dk_head_stride_in_elts = d;
    params.dv_head_stride_in_elts = d;
    params.do_ptr = do_packed_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
    params.attn_ds_ptr = attn_ds;
}

bool run_fwd_with_bias_mask(Launch_params<FMHA_fprop_params> &launch_params,
                            const bool configure) {
    bool status = true;
    if (launch_params.params.d == 32) {
        status = run_fmha_fwd_with_mask_bias_hdim32(launch_params, configure);
    } else if (launch_params.params.d == 64) {
        status = run_fmha_fwd_with_mask_bias_hdim64(launch_params, configure);
    } else if (launch_params.params.d == 128) {
        status = run_fmha_fwd_with_mask_bias_hdim128(launch_params, configure);
    }
    return status;
}

bool run_bwd_with_bias_mask(FMHA_dgrad_params &params,
                            cudaStream_t stream) {
    bool status = true;
    if (params.d == 32) {
        status = run_fmha_bwd_with_mask_bias_hdim32(params, stream);
    } else if (params.d == 64) {
        status = run_fmha_bwd_with_mask_bias_hdim64(params, stream);
    } else if (params.d == 128) {
        status = run_fmha_bwd_with_mask_bias_hdim128(params, stream);
    }
    return status;
}

#ifdef __cplusplus
extern "C" {
#endif


// For just alphafold2
bool flash_attn_fwd_with_bias_and_mask_(
        const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
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
    // printf("forward seed %jd offset %jd\b", seed, offset);
    FLASHATTNLIB_BEGIN_FUNC

    auto dprops = GetDeviceProperties(-1);
    bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;

    FLASH_ATTN_ASSERT_CHECK(is_sm8x || is_sm75);
    FLASH_ATTN_ASSERT_CHECK(batch_size > 0);
    FLASH_ATTN_ASSERT_CHECK((head_size % 8 == 0) && (head_size <= 128));

    int blocksize_c = head_size > 64 ? 128 : 256;
    // Need to round max_seqlen_k to multiples of blocksize_c
    int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if( max_seqlen_k_ <= 128 ) {
        max_seqlen_k = 128;
    } else if( max_seqlen_k_ <= 256 ) {
        max_seqlen_k = 256;
    }
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
    bool loop = max_seqlen_k > blocksize_c;

    void* o_tmp_ptr = workspace_ptr;
    // nullptr out to calculate workspace size
    if (out == nullptr) {
        if (loop) {
            *workspace_size = uint64_t(total_q) * num_heads * head_size * sizeof(float);
        } else {
            *workspace_size = 0;
        }
        return true;
    }

    int bias_mod_size = attn_bias ? bias_dims[0] : 0;
    if (attn_bias) {
        FLASH_ATTN_ASSERT_CHECK(bias_dims[1] == num_heads);
    }

    int mask_head_mod_size = attn_mask ? mask_dims[1] : 0;
    int mask_seq_mod_size  = attn_mask ? mask_dims[2] : 0;
    if (attn_mask) {
        FLASH_ATTN_ASSERT_CHECK(mask_dims[1] == 1 || mask_dims[1] == num_heads);
        FLASH_ATTN_ASSERT_CHECK(mask_dims[2] == 1 || mask_dims[2] == max_seqlen_q_);
    }

    if (attn_bias || attn_mask) {
        // When attn_bias or attn_mask is not nullptr, softmax_scale must be 1.
        FLASH_ATTN_ASSERT_CHECK(softmax_scale == 1.0f);
    }

    bool return_softmax = false;
    bool is_dropout = p_dropout > 0.f;
    Launch_params<FMHA_fprop_params> launch_params(dprops, stream, is_dropout, return_softmax);

    if (zero_tensors) {
        SetZero(out,  2, {total_q, num_heads, head_size}, stream);
        SetConstValue<float>(softmax_lse_ptr, -std::numeric_limits<float>::infinity(), uint64_t(batch_size) * num_heads * max_seqlen_q, stream);   
    }

    set_params_fprop_with_bias_mask(launch_params.params,
                                    batch_size,
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    num_heads,
                                    head_size,
                                    const_cast<void*>(q),
                                    const_cast<void*>(k),
                                    const_cast<void*>(v),
                                    const_cast<void*>(out),
                                    const_cast<int32_t*>(cu_seqlens_q),
                                    const_cast<int32_t*>(cu_seqlens_k),
                                    loop ? o_tmp_ptr : nullptr,
                                    nullptr,
                                    softmax_lse_ptr,
                                    p_dropout,
                                    softmax_scale,
                                    false, // is_causal
                                    is_bf16,
                                    num_splits,
                                    const_cast<void*>(attn_mask),
                                    const_cast<void*>(attn_bias),
                                    bias_mod_size,
                                    mask_head_mod_size,
                                    mask_seq_mod_size);
    run_fwd_with_bias_mask(launch_params, /*configure=*/ true);

    if (is_dropout) {
        launch_params.params.philox_args = PhiloxCudaState(seed, offset);
    }
    bool status = run_fwd_with_bias_mask(launch_params, /*configure=*/false);
    return status;
    FLASHATTNLIB_END_FUNC 
}


bool flash_attn_bwd_with_bias_and_mask_(
        const void *q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
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
    // printf("backward seed %jd offset %jd\b", seed, offset);
    FLASHATTNLIB_BEGIN_FUNC
    auto dprops = GetDeviceProperties(-1);
    bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm80 = dprops->major == 8 && dprops->minor == 0;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    FLASH_ATTN_ASSERT_CHECK(is_sm8x || is_sm75);

    bool is_dropout = p_dropout > 0.0;

    FLASH_ATTN_ASSERT_CHECK(batch_size > 0);
    FLASH_ATTN_ASSERT_CHECK((head_size % 8 == 0) && (head_size <= 128));
    if (head_size > 64) {  // TODO: eventually we should support SM86 and SM70 with d=128 as well
        FLASH_ATTN_ASSERT_CHECK(is_sm80);
    }

    int blocksize_c = (head_size > 64 || (is_sm75 && head_size > 32)) ? 128 : 256;
    int max_seqlen_k = ((max_seqlen_k_ + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if( max_seqlen_k_ <= 128 ) {
        max_seqlen_k = 128;
    } else if( max_seqlen_k_ <= 256 ) {
        max_seqlen_k = 256;
    }
    int max_seqlen_q = ((max_seqlen_q_ + 16 - 1) / 16) * 16;
    bool loop = max_seqlen_k > blocksize_c;

    void *dq_tmp_ptr = workspace_ptr;
    // nullptr out to calculate workspace size
    if (out == nullptr) {
        // There are two cases no need to allocate workspace:
        // 1) num_splits == 1
        // 2) num_splits == 0 for auto calculation, result to num_splits == 1
        // we do allocation for case 2 for simplicity
        if (num_splits == 1) {
            *workspace_size = 0;
        } else {
            *workspace_size = uint64_t(total_q) * num_heads * head_size * sizeof(float);
        }
        return true;
    }

    int bias_mod_size = 0;
    if (attn_bias) {
        // check attn_bias shape
        bias_mod_size = bias_dims[0];
        SetZero(dbias_ptr, 2, {batch_size, num_heads, max_seqlen_q_, max_seqlen_k_}, stream);
        FLASH_ATTN_ASSERT_CHECK(bias_dims[1] == num_heads);
    }

    int mask_head_mod_size = 0;
    int mask_seq_mod_size = 0;
    if (attn_mask) {
        // last two dimension
        mask_head_mod_size = mask_dims[1];
        mask_seq_mod_size = mask_dims[2];
        FLASH_ATTN_ASSERT_CHECK(mask_dims[1] == 1 || mask_dims[1] == num_heads);
        FLASH_ATTN_ASSERT_CHECK(mask_dims[2] == 1 || mask_dims[2] == max_seqlen_q_);
    }

    if (attn_bias || attn_mask) {
        // When attn_bias or attn_mask is not nullptr, softmax_scale must be 1.
        FLASH_ATTN_ASSERT_CHECK(softmax_scale == 1.0f);
    }
    
    if(zero_tensors) {
        SetZero(dq, 2, {total_q, num_heads, head_size}, stream);
        SetZero(dk, 2, {total_q, num_heads, head_size}, stream);
        SetZero(dv, 2, {total_q, num_heads, head_size}, stream);
        SetZero(dsoftmax_ptr, 4, {batch_size, num_heads, max_seqlen_q}, stream);  
    }

    FMHA_dgrad_params params;
    set_params_dgrad_with_bias_mask(params,
                                    batch_size,
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    num_heads,
                                    head_size,
                                    const_cast<void*>(q),
                                    const_cast<void*>(k),
                                    const_cast<void*>(v),
                                    const_cast<void*>(out),
                                    dq,
                                    dk,
                                    dv,
                                    const_cast<int32_t*>(cu_seqlens_q),
                                    const_cast<int32_t*>(cu_seqlens_k),
                                    loop ? dq_tmp_ptr : nullptr,
                                    const_cast<void*>(dout),
                                    const_cast<void*>(softmax_lse_ptr),
                                    dsoftmax_ptr,
                                    p_dropout,
                                    softmax_scale,
                                    false, // is_causal
                                    is_bf16,
                                    num_splits,
                                    attn_mask ? const_cast<void*>(attn_mask) : nullptr,
                                    attn_bias ? const_cast<void*>(attn_bias) : nullptr,
                                    attn_bias ? dbias_ptr : nullptr,
                                    bias_mod_size,
                                    mask_head_mod_size,
                                    mask_seq_mod_size);

    if(is_dropout) {
        params.philox_args = PhiloxCudaState(seed, offset);
    }
    bool status = run_bwd_with_bias_mask(params, stream);
    return status;
    FLASHATTNLIB_END_FUNC 
}

#ifdef __cplusplus
}
#endif

