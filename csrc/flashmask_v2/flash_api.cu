/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"

#define PADDLE_CHECK(__cond, message)                    \
      do {                                               \
        const bool __cond_var = (__cond);                \
        if (!__cond_var) {                               \
          ::std::string __err_msg = ::std::string("`") + \
                #__cond + "` check failed at " +         \
                __FILE__ + ":" +                         \
                ::std::to_string(__LINE__) +             \
                message;                                 \
          throw std::runtime_error(__err_msg);           \
        }                                                \
      } while (0)

#define CHECK_DEVICE(x) PADDLE_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) PADDLE_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) PADDLE_CHECK(x.is_contiguous(), #x " must be contiguous")

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    PADDLE_CHECK(params.num_splits >= 1, "num_splits should >= 1");
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        if (!params.is_e4m3) {
                            if (params.is_bf16) {
                                #ifndef FLASHMASK_V2_DISABLE_HDIM64
                                if (params.d <= 64) {
                                    if (params.dv > 64 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                    else {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHMASK_V2_DISABLE_HDIM96
                                if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHMASK_V2_DISABLE_HDIM128
                                if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHMASK_V2_DISABLE_HDIM192
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHMASK_V2_DISABLE_HDIM256
                                if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                            } else {
                                #ifndef FLASHMASK_V2_DISABLE_FP16
                                #ifndef FLASHMASK_V2_DISABLE_HDIM64
                                if (params.d <= 64) {
                                    if (params.dv > 64 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                    else {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHMASK_V2_DISABLE_HDIM96
                                if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHMASK_V2_DISABLE_HDIM128
                                if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHMASK_V2_DISABLE_HDIM192
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHMASK_V2_DISABLE_HDIM256
                                if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #else
                                PADDLE_CHECK(false, "This flash attention build does not support FP16.");
                                #endif
                            }
                        } else {
                            #ifndef FLASHMASK_V2_DISABLE_FP8
                            #ifndef FLASHMASK_V2_DISABLE_HDIM64
                            if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHMASK_V2_DISABLE_HDIM96
                            if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHMASK_V2_DISABLE_HDIM128
                            if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHMASK_V2_DISABLE_HDIM192
                            if (params.d <= 192) {
                                if (params.dv <= 128 && Arch == 90) {
                                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                } else {
                                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                }
                            }
                            #endif
                            #ifndef FLASHMASK_V2_DISABLE_HDIM256
                            if (params.d <= 256) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #else
                            #endif
                        }
                    });
                });
            });
        });
    });
}

void run_mha_fwd_combine(Flash_fwd_params &params, cudaStream_t stream, bool enable_pdl=false) {
    #ifndef FLASHMASK_V2_DISABLE_SPLIT
    // If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    // so that kBlockM is smaller and we have more parallelism.
    if (params.is_fp32) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<float, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<float, float, 128>(params, stream, enable_pdl);
        }
    } else if (params.is_bf16) {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>(params, stream, enable_pdl);
        }
    } else {
        if (params.dv <= 64) {
            run_mha_fwd_combine_<cutlass::half_t, float, 64>(params, stream, enable_pdl);
        } else {
            run_mha_fwd_combine_<cutlass::half_t, float, 128>(params, stream, enable_pdl);
        }
    }
    #else
    PADDLE_CHECK(false, "This flash attention build does not support combine kernels.");
    #endif
}

inline bool is_short_seqlen(Flash_fwd_params const& params) {
    return params.seqlen_k < 128 && params.seqlen_q < 128;
}

inline bool get_pagedkv_tma(Flash_fwd_params const& params) {
    if (params.arch < 90 || !params.page_table || params.leftpad_k || params.knew_ptr) { return false; }
    // This needs to match the kernel configs
    bool const short_seqlen = is_short_seqlen(params);
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, false /*paged_kv_non_TMA*/, params.softcap > 0.f, short_seqlen);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    int const kBlockN = std::get<1>(kBlockMN_kernel_args_sm90);
    // Heuristic: when seqlen_q <= kBlockM, we're not compute bound, and somehow using TMA is slower,
    // at least for MLA.
    return params.page_size % kBlockN == 0 && params.seqlen_q * (params.h / params.h_k) > kBlockM;
}

inline bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    #ifdef FLASHMASK_V2_DISABLE_PACKGQA
    return false;
    #else
    // params.page_table must already be set
    if (params.h == params.h_k) { return false; }
    // This needs to match the kernel configs
    bool const short_seqlen = is_short_seqlen(params);
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f, short_seqlen);
    int const kBlockM = std::get<0>(kBlockMN_kernel_args_sm90);
    return should_pack_gqa(params.cu_seqlens_q || params.seqused_q, params.seqlen_q, params.h / params.h_k, kBlockM);
    #endif
}

inline int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHMASK_V2_DISABLE_SPLIT
    return 1;
    #else
    // Always enable PackGQA for Split
    // params.page_table must already be set
    // This needs to match the kernel configs
    bool const varlen = params.cu_seqlens_q || params.cu_seqlens_k || params.seqused_q || params.seqused_k || params.leftpad_k;
    bool const short_seqlen = is_short_seqlen(params);
    auto kBlockMN_kernel_args_sm90 = tile_size_fwd_sm90(params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, false /*v_colmajor*/, params.page_table && !params.pagedkv_tma, params.softcap > 0.f, short_seqlen);
    // Strictly speaking we need to pass in (varlen && params.num_splits > 1) but num_splits
    // has not been set here. It's OK though because we might just underestimate kBlockN a bit
    auto kBlockMN_kernel_args_sm8x = tile_size_fwd_sm8x(params.arch == 86 || params.arch == 89, params.d_rounded, params.dv_rounded, params.is_causal, params.is_local, params.is_e4m3 ? 1 : 2 /*element_size*/, params.page_table, varlen, params.softcap > 0.f, params.knew_ptr);
    int const kBlockM = params.arch >= 90 ? std::get<0>(kBlockMN_kernel_args_sm90) : std::get<0>(kBlockMN_kernel_args_sm8x);
    int const kBlockN = params.arch >= 90 ? std::get<1>(kBlockMN_kernel_args_sm90) : std::get<1>(kBlockMN_kernel_args_sm8x);
    int seqlen_q_packgqa = params.seqlen_q * (params.h / params.h_k);
    // If is_local, we're not going to load all of seqlen_k
    int const seqlen_k_loaded = !params.is_local
        ? params.seqlen_k
        : std::max(0, std::min(params.seqlen_k, params.window_size_right + params.window_size_left + 1 + kBlockM));
    int const num_n_blocks = (seqlen_k_loaded + kBlockN - 1) / kBlockN;
    int const num_m_blocks = (seqlen_q_packgqa + kBlockM - 1) / kBlockM;
    int const size_one_kv_head = params.seqlen_k * (params.d + params.dv) * (params.is_e4m3 ? 1 : 2);
    // Always enable PackGQA for Split
    // If varlen, we use dynamic split, so this heuristic just needs to get an upper bound on num_splits.
    // We assume the case where there's 1 long sequence and the rest are short, i.e. pretending
    // that batch = 1.
    int total_mblocks = (params.num_splits_dynamic_ptr ? 1 : params.b) * params.h_k * num_m_blocks;
    return num_splits_heuristic(total_mblocks, params.num_sm, num_n_blocks, num_m_blocks, size_one_kv_head, params.is_causal || params.is_local, 128);
    #endif
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    #ifndef FLASHMASK_V2_DISABLE_BACKWARD
        // FP16_SWITCH(!params.is_bf16, [&] {
        //     HEADDIM_SWITCH(params.d, [&] {
        //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        //     });
        // });
    // printf("params.d = %d",params.d);
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (!params.is_bf16) {
                    #ifndef FLASHMASK_V2_DISABLE_FP16
                    #ifndef FLASHMASK_V2_DISABLE_HDIM64
                    if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::half_t, 64, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #ifndef FLASHMASK_V2_DISABLE_HDIM96
                    if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::half_t, 96, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #ifndef FLASHMASK_V2_DISABLE_HDIM128
                    if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::half_t, 128, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #ifndef FLASHMASK_V2_DISABLE_HDIM192
                    if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::half_t, 192, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #ifndef FLASHMASK_V2_DISABLE_HDIM256
                    if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::half_t, 256, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #else
                    PADDLE_CHECK(false, "This flash attention build does not support FP16.");
                    #endif
                } else {
                    #ifndef FLASHMASK_V2_DISABLE_HDIM64
                    if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #ifndef FLASHMASK_V2_DISABLE_HDIM96
                    if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #ifndef FLASHMASK_V2_DISABLE_HDIM128
                    if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #ifndef FLASHMASK_V2_DISABLE_HDIM192
                    if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    #ifndef FLASHMASK_V2_DISABLE_HDIM256
                    if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256, Has_softcap, Is_causal>(params, stream); }
                    #endif
                    PADDLE_CHECK(false, "This flash attention build does not support ");
                }
            });
        });
    });
    #endif
}

#ifdef __cplusplus
extern "C" {
#endif
Flash_fwd_params* flashmaskv2_create_fwd_params_handle() {
  Flash_fwd_params* params_handle = (Flash_fwd_params*)malloc(sizeof(Flash_fwd_params));
  if(params_handle) {
    *params_handle = Flash_fwd_params{};
  }
  return params_handle;
}

Flash_bwd_params* flashmaskv2_create_bwd_params_handle() {
  Flash_bwd_params* params_handle = (Flash_bwd_params*)malloc(sizeof(Flash_bwd_params));
  if(params_handle) {
    *params_handle = Flash_bwd_params{};
  }
  return params_handle;
}

void flashmaskv2_clear_fwd_params_handle(Flash_fwd_params* params_handle) {
  if(params_handle) {
    *params_handle = Flash_fwd_params{};
  }
}

void flashmaskv2_clear_bwd_params_handle(Flash_bwd_params* params_handle) {
  if(params_handle) {
    *params_handle = Flash_bwd_params{};
  }
}

Flash_fwd_params* flashmaskv2_cast_to_fwd_params_handle(Flash_bwd_params* params_handle) {
  return static_cast<Flash_fwd_params*>(params_handle);
}

void flashmaskv2_destroy_fwd_params_handle(Flash_fwd_params* params_handle) {
    PADDLE_CHECK(params_handle, "params_handle is nullptr");
    free(params_handle);
}

void flashmaskv2_destroy_bwd_params_handle(Flash_bwd_params* params_handle) {
    PADDLE_CHECK(params_handle, "params_handle is nullptr");
    free(params_handle);
}

void flashmaskv2_run_mha_fwd_combine(Flash_fwd_params* params_handle, cudaStream_t stream, bool enable_pdl=false) {
    run_mha_fwd_combine(*params_handle, stream, enable_pdl);
}

void flashmaskv2_run_mha_fwd(Flash_fwd_params* params_handle, cudaStream_t stream) {
    run_mha_fwd(*params_handle, stream);
}

void flashmaskv2_run_mha_bwd(Flash_bwd_params* params_handle, cudaStream_t stream) {
    // printf("point1\n");
    run_mha_bwd(*params_handle, stream);
}

bool flashmaskv2_get_pagedkv_tma(Flash_fwd_params* params_handle) {
    return get_pagedkv_tma(*params_handle);
}

bool flashmaskv2_get_pack_gqa(Flash_fwd_params* params_handle) {
    return get_pack_gqa(*params_handle);
}

int flashmaskv2_get_num_splits(Flash_fwd_params* params_handle) {
    return get_num_splits(*params_handle);
}

#define DEFINE_GETTER_SETTER(type, member) \
type flashmaskv2_fwd_params_get_##member(const Flash_fwd_params* params_handle) { return params_handle->member; } \
void flashmaskv2_fwd_params_set_##member(Flash_fwd_params* params_handle, type value) { params_handle->member = value; } \
type flashmaskv2_bwd_params_get_##member(const Flash_bwd_params* params_handle) { return params_handle->member; } \
void flashmaskv2_bwd_params_set_##member(Flash_bwd_params* params_handle, type value) { params_handle->member = value; }

// The QKV matrices.
DEFINE_GETTER_SETTER(void *, q_ptr)
DEFINE_GETTER_SETTER(void *, k_ptr)
DEFINE_GETTER_SETTER(void *, v_ptr)

// The stride between rows of the Q, K and V matrices.
DEFINE_GETTER_SETTER(int64_t, q_batch_stride)
DEFINE_GETTER_SETTER(int64_t, k_batch_stride)
DEFINE_GETTER_SETTER(int64_t, v_batch_stride)
DEFINE_GETTER_SETTER(int64_t, q_row_stride)
DEFINE_GETTER_SETTER(int64_t, k_row_stride)
DEFINE_GETTER_SETTER(int64_t, v_row_stride)
DEFINE_GETTER_SETTER(int64_t, q_head_stride)
DEFINE_GETTER_SETTER(int64_t, k_head_stride)
DEFINE_GETTER_SETTER(int64_t, v_head_stride)
DEFINE_GETTER_SETTER(int64_t, v_dim_stride)

// The number of heads.
DEFINE_GETTER_SETTER(int, h)
DEFINE_GETTER_SETTER(int, h_k)

// The O matrix (output).
DEFINE_GETTER_SETTER(void *, o_ptr)
DEFINE_GETTER_SETTER(void *, oaccum_ptr)

// The stride between rows of O.
DEFINE_GETTER_SETTER(int64_t, o_batch_stride)
DEFINE_GETTER_SETTER(int64_t, o_row_stride)
DEFINE_GETTER_SETTER(int64_t, o_head_stride)

// The pointer to the softmax sum.
DEFINE_GETTER_SETTER(void*, softmax_lse_ptr)
DEFINE_GETTER_SETTER(void*, softmax_lseaccum_ptr)

// For FP8 scaling
DEFINE_GETTER_SETTER(float *, q_descale_ptr)
DEFINE_GETTER_SETTER(float *, k_descale_ptr)
DEFINE_GETTER_SETTER(float *, v_descale_ptr)
DEFINE_GETTER_SETTER(int64_t, q_descale_batch_stride)
DEFINE_GETTER_SETTER(int64_t, q_descale_head_stride)
DEFINE_GETTER_SETTER(int64_t, k_descale_batch_stride)
DEFINE_GETTER_SETTER(int64_t, k_descale_head_stride)
DEFINE_GETTER_SETTER(int64_t, v_descale_batch_stride)
DEFINE_GETTER_SETTER(int64_t, v_descale_head_stride)

// The dimensions.
DEFINE_GETTER_SETTER(int, b)
DEFINE_GETTER_SETTER(int, seqlen_q)
DEFINE_GETTER_SETTER(int, seqlen_k)
DEFINE_GETTER_SETTER(int, seqlen_knew)
DEFINE_GETTER_SETTER(int, d)
DEFINE_GETTER_SETTER(int, seqlen_q_rounded)
DEFINE_GETTER_SETTER(int, seqlen_k_rounded)
DEFINE_GETTER_SETTER(int, d_rounded)
DEFINE_GETTER_SETTER(int, rotary_dim)
DEFINE_GETTER_SETTER(int, total_q)
DEFINE_GETTER_SETTER(int, total_k)
DEFINE_GETTER_SETTER(int, total_knew)
DEFINE_GETTER_SETTER(int, b_k)
DEFINE_GETTER_SETTER(int, dv)
DEFINE_GETTER_SETTER(int, dv_rounded)

// The scaling factors for the kernel.
DEFINE_GETTER_SETTER(float, scale_softmax)
DEFINE_GETTER_SETTER(float, softcap)

// array of length b+1 holding starting offset of each sequence.
DEFINE_GETTER_SETTER(int *, cu_seqlens_q)
DEFINE_GETTER_SETTER(int *, cu_seqlens_k)
DEFINE_GETTER_SETTER(int *, cu_seqlens_knew)
DEFINE_GETTER_SETTER(int *, leftpad_k)

// If provided, the actual length of each q/k sequence.
DEFINE_GETTER_SETTER(int *, seqused_q)
DEFINE_GETTER_SETTER(int *, seqused_k)

// The stride between rows of Oaccum.
DEFINE_GETTER_SETTER(int64_t, oaccum_split_stride)
DEFINE_GETTER_SETTER(int64_t, oaccum_batch_stride)
DEFINE_GETTER_SETTER(int64_t, oaccum_row_stride)
DEFINE_GETTER_SETTER(int64_t, oaccum_head_stride)

// The stride between rows of LSEaccum.
DEFINE_GETTER_SETTER(int64_t, lseaccum_split_stride)
DEFINE_GETTER_SETTER(int64_t, lseaccum_batch_stride)
DEFINE_GETTER_SETTER(int64_t, lseaccum_head_stride)

// The K_new and V_new matrices.
DEFINE_GETTER_SETTER(void *, knew_ptr)
DEFINE_GETTER_SETTER(void *, vnew_ptr)

// The stride between rows of the Q, K and V matrices.
DEFINE_GETTER_SETTER(int64_t, knew_batch_stride)
DEFINE_GETTER_SETTER(int64_t, vnew_batch_stride)
DEFINE_GETTER_SETTER(int64_t, knew_row_stride)
DEFINE_GETTER_SETTER(int64_t, vnew_row_stride)
DEFINE_GETTER_SETTER(int64_t, knew_head_stride)
DEFINE_GETTER_SETTER(int64_t, vnew_head_stride)

DEFINE_GETTER_SETTER(void *, qv_ptr)
DEFINE_GETTER_SETTER(int64_t, qv_batch_stride)
DEFINE_GETTER_SETTER(int64_t, qv_row_stride)
DEFINE_GETTER_SETTER(int64_t, qv_head_stride)

// The cos and sin matrices for rotary embedding.
DEFINE_GETTER_SETTER(void *, rotary_cos_ptr)
DEFINE_GETTER_SETTER(void *, rotary_sin_ptr)

// The indices to index into the KV cache.
DEFINE_GETTER_SETTER(int *, kv_batch_idx)

// Paged KV cache
DEFINE_GETTER_SETTER(int *, page_table)
DEFINE_GETTER_SETTER(int64_t, page_table_batch_stride)
DEFINE_GETTER_SETTER(int, page_size)
DEFINE_GETTER_SETTER(int, num_pages)
DEFINE_GETTER_SETTER(bool, pagedkv_tma)

// The dropout probability (probability of keeping an activation).
DEFINE_GETTER_SETTER(float, p_dropout)
// uint32_t p_dropout_in_uint;
// uint16_t p_dropout_in_uint16_t;
DEFINE_GETTER_SETTER(uint8_t, p_dropout_in_uint8_t)

// Scale factor of 1 / (1 - p_dropout).
DEFINE_GETTER_SETTER(float, rp_dropout)

// Local window size
DEFINE_GETTER_SETTER(int, window_size_left)
DEFINE_GETTER_SETTER(int, window_size_right)

// Pointer to the RNG seed (idx 0) and offset (idx 1).
DEFINE_GETTER_SETTER(uint64_t *, rng_state)

DEFINE_GETTER_SETTER(bool, is_bf16)
DEFINE_GETTER_SETTER(bool, is_fp32)
DEFINE_GETTER_SETTER(bool, is_e4m3)
DEFINE_GETTER_SETTER(bool, is_causal)
DEFINE_GETTER_SETTER(bool, is_local)

DEFINE_GETTER_SETTER(bool, is_rotary_interleaved)

DEFINE_GETTER_SETTER(int, num_splits)  // For split-KV version
DEFINE_GETTER_SETTER(bool, pack_gqa)

DEFINE_GETTER_SETTER(int *, tile_count_semaphore)
// int * __restrict__ num_m_blocks_ptr;
// int * __restrict__ num_n_blocks_ptr;
DEFINE_GETTER_SETTER(int *, num_splits_dynamic_ptr)
DEFINE_GETTER_SETTER(bool, skip_scheduler_metadata_computation)

DEFINE_GETTER_SETTER(int, arch)
DEFINE_GETTER_SETTER(int, num_sm)

DEFINE_GETTER_SETTER(int, h_flashmask)
DEFINE_GETTER_SETTER(int, h_h_flashmask_ratio)

DEFINE_GETTER_SETTER(int32_t *, lt_start_ptr)
DEFINE_GETTER_SETTER(int32_t *, lt_end_ptr)

DEFINE_GETTER_SETTER(int32_t *, ut_start_ptr)
DEFINE_GETTER_SETTER(int32_t *, ut_end_ptr)

DEFINE_GETTER_SETTER(int32_t *, flashmask_maxmin_ptr)

DEFINE_GETTER_SETTER(int32_t *, lt_start_nblockmax)
DEFINE_GETTER_SETTER(int32_t *, lt_start_nblockmin)

DEFINE_GETTER_SETTER(int32_t *, lt_end_nblockmax)
DEFINE_GETTER_SETTER(int32_t *, lt_end_nblockmin)

DEFINE_GETTER_SETTER(int32_t *, ut_start_nblockmax)
DEFINE_GETTER_SETTER(int32_t *, ut_start_nblockmin)

DEFINE_GETTER_SETTER(int32_t *, ut_end_nblockmax)
DEFINE_GETTER_SETTER(int32_t *, ut_end_nblockmin)

#define DEFINE_BWD_GETTER_SETTER(type, member) \
type flashmaskv2_bwd_params_get_##member(const Flash_bwd_params* params_handle) { return params_handle->member; } \
void flashmaskv2_bwd_params_set_##member(Flash_bwd_params* params_handle, type value) { params_handle->member = value; }

// The dO and dQKV matrices.
DEFINE_BWD_GETTER_SETTER(void *, do_ptr)
DEFINE_BWD_GETTER_SETTER(void *, dq_ptr)
DEFINE_BWD_GETTER_SETTER(void *, dk_ptr)
DEFINE_BWD_GETTER_SETTER(void *, dv_ptr)

// To accumulate dQ
DEFINE_BWD_GETTER_SETTER(void *, dq_accum_ptr)
DEFINE_BWD_GETTER_SETTER(void *, dk_accum_ptr)
DEFINE_BWD_GETTER_SETTER(void *, dv_accum_ptr)

// // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
// dimension void *__restrict__ dk_accum_ptr; void *__restrict__
// dv_accum_ptr;

// The stride between rows of the dO, dQ, dK and dV matrices.
DEFINE_BWD_GETTER_SETTER(int64_t, do_batch_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, do_row_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, do_head_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dq_batch_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dk_batch_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dv_batch_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dq_row_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dk_row_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dv_row_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dq_head_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dk_head_stride)
DEFINE_BWD_GETTER_SETTER(int64_t, dv_head_stride)

// The pointer to the softmax d sum.
DEFINE_BWD_GETTER_SETTER(void *, dsoftmax_sum)
DEFINE_BWD_GETTER_SETTER(void *, softmax_lse_log2_ptr)

DEFINE_BWD_GETTER_SETTER(int *, dq_semaphore)
DEFINE_BWD_GETTER_SETTER(int *, dk_semaphore)
DEFINE_BWD_GETTER_SETTER(int *, dv_semaphore)

DEFINE_BWD_GETTER_SETTER(bool, deterministic)
DEFINE_BWD_GETTER_SETTER(int64_t, dq_accum_split_stride)

#ifdef __cplusplus
}
#endif
