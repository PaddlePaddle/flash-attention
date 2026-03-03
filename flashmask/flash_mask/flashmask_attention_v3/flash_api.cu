/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 * Pradeep Ramani, Tri Dao.
 *
 * Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
                                #ifndef FLASHMASK_V3_DISABLE_HDIM64
                                if (params.d <= 64) {
                                    if (params.dv > 64 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                    else {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHMASK_V3_DISABLE_HDIM96
                                if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHMASK_V3_DISABLE_HDIM128
                                if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHMASK_V3_DISABLE_HDIM192
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHMASK_V3_DISABLE_HDIM256
                                if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                            } else {
                                #ifndef FLASHMASK_V3_DISABLE_FP16
                                #ifndef FLASHMASK_V3_DISABLE_HDIM64
                                if (params.d <= 64) {
                                    if (params.dv > 64 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 512, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                    else {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHMASK_V3_DISABLE_HDIM96
                                if (params.d <= 96) { return run_mha_fwd_<Arch, cutlass::half_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHMASK_V3_DISABLE_HDIM128
                                if (params.d <= 128) { return run_mha_fwd_<Arch, cutlass::half_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #ifndef FLASHMASK_V3_DISABLE_HDIM192
                                if (params.d <= 192) {
                                    if (params.dv <= 128 && Arch == 90) {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    } else {
                                        return run_mha_fwd_<Arch, cutlass::half_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                    }
                                }
                                #endif
                                #ifndef FLASHMASK_V3_DISABLE_HDIM256
                                if (params.d <= 256) { return run_mha_fwd_<Arch, cutlass::half_t, 256, 256, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                                #endif
                                #else
                                PADDLE_CHECK(false, "This flash attention build does not support FP16.");
                                #endif
                            }
                        } else {
                            #ifndef FLASHMASK_V3_DISABLE_FP8
                            #ifndef FLASHMASK_V3_DISABLE_HDIM64
                            if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHMASK_V3_DISABLE_HDIM96
                            if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHMASK_V3_DISABLE_HDIM128
                            if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream); }
                            #endif
                            #ifndef FLASHMASK_V3_DISABLE_HDIM192
                            if (params.d <= 192) {
                                if (params.dv <= 128 && Arch == 90) {
                                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 128, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                } else {
                                    return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, 192, Split, PagedKVNonTMA, Has_softcap, PackGQA>(params, stream);
                                }
                            }
                            #endif
                            #ifndef FLASHMASK_V3_DISABLE_HDIM256
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
    #ifndef FLASHMASK_V3_DISABLE_SPLIT
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

bool is_short_seqlen(Flash_fwd_params const& params) {
    return params.seqlen_k < 128 && params.seqlen_q < 128;
}

bool get_pagedkv_tma(Flash_fwd_params const& params) {
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

bool get_pack_gqa(Flash_fwd_params const& params) {
    // Always enable PackGQA for Sm8x or PagedKVNonTMA or Split to reduce compilation and binary size.
    // Has little effect on speed.
    if (params.arch < 90 || (params.page_table && !params.pagedkv_tma) || params.num_splits > 1) { return true; }
    #ifdef FLASHMASK_V3_DISABLE_PACKGQA
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

int get_num_splits(Flash_fwd_params const& params) {
    #ifdef FLASHMASK_V3_DISABLE_SPLIT
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
    #ifndef FLASHMASK_V3_DISABLE_BACKWARD
        // FP16_SWITCH(!params.is_bf16, [&] {
        //     HEADDIM_SWITCH(params.d, [&] {
        //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        //     });
        // });
    // printf("params.d = %d",params.d);
    ARCH_SWITCH(params.arch, Arch, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                    if (!params.is_bf16) {
                        #ifndef FLASHMASK_V3_DISABLE_FP16
                        #ifndef FLASHMASK_V3_DISABLE_HDIM64
                        if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::half_t, 64, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #ifndef FLASHMASK_V3_DISABLE_HDIM96
                        if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::half_t, 96, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #ifndef FLASHMASK_V3_DISABLE_HDIM128
                        if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::half_t, 128, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #ifndef FLASHMASK_V3_DISABLE_HDIM192
                        if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::half_t, 192, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #ifndef FLASHMASK_V3_DISABLE_HDIM256
                        if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::half_t, 256, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #else
                        PADDLE_CHECK(false, "This flash attention build does not support FP16.");
                        #endif
                    } else {
                        #ifndef FLASHMASK_V3_DISABLE_HDIM64
                        if (params.d <= 64) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #ifndef FLASHMASK_V3_DISABLE_HDIM96
                        if (params.d <= 96) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #ifndef FLASHMASK_V3_DISABLE_HDIM128
                        if (params.d <= 128) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #ifndef FLASHMASK_V3_DISABLE_HDIM192
                        if (params.d <= 192) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        #ifndef FLASHMASK_V3_DISABLE_HDIM256
                        if (params.d <= 256) { return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256, Has_softcap, Is_causal, Deterministic>(params, stream); }
                        #endif
                        PADDLE_CHECK(false, "This flash attention build does not support ");
                    }
                });
            });
        });
    });
    #endif
}
