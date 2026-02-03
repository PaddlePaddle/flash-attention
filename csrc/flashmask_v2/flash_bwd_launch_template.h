/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/device_kernel.h"  // For device_kernel
#include "cutlass/kernel_launch.h"  // For kernel_launch
#include "cutlass/cluster_launch.hpp"  // For ClusterLauncher

#include "static_switch.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_postprocess_kernel.h"
#include "tile_scheduler.hpp"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "mainloop_bwd_sm80.hpp"
#include "epilogue_bwd.hpp"
#include "flash_bwd_kernel_sm90.h"
#include "flash_bwd_kernel_sm80.h"
#include "utils.h"

#ifdef NVSHMEM_DISTRIBUTED_OVERLAP
#include "distributed/overlap_comm.cuh"
#endif

using namespace cute;

template <int Arch, int kHeadDim, int kBlockM, int kBlockN, typename Element,
          bool Is_causal, bool Is_local, bool Has_softcap, bool Varlen, bool Deterministic, bool GQA,
          bool Is_flashmask, bool Has_lt_end, bool Has_ut_start, bool Is_blockmask,
          int Stages_dO=2, int Stages_dS_or_QSm80=2,
          bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
          int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
          bool V_in_regs=false>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("point3\n");
    // flash::print_addr_value<<<1, 1,0,stream>>>(params.lt_start_ptr, 0);
    static_assert(!(Is_causal && Is_local), "Is_causal and Is_local cannot be true at the same time.");
    using ElementAccum = float;
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

    int const total_q_padded_rounded = cute::round_up(params.total_q + params.b * kBlockM, kBlockM);
    int const total_k_padded_rounded = cute::round_up(params.total_k + params.b * kBlockN, kBlockN);
    bool const is_varlen_q = params.cu_seqlens_q;
    bool const is_varlen_k = params.cu_seqlens_k;
    int seqlen_q = !is_varlen_q ? params.seqlen_q : params.total_q;
    int seqlen_k = !is_varlen_k ? params.seqlen_k : params.total_k;
    int seqlen_q_rounded = !is_varlen_q ? params.seqlen_q_rounded : total_q_padded_rounded;
    int seqlen_k_rounded = !is_varlen_k ? params.seqlen_k_rounded : total_k_padded_rounded;
    int batch_q = !is_varlen_q ? params.b : 1;
    int batch_k = !is_varlen_k ? params.b : 1;
    // printf("params.dv_ptr:%p\n",params.dv_ptr);
    // printf("seqlen_q_rounded:%d\n",seqlen_q_rounded);
    // printf("d_rounded:%d\n",params.d_rounded);

    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, ArchTag, /*Clear_dQaccum=*/true, Varlen>;

#define MakePreprocessKernelArgs(KernelFuncType, var_name)          \
    typename KernelFuncType::Arguments var_name {                   \
        static_cast<Element const*>(params.o_ptr),                  \
        {seqlen_q, params.d, params.h, batch_q},  /* Shape O */     \
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen_q ? params.o_batch_stride : 0},  /* Stride O */    \
        static_cast<Element const*>(params.do_ptr),                                                                     \
        {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen_q ? params.do_batch_stride : 0}, /* Stride dO */ \
        static_cast<float*>(params.dsoftmax_sum),                                                                       \
        {seqlen_q_rounded, params.h, batch_q},  /* Shape dP Sum */                                                      \
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  /* Stride dP Sum */           \
        static_cast<float*>(params.softmax_lse_ptr),                                                                    \
        {_1{}, seqlen_q, !is_varlen_q ? params.h * params.seqlen_q : 0},  /* stride_LSE */                              \
        static_cast<float*>(params.softmax_lse_log2_ptr),                                                               \
        {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  /* stride_LSE_log2 */         \
        static_cast<ElementAccum*>(params.dq_accum_ptr),                                                                \
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  /* shape_dQaccum */                                  \
        {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * seqlen_q_rounded * params.h : 0},  /* stride_dQaccum */ \
        params.b,               \
        params.dq_semaphore,    \
        params.cu_seqlens_q,    \
        params.seqused_q        \
    }

    MakePreprocessKernelArgs(PreprocessKernel, preprocess_args);

    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = cute::ceil_div(params.seqlen_q, kBlockM);
    dim3 grid_m(num_m_block, params.h, params.b);
    flash::flashmask_kernel_launch<PreprocessKernel>(grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream, preprocess_params, false /*launch_with_pdl*/);
    CHECK_CUDA(cudaGetLastError());
    // flash::print_addr_value<<<1, 1,0,stream>>>(params.lt_start_ptr, 0);
    // printf("point2\n");
    CHECK_CUDA_KERNEL_LAUNCH();

    auto call_preprocess_without_cleanup = [&](int seg_idx) {
        // split calls do not clear dQ accum
        using PreprocessKernelSplit = flash::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, ArchTag, /*Clear_dQaccum=*/false, Varlen>;
        MakePreprocessKernelArgs(PreprocessKernelSplit, split_args);
        typename PreprocessKernelSplit::Params split_params = PreprocessKernelSplit::to_underlying_arguments(split_args);
        flash::flashmask_kernel_launch<PreprocessKernelSplit>(grid_m, PreprocessKernelSplit::MaxThreadsPerBlock,
                        PreprocessKernelSplit::SharedStorageSize, stream, split_params, false /*launch_with_pdl*/);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA_KERNEL_LAUNCH();
    };  

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;  // Currently doesn't not support cluster
    // Stages_dS_or_QSm80 is Stages_dS if Sm90 and Stages if Sm80
    static constexpr int Stages = Arch >= 90 ? 2 : Stages_dS_or_QSm80;
    static constexpr int Stages_dS = Arch >= 90 ? Stages_dS_or_QSm80 : 1;
    using CollectiveMainloop = std::conditional_t<
        Arch >= 90,
        flash::CollectiveMainloopBwdSm90<Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm90,
            Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
            SdP_swapAB, dKV_swapAB, dQ_swapAB, Is_flashmask, Has_lt_end, Has_ut_start,Is_blockmask, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>,
        flash::CollectiveMainloopBwdSm80<Stages, Stages_dO, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm80,
            Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
            SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>
    >;
    using CollectiveEpilogue = std::conditional_t<
        !GQA,
        flash::CollectiveEpilogueBwd<TileShape_MNK, Element, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, dKV_swapAB, NumMmaWarpGroups * (Arch >= 90 ? 1 : cutlass::NumWarpsPerWarpGroup) / AtomLayoutNdKV>,
        flash::CollectiveEpilogueBwdGQA<TileShape_MNK, ElementAccum, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, Deterministic>
    >;
    using Scheduler = std::conditional_t<
        Arch >= 90,
        flash::BwdPreemptivePersistentTileScheduler<CollectiveMainloop::NumMmaThreads, cutlass::NumThreadsPerWarpGroup, Deterministic>,
        flash::SingleTileScheduler<Varlen, false /*Split*/, false /*PackGQA*/, kBlockN>
    >;
    using AttnKernel = std::conditional_t<
        Arch >= 90,
        flash::enable_sm90_or_later<flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>,
        flash::enable_sm80_to_sm89<flash::FlashAttnBwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>
    >;

    bool use_overlap = false, overlap_rs = false;
    int segment_idx = 0, segment_cnt = 1;
#ifdef NVSHMEM_DISTRIBUTED_OVERLAP
    use_overlap = params.nranks > 1 && (!flashmask::comm::is_singleton_null());
    std::unique_ptr<flash::flashmask::MaskPtrUpdater<kBlockN>> updater = nullptr;

    // in bwd, we don't need to set rank, cp_size for params, since overlap_comm instance already knew these in fwd
    // also, when nranks == 1, overlap mode is not used
    if (use_overlap) {
        auto& comm_singleton = flashmask::comm::singleton();
        segment_cnt = comm_singleton.num_segments();
        overlap_rs = segment_cnt > 1;

        // Chunk mask does not affect sync behavior, and it does not need to be computed in a splitted way.
        // Also, only one chunk (the local chunk) needs to be moved to the SR buffer. For RS-overlap, only the first
        // segment needs to call 'update_kv_buffer' since the following segments do not have local chunks. The
        // chunk_mask and update_kv_buffer can actually be called only once. AG and RS overlap are called 4 (num_segs) times.
        comm_singleton.wait_sr_buffer_empty();
        comm_singleton.reset_dkv_semaphores(stream);        // RS-overlap: reset dK, dV semaphores to all 0
        comm_singleton.compute_chunk_mask(params.lt_start_ptr, params.ut_end_ptr, stream, false /* fwd */);
        comm_singleton.update_kv_buffer((const Element*) params.k_ptr, (const Element*) params.v_ptr, false /*fwd*/);     // copy new KV data

        // seqlen_scale: when use_rs is true, this is chunks_per_seg (4 if CP16), otherwise this is cp_size
        params.seqlen_k *= comm_singleton.seqlen_scale();
        seqlen_k *= comm_singleton.seqlen_scale();
        params.k_batch_stride *= comm_singleton.seqlen_scale();
        params.v_batch_stride *= comm_singleton.seqlen_scale();
        
        // Re-route the KV data to the nvshmem_alloc SR buffer.
        params.k_ptr = comm_singleton.k_data();
        params.v_ptr = comm_singleton.v_data();
        // prepare mask_ptr updater and set params.num_segments to enable correct mask access in bwd kernel
        if (overlap_rs) {
            static constexpr int cp_chunk_size = 8192;
            updater = std::make_unique<flash::flashmask::MaskPtrUpdater<kBlockN>>(params, cp_chunk_size, comm_singleton.chunk_per_seg());
        }
    } else if (params.nranks > 1) {
        throw std::runtime_error("Overlap singleton instance is null but we try using overlap mechanism. This should be buggy.");
    }

SEGMENT_LOOP_START:
    if constexpr (Arch >= 90) {
        prepare_flashmask(params, stream, params.num_sm,
            use_overlap ? &flashmask::comm::singleton().wptr_init : nullptr,
            use_overlap ? flashmask::comm::singleton().get_block_cnt_semaphore() : nullptr);
    }

    if (use_overlap) {
        auto& comm_singleton = flashmask::comm::singleton();
        // Note(heqianyue): for RS-overlap, before the last computation kernel, communication kernel won't start
        comm_singleton.wait_wptr_init();        // wait until wptr is initialized
        if (overlap_rs) {  // RS-overlap splits the AG and attn kernel
            comm_singleton.run_overlap_splitted_ag_kernel(
                params.lt_start_ptr, params.ut_end_ptr, params.write_ptr, params.seqlen_k, segment_idx
            );
            // offset K and V by one chunk
            params.k_ptr = comm_singleton.k_data(segment_idx ? 1 : 0);
            params.v_ptr = comm_singleton.v_data(segment_idx ? 1 : 0);
            // make sure computation kernels are scheduled with SMs later than communication kernels
            comm_singleton.wait_reset_stream_coordinator(segment_idx > 0, stream);
        } else {
            comm_singleton.run_overlap_ag_kernel(
                params.lt_start_ptr, params.ut_end_ptr, params.write_ptr, params.seqlen_k, false /*fwd*/
            );
        }
    }
#else
    if constexpr (Arch >= 90) {
        prepare_flashmask(params, stream, params.num_sm);
    }
#endif  // NVSHMEM_DISTRIBUTED_OVERLAP

    if (segment_idx == 0) {
        // scanMinMax is called only once
        flash::flashmask::prepare_block_maxmin<kBlockN>(params, stream, false /* is_forward */, segment_cnt);
    } else {
        // reset grad semaphores, otherwise the program will hang
        call_preprocess_without_cleanup(segment_idx);
        if constexpr (Deterministic) {
            size_t total_bytes = (params.seqlen_k + kBlockN - 1) / kBlockN * params.b * params.h_k * sizeof(int);
            cudaMemsetAsync(params.dk_semaphore, 0, total_bytes, stream);
            cudaMemsetAsync(params.dv_semaphore, 0, total_bytes, stream);
        }
    }

    typename CollectiveMainloop::Arguments mainloop_args = [&] () {
        if constexpr(Arch >= 90)
            return typename CollectiveMainloop::Arguments {
                static_cast<Element const*>(params.q_ptr),
                {seqlen_q, params.d, params.h, batch_q},  // shape_Q
                {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},  // stride_Q
                static_cast<Element const*>(params.k_ptr),
                {seqlen_k, params.d, params.h_k, batch_k},  // shape_K
                {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},  // stride_K
                static_cast<Element const*>(params.v_ptr),
                {params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0},  // stride_V
                static_cast<Element const*>(params.do_ptr),
                {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen_q ? params.do_batch_stride : 0},  // stride_dO
                static_cast<ElementAccum*>(params.dq_accum_ptr),
                {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
                {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
                static_cast<float*>(params.softmax_lse_log2_ptr),
                {seqlen_q_rounded, params.h, batch_q},  // shape_LSE
                {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_LSE_log2
                static_cast<float*>(params.dsoftmax_sum),
                {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_dPsum
                params.scale_softmax,
                params.window_size_left, params.window_size_right,
                params.softcap,
                params.b,
                params.dq_semaphore,
                params.cu_seqlens_q, params.cu_seqlens_k,
                params.seqused_q, params.seqused_k,
                params.h_flashmask, params.h_h_flashmask_ratio,
                // RS-overlap: the following mask ptrs will be updated by the updater
                params.lt_start_ptr, params.lt_end_ptr,
                params.ut_start_ptr, params.ut_end_ptr,
                params.lt_start_nblockmax, params.lt_start_nblockmin,
                params.lt_end_nblockmax, params.lt_end_nblockmin,
                params.ut_start_nblockmax, params.ut_start_nblockmin,
                params.ut_end_nblockmax, params.ut_end_nblockmin,
                params.m_block_dim, params.n_block_dim,
                params.block_mask_ptr,
                params.write_ptr,
                segment_cnt
            };
        else
            return typename CollectiveMainloop::Arguments {
                static_cast<Element const*>(params.q_ptr),
                {seqlen_q, params.d, params.h, batch_q},  // shape_Q
                {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen_q ? params.q_batch_stride : 0},  // stride_Q
                static_cast<Element const*>(params.k_ptr),
                {seqlen_k, params.d, params.h_k, batch_k},  // shape_K
                {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen_k ? params.k_batch_stride : 0},  // stride_K
                static_cast<Element const*>(params.v_ptr),
                {params.v_row_stride, _1{}, params.v_head_stride, !is_varlen_k ? params.v_batch_stride : 0},  // stride_V
                static_cast<Element const*>(params.do_ptr),
                {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen_q ? params.do_batch_stride : 0},  // stride_dO
                static_cast<ElementAccum*>(params.dq_accum_ptr),
                {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
                {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
                static_cast<float*>(params.softmax_lse_log2_ptr),
                {seqlen_q_rounded, params.h, batch_q},  // shape_LSE
                {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_LSE_log2
                static_cast<float*>(params.dsoftmax_sum),
                {_1{}, seqlen_q_rounded, !is_varlen_q ? params.h * params.seqlen_q_rounded : 0},  // stride_dPsum
                params.scale_softmax,
                params.window_size_left, params.window_size_right,
                params.softcap,
                params.b,
                params.dq_semaphore,
                params.cu_seqlens_q, params.cu_seqlens_k,
                params.seqused_q, params.seqused_k};
            }();        
    // The case work with GQA is ugly but idk how to fix it.
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<typename CollectiveEpilogue::Element*>(!GQA ? params.dk_ptr : params.dk_accum_ptr),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k, params.d, params.h, batch_k};  // shape_dK
            } else {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k};  // shape_dKaccum
            }
        }(),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::StridedKV {params.dk_row_stride, _1{}, params.dk_head_stride, !is_varlen_k ? params.dk_batch_stride : 0};  // stride_dK
            } else {
                return typename CollectiveEpilogue::StridedKV {_1{}, params.d_rounded * seqlen_k_rounded, !is_varlen_k ? params.h_k * params.d_rounded * params.seqlen_k_rounded : 0};  // stride_dKaccum
            }
        }(),
        static_cast<typename CollectiveEpilogue::Element*>(!GQA ? params.dv_ptr : params.dv_accum_ptr),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::StridedKV {params.dv_row_stride, _1{}, params.dv_head_stride, !is_varlen_k ? params.dv_batch_stride : 0};  // stride_dV
            } else {
                return typename CollectiveEpilogue::StridedKV {_1{}, params.d_rounded * seqlen_k_rounded, !is_varlen_k ? params.h_k * params.d_rounded * params.seqlen_k_rounded : 0};  // stride_dVaccum
            }
        }(),
        params.h,
        params.dk_semaphore,
        params.dv_semaphore,
        params.cu_seqlens_k,
        params.seqused_k,
    };

    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_n, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_k,
        params.seqlen_q, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens_k, params.seqused_k
    };

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetLastError());
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // printf("tensor_size = %d\n",AttnKernel::TensorStorageSize);
    // printf("ppl_size = %d\n",AttnKernel::PipelineStorageSize);
    // int smem_size_q = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_q));
    // int smem_size_do = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_do));
    // int smem_size_ds = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_ds));
    // int smem_size_dqacc = [&] {
    //     if constexpr (Arch >= 90) {
    //         return sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dqacc));
    //     } else {
    //         return 0;
    //     }
    // }();
    // int smem_size_k = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v));
    // int smem_size_lse = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_lse));
    // int smem_size_dpsum = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dpsum));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, do = %d, ds = %d, dqacc = %d, lse = %d, dpsum = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_do, smem_size_ds, smem_size_dqacc, smem_size_lse, smem_size_dpsum);
    if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) flash::cutlass_flashmask_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLauncher::launch(
            grid_dims, cluster_dims, block_dims, smem_size, stream, kernel, kernel_params, false /*launch_with_pdl*/);
    } else {
        void const* kernel = (void const*) flash::cutlass_flashmask_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            int max_smem;
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
            // printf("smem_size = %d, max_smem = %d\n", smem_size, max_smem);
            CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            // printf("pass");
        }
        flash::flashmask_kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/);
    }
    CHECK_CUDA_KERNEL_LAUNCH();

    using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_MK, Element, ElementAccum, ArchTag,
        AttnKernel::CollectiveMainloop::NumMmaThreads,
        typename AttnKernel::CollectiveMainloop::TiledMmadQ,
        AttnKernel::CollectiveMainloop::dQ_swapAB
        >;

    if (segment_idx == segment_cnt - 1) {
        // only the last segments should call dQ post process to down cast dQ accum
        typename PostprocessKernel::Arguments postprocess_args {
            static_cast<ElementAccum const*>(params.dq_accum_ptr),
            {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
            {_1{}, seqlen_q_rounded * params.d_rounded, !is_varlen_q ? params.d_rounded * params.seqlen_q_rounded * params.h : 0}, // stride_dQaccum
            static_cast<Element*>(params.dq_ptr),
            {seqlen_q, params.d, params.h, batch_q},  // shape_dQ
            {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},  // stride_dQ
            params.scale_softmax,
            params.cu_seqlens_q,
            params.seqused_q
        };
        typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments(postprocess_args);
        int num_m_block_postprocess = cute::ceil_div(params.seqlen_q, get<0>(TileShape_MK{}));
        dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
        int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
        if (smem_size_postprocess >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(flash::cutlass_flashmask_kernel<PostprocessKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
        }
        flash::flashmask_kernel_launch<PostprocessKernel>(grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_params, false /*launch_with_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

    if constexpr (GQA) {
        using TileShape_NK = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
        using PostprocessKerneldKV = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_NK, Element, ElementAccum, ArchTag,
            AttnKernel::CollectiveEpilogue::NumEpilogueThreads,
            typename AttnKernel::CollectiveMainloop::TiledMmadKV,
            AttnKernel::CollectiveMainloop::dKV_swapAB
            >;

        Element* dk_buffer = static_cast<Element*>(params.dk_ptr);
        Element* dv_buffer = static_cast<Element*>(params.dv_ptr);
        const int batch_stride_dkv = params.d_rounded * params.seqlen_k_rounded * params.h_k;
        if (overlap_rs) {
            auto& comm_singleton = flashmask::comm::singleton();
            // post-process outputs to send buffer so that we can directly send it.
            dk_buffer = static_cast<Element*>(comm_singleton.dk_send());
            dv_buffer = static_cast<Element*>(comm_singleton.dv_send());
        }
        // Note(heqianyue): when RS-overlap is switched ON, the shape of dk_ptr (and dv_ptr) is (B, S_local, H, D)
        // while the dk_accum & dv_accum & dk_send, dv_send (NVSHMEM buffer) have shape (B, S_local * chunks_per_seg, H, D)
        // we therefore need to re-route the output of post-process kernels to dk_send, dv_send. The final reduced output
        // dk_ptr and dv_ptr requires is produced by rs_overlap_kernel.
        typename PostprocessKerneldKV::Arguments postprocess_dK_args {
            static_cast<ElementAccum const*>(params.dk_accum_ptr),
            {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k},  // shape_dKaccum
            {_1{}, seqlen_k_rounded * params.d_rounded, !is_varlen_k ? batch_stride_dkv : 0},  // stride_dKaccum
            dk_buffer,
            {seqlen_k, params.d, params.h_k, batch_k},  // shape_dK
            {params.dk_row_stride, _1{}, params.dk_head_stride, overlap_rs ? batch_stride_dkv : params.dk_batch_stride},  // stride_dK
            1.f,
            params.cu_seqlens_k,
            params.seqused_k
        };
        typename PostprocessKerneldKV::Params postprocess_dK_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dK_args);
        typename PostprocessKerneldKV::Arguments postprocess_dV_args {
            static_cast<ElementAccum const*>(params.dv_accum_ptr),
            {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k},  // shape_dVaccum
            {_1{}, seqlen_k_rounded * params.d_rounded, !is_varlen_k ? batch_stride_dkv : 0},  // stride_dVaccum
            dv_buffer,
            {seqlen_k, params.d, params.h_k, batch_k},  // shape_dV
            {params.dv_row_stride, _1{}, params.dv_head_stride, overlap_rs ? batch_stride_dkv : params.dv_batch_stride},  // stride_dV
            1.f,
            params.cu_seqlens_k,
            params.seqused_k
        };
        typename PostprocessKerneldKV::Params postprocess_dV_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dV_args);
        int num_n_block_postprocess = cute::ceil_div(params.seqlen_k, get<0>(TileShape_NK{}));
        dim3 grid_n_postprocess(num_n_block_postprocess, params.h_k, params.b);
        int smem_size_postprocess = PostprocessKerneldKV::SharedStorageSize;
        if (smem_size_postprocess >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(flash::cutlass_flashmask_kernel<PostprocessKerneldKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
        }
#ifdef NVSHMEM_DISTRIBUTED_OVERLAP
        if (overlap_rs && segment_idx) {
            // post-process kernel must wait for the RS-reduce finishing. Since we redirect the output buffer of post-process to dk/v_send
            // these two buffers are also used in RS-overlap (remote put and reduce), so we cannot overwrite these before they are released. 
            auto& comm_singleton = flashmask::comm::singleton();
            cudaStreamWaitEvent(stream, comm_singleton.send_done);      // wait RS-overlap producer finishing using dk/dv-send
            cudaStreamWaitEvent(stream, comm_singleton.reduce_done);    // wait RS-overlap producer finishing using dk/dv-send
        }
#endif  // NVSHMEM_DISTRIBUTED_OVERLAP
        flash::flashmask_kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dK_params, false /*launch_with_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
        flash::flashmask_kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dV_params, false /*launch_with_pdl*/);
        CHECK_CUDA_KERNEL_LAUNCH();
    }
#ifdef NVSHMEM_DISTRIBUTED_OVERLAP
    // TODO(heqianyue): we don't need to constrain the function with 'GQA', but
    // for simplicity of output pointer, GQA is required for now
    if (overlap_rs) {
        auto& comm_singleton = flashmask::comm::singleton();
        // dk_ptr and dv_ptr is the output
        comm_singleton.run_overlap_rs_kernel(
            static_cast<Element*>(params.dk_ptr),
            static_cast<Element*>(params.dv_ptr),
            segment_idx,
            stream
        );
        segment_idx ++;
        if (segment_idx < segment_cnt) {
            // Note(heqianyue): this is so damned, be extra careful here: for GQA, epilogue
            // does not set blocks to zero if they are masked. So we need to manually set zero
            if constexpr (GQA) {
                // need to set dk/v_accum to zero, so that the dK, dV from previous segments won't 
                // contaminate the later computation (GQA epilgue store_zero does nothing) 
                size_t accum_bytes = params.b * params.seqlen_k_rounded * params.h_k * params.d_rounded * sizeof(float);
                cudaMemsetAsync(params.dk_accum_ptr, 0, accum_bytes, stream);
                cudaMemsetAsync(params.dv_accum_ptr, 0, accum_bytes, stream);
            }
            updater->inplace_update();
            goto SEGMENT_LOOP_START;
        }
        // consumer (dKdV reduce) stream will record event for compute stream to wait for
        cudaStreamWaitEvent(stream, comm_singleton.reduce_done);
    }
#endif  // NVSHMEM_DISTRIBUTED_OVERLAP
}

template<int Arch, typename T, int kBlockM, int kBlockN, int kHeadDim, bool Is_causal, bool Is_local, bool Has_softcap,
         bool Is_flashmask_, bool Has_lt_end_, bool Has_ut_start_, bool Deterministic, bool Is_blockmask_, 
         int Stages_dO=2, int Stages_dS_or_QSm80=2,
         bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
         int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
         bool V_in_regs=false>
void run_mha_bwd_dispatch(Flash_bwd_params &params, cudaStream_t stream) {
    VARLEN_SWITCH(params.cu_seqlens_q != nullptr || params.cu_seqlens_k != nullptr, Varlen, [&] {
        BOOL_SWITCH(params.h != params.h_k, GQA, [&] {
            // run_flash_bwd<kHeadDim, kBlockM, kBlockN, T, Is_causal, Is_local, Has_softcap, Varlen, false, GQA, Stages_dO, Stages_dS_or_QSm80, SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ>(params, stream);   
            run_flash_bwd<Arch, kHeadDim, kBlockM, kBlockN, T, Is_causal, Is_local, Has_softcap, Varlen /*Varlen*/, Deterministic /*Deterministic*/, GQA, Is_flashmask_, Has_lt_end_, Has_ut_start_,Is_blockmask_, Stages_dO, Stages_dS_or_QSm80, SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>(params, stream);
        });
    });
}


template<int Arch, typename T, bool Has_softcap, bool Is_causal, bool Deterministic>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    // printf("point2-1\n");
    static constexpr bool Is_local = false;
    static constexpr bool Is_flashmask_ = true;
    BOOL_SWITCH(params.block_mask_ptr != nullptr, Is_blockmask_, [&]{
        FLASH_MASK_SWITCH(params.lt_end_ptr != nullptr, params.ut_start_ptr != nullptr, Has_lt_end, Has_ut_start, [&] {
            if constexpr (Arch >= 90) {
                if constexpr (Is_flashmask_ && !Is_causal) {
                   run_mha_bwd_dispatch<Arch, T, 64, 96, 64, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, Is_blockmask_, 2, 2, false, true, false, 2, 1, 2, 1, false>(params, stream);
                } else if constexpr (Is_causal && Has_softcap || Is_flashmask_) {
                    // register spill with 128 x 128
                    run_mha_bwd_dispatch<Arch, T, 96, 128, 64, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, Is_blockmask_, 2, 2, true, false, true, 2, 1, 2, 2, false>(params, stream);
                } else {
                    // With ShuffleStats we no longer have register spilling when Has_softcap and using 128 x 128 block.
                    run_mha_bwd_dispatch<Arch, T, 128, 128, 64, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, Is_blockmask_, 2, 2, true, false, false, 2, 1, 2, 2, false>(params, stream);
                }
            } else if constexpr (Arch == 86 || Arch == 89) {
                run_mha_bwd_dispatch<Arch, T, 64, 128, 64, Is_causal, Is_local, Has_softcap, 2, 2, false, false, false, 2, 2, 4, 2, true, Is_flashmask_>(params, stream);
                // run_mha_bwd_dispatch<Arch, T, 96, 96, 64, Is_causal, Is_local, Has_softcap, 1, 2, false, true, true, 2, 2, 4, 4, false>(params, stream);
                // run_mha_bwd_dispatch<Arch, T, 80, 128, 64, Is_causal, Is_local, Has_softcap, 1, 2, true, false, true, 2, 2, 4, 2, true>(params, stream);
                // run_mha_bwd_dispatch<Arch, T, 96, 128, 64, Is_causal, Is_local, Has_softcap, 1, 2, true, false, true, 2, 1, 8, 4, false>(params, stream);
            } else {
                run_mha_bwd_dispatch<Arch, T, 128, 128, 64, Is_causal, Is_local, Has_softcap, 2, 2, false, false, false, 2, 4, 4, 4, false, Is_flashmask_>(params, stream);
            }
        });
    });
}

template<int Arch, typename T, bool Has_softcap, bool Is_causal, bool Deterministic>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    static constexpr bool Is_local = false;
    static constexpr bool Is_flashmask_ = true;
    FLASH_MASK_SWITCH(params.lt_end_ptr != nullptr, params.ut_start_ptr != nullptr, Has_lt_end, Has_ut_start, [&] {
        if constexpr (Arch >= 90) {
            run_mha_bwd_dispatch<Arch, T, 64, 128, 96, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, 2, 2, true, false, false, 2, 1, 2, 1, true>(params, stream);
        } else if constexpr (Arch == 86 || Arch == 89) {
            run_mha_bwd_dispatch<Arch, T, 64, 128, 96, Is_causal, Is_local, Has_softcap, 1, 2, false, false, false, 2, 2, 4, 2, true, Is_flashmask_>(params, stream);
        } else {
            run_mha_bwd_dispatch<Arch, T, 64, 128, 96, Is_causal, Is_local, Has_softcap, 2, 2, false, false, false, 2, 2, 4, 2, false, Is_flashmask_>(params, stream);
        }
    });
}

template<int Arch, typename T, bool Has_softcap, bool Is_causal, bool Deterministic>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    static constexpr bool Is_local = false;
    static constexpr bool Is_flashmask_ = true;
    BOOL_SWITCH(params.block_mask_ptr != nullptr, Is_blockmask_, [&]{
        FLASH_MASK_SWITCH(params.lt_end_ptr != nullptr, params.ut_start_ptr != nullptr, Has_lt_end, Has_ut_start, [&] {
            if constexpr (Arch >= 90) {
                if constexpr (Is_causal || Is_local || Has_softcap) {
                    run_mha_bwd_dispatch<Arch, T, 64, 128, 128, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, Is_blockmask_, 2, 2, true, false, false, 2, 1, 2, 1, false>(params, stream);
                } else {
                    if ((params.seqlen_q >= 1024 || params.seqlen_k >= 1024) && !(Has_lt_end && Has_ut_start)) {
                    run_mha_bwd_dispatch<Arch, T, 64, 128, 128, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, Is_blockmask_, 2, 2, true, false, true, 2, 1, 2, 1, false>(params, stream);
                    } else {
                    run_mha_bwd_dispatch<Arch, T, 64, 64, 128, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, Is_blockmask_, 2, 2, false, true, false, 2, 1, 2, 1, false>(params, stream);
                    }
                }
            } else if constexpr (Arch == 86 || Arch == 89) {
                run_mha_bwd_dispatch<Arch, T, 64, 96, 128, Is_causal, Is_local, Has_softcap, 1, 2, false, false, false, 2, 2, 2, 2, true, Is_flashmask_>(params, stream);
            } else {
                run_mha_bwd_dispatch<Arch, T, 64, 128, 128, Is_causal, Is_local, Has_softcap, 2, 2, false, false, false, 2, 2, 2, 2, false, Is_flashmask_>(params, stream);
            }
        });
    });
}

template<int Arch, typename T, bool Has_softcap, bool Is_causal, bool Deterministic>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    static constexpr bool Is_local = false;
    static constexpr bool Is_flashmask_ = true;
    FLASH_MASK_SWITCH(params.lt_end_ptr != nullptr, params.ut_start_ptr != nullptr, Has_lt_end, Has_ut_start, [&] {
        if constexpr (Arch >= 90) {
            if (Has_lt_end && Has_ut_start) {
                run_mha_bwd_dispatch<Arch, T, 64, 48, 192, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, 1, 1, false, true, false, 3, 1, 1, 1, false>(params, stream);
            } else {
                run_mha_bwd_dispatch<Arch, T, 64, 96, 192, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, 1, 1, false, true, false, 3, 1, 1, 1, false>(params, stream);
            }
        } else if constexpr (Arch == 86 || Arch == 89) {
            run_mha_bwd_dispatch<Arch, T, 64, 64, 192, Is_causal, Is_local, Has_softcap, 1, 1, false, false, false, 2, 2, 2, 2, true, Is_flashmask_>(params, stream);
        } else {
            run_mha_bwd_dispatch<Arch, T, 64, 80, 192, Is_causal, Is_local, Has_softcap, 1, 2, false, true, false, 2, 4, 2, 2, false, Is_flashmask_>(params, stream);
        }
    });
}

template<int Arch, typename T, bool Has_softcap, bool Is_causal, bool Deterministic>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    static constexpr bool Is_local = false;
    static constexpr bool Is_flashmask_ = true;
    BOOL_SWITCH(params.block_mask_ptr != nullptr, Is_blockmask_, [&]{
        FLASH_MASK_SWITCH(params.lt_end_ptr != nullptr, params.ut_start_ptr != nullptr, Has_lt_end, Has_ut_start, [&] {
            if constexpr (Arch >= 90) {
                if (Has_lt_end && Has_ut_start) {
                    run_mha_bwd_dispatch<Arch, T, 64, 32, 256, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, Is_blockmask_, 1, 1, false, true, true, 2, 1, 1, 1, false>(params, stream);
                } else {
                    run_mha_bwd_dispatch<Arch, T, 64, 64, 256, Is_causal, Is_local, Has_softcap, Is_flashmask_, Has_lt_end, Has_ut_start, Deterministic, Is_blockmask_, 1, 1, false, true, true, 2, 1, 1, 1, false>(params, stream);
                }
            } else if constexpr (Arch == 86 || Arch == 89) {
                run_mha_bwd_dispatch<Arch, T, 32, 64, 256, Is_causal, Is_local, Has_softcap, 1, 1, false, false, false, 2, 2, 2, 1, true, Is_flashmask_>(params, stream);
            } else {
                run_mha_bwd_dispatch<Arch, T, 64, 64, 256, Is_causal, Is_local, Has_softcap, 1, 1, false, false, false, 2, 4, 2, 2, false, Is_flashmask_>(params, stream);
            }
        });
    });
}
