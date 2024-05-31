#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "philox.cuh"
namespace flash {

using namespace cute;

#define p(x) do {printf("\n%s\n",#x);print(x);} while(false)

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_B_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = decltype(size<1>(TileShape_MNK{}))::value / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    // This gives the correct layout, idk why.
    // auto t = make_tile(Layout<Shape<Shape<_8, _2>, _2>,
    //                           Stride<Stride<_1, _64>, _8> >{},
    // auto t = make_tile(Layout<Shape<_8, _2, _2>,
    //                           Stride<_1, _64, _8> >{},
    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{},       // (1, 64, 8) or (1, 32, 8)
                       make_layout(size<2>(TileShape_MNK{})));
    // if (cute::thread0()) {printf("make_tiled_copy_B_warpcontiguousN "); print(t); printf("\n");  }
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    using TileShape_MNK = typename TiledMMA::TiledShape_MNK;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = decltype(size<1>(TileShape_MNK{}))::value / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(make_layout(size<0>(TileShape_MNK{})),
                       Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{});       // (1, 64, 8) or (1, 32, 8)
    // if (cute::thread0()) {printf("make_tiled_copy_C_warpcontiguousN "); print(t); printf("\n");  }
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename TiledCopy>
inline __device__ void write_softmax_to_gmem(
    Tensor<Engine0, Layout0> const &tPrP, Tensor<Engine1, Layout1> &tPgP, TiledCopy gmem_tiled_copy_P
) {
#if 0
    // Reshape tOrP from (8, MMA_M, MMA_N) to (8, MMA_M * MMA_N)
    Layout l = tOrP.layout();
    Tensor tPrP = make_tensor(tOrP.data(), make_layout(get<0>(l), make_layout(get<1>(l), get<2>(l))));
    CUTE_STATIC_ASSERT_V(size<2>(tPgP) == _1{});
    CUTE_STATIC_ASSERT_V(size<1>(tPrP) == size<1>(tPgP));
    #pragma unroll
    for (int mi = 0; mi < size<1>(tPrP); ++mi) {
        cute::copy(gmem_tiled_copy_P, tPrP(_, mi), tPgP(_, mi, 0));
    }
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_first, bool Is_last, bool Is_attn_mask, bool Seq_parallel=false, typename Params>
inline __device__ void reduce_attn_scores_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {
    const bool Is_sparse_attn_mask = params.attn_mask_start_row_indices_ptr != nullptr;
    const int attn_mask_start_row = params.attn_mask_start_row;

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    __shared__ int32_t sparse_mask_smem_[Kernel_traits::kBlockN];
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    // constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{}))::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k || binfo.actual_seqlen_q == 0) return;

    // umiswing: residue is for predication of additional mask gmem access.
    // Additional mask for varlen qkv is supported, but a varlen mask is not supported.
    const int m_residue = params.seqlen_q % kBlockM ? params.seqlen_q % kBlockM : kBlockM;
    const int n_residue = params.seqlen_k % kBlockN ? params.seqlen_k % kBlockN : kBlockN;

    const int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);
    const int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q
        + (m_block_max - 1) * kBlockM;

    const uint64_t row_offset_mask = (uint64_t)((bidb * params.mask_head_mod_size
        + (bidh % params.mask_head_mod_size)) * params.mask_seq_q_mod_size
        + ((m_block_max - 1) * kBlockM % params.mask_seq_q_mod_size)) * params.seqlen_k
        + n_block * kBlockN;

    const index_t row_offset_sparse_mask = (bidb * params.mask_head_mod_size + bidh % params.mask_head_mod_size) * params.seqlen_k + n_block * kBlockN;

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded)
        * params.seqlen_k_rounded;

    // (b,n,1,s_k)
    const index_t offset_reduced_scores = (bidb * params.h + bidh) * params.seqlen_k_rounded;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gMask = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.attn_mask_ptr) + row_offset_mask),
                               Shape<Int<kBlockM>, Int<kBlockN>>{},
                               make_stride(params.seqlen_k, _1{}));
    Tensor gSparseMask = make_tensor(make_gmem_ptr(reinterpret_cast<int32_t *>(params.attn_mask_start_row_indices_ptr) + row_offset_sparse_mask),
                               Shape<Int<kBlockN>>{});

    // umiswing: should it be 16-bit?
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<float *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            // umiswing: i don't think it should be rounded.
                            make_stride(params.seqlen_k, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});

    // umiswing: I'm not sure should I keep double buffer?
    Tensor sK = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sP = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutPdS{});
    // sP and sdQ share the same memory so be careful
    Tensor sSparseMask = make_tensor(make_smem_ptr(reinterpret_cast<int32_t *>(sparse_mask_smem_)), Shape<Int<kBlockN>>{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto gmem_thr_copy_P = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp).get_thread_slice(tidx);
    Tensor tPgMask = gmem_thr_copy_P.partition_D(gMask);
    Tensor cMask = make_identity_tensor(shape(gMask));
    Tensor tPcMask = gmem_thr_copy_P.partition_D(cMask);

    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);

    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);

    // Partition sP and sdS to match the accumulator partitioning
    // This has to be tiled_mma_sdp, not tiled_mma_dkv
    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    int m_block = m_block_max - 1;
    int m_block_min = !Is_causal ? 0 : (n_block * kBlockN) / kBlockM;

    // umiswing: handle with the early exit case, write 0 to the reduced attn scores?
    // We might need to exit early and write 0 to dK and dV.
    // Otherwise we get wrong result for the case where we don't enter the for loop.
    // And we might read OOB elements from gQ and gdO.
    // TODO: what if we're not parallelizing, do we need to compute dot_do_o?
    if (Is_causal && m_block < m_block_min) {
        return;
    }

#if 0
umiswing: comment out the fucking double buffer.
    if (Double_buffer && m_block % 2 == 1) {  // Double buffer for sQ
        tQsQ.data() = tQsQ.data() + size(sQ);
        tSsQ.data() = tSsQ.data() + size(sQ);
    }
#endif

    if (!Is_first && !Seq_parallel) { __syncthreads(); }


    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, binfo.actual_seqlen_q - m_block * kBlockM
    );


    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((2, 2), MMA_N, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        // Using uint32_t row makes it 10us slower on d=128, not sure why.
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : 0;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::cp_async_fence();

    if (Is_sparse_attn_mask) {
        if (tidx < kBlockN) {
	        sSparseMask(tidx) = gSparseMask(tidx);
        }
	    __syncthreads();
    }

    auto atomMNK = typename decltype(tiled_mma_sdp)::AtomShape_MNK{};
    auto thrVMNK = typename decltype(tiled_mma_sdp)::ThrLayoutVMNK{};
    auto shape_MN = Shape<Int<kBlockM>, Int<kBlockN>>{};

    auto MMA_N = shape_div(size<1>(shape_MN), size<1>(atomMNK) * size<2>(thrVMNK));

    Tensor local_reduced_scores = make_tensor<float>(Shape<Int<2>, Int<MMA_N>>{}); // (2, MMA_N)

    for (; m_block >= m_block_min; --m_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);
        // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
        // umiswing: I think it should be Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (row=(2, MMA_M), col=(2, MMA_N))? Just check gemm() in utils.h
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        // Compute the exponential value.
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        // umiswing: add a fucking assert.
        for(int n=0; n < size<1>(scores); ++n) {
          for(int m=0;m<size<0>(scores);++m) {
            local_reduced_scores(n) += scores(m,n);
          }
        }

if(cute::thread0()) {
  p(kBlockM);
  p(kBlockN);
  p(MMA_N);
  p(acc_s);
  p(scores);
  p(local_reduced_scores);
  p(tiled_mma_sdp);
}

// umiswing: write attn scores out for debug
if (/*Return_softmax=*/true) {
    flash::write_attn_scores(scores,
                             reinterpret_cast<Element *>(params.p_ptr) + row_offset_p,
                             n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                             binfo.actual_seqlen_k, m_block * kBlockM + get<0>(taccScS_row(0)),
                             AtomLayoutMS * 16);
}
#if 0
umiswing: I guess we don't need this.
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);
#endif
        // if (cute::thread0()) { print(tPaP); }
        // __syncthreads();
        // if (cute::thread0()) { print(sP); }

        if (Double_buffer && m_block > m_block_min) {  // Double buffer for sQ
// umiswing: this is a stupid fake double buffer, just for debugging...
            __syncthreads();
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            flash::cp_async_fence();
        }

#if 0
umiswing: we only have one gemm now, should we keep double buffer?
        if (Double_buffer && m_block > m_block_min) {
            // Double buffer for sQ
            const int sQ_offset = m_block % 2 == 0 ? size(sQ) : -size(sQ);
            tQsQ.data() = tQsQ.data() + sQ_offset;
            tSsQ.data() = tSsQ.data() + sQ_offset;
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            flash::cp_async_fence();
        }
#endif

#if 0
umiswing: do we need to keep this sync?
        __syncthreads(); // Need syncthreads since we're writing to the same sdO location
#endif

        if (m_block > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
        }

        if (!Is_last) {
        } else {
        }

#if 0
umiswing: do we need to keep double buffer?
        if (Double_buffer) {  // Double buffer for sQ
        }
#endif
#if 0
// umiswing: actually we don't need this branch right now.
        if (!Double_buffer && m_block > m_block_min) {
            __syncthreads();
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            flash::cp_async_fence();
        }
#endif

#if 0
umiswing: still, I don't think we need this shits.
        if (Is_last) {
            __syncthreads();
        }
#endif

    }

    write_reduced_scores(local_reduced_scores,
                         reinterpret_cast<float *>(params.reduced_scores) + offset_reduced_scores,
                         n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                         binfo.actual_seqlen_k);

    // Epilogue

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    // If Is_last, there's already a __syncthreads() at the end of the loop.
    if (!Is_last) { __syncthreads(); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_attn_mask, bool Is_deterministic>
// inline __device__ void reduce_attn_scores_seqk_parallel(const Reduce_attn_scores_params &params) {
__global__ void reduce_attn_scores_seqk_parallel(const Reduce_attn_scores_params params) {
    const int n_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    if (Is_deterministic) {  // params.num_splits == 1, means grid.x = 1, blockIdx.x = 0;
        int loop_step_x = 0;
        for(int i = 0; i < params.seqlen_k; i+= kBlockN) {
           reduce_attn_scores_1colblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, false, false, Is_attn_mask, /*Seq_parallel=*/true>(params, bidb, bidh, loop_step_x);
           loop_step_x += 1;
        }
    } else {
        reduce_attn_scores_1colblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, false, false, Is_attn_mask, /*Seq_parallel=*/true>(params, bidb, bidh, n_block);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace flash
