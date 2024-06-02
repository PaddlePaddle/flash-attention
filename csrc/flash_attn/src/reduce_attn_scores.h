#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
namespace flash {

using namespace cute;

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

template <typename Engine, typename Layout>
inline __device__ void write_attn_scores(Tensor<Engine, Layout> &tensor,
                                         float * const gScores_ptr,
                                         const uint32_t max_seqlen_q,
                                         const uint32_t max_seqlen_k,
                                         const uint32_t row_idx_offset_,
                                         const uint32_t col_idx_offset_,
                                         const uint32_t warp_row_stride) {
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const uint32_t lane_id = threadIdx.x % 32;
    // const uint32_t row_idx_offset = row_idx_offset_ + lane_id / 4;
    const uint32_t row_idx_offset = row_idx_offset_;
    const uint32_t col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    const uint32_t row_idx_limit = max_seqlen_q;
    const uint32_t col_idx_limit = max_seqlen_k;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const uint32_t row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const uint32_t row_idx = row_idx_base + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const uint32_t col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const uint32_t col_idx = col_idx_base + j;
                    if (row_idx < row_idx_limit && col_idx < col_idx_limit) {
                        *(gScores_ptr + col_idx + row_idx*max_seqlen_k) = tensor(make_coord(i, mi), make_coord(j, nj));
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Engine, typename Layout, typename T>
inline __device__ void write_reduced_scores(Tensor<Engine, Layout> &rScores,
                                            T * const gScores_ptr,
                                            const uint32_t col_idx_offset_,
                                            const uint32_t max_seqlen_k) {
    // rScores has shape (2, MMA_M) umiswing: or just 2*MMA_M?
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;

    const uint32_t col_idx_limit = max_seqlen_k;
    #pragma unroll
    for (int nj = 0; nj < size<1>(rScores); ++nj) {
        const uint32_t col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<0>(rScores); ++j) {
            const uint32_t col_idx = col_idx_base + j;
            if (col_idx < col_idx_limit) {
                atomicAdd(gScores_ptr+col_idx, rScores(j,nj));
                // *(gScores_ptr+col_idx) = col_idx;
            }
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void reduce_attn_scores_1colblock(const Params &params, const int bidb, const int bidh, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    // constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int MMA_N_SdP = kBlockN / decltype(size<1>(typename Kernel_traits::TiledMmaSdP::TiledShape_MNK{}))::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (n_block * kBlockN >= binfo.actual_seqlen_k || binfo.actual_seqlen_q == 0) return;

    const int m_block_max = cute::ceil_div(binfo.actual_seqlen_q, kBlockM);

    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + (m_block_max - 1) * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + n_block * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q
        + (m_block_max - 1) * kBlockM;

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q)
        * params.seqlen_k;

    // (b,n,1,s_k)
    const index_t offset_reduced_scores = (bidb * params.h + bidh) * params.seqlen_k;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});

    Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;

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
    int m_block_min = 0;

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
        lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
    }

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
    );
    flash::cp_async_fence();

    auto atomMNK = typename decltype(tiled_mma_sdp)::AtomShape_MNK{};
    auto thrVMNK = typename decltype(tiled_mma_sdp)::ThrLayoutVMNK{};
    auto shape_MN = Shape<Int<kBlockM>, Int<kBlockN>>{};

    auto MMA_N = shape_div(size<1>(shape_MN), size<1>(atomMNK) * size<2>(thrVMNK));

    Tensor local_reduced_scores = make_tensor<float>(Shape<Int<2>, Int<MMA_N>>{}); // (2, MMA_N)
    cute::clear(local_reduced_scores);

    for (; m_block >= m_block_min; --m_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);
        // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (col=(2, MMA_N), row=(2, MMA_N))
        // umiswing: I think it should be Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (row=(2, MMA_M), col=(2, MMA_N)). Just check gemm() in utils.h
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        // Compute the exponential value.
        flash::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        if (Return_softmax) {
            flash::write_attn_scores(scores,
                                     reinterpret_cast<float *>(params.p_ptr) + row_offset_p,
                                     binfo.actual_seqlen_q,
                                     binfo.actual_seqlen_k,
                                     m_block * kBlockM + get<0>(taccScS_row(0)),
                                     n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                                     AtomLayoutMS * 16);
        }

        CUTE_STATIC_ASSERT_V(size(local_reduced_scores) == size<1>(scores));
        static_assert(decltype(size<0>(local_reduced_scores))::value == decltype(size<1,0>(scores))::value);
        static_assert(decltype(size<1>(local_reduced_scores))::value == decltype(size<1,1>(scores))::value);
        #pragma unroll
        for(int n=0; n < size<1>(scores); ++n) {
          #pragma unroll
          for(int m=0;m<size<0>(scores);++m) {
            local_reduced_scores(n) += scores(m,n);
          }
        }

        if (m_block > m_block_min) {
            // Advance gQ
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            flash::cp_async_fence();

            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
        }
    }

    write_reduced_scores(local_reduced_scores,
                         reinterpret_cast<float *>(params.reduced_scores) + offset_reduced_scores,
                         n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                         binfo.actual_seqlen_k);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Return_softmax>
__global__ void reduce_attn_scores_seqk_parallel(const Reduce_attn_scores_params params) {
    const int n_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    reduce_attn_scores_1colblock<Kernel_traits, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, n_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace flash
