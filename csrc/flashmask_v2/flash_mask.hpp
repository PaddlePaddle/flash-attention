#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "utils.h"

namespace flash {

  template <typename TiledMma,int kBlockM, int kBlockN,bool SwapAB, typename Engine, typename Layout>
  // CUTLASS_DEVICE
  __device__
  void apply_flashmask_bwd(Tensor<Engine, Layout> &tSrS, int const thread_idx, int32_t const * flashmask_index_smem_, const int32_t m_block) {

      // static_assert(!PackGQA);
      // static_assert(!SwapAB);

      auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);

      Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
      Tensor tScS = thread_mma.partition_C(cS);
      Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
      Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));

      static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;
      const int32_t * s_lt_start = flashmask_index_smem_;
      const int32_t * s_lt_end = flashmask_index_smem_ + kBlockN;
      const int32_t * s_ut_start = flashmask_index_smem_ + 2 * kBlockN;
      const int32_t * s_ut_end = flashmask_index_smem_ + 3 * kBlockN;

      #pragma unroll
      for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
        int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
        // __syncwarp();
        // printf("\n>>>>>> wsm debug row_idx:%d, thread_idx:%d\n", row_idx, thread_idx);
        // __syncwarp();
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          int const col_idx = get<Col>(tScS_rowcol(m, n)); // col_idx within a block
          if(row_idx >= s_lt_start[col_idx] && row_idx < s_lt_end[col_idx]){
              tSrS_rowcol(m, n) = -INFINITY;
              // printf("point1, row_idx:%d, col_idx:%d, thread_idx:%d, tSrS_rowcol(m, n):%f\n", row_idx, col_idx, thread_idx, tSrS_rowcol(m, n));
            }
          if(row_idx >= s_ut_start[col_idx] && row_idx < s_ut_end[col_idx]){
              tSrS_rowcol(m, n) = -INFINITY;
            //  printf("point2, row_idx:%d, col_idx:%d, thread_idx:%d, tSrS_rowcol(m, n):%f\n", row_idx, col_idx, thread_idx, tSrS_rowcol(m, n));
          }
          // printf("\n>>>>>> wsm debug s_ltstart_col :%d, s_ltend_col:%d,s_utstart_col:%d, s_utend_col:%d, col_idx: %d\n", s_lt_start[col_idx], s_lt_end[col_idx], s_ut_start[col_idx], s_ut_end[col_idx], col_idx);
          // printf("\n>>>>>> wsm debug row_idx:%d, col_idx:%d,m:%d,n:%d, thread_idx:%d, tSrS_rowcol(m, n):%f\n", row_idx, col_idx,m,n, thread_idx, tSrS_rowcol(m, n));
        }
      }
    }

// };

namespace flashmask {

  template<int kBlockN>
  __global__
  void scanMaxMinKernel(
      const int *input, int b, int n, int *maxo, int *mino) {
    int bid = threadIdx.y + blockIdx.y * blockDim.y;
    if (bid >= b) {
      return;
    }
    int i_offset = bid * n;
    input = input + i_offset;
    const int o_n = ((n + kBlockN - 1) / kBlockN + 3) /4 * 4;
    // constexpr int nums = kBlockN / 32;  // ensure N % 32 == 0
    constexpr int nums = (kBlockN + 31) / 32;
    int warpId = blockIdx.x;      // ensure blockDim.x == 32
    int tid = threadIdx.x;
    int lane_id = threadIdx.x % 32;
    int maxv, minv;
    int idx = warpId * kBlockN + tid;
    if (warpId * kBlockN + kBlockN > n) {
      maxv = 0;
      minv = INT_MAX;
  #pragma unroll
      for (int i = 0; i < nums; i++) {
        if (idx < n && lane_id + i * 32 < kBlockN) {
          maxv = max(maxv, input[idx]);
          minv = min(minv, input[idx]);
        }
        idx += 32;
      }
    } else {
      maxv = 0;
      minv = INT_MAX;
  #pragma unroll
      for (int i = 0; i < nums; i++) {
        if(lane_id + i * 32 < kBlockN) {
          maxv = max(maxv, input[idx]);
          minv = min(minv, input[idx]);
          idx += 32;
        }
      }
    }
    __syncwarp();
    maxv = __reduce_max_sync(0xffffffff, maxv);
    minv = __reduce_min_sync(0xffffffff, minv);
    if (tid == 0) {
      maxo[bid * o_n + warpId] = maxv;
      mino[bid * o_n + warpId] = minv;
    }
  }

  template <int kBlockN>
  void scanMaxMinGpu(
      const int *input, int b, int n, int *maxo, int *mino, cudaStream_t stream) {
    // static_assert(kBlockN % 32 == 0, "kBlockN must be a multiple of 32");
    dim3 block(32, 4);
    dim3 grid((n + kBlockN - 1) / kBlockN, (b + 3) / 4);
    scanMaxMinKernel<kBlockN><<<grid, block, 0, stream>>>(input, b, n, maxo, mino);
  }

  template <int kBlockN>
  void prepare_block_maxmin(Flash_fwd_params &params, cudaStream_t stream) {
    if (params.lt_start_ptr == nullptr &&
        params.ut_end_ptr == nullptr) {
      return;
    }
    int *nblock_smask = params.flashmask_maxmin_ptr;

    const int nblock_seqlen = ((params.seqlen_k + kBlockN - 1) / kBlockN + 3) / 4 * 4; // umiswing: padding for int4 load
    const int nblock_masklen = params.b * params.h_flashmask * nblock_seqlen;

    params.lt_start_nblockmax = nblock_smask;
    params.lt_start_nblockmin = nblock_smask + nblock_masklen;
    params.ut_end_nblockmax = nblock_smask + 2 * nblock_masklen;
    params.ut_end_nblockmin = nblock_smask + 3 * nblock_masklen;
    params.lt_end_nblockmax = nblock_smask + 4 * nblock_masklen;
    params.lt_end_nblockmin = nblock_smask + 5 * nblock_masklen;
    params.ut_start_nblockmax = nblock_smask + 6 * nblock_masklen;
    params.ut_start_nblockmin = nblock_smask + 7 * nblock_masklen;
    if (params.lt_start_ptr != nullptr) {
      scanMaxMinGpu<kBlockN>(
          params.lt_start_ptr,
          params.b * params.h_flashmask,
          params.seqlen_k,
          params.lt_start_nblockmax,
          params.lt_start_nblockmin,
          stream);
    } else {
      params.lt_start_nblockmax = nullptr;
      params.lt_start_nblockmin = nullptr;
    }
    if (params.ut_end_ptr != nullptr) {
      scanMaxMinGpu<kBlockN>(
                    params.ut_end_ptr,
                    params.b * params.h_flashmask,
                    params.seqlen_k,
                    params.ut_end_nblockmax,
                    params.ut_end_nblockmin,
                    stream);
    } else {
      params.ut_end_nblockmax = nullptr;
      params.ut_end_nblockmin = nullptr;
    }
    if (params.lt_end_ptr != nullptr) {
      scanMaxMinGpu<kBlockN>(
          params.lt_end_ptr,
          params.b * params.h_flashmask,
          params.seqlen_k,
          params.lt_end_nblockmax,
          params.lt_end_nblockmin,
          stream);
    } else {
      params.lt_end_nblockmax = nullptr;
      params.lt_end_nblockmin = nullptr;
    }
    if (params.ut_start_ptr != nullptr) {
      scanMaxMinGpu<kBlockN>(
          params.ut_start_ptr,
          params.b * params.h_flashmask,
          params.seqlen_k,
          params.ut_start_nblockmax,
          params.ut_start_nblockmin,
          stream);
    } else {
      params.ut_start_nblockmax = nullptr;
      params.ut_start_nblockmin = nullptr;
    }
  }
} // namespace flashmask
} // namespace flash
