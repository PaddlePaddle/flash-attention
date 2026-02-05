#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "utils.h"

namespace flash {

  template <typename TiledMma, int kBlockM, int kBlockN, bool SwapAB, bool Has_ut_start, bool Is_causal, typename Engine, typename Layout>
  // CUTLASS_DEVICE
  __device__
  void apply_flashmask_bwd(Tensor<Engine, Layout> &tSrS, int const thread_idx, const int32_t* const __restrict__ flashmask_index_smem_, const int32_t m_block) {

      // static_assert(!PackGQA);
      // static_assert(!SwapAB);

      const auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);

      const Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
      const Tensor tScS = thread_mma.partition_C(cS);
      Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
      const Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));

      static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;
      const int32_t* const s_lt_start = flashmask_index_smem_;
      const int32_t* const s_lt_end = flashmask_index_smem_ + kBlockN;
      const int32_t* const s_ut_start = flashmask_index_smem_ + 2 * kBlockN;
      const int32_t* const s_ut_end = flashmask_index_smem_ + 3 * kBlockN;

      #pragma unroll
      for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
        int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
        // __syncwarp();
        // printf("\n>>>>>> wsm debug row_idx:%d, thread_idx:%d\n", row_idx, thread_idx);
        // __syncwarp();
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          int const col_idx = get<Col>(tScS_rowcol(m, n)); // col_idx within a block

          // Note(heqianyue): causal masking will be processed in generic fa-v3 `mask.apply`, so if causal, there is no need to apply mask again
          if constexpr (Is_causal) {
            // Note(heqianyue): if Has_lt_end == false, row_idx < s_lt_end[col_idx] is entirely unnecessary, but if we just
            // throw it away, for sliding window and document mask, we might have about 3% performance loss
            // due to if both predicates are present, some of the FSEL instructions are selectively performed
            // instead of performed unconditionally. Through removing the latter predicate can save a lot of
            // instructions (193 --> 99), we will actually store more / use more regs. This is basically a 
            // trade-off for speed and no performance recession
            if (row_idx >= s_lt_start[col_idx] && row_idx < s_lt_end[col_idx])
                tSrS_rowcol(m, n) = -INFINITY;
          } else {
            if constexpr (Has_ut_start) {
              // Note(heqianyue): currently, if we have ut_start, we will definitely have lt_end
              // but if we have a new mask type other than global swin, the constraint might be violated
              if (row_idx >= s_lt_start[col_idx] && row_idx < s_lt_end[col_idx])
                  tSrS_rowcol(m, n) = -INFINITY;
              if (row_idx >= s_ut_start[col_idx] && row_idx < s_ut_end[col_idx])
                  tSrS_rowcol(m, n) = -INFINITY;
            } else {
              // Note(heqianyue): we don't have lt_start, lt_end, nullptr and ut_end composition, maybe in the future
              if (row_idx >= s_lt_start[col_idx])
                  tSrS_rowcol(m, n) = -INFINITY;
              if (row_idx < s_ut_end[col_idx])
                  tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      }
  }

// };

namespace flashmask {

  // make sure the following value is the same with the CooperativeMainLoopImpl
  // for example, sm90 is 16 * 1024.
  static constexpr int flashmask_buffer_length = 16 * 1024;

  // Note(heqianyue): this kernel is currently only used for fwd and sm90 (flashmask v3)
  // for fully aligned minmax with no excessive global sector
  template<int kBlockN, bool aligned_chunk = false>
  __global__
  void scanMaxMinChunkedKernel(
      const int *input, int b, int n, int *maxo, int *mino) {
    int bid = threadIdx.y + blockIdx.y * blockDim.y;
    if (bid >= b) {
      return;
    }
    int i_offset = bid * n;
    input = input + i_offset;

    const int nblock_seqlen = ((n + kBlockN - 1) / kBlockN + 3) & 0xfffffffc;  // umiswing: padding for int4 load

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
      if constexpr (aligned_chunk) {
        // the length of the buffer that actually takes part in computation
        constexpr int chunk_valid_length = ((flashmask_buffer_length + kBlockN - 1) / kBlockN + 3) & 0xfffffffc;
        // the padded length for the sake of 128B aligned reading sector (ceil to multiple of 32)
        constexpr int chunk_padded_length = ((flashmask_buffer_length + kBlockN - 1) / kBlockN + 31) & 0xffffffe0;

        const int num_chunk = (nblock_seqlen + chunk_valid_length - 1) / chunk_valid_length;
        const int total_length = num_chunk * chunk_padded_length;
        // TODO(heqianyue): This can be made faster by fast div mod, but I suppose this will not be a bottleneck
        const int chunk_id = warpId / chunk_valid_length;
        const int within_chunk_id = warpId % chunk_valid_length;

        // stores chunk (there will be 'padding -- invalid data' on the tail) continuously
        maxo[bid * total_length + chunk_padded_length * chunk_id + within_chunk_id] = maxv;
        mino[bid * total_length + chunk_padded_length * chunk_id + within_chunk_id] = minv;
      } else {
        // stores data continuously
        maxo[bid * nblock_seqlen + warpId] = maxv;
        mino[bid * nblock_seqlen + warpId] = minv;
      }
    }
  }

  template <int kBlockN>
  void scanMaxMinGpu(
      const int *input, int b, int n, int *maxo, int *mino, cudaStream_t stream, bool use_aligned_chunk = false) {
    // static_assert(kBlockN % 32 == 0, "kBlockN must be a multiple of 32");
    dim3 block(32, 4);
    dim3 grid((n + kBlockN - 1) / kBlockN, (b + 3) / 4);
    if (use_aligned_chunk)
      scanMaxMinChunkedKernel<kBlockN, true><<<grid, block, 0, stream>>>(input, b, n, maxo, mino);
    else
      scanMaxMinChunkedKernel<kBlockN, false><<<grid, block, 0, stream>>>(input, b, n, maxo, mino);
  }

  template <int kBlockN>
  void prepare_block_maxmin(Flash_fwd_params &params, cudaStream_t stream, bool is_forward = false) {
    if (params.lt_start_ptr == nullptr &&
        params.ut_end_ptr == nullptr) {
      return;
    }
    int *nblock_smask = params.flashmask_maxmin_ptr;

    // only used in forward pass and SM90 (FlashMaskV3)
    const int nblock_seqlen = ((params.seqlen_k + kBlockN - 1) / kBlockN + 3) & 0xfffffffc; // umiswing: padding for int4 load
    int nblock_masklen = 0;

    const bool use_aligned_chunk = params.arch == 90 && is_forward; 

    if (use_aligned_chunk) {
      constexpr int chunk_valid_length = ((flashmask_buffer_length + kBlockN - 1) / kBlockN + 3) & 0xfffffffc;
      constexpr int chunk_padded_length = ((flashmask_buffer_length + kBlockN - 1) / kBlockN + 31) & 0xffffffe0;
      const int num_chunk = (nblock_seqlen + chunk_valid_length - 1) / chunk_valid_length;
      nblock_masklen = params.b * params.h_flashmask * num_chunk * chunk_padded_length;
    } else {
      nblock_masklen = params.b * params.h_flashmask * nblock_seqlen;
    }

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
          stream,
          use_aligned_chunk);
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
                    stream,
                    use_aligned_chunk);
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
          stream,
          use_aligned_chunk);
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
          stream,
          use_aligned_chunk);
    } else {
      params.ut_start_nblockmax = nullptr;
      params.ut_start_nblockmin = nullptr;
    }
  }
} // namespace flashmask
} // namespace flash
