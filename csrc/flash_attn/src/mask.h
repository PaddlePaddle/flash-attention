#include <cuda_runtime.h>
template <int N>
__global__ void scanMaxMinKernel(
    const int *input, int b, int n, int *maxo, int *mino) {
  int bid = threadIdx.y + blockIdx.y * blockDim.y;
  if (bid >= b) {
    return;
  }
  int i_offset = bid * n;
  const int o_n = (n + N - 1) / N;
  input = input + i_offset;
  constexpr int nums = N / 32;  // ensure N % 32 == 0
  int warpId = blockIdx.x;      // ensure blockDim.x == 32
  int tid = threadIdx.x;
  int maxv, minv;
  int idx = warpId * N + tid;
  if (warpId * N + N > n) {
    maxv = 0;
    minv = INT_MAX;
#pragma unroll
    for (int i = 0; i < nums; i++) {
      if (idx < n) {
        maxv = max(maxv, input[idx]);
        minv = min(minv, input[idx]);
      }
      idx += 32;
    }
  } else {
    maxv = input[idx];
    minv = input[idx];
#pragma unroll
    for (int i = 1; i < nums; i++) {
      idx += 32;
      maxv = max(maxv, input[idx]);
      minv = min(minv, input[idx]);
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

template <int N>
// requires N % 32 == 0
void scanMaxMinGpu(
    const int *input, int b, int n, int *maxo, int *mino, cudaStream_t stream) {
  static_assert(N % 32 == 0, "N must be a multiple of 32");
  dim3 block(32, 4);
  dim3 grid((n + N - 1) / N, (b + 3) / 4);
  scanMaxMinKernel<N><<<grid, block, 0, stream>>>(input, b, n, maxo, mino);
}

template <typename Kernel_traits>
void prepare_sparsemask(Flash_fwd_params &params, cudaStream_t stream) {
  if (!params.enable_mask_bypass) {
    return;
  }
  if (params.attn_mask_start_row_indices_ptr == nullptr &&
      params.attn_mask_end_row_indices_ptr == nullptr) {
    params.attn_sparsemask_down_nblockmax = nullptr;
    params.attn_sparsemask_down_nblockmin = nullptr;
    params.attn_sparsemask_up_nblockmax = nullptr;
    params.attn_sparsemask_up_nblockmin = nullptr;
    return;
  }
  int *nblock_smask = params.flashmask_maxmin_ptr;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  const int nblock_seqlen = (params.seqlen_k + kBlockN - 1) / kBlockN;
  const int nblock_masklen = params.b * params.h_sparsemask * nblock_seqlen;
  params.attn_sparsemask_down_nblockmax = nblock_smask;
  params.attn_sparsemask_down_nblockmin = nblock_smask + nblock_masklen;
  params.attn_sparsemask_up_nblockmax = nblock_smask + 2 * nblock_masklen;
  params.attn_sparsemask_up_nblockmin = nblock_smask + 3 * nblock_masklen;
  if (params.attn_mask_start_row_indices_ptr != nullptr) {
    scanMaxMinGpu<kBlockN>(
        static_cast<const int *>(params.attn_mask_start_row_indices_ptr),
        params.b * params.h_sparsemask,
        params.seqlen_k,
        params.attn_sparsemask_down_nblockmax,
        params.attn_sparsemask_down_nblockmin,
        stream);
  } else {
    params.attn_sparsemask_down_nblockmax = nullptr;
    params.attn_sparsemask_down_nblockmin = nullptr;
  }
  if (params.attn_mask_end_row_indices_ptr != nullptr) {
    scanMaxMinGpu<kBlockN>(
        static_cast<const int *>(params.attn_mask_end_row_indices_ptr),
        params.b * params.h_sparsemask,
        params.seqlen_k,
        params.attn_sparsemask_up_nblockmax,
        params.attn_sparsemask_up_nblockmin,
        stream);
  } else {
    params.attn_sparsemask_up_nblockmax = nullptr;
    params.attn_sparsemask_up_nblockmin = nullptr;
  }
}