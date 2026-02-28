#include "paddle/extension.h"

#define CHECK_CUDA_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

int get_kBlockN(int head_size_rounded, bool is_flashmask, bool is_causal, bool has_softcap,
                bool is_local, int seqlen_q, int seqlen_k, bool has_lt_end, bool has_ut_start) {
    if (head_size_rounded <= 64) {
        if (is_flashmask && !is_causal) {
            return 96;
        } else if ((is_causal && has_softcap) || is_flashmask) {
            return 128;
        } else {
            return 128;
        }
    } else if (head_size_rounded <= 128) {
        if (is_causal || is_local || has_softcap) {
            return 128;
        } else {
            if ((seqlen_q >= 1024 || seqlen_k >= 1024) && !(has_lt_end && has_ut_start)) {
                return 128;
            } else {
                return 64;
            }
        }
    } else if (head_size_rounded <= 256) {
        if (has_lt_end && has_ut_start) {
            return 32;
        } else {
            return 64;
        }
    } else {
        // 不支持的情况
        throw std::runtime_error("head_size_rounded not supported");
    }
}

template<int kBlockN>
__global__
void scanMaxMinChunkedKernel(
    const int *input, int b, int n, int *maxo, int *mino) {
    int bid = threadIdx.y + blockIdx.y * blockDim.y;
    if (bid >= b) return;
    int i_offset = bid * n;
    input = input + i_offset;
    const int nblock_seqlen = ((n + kBlockN - 1) / kBlockN + 3) & 0xfffffffc;
    constexpr int nums = (kBlockN + 31) / 32;
    int warpId = blockIdx.x;
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
        maxo[bid * nblock_seqlen + warpId] = maxv;
        mino[bid * nblock_seqlen + warpId] = minv;
    }
}

// Enum for pointer dispatching in reduce_workload_kernel
enum PtrDispatch { SINGLE_PTR = 1, DUAL_PTR = 2, FULL_PTR = 4 };

template<int kBlockM, int PTR_DISPATCH_TAG, bool is_causal>
__global__ void reduce_workload_kernel(
    const int* LTStartMax, const int* LTStartMin,
    const int* LTEndMax, const int* LTEndMin,
    const int* UTStartMax, const int* UTStartMin,
    const int* UTEndMax, const int* UTEndMin,
    int* workload, // [B, H, Tr, 1]
    int BH, int Tr, int Tc, int S,
    int Br // m_block_size
) {
    int bh = blockIdx.y;
    int tr = blockIdx.x;
    int tc = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    if(tr >= Tr) return;

    int wl = 0;
    bool fully_masked = true;
    bool partially_masked = false;
    int lt_start_max = INT_MAX;
    int lt_start_min = INT_MAX;
    int lt_end_max = INT_MAX;
    int lt_end_min = INT_MAX;
    int ut_start_max = INT_MIN;
    int ut_start_min = INT_MIN;
    int ut_end_max = INT_MIN;
    int ut_end_min = INT_MIN;

    __shared__ int smem[32];

    const int idx = bh * Tc + tc;
    const int q_idx = bh * Tr + tr;

    const int m_block_s = q_idx * kBlockM;
    const int m_block_e = m_block_s + kBlockM < S ? m_block_s + kBlockM : S;

    lt_start_max = tc < Tc ? LTStartMax[idx] : INT_MAX;
    lt_start_min = tc < Tc ? LTStartMin[idx] : INT_MAX;

    // 分支展开
    if constexpr (PTR_DISPATCH_TAG == FULL_PTR) {
        lt_end_max = tc < Tc ? LTEndMax[idx] : INT_MAX;
        lt_end_min = tc < Tc ? LTEndMin[idx] : INT_MAX;
        ut_start_max = tc < Tc ? UTStartMax[idx] : INT_MIN;
        ut_start_min = tc < Tc ? UTStartMin[idx] : INT_MIN;
        ut_end_max = tc < Tc ? UTEndMax[idx] : INT_MIN;
        ut_end_min = tc < Tc ? UTEndMin[idx] : INT_MIN;

        fully_masked = (m_block_s >= lt_start_max && m_block_e <= lt_end_min) ||
                       (m_block_s >= ut_start_max && m_block_e <= ut_end_min);
        partially_masked = (m_block_s < lt_end_max && m_block_e > lt_start_min) ||
                           (m_block_s < ut_end_max && m_block_e > ut_start_min);
    }
    else if constexpr (PTR_DISPATCH_TAG == DUAL_PTR) {
        if constexpr (is_causal) {
            lt_end_max = tc < Tc ? LTEndMax[idx] : INT_MAX;
            lt_end_min = tc < Tc ? LTEndMin[idx] : INT_MAX;
            fully_masked = m_block_s >= lt_start_max && m_block_e <= lt_end_min;
            partially_masked = m_block_s < lt_end_max && m_block_e > lt_start_min;
        } else {
            ut_end_max = tc < Tc ? UTEndMax[idx] : INT_MIN;
            ut_end_min = tc < Tc ? UTEndMin[idx] : INT_MIN;
            fully_masked = (m_block_s >= lt_start_max) || (m_block_e <= ut_end_min);
            partially_masked = (m_block_e > lt_start_min) || (m_block_s < ut_end_max);
        }
    }
    else if constexpr (PTR_DISPATCH_TAG == SINGLE_PTR) {
        fully_masked = m_block_s >= lt_start_max;
        partially_masked = m_block_e > lt_start_min;
    }

    if(tc >= Tc){
        fully_masked = true;
        partially_masked = false;
    }
    wl = fully_masked ? 0 : 1;

    unsigned mask = 0xffffffff;
    // warp reduce sum
    int wl_sum = wl;
    for (int offset = 16; offset > 0; offset >>= 1) {
        wl_sum += __shfl_down_sync(mask, wl_sum, offset);
    }
    if (laneId == 0) {
        smem[warpId] = wl_sum;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        int val = (threadIdx.x < (blockDim.x + 31)/32) ? smem[threadIdx.x] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x == 0) {
            workload[q_idx] = val;
        }
    }
}

__global__ void indices_to_chunks_kernel(
    const int* startend_row_indices,
    const int* chunk_bucket_indices,
    int* chunked_result,
    int num_rows,
    int num_buckets,
    int chunk_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int max_chunk_index = 0;
    int row_val = startend_row_indices[row];

    for (int bucket = 0; bucket < num_buckets; ++bucket) {
        int bucket_idx = chunk_bucket_indices[bucket];
        int chunk_start = bucket_idx * chunk_size;
        int local_index = row_val - chunk_start;
        local_index = max(local_index, 0);
        local_index = min(local_index, chunk_size);

        if (local_index > 0) {
            local_index += bucket * chunk_size;
        }

        if (bucket == 0 || local_index > max_chunk_index) {
            max_chunk_index = local_index;
        }
    }
    chunked_result[row] = max_chunk_index;
}

__global__ void indices_rerank_kernel(
    const int* startend_row_indices,
    int* output_reranked_indices,
    const int* chunk_indices,
    int batch_size,
    int num_heads,
    int seq_len,
    int feature_dim,
    int num_chunks,
    int chunk_size
) {
    int output_seq_len = num_chunks * chunk_size;
    int total_elements = batch_size * output_seq_len * num_heads * feature_dim;
    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total_elements) return;

    int d = flat_idx % feature_dim;
    int s_out = (flat_idx / feature_dim) % output_seq_len;
    int h = (flat_idx / feature_dim / output_seq_len) % num_heads;
    int b = (flat_idx / feature_dim / output_seq_len / num_heads) % batch_size;

    int chunk_id = s_out / chunk_size;
    int chunk_offset = s_out % chunk_size;
    int src_s = chunk_indices[chunk_id] * chunk_size + chunk_offset;

    if (src_s >= seq_len) return;

    int src_flat_idx = ((b * num_heads + h) * seq_len + src_s) * feature_dim + d;
    int dst_flat_idx = flat_idx;

    output_reranked_indices[dst_flat_idx] = startend_row_indices[src_flat_idx];
}




// ============================================================================
//                          ScanMaxMin Operator
// ============================================================================

std::vector<paddle::Tensor> scan_max_min_cuda(
    const paddle::Tensor& input,
    const int head_size_rounded,
    const int seq_len_q,
    const int seq_len_k,
    const int blocksize = -1,
    const bool is_causal = false,
    const float softcap = 0.0,
    const int window_size_left = 0,
    const int window_size_right = 0) {
    CHECK_CUDA_INPUT(input);

    const auto batch_size = input.shape()[0];
    const auto num_heads = input.shape()[1];
    const auto num_sequences = input.shape()[2];
    const auto head_dim = input.shape()[3];

    PADDLE_ENFORCE_EQ(
        num_sequences,
        seq_len_k,
        common::errors::InvalidArgument(
            "Input tensor's third dimension (num_sequences) must be equal to seq_len_k."));

    const bool is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
    const bool is_flashmask = true;
    const bool has_softcap = softcap > 0.0;
    const bool has_lt_end = !is_causal && head_dim >= 2;
    const bool has_ut_start = head_dim == 4;

    const int kernel_block_size_n =
        blocksize > 0 ? blocksize : get_kBlockN(head_size_rounded,
                                                is_flashmask,
                                                is_causal,
                                                has_softcap,
                                                is_local,
                                                seq_len_q,
                                                seq_len_k,
                                                has_lt_end,
                                                has_ut_start);

    // Pad the number of blocks to be a multiple of 4 for performance
    const int num_blocks_seqlen =
        ((num_sequences + kernel_block_size_n - 1) / kernel_block_size_n + 3) & 0xfffffffc;

    std::vector<int64_t> output_shape = {batch_size, num_blocks_seqlen};
    auto max_output = paddle::empty(output_shape, input.dtype(), input.place());
    auto min_output = paddle::empty(output_shape, input.dtype(), input.place());

    // Launch kernel
    dim3 block_dim(32, 4);
    dim3 grid_dim((num_sequences + kernel_block_size_n - 1) / kernel_block_size_n,
                  (batch_size + 3) / 4);
    
    const cudaStream_t stream = input.stream();

    switch (kernel_block_size_n) {
        case 32:
            scanMaxMinChunkedKernel<32><<<grid_dim, block_dim, 0, stream>>>(
                input.data<int>(), batch_size, num_sequences,
                max_output.data<int>(), min_output.data<int>());
            break;
        case 64:
            scanMaxMinChunkedKernel<64><<<grid_dim, block_dim, 0, stream>>>(
                input.data<int>(), batch_size, num_sequences,
                max_output.data<int>(), min_output.data<int>());
            break;
        case 96:
            scanMaxMinChunkedKernel<96><<<grid_dim, block_dim, 0, stream>>>(
                input.data<int>(), batch_size, num_sequences,
                max_output.data<int>(), min_output.data<int>());
            break;
        case 128:
            scanMaxMinChunkedKernel<128><<<grid_dim, block_dim, 0, stream>>>(
                input.data<int>(), batch_size, num_sequences,
                max_output.data<int>(), min_output.data<int>());
            break;
        default:
            PD_THROW("Unsupported kernel_block_size_n: %d", kernel_block_size_n);
    }
    return {max_output, min_output};
}

std::vector<paddle::Tensor> ScanMaxMin(
    const paddle::Tensor& input,
    int head_size_rounded,
    int seq_len_q,
    int seq_len_k,
    int blocksize,
    bool is_causal,
    float softcap,
    int window_size_left,
    int window_size_right) {
#ifdef PADDLE_WITH_CUDA
    if (input.is_gpu()) {
        return scan_max_min_cuda(input,
                                 head_size_rounded,
                                 seq_len_q,
                                 seq_len_k,
                                 blocksize,
                                 is_causal,
                                 softcap,
                                 window_size_left,
                                 window_size_right);
    }
#endif
    PD_THROW("Unsupported device: ScanMaxMin operator is only available for CUDA.");
}


// ============================================================================
//                          ReduceWorkload Operator
// ============================================================================

template <int kBlockM>
void launch_reduce_workload_kernel(
    const paddle::Tensor& lt_start_max,
    const paddle::Tensor& lt_start_min,
    const paddle::optional<paddle::Tensor>& lt_end_max,
    const paddle::optional<paddle::Tensor>& lt_end_min,
    const paddle::optional<paddle::Tensor>& ut_start_max,
    const paddle::optional<paddle::Tensor>& ut_start_min,
    const paddle::optional<paddle::Tensor>& ut_end_max,
    const paddle::optional<paddle::Tensor>& ut_end_min,
    paddle::Tensor& workload,
    int batch_times_heads,
    int num_row_blocks,
    int num_col_blocks,
    int stride,
    int row_block_size,
    bool is_causal,
    cudaStream_t stream) {
    
    dim3 block_dim(1024, 1);
    dim3 grid_dim(num_row_blocks, batch_times_heads);

    int ptr_dispatch_tag = SINGLE_PTR;
    if (lt_end_max || ut_end_max) {
        ptr_dispatch_tag = DUAL_PTR;
        if (ut_start_max) {
            ptr_dispatch_tag = FULL_PTR;
        }
    }

    int* workload_ptr = workload.data<int>();
    const int* lt_start_max_ptr = lt_start_max.data<int>();
    const int* lt_start_min_ptr = lt_start_min.data<int>();
    const int* lt_end_max_ptr = lt_end_max ? lt_end_max.get().data<int>() : nullptr;
    const int* lt_end_min_ptr = lt_end_min ? lt_end_min.get().data<int>() : nullptr;
    const int* ut_start_max_ptr = ut_start_max ? ut_start_max.get().data<int>() : nullptr;
    const int* ut_start_min_ptr = ut_start_min ? ut_start_min.get().data<int>() : nullptr;
    const int* ut_end_max_ptr = ut_end_max ? ut_end_max.get().data<int>() : nullptr;
    const int* ut_end_min_ptr = ut_end_min ? ut_end_min.get().data<int>() : nullptr;

    if (ptr_dispatch_tag == FULL_PTR) {
        reduce_workload_kernel<kBlockM, FULL_PTR, false><<<grid_dim, block_dim, 0, stream>>>(
            lt_start_max_ptr, lt_start_min_ptr, lt_end_max_ptr, lt_end_min_ptr,
            ut_start_max_ptr, ut_start_min_ptr, ut_end_max_ptr, ut_end_min_ptr,
            workload_ptr, batch_times_heads, num_row_blocks, num_col_blocks, stride, row_block_size);
    } else if (ptr_dispatch_tag == DUAL_PTR) {
        if (is_causal) {
            reduce_workload_kernel<kBlockM, DUAL_PTR, true><<<grid_dim, block_dim, 0, stream>>>(
                lt_start_max_ptr, lt_start_min_ptr, lt_end_max_ptr, lt_end_min_ptr,
                ut_start_max_ptr, ut_start_min_ptr, ut_end_max_ptr, ut_end_min_ptr,
                workload_ptr, batch_times_heads, num_row_blocks, num_col_blocks, stride, row_block_size);
        } else {
            reduce_workload_kernel<kBlockM, DUAL_PTR, false><<<grid_dim, block_dim, 0, stream>>>(
                lt_start_max_ptr, lt_start_min_ptr, lt_end_max_ptr, lt_end_min_ptr,
                ut_start_max_ptr, ut_start_min_ptr, ut_end_max_ptr, ut_end_min_ptr,
                workload_ptr, batch_times_heads, num_row_blocks, num_col_blocks, stride, row_block_size);
        }
    } else if (ptr_dispatch_tag == SINGLE_PTR) {
        reduce_workload_kernel<kBlockM, SINGLE_PTR, false><<<grid_dim, block_dim, 0, stream>>>(
            lt_start_max_ptr, lt_start_min_ptr, lt_end_max_ptr, lt_end_min_ptr,
            ut_start_max_ptr, ut_start_min_ptr, ut_end_max_ptr, ut_end_min_ptr,
            workload_ptr, batch_times_heads, num_row_blocks, num_col_blocks, stride, row_block_size);
    } else {
        PD_THROW("Unknown pointer dispatch tag.");
    }
}

std::vector<paddle::Tensor> reduce_workload_cuda(
    const paddle::Tensor& lt_start_max,
    const paddle::Tensor& lt_start_min,
    const paddle::optional<paddle::Tensor>& lt_end_max,
    const paddle::optional<paddle::Tensor>& lt_end_min,
    const paddle::optional<paddle::Tensor>& ut_start_max,
    const paddle::optional<paddle::Tensor>& ut_start_min,
    const paddle::optional<paddle::Tensor>& ut_end_max,
    const paddle::optional<paddle::Tensor>& ut_end_min,
    int batch_size,
    int num_heads,
    int num_row_blocks,
    int num_col_blocks,
    int stride,
    int row_block_size,
    bool is_causal,
    int m_block_size) {
    
    const int kBlockM = m_block_size;
    const int batch_times_heads = batch_size * num_heads;
    
    // Allocate output tensor
    std::vector<int64_t> output_shape = {batch_size, num_heads, num_row_blocks, 1};
    auto workload = paddle::empty(output_shape, lt_start_max.dtype(), lt_start_max.place());
    
    cudaStream_t stream = lt_start_max.stream();

    switch (kBlockM) {
        case 64:
            launch_reduce_workload_kernel<64>(
                lt_start_max, lt_start_min, lt_end_max, lt_end_min, ut_start_max,
                ut_start_min, ut_end_max, ut_end_min, workload, batch_times_heads,
                num_row_blocks, num_col_blocks, stride, row_block_size, is_causal, stream);
            break;
        case 96:
            launch_reduce_workload_kernel<96>(
                lt_start_max, lt_start_min, lt_end_max, lt_end_min, ut_start_max,
                ut_start_min, ut_end_max, ut_end_min, workload, batch_times_heads,
                num_row_blocks, num_col_blocks, stride, row_block_size, is_causal, stream);
            break;
        case 128:
            launch_reduce_workload_kernel<128>(
                lt_start_max, lt_start_min, lt_end_max, lt_end_min, ut_start_max,
                ut_start_min, ut_end_max, ut_end_min, workload, batch_times_heads,
                num_row_blocks, num_col_blocks, stride, row_block_size, is_causal, stream);
            break;
        default:
            PD_THROW("Unsupported m_block_size: %d", kBlockM);
    }
    return {workload};
}

std::vector<paddle::Tensor> ReduceWorkloadOp(
    const paddle::Tensor& lt_start_max,
    const paddle::Tensor& lt_start_min,
    const paddle::optional<paddle::Tensor>& lt_end_max,
    const paddle::optional<paddle::Tensor>& lt_end_min,
    const paddle::optional<paddle::Tensor>& ut_start_max,
    const paddle::optional<paddle::Tensor>& ut_start_min,
    const paddle::optional<paddle::Tensor>& ut_end_max,
    const paddle::optional<paddle::Tensor>& ut_end_min,
    int batch_size,
    int num_heads,
    int num_row_blocks,
    int num_col_blocks,
    int stride,
    int row_block_size,
    bool is_causal,
    int m_block_size) {
#ifdef PADDLE_WITH_CUDA
    if (lt_start_max.is_gpu()) {
        return reduce_workload_cuda(lt_start_max,
                                    lt_start_min,
                                    lt_end_max,
                                    lt_end_min,
                                    ut_start_max,
                                    ut_start_min,
                                    ut_end_max,
                                    ut_end_min,
                                    batch_size,
                                    num_heads,
                                    num_row_blocks,
                                    num_col_blocks,
                                    stride,
                                    row_block_size,
                                    is_causal,
                                    m_block_size);
    }
#endif
    PD_THROW("Unsupported device: ReduceWorkload operator is only available for CUDA.");
}


// ============================================================================
//                       IndicesToChunks & IndicesRerank Operators
// ============================================================================

std::vector<paddle::Tensor> IndicesToChunksOp(
    const paddle::Tensor& row_indices,
    const paddle::Tensor& chunk_bucket_indices,
    int chunk_size) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_EQ(row_indices.is_gpu(), true,
                      common::errors::InvalidArgument("Input 'row_indices' must be a CUDA tensor."));
    
    auto chunked_result = paddle::empty_like(row_indices);
    
    const int num_rows = row_indices.numel();
    const int num_buckets = chunk_bucket_indices.numel();
    const int num_threads_per_block = 256;
    const int num_blocks = (num_rows + num_threads_per_block - 1) / num_threads_per_block;

    indices_to_chunks_kernel<<<num_blocks, num_threads_per_block, 0, row_indices.stream()>>>(
        row_indices.data<int>(),
        chunk_bucket_indices.data<int>(),
        chunked_result.data<int>(),
        num_rows,
        num_buckets,
        chunk_size);
        
    return {chunked_result};
#else
    PD_THROW("Unsupported device: IndicesToChunks operator is only available for CUDA.");
#endif
}

std::vector<paddle::Tensor> IndicesRerankOp(
    const paddle::Tensor& input_row_indices,
    const paddle::Tensor& chunk_indices,
    int batch_size,
    int num_heads,
    int seq_len,
    int feature_dim,
    int num_chunks,
    int chunk_size) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_EQ(input_row_indices.is_gpu(), true,
                      common::errors::InvalidArgument("Input 'input_row_indices' must be a CUDA tensor."));

    const int output_seq_len = num_chunks * chunk_size;
    auto reranked_indices = paddle::empty({batch_size, num_heads, output_seq_len, feature_dim},
                                          input_row_indices.dtype(),
                                          input_row_indices.place());
    
    const int total_elements = batch_size * output_seq_len * num_heads * feature_dim;
    const int num_threads_per_block = 256;
    const int num_blocks = (total_elements + num_threads_per_block - 1) / num_threads_per_block;

    indices_rerank_kernel<<<num_blocks, num_threads_per_block, 0, input_row_indices.stream()>>>(
        input_row_indices.data<int>(),
        reranked_indices.data<int>(),
        chunk_indices.data<int>(),
        batch_size,
        num_heads,
        seq_len,
        feature_dim,
        num_chunks,
        chunk_size);
        
    return {reranked_indices};
#else
    PD_THROW("Unsupported device: IndicesRerank operator is only available for CUDA.");
#endif
}


// ============================================================================
//                          Operator Registrations
// ============================================================================

PD_BUILD_OP(scan_max_min)
    .Inputs({"Input"})
    .Outputs({"MaxOut", "MinOut"})
    .Attrs({"head_size_rounded: int",
            "seq_len_q: int",
            "seq_len_k: int",
            "blocksize: int",
            "is_causal: bool",
            "softcap: float",
            "window_size_left: int",
            "window_size_right: int"})
    .SetKernelFn(PD_KERNEL(ScanMaxMin));

PD_BUILD_OP(reduce_workload)
    .Inputs({"LTStartMax", "LTStartMin", 
             paddle::Optional("LTEndMax"), paddle::Optional("LTEndMin"),
             paddle::Optional("UTStartMax"), paddle::Optional("UTStartMin"),
             paddle::Optional("UTEndMax"), paddle::Optional("UTEndMin")})
    .Outputs({"Workload"})
    .Attrs({"batch_size: int",
            "num_heads: int",
            "num_row_blocks: int",
            "num_col_blocks: int",
            "stride: int",
            "row_block_size: int",
            "is_causal: bool",
            "m_block_size: int"})
    .SetKernelFn(PD_KERNEL(ReduceWorkloadOp));

PD_BUILD_OP(indices_to_chunks)
    .Inputs({"RowIndices", "ChunkBucketIndices"})
    .Outputs({"ChunkedResult"})
    .Attrs({"chunk_size: int"})
    .SetKernelFn(PD_KERNEL(IndicesToChunksOp));

PD_BUILD_OP(indices_rerank)
    .Inputs({"InputRowIndices", "ChunkIndices"})
    .Outputs({"RerankedIndices"})
    .Attrs({"batch_size: int",
            "num_heads: int",
            "seq_len: int",
            "feature_dim: int",
            "num_chunks: int",
            "chunk_size: int"})
    .SetKernelFn(PD_KERNEL(IndicesRerankOp));