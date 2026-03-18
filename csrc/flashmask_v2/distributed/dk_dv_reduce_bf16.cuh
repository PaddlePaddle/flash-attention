#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <cutlass/bfloat16.h>
#include <cutlass/array.h>

namespace flashmask {

using bf16 = cutlass::bfloat16_t;
using bf16x4 = cutlass::Array<bf16, 4>;

// bf16x4 and float4 conversion, so that we can use fp32 accumulation
__device__ __forceinline__ float4 to_float4(bf16x4 in) {
    return {
        static_cast<float>(in[0]),
        static_cast<float>(in[1]),
        static_cast<float>(in[2]),
        static_cast<float>(in[3])
    };
}

__device__ __forceinline__ bf16x4 to_bf16x4(float4 in) {
    bf16x4 out;
    out[0] = static_cast<bf16>(in.x);
    out[1] = static_cast<bf16>(in.y);
    out[2] = static_cast<bf16>(in.z);
    out[3] = static_cast<bf16>(in.w);
    return out;
}

/**
 * Note that the input buffer and output buffer has shape mismatch:
 * @param dx_send_recv the shape is (B, S_local * num_chunks, H, D)
 * @param dx_accum the shape is (B, S_local, H, D)
 * 
 * So we need to calculate different batch stride for input and output
*/
template <int S_chunk = 8192, int num_chunks = 4, bool is_first = true>
__global__ __launch_bounds__(128, 8) 
void ReducedKdVKernel(
    const bf16* __restrict__ dk_recv,
    const bf16* __restrict__ dv_recv,
    bf16* __restrict__ dk_accum,
    bf16* __restrict__ dv_accum,
    const int num_tasks_per_batch       // S_chunk * H * D / 512
) {
    static constexpr int elem_per_block = 512;
    const int b = blockIdx.y;           // batch
    
    const int elem_per_chunk = num_tasks_per_batch * elem_per_block;    // chunk stride
    const int b_offset_accum = b * elem_per_chunk;
    const int b_offset_sr = b_offset_accum * num_chunks;

    // task offset is small_chunk offset + thread offset
    auto reduce_op = [&](
        const bf16* const __restrict__ src_recv,
        bf16* const __restrict__ dst_accum, int task_offset
    ) {
        // step 1. load values to SMEM
        float4 acc = make_float4(0, 0, 0, 0);
        if constexpr (!is_first) {
            acc = to_float4(*reinterpret_cast<const bf16x4*>(dst_accum + b_offset_accum + task_offset));
        }
        // step 2. use higher precision to do the reduce
        const int base_offset = b_offset_sr + task_offset;
        #pragma unroll
        for (int c = 0; c < num_chunks; ++c) {
            float4 temp_v = to_float4(
                *reinterpret_cast<const bf16x4*>(src_recv + c * elem_per_chunk + base_offset)
            );
            acc.x += temp_v.x;
            acc.y += temp_v.y;
            acc.z += temp_v.z;
            acc.w += temp_v.w;
        }
        
        auto result = to_bf16x4(acc);
        // step 3. store the accumulated results
        *reinterpret_cast<bf16x4*>(dst_accum + b_offset_accum + task_offset) = result;
    };

    // TODO(heqianyue): batch = 1 the following seems correct, what about batch > 1 ?
    for (int task_idx = blockIdx.x; task_idx < num_tasks_per_batch; task_idx += gridDim.x) {
        const int task_offset = task_idx * elem_per_block + 4 * threadIdx.x;

        reduce_op(dk_recv, dk_accum, task_offset);
        reduce_op(dv_recv, dv_accum, task_offset);
    }
}

#define ChunkDipatchKernelLaunch(num_chunk, is_first)                                   \
    switch (num_chunk) {                                                                \
        case 4: { ReducedKdVKernel<S_chunk_exp, 4, is_first><<<grid, 128, 0, stream>>>( \
            dk_recv, dv_recv, dk_accum, dv_accum, num_tasks_per_chunk); break; }        \
        case 2: { ReducedKdVKernel<S_chunk_exp, 2, is_first><<<grid, 128, 0, stream>>>( \
            dk_recv, dv_recv, dk_accum, dv_accum, num_tasks_per_chunk); break; }        \
        case 8: { ReducedKdVKernel<S_chunk_exp, 8, is_first><<<grid, 128, 0, stream>>>( \
            dk_recv, dv_recv, dk_accum, dv_accum, num_tasks_per_chunk); break; }        \
    default:                                                                            \
        throw std::invalid_argument("Num chunk per segment must be 2 or 4. Other segment size is not supported."); \
    }

/**
 * This function calls the dK, dV reduce kernel.
 * @param is_first The first segment to call this function has special
 *  behavior: load from the first chunk of dk/dv_send, then the rest of
 *  the chunks are loaded from dk/dv_recv. If false, we will load from
 *  dk_accum and dv_accum.
*/
void launch_dk_dv_reduce(
    const bf16* dk_recv,
    const bf16* dv_recv,
    bf16* dk_accum, bf16* dv_accum,
    int B, int S_chunk, int H, int D,
    int num_chunks, bool is_first, cudaStream_t stream
) {
    // 128 threads, each reduces 4 bf16
    static constexpr int elem_per_block = 512;
    int elem_per_chunk = S_chunk * H * D;
    // a typical value: 8192 * 8 * 128 / 512 = 16384
    int num_tasks_per_chunk = elem_per_chunk / elem_per_block;

    // typically, B = 1, so we have 2048 CTAs --> 16 CTAs per SM = 128 SMs
    // the reduce speed shouldn't be a bottleneck, so it's OK to allocate more SMs
    dim3 grid(std::max(2048 / B, 128), B);

#define ReduceDispatchBody(_S_chunk_val)                                            \
    do {                                                                             \
        static constexpr int S_chunk_exp = _S_chunk_val;                            \
        if (is_first) {                                                              \
            ChunkDipatchKernelLaunch(num_chunks, true);                              \
        } else {                                                                     \
            ChunkDipatchKernelLaunch(num_chunks, false);                              \
        }                                                                            \
    } while(0)

    switch (S_chunk) {
        case 4096:   { ReduceDispatchBody(4096);   break; }
        case 8192:   { ReduceDispatchBody(8192);   break; }
        case 16384:  { ReduceDispatchBody(16384);  break; }
        case 32768:  { ReduceDispatchBody(32768);  break; }
        case 65536:  { ReduceDispatchBody(65536);  break; }
        case 131072: { ReduceDispatchBody(131072); break; }
    default:
        throw std::invalid_argument(
            "[FlashMask Overlap] S_chunk must be one of {4096, 8192, 16384, 32768, 65536, 131072}, got: "
            + std::to_string(S_chunk));
    }
#undef ReduceDispatchBody
}

#undef ChunkDipatchKernelLaunch

}   // namespace flashmask