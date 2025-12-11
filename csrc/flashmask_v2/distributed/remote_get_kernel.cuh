#pragma once
#include <cuda_runtime.h>
#include "sr_buffer.cuh"
#include "cutlass/fast_math.h"

#define NVSHMEM_DEBUG

inline void CheckCudaErrorAux(const char *file, unsigned line,
                                       const char *statement, cudaError_t err) {
    if (err == cudaSuccess)
        return;
    printf("%s returned %s(%d) at %s:%u\n", statement, cudaGetErrorString(err),
           err, file, line);
    exit(1);
}

#if defined(NVSHMEM_DEBUG)
#define CUDA_DEBUG_CHECK(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
#else
#define CUDA_DEBUG_CHECK(value) do {} while(0)
#endif

/**
 * This kernel assumes num head is small, like <= 3, so that we directly copy
 * the entire head dimension, without the need to calculate index
 * 
 * @param column_mask tensor shape (B, 1, S, 2) for now, in the future when
 *      more masks need support, this can be easily extended
 * @param S_stride stride seqluence length: each warp is responsible for transfering
 *      one row. Passing D means we only transfer one row, passing H * D means we are
 *      transfering the same rows on different heads.
 * 
 * gridDim.x: total number of rows to be copied / num_warps per block (4)
 * blockDim.x: number of warps (4) * 32
 * 
 * num_blocks & num_warps & S are made compile-time known to alleviate integer op pressure
 * 
 * Note that: using this kernel, we can either fetch the entire row on different heads (H * D)
 * or single head, depending on the `S_stride` and `rows_per_warp`
*/
template <typename T, int num_blocks=16, int num_warps=16>
__global__ __launch_bounds__(128, 16) void SparseKVFewHeadRemoteGetKernel(
    T* const __restrict__ sr_buffer,
    const int* const __restrict__ column_mask,
    const int S_stride,                 // H * D
    const int remote_pe,
    const int rows_per_warp,            // (H) * S / num_blocks / num_warps
    const int num_elem = 0              // (B*S*H*D), offset for V
) {
    // assume num_head is even, so that we can transfer two rows per warp
    // and only need to read mask once (since heads reuse the same mask, 
    // even head num won't result in two rows in a warp mapped to different seq)
    constexpr int seqlen_stride = num_warps * num_blocks;
    const int warp_id = threadIdx.x >> 5;

    for (int i = 0, seqlen_offset = blockIdx.x * num_warps + warp_id;
        i < rows_per_warp; 
        i++, seqlen_offset += seqlen_stride
    ) {
        // TODO(heqianyue): We only support LTS + UTE and multi-head shared mask currently, this should be extended
        const int2 lts_ute = *(reinterpret_cast<const int2*>(&column_mask[seqlen_offset * 2]));
        if (lts_ute.x > lts_ute.y) {         // mask does not cover the whole row of KV
            const int offset = seqlen_offset * S_stride;
            // actually, two calls can be merged
            nvshmemx_getmem_nbi_warp(
                        &sr_buffer[offset],
                        &sr_buffer[offset],
                        S_stride * sizeof(T), remote_pe
            );
            nvshmemx_getmem_nbi_warp(
                    &sr_buffer[offset + num_elem],
                    &sr_buffer[offset + num_elem],
                    S_stride * sizeof(T), remote_pe
            );
        }
        // ensures all transfer is completed. If we use `nvshmemx_getmem_warp` (blocking ver), the following line should be commented
        nvshmem_quiet();
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(wptr, num_warps);     // 4 rows are done, move the write ptr
    }
}


/**
 * This kernel does not copy an entire chunk of (B, S, H, D)
 * instead, it progresses firstly along `S`, then `H`, then `D`
 * with each warp responsible of copying two rows of D, for example:
 * 
 * warp 0 of block i copys (i, j, 2)-th row and (i, j, 3)-th row
*/
template <typename T, int num_blocks, int num_warps, int S, int D>
__global__ __launch_bounds__(128, 16) void SparseKVMultiHeadRemoteGetKernel(
    T* const __restrict__ sr_buffer,
    int* const __restrict__ wptr,
    const int* const __restrict__ column_mask,
    const cutlass::FastDivmod head_divmod,
    const int num_head,                     // head / 2
    const int total_work                    // B * S * H
) {
    const int warp_id = threadIdx.x >> 5;
    constexpr int work_stride = num_warps * num_blocks;
    // Use static scheduling (this might lead to workload imbalance)
    for (int work_id = blockIdx.x * num_warps + warp_id; work_id < total_work; work_id += work_stride) {
        int seqlen_id = work_id % S;            // S must be a power of 2, so this is fast
        int batch_id, head_id;
        batch_id = head_divmod.divmod(head_id, work_id / S);
        // TODO (heqianyue): can we simplify this?
        const int offset = ((batch_id * S + seqlen_id) * num_head + head_id) * D;
        const int mask_id = (batch_id * S + seqlen_id) * 2;
        const int2 lts_ute = *(reinterpret_cast<const int2*>(&column_mask[mask_id]));
        if (lts_ute.x > lts_ute.y) {         // mask does not cover the whole row of KV
            // copy data on two different heads (K and V)
            nvshmemx_getmem_nbi_warp(
                        &sr_buffer[offset],
                        &sr_buffer[offset],
                        D * sizeof(T) * 2, remote_pe
            );
            nvshmemx_getmem_nbi_warp(
                        &sr_buffer[offset + total_work * D],
                        &sr_buffer[offset + total_work * D],
                        D * sizeof(T) * 2, remote_pe
            );
        }
        nvshmem_quiet();
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(wptr, num_warps);     // move the write ptr
    }
}

// returns the team and stride between teams
inline nvshmem_team_t simple_collective_topology_setter(int my_global_pe, int& stride) {
    int n_pes = nvshmem_n_pes();
    if (n_pes == 4) {       
        // the launch world only has 4 GPUs, we consider this a single node use case
        // the world group is the CP communication group
        stride = 1;
        return NVSHMEM_TEAM_WORLD; 
    } else {
        stride = n_pes / 4;
        // for example: 2 node (4, 4, 16, 4) ---> comm group [0, 4, 8, 12], stride 4
        // 4 node （8, 4, 32, 4） ---> comm group [0, 8, 16, 24], stride 8
        int my_group_start_pe = my_global_pe % cp_group_stride;

        nvshmem_team_t cp_team;
        nvshmem_team_config_t config; // Default config
        long config_mask = 0;

        int status = nvshmem_team_split_strided(
            NVSHMEM_TEAM_WORLD,  // Parent team
            my_group_start_pe,   // Start PE (rank in parent)
            stride,              // Stride
            cp_group_size,       // Size (num PEs in new team)
            &config,
            config_mask,
            &cp_team
        );
        return cp_team;
    }
}

template <typename KVType>
void overlap_comm_kernel(
    const KVType* const k_data,
    const KVType* const v_data,
    SRBuffer<KVType>& kv_buffer,
    int b_kv,
    int s_kv,
    int h_kv,
    int d_kv,
    int cp_size
) {
    // TODO(heqianyue): I don't know how to effectively initialize nvshmem context? Using MPI again?
    cudaStream_t comm_stream;
    cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking);
    size_t numel = b_kv * s_kv * h_kv * d_kv * cp_size;
    
    int my_global_pe = nvshmem_my_pe(), stride = 0;
    nvshmem_team_t cp_team = simple_collective_topology_setter(my_global_pe, stride);

    // copy to the correct position
    CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer.k_data() + my_global_pe / stride, k_data, 
                            numel * sizeof(KVType), cudaMemcpyDeviceToDevice, comm_stream));
    CUDA_DEBUG_CHECK(cudaMemcpyAsync(kv_buffer.v_data() + my_global_pe / stride, v_data, 
                            numel * sizeof(KVType), cudaMemcpyDeviceToDevice, comm_stream));

    constexpr int num_warp = 4;
    constexpr int num_blocks = 64;          // 32 reg, 128 thread, one SM of H800 can hold 16 blocks, we use 4 SM

    if (h_kv > 4) {
        // `SparseKVMultiHeadRemoteGetKernel`
    } else {
        // `SparseKVFewHeadRemoteGetKernel`
    }   

    CUDA_DEBUG_CHECK(cudaStreamSynchronize(comm_stream));
    cudaStreamDestroy(comm_stream);
}
