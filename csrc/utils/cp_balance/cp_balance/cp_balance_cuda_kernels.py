import cupy as cp
import numpy as np
import paddle
from paddle.utils.cpp_extension import load
import flashmask_cpbalance_cudaops as cp_balance_ops
    
def scanMaxMinChunkedKernel(input_tensor, Bc, B, H, S):
    maxo,mino = cp_balance_ops.scan_max_min(
        input_tensor,
        H,
        S,
        S,
        Bc,
        False,
        0.0,
        0,
        0
    )

    # 取出结果（假设每行只有一个warp，maxo[:, 0]）
    # LTStartMax_gpu = cp.asnumpy(maxo[:, 0])
    # LTStartMin_gpu = cp.asnumpy(mino[:, 0])
    # print(maxo)
    return maxo, mino


def reduce_workload(start_row_maxmin_indice_list, B, H, Tr, Tc, Br, S):
    (
        LTStartMax,
        LTStartMin,
        LTEndMax,
        LTEndMin,
        UTStartMax,
        UTStartMin,
        UTEndMax,
        UTEndMin,
    ) = start_row_maxmin_indice_list
    
    workload = cp_balance_ops.reduce_workload(
        LTStartMax, LTStartMin, LTEndMax, LTEndMin, UTStartMax, UTStartMin, UTEndMax, UTEndMin,
        B, H, Tr, Tc, S, Br, False, 128
    )
    
    return workload

def indices_to_chunks_cuda(startend_row_indices, bucket_idx, chunksize=2048):
    result = cp_balance_ops.indices_to_chunks(startend_row_indices, bucket_idx, chunksize)
    return result

def indices_rerank_cuda(startend_row_indices, indices, balance_chunk_size=2048):
    B, H, S, D = startend_row_indices.shape
    num_chunks = (S + balance_chunk_size - 1) // balance_chunk_size
    startend_row_indices_rerank = cp_balance_ops.indices_rerank(startend_row_indices, indices, B, H, S,D,num_chunks,balance_chunk_size)
    return startend_row_indices_rerank
