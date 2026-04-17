# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import flashmask_cpbalance_cudaops as cp_balance_ops

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
