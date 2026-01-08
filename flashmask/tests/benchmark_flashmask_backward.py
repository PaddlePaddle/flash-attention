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

import json
import numpy as np
from functools import partial
from typing import Optional, List
from tabulate import tabulate
import time
import paddle
from paddle.nn.functional.flash_attention import flashmask_attention
import random
import os

np.random.seed(0)
random.seed(0)

def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = paddle.quantile(times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(paddle, return_mode)(times).item()

def cal_flops(B, H, Sq, Sk, D, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def cal_tflops(flops, time_ms):
    if time_ms == 0.0:
        return 0.0
    return  flops * (1e3 / time_ms) / 1e12

def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: paddle.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    fn()

    paddle.device.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = paddle.empty([int(cache_size // 4)], dtype=paddle.int32)
    else:
        cache = paddle.empty([int(cache_size)], dtype=paddle.int8)

    # Estimate the runtime of the function
    start_event = paddle.device.Event(enable_timing=True)
    end_event = paddle.device.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    paddle.device.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    n_warmup = 10
    n_repeat = 50
    start_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        #cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    paddle.device.synchronize()
    times = paddle.to_tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=paddle.float32)
    return _summarize_statistics(times, quantiles, return_mode)

def test_mask(
    generate_mask_fn,
    B: int = 16,
    S: int = 8192,
    H: int = 16,
    D: int = 64,
    dtype = 'bf16',
    fm_version = 1,
):

    #B = 1
    #S = 8192
    #H = 4
    #D = 128

    if dtype == 'bf16':
        data_type = paddle.bfloat16
    else:
        data_type = paddle.float16

    query = paddle.randn([B, S, H, D], dtype=data_type)
    key = paddle.randn([B, S, H, D], dtype=data_type)
    value = paddle.randn([B, S, H, D], dtype=data_type)
    output_grad = paddle.randn([B, S, H, D], dtype=data_type)

    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False

    startend_row_indices, causal = None, True
    if generate_mask_fn is not None:
        startend_row_indices, causal = generate_mask_fn(B, S, H, D)

    output, log_sum_exp = flashmask_attention(
        query,
        key,
        value,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_softmax_lse=True,
        training=True,
    )

    def flashmask_bwd():
        if fm_version == 4:
            from flash_mask.cute import flashmask_utils as fm
            from flash_mask.cute.interface import _flash_attn_bwd
            log_sum_exp_fa4 = log_sum_exp
            if log_sum_exp.shape[-1] != S:
                new_shape = (log_sum_exp.shape[0], log_sum_exp.shape[1], S)
                num = np.prod(log_sum_exp.shape[:2]) * S
                log_sum_exp_fa4 = log_sum_exp.flatten()[:num].reshape(new_shape)

            flashmask_info = None
            if startend_row_indices is not None:
                flashmask_info = fm.FlashMaskInfoPaddle(
                    startend_row_indices=startend_row_indices,
                    is_causal=causal,
                )
            fm4_query_grad, fm4_key_grad, fm4_value_grad = _flash_attn_bwd(
                query,
                key,
                value,
                output,
                output_grad,
                log_sum_exp_fa4,
                flashmask_info,
                causal=causal,
            )
        elif fm_version == 3:
            if startend_row_indices is not None:
                fm_query_grad, fm_key_grad, fm_value_grad = paddle._C_ops.flashmask_attention_v2_grad(
                    query,
                    key,
                    value,
                    output,
                    log_sum_exp,
                    startend_row_indices,
                    None,
                    output_grad,
                    query.shape[-1] ** (-0.5),
                    causal,
                )
            else:
                fm_query_grad, fm_key_grad, fm_value_grad = paddle._C_ops.flash_attn_v3_grad(
                    query,
                    key,
                    value,
                    output,
                    log_sum_exp,
                    output_grad,
                    query.shape[-1] ** (-0.5),  # default softmax_scale
                    causal,
                    -1,  # window_size_left
                    -1,  # window_size_right
                    0.0,  # softcap
                    0,  # sm_margin
                )
        else:
            if startend_row_indices is not None:
                fm_query_grad, fm_key_grad, fm_value_grad = paddle._C_ops.flashmask_attention_grad(
                    query,
                    key,
                    value,
                    startend_row_indices,
                    output,
                    log_sum_exp,
                    paddle.zeros(shape=[query.shape[1], query.shape[2]], dtype=paddle.int64),
                    output_grad,
                    0.0,  # dropout probability
                    causal,
                )
            else:
                fm_query_grad, fm_key_grad, fm_value_grad = paddle._C_ops.flash_attn_grad(
                    query,
                    key,
                    value,
                    output,
                    log_sum_exp,
                    paddle.zeros(shape=[query.shape[1], query.shape[2]], dtype=paddle.int64),
                    None,
                    output_grad,
                    0.0,  # dropout probability
                    causal,
                )

    sparsity = flashmask_block_sparsity(causal, startend_row_indices, B, H, S)
    density = 1.0 - sparsity 

    fwd_time_ms = 0.0

    bwd_time_ms = do_bench(flashmask_bwd)

    total_time_ms = fwd_time_ms + bwd_time_ms

    fwd_flops = density * cal_flops(B, H, S, S, D, mode='fwd')
    bwd_flops = density * cal_flops(B, H, S, S, D, mode='bwd')
    total_flops = density * cal_flops(B, H, S, S, D, mode='fwd_bwd')

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, sparsity

def flashmask_block_sparsity(
    causal,
    flashmask,
    B=None,
    H=None,
    S=None,
    KV_BLOCK_SIZE=128,
    Q_BLOCK_SIZE=128,
    ):

    if flashmask is None and not causal:
        return 0.0
    elif flashmask is None and causal:
        Br = Q_BLOCK_SIZE
        Bc = KV_BLOCK_SIZE
        Tr = S // Br
        Tc = S // Bc
        total_size = B * H * S * S
        num_sparse_blocks = Tr * (Tc - 1) // 2 * B * H
        sparsity = ((num_sparse_blocks * Bc * Br) / total_size)
        return sparsity

    LTS = None
    LTE = None
    UTS = None
    UTE = None
    if flashmask.shape[-1] == 4:
        LTS, LTE, UTS, UTE = flashmask.split(4, axis=-1)
        LTS = LTS.squeeze(-1)
        LTE = LTE.squeeze(-1)
        UTS = UTS.squeeze(-1)
        UTE = UTE.squeeze(-1)
    elif flashmask.shape[-1] == 2 and causal:
        LTS, LTE = flashmask.split(2, axis=-1)
        LTS = LTS.squeeze(-1)
        LTE = LTE.squeeze(-1)
    elif flashmask.shape[-1] == 2 and not causal:
        LTS, UTE = flashmask.split(2, axis=-1)
        LTS = LTS.squeeze(-1)
        UTE = UTE.squeeze(-1)
    else:
        LTS = flashmask.squeeze(-1)

    Br = Q_BLOCK_SIZE
    Bc = KV_BLOCK_SIZE
    
    if LTS is not None:
        B, H, S = LTS.shape
    if LTE is not None:
        B, H, S = LTE.shape
    if UTS is not None:
        B, H, S = UTS.shape
    if UTE is not None:
        B, H, S = UTE.shape
    
    Tr = S // Br
    Tc = S // Bc

    if LTS is not None:
        LTS = LTS.cpu().detach().numpy()
    else:
        LTS = np.full((B, H, S), S, dtype=np.int32)
    LTStartMax = np.array(LTS).reshape([B, H, -1, Bc]).max(axis=-1)
    LTStartMin = np.array(LTS).reshape([B, H, -1, Bc]).min(axis=-1)

    if LTE is not None:
        LTE = LTE.cpu().detach().numpy()
    else:
        LTE = np.full((B, H, S), S, dtype=np.int32)
    LTEndMax = np.array(LTE).reshape([B, H, -1, Bc]).max(-1)
    LTEndMin = np.array(LTE).reshape([B, H, -1, Bc]).min(-1)
    
    if UTS is not None:
        UTS = UTS.cpu().detach().numpy()
    else:
        UTS = np.full((B, H, S,), 0, dtype=np.int32)
    UTStartMax = np.array(UTS).reshape([B, H, -1, Bc]).max(-1)
    UTStartMin = np.array(UTS).reshape([B, H, -1, Bc]).min(-1)

    if UTE is not None:
        UTE = UTE.cpu().detach().numpy()
    else:
        UTE = np.tile(np.arange(S, dtype=np.int32).reshape(1, 1, S), (B, H, 1))
    UTEndMax = np.array(UTE).reshape([B, H, -1, Bc]).max(-1)
    UTEndMin = np.array(UTE).reshape([B, H, -1, Bc]).min(-1)

    
    num_dense_blocks = 0
    for bsz in range(B):
        for head in range(H):
            for i in range(Tr):
                for j in range(Tc):
                    if causal and j > i:
                        #print('S', end="")
                        continue
                    if i * Br >= LTStartMax[bsz, head, j] and (i+1) * Br <= LTEndMin[bsz, head, j]:
                        #print('S', end="")
                        continue
                    if i * Br >= UTStartMax[bsz, head, j] and (i+1) * Br <= UTEndMin[bsz, head, j]:
                        #print('S', end="")
                        continue
            
                    if (i+1) * Br > LTStartMin[bsz, head, j] and i * Br < LTEndMax[bsz, head, j]:
                        #print('A', end="")
                        num_dense_blocks += 1
                        continue
                    if (i+1) * Br > UTStartMin[bsz, head, j] and i * Br < UTEndMax[bsz, head, j]:
                        #print('A', end="")
                        num_dense_blocks += 1
                        continue
            
                    #print('C', end="")
                    num_dense_blocks += 1
                #print()

    num_sparse_blocks = B * H * Tc * Tr - num_dense_blocks
    total_size = B * H * S * S
    sparsity = ((num_sparse_blocks * Bc * Br) / total_size)
    return sparsity


def generate_none_mask(B, S, H, D, causal=True):
    return None, causal

def generate_sliding_window_mask(B, S, H, D, window_size=1024):
    startend_row_indices = paddle.arange(
        window_size, S + window_size, dtype="int32"
    ).reshape((1, 1, S, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=S
    ).repeat_interleave(B, 0)

    causal=True
    return startend_row_indices, causal

def generate_causal_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 1
    padding = S - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    startend_row_indices = np.repeat(seq_cusums, doc_seq_lens)
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 1
    padding = S - np.sum(doc_seq_lens)

    down_left_row_indices = []
    up_right_row_indices = []

    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
    return startend_row_indices, causal

def generate_share_question_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):

    total_seq_len = sum([sum(doc) for doc in doc_seq_lens])
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 1
    padding = S - total_seq_len
    if padding > 0:
        doc_seq_lens.append([padding])

    startend_row_indices = []
    seqlen_so_far = 0
    for doc in doc_seq_lens:
        assert len(doc) >= 1
        doc_len = sum(doc)
        for idx, seqlen in enumerate(doc):
            if idx == 0:
                startend_row_indices.extend([seqlen_so_far + doc_len] * doc[idx])
            else:
                startend_row_indices.extend([seqlen_so_far + seqlen] * doc[idx])
            seqlen_so_far += seqlen

    assert seqlen_so_far == S

    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_global_sliding_window_mask(B, S, H, D, global_token=16, window_size=(512, 512)):
    assert len(window_size) == 2
    left_window_size, right_window_size = window_size

    down_left_start_row_indices = []
    down_left_end_row_indices = []
    up_right_start_row_indices = []
    up_right_end_row_indices = []

    down_left_start_row_indices = paddle.arange(
        left_window_size + 1, S + left_window_size + 1, dtype="int32"
    ).clip(max=S)
    down_left_start_row_indices[:global_token] = 0
    down_left_start_row_indices = down_left_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    down_left_end_row_indices = paddle.full([S], S, dtype="int32")
    down_left_end_row_indices[:global_token] = 0
    down_left_end_row_indices = down_left_end_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_start_row_indices = paddle.full([S], global_token, dtype="int32")
    up_right_start_row_indices[:global_token+right_window_size+1] = 0
    up_right_start_row_indices = up_right_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_end_row_indices = paddle.arange(
        -right_window_size, S - right_window_size, dtype="int32"
    )
    up_right_end_row_indices[:global_token+right_window_size+1] = 0
    up_right_end_row_indices = up_right_end_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_start_row_indices, down_left_end_row_indices, up_right_start_row_indices, up_right_end_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_causal_blockwise_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - np.sum(doc_seq_lens)

    start_row_indices = []
    cur_len_so_far = doc_seq_lens[0]
    for i in range(len(doc_seq_lens)):
        start_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1]
    if padding > 0:
        start_row_indices.extend([cur_len_so_far] * padding)
    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    seq_cusums = np.cumsum(doc_seq_lens)
    end_row_indices = [seq_cusums[-2]] * seq_cusums[-2] + [seq_cusums[-1]] * doc_seq_lens[-1] + [S] * padding
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

def generate_prefix_lm_document_mask(B, S, H, D, doc_seq_lens=[(1024, 2538), (1742, 1742), (512, 3213)]):
    """
    tuple(prefix_length, seq_length)
    """
    assert len(doc_seq_lens) >= 2
    total_seq_len = 0
    for prefix_length, seq_length in doc_seq_lens:
        total_seq_len += seq_length
    assert total_seq_len <= S
    padding = S - total_seq_len

    down_left_row_indices = []
    cur_len_so_far = doc_seq_lens[0][1]
    for i in range(len(doc_seq_lens)):
        down_left_row_indices.extend([cur_len_so_far] * doc_seq_lens[i][1])
        if i < len(doc_seq_lens) - 1:
            cur_len_so_far += doc_seq_lens[i+1][1]
    if padding > 0:
        down_left_row_indices.extend([cur_len_so_far] * padding)
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    up_right_row_indices = []
    cur_len_so_far = 0
    for prefix_length, seq_length in doc_seq_lens:
        up_right_row_indices.extend([cur_len_so_far] * prefix_length + list(range(cur_len_so_far+prefix_length, cur_len_so_far+seq_length)))
        cur_len_so_far += seq_length
    if padding > 0:
        up_right_row_indices.extend([total_seq_len] * padding)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_prefix_lm_causal_mask(B, S, H, D, prefix_length=1024):
    """
    tuple(prefix_length, seq_length)
    """
    assert prefix_length <= S
    down_left_row_indices = paddle.full([S], S, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor([0] * prefix_length + list(range(prefix_length, S)), dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)

    causal = False
    return startend_row_indices, causal

def generate_qk_sparse_mask(B, S, H, D, maskout_pair=[(1024, 538), (2358, 1700)]):
    """
    tuple(offset, maskout_len)
    """
    start_row_indices = []
    end_row_indices  = []
    last_offset = 0
    for offset, maskout_len in maskout_pair:
        assert offset >= last_offset
        start_row_indices.extend(list(range(last_offset, offset)))
        end_row_indices.extend(list(range(last_offset, offset)))

        start_row_indices.extend(list(range(offset, offset+maskout_len)))
        end_row_indices.extend([offset+maskout_len]*(maskout_len))

        last_offset = offset + maskout_len

    last_offset <= S
    start_row_indices.extend(list(range(last_offset, S)))
    end_row_indices.extend(list(range(last_offset, S)))

    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

def generate_random_eviction_mask(B, S, H, D, start_row=4096):
    start_rows_list = []
    for bz_idx in range(B):
        for head_idx in range(H):
            start_rows = np.array([S+1] * S)
            mask_pos = np.random.choice(S-1, S - start_row, replace=False)
            index = np.arange(start_row, S)
            mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            start_rows[mask_pos] = index
            start_rows_list.append(start_rows)
    startend_row_indices = paddle.to_tensor(start_rows_list, dtype=paddle.int32).reshape((B, H, S, 1))
    causal = True
    return startend_row_indices, causal

def split_sequence(sequence_length):
    if sequence_length < 3:
        raise ValueError("序列长度必须至少为 3，以保证能够分配给一个 Question 和两个 Answer。")
    
    # 确定 Answer 的数量
    num_answers = random.randint(2, 6)
    
    # 初始化分配的长度
    lengths = [1] * (num_answers + 1)  # 至少给每个部分分配一个长度，确保为正整数
    
    # 剩余的长度需要分配
    remaining_length = sequence_length - sum(lengths)
    
    # 随机分配剩余的长度
    for _ in range(remaining_length):
        # 随机选择一个位置增加长度
        index = random.randint(0, num_answers)
        lengths[index] += 1

    return lengths

kernel_test_seq_info = [
    {
        "total_length": 8192,
        "document_count_range": (2, 6),
        "samples": [
            {
                "doc_list": [(885, 3145), (296, 2203), (528, 2844)],
                "qksparse_mask": [0, 1, 0],
            },
            {
                "doc_list": [(321, 2481), (245, 1388), (428, 1978), (357, 2345)],
                "qksparse_mask": [0, 0, 1, 0],
            },
            {
                "doc_list": [(245, 1041), (737, 2718), (391, 1420), (275, 1191), (328, 1822)],
                "qksparse_mask": [0, 1, 1, 0, 0],
            },
            {
                "doc_list": [(310, 1858), (218, 872), (156, 529), (340, 2059), (138, 1089), (416, 1785)],
                "qksparse_mask": [0, 1, 0, 0, 1, 0],
            },
            {
                "doc_list": [(114, 1144), (64, 407), (483, 1709), (300, 1187), (168, 1138), (150, 915), (490, 1692)],
                "qksparse_mask": [0, 0, 1, 0, 0, 1, 0],
            },
        ]
    },
    {
        "total_length": 32768,
        "document_count_range": (10, 14),
        "samples": [
            {
                "doc_list": [(884, 3482), (956, 4024), (922, 4638), (864, 4022), (308, 2879), (185, 956), (450, 3586), (750, 2682), (707, 3546), (705, 2953)],
                "qksparse_mask": [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
            },
            {
                "doc_list": [(929, 3316), (303, 1441), (654, 3867), (570, 2039), (438, 3531), (799, 4508), (560, 2923), (398, 2440), (452, 1876), (350, 2495), (740, 4332)],
                "qksparse_mask": [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
            },
            {
                "doc_list": [(521, 2427), (703, 2802), (689, 4031), (474, 2121), (516, 3054), (225, 2202), (364, 3554), (443, 2698), (402, 2481), (971, 3376), (392, 2255), (372, 1767)],
                "qksparse_mask": [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            },
            {
                "doc_list": [(325, 1571), (449, 2074), (508, 2095), (341, 1318), (513, 2934), (378, 1368), (423, 3126), (852, 3714), (1053, 5720), (367, 1711), (803, 2756), (742, 2635), (314, 1746)],
                "qksparse_mask": [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
            },
            {
                "doc_list": [(360, 1478), (581, 3207), (225, 839), (266, 1721), (742, 3329), (666, 3018), (276, 2302), (424, 2872), (381, 2381), (455, 2022), (312, 1995), (597, 2786), (406, 2373), (310, 2445)],
                "qksparse_mask": [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            },
        ]
    },
    {
        "total_length": 131072,
        "document_count_range": (11, 15),
        "samples": [
            {
                "doc_list": [(1969, 11610), (2478, 12300), (3805, 16924), (621, 3872), (2045, 13367), (1336, 6646), (3579, 15715), (3981, 15867), (718, 6181), (4431, 17929), (3019, 10661)],
                "qksparse_mask": [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
            },
            {
                "doc_list": [(3140, 12589), (2271, 11764), (1252, 7263), (2881, 15921), (1300, 6369), (1260, 7899), (1634, 7457), (1994, 10330), (2780, 14946), (2591, 12289), (2239, 12603), (1506, 11642)],
                "qksparse_mask": [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
            },
            {
                "doc_list": [(2032, 13561), (2131, 8279), (2443, 15840), (1345, 7267), (2103, 13986), (2307, 11556), (1916, 9207), (1156, 6647), (3089, 12986), (1336, 4941), (1824, 6942), (1287, 7772), (2912, 12088)],
                "qksparse_mask": [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            },
            {
                "doc_list": [(3612, 15286), (930, 8873), (1364, 4782), (1321, 5300), (2914, 17395), (4326, 15612), (1339, 9340), (2096, 9311), (403, 2016), (1530, 7183), (1650, 7695), (1968, 11782), (1109, 5028), (2688, 11469)],
                "qksparse_mask": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            },
            {
                "doc_list": [(1556, 5668), (2217, 9799), (1869, 6686), (1089, 3796), (934, 6543), (3034, 13966), (1302, 12018), (3739, 13309), (1563, 9541), (3327, 11602), (1451, 5941), (2250, 11404), (995, 5651), (1082, 3778), (1273, 11370)],
                "qksparse_mask": [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0],
            },
        ]
    }
]


def main(examples: List[str] = ["all"], dtype='bf16', fm_version=1, suffix="_base", overwrite=True):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    if fm_version == 1:
        paddle.set_flags({'FLAGS_flash_attn_version': 2})
    elif fm_version == 3:
        paddle.set_flags({'FLAGS_flash_attn_version': 3})
    elif fm_version == 4:
        pass
    else:
        raise ArgumentError(f"fm_version must be 1, 3 or 4, but got {fm_version}")
            
    doc_seq_lens_list = []
    for item in kernel_test_seq_info:
        total_length = item['total_length']
        for sample in item['samples']:
            doc_list = sample['doc_list']
            qksparse_mask = sample['qksparse_mask']
            doc_seq_lens_list.append((total_length, doc_list, qksparse_mask))

    #doc_seq_lens_list = doc_seq_lens_list[::-1]

    for D in [128]:
    #for D in [64, 128, 256]:
        H = 4096 // D
        for idx, (S, prefix_doc_seq_lens, qksparse_mask) in enumerate(doc_seq_lens_list):
            B = 128 * 1024 // S

            doc_seq_lens = [x[1] for x in prefix_doc_seq_lens]
            maskout_pair = []
            offset = 0
            print(f"{B}_{S}_{H}_{D}_{idx}_{dtype}")
            if not overwrite:
                if os.path.exists(f"{dtype}{suffix}/flashmaskv{fm_version}_{B}_{S}_{H}_{D}_{idx}.csv"):
                    print(f"{dtype}{suffix}/flashmaskv{fm_version}_{B}_{S}_{H}_{D}_{idx}.csv already exists, skipping. To enable overwrite, use: --overwrite (True by default).")
                    continue
            if sum(qksparse_mask) == 0:
                maskout_pair = [(1024, 538), (2358, 1700)]
            else:
                for is_maskout, doc_seq in zip(qksparse_mask, doc_seq_lens):
                    if is_maskout:
                        maskout_pair.append((offset, doc_seq))
                    offset += doc_seq

            share_qa_docs = [split_sequence(doc_seq) for doc_seq in doc_seq_lens]

            available_examples = {
                "Full": lambda: test_mask(generate_mask_fn=partial(generate_none_mask, causal=False), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Causal": lambda: test_mask(generate_mask_fn=partial(generate_none_mask, causal=True), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Sliding Window": lambda: test_mask(generate_mask_fn=partial(generate_sliding_window_mask, window_size=int(S*0.0625)), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Causal Document Mask": lambda: test_mask(generate_mask_fn=partial(generate_causal_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Document Mask": lambda: test_mask(generate_mask_fn=partial(generate_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Share Question Mask": lambda: test_mask(generate_mask_fn=partial(generate_share_question_mask, doc_seq_lens=share_qa_docs), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                #"Global Sliding Window": lambda: test_mask(generate_mask_fn=partial(generate_global_sliding_window_mask, global_token=16, window_size=(int(S*0.0625), int(S*0.0625))), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Causal Blockwise Mask": lambda: test_mask(generate_mask_fn=partial(generate_causal_blockwise_mask, doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Prefix LM Document Mask": lambda: test_mask(generate_mask_fn=partial(generate_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Prefix LM Causal Mask": lambda: test_mask(generate_mask_fn=partial(generate_prefix_lm_causal_mask, prefix_length=int(S*0.5)), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "QK-sparse Mask": lambda: test_mask(generate_mask_fn=partial(generate_qk_sparse_mask, maskout_pair=maskout_pair), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
                "Random Eviction Mask": lambda: test_mask(generate_mask_fn=partial(generate_random_eviction_mask, start_row=S//2), B=B, S=S, H=H, D=D, dtype=dtype, fm_version=fm_version),
            }

            if "all" in examples:
                ex_to_run = list(available_examples.keys())
            else:
                ex_to_run = examples

            results = []
            for ex in ex_to_run:
                if ex in available_examples:
                    print(ex)
                    fw_time, bw_time, total_time, fw_flops, bw_flops, total_flops, fw_tflops, bw_tflops, total_tflops, sparsity = available_examples[ex]()
                    results.append([ex, f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}", f"{fw_flops:.4f}", f"{bw_flops:.4f}", f"{total_flops:.4f}", f"{fw_tflops:.4f}", f"{bw_tflops:.4f}", f"{total_tflops:4f}", f"{sparsity:.4f}"])
                else:
                    print(f"Warning: Unknown example key '{ex}'. Skipping.")

            # Usage in your results formatting:
            headers = [
                "Operation",
                "FW Time (ms)",
                "BW Time (ms)",
                "TOTAL Time (ms)",
                "FW FLOPs",
                "BW FLOPs",
                "TOTAL FLOPs",
                "FW TFLOPs/s",
                "BW TFLOPs/s",
                "TOTAL TFLOPs/s",
                "Sparsity",
            ]
            print(
                tabulate(
                    results,
                    headers=headers,
                    tablefmt="grid",
                )
            )
            content2=tabulate(results, headers=headers, tablefmt="tsv")
            os.makedirs(f"{dtype}{suffix}", exist_ok=True)
            text_file = open(f"{dtype}{suffix}/flashmaskv{fm_version}_{B}_{S}_{H}_{D}_{idx}.csv","w")
            text_file.write(content2)
            text_file.close()

if __name__ == "__main__":
    #from jsonargparse import ArgumentParser
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run specific examples or all examples.")
    parser.add_argument(
        "--examples",
        type=str,
        nargs="+",
        default=["all"],
        help="List of examples to run. Use space to separate multiple examples. "
        "Available options: causal, alibi, sliding_window, prefix_lm, "
        "document, softcap, softcap_approx, or 'all' to run all examples.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16"
    )
    parser.add_argument(
        "--fm_version",
        type=int,
        default=1
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=""
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true", default=True)
    group.add_argument("--no-overwrite", action="store_false", dest="overwrite")

    args = parser.parse_args()
    main(**vars(args))
