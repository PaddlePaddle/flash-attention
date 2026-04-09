import numpy as np
from functools import partial
from typing import Optional, List
from tabulate import tabulate
import paddle
import os
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import flashmask_attention
from cp_balance.context_parallel_utils import flashmask_attention_cp,scatter_balance, all_gather_balance
from cp_balance.cp_balance import assign_tasks_heap, get_q_workload,balance_flashmask_input, tensor_md5,balance_alltoall

import paddle.distributed.fleet as fleet
import time

cp_size = 4
strategy = fleet.DistributedStrategy()

strategy.hybrid_configs = {
  "dp_degree": 1,
  "mp_degree": 2,
  "pp_degree": 1,
  "sharding_degree": 4,
  "sep_degree": 1,
  "ep_degree":  8,
  "moe_sharding_degree": 1,
  "cp_degree": cp_size,
  "order": ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"]
}

fleet.init(is_collective=True, strategy=strategy)
cp_group = fleet.get_hybrid_communicate_group().get_context_parallel_group()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def from_paddle(x: paddle.Tensor):
    if x.dtype == paddle.bfloat16 or x.dtype == "bfloat16":
      return torch.from_numpy(x.view("uint16").numpy()).to("cuda").view(torch.bfloat16)
    elif x.dtype == paddle.float32 or x.dtype == "float32":
      return torch.from_numpy(x.numpy()).to("cuda")
    else:
      assert False
      
def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = paddle.quantile(times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(paddle, return_mode)(times).item()

def split_sequence(sequence_length, num_answers=2):
    if sequence_length < num_answers + 1:
        raise ValueError(f"序列长度必须至少为 {num_answers + 1}")

    base = sequence_length // (num_answers + 1)
    extra = sequence_length % (num_answers + 1)
    # 前extra个部分多加1
    lengths = [base + (1 if i < extra else 0) for i in range(num_answers + 1)]

    return lengths

def do_bench_flashmaskcp(q_local, k_local, v_local, o_grad_local, startend_row_indices, group, is_causal,bucket = None, warmup=50, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
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
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    rank = paddle.distributed.get_rank()
    out_local = flashmask_attention_cp(q_local, k_local, v_local, startend_row_indices, causal=is_causal)
    # print('pt00')
    out_local.backward(o_grad_local)
    # print('pt0')
    paddle.distributed.barrier(group=cp_group)
    paddle.device.synchronize()
    # print('here')

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
        time.sleep(0.1)
        cache.zero_()
        out_local = flashmask_attention_cp(q_local, k_local, v_local, startend_row_indices, causal=is_causal, bucket=bucket)
        out_local.backward(o_grad_local)
        paddle.distributed.barrier(group=cp_group)
        paddle.device.synchronize()
    end_event.record()
    paddle.distributed.barrier(group=cp_group)
    paddle.device.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # print('pt2')
    # compute number of warmup and repeat
    n_warmup = max(3, int(warmup / estimate_ms))
    n_repeat = max(5, int(rep / estimate_ms))
    start_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        time.sleep(0.1)
        out_local =flashmask_attention_cp(q_local, k_local, v_local, startend_row_indices, causal=is_causal, bucket=bucket)
        out_local.backward(o_grad_local, retain_graph=True)
        paddle.distributed.barrier(group=cp_group)
        paddle.device.synchronize()
    # Benchmark
    times_fwd = []
    times_bwd = []
    for i in range(n_repeat):
        time.sleep(0.1)
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        paddle.distributed.barrier(group=cp_group)
        paddle.device.synchronize()
        paddle.base.core.nvprof_nvtx_push(f"flashmask_cp_fwd_{rank}")
        t0 = time.perf_counter()
        out_local = flashmask_attention_cp(q_local, k_local, v_local, startend_row_indices, causal=is_causal, bucket=bucket)
        paddle.distributed.barrier(group=cp_group)
        paddle.device.synchronize()
        paddle.base.core.nvprof_nvtx_pop()
        paddle.base.core.nvprof_nvtx_push(f"flashmask_cp_bwd_{rank}")
        t1 = time.perf_counter()
        out_local.backward(o_grad_local, retain_graph=True)
        paddle.distributed.barrier(group=cp_group)
        paddle.device.synchronize()
        paddle.base.core.nvprof_nvtx_pop()
        t2 = time.perf_counter()
        times_fwd.append(1000 * (t1 - t0))
        times_bwd.append(1000 * (t2 - t1))
        
    # Record clocks
    print(times_bwd)
    paddle.distributed.barrier(group=cp_group)
    paddle.device.synchronize()
    # print('pt3')
    return sum(times_fwd) / n_repeat, sum(times_bwd) / n_repeat
    
def cp_flashmask_balance_bench(q, k, v, startend_row_indices, is_causal,o_grad,bucket):
    group = cp_group
    rank = group.rank
    print(f"rank: {rank}")
    q_blocksize = (int)(q.shape[1] // (2 * cp_size))
    k_blocksize = (int)(k.shape[1] // cp_size)
    q_local_1 = q[:, rank*q_blocksize:(rank+1)*q_blocksize, :, :]
    q_local_2 = q[:, (cp_size *2 -rank -1)*q_blocksize:(cp_size *2 -rank)*q_blocksize, :, :]
    q_local = paddle.concat([q_local_1, q_local_2], axis=1).detach()
    k_local = k[:, rank*k_blocksize:(rank+1)*k_blocksize, :, :].detach().contiguous()
    v_local = v[:, rank*k_blocksize:(rank+1)*k_blocksize, :, :].detach().contiguous()
    o_grad_local_1 = o_grad[:, rank * q_blocksize : (rank + 1) * q_blocksize, :, :].detach()
    o_grad_local_2 = o_grad[:, (cp_size * 2 - rank - 1) * q_blocksize : (cp_size * 2 - rank) * q_blocksize, :, :].detach()
    o_grad_local = paddle.concat([o_grad_local_1, o_grad_local_2], axis=1).contiguous()

    
    q_local.stop_gradient = False
    k_local.stop_gradient = False
    v_local.stop_gradient = False
    # startend_row_indices.stop_gradient = False

    cp_fwd_time, cp_bwd_time = do_bench_flashmaskcp(q_local, k_local, v_local, o_grad_local, startend_row_indices, group, is_causal, bucket)
    # print(f"cp balance fwd+bwd time: {cp_fwd_bwd_time} ms\n")
    return cp_fwd_time, cp_bwd_time

def test_cp_famask(
    startend_row_indices,
    B: int = 16,
    S: int = 8192,
    H: int = 16,
    D: int = 64,
    dtype = 'bf16',
):
    """
    测试上下文并行FlashMask注意力机制的性能基准
    
    该函数用于测试在分布式并行环境中FlashMask注意力机制的前向传播和后向传播性能，
    支持不同类型的注意力掩码生成策略。

    Args:
        generate_mask_fn: 注意力掩码生成函数，用于生成startend_row_indices和因果关系标记
        B: 批次大小，默认16
        S: 序列长度，默认8192
        H: 注意力头数，默认16
        D: 每个注意力头的维度，默认64
        dtype: 数据类型，默认'bf16'

    Returns:
        tuple: 包含前向传播时间和后向传播时间的元组 (fwd_time, bwd_time)，单位为毫秒
    """
    # paddle.seed(2024)
    paddle.seed(2024)
    # batch_size = 1
    total_q = S
    total_k = S
    batch_size = B
    num_head = H
    num_head_q = 12 * H
    head_size = D
    rank = cp_group.rank
    # total_k = total_q * 2
    if rank == 0:
        query = paddle.randn([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
        key = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        value = paddle.randn([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        o_grad = paddle.randn([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
    else:
        query = paddle.empty([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)
        key = paddle.empty([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        value = paddle.empty([batch_size, total_k, num_head, head_size], dtype=paddle.bfloat16)
        o_grad = paddle.empty([batch_size, total_q, num_head_q, head_size], dtype=paddle.bfloat16)

    # 广播到所有 rank
    paddle.distributed.broadcast(query, src=cp_group.ranks[0],group=cp_group)
    paddle.distributed.broadcast(key, src=cp_group.ranks[0],group=cp_group)
    paddle.distributed.broadcast(value, src=cp_group.ranks[0],group=cp_group)
    paddle.distributed.broadcast(o_grad,src = cp_group.ranks[0],group=cp_group)
    paddle.distributed.barrier(group=cp_group)
    query.stop_gradient = False
    key.stop_gradient = False
    value.stop_gradient = False

    causal = False
    balance_q_chunksize = 2048
    
    workload = get_q_workload(startend_row_indices, balance_q_chunksize, 128,128)
    
    # print(workload)
    total_workload = paddle.sum(workload,axis = 1)
    buckets, bucket_weights,cuts = assign_tasks_heap(workload.reshape(-1,2).cpu().numpy(), cp_size)
    buckets0 = buckets.copy()
    hcg = fleet.get_hybrid_communicate_group()
        
    ref_o = flashmask_attention(query, key, value, startend_row_indices, causal=causal)
    ref_o.backward(o_grad)
    x = query.reshape([batch_size, total_q, -1]).detach().contiguous()
    local_qs = []
    local_ks = []
    local_vs = []
    local_ograds =[]
    local_ref_os = []
    local_ref_grad_qs = []
    local_ref_grad_ks = []
    local_ref_grad_vs = []
    local_xs = []
    for(_, idx) in buckets[rank]:
        local_qs.append(query[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:,:])
        local_ks.append(key[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:,:])
        local_vs.append(value[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:,:])
        local_ograds.append(o_grad[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:,:])
        local_ref_os.append(ref_o[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:,:])
        local_ref_grad_qs.append(query.grad[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:,:])
        local_ref_grad_ks.append(key.grad[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:,:])
        local_ref_grad_vs.append(value.grad[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:,:])
        local_xs.append(x[:,idx * balance_q_chunksize:(idx+1) * balance_q_chunksize,:])
    
    local_q = paddle.concat(local_qs, axis=1).detach().contiguous()
    local_ref_q = local_q.detach()
    local_k = paddle.concat(local_ks, axis=1).detach().contiguous()
    local_v = paddle.concat(local_vs, axis=1).detach().contiguous()
    local_o_grad = paddle.concat(local_ograds, axis=1)
    local_ref_o = paddle.concat(local_ref_os, axis=1)
    local_ref_grad_q = paddle.concat(local_ref_grad_qs, axis=1)
    local_ref_grad_k = paddle.concat(local_ref_grad_ks, axis=1)
    local_ref_grad_v = paddle.concat(local_ref_grad_vs, axis=1)
    local_ref_x = paddle.concat(local_xs, axis=1).detach().contiguous()
    
    local_startend_row_indices, buckets = balance_flashmask_input( startend_row_indices, cp_size, rank,balance_chunk_size= balance_q_chunksize)
    assert buckets0 == buckets
    local_q = scatter_balance(query, group = cp_group, axis=1,mode = "balanced_swap", buckets = buckets).detach().contiguous()
    local_x = scatter_balance(x, group = cp_group, axis=1,mode = "balanced_swap", buckets = buckets)
    gather_x = all_gather_balance(local_x, group = cp_group, axis=1,mode = "balanced_swap", buckets = buckets)
    
    local_k.stop_gradient = False
    local_v.stop_gradient = False
    # local_q = query[:,rank * S // cp_size:(rank+1)  * S // cp_size,:,:].detach().contiguous()
    # local_o_grad = o_grad[:,rank * S // cp_size:(rank+1)  * S // cp_size,:,:]
    # print(local_o_grad)
    print(local_q.shape)
    print(local_startend_row_indices.shape)
    local_q.stop_gradient = False
    local_o = flashmask_attention_cp(local_q, local_k, local_v, local_startend_row_indices, causal=causal, mode="balance_q")
    local_ref_o1 = ref_o[:,rank * S // cp_size:(rank+1)  * S // cp_size,:,:]
    print(f"rank: {rank}")
    local_o.backward(local_o_grad)
    paddle.distributed.barrier(group=cp_group)
    
    if(rank == 0):
        print(f"total_workload: {total_workload}")
        print(bucket_weights)
        print(cuts)
        for i, bucket in enumerate(buckets):
            workload_sum = 0
            for item in bucket:
                workload_sum += item[0]
            print(f"Bucket {i+1}: {workload_sum}")
            print(f"Bucket {i+1}: {bucket}")
            print(f"Bucket {i+1}: {len(bucket)}")
        # x_np = buckets[0].numpy()
        # np.savetxt('buckets.txt', x_np.reshape(-1, x_np.shape[-1]), fmt='%d')
    # strict_check(local_o[:,:,0,0].flatten(), local_o1[:,:,0,0].flatten())
    # strict_check(local_o1[:,:,0,0].flatten(), local_ref_o[:,:,0,0].flatten())
    strict_check(local_q.flatten(), local_ref_q.flatten())
    strict_check(local_o.flatten(), local_ref_o.flatten())
    strict_check(local_k.grad.flatten(), local_ref_grad_k.flatten())
    strict_check(local_v.grad.flatten(), local_ref_grad_v.flatten())
    strict_check(local_q.grad.flatten(), local_ref_grad_q.flatten())
    strict_check(local_x.flatten(), local_ref_x.flatten())
    strict_check(gather_x, x)
    
    # startend_row_indices = regroup_chunks_by_buckets(startend_row_indices, buckets)
    # print(buckets)
    # if generate_mask_fn is not None:
    #     print("enter",generate_mask_fn)
    #     startend_row_indices = generate_mask_fn()
        # startend_row_indices, causal = generate_mask_fn(total_q)
        
    # print(startend_row_indices)
    # paddle.set_printoptions(precision=None, threshold=10000000, edgeitems=None, sci_mode=None, linewidth=None)

    # fwd_time, bwd_time = cp_flashmask_balance_bench(query, key, value, startend_row_indices, causal,o_grad,buckets[rank])
    fwd_time, bwd_time = 0,0
    paddle.device.synchronize()
    # out1.backward(o_grad1)
    # paddle.device.synchronize()
    
    # print("pypt2:")
    # print(startend_row_indices)
    total_time = fwd_time + bwd_time
    return fwd_time, bwd_time, total_time
    # with open("execution_times.txt", "a") as log_file:
    #     log_file.write(f"bsz: {batch_size},num_head_k: {num_head},num_head_q: {num_head * 4},hsz: {head_size},seqlen: {total_q}, flashattnv1: {flashattnv1_time:.6f}s, "
    #                     f"flashattnv2: {flashattnv2_time:.6f}s\n")
    # for x,y in [(out1,out),(dq1,query.grad),(dk1,key.grad),(dv1,value.grad)]:
    #     strict_check(x.flatten(), y.flatten())
    # for x,y in [(out1,out)]:
    #     strict_check(x.flatten(), y.flatten())
    
def strict_check(x, y):
    if isinstance(x, paddle.Tensor):
        if x.dtype == paddle.bfloat16 or x.dtype == "float16":
          # x = x.view("float16").numpy()
          x = x.cast("float32").numpy()
        else:
          x = x.numpy()
    else:
      assert False

    # if isinstance(y, torch.Tensor):
    #     if y.dtype == torch.bfloat16 or y.dtype == "bfloat16":
    #       # x = x.view("float16").numpy()
    #       y = y.to(torch.float32).detach().cpu().numpy()
    #     else:
    #       y = y.detach().cpu().numpy()

    if isinstance(y, paddle.Tensor):
        if y.dtype == paddle.bfloat16 or y.dtype == "float16":
          # y = y.view("float16").numpy()
          y = y.cast("float32").numpy()
        else:
          y = y.numpy()

    try:
        print(f"{x=}, {y=}")
        np.testing.assert_allclose(x.flatten(), y.flatten(),rtol=1e-2, atol=1e-2)
    except Exception as e:
        print('---------------')
        idx = np.where(~(x == y))
        print(f"fail idx: {idx=}")
        print(f"shape:'{x.shape}'")
        # print(f"fail idx:'{np.unique(idx[0])}'")
        print(x[idx])
        print(y[idx])
        raise e
    

def ele_check(x, y):
    if isinstance(x, paddle.Tensor):
        if x.dtype == paddle.bfloat16 or x.dtype == "bfloat16":
          # x = x.view("uint16").numpy()
          x = x.cast("float32").numpy()
        else:
          x = x.numpy()
    else:
      assert False

    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16 or y.dtype == "bfloat16":
          # x = x.view("uint16").numpy()
          y = y.to(torch.float32).detach().cpu().numpy()
        else:
          y = y.detach().cpu().numpy()

    # if isinstance(y, paddle.Tensor):
    #     if y.dtype == paddle.bfloat16 or y.dtype == "bfloat16":
    #       # y = y.view("uint16").numpy()
    #       y = y.cast("float32").numpy()
    #     else:
    #       y = y.numpy()

    try:
        print(f"{x=}, {y=}")
        np.testing.assert_allclose(np.sort(x.flatten()), np.sort(y.flatten()),rtol=1e-3, atol=1e-6)
    except Exception as e:
        print('---------------')
        idx = np.where(~(x == y))
        print(f"fail idx: {idx=}")
        print(f"shape:'{x.shape}'")
        # print(f"fail idx:'{np.unique(idx[0])}'")
        print(x[idx])
        print(y[idx])
        raise e

def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
    if startend_row_indices is None:
        return None
    bz, num_head, seq_len, bound_num = startend_row_indices.shape
    m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
    has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
    for bi in range(bz):
        for hi in range(num_head):
            for j in range(seq_len):
                downstart = startend_row_indices[bi, hi, j, 0]
                if has_end:
                    downend = startend_row_indices[bi, hi, j, 1]
                    m[bi, hi, downstart:downend, j] = -np.inf
                else:
                    m[bi, hi, downstart:, j] = -np.inf
                if causal:
                    m[bi, hi, :j, j] = -np.inf
                else:
                    if has_end:
                        upstart = startend_row_indices[bi, hi, j, 2]
                        upend = startend_row_indices[bi, hi, j, 3]
                        m[bi, hi, upstart:upend, j] = -np.inf
                    else:
                        upend = startend_row_indices[bi, hi, j, 1]
                        m[bi, hi, :upend, j] = -np.inf
    return m

def generate_none_mask(B, S, H, D, causal=True):
    return None, causal

def generate_ones_mask(B, S, H, D):
    startend_row_indices = paddle.zeros(
        shape=(B, H, S, 2), dtype="int32"
    )
    startend_row_indices[:,:,:,0]=S
    causal = False
    return startend_row_indices, causal

def generate_causal_mask(B,S,H,D):
    startend_row_indices = paddle.zeros(
        shape=(B, H, S, 1), dtype="int32"
    )
    startend_row_indices[:,:,:,0]=S
    causal = True
    return startend_row_indices, causal

def generate_sliding_window_mask(B, S, H, D, window_size=1024):
    startend_row_indices = paddle.arange(
        window_size, S + window_size, dtype="int32"
    ).reshape((1, 1, S, 1))
    startend_row_indices = paddle.clip(
        startend_row_indices, max=S
    ).repeat_interleave(B, 0)

    causal=True
    return startend_row_indices, causal

# def generate_causal_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
def generate_causal_document_mask(B,S,H,D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S, f"{total_seq_len=}, {S=}"
    padding = S - np.sum(doc_seq_lens)
    doc_seq_lens[-1] += padding
    seq_cusums = np.cumsum(doc_seq_lens)

    startend_row_indices = np.repeat(seq_cusums, doc_seq_lens)
    startend_row_indices = paddle.to_tensor(startend_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1))
    startend_row_indices = startend_row_indices.repeat_interleave(B, 0)
    
    causal = True
    return startend_row_indices, causal

def generate_upper_document_mask(B,S,H,D, doc_seq_lens=[2538, 1742, 3213],padding_size = 256):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    padding = S - np.sum(doc_seq_lens)

    up_right_row_indices = []

    cur_len_so_far = 0
    for i in range(len(doc_seq_lens)):
        up_right_row_indices.extend([cur_len_so_far] * doc_seq_lens[i])
        if i < len(doc_seq_lens) -1:
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    down_left_row_indices =  paddle.ones_like(up_right_row_indices) * (S - padding_size)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
    return startend_row_indices, causal

def generate_document_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
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
            cur_len_so_far += doc_seq_lens[i]
    if padding > 0:
        up_right_row_indices.extend([cur_len_so_far] * padding)
    
    down_left_row_indices = paddle.to_tensor(down_left_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    up_right_row_indices = paddle.to_tensor(up_right_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
    
    causal = False
    return startend_row_indices, causal

def generate_share_question_mask(B, S, H, D, doc_seq_lens=[2538, 1742, 3213]):
    total_seq_len = np.sum(doc_seq_lens)
    assert total_seq_len <= S
    assert len(doc_seq_lens) >= 3
    padding = S - total_seq_len

    startend_row_indices = [S] * doc_seq_lens[0]

    cur_len_so_far = doc_seq_lens[0]
    for idx in range(1, len(doc_seq_lens)):
        cur_len_so_far += doc_seq_lens[idx]
        startend_row_indices.extend([cur_len_so_far] * doc_seq_lens[idx])

    if padding > 0:
        startend_row_indices.extend([cur_len_so_far] * padding)
        
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
    down_left_start_row_indices[:global_token] = S
    down_left_start_row_indices =  down_left_start_row_indices.reshape((1, 1, S, 1)).repeat_interleave(B, 0)

    down_left_end_row_indices = paddle.full([S], S, dtype="int32").reshape((1, 1, S, 1)).repeat_interleave(B, 0)

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
        assert offset > last_offset
        start_row_indices.extend([S]*(offset-last_offset))
        end_row_indices.extend([S]*(offset-last_offset))

        start_row_indices.extend(list(range(offset, offset+maskout_len)))
        end_row_indices.extend([offset+maskout_len]*(maskout_len))

        last_offset = offset + maskout_len

    last_offset <= S
    start_row_indices.extend([S]*(S-last_offset))
    end_row_indices.extend([S]*(S-last_offset))

    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
    startend_row_indices = paddle.concat([start_row_indices, end_row_indices], axis=-1)

    causal = True
    return startend_row_indices, causal

#def generate_hash_sparse_mask(B, S, H, D, maskout_pair=[(1024, 538), (2358, 1700)]):
#    """
#    tuple(offset, maskout_len)
#    """
#    start_row_indices = []
#    end_row_indices  = []
#    last_offset = 0
#    for offset, maskout_len in maskout_pair:
#        assert offset > last_offset
#        start_row_indices.append([S]*(offset-last_offset))
#        end_row_indices.append([S]*(offset-last_offset))
#
#        start_row_indices.append(list(range(offset, offset+maskout_len)))
#        end_row_indices.append([offset+maskout_len]*(maskout_len))
#
#        last_offset = offset + maskout_len
#
#    last_offset <= S
#    start_row_indices.append([S]*(S-last_offset))
#    end_row_indices.append([S]*(S-last_offset))
#
#    start_row_indices = paddle.to_tensor(start_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
#    end_row_indices = paddle.to_tensor(end_row_indices, dtype=paddle.int32).reshape((1, 1, S, 1)).repeat_interleave(B, 0)
#    startend_row_indices = paddle.concat([down_left_row_indices, up_right_row_indices], axis=-1)
#
#    causal = False
#    return startend_row_indices, causal


def generate_random_eviction_mask(B, S, H, D, start_row=4096):
    np.random.seed(0)
    start_rows_list = []
    for bz_idx in range(B):
        for head_idx in range(H):
            start_rows = np.array([S+1] * S)
            mask_pos = np.random.choice(S-1, S - start_row, replace=False)
            index = np.arange(start_row, S)
            mask_pos = np.concatenate([mask_pos[mask_pos < index - 1], mask_pos[mask_pos >= index - 1]])
            start_rows[mask_pos] = index
            min_index = np.arange(1,S+1)
            start_rows = np.maximum(start_rows, min_index)
            start_rows_list.append(start_rows)
    startend_row_indices = paddle.to_tensor(start_rows_list, dtype=paddle.int32).reshape((B, H, S, 1))
    causal = True
    return startend_row_indices, causal

def gen_varlen(cu_seqlens):
  # 初始化mask
  mask = np.zeros((1, 1, 32768, 2), dtype=np.int32)
  mask[0, 0, :, 0] = np.arange(32768, dtype=np.int32)
  mask[0, 0, :, 1] = np.arange(32768, dtype=np.int32)

  # 按规则填充mask[..., 0]
  for i in range(len(cu_seqlens)-1):
      start = cu_seqlens[i]
      end = cu_seqlens[i+1]
      mask[0, 0, start:end, 0] = cu_seqlens[i+1]

  # mask[..., 1] = i
  mask[0, 0, :, 1] = np.arange(32768, dtype=np.int32)

  # 转为 paddle tensor
  mask_tensor = paddle.to_tensor(mask, dtype='int32')
  print(mask_tensor)
  return mask_tensor


def gen_varlen_causal(cu_seqlens):
  # 初始化mask
  mask = np.zeros((1, 1, 32768, 1), dtype=np.int32)
  mask[0, 0, :, 0] = np.arange(32768, dtype=np.int32)
#   mask[0, 0, :, 1] = np.arange(32768, dtype=np.int32)

  # 按规则填充mask[..., 0]
  for i in range(len(cu_seqlens)-1):
      start = cu_seqlens[i]
      end = cu_seqlens[i+1]
      mask[0, 0, start:end, 0] = cu_seqlens[i+1]

  # mask[..., 1] = i
#   mask[0, 0, :, 1] = np.arange(32768, dtype=np.int32)

  # 转为 paddle tensor
  mask_tensor = paddle.to_tensor(mask, dtype='int32')
  print(mask_tensor)
  return mask_tensor

def main(examples: List[str] = ["all"], dtype='bf16'):
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    total_length = 0
    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    doc_seq_lens_list = []
    rank = paddle.distributed.get_rank()
    
    
            
    #doc_seq_lens_list = doc_seq_lens_list[::-1]
    for D in [128]:
        H = 1
        # print(doc_seq_lens_list)
        results = []
        for idx in range(0,50):
            B = 2
            startend_row_indices = paddle.load(f'/root/paddlejob/workspace/env_run/xiehaoyang/flashmask/flashmask-cp/cp_balance/dump_32k_startend_row_indices/startend_row_indices_{idx}.pdparams')
            startend_row_indices = startend_row_indices.repeat(B, 1, 1, 1)
            print(startend_row_indices)
            S = 32768
            print(f"{B}_{S}_{H}_{D}_{idx}_{dtype}")


            available_examples = {
                # "Full": lambda: test_cp_famask(generate_mask_fn=partial(generate_none_mask, causal=False), B=B, S=S, H=H, D=D, dtype=dtype),
                # "Causal": lambda: test_cp_famask(generate_mask_fn=partial(generate_none_mask, causal=True), B=B, S=S, H=H, D=D, dtype=dtype),
                # "Sliding Window": lambda: test_cp_famask(generate_mask_fn=partial(generate_sliding_window_mask, window_size=int(S*0.0625)), B=B, S=S, H=H, D=D, dtype=dtype),
                # "Causal Document Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_causal_document_mask, doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                "Document Mask": lambda: test_cp_famask(startend_row_indices = startend_row_indices, B=B, S=S, H=H, D=D, dtype=dtype),
                # "Share Question Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_share_question_mask, doc_seq_lens=share_qa_docs), B=B, S=S, H=H, D=D, dtype=dtype),
                # "Global Sliding Window": lambda: test_cp_famask(generate_mask_fn=partial(generate_global_sliding_window_mask, global_token=16, window_size=(int(S*0.0625), int(S*0.0625))), B=B, S=S, H=H, D=D, dtype=dtype),
                # "Causal Blockwise Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_causal_blockwise_mask, doc_seq_lens=doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                # "Prefix LM Document Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_prefix_lm_document_mask, doc_seq_lens=prefix_doc_seq_lens), B=B, S=S, H=H, D=D, dtype=dtype),
                # "Prefix LM Causal Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_prefix_lm_causal_mask, prefix_length=int(S*0.5)), B=B, S=S, H=H, D=D, dtype=dtype),
                # "QK-sparse Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_qk_sparse_mask, maskout_pair=maskout_pair), B=B, S=S, H=H, D=D, dtype=dtype),
                # "Random Eviction Mask": lambda: test_cp_famask(generate_mask_fn=partial(generate_random_eviction_mask, start_row=S//2), B=B, S=S, H=H, D=D, dtype=dtype),
            }

            global total_num
            total_num = len(available_examples)

            if "all" in examples:
                ex_to_run = list(available_examples.keys())
            else:
                ex_to_run = examples

            for ex in ex_to_run:
                if ex in available_examples:
                    print(ex)
                    fw_time, bw_time, total_time = available_examples[ex]()
                    results.append([idx, f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}"])
                    print(fw_time, bw_time)
                else:
                    print(f"Warning: Unknown example key '{ex}'. Skipping.")
            # if(idx >= 3):
                # return

        # print(f'avg_fwd_time:{sum([float(result[1][:-1]) for result in results]) / len(results)} avg_bwd_time:{sum([float(result[2][:-1]) for result in results]) / len(results)}')
        headers = [
            "Idx",
            "FW Time (ms)",
            "BW Time (ms)",
            "TOTAL Time (ms)",
        ]
        print(
            tabulate(
                results,
                headers=headers,
                tablefmt="grid",
            )
        )
        content2=tabulate(results, headers=headers, tablefmt="tsv")
        os.makedirs(f"{dtype}_dist_test_dump", exist_ok=True)
        text_file = open(f"{dtype}_dist_test_dump/flashmask_{rank}_{B}_{S}_{H}_{D}.csv","w")
        text_file.write(content2)
        text_file.close()
        paddle.device.synchronize()
        paddle.distributed.barrier()

        print(f'avg_fwd_time:{sum([float(result[1][:-1]) for result in results]) / len(results)} avg_bwd_time:{sum([float(result[2][:-1]) for result in results]) / len(results)}')
                # assert False

if __name__ == "__main__":
    try:
        from jsonargparse import ArgumentParser
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
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

    args = parser.parse_args()
    main(**vars(args))
