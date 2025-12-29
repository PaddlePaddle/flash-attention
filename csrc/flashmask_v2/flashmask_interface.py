# Copyright (c) 2023, Tri Dao.

from typing import Optional, Union, List, Tuple

import torch
import torch.nn as nn

# isort: off
# We need to import the CUDA kernels after importing torch
import flashmask._C # Registers operators with PyTorch

# isort: on

flashmask_cuda = torch.ops.flashmask

class FlashMaskFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, startend_row_indices, block_mask, softmax_scale, causal, window_size_left, window_size_right, softcap, deterministic):
        # 1. 调用 C++ 前向算子
        # 使用关键字参数调用，避免参数位置对不上的问题
        out, lse = flashmask_cuda.fwd(
            q=q,
            k=k,
            v=v,
            startend_row_indices=startend_row_indices,
            block_mask=block_mask,
            softmax_scale=softmax_scale,
            is_causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softcap=softcap
            # 其他参数 C++ 端有默认值 None/0，这里不用传
        )
        
        # 2. 保存用于反向传播的 Tensor 和参数
        ctx.save_for_backward(q, k, v, out, lse, startend_row_indices, block_mask)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        # 1. 取出保存的 Tensor
        q, k, v, out, lse, startend_row_indices, block_mask = ctx.saved_tensors
        
        # 2. 调用 C++ 反向算子
        # 注意：这里的参数名必须和 flash_api.cpp 中 m.def("bwd(...") 定义的一致
        dq, dk, dv = flashmask_cuda.bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            out=out,
            softmax_lse=lse,
            dq=None, # 可选，C++ 会自动分配
            dk=None,
            dv=None,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            startend_row_indices=startend_row_indices, # 传入 mask
            block_mask=block_mask,                     # 传入 block mask
            max_seqlen_q=None,
            max_seqlen_k=None,
            softmax_scale=ctx.softmax_scale,
            is_causal=ctx.causal,
            window_size_left=ctx.window_size_left,
            window_size_right=ctx.window_size_right,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
            sm_margin=0
        )
        
        # 3. 返回梯度
        # 顺序必须对应 forward 的输入: 
        # (q, k, v, startend_row_indices, block_mask, softmax_scale, causal, window_size_left, window_size_right, softcap, deterministic)
        return dq, dk, dv, None, None, None, None, None, None, None, None

def flashmask_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    startend_row_indices: Optional[torch.Tensor] = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: Optional[Union[int, Tuple[int, int]]] = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: Optional[torch.Tensor] = None, # Paddle 中通常传入 Tensor，PyTorch 中如果需要手动控制随机种子通常传 Generator 或 tuple
    rng_name: str = "",          # Paddle 特有：用于选择随机数生成器，PyTorch 中通常忽略或使用 generator 参数
    training: bool = True,
    name: Optional[str] = None,  # Paddle 特有：静态图命名，PyTorch 动态图不需要
    softmax_scale: Optional[float] = None,
    block_mask: Optional[torch.Tensor] = None,
):
    """
    FlashMask: Official Implementation (PyTorch Port)
    
    Args:
        query (torch.Tensor): [batch_size, seq_len, num_heads, head_dim]
        key (torch.Tensor): [batch_size, seq_len, num_heads, head_dim]
        value (torch.Tensor): [batch_size, seq_len, num_heads, head_dim]
        startend_row_indices (torch.Tensor): 
            A column-wise sparse attention mask row indices tensor.
            Shape: [batch_size, num_heads, seq_len, {1, 2, 4}]
            Dtype: torch.int32
        ...
    """
    r"""
    FlashMask: Official Implementation

    This module provides the official implementation of the FlashMask algorithm as described in the paper. For more details, please refer to the paper available at: https://arxiv.org/abs/2410.01359.

    The core equation utilized in FlashMask is as follows:

    .. math::

        \text{result} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}} + M\right) \cdot V

    In this equation:

        - ``Q``, ``K``, and ``V`` are the input tensors to the attention module.
        - All these tensors share the same dimensions.
        - ``d`` denotes the size of the last dimension of these tensors.
        - ``M`` represents the column-wise sparse mask introduced by FlashMask.

    Args:
        query (torch.Tensor):  The query tensor in the attention module.
            A 4-D tensor with shape [batch_size, q_seq_len, num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        key (torch.Tensor): The key tensor in the attention module.
            A 4-D tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        value (torch.Tensor): The value tensor in the attention module.
            A 4-D tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        startend_row_indices (torch.Tensor):
            A column-wise sparse attention mask row indices tensor.
            A 4-D tensor with shape [batch_size, k_num_heads, k_seq_len, {1, 2, 4}].
            The dtype must be int32. k_num_heads can be 1 or the same as key's num_heads. When num_heads is 1, it will be broadcast to match key's num_heads.
            Depending on the value of the causal parameter, startend_row_indices can take different shapes and meanings.

            - When `causal=True` and the shape is [batch_size, k_num_heads, k_seq_len, 1],
              indicating unidirectional attention. The value represents the starting row index of the left
              lower triangular mask in the dense mask. The value startend_row_indices[..., 0] indicates that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) will be masked.
            - When `causal=True` and the shape is [batch_size, k_num_heads, k_seq_len, 2],
              indicating unidirectional attention. The values represent the starting and ending row indices of
              the left lower triangular mask in the dense mask. The values startend_row_indices[..., 0:2] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) but above the startend_row_indices[..., 1]-th row (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, k_num_heads, k_seq_len, 2],
              indicating bidirectional attention. The values represent the starting row index of the left
              lower triangular mask and the ending row index of the right upper triangular mask in the dense mask. The values startend_row_indices[..., 0:2] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) will be masked, and elements in the upper right triangle starting from the startend_row_indices[..., 1]-th row upwards (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, k_num_heads, k_seq_len, 4] ,
              indicating bidirectional attention. The values represent the start and end row indices of the
              left lower triangular mask and the start and end row indices of the right upper triangular mask in the dense mask. The values startend_row_indices[..., 0:4] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) but above the startend_row_indices[..., 1] row (exclusive) will be masked, and elements in the upper right triangle starting from the startend_row_indices[..., 2]-th row downwards (inclusive) but above the startend_row_indices[..., 3] row (exclusive) will be masked.

        dropout (float): The dropout ratio. Default is 0.0.
        causal (bool): Whether to enable causal mode. Default is False.
        window_size (int|tuple, optional): Indicates the window size of sliding window local attention.
            If causal mode is enabled, Query at position i will only attend to keys between [i - window_size, i] or [i - window_size[0], i].
            If causal mode is disabled, Query at position i will only attend to keys between [i - window_size, i + window_size] or [i - window_size[0], i + window_size[1]].
        return_softmax_lse (bool): Whether to return the log-sum-exp of the softmax. Default is False.
        return_seed_offset (bool): Whether to return the random seed offset. Default is False.
        fixed_seed_offset (torch.Tensor, optional): With fixed seed, offset for dropout mask.
        rng_name (str): The name to select Generator. (Note: In PyTorch, this is typically unused or replaced by a torch.Generator object, kept here for interface compatibility).
        training (bool): Whether the module is in training mode. Default is True.
        name (str, optional): Name of the operation. Default is None. (Note: Unused in PyTorch).
        block_mask (torch.Tensor, optional):
            A 4-D integer mask tensor indicating whether each block in the attention matrix should be kept or masked. Must be used together with flashmask.
            The shape should be [batch_size, num_heads, blocklen_q, blocklen_k], where:

            blocklen_q = ceil(seqlen_q / 128), i.e., block_mask.shape[2] must be (seqlen_q + 127) // 128
            blocklen_k = ceil(seqlen_k / 128), i.e., block_mask.shape[3] must be (seqlen_k + 127) // 128
            block_mask.shape[1] (number of heads) must match the num_heads dimension of the flashmask
            Both seqlen_q and seqlen_k must be less than or equal to 128 * 1024
            The dtype should be int32, and each element should be either 0 or 1.
            A value of 1 indicates that the corresponding block is kept (not masked), while 0 means the block is masked.

            Usage Notes:

            Only supported when blockdim_q = blockdim_k = 128 now.
            Only supported when headdim = 128 now.
            This argument must be provided together with flashmask.
            The mask will be applied at the block level: each [i, j] position in block_mask controls whether the corresponding [128 x 128] block in the attention matrix is masked.
            Any mismatch in expected shape or head dimension will raise an error.


    Returns
        torch.Tensor. The computed attention result with the same shape as the input `query`.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Hint:
        This API supports GQA.

    To convert FlashMask's `startend_row_indices` to `dense_mask`, use the code below:

    .. code-block:: python

        >>> import torch
        >>> import numpy as np
        >>> def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
        ...     if startend_row_indices is None:
        ...         return None
        ...     bz, num_head, seq_len, bound_num = startend_row_indices.shape
        ...     m = torch.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
        ...     has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
        ...     for bi in range(bz):
        ...         for hi in range(num_head):
        ...             for j in range(seq_len):
        ...                 downstart = startend_row_indices[bi, hi, j, 0]
        ...                 if has_end:
        ...                     downend = startend_row_indices[bi, hi, j, 1]
        ...                     m[bi, hi, downstart:downend, j] = -np.inf
        ...                 else:
        ...                     m[bi, hi, downstart:, j] = -np.inf
        ...                 if causal:
        ...                     m[bi, hi, :j, j] = -np.inf
        ...                 else:
        ...                     if has_end:
        ...                         upstart = startend_row_indices[bi, hi, j, 2]
        ...                         upend = startend_row_indices[bi, hi, j, 3]
        ...                         m[bi, hi, upstart:upend, j] = -np.inf
        ...                     else:
        ...                         upend = startend_row_indices[bi, hi, j, 1]
        ...                         m[bi, hi, :upend, j] = -np.inf
        ...     return m

    For `Causal Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> startend_row_indices = torch.tensor([8]*10, dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> print(startend_row_indices)
        tensor([[[[8],
                  [8],
                  [8],
                  [8],
                  [8],
                  [8],
                  [8],
                  [8],
                  [8],
                  [8]]]], device='cuda:0', dtype=torch.int32)
        >>> # doctest: -SKIP


    For `Sliding Window Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> startend_row_indices = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> print(startend_row_indices)
        tensor([[[[ 3],
                  [ 4],
                  [ 5],
                  [ 6],
                  [ 7],
                  [ 8],
                  [ 9],
                  [10],
                  [10],
                  [10]]]], device='cuda:0', dtype=torch.int32)
        >>> # doctest: -SKIP

    For `Causal Document Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> startend_row_indices = torch.tensor([4, 4, 4, 4, 7, 7, 7, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> print(startend_row_indices)
        tensor([[[[ 4],
                  [ 4],
                  [ 4],
                  [ 4],
                  [ 7],
                  [ 7],
                  [ 7],
                  [10],
                  [10],
                  [10]]]], device='cuda:0', dtype=torch.int32)
        >>> # doctest: -SKIP

    For `Document Mask`, where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> LTS = torch.tensor([4, 4, 4, 4, 7, 7, 7, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> UTE = torch.tensor([0, 0, 0, 0, 4, 4, 4, 7, 7, 7], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> startend_row_indices = torch.cat([LTS, UTE], dim=-1)
        >>> print(startend_row_indices)
        tensor([[[[ 4,  0],
                  [ 4,  0],
                  [ 4,  0],
                  [ 4,  0],
                  [ 7,  4],
                  [ 7,  4],
                  [ 7,  4],
                  [10,  7],
                  [10,  7],
                  [10,  7]]]], device='cuda:0', dtype=torch.int32)
        >>> # doctest: -SKIP

    For `Share Question Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
          [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> startend_row_indices = torch.tensor([10, 10, 10, 10, 7, 7, 7, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> print(startend_row_indices)
        tensor([[[[10],
                  [10],
                  [10],
                  [10],
                  [ 7],
                  [ 7],
                  [ 7],
                  [10],
                  [10],
                  [10]]]], device='cuda:0', dtype=torch.int32)
        >>> # doctest: -SKIP

    For `Global + Sliding Window Mask`, where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')

       [[[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
          [1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
          [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
          [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]]]])

        >>> import torch
        >>> LTS = torch.tensor([10, 10, 4, 5, 6, 7, 8, 9, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> LTE = torch.tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> UTS = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2, 2, 2], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> UTE = torch.tensor([0, 0, 0, 0, 3, 4, 5, 6, 7, 8], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> startend_row_indices = torch.cat([LTS, LTE, UTS, UTE], dim=-1)
        >>> print(startend_row_indices)
        tensor([[[[10, 10,  0,  0],
                  [10, 10,  0,  0],
                  [ 4, 10,  0,  0],
                  [ 5, 10,  0,  0],
                  [ 6, 10,  2,  3],
                  [ 7, 10,  2,  4],
                  [ 8, 10,  2,  5],
                  [ 9, 10,  2,  6],
                  [10, 10,  2,  7],
                  [10, 10,  2,  8]]]], device='cuda:0', dtype=torch.int32)
        >>> # doctest: -SKIP

    For `Causal Blockwise Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> LTS = torch.tensor([4, 4, 4, 4, 10, 10, 10, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> LTE = torch.tensor([7, 7, 7, 7, 10, 10, 10, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> startend_row_indices = torch.cat([LTS, LTE], dim=-1)
        >>> print(startend_row_indices)
        tensor([[[[ 4,  7],
                  [ 4,  7],
                  [ 4,  7],
                  [ 4,  7],
                  [10, 10],
                  [10, 10],
                  [10, 10],
                  [10, 10],
                  [10, 10],
                  [10, 10]]]], device='cuda:0', dtype=torch.int32)
        >>> # doctest: -SKIP

    For `Prefix LM Document Mask`, where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> LTS = torch.tensor([3, 3, 3, 5, 5, 10, 10, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> UTE = torch.tensor([0, 0, 2, 3, 3, 5, 5, 7, 8, 9], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> startend_row_indices = torch.cat([LTS, UTE], dim=-1)
        >>> print(startend_row_indices)
        tensor([[[[ 3,  0],
                  [ 3,  0],
                  [ 3,  2],
                  [ 5,  3],
                  [ 5,  3],
                  [10,  5],
                  [10,  5],
                  [10,  7],
                  [10,  8],
                  [10,  9]]]], device='cuda:0', dtype=torch.int32)
        >>> # doctest: -SKIP

    For `Prefix LM Causal Mask`, where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> LTS = torch.tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> UTE = torch.tensor([0, 0, 0, 0, 0, 5, 6, 7, 8, 9], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> startend_row_indices = torch.cat([LTS, UTE], dim=-1)
        >>> print(startend_row_indices)
        tensor([[[[10,  0],
                  [10,  0],
                  [10,  0],
                  [10,  0],
                  [10,  0],
                  [10,  5],
                  [10,  6],
                  [10,  7],
                  [10,  8],
                  [10,  9]]]], device='cuda:0', dtype=torch.int32)

    For `QK-sparse Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import torch
        >>> LTS = torch.tensor([10, 10, 2, 3, 4, 5, 6, 7, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> LTE = torch.tensor([10, 10, 5, 5, 5, 5, 8, 8, 10, 10], dtype=torch.int32).reshape(1, 1, 10, 1).cuda()
        >>> startend_row_indices = torch.cat([LTS, LTE], dim=-1)
        >>> print(startend_row_indices)
        tensor([[[[10, 10],
                  [10, 10],
                  [ 2,  5],
                  [ 3,  5],
                  [ 4,  5],
                  [ 5,  5],
                  [ 6,  8],
                  [ 7,  8],
                  [10, 10],
                  [10, 10]]]], device='cuda:0', dtype=torch.int32)

        >>> # doctest: -SKIP
    """

    if window_size is not None:
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        sq = query.shape[1]
        bsz = query.shape[0]
        assert startend_row_indices is None, (
            "can't use window_size with startend_row_indices"
        )
        
        # 关键点：获取输入 tensor 的 device，确保新建的 tensor 在同一设备上
        device = query.device

        if causal:
            # paddle.arange -> torch.arange
            startend_row_indices = torch.arange(
                window_size[0] + 1, sq + window_size[0] + 1, dtype=torch.int32, device=device
            ).reshape(1, 1, sq, 1)
            
            # paddle.clip -> torch.clamp
            startend_row_indices = torch.clamp(
                startend_row_indices, max=sq
            ).repeat_interleave(bsz, dim=0)

        else:
            # paddle.empty -> torch.empty
            startend_row_indices = torch.empty((1, 1, sq, 2), dtype=torch.int32, device=device)
            startend_row_indices[0, 0, :, 0] = torch.arange(
                window_size[0] + 1, sq + window_size[0] + 1, dtype=torch.int32, device=device
            )
            startend_row_indices[0, 0, :, 1] = torch.arange(
                -window_size[1], sq - window_size[1], dtype=torch.int32, device=device
            )
            # paddle.clip -> torch.clamp
            startend_row_indices = torch.clamp(
                startend_row_indices, min=0, max=sq
            ).repeat_interleave(bsz, dim=0)

    if block_mask is not None:
        # xhy: can set a full startend_row_indices for block_mask_attn when using block_mask_attn?
        assert startend_row_indices is not None, (
            "must provide startend_row_indices when using block_mask_attn"
        )
    if startend_row_indices is None:
        raise ValueError(
            "startend_row_indices cannot be None when calling flashmask_attention. "
            "This API is dedicated to FlashMask functionality. "
            "If you intended to use standard FlashAttention (without sparse mask), "
            "please import and use 'flash_attn_func' from the 'flash_attn' library directly."
        )

    else:
        assert startend_row_indices.dtype == torch.int32, (
            f"startend_row_indices.dtype must be torch.int32, but got {startend_row_indices.dtype}"
        )
        assert len(startend_row_indices.shape) == 4, (
            f"startend_row_indices rank must be 4, but got {startend_row_indices.shape}"
        )
        assert startend_row_indices.shape[0] == key.shape[0], (
            f"startend_row_indices.shape[0] must be equal to batch_size, but got {startend_row_indices.shape[0]} and {key.shape[0]}"
        )
        assert startend_row_indices.shape[2] == key.shape[1], (
            f"startend_row_indices.shape[2] must be equal to seqlen_k, but got {startend_row_indices.shape[2]} and {key.shape[2]}"
        )
        assert startend_row_indices.shape[1] in [1, key.shape[2]], (
            "startend_row_indices head_num must be equal to 1(broadcast) or head_num_k."
        )

        if block_mask is not None:
            assert block_mask.dtype == torch.int32, (
                f"block_mask.dtype must be torch.int32, but got {block_mask.dtype}"
            )
            assert block_mask.shape[0] == key.shape[0], (
                f"block_mask.shape[0] must be equal to batch_size, but got {block_mask.shape[0]} and {key.shape[0]}"
            )
            assert block_mask.shape[1] == startend_row_indices.shape[1], (
                f"block_mask.shape[1] must be equal to startend_row_indices.shape[1], but got {block_mask.shape[1]} and {key.shape[2]}"
            )
            assert block_mask.shape[2] == (query.shape[1] + 127) // 128, (
                "block_size must be 128 when using block_mask_attn"
            )
            assert block_mask.shape[3] == (key.shape[1] + 127) // 128, (
                "block_size must be 128 when using block_mask_attn"
            )
            assert key.shape[3] == 128, (
                "headdim must be 128 when using block_mask_attn"
            )

        if causal:
            if startend_row_indices.shape[-1] == 1:
                has_end = False
            elif startend_row_indices.shape[-1] == 2:
                has_end = True
            else:
                raise ValueError(
                    f"Invalid shape of startend_row_indices, when causal is True, the last dimension should be either 1 or 2 but got {startend_row_indices.shape[-1]}"
                )
        else:
            if startend_row_indices.shape[-1] == 2:
                has_end = False
            elif startend_row_indices.shape[-1] == 4:
                has_end = True
            else:
                raise ValueError(
                    f"Invalid shape of startend_row_indices, when causal is False, the last dimension should be either 2 or 4 but got {startend_row_indices.shape[-1]}"
                )

        current_device_type = query.device.type
        flag_cudnn_deterministic = torch.are_deterministic_algorithms_enabled()

        # 疑惑
        if current_device_type == "cuda":
            major, _ = torch.cuda.get_device_capability(query.device)
            flag_flash_attn_version = 3 if major >= 9 else 2
        else:
            flag_flash_attn_version = 2

        if (
            "xpu" not in current_device_type
            and flag_cudnn_deterministic
        ):
            assert block_mask is None, (
                " blockmask attention no supports deterministic now ."
            )
        
        if "xpu" in current_device_type:
            fa_version = 2
        elif (
            flag_flash_attn_version == 3
            and flag_cudnn_deterministic
            and query.shape[3] > 128
        ):
            fa_version = 2
        else:
            fa_version = flag_flash_attn_version

        if fa_version == 2:
            raise NotImplementedError("FlashMask v1 is not supported. Please use FlashMask v2.")
        
        elif fa_version == 3:
            assert dropout == 0.0, (
                "flashmask_attention_v2 does not support dropout"
            )
            assert not return_seed_offset, (
                "flashmask_attention_v2 does not support return seed_offset"
            )
            assert fixed_seed_offset is None, (
                "flashmask_attention_v2 does not support setting seed_offset"
            )
            # 在 PyTorch 接口中 rng_name 通常默认为 "" 或 None，保留此检查以确保兼容性
            assert rng_name == "", (
                "flashmask_attention_v2 does not support setting rng_name"
            )
            assert training, (
                "flashmask_attention_v2 does not support setting training to False"
            )
            
            assert name is None, (
                "flashmask_attention_v2 does not support setting name"
            )

            if softmax_scale is None:
                softmax_scale = query.shape[-1] ** (-0.5)

            window_size_left = -1
            window_size_right = -1
            softcap = 0.0
            deterministic = flag_cudnn_deterministic

            # 调用 PyTorch 注册的算子 (flashmask_cuda.flashmask_attention_v2)
            (
                out,
                result_softmax_lse,
            ) = FlashMaskFunc.apply(
                query,
                key,
                value,
                startend_row_indices,
                block_mask,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                softcap,
                deterministic
            )
        else:
            raise ValueError(f"Invalid flash attention version: {fa_version}")

    outputs = [out]
    if return_softmax_lse:
        outputs += [result_softmax_lse]
    if return_seed_offset:
        outputs += [result_seed_offset]
        
    if len(outputs) == 1:
        return outputs[0]
    else:
        return tuple(outputs)

    