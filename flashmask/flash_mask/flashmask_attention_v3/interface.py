import paddle
from paddle import Tensor
from typing import Optional

# [BQW_CHANGE] 使用 paddle._C_ops._run_custom_op 调用 PD_BUILD_OP 注册的自定义算子
# Paddle CUDAExtension 生成的自定义算子通过全局注册表访问
# SO 文件在 flash_mask/__init__.py 中通过 load_op_meta_info_and_register_op 加载
from paddle import _C_ops

def flashmask_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    startend_row_indices: Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: Tensor | None = None,
):
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
        query (Tensor):  The query tensor in the attention module.
            A 4-D tensor with shape [batch_size, q_seq_len, num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        key (Tensor): The key tensor in the attention module.
            A 4-D tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        value (Tensor): The value tensor in the attention module.
            A 4-D tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        startend_row_indices(Tensor):
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
        fixed_seed_offset(Tensor, optional): With fixed seed, offset for dropout mask.
        rng_name (str): The name to select Generator.
        training (bool): Whether the module is in training mode. Default is True.
        name (str, optional): Name of the operation. Default is None. Normally, users do not need to set this property.
            For more information, refer to :ref:`api_guide_Name` .
        block_mask (tensor, optional):
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
        Tensor. The computed attention result with the same shape as the input `query`.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Hint:
        This API supports GQA.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle
            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 10, 2, 32),dtype="bfloat16") # shape: [batch_size, seq_len, num_heads, head_dim]
            >>> k = paddle.rand((1, 10, 2, 32),dtype="bfloat16") # shape: [batch_size, seq_len, num_heads, head_dim]
            >>> v = paddle.rand((1, 10, 2, 32),dtype="bfloat16") # shape: [batch_size, seq_len, num_heads, head_dim]
            >>> startend_row_indices = paddle.to_tensor([8]*10 + [5]*10, dtype="int32").reshape([1, 2, 10, 1])
            >>> output = paddle.nn.functional.flashmask_attention(q, k, v, startend_row_indices, causal=True)
            >>> print(output)
            Tensor(shape=[1, 10, 2, 32], dtype=bfloat16, place=Place(gpu:0), stop_gradient=True,
                [[[[0.82421875, 0.27539062, 0.80859375, 0.98046875, 0.00251770,
                    0.41992188, 0.17285156, 0.11767578, 0.42773438, 0.31250000,
                    0.34570312, 0.70312500, 0.29296875, 0.44531250, 0.51562500,
                    0.96093750, 0.85546875, 0.15625000, 0.34765625, 0.98437500,
                    0.96484375, 0.45312500, 0.33593750, 0.56640625, 0.07714844,
                    0.43750000, 0.83984375, 0.66796875, 0.93750000, 0.24804688,
                    0.51171875, 0.55468750],
                    [0.54687500, 0.74609375, 0.43164062, 0.32421875, 0.10693359,
                    0.37304688, 0.53906250, 0.17187500, 0.57421875, 0.75000000,
                    0.13378906, 0.57031250, 0.19531250, 0.01403809, 0.29101562,
                    0.14257812, 0.07568359, 0.88671875, 0.75390625, 0.17089844,
                    0.87109375, 0.93359375, 0.89843750, 0.58203125, 0.75390625,
                    0.27539062, 0.67968750, 0.24804688, 0.57812500, 0.67578125,
                    0.92578125, 0.98046875]],

                    [[0.59765625, 0.62890625, 0.62109375, 0.75781250, 0.03295898,
                    0.64062500, 0.27929688, 0.20800781, 0.72265625, 0.52343750,
                    0.53125000, 0.61718750, 0.57421875, 0.56640625, 0.65625000,
                    0.48242188, 0.68359375, 0.42968750, 0.26562500, 0.86718750,
                    0.83203125, 0.40820312, 0.38281250, 0.59765625, 0.43945312,
                    0.22851562, 0.86328125, 0.51562500, 0.89453125, 0.62500000,
                    0.50390625, 0.67968750],
                    [0.34765625, 0.61328125, 0.58593750, 0.60156250, 0.43164062,
                    0.41601562, 0.71093750, 0.59765625, 0.53515625, 0.78125000,
                    0.13867188, 0.30664062, 0.48828125, 0.04394531, 0.24316406,
                    0.18847656, 0.10644531, 0.71093750, 0.69140625, 0.35937500,
                    0.44531250, 0.81640625, 0.44140625, 0.64062500, 0.81640625,
                    0.61328125, 0.72265625, 0.53125000, 0.49414062, 0.59765625,
                    0.54296875, 0.61328125]],

                    [[0.65234375, 0.47656250, 0.71875000, 0.64843750, 0.23828125,
                    0.61328125, 0.29101562, 0.26562500, 0.54296875, 0.60937500,
                    0.67187500, 0.67578125, 0.64062500, 0.41406250, 0.47656250,
                    0.40820312, 0.66406250, 0.39453125, 0.39453125, 0.62109375,
                    0.58593750, 0.31054688, 0.31835938, 0.45703125, 0.52343750,
                    0.43164062, 0.64453125, 0.49804688, 0.82812500, 0.48242188,
                    0.38476562, 0.59375000],
                    [0.44921875, 0.62109375, 0.50390625, 0.51562500, 0.51953125,
                    0.57812500, 0.78515625, 0.73437500, 0.60546875, 0.55078125,
                    0.30273438, 0.23339844, 0.60546875, 0.33007812, 0.23242188,
                    0.30468750, 0.34570312, 0.70703125, 0.72656250, 0.58593750,
                    0.40234375, 0.62109375, 0.62109375, 0.69531250, 0.66796875,
                    0.51562500, 0.45898438, 0.67968750, 0.48828125, 0.50000000,
                    0.54687500, 0.71875000]],

                    [[0.67578125, 0.50000000, 0.58203125, 0.62109375, 0.43554688,
                    0.69531250, 0.30273438, 0.24023438, 0.57812500, 0.63671875,
                    0.51171875, 0.52734375, 0.60546875, 0.45507812, 0.42382812,
                    0.46093750, 0.55859375, 0.34960938, 0.39453125, 0.57031250,
                    0.55078125, 0.47265625, 0.24609375, 0.51953125, 0.46093750,
                    0.49218750, 0.49609375, 0.60156250, 0.76953125, 0.57421875,
                    0.40429688, 0.57031250],
                    [0.45703125, 0.71093750, 0.58984375, 0.43164062, 0.54296875,
                    0.57031250, 0.72265625, 0.61328125, 0.64453125, 0.50781250,
                    0.28125000, 0.19531250, 0.60546875, 0.40625000, 0.18554688,
                    0.33203125, 0.40039062, 0.58593750, 0.79687500, 0.45507812,
                    0.32812500, 0.58203125, 0.70703125, 0.64453125, 0.53906250,
                    0.57421875, 0.48828125, 0.53515625, 0.49804688, 0.50000000,
                    0.48437500, 0.55468750]],

                    [[0.64453125, 0.43164062, 0.54687500, 0.53125000, 0.42187500,
                    0.71484375, 0.30273438, 0.21484375, 0.50390625, 0.69531250,
                    0.58203125, 0.51562500, 0.61328125, 0.41992188, 0.40039062,
                    0.46679688, 0.58984375, 0.39062500, 0.41992188, 0.49023438,
                    0.47851562, 0.47070312, 0.30078125, 0.50390625, 0.47656250,
                    0.44921875, 0.43164062, 0.63671875, 0.78125000, 0.60156250,
                    0.48242188, 0.58203125],
                    [0.52343750, 0.69921875, 0.58984375, 0.35156250, 0.49218750,
                    0.58593750, 0.71093750, 0.59375000, 0.66406250, 0.49414062,
                    0.24023438, 0.18554688, 0.66796875, 0.50000000, 0.23144531,
                    0.29882812, 0.49414062, 0.57031250, 0.70312500, 0.42773438,
                    0.35351562, 0.47460938, 0.73437500, 0.53125000, 0.47070312,
                    0.49609375, 0.50000000, 0.55078125, 0.50000000, 0.45898438,
                    0.45703125, 0.61328125]],

                    [[0.63671875, 0.41210938, 0.52734375, 0.56640625, 0.44531250,
                    0.64843750, 0.37890625, 0.31250000, 0.56640625, 0.62890625,
                    0.53125000, 0.51562500, 0.54296875, 0.50781250, 0.35546875,
                    0.41601562, 0.55468750, 0.36914062, 0.35937500, 0.45117188,
                    0.46875000, 0.49609375, 0.28710938, 0.50000000, 0.49609375,
                    0.50000000, 0.51562500, 0.57031250, 0.77734375, 0.62109375,
                    0.43164062, 0.50781250],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]],

                    [[0.62109375, 0.44531250, 0.46875000, 0.61328125, 0.39062500,
                    0.60156250, 0.41015625, 0.28710938, 0.58984375, 0.67968750,
                    0.55859375, 0.48632812, 0.51562500, 0.42382812, 0.37695312,
                    0.46679688, 0.54687500, 0.44921875, 0.33789062, 0.36328125,
                    0.49023438, 0.44140625, 0.25000000, 0.45312500, 0.43945312,
                    0.45507812, 0.46679688, 0.57812500, 0.65625000, 0.64062500,
                    0.42382812, 0.57031250],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]],

                    [[0.62500000, 0.47070312, 0.51562500, 0.61328125, 0.36718750,
                    0.66406250, 0.37890625, 0.28320312, 0.65625000, 0.66015625,
                    0.48632812, 0.53906250, 0.46679688, 0.47851562, 0.43359375,
                    0.45703125, 0.47070312, 0.39843750, 0.32617188, 0.37304688,
                    0.49023438, 0.50390625, 0.27148438, 0.46679688, 0.37695312,
                    0.49023438, 0.47265625, 0.58593750, 0.64453125, 0.60156250,
                    0.38476562, 0.62109375],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]],

                    [[0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]],

                    [[0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]]]])
            >>> # doctest: -SKIP


    To convert FlashMask's `startend_row_indices` to `dense_mask`, use the code below:

    .. code-block:: python

        >>> import paddle
        >>> import numpy as np
        >>> def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
        ...     if startend_row_indices is None:
        ...         return None
        ...     bz, num_head, seq_len, bound_num = startend_row_indices.shape
        ...     m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
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
        >>> import paddle
        >>> startend_row_indices = paddle.to_tensor([8]*10, dtype="int32").reshape([1, 1, 10, 1])
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[8],
                [8],
                [8],
                [8],
                [8],
                [8],
                [8],
                [8],
                [8],
                [8]]]])
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
        >>> import paddle
        >>> startend_row_indices = paddle.to_tensor([3, 4, 5, 6, 7, 8, 9, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[3 ],
                [4 ],
                [5 ],
                [6 ],
                [7 ],
                [8 ],
                [9 ],
                [10],
                [10],
                [10]]]])
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
        >>> import paddle
        >>> startend_row_indices = paddle.to_tensor([4, 4, 4, 4, 7, 7, 7, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 ],
                [4 ],
                [4 ],
                [4 ],
                [7 ],
                [7 ],
                [7 ],
                [10],
                [10],
                [10]]]])
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
        >>> import paddle
        >>> LTS = paddle.to_tensor([4, 4, 4, 4, 7, 7, 7, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTE = paddle.to_tensor([0, 0, 0, 0, 4, 4, 4, 7, 7, 7], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 , 0 ],
                [4 , 0 ],
                [4 , 0 ],
                [4 , 0 ],
                [7 , 4 ],
                [7 , 4 ],
                [7 , 4 ],
                [10, 7 ],
                [10, 7 ],
                [10, 7 ]]]])
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
        >>> import paddle
        >>> startend_row_indices = paddle.to_tensor([10, 10, 10, 10, 7, 7, 7, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10],
                [10],
                [10],
                [10],
                [7 ],
                [7 ],
                [7 ],
                [10],
                [10],
                [10]]]])
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

        >>> import paddle
        >>> LTS = paddle.to_tensor([10, 10, 4, 5, 6, 7, 8, 9, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> LTE = paddle.to_tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTS = paddle.to_tensor([0, 0, 0, 0, 2, 2, 2, 2, 2, 2], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTE = paddle.to_tensor([0, 0, 0, 0, 3, 4, 5, 6, 7, 8], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, LTE, UTS, UTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 4], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10, 10, 0 , 0 ],
                [10, 10, 0 , 0 ],
                [4 , 10, 0 , 0 ],
                [5 , 10, 0 , 0 ],
                [6 , 10, 2 , 3 ],
                [7 , 10, 2 , 4 ],
                [8 , 10, 2 , 5 ],
                [9 , 10, 2 , 6 ],
                [10, 10, 2 , 7 ],
                [10, 10, 2 , 8 ]]]])
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
        >>> import paddle
        >>> LTS = paddle.to_tensor([4, 4, 4, 4, 10, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> LTE = paddle.to_tensor([7, 7, 7, 7, 10, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, LTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 , 7 ],
                [4 , 7 ],
                [4 , 7 ],
                [4 , 7 ],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10]]]])
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
        >>> import paddle
        >>> LTS = paddle.to_tensor([3, 3, 3, 5, 5, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTE = paddle.to_tensor([0, 0, 2, 3, 3, 5, 5, 7, 8, 9], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[3 , 0 ],
                [3 , 0 ],
                [3 , 2 ],
                [5 , 3 ],
                [5 , 3 ],
                [10, 5 ],
                [10, 5 ],
                [10, 7 ],
                [10, 8 ],
                [10, 9 ]]]])
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
        >>> import paddle
        >>> LTS = paddle.to_tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTE = paddle.to_tensor([0, 0, 0, 0, 0, 5, 6, 7, 8, 9], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10, 0 ],
                [10, 0 ],
                [10, 0 ],
                [10, 0 ],
                [10, 0 ],
                [10, 5 ],
                [10, 6 ],
                [10, 7 ],
                [10, 8 ],
                [10, 9 ]]]])

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
        >>> import paddle
        >>> LTS = paddle.to_tensor([10, 10, 2, 3, 4, 5, 6, 7, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> LTE = paddle.to_tensor([10, 10, 5, 5, 5, 5, 8, 8, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, LTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10, 10],
                [10, 10],
                [2 , 5 ],
                [3 , 5 ],
                [4 , 5 ],
                [5 , 5 ],
                [6 , 8 ],
                [7 , 8 ],
                [10, 10],
                [10, 10]]]])

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
        if causal:
            startend_row_indices = paddle.arange(
                window_size[0] + 1, sq + window_size[0] + 1, dtype="int32"
            ).reshape((1, 1, sq, 1))
            startend_row_indices = paddle.clip(
                startend_row_indices, max=sq
            ).repeat_interleave(bsz, 0)

        else:
            startend_row_indices = paddle.empty((1, 1, sq, 2), dtype="int32")
            startend_row_indices[0, 0, :, 0] = paddle.arange(
                window_size[0] + 1, sq + window_size[0] + 1, dtype="int32"
            )
            startend_row_indices[0, 0, :, 1] = paddle.arange(
                -window_size[1], sq - window_size[1], dtype="int32"
            )
            startend_row_indices = paddle.clip(
                startend_row_indices, min=0, max=sq
            ).repeat_interleave(bsz, 0)

    if block_mask is not None:
        # xhy: can set a full startend_row_indices for block_mask_attn when using block_mask_attn?
        assert startend_row_indices is not None, (
            "must provide startend_row_indices when using block_mask_attn"
        )

    if startend_row_indices is None:
        (
            out,
            result_softmax,
            result_softmax_lse,
            result_seed_offset,
        ) = _C_ops.flash_attn(
            query,
            key,
            value,
            fixed_seed_offset,
            None,
            dropout,
            causal,
            False,
            not training,
            rng_name,
        )

    else:
        assert startend_row_indices.dtype == paddle.int32, (
            f"startend_row_indices.dtype must be paddle.int32, but got {startend_row_indices.dtype}"
        )
        assert len(startend_row_indices.shape) == 4, (
            f"startend_row_indices rank must be 4,but got {startend_row_indices.shape}"
        )

        assert startend_row_indices.shape[0] == key.shape[0], (
            f"startend_row_indices.shape[0] must be equal to batch_size, but got {startend_row_indices.shape[0]} and {key.shape[0]}"
        )

        assert startend_row_indices.shape[2] == key.shape[1], (
            f"startend_row_indices.shape[2] must be equal to seqlen_k, but got {startend_row_indices.shape[2]} and {key.shape[2]}"
        )
        assert startend_row_indices.shape[1] in [
            1,
            query.shape[2],
            key.shape[2],
        ], (
            "startend_row_indices head_num must be equal to 1(broadcast) or head_num_q or head_num_k."
        )

        if block_mask is not None:
            assert block_mask.dtype == paddle.int32, (
                f"block_mask.dtype must be paddle.int32, but got {block_mask.dtype}"
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

        if (
            "xpu" not in paddle.get_device()
            and paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ]
        ):
            assert block_mask is None, (
                " blockmask attention no supports deterministic now ."
            )

        if "xpu" in paddle.get_device():
            fa_version = 2
        elif (
            paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
                "FLAGS_flash_attn_version"
            ]
            == 3
            and paddle.base.framework.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ]
            and query.shape[3] > 128
        ):
            fa_version = 2
        else:
            fa_version = paddle.base.framework.get_flags(
                ["FLAGS_flash_attn_version"]
            )["FLAGS_flash_attn_version"]

        if fa_version == 2:
            assert softmax_scale is None, (
                "flashmask_attention does not support setting softmax_scale, use flashmask_attention_v2 instead"
            )

            assert block_mask is None, (
                " blockmask attention only supports sm >= 90 now."
            )

            (
                out,
                result_softmax,
                result_softmax_lse,
                result_seed_offset,
            ) = _C_ops.flashmask_attention(
                query,
                key,
                value,
                startend_row_indices,
                fixed_seed_offset,
                dropout,
                causal,
                False,
                not training,
                rng_name,
            )

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

            # (
            #     out,
            #     result_softmax_lse,
            # ) = _C_ops.flashmask_attention_v2(
            #     query,
            #     key,
            #     value,
            #     startend_row_indices,
            #     block_mask,
            #     softmax_scale,
            #     causal,
            # )

            (
                out,
                result_softmax_lse,
            ) = _C_ops._run_custom_op(
                "flashmask_attention_v2",
                query,
                key,
                value,
                startend_row_indices,
                block_mask,
                softmax_scale,
                causal,
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
        return outputs
