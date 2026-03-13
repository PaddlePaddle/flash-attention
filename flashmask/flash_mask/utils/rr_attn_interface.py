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

import paddle
import paddle.nn.functional as F

from .rr_attn_estimate_triton_op import (
    rr_attn_estimate_triton_func,
)


def rr_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    startend_row_indices: paddle.Tensor,
    *,
    threshold: float = 1.0,
    stride: int = 8,
    causal: bool = False,
    dropout: float = 0.0,
    training: bool = True,
    keep_sink: bool = True,
    keep_last: bool = True,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
) -> paddle.Tensor:
    """
    RRAttention using threshold-based sparse pattern estimation.

    This function implements an efficient attention mechanism that uses a
    threshold-based strategy to reduce computation in attention blocks. It estimates
    attention patterns using a custom triton kernel and applies sparse attention with
    FlashMask.

    Args:
        query (paddle.Tensor):
            Query tensor with shape [batch_size, seq_len_q, num_heads, head_dim]
        key (paddle.Tensor):
            Key tensor with shape [batch_size, seq_len_k, num_heads, head_dim]
        value (paddle.Tensor):
            Value tensor with shape [batch_size, seq_len_k, num_heads, head_dim]
        startend_row_indices (paddle.Tensor):
            See flashmask_attention for details.
        threshold (float, optional):
            Sparsity threshold in range [0, 1]. Higher values produce sparser patterns.
            Default: 1.0 (full attention)
        stride (int, optional):
            Stride for attention pattern estimation. Controls granularity of block processing.
            Default: 8
        causal (bool, optional):
            Whether to apply causal masking. Default: False
        dropout (float, optional):
            Dropout probability for attention weights. Default: 0.0
        training (bool, optional):
            Whether in training mode. Default: True
        return_softmax_lse (bool, optional):
            Whether to return log-sum-exp values. Default: False
        return_seed_offset (bool, optional):
            Whether to return seed offset. Default: False

    Returns:
        paddle.Tensor:
            Attention output with shape [batch_size, seq_len_q, num_heads, head_dim]
            If return_softmax_lse is True and return_seed_offset is True, returns tuple:
            (output, softmax_lse, seed_offset)

    Raises:
        ValueError: If input tensors have incompatible shapes or invalid parameters
        RuntimeError: If triton kernel execution fails

    Example:
        >>> import paddle
        >>> from rr_attn_interface import rr_attention
        >>>
        >>> # Create sample tensors
        >>> batch_size, seq_len_q, seq_len_k, num_heads, head_dim = 2, 512, 512, 8, 64
        >>> query = paddle.randn([batch_size, seq_len_q, num_heads, head_dim])
        >>> key = paddle.randn([batch_size, seq_len_k, num_heads, head_dim])
        >>> value = paddle.randn([batch_size, seq_len_k, num_heads, head_dim])
        >>>
        >>> # Apply RR attention with threshold 0.8
        >>> output = rr_attention(
        ...     query, key, value, startend_row_indices,
        ...     threshold=0.8, causal=True, training=True
        ... )
        >>> print(output.shape)  # [2, 512, 8, 64]

    Note:
        - When threshold=1.0, falls back to standard flashmask_attention
        - The sparse pattern is constructed by boundary_mask and top-p mask
    """

    # Fast path: full attention when threshold is maximum
    if startend_row_indices is None or threshold == 1.0:
        return F.flashmask_attention(
            query,
            key,
            value,
            startend_row_indices=startend_row_indices,
            dropout=dropout,
            causal=causal,
            training=training,
            return_softmax_lse=return_softmax_lse,
            return_seed_offset=return_seed_offset,
        )

    # Ensure startend_row_indices has same number of heads as query
    _, num_heads_q = query.shape[0], query.shape[2]
    num_heads_indices = startend_row_indices.shape[1]
    if num_heads_indices != num_heads_q:
        if num_heads_q % num_heads_indices != 0:
            raise ValueError(
                f"query heads ({num_heads_q}) must be divisible by "
                f"startend_row_indices heads ({num_heads_indices})"
            )
        repeat_factor = num_heads_q // num_heads_indices
        startend_row_indices = startend_row_indices.repeat_interleave(
            repeat_factor, axis=1
        ).contiguous()

    with paddle.no_grad():
        attn_sums, boundary_mask, topp_mask = rr_attn_estimate_triton_func(
            q=query,
            k=key,
            startend_row_indices=startend_row_indices,
            stride=stride,
            threshold=threshold,
            causal=causal,
        )

        # Combine masks: boundary protection + top-p sparsity
        block_mask = paddle.logical_or(boundary_mask, topp_mask).astype(paddle.int32)
        if keep_sink:
            block_mask[:, :, :, 0] = 1
        if keep_last:
            block_mask[:, :, -1, :] = 1

    # Apply sparse attention with computed block mask
    return F.flashmask_attention(
        query,
        key,
        value,
        startend_row_indices=startend_row_indices,
        dropout=dropout,
        causal=causal,
        training=training,
        return_softmax_lse=return_softmax_lse,
        return_seed_offset=return_seed_offset,
        block_mask=block_mask,
    )


# if __name__ == '__main__':
#     query = paddle.randn((1, 128, 8, 128), dtype=paddle.bfloat16)
#     key = paddle.randn((1, 128, 2, 128), dtype=paddle.bfloat16)
#     value = paddle.randn((1, 128, 2, 128), dtype=paddle.bfloat16)

#     startend_row_indices = paddle.full([1, 1, 128, 1], 128, dtype=paddle.int32)
#     print(rr_attention(query, key, value, startend_row_indices, threshold=0.5, causal=True))
