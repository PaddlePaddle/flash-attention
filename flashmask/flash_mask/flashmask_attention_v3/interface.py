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
            The dtype must be int32.
        dropout (float): The dropout ratio. Default is 0.0.
        causal (bool): Whether to enable causal mode. Default is False.
        window_size (int|tuple, optional): Indicates the window size of sliding window local attention.
        return_softmax_lse (bool): Whether to return the log-sum-exp of the softmax. Default is False.
        return_seed_offset (bool): Whether to return the random seed offset. Default is False.
        fixed_seed_offset(Tensor, optional): With fixed seed, offset for dropout mask.
        rng_name (str): The name to select Generator.
        training (bool): Whether the module is in training mode. Default is True.
        name (str, optional): Name of the operation.
        softmax_scale (float, optional): The scaling factor for softmax.
        block_mask (tensor, optional): A 4-D integer mask tensor for block-level masking.

    Returns:
        Tensor. The computed attention result with the same shape as the input `query`.
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
        assert startend_row_indices is not None, (
            "must provide startend_row_indices when using block_mask_attn"
        )

    if startend_row_indices is None:
        raise ValueError(
            "flashmask_attention requires startend_row_indices. "
            "For standard flash attention without mask, use paddle.nn.functional.flash_attention."
        )

    assert startend_row_indices.dtype == paddle.int32, (
        f"startend_row_indices.dtype must be paddle.int32, but got {startend_row_indices.dtype}"
    )
    assert len(startend_row_indices.shape) == 4, (
        f"startend_row_indices rank must be 4, but got {startend_row_indices.shape}"
    )
    assert startend_row_indices.shape[0] == key.shape[0], (
        f"startend_row_indices.shape[0] must be equal to batch_size, but got {startend_row_indices.shape[0]} and {key.shape[0]}"
    )
    assert startend_row_indices.shape[2] == key.shape[1], (
        f"startend_row_indices.shape[2] must be equal to seqlen_k, but got {startend_row_indices.shape[2]} and {key.shape[1]}"
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
            f"block_mask.shape[0] must be equal to batch_size"
        )
        assert block_mask.shape[1] == startend_row_indices.shape[1], (
            f"block_mask.shape[1] must be equal to startend_row_indices.shape[1]"
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
            pass
        elif startend_row_indices.shape[-1] == 2:
            pass
        else:
            raise ValueError(
                f"Invalid shape of startend_row_indices, when causal is True, the last dimension should be either 1 or 2 but got {startend_row_indices.shape[-1]}"
            )
    else:
        if startend_row_indices.shape[-1] == 2:
            pass
        elif startend_row_indices.shape[-1] == 4:
            pass
        else:
            raise ValueError(
                f"Invalid shape of startend_row_indices, when causal is False, the last dimension should be either 2 or 4 but got {startend_row_indices.shape[-1]}"
            )

    # fa_version = 3 only
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

    # [BQW_CHANGE] 使用 _C_ops._run_custom_op 调用注册的自定义算子
    # Paddle 的 PD_BUILD_OP 注册后，通过此方式调用
    outs = _C_ops._run_custom_op(
        "flashmask_attention_v2",
        query,
        key,
        value,
        startend_row_indices,
        block_mask,
        softmax_scale,
        causal,
    )
    out = outs[0]
    result_softmax_lse = outs[1]

    outputs = [out]
    if return_softmax_lse:
        outputs += [result_softmax_lse]
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs
