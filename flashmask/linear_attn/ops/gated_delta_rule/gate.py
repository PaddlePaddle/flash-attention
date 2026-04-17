# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import paddle
import paddle.nn.functional as F
import triton
import triton.language as tl

from linear_attn.ops.utils.index import prepare_chunk_indices
from linear_attn.ops.utils.op import exp
from linear_attn.ops.utils.softplus import softplus
from linear_attn.utils import autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, input_guard
from linear_attn.triton_utils import enable_compat_on_triton_kernel


def naive_gdn_gate(
    g: paddle.Tensor,
    A_log: paddle.Tensor,
    dt_bias: paddle.Tensor | None = None,
    output_dtype: paddle.dtype = paddle.float32,
) -> paddle.Tensor:
    """
    Paddle reference implementation for GDN gate computation.

    Computes: ``g = -A_log.exp() * softplus(g + dt_bias)``

    Args:
        g (paddle.Tensor):
            Input tensor of shape `[..., HV]`.
        A_log (paddle.Tensor):
            Decay parameter tensor with `HV` elements.
        dt_bias (paddle.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[HV]`.

    Returns:
        Output tensor of shape `[..., HV]`.
    """
    g = g.cast(paddle.float32)
    if dt_bias is not None:
        g = g + dt_bias.cast(paddle.float32)
    return (-A_log.cast(paddle.float32).exp() * F.softplus(g)).cast(output_dtype)


@enable_compat_on_triton_kernel
@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_chunk_cumsum_scalar_kernel(
    g,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_gate = -exp(b_A) * softplus(b_g)

    b_o = tl.cumsum(b_gate, axis=0)
    if REVERSE:
        b_z = tl.sum(b_gate, axis=0)
        b_o = -b_o + b_z[None] + b_gate
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@enable_compat_on_triton_kernel
@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['H', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dg = tl.make_block_ptr(dg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_dyg = tl.make_block_ptr(dyg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_dyg = tl.load(p_dyg, boundary_check=(0,)).to(tl.float32)

    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)

    # gate = -exp(A_log) * softplus(g + bias)
    # d(gate)/d(g) = -exp(A_log) * sigmoid(g + bias)   (softplus' = sigmoid)
    # d(gate)/d(A_log) = -exp(A_log) * softplus(g + bias) = gate
    b_neg_expA = -exp(b_A)
    b_yg = b_neg_expA * softplus(b_g)
    b_dg = b_neg_expA * (b_dyg * tl.sigmoid(b_g))
    b_dA = tl.sum(b_dyg * b_yg, 0)

    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))
    tl.store(dA + i_t * H + i_h, b_dA)


@input_guard
def gdn_gate_chunk_cumsum(
    g: paddle.Tensor,
    A_log: paddle.Tensor,
    chunk_size: int,
    scale: float = None,
    dt_bias: paddle.Tensor | None = None,
    cu_seqlens: paddle.Tensor | None = None,
    chunk_indices: paddle.Tensor | None = None,
    output_dtype: paddle.dtype | None = paddle.float32,
) -> paddle.Tensor:
    B, T, H = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = paddle.empty_like(g).cast(output_dtype or g.dtype)
    gdn_gate_chunk_cumsum_scalar_kernel[(NT, B * H)](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        REVERSE=False,
    )
    return o


def gdn_gate_bwd(
    g: paddle.Tensor,
    A_log: paddle.Tensor,
    dt_bias: paddle.Tensor | None,
    dyg: paddle.Tensor,
) -> tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor | None]:
    H = g.shape[-1]
    T = g.numel().item() // H
    BT = 32
    NT = triton.cdiv(T, BT)

    dg = paddle.empty_like(g).cast(paddle.float32)
    dA = paddle.empty([NT, H], dtype=paddle.float32)

    gdn_gate_bwd_kernel[(NT, H)](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        dyg=dyg,
        dg=dg,
        dA=dA,
        T=T,
        H=H,
        BT=BT,
    )

    # Compute dbias from dg while still in float32 (before casting to g.dtype which may be float16).
    # Paddle's .sum() does not promote float16 to float32 for accumulation like PyTorch does.
    dbias = dg.reshape([-1, H]).sum(axis=0).cast(dt_bias.dtype) if dt_bias is not None else None
    dg = dg.reshape(g.shape).cast(g.dtype)
    dA = dA.sum(axis=0).reshape(A_log.shape).cast(A_log.dtype)

    return dg, dA, dbias


@enable_compat_on_triton_kernel
@triton.heuristics({
    'HAS_BIAS': lambda args: args['dt_bias'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=['H'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def gdn_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    yg,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_yg = tl.make_block_ptr(yg + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h).to(tl.float32)
    b_yg = -exp(b_A) * softplus(b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0,))


def gdn_gate_fwd(
    g: paddle.Tensor,
    A_log: paddle.Tensor,
    dt_bias: paddle.Tensor | None = None,
    output_dtype: paddle.dtype = paddle.float32,
) -> paddle.Tensor:
    H = g.shape[-1]
    T = g.numel().item() // H

    yg = paddle.empty_like(g).cast(output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta['BT']), H)

    gdn_gate_fwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        yg=yg,
        T=T,
        H=H,
    )
    return yg


class GDNGateFunction(paddle.autograd.PyLayer):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: paddle.Tensor,
        A_log: paddle.Tensor,
        dt_bias: paddle.Tensor | None = None,
        output_dtype: paddle.dtype = paddle.float32,
    ) -> paddle.Tensor:
        yg = gdn_gate_fwd(g=g, A_log=A_log, dt_bias=dt_bias, output_dtype=output_dtype)
        ctx.save_for_backward(g, A_log, dt_bias)
        ctx.output_dtype = output_dtype
        # Paddle PyLayer backward must return exactly as many values as tensor inputs.
        _forward_args = [g, A_log, dt_bias, output_dtype]
        ctx._tensor_mask = tuple(isinstance(a, paddle.Tensor) for a in _forward_args)
        ctx._needs_grad = tuple(
            isinstance(a, paddle.Tensor) and not a.stop_gradient for a in _forward_args
        )
        return yg

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dyg: paddle.Tensor):
        g, A_log, dt_bias = ctx.saved_tensor()
        dg, dA, dbias = gdn_gate_bwd(g=g, A_log=A_log, dt_bias=dt_bias, dyg=dyg)
        all_grads = [dg, dA, dbias, None]
        return tuple(
            g if needs_grad else None
            for g, is_tensor, needs_grad in zip(all_grads, ctx._tensor_mask, ctx._needs_grad)
            if is_tensor
        )


def fused_gdn_gate(
    g: paddle.Tensor,
    A_log: paddle.Tensor,
    dt_bias: paddle.Tensor | None = None,
    output_dtype: paddle.dtype = paddle.float32,
) -> paddle.Tensor:
    r"""
    Fused GDN gate computation with autograd support.

    Computes: ``g = -A_log.exp() * softplus(g + dt_bias)``

    Args:
        g (paddle.Tensor):
            Input tensor of shape `[..., HV]`.
        A_log (paddle.Tensor):
            Decay parameter tensor with `HV` elements.
        dt_bias (paddle.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[HV]`.
        output_dtype (paddle.dtype):
            The dtype of the output tensor. Default: `paddle.float32`.

    Returns:
        Output tensor of shape `[..., HV]`.
    """
    return GDNGateFunction.apply(g, A_log, dt_bias, output_dtype)
