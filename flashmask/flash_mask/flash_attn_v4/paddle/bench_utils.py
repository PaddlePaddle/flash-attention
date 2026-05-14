# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
"""Shared benchmark utilities: attention_ref, cuDNN helpers, flops calculation — Paddle backend."""

import math
import paddle
import paddle.nn.functional as F


# ── FLOPS calculation ────────────────────────────────────────────────────────


def flops(
    batch,
    nheads,
    seqlen_q,
    seqlen_k,
    headdim,
    headdim_v,
    causal=False,
    window_size=(None, None),
    has_qv=False,
):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (None, None):
            avg_seqlen = seqlen_k
        else:
            row_idx = paddle.arange(seqlen_q, dtype=paddle.float32, place=paddle.CUDAPlace(0))
            col_left = (
                paddle.maximum(
                    row_idx + seqlen_k - seqlen_q - window_size[0],
                    paddle.zeros([1], dtype=paddle.float32),
                )
                if window_size[0] is not None
                else paddle.zeros_like(row_idx)
            )
            col_right = (
                paddle.minimum(
                    row_idx + seqlen_k - seqlen_q + window_size[1],
                    paddle.full([1], seqlen_k - 1, dtype=paddle.float32),
                )
                if window_size[1] is not None
                else paddle.full_like(row_idx, seqlen_k - 1)
            )
            avg_seqlen = (col_right - col_left + 1).mean().item()
    eff_headdim = headdim + headdim_v if has_qv else headdim
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (eff_headdim + headdim_v)


# ── Bandwidth calculation ────────────────────────────────────────────────────


def bandwidth_fwd_bytes(
    batch, nheads, nheads_kv, seqlen_q, seqlen_k, headdim, headdim_v, dtype_bytes=2, has_qv=False
):
    """HBM traffic for one attention pass: read Q,K,V + write O."""
    q = batch * nheads * seqlen_q * headdim
    qv = batch * nheads * seqlen_q * headdim_v if has_qv else 0
    k = batch * nheads_kv * seqlen_k * headdim
    v = batch * nheads_kv * seqlen_k * headdim_v
    o = batch * nheads * seqlen_q * headdim_v
    return (q + qv + k + v + o) * dtype_bytes


def bandwidth_bwd_bytes(
    batch, nheads, nheads_kv, seqlen_q, seqlen_k, headdim, headdim_v, dtype_bytes=2
):
    """HBM traffic for one attention pass: read Q,K,V,dO + write dQ,dK,dV."""
    q = batch * nheads * seqlen_q * headdim
    k = batch * nheads_kv * seqlen_k * headdim
    v = batch * nheads_kv * seqlen_k * headdim_v
    do = batch * nheads * seqlen_q * headdim_v
    dq = q
    dk = k
    dv = v
    return (q + k + v + do + dq + dk + dv) * dtype_bytes


# ── Reference attention ─────────────────────────────────────────────────────

_attention_ref_mask_cache = {}


def attention_ref(q, k, v, causal=False):
    """Standard attention reference implementation.

    Args:
        q, k, v: (batch, seqlen, nheads, headdim) tensors.
        causal: whether to apply causal mask.
    """
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = paddle.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    if causal:
        seqlen_q, seqlen_k = scores.shape[-2], scores.shape[-1]
        cache_key = seqlen_q
        if cache_key not in _attention_ref_mask_cache:
            mask = paddle.tril(
                paddle.ones([seqlen_q, seqlen_k], dtype=paddle.bool)
            )
            _attention_ref_mask_cache[cache_key] = mask
        else:
            mask = _attention_ref_mask_cache[cache_key]
        # Apply causal mask: masked positions (upper triangle) get -inf
        scores = paddle.where(mask, scores, paddle.full_like(scores, float("-inf")))
    attn = F.softmax(scores, axis=-1)
    return paddle.einsum("bhts,bshd->bthd", attn, v)


# ── cuDNN graph helpers ─────────────────────────────────────────────────────
# cuDNN integration is not available for the Paddle backend path.
# These stubs raise NotImplementedError to give a clear message.

_PADDLE_TO_CUDNN_DTYPE = {
    paddle.float16: "HALF",
    paddle.bfloat16: "BFLOAT16",
    paddle.float32: "FLOAT",
    paddle.int32: "INT32",
    paddle.int64: "INT64",
}


def cudnn_fwd_setup(q, k, v, causal=False, window_size_left=None):
    raise NotImplementedError(
        "cudnn_fwd_setup is not supported in the Paddle backend. "
        "Use attention_ref() for a reference implementation."
    )


def cudnn_bwd_setup(q, k, v, o, g, lse, causal=False, window_size_left=None):
    raise NotImplementedError(
        "cudnn_bwd_setup is not supported in the Paddle backend. "
        "Use the Paddle autograd backend for gradient computation."
    )
