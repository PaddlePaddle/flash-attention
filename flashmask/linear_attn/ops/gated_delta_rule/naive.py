# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Adapted for PaddlePaddle

import paddle


def naive_recurrent_gated_delta_rule(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    beta: paddle.Tensor,
    g: paddle.Tensor,
    scale: float = None,
    initial_state: paddle.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PaddlePaddle implementation of recurrent gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g: [B, T, H]
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    q, k, v, beta, g = map(
        lambda x: x.transpose([0, 2, 1, 3]).contiguous().cast(paddle.float32) if x.ndim == 4
        else x.transpose([0, 2, 1]).contiguous().cast(paddle.float32),
        [q, k, v, beta, g]
    )
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = paddle.zeros([B, H, T, V], dtype=v.dtype)
    h = paddle.zeros([B, H, K, V], dtype=v.dtype)
    if initial_state is not None:
        h = initial_state.cast(paddle.float32)
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k.unsqueeze(-1)).sum(-2)
        b_v = b_v * b_beta.unsqueeze(-1)
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = paddle.einsum('bhd,bhdm->bhm', b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose([0, 2, 1, 3]).contiguous()
    return o, h
