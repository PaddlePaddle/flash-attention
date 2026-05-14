# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Original portions of this file are licensed under the MIT License.
# See the LICENSE-MIT file or the original project license for details.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
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
from einops import rearrange


def naive_recurrent_kda(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: float | None = None,
    initial_state: paddle.Tensor | None = None,
    output_final_state: bool = False,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = [x.cast(paddle.float32) for x in [q, k, v, g, beta]]
    q = q * scale

    S = paddle.zeros([B, H, K, V], dtype=q.dtype)
    if initial_state is not None:
        S += initial_state
    o = paddle.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + paddle.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = paddle.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.cast(dtype), S


def naive_chunk_kda(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    g: paddle.Tensor,
    beta: paddle.Tensor,
    scale: float | None = None,
    initial_state: paddle.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    q, k, v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).cast(paddle.float32) for x in [q, k, v, g, beta]]
    q = q * scale
    g = g.cumsum(-2)

    # note that diagonal is masked.
    mask = paddle.triu(paddle.ones([BT, BT], dtype='bool'), diagonal=0)

    A = paddle.zeros([*q.shape[:-1], BT], dtype=paddle.float32)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = paddle.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -(A * (~mask).cast(A.dtype))
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + paddle.eye(BT, dtype=paddle.float32)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = paddle.zeros([B, H, K, V], dtype=q.dtype)
    if initial_state is not None:
        S += initial_state
    o = paddle.zeros_like(v)
    mask = paddle.triu(paddle.ones([BT, BT], dtype='bool'), diagonal=1)
    for i in range(0, NT):
        # [B, H, BT, ...]
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        A = paddle.zeros([B, H, BT, BT], dtype=paddle.float32)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            A[..., j] = paddle.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        A = A * (~mask).cast(A.dtype)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + A @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').cast(dtype), S
