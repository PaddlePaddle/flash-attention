# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
import math
from contextlib import nullcontext
from functools import wraps
from typing import Optional

import paddle
import paddle.nn.functional as F
from einops import rearrange, repeat


class IndexFirstAxis(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim = input.shape[0]
        other_shape = input.shape[1:]
        second_dim = 1
        for d in other_shape:
            second_dim *= d
        # gather along first axis
        flat_input = input.reshape([input.shape[0], second_dim])
        expanded_indices = indices.unsqueeze(1).expand([indices.shape[0], second_dim])
        result = paddle.gather(flat_input, indices)
        return result.reshape([-1, *other_shape])

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensor()
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        flat_grad = grad_output.reshape([grad_output.shape[0], -1])
        grad_input = paddle.zeros(
            [ctx.first_axis_dim, flat_grad.shape[1]],
            dtype=grad_output.dtype,
        )
        # scatter add
        expanded_indices = indices.unsqueeze(1).expand([indices.shape[0], flat_grad.shape[1]])
        grad_input = paddle.scatter(grad_input, indices, flat_grad)
        return grad_input.reshape([ctx.first_axis_dim, *other_shape]), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = paddle.zeros(
            [first_axis_dim, *values.shape[1:]], dtype=values.dtype
        )
        output = paddle.scatter(output, indices, values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensor()
        grad_values = paddle.gather(grad_output, indices)
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(axis=-1).cast(paddle.int32)
    used_seqlens_in_batch = attention_mask.sum(axis=-1).cast(paddle.int32)
    # No FakeTensorMode in Paddle — always use the real path
    indices = paddle.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(
        paddle.cumsum(seqlens_in_batch, axis=0).cast(paddle.int32),
        [1, 0],
    )
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def generate_random_padding_mask(
    max_seqlen, batch_size, device, mode="random", zero_lengths=False, min_seqlen=None
):
    assert mode in ["full", "random", "third"]
    min_seqlen = min_seqlen if min_seqlen is not None else 0 if zero_lengths else 1
    if mode == "full":
        lengths = paddle.full([batch_size, 1], max_seqlen, dtype=paddle.int32)
    elif mode == "random":
        lengths = paddle.randint(
            max(min_seqlen, max_seqlen - 20),
            max_seqlen + 1,
            shape=[batch_size, 1],
        )
    else:
        lengths = paddle.randint(
            max(min_seqlen, max_seqlen // 3),
            max_seqlen + 1,
            shape=[batch_size, 1],
        )

    if zero_lengths:
        for i in range(batch_size):
            if i % 5 == 0:
                lengths[i] = 0
        lengths[-1] = 0
    padding_mask = (
        repeat(paddle.arange(max_seqlen, dtype=paddle.int64), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    qv=None,
    kvpacked=False,
    qkvpacked=False,
    query_unused_mask=None,
    key_unused_mask=None,
):
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    d_v = v.shape[-1]
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == [batch_size, seqlen_k, nheads_k, d]
    assert v.shape == [batch_size, seqlen_k, nheads_k, d_v]
    if query_unused_mask is not None or key_unused_mask is not None:
        assert not kvpacked
        assert not qkvpacked

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, seqused_q = unpad_input(
            q, query_padding_mask, query_unused_mask
        )
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
        qv_unpad = rearrange(qv, "b s ... -> (b s) ...")[indices_q] if qv is not None else None
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = paddle.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=paddle.int32
        )
        seqused_q = None
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )
        qv_unpad = rearrange(qv, "b s ... -> (b s) ...") if qv is not None else None

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, seqused_k = unpad_input(
            k, key_padding_mask, key_unused_mask
        )
        v_unpad, *_ = unpad_input(v, key_padding_mask, key_unused_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = paddle.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=paddle.int32
        )
        seqused_k = None
        max_seqlen_k = seqlen_k

    def _detach_grad(t):
        t = t.detach()
        t.stop_gradient = False
        return t

    if qkvpacked:
        assert paddle.all(query_padding_mask == key_padding_mask)
        assert nheads == nheads_k
        qkv_unpad = paddle.stack([q_unpad, k_unpad, v_unpad], axis=1)
        qkv = paddle.stack([q, k, v], axis=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            _detach_grad(qkv_unpad),
            cu_seqlens_q,
            max_seqlen_q,
            _detach_grad(qkv),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = paddle.stack([k_unpad, v_unpad], axis=1)
        kv = paddle.stack([k, v], axis=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            _detach_grad(q_unpad),
            _detach_grad(kv_unpad),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            _detach_grad(q),
            _detach_grad(kv),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            _detach_grad(q_unpad),
            _detach_grad(k_unpad),
            _detach_grad(v_unpad),
            qv_unpad.detach() if qv is not None else None,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            _detach_grad(q),
            _detach_grad(k),
            _detach_grad(v),
            qv.detach() if qv is not None else None,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(None, None),
    sink_token_length=0,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    row_idx = rearrange(
        paddle.arange(seqlen_q, dtype=paddle.int64), "s -> s 1"
    )
    col_idx = paddle.arange(seqlen_k, dtype=paddle.int64)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = paddle.where(
            col_idx >= key_leftpad,
            col_idx - key_leftpad,
            paddle.full_like(col_idx, 2**32),
        )
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] is None:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = paddle.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        if window_size[1] is None:
            local_mask_left = col_idx > sk
        else:
            local_mask_left = col_idx > paddle.minimum(
                row_idx + sk - sq + paddle.to_tensor(window_size[1], dtype=paddle.int64),
                sk.cast(paddle.int64) if not isinstance(sk, int) else paddle.to_tensor(sk, dtype=paddle.int64),
            )
        return paddle.logical_or(
            local_mask_left,
            paddle.logical_and(
                col_idx < row_idx + sk - sq - window_size[0],
                col_idx >= sink_token_length,
            ),
        )


def construct_chunk_mask(
    seqlen_q,
    seqlen_k,
    attention_chunk,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    row_idx = rearrange(
        paddle.arange(seqlen_q, dtype=paddle.int64), "s -> s 1"
    )
    col_idx = paddle.arange(seqlen_k, dtype=paddle.int64)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = paddle.where(
            col_idx >= key_leftpad,
            col_idx - key_leftpad,
            paddle.full_like(col_idx, 2**32),
        )
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sk = paddle.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
    col_limit_left_chunk = row_idx + sk - sq - (row_idx + sk - sq) % attention_chunk
    return paddle.logical_or(
        col_idx < col_limit_left_chunk,
        col_idx >= col_limit_left_chunk + attention_chunk,
    )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(None, None),
    attention_chunk=0,
    sink_token_length=0,
    learnable_sink: Optional[paddle.Tensor] = None,
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
    return_lse=False,
    gather_kv_indices=None,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.cast(paddle.float32), k.cast(paddle.float32), v.cast(paddle.float32)
        qv = qv.cast(paddle.float32) if qv is not None else None
    if q_descale is not None:
        q_descale = repeat(q_descale, "b h -> b 1 (h g) 1", g=q.shape[2] // k.shape[2])
        q = (q.cast(paddle.float32) * q_descale).cast(q.dtype)
        qv = (qv.cast(paddle.float32) * q_descale).cast(qv.dtype) if qv is not None else None
    if k_descale is not None:
        k = (k.cast(paddle.float32) * rearrange(k_descale, "b h -> b 1 h 1")).cast(k.dtype)
    if v_descale is not None:
        v = (v.cast(paddle.float32) * rearrange(v_descale, "b h -> b 1 h 1")).cast(v.dtype)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    dv = v.shape[-1]
    softmax_scale = 1.0 / math.sqrt(d if qv is None else d + dv)
    if not reorder_ops:
        scores = paddle.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = paddle.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if qv is not None:
        scores = scores + paddle.einsum("bthd,bshd->bhts", qv * softmax_scale, v)
    if softcap > 0:
        scores = paddle.tanh(scores / softcap) * softcap
    if key_padding_mask is not None:
        inv_mask = rearrange(~key_padding_mask, "b s -> b 1 1 s")
        scores = paddle.where(inv_mask, paddle.full_like(scores, float("-inf")), scores)
    local_mask = None
    if window_size[0] is not None or window_size[1] is not None:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            sink_token_length,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
        )
    if attention_chunk > 0:
        chunk_mask = construct_chunk_mask(
            seqlen_q,
            seqlen_k,
            attention_chunk,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
        )
        local_mask = (
            paddle.logical_or(local_mask, chunk_mask) if local_mask is not None else chunk_mask
        )
    if gather_kv_indices is not None:
        batch = q.shape[0]
        topk_len = gather_kv_indices.shape[2]
        if topk_len < seqlen_k:
            topk_index_mask = paddle.zeros(
                [batch, seqlen_q, seqlen_k], dtype=paddle.bool
            )
            topk_index_mask = paddle.put_along_axis(
                topk_index_mask,
                gather_kv_indices,
                paddle.ones_like(gather_kv_indices, dtype=paddle.bool),
                axis=-1,
            )
            inv_topk_mask = rearrange(~topk_index_mask, "b t s -> b 1 t s")
            scores = paddle.where(inv_topk_mask, paddle.full_like(scores, float("-inf")), scores)
    if local_mask is not None:
        scores = paddle.where(local_mask, paddle.full_like(scores, float("-inf")), scores)
    if attn_bias is not None:
        scores = scores + attn_bias
    # After all masks are applied, before softmax:
    # scores shape: [b, h, t, s]
    lse = paddle.logsumexp(scores, axis=-1)  # [b, h, t]
    if learnable_sink is None:
        attention = F.softmax(scores, axis=-1).cast(v.dtype)
    else:
        scores_fp32 = scores.cast(paddle.float32)
        logits_max = paddle.amax(scores_fp32, axis=-1, keepdim=True)
        learnable_sink = rearrange(learnable_sink, "h -> h 1 1")
        logits_or_sinks_max = paddle.maximum(learnable_sink, logits_max)
        unnormalized_scores = paddle.exp(scores_fp32 - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(axis=-1, keepdim=True) + paddle.exp(
            learnable_sink - logits_or_sinks_max
        )
        # LSE with sink: log(Z) = log(normalizer) + max
        lse = (paddle.log(normalizer.squeeze(-1)) + logits_or_sinks_max.squeeze(-1)).cast(dtype_og)
        attention = (unnormalized_scores / normalizer).cast(v.dtype)
    if query_padding_mask is not None:
        inv_q_mask = rearrange(~query_padding_mask, "b s -> b 1 s 1")
        attention = paddle.where(inv_q_mask, paddle.zeros_like(attention), attention)
    if key_padding_mask is not None:
        inv_k_mask = rearrange(~key_padding_mask, "b s -> b 1 1 s")
        attention = paddle.where(inv_k_mask, paddle.zeros_like(attention), attention)
    if local_mask is not None:
        all_masked = paddle.all(local_mask, axis=-1, keepdim=True)
        attention = paddle.where(all_masked, paddle.zeros_like(attention), attention)
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = paddle.where(~dropout_mask, paddle.zeros_like(attention), attention)
    else:
        attention_drop = attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.cast(intermediate_dtype).cast(attention_drop.dtype)
    output = paddle.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        inv_q_mask2 = rearrange(~query_padding_mask, "b s -> b s 1 1")
        output = paddle.where(inv_q_mask2, paddle.zeros_like(output), output)
    if return_lse:
        return output.cast(dtype_og), attention.cast(dtype_og), lse.cast(dtype_og)
    return output.cast(dtype_og), attention.cast(dtype_og)


def maybe_fake_tensor_mode(fake: bool = True):
    """
    No-op wrapper for Paddle: FakeTensorMode is a torch concept.
    Decorator simply calls the function in a nullcontext.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with nullcontext():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def is_fake_mode() -> bool:
    """Always False in Paddle (no FakeTensorMode equivalent)."""
    return False
