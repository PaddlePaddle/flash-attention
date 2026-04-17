# -*- coding: utf-8 -*-
# Tests for KDA (Kimi Delta Attention) operators on PaddlePaddle
# Aligned with tests/ops/test_kda.py

import paddle
import paddle.nn.functional as F
import pytest

from linear_attn.ops.kda import chunk_kda, fused_recurrent_kda
from linear_attn.ops.kda.fused_recurrent import fused_recurrent_kda_fwd
from linear_attn.ops.kda.gate import fused_kda_gate, naive_kda_gate, naive_kda_lowerbound_gate
from linear_attn.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda

from .conftest import assert_close


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 64, 1, 1, paddle.float32),
            (2, 512, 3, 60, 1, 1, paddle.float32),
            (4, 1024, 4, 128, 0.1, 1, paddle.float32),
            (4, 1024, 4, 128, 1, 10, paddle.float32),
        ]
    ],
)
def test_naive_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype,
):
    paddle.seed(42)
    q = paddle.rand([B, T, H, D], dtype=dtype)
    k = paddle.rand([B, T, H, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    g = F.log_sigmoid(paddle.randn([B, T, H, D], dtype=paddle.float32)) / gate_logit_normalizer
    beta = paddle.randn([B, T, H], dtype=dtype).sigmoid()
    h0 = paddle.randn([B, H, D, D], dtype=paddle.float32)

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = naive_chunk_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "use_qk_l2norm_in_kernel", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-use_qk_l2norm_in_kernel{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 64, 1, 1, False, paddle.float32),
            (2, 512, 3, 60, 1, 1, False, paddle.float32),
            (3, 1000, 4, 100, 0.1, 1, True, paddle.float32),
            (4, 1024, 4, 128, 0.1, 1, False, paddle.float32),
        ]
    ],
)
def test_fused_recurrent(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    dtype,
):
    paddle.seed(42)
    q = paddle.rand([B, T, H, D], dtype=dtype)
    k = paddle.rand([B, T, H, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    g = F.log_sigmoid(paddle.randn([B, T, H, D], dtype=paddle.float32)) / gate_logit_normalizer
    beta = paddle.randn([B, T, H], dtype=dtype).sigmoid()
    h0 = paddle.randn([B, H, D, D], dtype=paddle.float32)

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    tri, tri_ht = fused_recurrent_kda(
        q=F.normalize(q.clone(), p=2, axis=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, axis=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 64, 1, 64, 1, 1, paddle.float32),
            (2, 512, 3, 60, 1, 1, paddle.float32),
            (3, 1000, 4, 100, 0.1, 1, paddle.float32),
            (4, 1024, 4, 128, 0.1, 1, paddle.float32),
        ]
    ],
)
def test_fused_recurrent_transpose_state(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype,
):
    paddle.seed(42)
    q = paddle.rand([B, T, H, D], dtype=dtype)
    k = paddle.rand([B, T, H, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    g = F.log_sigmoid(paddle.randn([B, T, H, D], dtype=paddle.float32)) / gate_logit_normalizer
    beta = paddle.randn([B, T, H], dtype=dtype).sigmoid()
    h0_kv = paddle.randn([B, H, D, D], dtype=paddle.float32)
    h0_vk = h0_kv.transpose([0, 1, 3, 2]).contiguous()

    ref, ref_ht = fused_recurrent_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        transpose_state_layout=False,
    )
    tri, tri_ht = fused_recurrent_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        transpose_state_layout=True,
    )
    assert_close("o", ref, tri, 1e-4)
    assert_close("ht", ref_ht, tri_ht.transpose([0, 1, 3, 2]), 1e-4)


@pytest.mark.parametrize(
    ("B", "H", "D", "scale", "gate_logit_normalizer", "use_qk_l2norm_in_kernel", "use_gate_in_kernel", "safe_gate", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-H{}-D{}-scale{}-norm{}-qk_l2{}-gate{}-safe_gate{}-dtype{}".format(*test),
        )
        for test in [
            (16, 16, 128, 0.1, 1.0, True, False, False, paddle.bfloat16),
            (32, 8, 64, 1.0, 1.0, False, False, False, paddle.float16),
            (16, 16, 128, 0.1, 1.0, True, True, False, paddle.bfloat16),
            (32, 8, 64, 1.0, 1.0, False, True, False, paddle.float16),
            (7, 32, 128, 0.5, 0.5, True, True, True, paddle.bfloat16),
        ]
    ],
)
def test_fused_recurrent_vllm_decode(
    B: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    dtype,
):
    """Test vLLM-style decoding with continuous batching and paged state storage."""
    paddle.seed(42)

    # Setup cache pool and inputs
    max_cache_slots = B * 3
    state_pool = paddle.randn([max_cache_slots, H, D, D], dtype=paddle.float32)
    state_indices = paddle.randperm(max_cache_slots)[:B].cast(paddle.int32)

    # Fill unaccessed slots with a huge value to detect out-of-bound access
    HUGE_VALUE = 1e30
    mask = paddle.ones([max_cache_slots], dtype='bool')
    mask[state_indices.cast(paddle.int64)] = False
    state_pool[mask] = HUGE_VALUE

    T = 1
    total_tokens = B * T

    q = paddle.rand([1, total_tokens, H, D], dtype=dtype)
    k = paddle.rand([1, total_tokens, H, D], dtype=dtype)
    v = paddle.rand([1, total_tokens, H, D], dtype=dtype)
    g = paddle.randn([1, total_tokens, H, D], dtype=paddle.float32 if not use_gate_in_kernel else dtype)

    if use_gate_in_kernel:
        A_log = paddle.log(paddle.uniform([1, 1, H, 1], dtype=paddle.float32, min=1, max=16)).squeeze()
        dt_bias = paddle.randn([H * D], dtype=paddle.float32)
        lower_bound = -5.0 if safe_gate else None
        naive_kda_gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    else:
        g = F.log_sigmoid(g) / gate_logit_normalizer
        A_log = None
        dt_bias = None
        lower_bound = None
        naive_kda_gate_fn = None

    beta = paddle.randn([1, total_tokens, H], dtype=dtype).sigmoid()

    cu_seqlens = paddle.arange(0, total_tokens + 1, step=T, dtype=paddle.int32)
    ref_state_pool = state_pool.clone()
    tri_state_pool = state_pool.clone()

    # Reference implementation (loop over batch)
    ref_outputs = []
    for i in range(B):
        start, end = i, i + 1
        slot_idx = state_indices[i].item()

        q_i = q[:, start:end].clone()
        k_i = k[:, start:end].clone()
        v_i = v[:, start:end].clone()
        g_i = g[:, start:end].clone()
        beta_i = beta[:, start:end].clone()

        h_init = ref_state_pool[slot_idx].clone().unsqueeze(0)
        ref_o_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q_i, p=2, axis=-1),
            k=F.normalize(k_i, p=2, axis=-1),
            v=v_i,
            g=(naive_kda_gate_fn(g_i, A_log, dt_bias) if use_gate_in_kernel else g_i),
            beta=beta_i,
            scale=scale,
            initial_state=h_init,
            output_final_state=True
        )
        ref_outputs.append(ref_o_i)
        ref_state_pool[slot_idx] = ref_ht_i.squeeze(0)

    ref_out = paddle.concat(ref_outputs, axis=1)

    # Triton kernel
    q_in = q.clone()
    k_in = k.clone()
    if not use_qk_l2norm_in_kernel:
        q_in = F.normalize(q_in, p=2, axis=-1)
        k_in = F.normalize(k_in, p=2, axis=-1)

    tri_out, _ = fused_recurrent_kda_fwd(
        q=q_in,
        k=k_in,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=tri_state_pool,
        scale=scale,
        output_final_state=False,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )

    # Verify results
    assert_close("o", ref_out, tri_out, 0.005)
    assert_close("ht", ref_state_pool[state_indices.cast(paddle.int64)],
                 tri_state_pool[state_indices.cast(paddle.int64)], 0.005)

    mask = paddle.ones([max_cache_slots], dtype='bool')
    mask[state_indices.cast(paddle.int64)] = False
    assert_close("Untouched ht", ref_state_pool[mask], tri_state_pool[mask], 0.0)


@pytest.mark.parametrize(
    (
        "B", "T", "H", "D", "scale", "gate_logit_normalizer",
        "mask_p", "use_qk_l2norm_in_kernel", "use_gate_in_kernel",
        "dtype", "safe_gate", "disable_recompute",
    ),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-mask_p{}-qk_l2norm{}-gate{}-dtype{}-safe_gate{}-disable_recompute{}".format(
                *test),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, 0, False, False, paddle.float16, True, False),
            (2, 500, 3, 60, 1, 1, 0, False, False, paddle.float16, True, True),
            (2, 1000, 3, 64, 0.1, 1, 0.5, False, False, paddle.float16, False, True),
            (3, 1024, 4, 100, 1, 0.1, 0, False, False, paddle.float16, False, False),
            (4, 1024, 4, 128, 0.1, 1, 0, False, False, paddle.float16, True, True),
            (4, 1024, 4, 128, 0.1, 1, 0, True, False, paddle.float16, True, False),
            (2, 1500, 4, 128, 0.1, 10, 0, False, True, paddle.float16, False, True),
            (4, 2048, 8, 64, 0.1, 1, 0, False, True, paddle.float16, True, True),
        ]
    ],
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    mask_p: float,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    dtype,
    safe_gate: bool,
    disable_recompute: bool,
):
    paddle.seed(42)
    q = paddle.rand([B, T, H, D], dtype=dtype)
    k = paddle.rand([B, T, H, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    g = paddle.randn([B, T, H, D], dtype=paddle.float32 if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = paddle.randn([H], dtype=paddle.float32)
        dt_bias = paddle.randn([H * D], dtype=paddle.float32)
    else:
        g = F.log_sigmoid(g) / gate_logit_normalizer
        g = g * (paddle.rand(g.shape, dtype=g.dtype) > mask_p).cast(g.dtype)
    if safe_gate:
        lower_bound = -5.0
        if not use_gate_in_kernel:
            g = g.clip(-5, 0)
        naive_kda_gate_fn = naive_kda_lowerbound_gate
    else:
        lower_bound = None
        naive_kda_gate_fn = naive_kda_gate

    beta = paddle.randn([B, T, H], dtype=dtype).sigmoid()
    h0 = paddle.randn([B, H, D, D], dtype=paddle.float32)

    if use_gate_in_kernel:
        A_log.stop_gradient = False
        dt_bias.stop_gradient = False
    for t in [q, k, v, g, beta, h0]:
        t.stop_gradient = False

    do = paddle.randn(v.shape, dtype=v.dtype)
    dht = paddle.randn(h0.shape, dtype=h0.dtype)

    ref, ref_ht = naive_recurrent_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=(naive_kda_gate_fn(g, A_log, dt_bias) if use_gate_in_kernel else g.clone()),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    if use_gate_in_kernel:
        ref_dA = A_log.grad.clone()
        A_log.clear_gradient()
        ref_dbias = dt_bias.grad.clone()
        dt_bias.clear_gradient()
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        g.grad.clone(), beta.grad.clone(), h0.grad.clone()
    )
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    g.clear_gradient()
    beta.clear_gradient()
    h0.clear_gradient()

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, axis=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, axis=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        disable_recompute=disable_recompute,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    if use_gate_in_kernel:
        tri_dA = A_log.grad.clone()
        A_log.clear_gradient()
        tri_dbias = dt_bias.grad.clone()
        dt_bias.clear_gradient()
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        g.grad.clone(), beta.grad.clone(), h0.grad.clone()
    )

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.008)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.008)
    assert_close("dg", ref_dg, tri_dg, 0.02)
    assert_close("db", ref_db, tri_db, 0.02)
    if use_gate_in_kernel:
        assert_close("dA", ref_dA, tri_dA, 0.003, warning=True)
        assert_close("dbias", ref_dbias, tri_dbias, 0.008)
    assert_close("dh0", ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 64, 1, 1, paddle.float16),
            (2, 500, 3, 60, 1, 1, paddle.float16),
            (3, 1024, 4, 128, 0.1, 1, paddle.float16),
            (4, 2048, 8, 64, 0.1, 1, paddle.float16),
        ]
    ],
)
def test_chunk_transpose_state(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype,
):
    paddle.seed(42)
    q = paddle.rand([B, T, H, D], dtype=dtype)
    k = paddle.rand([B, T, H, D], dtype=dtype)
    v = paddle.rand([B, T, H, D], dtype=dtype)
    g = F.log_sigmoid(paddle.randn([B, T, H, D], dtype=paddle.float32)) / gate_logit_normalizer
    beta = paddle.randn([B, T, H], dtype=dtype).sigmoid()
    h0_kv = paddle.randn([B, H, D, D], dtype=paddle.float32)
    h0_vk = h0_kv.transpose([0, 1, 3, 2]).contiguous()

    for t in [q, k, v, g, beta, h0_kv, h0_vk]:
        t.stop_gradient = False

    do = paddle.randn(v.shape, dtype=v.dtype)
    dht_vk = paddle.randn([B, H, D, D], dtype=paddle.float32)
    dht_kv = dht_vk.transpose([0, 1, 3, 2]).contiguous()

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_vk.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        transpose_state_layout=True,
    )
    ((tri * do).sum() + (tri_ht * dht_vk).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        g.grad.clone(), beta.grad.clone(), h0_vk.grad.clone()
    )
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    g.clear_gradient()
    beta.clear_gradient()
    h0_vk.clear_gradient()

    ref, ref_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=F.normalize(k.clone(), p=2, axis=-1),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        scale=scale,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        transpose_state_layout=False,
    )
    ((ref * do).sum() + (ref_ht * dht_kv).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        g.grad.clone(), beta.grad.clone(), h0_kv.grad.clone()
    )

    assert_close("o", ref, tri, 1e-4)
    assert_close("ht", ref_ht, tri_ht.transpose([0, 1, 3, 2]), 1e-4)
    assert_close("dq", ref_dq, tri_dq, 1e-4)
    assert_close("dk", ref_dk, tri_dk, 1e-4)
    assert_close("dv", ref_dv, tri_dv, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("db", ref_db, tri_db, 1e-4)
    assert_close("dh0", ref_dh0, tri_dh0.transpose([0, 1, 3, 2]), 1e-4)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype", "use_gate_in_kernel", "safe_gate", "disable_recompute"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}-gate{}-safe_gate{}-disable_recompute{}".format(*test))
        for test in [
            (4, 60, 0.1, [0, 15], paddle.float16, True, False, False),
            (4, 64, 0.9, [0, 256, 500, 1000], paddle.float16, True, False, False),
            (4, 128, 0.5, [0, 256, 500, 1000], paddle.float16, False, False, False),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], paddle.float16, True, False, False),
            (4, 256, 0, [0, 100, 300, 1200, 3000, 4096], paddle.float16, False, True, True),
        ]
    ],
)
def test_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list,
    dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
):
    paddle.seed(42)
    cu_seqlens_t = paddle.to_tensor(cu_seqlens, dtype=paddle.int64)
    cu_seqlens_cpu = paddle.to_tensor(cu_seqlens, dtype=paddle.int64)
    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1

    q = paddle.randn([1, T, H, D], dtype=dtype)
    k = F.normalize(paddle.randn([1, T, H, D], dtype=paddle.float32), p=2, axis=-1).cast(dtype)
    v = paddle.randn([1, T, H, D], dtype=dtype)
    g = paddle.randn([1, T, H, D], dtype=paddle.float32 if not use_gate_in_kernel else dtype)
    if use_gate_in_kernel:
        A_log = paddle.log(paddle.uniform([1, 1, H, 1], dtype=paddle.float32, min=1, max=16))
        dt_bias = paddle.randn([H * D], dtype=paddle.float32)
    else:
        g = F.log_sigmoid(g)
        g = g * (paddle.rand(g.shape, dtype=g.dtype) > mask_p).cast(g.dtype)
    mask = (paddle.rand(g.shape, dtype=g.dtype) > mask_p).cast(g.dtype)
    g = g * mask + (1 - mask) * (-1000)
    if safe_gate:
        assert use_gate_in_kernel is False
        g = g.clip(-5, 0)

    beta = paddle.rand([1, T, H], dtype=dtype).sigmoid()
    h0 = paddle.randn([N, H, D, D], dtype=paddle.float32)

    for t in [q, k, v, g, beta, h0]:
        t.stop_gradient = False
    if use_gate_in_kernel:
        A_log.stop_gradient = False
        dt_bias.stop_gradient = False
    do = paddle.randn(v.shape, dtype=v.dtype)
    dht = paddle.rand(h0.shape, dtype=h0.dtype)

    tri, tri_ht = chunk_kda(
        q=F.normalize(q.clone(), p=2, axis=-1),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        A_log=(A_log.clone() if use_gate_in_kernel else None),
        dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        cu_seqlens_cpu=cu_seqlens_cpu,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dg, tri_db, tri_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        g.grad.clone(), beta.grad.clone(), h0.grad.clone()
    )
    q.clear_gradient()
    k.clear_gradient()
    v.clear_gradient()
    g.clear_gradient()
    beta.clear_gradient()
    h0.clear_gradient()
    if use_gate_in_kernel:
        tri_dA = A_log.grad.clone()
        A_log.clear_gradient()
        tri_dbias = dt_bias.grad.clone()
        dt_bias.clear_gradient()

    ref_list = []
    ref_ht_list = []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        ref_i, ref_ht_i = naive_recurrent_kda(
            q=F.normalize(q[:, s:e], p=2, axis=-1),
            k=k[:, s:e],
            v=v[:, s:e],
            beta=beta[:, s:e],
            g=(naive_kda_gate(g[:, s:e].cast(paddle.float32), A_log.cast(paddle.float32),
               dt_bias.cast(paddle.float32)) if use_gate_in_kernel else g[:, s:e]),
            initial_state=h0[i],
            output_final_state=True,
        )
        ref_list.append(ref_i)
        ref_ht_list.append(ref_ht_i)
    ref = paddle.concat(ref_list, axis=1)
    ref_ht = paddle.concat(ref_ht_list, axis=0)

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dg, ref_db, ref_dh0 = (
        q.grad.clone(), k.grad.clone(), v.grad.clone(),
        g.grad.clone(), beta.grad.clone(), h0.grad.clone()
    )
    if use_gate_in_kernel:
        ref_dA = A_log.grad.clone()
        ref_dbias = dt_bias.grad.clone()

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.007)
    assert_close("dk", ref_dk, tri_dk, 0.008)
    assert_close("dv", ref_dv, tri_dv, 0.007)
    assert_close("dg", ref_dg, tri_dg, 0.015)
    assert_close("db", ref_db, tri_db, 0.015)
    assert_close("dh0", ref_dh0, tri_dh0, 0.007)
    if use_gate_in_kernel:
        assert_close("dA", ref_dA, tri_dA, 0.008, warning=True)
        assert_close("dbias", ref_dbias, tri_dbias, 0.005)


@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype", "use_gate_in_kernel", "safe_gate", "disable_recompute"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}-gate{}-safe_gate{}-disable_recompute{}".format(*test))
        for test in [
            (4, 60, 0.1, [0, 8192], paddle.float16, True, False, False),
            (4, 64, 0.9, [0, 256, 500, 1000], paddle.float16, True, False, False),
            (4, 128, 0.5, [0, 256, 500, 1000], paddle.float16, False, False, False),
            (4, 100, 0, [0, 15, 100, 300, 1200, 2000], paddle.float16, True, False, False),
            (4, 256, 0, [0, 100, 300, 1200, 3000, 4096], paddle.float16, False, True, True),
        ]
    ],
)
def test_chunk_varlen_prefill(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list,
    dtype,
    use_gate_in_kernel: bool,
    safe_gate: bool,
    disable_recompute: bool,
):
    paddle.seed(42)
    with paddle.no_grad():
        cu_seqlens_t = paddle.to_tensor(cu_seqlens, dtype=paddle.int64)
        cu_seqlens_cpu = paddle.to_tensor(cu_seqlens, dtype=paddle.int64)
        T = cu_seqlens[-1]
        N = len(cu_seqlens) - 1

        q = paddle.randn([1, T, H, D], dtype=dtype)
        k = F.normalize(paddle.randn([1, T, H, D], dtype=paddle.float32), p=2, axis=-1).cast(dtype)
        v = paddle.randn([1, T, H, D], dtype=dtype)
        g = paddle.randn([1, T, H, D], dtype=paddle.float32 if not use_gate_in_kernel else dtype)
        if use_gate_in_kernel:
            A_log = paddle.log(paddle.uniform([1, 1, H, 1], dtype=paddle.float32, min=1, max=16))
            dt_bias = paddle.randn([H * D], dtype=paddle.float32)
        else:
            g = F.log_sigmoid(g)
            g = g * (paddle.rand(g.shape, dtype=g.dtype) > mask_p).cast(g.dtype)
        mask = (paddle.rand(g.shape, dtype=g.dtype) > mask_p).cast(g.dtype)
        g = g * mask + (1 - mask) * (-1000)
        if safe_gate:
            assert use_gate_in_kernel is False
            g = g.clip(-5, 0)

        beta = paddle.rand([1, T, H], dtype=dtype).sigmoid()
        h0 = paddle.randn([N, H, D, D], dtype=paddle.float32)

        tri, tri_ht = chunk_kda(
            q=F.normalize(q.clone(), p=2, axis=-1),
            k=k.clone(),
            v=v.clone(),
            g=g.clone(),
            beta=beta.clone(),
            A_log=(A_log.clone() if use_gate_in_kernel else None),
            dt_bias=(dt_bias.clone() if use_gate_in_kernel else None),
            initial_state=h0.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens_t,
            cu_seqlens_cpu=cu_seqlens_cpu,
            use_gate_in_kernel=use_gate_in_kernel,
            safe_gate=safe_gate,
            disable_recompute=disable_recompute,
        )

        ref_list = []
        ref_ht_list = []
        for i in range(N):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            ref_i, ref_ht_i = naive_recurrent_kda(
                q=F.normalize(q[:, s:e], p=2, axis=-1),
                k=k[:, s:e],
                v=v[:, s:e],
                beta=beta[:, s:e],
                g=(naive_kda_gate(g[:, s:e].cast(paddle.float32), A_log.cast(paddle.float32),
                   dt_bias.cast(paddle.float32)) if use_gate_in_kernel else g[:, s:e]),
                initial_state=h0[i],
                output_final_state=True,
            )
            ref_list.append(ref_i)
            ref_ht_list.append(ref_ht_i)
        ref = paddle.concat(ref_list, axis=1)
        ref_ht = paddle.concat(ref_ht_list, axis=0)

        assert_close("o", ref, tri, 0.005)
        assert_close("ht", ref_ht, tri_ht, 0.005)


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "HAS_BIAS", "LOWER_BOUND"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-bias{}-lowerbound{}".format(*test))
        for test in [
            (1, 2, 2, 12, False, -5.0),
            (1, 32, 2, 16, False, -5.0),
            (2, 64, 4, 32, False, -5.0),
            (4, 128, 8, 64, False, -5.0),
            (4, 128, 8, 128, False, None),
            (1, 2, 2, 12, True, None),
            (1, 32, 2, 16, True, None),
            (2, 64, 4, 32, True, None),
            (4, 128, 8, 64, True, None),
            (4, 128, 8, 128, True, None),
        ]
    ],
)
def test_gate(
    B: int,
    T: int,
    H: int,
    D: int,
    HAS_BIAS: bool,
    LOWER_BOUND,
):
    paddle.seed(42)
    g = paddle.randn([B, T, H, D], dtype=paddle.float32) * 10
    A_log = paddle.log(paddle.uniform([1, 1, H, 1], dtype=paddle.float32, min=1, max=16))
    dt_bias = paddle.randn([H * D], dtype=paddle.float32) if HAS_BIAS else None
    g.stop_gradient = False
    A_log.stop_gradient = False
    if dt_bias is not None:
        dt_bias.stop_gradient = False
    do = paddle.randn([B, T, H, D], dtype=paddle.float32)

    if LOWER_BOUND is not None:
        ref = naive_kda_lowerbound_gate(
            g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None, LOWER_BOUND
        )
    else:
        ref = naive_kda_gate(
            g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
        )
    tri = fused_kda_gate(
        g.clone(), A_log.clone(), dt_bias.clone() if dt_bias is not None else None,
        lower_bound=LOWER_BOUND
    )
    (ref * do).sum().backward(retain_graph=True)

    ref_dg = g.grad.clone()
    ref_dA = A_log.grad.clone()
    ref_dbias = dt_bias.grad.clone() if dt_bias is not None else None
    g.clear_gradient()
    A_log.clear_gradient()
    if dt_bias is not None:
        dt_bias.clear_gradient()

    ((tri * do).sum()).backward(retain_graph=True)
    tri_dg = g.grad.clone()
    tri_dA = A_log.grad.clone()
    tri_dbias = dt_bias.grad.clone() if dt_bias is not None else None

    assert_close("o", ref, tri, 1e-4)
    assert_close("dg", ref_dg, tri_dg, 1e-4)
    assert_close("dA", ref_dA, tri_dA, 1e-4)
    if HAS_BIAS:
        assert_close("dbias", ref_dbias, tri_dbias, 1e-4)


@pytest.mark.parametrize("dtype", [paddle.bfloat16])
def test_chunk_return_intermediate_states(dtype):
    """Test that return_intermediate_states=True works in inference mode and returns h with correct shape."""
    paddle.seed(42)
    B, T, H, D = 2, 1024, 4, 128
    chunk_size = 64

    with paddle.no_grad():
        q = paddle.randn([B, T, H, D], dtype=dtype)
        k = paddle.randn([B, T, H, D], dtype=dtype)
        v = paddle.randn([B, T, H, D], dtype=dtype)
        g = paddle.randn([B, T, H, D], dtype=dtype)
        beta = paddle.rand([B, T, H], dtype=dtype)

        # Test equal-length sequences
        o, final_state, h = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            return_intermediate_states=True,
            disable_recompute=False,
        )

        # Verify shapes
        assert list(o.shape) == [B, T, H, D], f"Output shape mismatch: {o.shape}"
        assert list(final_state.shape) == [B, H, D, D], f"Final state shape mismatch: {final_state.shape}"

        expected_nt = (T + chunk_size - 1) // chunk_size
        assert list(h.shape) == [B, expected_nt, H, D, D], f"h shape mismatch: {h.shape}"
        assert h.dtype == dtype, f"h dtype should be {dtype}, got: {h.dtype}"

        # Test variable-length sequences
        total_tokens = 1024
        N = 2
        seq_len = total_tokens // N
        cu_seqlens = paddle.to_tensor([0, seq_len, total_tokens], dtype=paddle.int64)

        q_varlen = paddle.randn([1, total_tokens, H, D], dtype=dtype)
        k_varlen = paddle.randn([1, total_tokens, H, D], dtype=dtype)
        v_varlen = paddle.randn([1, total_tokens, H, D], dtype=dtype)
        g_varlen = paddle.randn([1, total_tokens, H, D], dtype=dtype)
        beta_varlen = paddle.rand([1, total_tokens, H], dtype=dtype)

        o_varlen, final_state_varlen, h_varlen = chunk_kda(
            q=q_varlen,
            k=k_varlen,
            v=v_varlen,
            g=g_varlen,
            beta=beta_varlen,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            return_intermediate_states=True,
            disable_recompute=False,
        )

        assert list(o_varlen.shape) == [1, total_tokens, H, D], f"Varlen output shape mismatch: {o_varlen.shape}"
        assert list(final_state_varlen.shape) == [N, H, D, D], f"Varlen final state shape mismatch: {final_state_varlen.shape}"
        assert h_varlen.shape[0] == 1, f"Varlen h batch dim should be 1, got: {h_varlen.shape[0]}"
        assert list(h_varlen.shape[2:]) == [H, D, D], f"Varlen h dims mismatch: {h_varlen.shape[2:]}"
        assert h_varlen.dtype == dtype, f"Varlen h dtype should be {dtype}, got: {h_varlen.dtype}"
