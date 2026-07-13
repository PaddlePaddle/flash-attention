"""Accuracy tests for the fused block-score (block_logit) epilogue of the SM100
FlashAttention forward kernel.

When ``_flash_attn_fwd`` is called with a preallocated ``block_logit`` tensor it
additionally writes, for every (query, key-block) pair, the maximum of the
post-score_mod, post-mask, SCALED ``softmax_scale * q @ k^T`` logit -- i.e. the
exact value fed into softmax, INCLUDING any score_mod bias. Storing the scaled
value (rather than the raw ``q @ k^T``) puts every head on one head-independent
scale, so a downstream ``block_logit - LSE`` gives log(max attention weight in
the block), which is comparable across heads. It is computed inside the softmax
epilogue at (nearly) zero extra cost. Downstream this feeds a Top-K block selection
(HySparse sparse attention), so it must:

  * respect exactly the same causal / flashmask masking as the attention itself
    (masked positions contribute ``-inf`` and never leak into a block max);
  * be numerically faithful to the reference block max on the finite positions;
  * leave the returned ``(out, lse)`` unchanged.

These tests run only on SM 10.x (Blackwell); the whole cute interface asserts
that anyway. Run:

    PYTHONPATH=. python -m pytest flashmask/tests/test_block_score_fusion.py -q
"""

import math

import paddle
import pytest

from flash_mask.cute.interface import _flash_attn_fwd

_NEG_INF = float("-inf")


def _sm100_or_skip():
    if not paddle.device.is_compiled_with_cuda():
        pytest.skip("CUDA not available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        pytest.skip(f"block-score fusion requires SM 10.x; got SM {major}.x")


def _num_blocks(seqlen_k, block_size):
    return (seqlen_k + block_size - 1) // block_size


def ref_block_logit(q, k, causal, block_size, num_blocks, scale=1.0):
    """Reference: per-(query, key-block) max of the SCALED ``scale * q@k^T`` logit,
    with the same causal masking the kernel applies (bottom-right aligned).

    q: [B, S, H, D]  k: [B, Sk, H, D]  ->  [B, H, S, num_blocks] (fp32)
    """
    b, s, h, d = q.shape
    sk = k.shape[1]
    qf = q.astype("float32").transpose([0, 2, 1, 3])   # [B,H,S,D]
    kf = k.astype("float32").transpose([0, 2, 1, 3])   # [B,H,Sk,D]
    scores = paddle.matmul(qf, kf, transpose_y=True) * scale  # [B,H,S,Sk] scaled logit
    if causal:
        row = paddle.arange(s).reshape([s, 1])
        col = paddle.arange(sk).reshape([1, sk])
        # bottom-right aligned causal: mask key j when j > i + (sk - s)
        masked = (col > row + (sk - s)).reshape([1, 1, s, sk])
        scores = paddle.where(masked, paddle.full_like(scores, _NEG_INF), scores)
    pad = num_blocks * block_size - sk
    if pad > 0:
        scores = paddle.concat(
            [scores, paddle.full([b, h, s, pad], _NEG_INF, dtype="float32")], axis=-1
        )
    scores = scores.reshape([b, h, s, num_blocks, block_size])
    return scores.max(axis=-1)  # [B,H,S,num_blocks]


def ref_block_logit_bf16(q, k, causal, block_size, num_blocks, scale=1.0):
    """Same as ref_block_logit but with a bf16-precision matmul, to bound the
    finite-value error against the kernel's bf16 tensor-core MMA."""
    b, s, h, d = q.shape
    sk = k.shape[1]
    qf = q.transpose([0, 2, 1, 3])
    kf = k.transpose([0, 2, 1, 3])
    scores = paddle.matmul(qf, kf, transpose_y=True).astype("float32") * scale
    if causal:
        row = paddle.arange(s).reshape([s, 1])
        col = paddle.arange(sk).reshape([1, sk])
        masked = (col > row + (sk - s)).reshape([1, 1, s, sk])
        scores = paddle.where(masked, paddle.full_like(scores, _NEG_INF), scores)
    pad = num_blocks * block_size - sk
    if pad > 0:
        scores = paddle.concat(
            [scores, paddle.full([b, h, s, pad], _NEG_INF, dtype="float32")], axis=-1
        )
    scores = scores.reshape([b, h, s, num_blocks, block_size])
    return scores.max(axis=-1)


def _run_fwd(q, k, v, causal, block_size):
    b, s, h, d = q.shape
    sk = k.shape[1]
    dv = v.shape[-1]
    nb = _num_blocks(sk, block_size)
    sm_scale = 1.0 / math.sqrt(d)
    block_logit = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    out, lse = _flash_attn_fwd(
        q, k, v,
        softmax_scale=sm_scale,
        causal=causal,
        return_lse=True,
        block_logit=block_logit,
        block_size=block_size,
        pack_gqa=False,
    )
    return out, lse, block_logit, sm_scale


def _assert_block_logit(got, ref, ref_bf16, tol):
    """got/ref: [B,H,S,num_blocks] fp32. Check the masked pattern matches and the
    finite values match the (bf16) reference within tol.

    A block is "masked" when every key column in it was masked out, so its max is
    -inf. Different code paths spell that either as true -inf (kernel) or as
    -FLT_MAX (paddle's reduce over a -inf-filled slice), so we treat any value
    below -1e30 as masked rather than comparing the bit pattern directly.
    """
    import numpy as np

    got_np = got.astype("float32").numpy()
    ref_np = ref.astype("float32").numpy()
    ref_bf16_np = ref_bf16.astype("float32").numpy()

    MASK_THR = -1e30
    got_masked = got_np <= MASK_THR
    ref_masked = ref_np <= MASK_THR
    n_pattern_mismatch = int((got_masked != ref_masked).sum())
    assert n_pattern_mismatch == 0, f"masked pattern mismatch: {n_pattern_mismatch} entries"

    finite = ~ref_masked
    assert finite.any(), "no finite reference entries to compare"
    diff = np.abs(got_np[finite] - ref_bf16_np[finite])
    maxdiff = float(diff.max())
    assert maxdiff <= tol, f"block_logit finite max|diff|={maxdiff:.4e} > tol={tol:.4e}"
    return maxdiff


# ---------------------------------------------------------------------------
# Multi-dimension accuracy: head dims, head counts, causal/non-causal.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "b, s, sk, h, d, dv",
    [
        (1, 256, 256, 8, 64, 64),
        (2, 512, 512, 4, 128, 128),
        (1, 512, 512, 8, 256, 256),   # the real HySparse full-attn config (split-D)
        (1, 1024, 1024, 2, 256, 256),
        (2, 640, 640, 6, 128, 128),
    ],
)
@pytest.mark.parametrize("block_size", [64])
def test_block_logit_multi_dim(b, s, sk, h, d, dv, causal, block_size):
    _sm100_or_skip()
    paddle.seed(0)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, sk, h, d], dtype="bfloat16")
    v = paddle.randn([b, sk, h, dv], dtype="bfloat16")

    _, _, block_logit, sm_scale = _run_fwd(q, k, v, causal, block_size)
    nb = _num_blocks(sk, block_size)
    ref = ref_block_logit(q, k, causal, block_size, nb, scale=sm_scale)
    ref_bf16 = ref_block_logit_bf16(q, k, causal, block_size, nb, scale=sm_scale)
    # block_logit now stores the SCALED logit (~O(1)); tol tracks the bf16 MMA
    # error scaled by softmax_scale (raw tol 0.06*sqrt(d) times sm_scale=1/sqrt(d)).
    tol = 0.06 * math.sqrt(d) * sm_scale
    _assert_block_logit(block_logit, ref, ref_bf16, tol)


# ---------------------------------------------------------------------------
# block_size variants (must divide the kernel n_block_size=128).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("causal", [True, False])
def test_block_logit_block_sizes(block_size, causal):
    _sm100_or_skip()
    b, s, sk, h, d, dv = 1, 512, 512, 4, 128, 128
    paddle.seed(1)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, sk, h, d], dtype="bfloat16")
    v = paddle.randn([b, sk, h, dv], dtype="bfloat16")

    _, _, block_logit, sm_scale = _run_fwd(q, k, v, causal, block_size)
    nb = _num_blocks(sk, block_size)
    assert block_logit.shape == [b, h, s, nb]
    ref = ref_block_logit(q, k, causal, block_size, nb, scale=sm_scale)
    ref_bf16 = ref_block_logit_bf16(q, k, causal, block_size, nb, scale=sm_scale)
    _assert_block_logit(block_logit, ref, ref_bf16, tol=0.06 * math.sqrt(d) * sm_scale)


# ---------------------------------------------------------------------------
# Corner cases: ragged seqlens, seqlen_k not a multiple of block_size,
# seqlen shorter than a block, single query, seqlen_q != seqlen_k.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "b, s, sk, h, d, dv, block_size",
    [
        (1, 250, 250, 4, 128, 128, 64),    # sk not multiple of block_size -> partial last block
        (1, 130, 130, 4, 128, 128, 64),    # just over two blocks
        (1, 63, 63, 4, 128, 128, 64),      # shorter than one block
        (1, 1, 1, 4, 128, 128, 64),        # single token
        (2, 300, 512, 4, 128, 128, 64),    # seqlen_q < seqlen_k (bottom-right causal)
        (2, 512, 300, 4, 128, 128, 64),    # seqlen_q > seqlen_k
        (1, 511, 513, 2, 256, 256, 64),    # odd ragged, split-D head dim
    ],
)
def test_block_logit_corner_cases(b, s, sk, h, d, dv, block_size, causal):
    _sm100_or_skip()
    paddle.seed(2)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, sk, h, d], dtype="bfloat16")
    v = paddle.randn([b, sk, h, dv], dtype="bfloat16")

    _, _, block_logit, sm_scale = _run_fwd(q, k, v, causal, block_size)
    nb = _num_blocks(sk, block_size)
    assert block_logit.shape == [b, h, s, nb]
    ref = ref_block_logit(q, k, causal, block_size, nb, scale=sm_scale)
    ref_bf16 = ref_block_logit_bf16(q, k, causal, block_size, nb, scale=sm_scale)
    _assert_block_logit(block_logit, ref, ref_bf16, tol=0.06 * math.sqrt(d) * sm_scale)


# ---------------------------------------------------------------------------
# The fused epilogue must not perturb the attention output / lse. Compare a run
# WITH block_logit against a run WITHOUT it (same inputs).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("d", [128, 256])
def test_block_logit_does_not_change_output(causal, d):
    _sm100_or_skip()
    b, s, sk, h = 1, 512, 512, 4
    paddle.seed(3)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, sk, h, d], dtype="bfloat16")
    v = paddle.randn([b, sk, h, d], dtype="bfloat16")
    sm_scale = 1.0 / math.sqrt(d)

    out_ref, lse_ref = _flash_attn_fwd(
        q, k, v, softmax_scale=sm_scale, causal=causal, return_lse=True, pack_gqa=False
    )
    out_fused, lse_fused, _, _ = _run_fwd(q, k, v, causal, block_size=64)

    import numpy as np

    o0 = out_ref.astype("float32").numpy()
    o1 = out_fused.astype("float32").numpy()
    assert np.array_equal(o0, o1), (
        f"out changed by fusion: max|diff|={np.abs(o0 - o1).max():.3e}"
    )
    l0 = lse_ref.astype("float32").numpy()
    l1 = lse_fused.astype("float32").numpy()
    assert np.allclose(l0, l1, atol=0, rtol=0), (
        f"lse changed by fusion: max|diff|={np.abs(l0 - l1).max():.3e}"
    )


# ---------------------------------------------------------------------------
# GQA: qhead_per_kvhead > 1. block_logit is indexed by the QUERY head, and each
# query head must see its own kv head broadcast. Reference repeats kv heads.
# ---------------------------------------------------------------------------
def ref_block_logit_gqa(q, k, causal, block_size, num_blocks, scale=1.0):
    b, s, hq, d = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    qf = q.astype("float32").transpose([0, 2, 1, 3])   # [B,Hq,S,D]
    kf = k.astype("float32").transpose([0, 2, 1, 3])   # [B,Hkv,Sk,D]
    kf = paddle.repeat_interleave(kf, hq // hkv, axis=1)
    scores = paddle.matmul(qf, kf, transpose_y=True) * scale
    if causal:
        row = paddle.arange(s).reshape([s, 1])
        col = paddle.arange(sk).reshape([1, sk])
        masked = (col > row + (sk - s)).reshape([1, 1, s, sk])
        scores = paddle.where(masked, paddle.full_like(scores, _NEG_INF), scores)
    pad = num_blocks * block_size - sk
    if pad > 0:
        scores = paddle.concat(
            [scores, paddle.full([b, hq, s, pad], _NEG_INF, dtype="float32")], axis=-1
        )
    scores = scores.reshape([b, hq, s, num_blocks, block_size])
    return scores.max(axis=-1)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("hq, hkv", [(8, 2), (8, 1), (4, 2)])
def test_block_logit_gqa(hq, hkv, causal):
    _sm100_or_skip()
    b, s, sk, d = 1, 512, 512, 128
    block_size = 64
    paddle.seed(5)
    q = paddle.randn([b, s, hq, d], dtype="bfloat16")
    k = paddle.randn([b, sk, hkv, d], dtype="bfloat16")
    v = paddle.randn([b, sk, hkv, d], dtype="bfloat16")
    nb = _num_blocks(sk, block_size)
    block_logit = paddle.full([b, hq, s, nb], _NEG_INF, dtype="float32")
    sm_scale = 1.0 / math.sqrt(d)
    _flash_attn_fwd(
        q, k, v, softmax_scale=sm_scale, causal=causal,
        return_lse=True, block_logit=block_logit, block_size=block_size, pack_gqa=False,
    )
    ref = ref_block_logit_gqa(q, k, causal, block_size, nb, scale=sm_scale)
    ref_bf16 = ref  # inputs are bf16 already; fp32 matmul of bf16 values
    _assert_block_logit(block_logit, ref, ref_bf16, tol=0.06 * math.sqrt(d) * sm_scale)


# ---------------------------------------------------------------------------
# Negative: block_logit is indexed by the QUERY head and the query row. Under
# pack_gqa=True the query heads are packed into the M/row dim and head_idx is
# the KV head, so the write would target wrong locations. The interface MUST
# reject block_logit + pack_gqa=True. pack_gqa also DEFAULTS to
# (qhead_per_kvhead > 1), so a GQA caller who forgets to pass pack_gqa=False
# must still be rejected (not silently produce garbage).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pack_gqa", [True, None])
def test_block_logit_rejects_pack_gqa(pack_gqa):
    _sm100_or_skip()
    b, s, sk, hq, hkv, d = 1, 512, 512, 8, 2, 128
    block_size = 64
    paddle.seed(6)
    q = paddle.randn([b, s, hq, d], dtype="bfloat16")
    k = paddle.randn([b, sk, hkv, d], dtype="bfloat16")
    v = paddle.randn([b, sk, hkv, d], dtype="bfloat16")
    nb = _num_blocks(sk, block_size)
    block_logit = paddle.full([b, hq, s, nb], _NEG_INF, dtype="float32")
    sm_scale = 1.0 / math.sqrt(d)
    # pack_gqa=None -> defaults to (qhead_per_kvhead=4 > 1) -> True -> must reject.
    with pytest.raises(AssertionError, match="pack_gqa"):
        _flash_attn_fwd(
            q, k, v, softmax_scale=sm_scale, causal=False,
            return_lse=True, block_logit=block_logit, block_size=block_size,
            pack_gqa=pack_gqa,
        )


# ---------------------------------------------------------------------------
# Large multi-n-tile (sk >> n_block_size=128): exercises cross-tile column
# addressing (block idx = n_block * blocks_per_ntile + b) and confirms that for
# NON-causal every block is written (no reliance on the -inf pre-fill).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_block_logit_large_multitile_noncausal(block_size):
    _sm100_or_skip()
    import numpy as np

    b, s, sk, h, d = 1, 2048, 2048, 2, 128
    paddle.seed(7)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, sk, h, d], dtype="bfloat16")
    v = paddle.randn([b, sk, h, d], dtype="bfloat16")
    nb = _num_blocks(sk, block_size)
    sentinel = 987654.0
    block_logit = paddle.full([b, h, s, nb], sentinel, dtype="float32")
    sm_scale = 1.0 / math.sqrt(d)
    _flash_attn_fwd(
        q, k, v, softmax_scale=sm_scale, causal=False,
        return_lse=True, block_logit=block_logit, block_size=block_size, pack_gqa=False,
    )
    got = block_logit.numpy()
    # Non-causal, sk a multiple of block_size => every block visited & written.
    n_unwritten = int((got == sentinel).sum())
    assert n_unwritten == 0, f"{n_unwritten} blocks left unwritten in a full non-causal run"
    ref = ref_block_logit(q, k, False, block_size, nb, scale=sm_scale).numpy()
    maxdiff = float(np.abs(got - ref).max())
    assert maxdiff <= 0.06 * math.sqrt(d) * sm_scale, f"large multitile max|diff|={maxdiff:.4e}"


# ---------------------------------------------------------------------------
# Contract: the kernel only writes key-blocks the attention loop VISITS. Under
# causal, fully-future n-tiles are skipped entirely, so their block_logit
# entries are NEVER written and keep whatever the caller put there. This test
# pins that contract (init with a poison value): reachable blocks must be
# overwritten (no poison), and the masked/finite pattern must match the -inf
# reference on the entries the kernel is responsible for. A regression that
# starts writing future blocks -- or stops writing reachable ones -- is caught.
# ---------------------------------------------------------------------------
def test_block_logit_unvisited_future_blocks_untouched_causal():
    _sm100_or_skip()
    import numpy as np

    b, s, sk, h, d = 1, 512, 512, 2, 128
    block_size = 64
    paddle.seed(8)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, sk, h, d], dtype="bfloat16")
    v = paddle.randn([b, sk, h, d], dtype="bfloat16")
    nb = _num_blocks(sk, block_size)
    poison = 987654.0
    sm_scale = 1.0 / math.sqrt(d)

    # Run twice: once with the correct -inf init (functional), once with poison
    # to observe exactly which entries the kernel writes.
    bl_inf = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    _flash_attn_fwd(q, k, v, softmax_scale=sm_scale, causal=True,
                    return_lse=True, block_logit=bl_inf, block_size=block_size, pack_gqa=False)
    bl_poison = paddle.full([b, h, s, nb], poison, dtype="float32")
    _flash_attn_fwd(q, k, v, softmax_scale=sm_scale, causal=True,
                    return_lse=True, block_logit=bl_poison, block_size=block_size, pack_gqa=False)

    inf_np = bl_inf.astype("float32").numpy()
    poison_np = bl_poison.astype("float32").numpy()

    written = poison_np != poison            # entries the kernel actually wrote
    # (1) Wherever the kernel wrote, both runs must agree bit-for-bit.
    assert np.array_equal(inf_np[written], poison_np[written]), "written entries differ across inits"
    # (2) The -inf run must equal the reference everywhere (masked + finite).
    ref = ref_block_logit(q, k, True, block_size, nb, scale=sm_scale).numpy()
    _assert_block_logit(bl_inf, paddle.to_tensor(ref), paddle.to_tensor(ref),
                        tol=0.06 * math.sqrt(d) * sm_scale)
    # (3) There MUST exist future blocks the kernel skipped -> proves the caller
    #     is responsible for pre-filling them (documents the API contract).
    assert (~written).any(), "expected some fully-future blocks to be left unwritten under causal"
    # Every skipped entry is a masked (-inf) entry in the correct reference.
    assert (ref[~written] <= -1e30).all(), "an unwritten entry was NOT a masked block in the reference"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
