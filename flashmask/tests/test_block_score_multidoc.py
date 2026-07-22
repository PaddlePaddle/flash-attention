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

"""Multi-document accuracy tests for the fused block-score (block_logit) epilogue.

``test_block_score_fusion.py`` only exercises the ABSOLUTE (single-document,
bos=0) bucketing fast path -- it never passes ``block_bos``. This file closes
that gap: it drives the DOCUMENT-RELATIVE bucketing path (the
``if const_expr(self.has_block_bos)`` branch of ``softmax_step`` in
``flash_fwd_sm100.py``), which computes each key column's block as

    rel = floor((abs_col - bos) / block_size)

with a per-query-row ``bos`` (document start). A bug in the carry / straddle /
negative-base arithmetic there would silently mis-place every block coordinate
and corrupt the HySparse Top-K selection, yet go completely uncaught today.

The test mirrors the PRODUCTION caller exactly (``block_score_fa4.py``):
document masking flows through FA4's flashmask ``startend_row_indices`` (the
per-token exclusive-doc-end convention shared with
``paddlefleet.transformer.multi_latent_attention.build_hysparse_valid_range``
and ``utils.get_doc_lens``) plus ``causal=True``; the per-row document start
``bos == valid_range[..., 0]`` is threaded in as ``block_bos``.

Two independent oracles are used:

  * a DIRECT host reference (``ref_block_logit_multidoc``) that buckets the
    same-document, causal scores by document-relative block from first
    principles; and
  * PACK-EQUIVALENCE -- the packed (bos>0) block_logit for each document must be
    bit-identical to running that document ALONE as a standalone causal
    sequence (bos=0). This is the property the feature exists to guarantee.

These tests run only on SM 10.x (Blackwell); they skip otherwise.

    PYTHONPATH=. python -m pytest flashmask/tests/test_block_score_multidoc.py -q
"""

import math

import numpy as np
import paddle
import pytest

from flash_mask.cute.interface import _flash_attn_fwd

_NEG_INF = float("-inf")
_MASK_THR = -1e30  # any value below this is treated as a masked (-inf) block


def _sm100_or_skip():
    if not paddle.device.is_compiled_with_cuda():
        pytest.skip("CUDA not available")
    major = paddle.device.cuda.get_device_capability()[0]
    if major != 10:
        pytest.skip(f"block-score fusion requires SM 10.x; got SM {major}.x")


def _num_blocks(seqlen_k, block_size):
    return (seqlen_k + block_size - 1) // block_size


def _doc_bounds(doc_lens):
    """[(start, end)] cumulative document boundaries for the packed sequence."""
    bounds, off = [], 0
    for L in doc_lens:
        bounds.append((off, off + L))
        off += L
    return bounds


def _causal_document_flashmask(doc_lens, batch=1):
    """Flashmask ``startend_row_indices`` [B, 1, S, 1] int32 for causal-document.

    Value at key column ``t`` is the EXCLUSIVE end of the document containing
    ``t`` -- the exact convention ``generate_causal_document_mask`` uses and that
    ``build_hysparse_valid_range`` / ``get_doc_lens`` read from ``[:, 0, :, 0]``.
    Combined with ``causal=True`` this yields per-token same-document causal
    masking (query ``i`` sees key ``j`` iff ``j <= i`` and ``i < doc_end(j)``).
    """
    s = sum(doc_lens)
    de = np.zeros([s], dtype=np.int32)
    for ds, dee in _doc_bounds(doc_lens):
        de[ds:dee] = dee
    t = paddle.to_tensor(de, dtype="int32").reshape([1, 1, s, 1])
    return t.expand([batch, 1, s, 1]).contiguous()


def _block_bos_from_doc_lens(doc_lens, batch=1):
    """Per-query document start ``bos`` [B, S] int32 (== valid_range[..., 0])."""
    s = sum(doc_lens)
    bos = np.zeros([s], dtype=np.int32)
    for ds, dee in _doc_bounds(doc_lens):
        bos[ds:dee] = ds
    t = paddle.to_tensor(bos, dtype="int32").reshape([1, s])
    return t.expand([batch, s]).contiguous()


def _scores_bhss(q, k, scale, as_bf16_matmul):
    """Scaled logits [B,H,S,Sk] fp32. ``as_bf16_matmul`` keeps the QK matmul in
    bf16 (kernel-faithful) vs. fp32 (exact) so the caller can bound bf16 error."""
    qf = q.transpose([0, 2, 1, 3])   # [B,H,S,D]
    kf = k.transpose([0, 2, 1, 3])   # [B,H,Sk,D]
    if as_bf16_matmul:
        scores = paddle.matmul(qf, kf, transpose_y=True).astype("float32") * scale
    else:
        scores = paddle.matmul(
            qf.astype("float32"), kf.astype("float32"), transpose_y=True
        ) * scale
    return scores.numpy()  # [B,H,S,Sk]


def _bucket_multidoc(scores_np, doc_lens, block_size, num_blocks):
    """Document-relative block max of same-doc causal scores -> [B,H,S,nb] fp32.

    For query row ``i`` in document ``[ds, de)`` the valid keys are the causal
    same-document columns ``[ds, i]``; each is bucketed to the DOCUMENT-relative
    block ``(c - ds) // block_size``. Blocks with no valid column stay -inf.
    """
    b, h, s, _ = scores_np.shape
    out = np.full([b, h, s, num_blocks], _NEG_INF, dtype=np.float32)
    for ds, de in _doc_bounds(doc_lens):
        for i in range(ds, de):
            ncols = i - ds + 1
            rel = (np.arange(ncols) // block_size)  # relative col -> rel block
            row = scores_np[:, :, i, ds:i + 1]      # [b,h,ncols]
            for j in range(int(rel.max()) + 1):
                sel = rel == j
                if sel.any():
                    out[:, :, i, j] = np.maximum(out[:, :, i, j], row[:, :, sel].max(axis=-1))
    return out


def ref_block_logit_multidoc(q, k, doc_lens, block_size, num_blocks, scale, bf16_matmul):
    return _bucket_multidoc(
        _scores_bhss(q, k, scale, bf16_matmul), doc_lens, block_size, num_blocks
    )


def _run_multidoc(q, k, v, doc_lens, block_size, batch):
    """Packed multi-doc run mirroring block_score_fa4: causal + doc flashmask +
    per-row bos. Returns (block_logit [B,H,S,nb], sm_scale)."""
    b, s, h, d = q.shape
    nb = _num_blocks(s, block_size)
    sm_scale = 1.0 / math.sqrt(d)
    block_logit = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    startend = _causal_document_flashmask(doc_lens, batch=batch)
    block_bos = _block_bos_from_doc_lens(doc_lens, batch=batch)
    _flash_attn_fwd(
        q, k, v,
        softmax_scale=sm_scale,
        causal=True,
        return_lse=True,
        startend_row_indices=startend,
        block_logit=block_logit,
        block_size=block_size,
        block_bos=block_bos,
        pack_gqa=False,
    )
    return block_logit, sm_scale


def _run_solo(q, k, v, block_size):
    """Single standalone causal document (bos=0, no flashmask, no block_bos)."""
    b, s, h, d = q.shape
    nb = _num_blocks(s, block_size)
    sm_scale = 1.0 / math.sqrt(d)
    block_logit = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    _flash_attn_fwd(
        q, k, v,
        softmax_scale=sm_scale,
        causal=True,
        return_lse=True,
        block_logit=block_logit,
        block_size=block_size,
        pack_gqa=False,
    )
    return block_logit


def _assert_block_logit(got, ref_fp32, ref_bf16, tol):
    """Masked pattern must match the fp32 reference bit-for-bit; finite values
    must match the bf16 reference within ``tol``."""
    got_np = np.asarray(got.astype("float32").numpy())
    ref_np = np.asarray(ref_fp32)
    ref_bf16_np = np.asarray(ref_bf16)
    assert got_np.shape == ref_np.shape, f"shape {got_np.shape} != {ref_np.shape}"
    got_masked = got_np <= _MASK_THR
    ref_masked = ref_np <= _MASK_THR
    n_bad = int((got_masked != ref_masked).sum())
    assert n_bad == 0, f"masked pattern mismatch: {n_bad} entries"
    finite = ~ref_masked
    assert finite.any(), "no finite reference entries to compare"
    maxdiff = float(np.abs(got_np[finite] - ref_bf16_np[finite]).max())
    assert maxdiff <= tol, f"finite max|diff|={maxdiff:.4e} > tol={tol:.4e}"
    return maxdiff


# Document layouts. Deliberately NOT 64-aligned (the packed-training regime):
# doc starts land mid-block so the relative-bucket carry/straddle path is hit.
#   [40,88,133,27] -> starts 0,40,128,261 (40%64=40, 261%64=5 straddle; 128 aligned)
#   [64,128,96]    -> starts 0,64,192 (all 64-aligned: carry must degenerate to 0)
#   [50,100,70]    -> starts 0,50,150
#   [1,200,87]     -> a length-1 doc then a long one (tiny-doc corner)
#   [37,91,160]    -> starts 0,37,128
_DOC_LAYOUTS = [
    [40, 88, 133, 27],
    [64, 128, 96],
    [50, 100, 70],
    [1, 200, 87],
    [37, 91, 160],
]


# ---------------------------------------------------------------------------
# (1) Direct host reference: document-relative block max from first principles.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc_lens", _DOC_LAYOUTS)
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("h, d", [(4, 128), (2, 256)])  # d=256 exercises split-D
def test_multidoc_direct_reference(doc_lens, block_size, h, d):
    _sm100_or_skip()
    b = 1
    s = sum(doc_lens)
    paddle.seed(0)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, s, h, d], dtype="bfloat16")
    v = paddle.randn([b, s, h, d], dtype="bfloat16")

    block_logit, sm_scale = _run_multidoc(q, k, v, doc_lens, block_size, batch=b)
    nb = _num_blocks(s, block_size)
    assert list(block_logit.shape) == [b, h, s, nb]
    ref = ref_block_logit_multidoc(q, k, doc_lens, block_size, nb, sm_scale, bf16_matmul=False)
    ref_bf16 = ref_block_logit_multidoc(q, k, doc_lens, block_size, nb, sm_scale, bf16_matmul=True)
    _assert_block_logit(block_logit, ref, ref_bf16, tol=0.06 * math.sqrt(d) * sm_scale)


# ---------------------------------------------------------------------------
# (2) Pack-equivalence: each packed document's block_logit must be bit-identical
#     to running that document ALONE (bos=0). This is the property the fixture
#     exists to guarantee -- if it holds, document-relative bucketing is correct.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc_lens", _DOC_LAYOUTS)
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("h, d", [(4, 128), (2, 256)])
def test_multidoc_pack_equivalence(doc_lens, block_size, h, d):
    _sm100_or_skip()
    b = 1
    s = sum(doc_lens)
    paddle.seed(1)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, s, h, d], dtype="bfloat16")
    v = paddle.randn([b, s, h, d], dtype="bfloat16")

    packed, _ = _run_multidoc(q, k, v, doc_lens, block_size, batch=b)
    packed_np = packed.astype("float32").numpy()

    for ds, de in _doc_bounds(doc_lens):
        L = de - ds
        nb_d = _num_blocks(L, block_size)
        # Standalone run of just this document (contiguous slice, same tokens).
        qd = q[:, ds:de].contiguous()
        kd = k[:, ds:de].contiguous()
        vd = v[:, ds:de].contiguous()
        solo = _run_solo(qd, kd, vd, block_size).astype("float32").numpy()  # [b,h,L,nb_d]

        packed_doc = packed_np[:, :, ds:de, :]  # [b,h,L,nb_packed]
        # (a) relative blocks that exist in the standalone doc must match exactly.
        got = packed_doc[:, :, :, :nb_d]
        # Compare masked pattern + finite values bit-for-bit (same kernel, same
        # tokens, identical document-relative masking => bit-identical).
        got_masked = got <= _MASK_THR
        solo_masked = solo <= _MASK_THR
        assert np.array_equal(got_masked, solo_masked), (
            f"doc[{ds},{de}) bs={block_size}: masked pattern differs packed vs solo"
        )
        fin = ~solo_masked
        assert np.array_equal(got[fin], solo[fin]), (
            f"doc[{ds},{de}) bs={block_size}: finite block_logit not bit-identical "
            f"packed vs solo (max|diff|={np.abs(got[fin] - solo[fin]).max():.3e})"
        )
        # (b) relative blocks BEYOND the document have no valid key -> stay -inf.
        beyond = packed_doc[:, :, :, nb_d:]
        assert (beyond <= _MASK_THR).all(), (
            f"doc[{ds},{de}) bs={block_size}: out-of-document relative blocks not -inf"
        )


# ---------------------------------------------------------------------------
# (3) Focused straddle: a document whose start is deep inside a block forces the
#     carry (mi >= t) path for many columns. Pin it against the direct reference.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_multidoc_unaligned_start_straddle(block_size):
    _sm100_or_skip()
    # Second doc starts at 40: for bs=64 that is 40 tokens into a block (heavy
    # straddle); for bs=32 it is 8 into the 2nd block; for bs=128, 40 into block 0.
    doc_lens = [40, 200, 72]
    b, h, d = 1, 4, 128
    s = sum(doc_lens)
    paddle.seed(2)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, s, h, d], dtype="bfloat16")
    v = paddle.randn([b, s, h, d], dtype="bfloat16")

    block_logit, sm_scale = _run_multidoc(q, k, v, doc_lens, block_size, batch=b)
    nb = _num_blocks(s, block_size)
    ref = ref_block_logit_multidoc(q, k, doc_lens, block_size, nb, sm_scale, bf16_matmul=False)
    ref_bf16 = ref_block_logit_multidoc(q, k, doc_lens, block_size, nb, sm_scale, bf16_matmul=True)
    _assert_block_logit(block_logit, ref, ref_bf16, tol=0.06 * math.sqrt(d) * sm_scale)


# ---------------------------------------------------------------------------
# (4) Degenerate: block_bos all-zero (single document semantics) must reproduce
#     the ABSOLUTE bucketing -- i.e. equal a plain single-doc causal run.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_size", [32, 64, 128])
def test_multidoc_zero_bos_equals_absolute(block_size):
    _sm100_or_skip()
    b, s, h, d = 1, 512, 4, 128
    paddle.seed(3)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, s, h, d], dtype="bfloat16")
    v = paddle.randn([b, s, h, d], dtype="bfloat16")
    nb = _num_blocks(s, block_size)
    sm_scale = 1.0 / math.sqrt(d)

    # Explicit all-zero block_bos (relative path forced, bos==0 everywhere).
    bl_rel = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    _flash_attn_fwd(
        q, k, v, softmax_scale=sm_scale, causal=True, return_lse=True,
        block_logit=bl_rel, block_size=block_size,
        block_bos=paddle.zeros([b, s], dtype="int32"), pack_gqa=False,
    )
    # No block_bos at all (absolute fast path).
    bl_abs = paddle.full([b, h, s, nb], _NEG_INF, dtype="float32")
    _flash_attn_fwd(
        q, k, v, softmax_scale=sm_scale, causal=True, return_lse=True,
        block_logit=bl_abs, block_size=block_size, pack_gqa=False,
    )
    a = bl_rel.astype("float32").numpy()
    c = bl_abs.astype("float32").numpy()
    a_m, c_m = a <= _MASK_THR, c <= _MASK_THR
    assert np.array_equal(a_m, c_m), "bos=0 relative path masked pattern != absolute path"
    assert np.array_equal(a[~c_m], c[~c_m]), (
        "bos=0 relative path finite values != absolute path (should be bit-identical)"
    )


# ---------------------------------------------------------------------------
# (5) GQA layout with a document mask: block_logit is indexed by the QUERY head;
#     each query head sees its own kv-head broadcast, bucketed document-relative.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hq, hkv", [(8, 2), (4, 1)])
@pytest.mark.parametrize("block_size", [64])
def test_multidoc_gqa(hq, hkv, block_size):
    _sm100_or_skip()
    doc_lens = [40, 88, 133, 27]
    b, d = 1, 128
    s = sum(doc_lens)
    paddle.seed(4)
    q = paddle.randn([b, s, hq, d], dtype="bfloat16")
    k = paddle.randn([b, s, hkv, d], dtype="bfloat16")
    v = paddle.randn([b, s, hkv, d], dtype="bfloat16")
    nb = _num_blocks(s, block_size)
    sm_scale = 1.0 / math.sqrt(d)
    block_logit = paddle.full([b, hq, s, nb], _NEG_INF, dtype="float32")
    _flash_attn_fwd(
        q, k, v, softmax_scale=sm_scale, causal=True, return_lse=True,
        startend_row_indices=_causal_document_flashmask(doc_lens, batch=b),
        block_logit=block_logit, block_size=block_size,
        block_bos=_block_bos_from_doc_lens(doc_lens, batch=b), pack_gqa=False,
    )
    # Reference: repeat kv heads to query heads, then document-relative bucket.
    kf = paddle.repeat_interleave(k, hq // hkv, axis=2)
    ref = ref_block_logit_multidoc(q, kf, doc_lens, block_size, nb, sm_scale, bf16_matmul=False)
    ref_bf16 = ref_block_logit_multidoc(q, kf, doc_lens, block_size, nb, sm_scale, bf16_matmul=True)
    _assert_block_logit(block_logit, ref, ref_bf16, tol=0.06 * math.sqrt(d) * sm_scale)


# ---------------------------------------------------------------------------
# (6) Mask/data batch: the flashmask + block_bos derived per document must be
#     consistent across a batch > 1 (all sequences share one document layout).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block_size", [64])
def test_multidoc_batched(block_size):
    _sm100_or_skip()
    doc_lens = [50, 100, 70]
    b, h, d = 3, 4, 128
    s = sum(doc_lens)
    paddle.seed(5)
    q = paddle.randn([b, s, h, d], dtype="bfloat16")
    k = paddle.randn([b, s, h, d], dtype="bfloat16")
    v = paddle.randn([b, s, h, d], dtype="bfloat16")

    block_logit, sm_scale = _run_multidoc(q, k, v, doc_lens, block_size, batch=b)
    nb = _num_blocks(s, block_size)
    ref = ref_block_logit_multidoc(q, k, doc_lens, block_size, nb, sm_scale, bf16_matmul=False)
    ref_bf16 = ref_block_logit_multidoc(q, k, doc_lens, block_size, nb, sm_scale, bf16_matmul=True)
    _assert_block_logit(block_logit, ref, ref_bf16, tol=0.06 * math.sqrt(d) * sm_scale)


# ---------------------------------------------------------------------------
# (7) Alignment cross-check: the block_bos this test threads in must be exactly
#     what the production derivation ``build_hysparse_valid_range`` produces from
#     the SAME flashmask (pure host math -- no GPU needed). Skips if PaddleFleet
#     is not importable in the test environment.
# ---------------------------------------------------------------------------
def test_block_bos_matches_build_hysparse_valid_range():
    try:
        from paddlefleet.transformer.multi_latent_attention import (
            build_hysparse_valid_range,
        )
    except Exception as exc:  # pragma: no cover - env-dependent import
        pytest.skip(f"paddlefleet not importable: {exc}")

    doc_lens = [40, 88, 133, 27]
    s = sum(doc_lens)
    mask = _causal_document_flashmask(doc_lens, batch=1)  # [1,1,S,1] excl doc-end
    vr = build_hysparse_valid_range(mask, s, 1)            # [1,S,2] int32
    bos_ref = vr[..., 0].astype("int32").numpy()[0]        # [S]
    bos_got = _block_bos_from_doc_lens(doc_lens, batch=1).numpy()[0]
    assert np.array_equal(bos_got, bos_ref), (
        f"block_bos {bos_got.tolist()} != build_hysparse_valid_range "
        f"{bos_ref.tolist()}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
