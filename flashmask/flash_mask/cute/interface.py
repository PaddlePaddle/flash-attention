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

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import math
from typing import Optional, Tuple, Callable, Union

import paddle

import cuda.bindings.driver as cuda
from dataclasses import dataclass
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_mask.cute import utils
from flash_mask.cute.cute_dsl_utils import make_fake_tensor
from flash_mask.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_mask.cute.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_mask.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_mask.cute.flash_bwd_sink import FlashAttentionBackwardDsink
from flash_mask.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_mask.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_mask.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_mask.cute.flash_bwd_sm100_bigd import (
    FlashAttentionBackwardSm100BigD,
    bigd_host_config,
)
from flash_mask.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_mask.cute.flash_fwd_combine import FlashAttentionForwardCombine
from flash_mask.cute.flashmask_utils import (
    FlashMaskInfoPaddle,
    prepare_block_maxmin,
    to_cute_flashmask_info,
    reduce_block_count,
    compute_flashmask_block_lists,
    build_flashmask_block_lists,
)

from flash_mask.cute.block_sparsity import (
    BlockSparseTensorsPaddle,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_tensors,
)

try:
    from ..utils import accum_zero_axis1_kv
except ImportError:
    accum_zero_axis1_kv = None


def _get_overlap_runtime():
    try:
        from flash_mask.overlap import overlap_runtime
    except ImportError as exc:
        raise RuntimeError(
            "FM-4 overlap support requires the 'ovl' build component"
        ) from exc
    return overlap_runtime


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.strides[-1] != 1 else x


paddle2cute_dtype_map = {
    paddle.float16: cutlass.Float16,
    paddle.bfloat16: cutlass.BFloat16,
    paddle.float32: cutlass.Float32,
}


def _get_fa_version():
    return paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]


def _is_valid_flash_dims(query, key, value, fa_version=2):
    q_headdim, k_headdim, v_headdim = query.shape[-1], key.shape[-1], value.shape[-1]
    if (
        (q_headdim <= 128 and k_headdim <= 128 and v_headdim <= 128)
        or (q_headdim == 192 and k_headdim == 192 and v_headdim == 128)
        or (q_headdim == 256 and k_headdim == 256 and v_headdim == 256)
    ):
        return True
    # SM100 (fa4) large head_dim, forward + backward.
    # has been run end to end.
    #   512/512 : forward + backward verified
    if fa_version == 4:
        if (
            (q_headdim == 512 and k_headdim == 512 and v_headdim == 512)
        ):
            return True
    return False


def _is_non_4vec_startend(startend_row_indices):
    return startend_row_indices is None or startend_row_indices.shape[-1] != 4


def _is_cutedsl_kernel_supported(query, key, value, startend_row_indices=None):
    fa_version = _get_fa_version()
    if not _is_valid_flash_dims(query, key, value, fa_version):
        return False
    if fa_version == 3:                       # SM90
        return True
    if fa_version == 4:                       # SM100
        return _is_non_4vec_startend(startend_row_indices)
    return False

def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)

@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warp groups (total threads = (num_wg + 1) * 128)
    dQ_single_wg: bool = False

def _tile_size_bwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=None, flashmask=False, deterministic=False):
    """Return BwdConfig for SM90.

    Configs based on C++ FA3 hopper/flash_bwd_launch_template.h,
    benchmarked on H100 SXM.
    """
    if head_dim <= 64:
        # C++ FA3: 128, 128, 64, ..., 2, 2, true, false, false, 2, 1, 2, 2
        return BwdConfig(
            m_block_size=128, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        # C++ FA3: 64, 128, 96, dQ_swapAB=False
        return BwdConfig(
            m_block_size=64, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        # flashmask stages per-n_block startend_row_indices in extra smem; the
        # non-causal m=80 tile has no headroom for it, so fall back to m=64.
        if flashmask:
            m_block_size = 64
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        if hdimv128:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=2, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
        else:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=1, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
    else:
        # hdim 256: mirror C++ FA3 run_mha_bwd_hdim256 (Arch>=90); dKV/dQ MMAs
        # use swapAB. The dQ-accumulation path drives the smem budget and must
        # match flash_bwd_sm90's dQacc_use_TMA (== deterministic for d256):
        #   - non-deterministic: dQ is atomicAdd'd straight to gmem, so the 64KB
        #     dQ smem staging buffer is dropped. That headroom buys a 2-stage Q
        #     pipeline and the larger dense N=80 tile (~20% fewer KV-blocks;
        #     flashmask keeps N=64 to match its block-list tiling).
        #   - deterministic: ordered accumulation forces the TMA-staging path,
        #     re-adding the 64KB buffer. 2-stage Q / N=80 would then overflow
        #     Hopper's ~227KB smem (CUDA_ERROR_INVALID_VALUE at launch), so use
        #     a 1-stage Q pipeline and N=64.
        if deterministic:
            n_block_size, num_stages_Q = 64, 1
        else:
            n_block_size, num_stages_Q = (64 if flashmask else 80), 2
        return BwdConfig(
            m_block_size=64, n_block_size=n_block_size,
            num_stages_Q=num_stages_Q, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=True,
            AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
        )


def _make_fake_bwd_tensors(dtype, has_gqa, deterministic=False, cluster_size=1):
    """FA4-style fake bwd tensors with all dims as cute.sym_int and stride
    divisibility hints (so 128-bit alignment is guaranteed at compile time).
    Non-varlen only (flashmask does not support varlen)."""
    sym = cute.sym_int
    div = 128 // dtype.width  # 8 for bf16/fp16
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    mQ = make_fake_tensor(dtype, (b, seqlen_q, h_q, d), divisibility=div)
    mO = make_fake_tensor(dtype, (b, seqlen_q, h_q, d_v), divisibility=div)
    mdO = make_fake_tensor(dtype, (b, seqlen_q, h_q, d_v), divisibility=div)
    mK = make_fake_tensor(dtype, (b, seqlen_k, h_kv, d), divisibility=div)
    mV = make_fake_tensor(dtype, (b, seqlen_k, h_kv, d_v), divisibility=div)
    mdQ = make_fake_tensor(dtype, (b, seqlen_q, h_q, d), divisibility=div)
    mdK = make_fake_tensor(dtype, (b, seqlen_k, h_kv, d), divisibility=div)
    mdV = make_fake_tensor(dtype, (b, seqlen_k, h_kv, d_v), divisibility=div)
    mLSE = make_fake_tensor(cutlass.Float32, (b, h_q, seqlen_q), divisibility=1)
    mLSElog2 = make_fake_tensor(cutlass.Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    mPdPsum = make_fake_tensor(cutlass.Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    mdQaccum = make_fake_tensor(cutlass.Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        mdKaccum = make_fake_tensor(cutlass.Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
        mdVaccum = make_fake_tensor(cutlass.Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
    if not deterministic:
        mdQ_semaphore = None
        mdK_semaphore, mdV_semaphore = None, None
    else:
        num_m_blocks = sym()
        mdQ_semaphore = make_fake_tensor(
            cutlass.Int32, (b, h_q, num_m_blocks, cluster_size), divisibility=1
        )
        if not has_gqa:
            mdK_semaphore, mdV_semaphore = None, None
        else:
            num_n_blocks = sym()
            mdK_semaphore = make_fake_tensor(
                cutlass.Int32, (b, h_kv, num_n_blocks, 2), divisibility=1
            )
            mdV_semaphore = make_fake_tensor(
                cutlass.Int32, (b, h_kv, num_n_blocks, 2), divisibility=1
            )
    return (
        mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum,
        mdQaccum, mdKaccum, mdVaccum, mdQ_semaphore, mdK_semaphore, mdV_semaphore,
    )


def _flash_attn_fwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    seqused_q: Optional[paddle.Tensor] = None,
    seqused_k: Optional[paddle.Tensor] = None,
    page_table: Optional[paddle.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[paddle.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsPaddle] = None,
    return_lse: bool = False,
    out: Optional[paddle.Tensor] = None,
    lse: Optional[paddle.Tensor] = None,
    aux_tensors: Optional[list[paddle.Tensor]] = None,
    startend_row_indices: Optional[paddle.Tensor] = None,
    block_logit: Optional[paddle.Tensor] = None,
    block_size: int = 64,
    block_bos: Optional[paddle.Tensor] = None,
    group=None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
        block_logit: Optional pre-allocated fp32 tensor [batch, num_heads, seqlen_q, num_blocks] that,
            when provided, receives the fused per-(query, key-block) max of the post-score_mod, post-mask,
            SCALED logit `softmax_scale * q*k^T` (INCLUDING any score_mod bias) -- i.e. the exact value fed
            into softmax. Storing the scaled logit puts every head on one head-independent scale, so a
            downstream `block_logit - LSE` yields log(max attention weight in the block), which is comparable
            across heads for cross-head Top-K block selection. Computed in the softmax epilogue and written in
            place; kept out of the returned tuple. num_blocks must be >= ceil(seqlen_k / block_size). The fused
            reduction respects the same causal / flashmask masking applied to the attention. Only supported on
            SM 10.x (non split-KV).
            IMPORTANT: the kernel only writes key-blocks that the attention loop actually visits. Key-blocks
            that are entirely skipped by masking (e.g. fully-future blocks under causal / flashmask, or blocks
            past seqlen_k when num_blocks*block_size > seqlen_k) are NEVER written. The caller MUST
            pre-initialize block_logit to -inf so those entries read as -inf (masked) rather than stale
            garbage; do not pass an uninitialized / reused buffer.
        block_size: Key-block width (in tokens) used to bucket the fused block-logit reduction. Must divide
            the kernel's n_block_size. Ignored when block_logit is None.
    """

    assert cu_seqlens_q is None, "cu_seqlens_q must be None (varlen is not supported in flashmask)"
    assert cu_seqlens_k is None, "cu_seqlens_k must be None (varlen is not supported in flashmask)"

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]

    compute_capability = (
        paddle.device.cuda.get_device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    # --- SM100 large head_dim: the folded-accumulator single pass ---
    # head_dim > 256 keeps head_dim_v whole and runs the folded (m_block_size == 64)
    # accumulator config below. There used to be a fallback that split head_dim_v into
    # several <= 256-wide passes; it is gone because every extra pass costs a FULL
    # redundant Q*K^T. For d=dv=512 the split is 1.5x the useful FLOPs (measured 49.5ms /
    # 711 effective TFLOPs/s, vs 37.8ms / 931 effective for the single pass on
    # b16 s8192 h16 d512), and at dv=128 it is 2.5x (86.9ms / 405 effective).
    #
    # The features the folded accumulator cannot express are therefore not supported at
    # head_dim_v > 256 -- assert instead of silently taking a 1.3-2.3x slower route. See
    # also the folded_acc asserts in FlashAttentionForwardSm100.__init__.
    if compute_capability == 10 and head_dim > 256 and v.shape[-1] > 256:
        assert v.shape[-1] <= 512, (
            f"head_dim={head_dim}/{v.shape[-1]}: the folded accumulator supports "
            "head_dim_v <= 512 (O takes head_dim_v / 2 TMEM columns)"
        )
        assert num_splits == 1, (
            "head_dim_v > 256 does not support split-KV: the folded accumulator shares "
            "each query row between two threads, which the split-KV row map cannot express"
        )
        assert block_logit is None, (
            "head_dim_v > 256 does not support block_logit (folded accumulator)"
        )
        assert page_table is None, (
            "head_dim_v > 256 does not support paged KV (folded accumulator)"
        )
        assert block_sparse_tensors is None, (
            "head_dim_v > 256 does not support block sparsity (folded accumulator)"
        )

    # num_stages=2 exceeds Hopper's ~227KB per-block shared-memory limit
    # (e.g. hdim=256 needs ~320KB -> cuLaunchKernel CUDA_ERROR_INVALID_VALUE), so
    # shrink the N tile to 64. Set BEFORE the flashmask nblock max/min + block-list
    # generation below so prepare_block_maxmin / reduce_block_count /
    # compute_flashmask_block_lists and the kernel all agree on n_block_size
    # (otherwise the per-n_block arrays disagree in length -> reshape error).
    if compute_capability == 9 and head_dim > 128:
        n_block_size = 64
        # d256/dv256 dense fwd is tensor-core bound; a larger KV tile (80 vs 64,
        # matching mainline FA3) cuts KV-loop iterations/overhead and feeds the
        # MMA better. smem at N=80 (sQ 64KB + sK 80KB + sV 80KB ~= 224KB, no P
        # since mma_pv_is_rs) fits Hopper's ~227KB. Only widen the dense
        # (non-flashmask) d256 case: flashmask keeps 64 so the block max/min scan
        # + block-list tiling (kBlockN) agree (and leave headroom for s_rowidx);
        # head_dim=192 keeps 64 (dv=128 tile is a different smem shape).
        if head_dim == 256 and v.shape[-1] == 256 and startend_row_indices is None:
            n_block_size = 80

    # SM100 per-pass config for head_dim=576 (head_dim_v <= 256 after the split above).
    # m_block stays 128: the softmax TMEM load atom needs a QK accumulator with M=128.
    # With m=128 the KV tile must shrink to 32 to fit SMEM, since Q alone takes
    # m*576*2B = 144KB (sO overlaps sQ) and K/V share one buffer of kv_stage*n*576*2B
    # (72KB at n=32, kv_stage=2) -> ~216KB of the ~227KB budget.
    # n_block must stay a multiple of 32 (tilePlikeFP32 = n//32*16).
    is_bigd_fwd = compute_capability == 10 and head_dim > 256
    if is_bigd_fwd:
        m_block_size = 128
        n_block_size = 32

    # SM100 big-d (d>256, dv<=256 per pass): the 1-CTA config is forced to n=32 because
    # sQ alone is m*d*2B = 128KB at m=128, which caps FLOP-per-SMEM-byte at ~35 (vs the
    # ~85 needed to saturate the tensor core) and is why hd512 fwd collapses to ~103ms.
    # A 2-CTA (tcgen05 CTA-pair) UMMA splits the MMA's N across the pair, so sK/sV per CTA
    # is halved and n can double at identical SMEM: measured per-CTA footprint for
    # (m=128, n=64, d=512, dv=256, q_stage=1, kv_stage=2) is sQ 128K + sK 64K + sV 32K,
    # i.e. exactly the 1-CTA n=32 footprint, while the pair's MMA tile grows from
    # 128x32 to 256x64 (4x the FLOPs per KV tile load).
    #
    # m_block_size stays 128 for every per-pass (dv <= 256) config. The MMA's M is split
    # across the pair, so a CTA owns m_block_size accumulator rows; every TMEM copy in
    # softmax/correction/epilogue is a 128-thread (4-warp) tiled copy whose thread->row map
    # is 1:1 only when the CTA owns 128 rows. At m_block_size=64 CuTe spreads those 64 rows
    # over all 128 TMEM lanes by splitting N (thread t and t+64 then share a row, each
    # holding half of it); that FOLDED layout is what the single-pass config below uses, and
    # it needs the row_max exchange / P-in-SMEM / physical-lane epilogue handling in
    # flash_fwd_sm100.py. Do not set m_block_size=64 for a config that does not want it.
    use_2cta_instrs = False
    if is_bigd_fwd:
        # Enabled for both the dense and the flashmask path. An earlier "flashmask hangs
        # with the CTA pair" report turned out to be the test harness running out of
        # memory while building its reference attn_bias (cuda-gdb showed no resident
        # kernel; the stack was a paddle `set_value_` strided copy), not a kernel
        # deadlock, so there is no reason to keep flashmask on the slower 1-CTA config.
        use_2cta_instrs = True
    if use_2cta_instrs:
        # m_block_size 128 keeps one accumulator row per thread. 64 folds the accumulator (64
        # rows spread over all 128 TMEM lanes, N split between the lane halves), which HALVES
        # its column cost: O(64x512) f32 takes 256 of the 512 TMEM columns, so dv=512 fits in
        # ONE pass (O 256 + S 32 + P 16 = 304). That is the only way to avoid a 1.5x QK
        # redundancy; the price is half the arithmetic intensity per KV tile (the pair covers
        # 128 query rows instead of 256).
        if v.shape[-1] > 256:
            m_block_size = 64
        else:
            m_block_size = 128
        # n=64 doubles the 1-CTA KV tile at identical per-CTA SMEM (the pair splits the MMA's
        # N, so sK/sV per CTA is halved). The folded config additionally requires
        # n_block_size * sizeof(dtype) == 128B, i.e. one P row is exactly one swizzle period
        # of the A-operand SMEM layout (asserted in softmax_loop).
        n_block_size = 64
    fwd_cta_group_size = 2 if use_2cta_instrs else 1

    # Each SM100 CTA processes q_stage * m_block_size query rows; Split-D
    # (d>192, d==dv) uses q_stage=1 to fit the TMEM budget. Must match
    # FlashAttentionForwardSm100.q_stage (= 1 if is_split_d else 2). Computed
    # once here so the flashmask valid_block_count and the block-sparse M-block
    # normalization below share the identical M granularity.
    q_stage = 1 if ((head_dim > 192 and head_dim == v.shape[-1]) or is_bigd_fwd) else 2
    # Query rows per work tile. This is the M granularity that the flashmask
    # valid_block_count / block lists are built at; with a 2-CTA UMMA one work tile is
    # handled by a CTA *pair*, so it covers cta_group_size * q_stage * m_block_size rows.
    fwd_m_tile_rows = q_stage * m_block_size * fwd_cta_group_size
    # FM-4 overlap: when a CP `group` is given, K/V come in LOCAL (B, S_local, H, D)
    # and the gathered KV lives in the NVSHMEM SRBuffer. Bootstrap+init the comm
    # singleton once, run the sparse all-gather on the internal comm_stream, then
    # below swap the K/V cute tensors for SRBuffer-backed views with
    # seqlen_k = S_local*nranks. The SM100 load warp gates each K/V tile on the
    # non-splitted AG kernel's write_ptr row frontier.

    enable_overlap = group is not None and group.world_size > 1
    if enable_overlap and compute_capability != 10:
        raise NotImplementedError("FM-4 overlap fwd is only supported on SM100")
    overlap_view_args = None
    overlap_bhsd_layout = None
    if enable_overlap:
        overlap_runtime = _get_overlap_runtime()
        assert startend_row_indices is not None, (
            "overlap mode requires startend_row_indices (the post-AG mask)"
        )
        assert page_table is None, "overlap mode does not support paged KV"
        assert cu_seqlens_k is None, "overlap mode does not support varlen K"
        assert k.dtype == paddle.bfloat16, "overlap SRBuffer is bf16"
        # causal would make startend_row_indices col1 hold lt_end (not ut_end), but
        # the comm-side compute_chunk_mask requires a non-null ut_end (overlap_comm.cu
        # :453); FM-3 overlap forbids causal for the same reason (overlap_flashmask.py
        # :335). Guarding here keeps the col mapping in _sparse_chunk_mask_cols exact.
        assert not causal, "overlap mode does not support causal yet"
        overlap_runtime.ensure_initialized(
            k, v, group, mask_head=startend_row_indices.shape[1]
        )
        overlap_bhsd_layout = overlap_runtime.use_bhsd_layout()
        overlap_stream = overlap_runtime.current_stream_handle()
        # Launch AG and retain the gathered SRBuffer view; its FULL S_total shape
        # drives host-side validation and the runtime-dimension kernel arguments.
        overlap_ag_args = overlap_runtime.start_forward_ag(
            k, v, startend_row_indices, overlap_stream
        )
        overlap_view_args = overlap_ag_args.view

    cute_flashmask_info = None
    if startend_row_indices is not None:
        fm_batch_size = startend_row_indices.shape[0]
        fm_heads = startend_row_indices.shape[1]
        num_m_blocks = (seqlen_q + fwd_m_tile_rows - 1) // fwd_m_tile_rows
        flashmask_info = FlashMaskInfoPaddle(
            is_causal=causal,
            startend_row_indices=startend_row_indices,
        )
        # valid_block_count (produced by reduce_block_count) feeds only the SM100
        # path and the bwd 2CTA density heuristic. The SM90 fwd kernel iterates the
        # block-sparse lists built below (build_flashmask_block_lists) and never
        # reads valid_block_count, so skip both the extra [b,h,num_m_blocks] alloc
        # and the reduce_block_count kernel launch there. Per-call fixed overhead
        # dominates the high-sparsity / short-seq configs, so this is pure win.
        if compute_capability != 9:
            flashmask_info.valid_block_count = paddle.empty([fm_batch_size, fm_heads, num_m_blocks], dtype=paddle.int32)
        prepare_block_maxmin(flashmask_info, kBlockN=n_block_size)
        cute_flashmask_info = to_cute_flashmask_info(flashmask_info)
        if compute_capability != 9:
            reduce_block_count(cute_flashmask_info, causal, fwd_m_tile_rows, n_block_size, seqlen_q)

    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == paddle.int32, "page_table must be int32"
        assert page_table.strides[-1] == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == [batch_size, max_num_pages_per_seq]
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    elif enable_overlap:
        # K/V are still the LOCAL paddle tensors here (used only for update_kv +
        # dtype); the kernel reads the gathered SRBuffer, so seqlen_k is the full
        # gathered length S_total = S_local * nranks from the SRBuffer view.
        num_pages, page_size = None, None
        seqlen_k = overlap_view_args.shape[1]
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if enable_overlap:
        # Skip the [batch, seqlen_k, ...] shape assert: k/v are LOCAL (S_local),
        # while seqlen_k is the gathered S_total. The SRBuffer view (built below
        # from overlap_view_args) carries the gathered shape into the kernel.
        assert num_head_kv == overlap_view_args.shape[2]
        assert head_dim == overlap_view_args.shape[3]
    elif cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == [batch_size, seqlen_k, num_head_kv, head_dim], (
                f"expect k with shape {[batch_size, seqlen_k, num_head_kv, head_dim]}, received {k.shape=}"
            )
            assert v.shape == [batch_size, seqlen_k, num_head_kv, head_dim_v]
        else:
            assert k.shape == [num_pages, page_size, num_head_kv, head_dim]
            assert v.shape == [num_pages, page_size, num_head_kv, head_dim_v]
    else:
        assert k.shape == [seqlen_k, num_head_kv, head_dim]
        assert v.shape == [seqlen_k, num_head_kv, head_dim_v]
        assert cu_seqlens_k.shape == [
            batch_size + 1,
        ], "cu_seqlens_k must have shape (batch_size + 1,)"

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == [
            batch_size + 1,
        ], "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == [
        batch_size,
    ], "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == [
        batch_size,
    ], "seqused_k must have shape (batch_size,)"
    assert q.dtype in [paddle.float16, paddle.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == paddle.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.strides[0] == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == [
            num_head,
        ]
        assert learnable_sink.dtype == paddle.bfloat16, "learnable_sink must be bfloat16"

    assert all(
        t is None or t.place.is_gpu_place()
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    # head_dim <= 256 for the symmetric / Split-D paths. Larger head_dim is the SM100
    # big-d path (Split-D with q_stage=1); head_dim_v > 256 stays whole on the folded
    # (m_block_size == 64) accumulator, which supports up to head_dim_v = 512.
    assert (
        head_dim <= 256
        or (compute_capability == 10 and head_dim_v <= 512)
    ), (
        "head_dim must be <= 256, or head_dim>256 with head_dim_v<=512 on SM100 "
        f"(got {head_dim}/{head_dim_v})"
    )
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_paddle_dtype = q.dtype
    place = q.place
    q_batch_seqlen_shape = (
        [batch_size, seqlen_q]
        if cu_seqlens_q is None
        else [
            total_q,
        ]
    )
    lse_shape = [batch_size, num_head, seqlen_q] if cu_seqlens_q is None else [num_head, total_q]
    requires_grad = not (q.stop_gradient and k.stop_gradient and v.stop_gradient)

    if out is None:
        # then stored, so fully-masked rows/blocks (incl. generate_empty_mask) come
        # out as 0. So on SM90 use an uninitialized buffer to skip the ~1GB O
        # zero-fill that dominates short-seq / high-sparsity calls (~5% fwd speedup).
        # On SM100 the flashmask fwd skips the O store for a fully-masked query block
        # (valid_block_count == 0) and has no zero-writer for it, so that O tile would
        # be left uninitialized -- keep paddle.zeros on SM100.
        out = (paddle.empty if compute_capability == 9 else paddle.zeros)(
            shape=[*q_batch_seqlen_shape, num_head, head_dim_v], dtype=out_paddle_dtype
        )
    else:
        expected_out_shape = [*q_batch_seqlen_shape, num_head, head_dim_v]
        assert out.shape == expected_out_shape, (
            f"out tensor shape {out.shape} does not match expected shape {expected_out_shape}"
        )
        assert out.dtype == out_paddle_dtype, (
            f"out tensor dtype {out.dtype} does not match expected dtype {out_paddle_dtype}"
        )
        assert out.place.is_gpu_place(), (
            f"out tensor device {out.place} does not match input device"
        )

    if lse is None:
        lse = (
            paddle.full(shape=lse_shape, fill_value=float('-inf'), dtype=paddle.float32)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        assert lse.shape == lse_shape, (
            f"lse tensor shape {lse.shape} does not match expected shape {lse_shape}"
        )
        assert lse.dtype == paddle.float32, (
            f"lse tensor dtype {lse.dtype} does not match expected dtype paddle.float32"
        )
        assert lse.place.is_gpu_place(), "lse tensor must be on CUDA device"

    dtype = paddle2cute_dtype_map[q.dtype]
    (
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        learnable_sink_tensor,
    ) = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if t is not None
        else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    page_table_tensor = (
        from_dlpack(page_table.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
        if page_table is not None
        else None
    )

    # flashmask (SM90): route through the block-sparse list-driven path so the
    # kernel iterates only surviving KV blocks (arbitrary/mid-range skip) while
    # keeping intra_wg_overlap. Build per-(b,h,m_block) surviving-block lists from
    # the flashmask nblock max/min (kBlockM=m_block_size, kBlockN=n_block_size).
    # split_full=True: partially-masked blocks go to the mask list (they get the
    # per-element flashmask apply + rowidx staging), fully-visible blocks go to the
    # full list (no rowidx pipeline / no element mask). This keeps correctness for
    # partial masks (sliding-window etc.) while avoiding the per-block rowidx
    # mbarrier sync on the fully-visible blocks (perf).
    if (
        compute_capability == 9
        and cute_flashmask_info is not None
        and block_sparse_tensors is None
        and seqlen_q is not None
    ):
        fm_full_cnt, fm_full_idx, fm_mask_cnt, fm_mask_idx = build_flashmask_block_lists(
            flashmask_info,
            causal,
            m_block_size,
            n_block_size,
            seqlen_q,
            seqlen_k,
            num_head,
            split_full=True,
        )
        # mask list = partial blocks (element-masked), full list = fully-visible
        # blocks (no element mask). Both are consumed; the producer stages rowidx
        # only for the (contiguous, processed-first) mask-list prefix.
        block_sparse_tensors = BlockSparseTensorsPaddle(
            mask_block_cnt=fm_mask_cnt,
            mask_block_idx=fm_mask_idx,
            full_block_cnt=fm_full_cnt,
            full_block_idx=fm_full_idx,
        )

    sparse_tensors = None
    if block_sparse_tensors is not None:
        if seqlen_q is None:
            raise ValueError(
                "Block sparsity requires fixed-length sequences (seqlen_q must be known)."
            )
        m_block_size_block = m_block_size
        if compute_capability == 10:
            # One SM100 CTA handles q_stage * tile_m rows; keep this in lockstep
            # with the flashmask valid_block_count granularity above.
            m_block_size_block = fwd_m_tile_rows
        expected_m_blocks = (seqlen_q + m_block_size_block - 1) // m_block_size_block
        expected_n_blocks = (seqlen_k + n_block_size - 1) // n_block_size
        block_sparse_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=(batch_size, num_head, expected_m_blocks),
            expected_index_shape=(batch_size, num_head, expected_m_blocks, expected_n_blocks),
        )
        sparse_tensors = to_cute_block_sparse_tensors(block_sparse_tensors)

    use_block_sparsity = sparse_tensors is not None

    if mask_mod is None:
        if causal:
            window_size_right = 0
        local = window_size_left is not None or window_size_right is not None
        if window_size_left is not None or window_size_right is not None:
            if window_size_left is None and window_size_right == 0:
                causal, local = True, False
            else:
                causal, local = False, True
    else:
        causal, local = False, False

    current_stream = cuda.CUstream(paddle.device.current_stream().stream_base.cuda_stream)

    # NOTE: do NOT bump n_block_size to 192 for the dense d=128 (non-causal,
    # non-flashmask) case. tile_n=192 with num_stages=2 needs ~225 KiB smem, which
    # hits Hopper's ~228 KiB ceiling -> 1 block/SM and near-zero L1, making Full
    # ~6% slower than tile_n=128 (FA4 uses 128 and reaches ~647 vs ~613 TFLOP/s
    # here). Keep the default 128 to match FA4's dense config.
    if compute_capability == 10:
        # TODO: fix the varlen case
        if (
            pack_gqa
            and (128 % qhead_per_kvhead != 0)
            or (cu_seqlens_q is not None or seqused_q is not None)
        ):
            pack_gqa = False
        # TODO: fix GQA + SplitKV + non-varlen
        if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
            pack_gqa = False
        # The big-d fwd configs cannot pack q heads into the M dim. PackGQA derives each
        # row's q_head (and its O destination) straight from tidx, which neither Split-D's
        # q_stage=1 row map nor the folded (m_block_size=64) accumulator provides -- in the
        # folded layout a row is shared by threads t and t + m_block_size. Gate it here
        # instead of letting the kernel assert. GQA/MQA stays correct via qhead_per_kvhead,
        # just without the packing optimization (which only pays off when seqlen_q per head
        # is too small to fill the M tile, i.e. decode).
        if is_bigd_fwd:
            pack_gqa = False
        # Split-D (q_stage=1) for d=dv=256, and for the head_dim=576 per-pass config:
        # both need q_stage=1 so O (head_dim_v cols) fits alongside S/P in the 512-col TMEM.
        is_split_d = (head_dim > 192 and head_dim == head_dim_v) or is_bigd_fwd

    if num_splits < 1:
        max_seqlen_k = (
            seqlen_k
            if cu_seqlens_k is None
            else (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
        )
        max_seqlen_q = (
            seqlen_q
            if cu_seqlens_q is None
            else (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
        )
        seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
        seqlen_k_loaded = (
            max_seqlen_k
            if not local
            else max(0, min(max_seqlen_k, window_size_right + window_size_left + 1 + m_block_size))
        )
        num_n_blocks = (seqlen_k_loaded + n_block_size - 1) // n_block_size
        num_m_blocks = (seqlen_q_packgqa + m_block_size - 1) // m_block_size
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_splits = num_splits_heuristic(
            total_mblocks,
            paddle.device.cuda.get_device_properties(place.gpu_device_id()).multi_processor_count,
            num_n_blocks,
            128,
        )

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = paddle.empty(
            shape=[num_splits, *q_batch_seqlen_shape, num_head, head_dim_v], dtype=paddle.float32
        )
        lse_partial = paddle.empty(shape=[num_splits, *lse_shape], dtype=paddle.float32)

    q_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, out if not is_split_kv else out_partial)
    ]
    if enable_overlap:
        # K/V live in the NVSHMEM SRBuffer (gathered device memory), so there is no
        # dlpack capsule to wrap; the kernel builds the views from their addr instead.
        k_tensor = None
        v_tensor = None
    else:
        k_tensor, v_tensor = [
            from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (k, v)
        ]
    if is_split_kv:
        lse_tensor = from_dlpack(lse_partial.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse_partial.ndim - 1
        )
    elif lse is not None:
        lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse.ndim - 1
        )
    else:
        lse_tensor = None

    if block_logit is not None:
        assert compute_capability == 10, (
            "block_logit (fused block-score) is only supported on SM 10.x"
        )
        assert not is_split_kv, "block_logit is not supported with split-KV"
        assert not pack_gqa, (
            "block_logit requires pack_gqa=False. block_logit is indexed by the "
            "query head (head_idx) and the query row (m_block*m_block_size+tidx); "
            "under pack_gqa=True the query heads are packed into the M/row dim and "
            "head_idx is the KV head, so the block-score write would target wrong "
            "locations. NOTE: pack_gqa defaults to (qhead_per_kvhead > 1), so GQA "
            "callers MUST pass pack_gqa=False explicitly when requesting block_logit."
        )
        assert block_logit.dtype == paddle.float32, (
            f"block_logit must be float32; got {block_logit.dtype}"
        )
        assert block_logit.place.is_gpu_place(), "block_logit must be on CUDA"
        assert block_logit.ndim == 4, (
            f"block_logit must be [batch, num_heads, seqlen_q, num_blocks]; got ndim={block_logit.ndim}"
        )
        block_seqlen_k = seqlen_k if enable_overlap else k.shape[1]
        _nb_min = (block_seqlen_k + block_size - 1) // block_size
        assert block_logit.shape[-1] >= _nb_min, (
            f"block_logit num_blocks={block_logit.shape[-1]} < ceil(seqlen_k/block_size)={_nb_min}"
        )
        block_logit_tensor = from_dlpack(
            block_logit.detach(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=block_logit.ndim - 1)
    else:
        block_logit_tensor = None

    # Optional per-query document start (bos) for DOCUMENT-relative block
    # bucketing (pack-equivalence). Without it the kernel buckets by absolute
    # packed-sequence block, which is only correct for single-document (bos==0)
    # inputs. Shape [B, S] int32, aligned with block_logit's query dim.
    if block_bos is not None:
        assert block_logit is not None, (
            "block_bos requires block_logit (it drives its relative bucketing)"
        )
        assert block_bos.dtype == paddle.int32, (
            f"block_bos must be int32; got {block_bos.dtype}"
        )
        assert block_bos.place.is_gpu_place(), "block_bos must be on CUDA"
        assert block_bos.ndim == 2, (
            f"block_bos must be [B, S]; got ndim={block_bos.ndim}"
        )
        assert list(block_bos.shape) == [batch_size, seqlen_q], (
            f"block_bos must be [B={batch_size}, S={seqlen_q}]; got {block_bos.shape}"
        )
        block_bos_tensor = from_dlpack(
            block_bos.detach(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=block_bos.ndim - 1)
    else:
        block_bos_tensor = None

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )
    if score_mod is not None:
        if is_varlen:
            raise NotImplementedError(
                "score_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    if mask_mod is not None:
        if is_varlen:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "Block sparsity is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )

    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [from_dlpack(buf).mark_layout_dynamic() for buf in aux_tensors]

    # Build SRBuffer views inside the MLIR context. Dimensions stay runtime Int32;
    # making this layout static changes the TMA descriptor and reads wrong bytes.
    if enable_overlap:
        overlap_view = overlap_ag_args.view
        overlap_k_addr = cutlass.Int64(overlap_view.k_addr)
        overlap_v_addr = cutlass.Int64(overlap_view.v_addr)
        overlap_write_ptr_addr = cutlass.Int64(overlap_ag_args.write_ptr.data_ptr())
        _ob, _os, _oh, _od = overlap_view.shape
        overlap_b = cutlass.Int32(_ob)
        overlap_s = cutlass.Int32(_os)
        overlap_h = cutlass.Int32(_oh)
        overlap_d = cutlass.Int32(_od)
        overlap_kv_chunk_size = overlap_ag_args.kv_chunk_size
    else:
        overlap_k_addr = None
        overlap_v_addr = None
        overlap_write_ptr_addr = None
        overlap_b = None
        overlap_s = None
        overlap_h = None
        overlap_d = None
        overlap_kv_chunk_size = None
        overlap_bhsd_layout = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        len(aux_tensors) if aux_tensors is not None else 0,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        is_split_kv,
        pack_gqa,
        compute_capability,
        page_size not in [None, 128],  # paged KV non-TMA
        # flashmask
        startend_row_indices.shape[3] if startend_row_indices is not None else None,
        is_split_d if compute_capability == 10 else False,
        use_2cta_instrs,
        block_logit is None,
        block_size,
        block_bos is None,
    ) + (
        # SRBuffer K/V require a distinct artifact only when overlap is active.
        (overlap_bhsd_layout, overlap_kv_chunk_size) if enable_overlap else ()
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        if compute_capability == 9:
            assert page_table is None, "paged KV not supported on SM 9.0"
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            # fa_fwd = FlashAttentionForwardSm80(
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=m_block_size,
                tile_n=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif compute_capability == 10:
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                is_persistent=not causal
                and not local
                and cu_seqlens_q is None
                and seqused_q is None
                and not is_split_kv,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                paged_kv_non_tma=page_size not in [None, 128],
                is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                is_split_d=is_split_d,
                has_block_logit=block_logit is not None,
                block_size=block_size,
                has_block_bos=block_bos is not None,
                use_2cta_instrs=use_2cta_instrs,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        # TODO: check @can_implement
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            mCuSeqlensQ=cu_seqlens_q_tensor,
            mCuSeqlensK=cu_seqlens_k_tensor,
            mSeqUsedQ=seqused_q_tensor,
            mSeqUsedK=seqused_k_tensor,
            mPageTable=page_table_tensor,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink_tensor,
            blocksparse_tensors=sparse_tensors,
            aux_tensors=cute_aux_tensors,
            flashmask_info=cute_flashmask_info,
            **(
                {
                    "mBlockLogit": block_logit_tensor,
                    "mBlockBos": block_bos_tensor,
                }
                if compute_capability == 10
                else {}
            ),
            **(
                {
                    "overlap_k_addr": overlap_k_addr,
                    "overlap_v_addr": overlap_v_addr,
                    "overlap_write_ptr_addr": overlap_write_ptr_addr,
                    "overlap_b": overlap_b,
                    "overlap_s": overlap_s,
                    "overlap_h": overlap_h,
                    "overlap_d": overlap_d,
                    "overlap_kv_chunk_size": overlap_kv_chunk_size,
                    "overlap_bhsd_layout": overlap_bhsd_layout,
                }
                if enable_overlap
                else {}
            ),
            stream=current_stream,
        )
    # Runtime address and shape scalars are re-supplied below; compile-time
    # overlap_kv_chunk_size is captured by the compiled callable.

    _flash_attn_fwd.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        mCuSeqlensQ=cu_seqlens_q_tensor,
        mCuSeqlensK=cu_seqlens_k_tensor,
        mSeqUsedQ=seqused_q_tensor,
        mSeqUsedK=seqused_k_tensor,
        mPageTable=page_table_tensor,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        learnable_sink=learnable_sink_tensor,
        blocksparse_tensors=sparse_tensors,
        aux_tensors=cute_aux_tensors,
        flashmask_info=cute_flashmask_info,
        **(
            {
                "mBlockLogit": block_logit_tensor,
                "mBlockBos": block_bos_tensor,
            }
            if compute_capability == 10
            else {}
        ),
        **(
            {
                "overlap_k_addr": overlap_k_addr,
                "overlap_v_addr": overlap_v_addr,
                "overlap_write_ptr_addr": overlap_write_ptr_addr,
                "overlap_b": overlap_b,
                "overlap_s": overlap_s,
                "overlap_h": overlap_h,
                "overlap_d": overlap_d,
            }
            if enable_overlap
            else {}
        ),
        stream=current_stream,
    )
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def _flash_attn_bwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    out: paddle.Tensor,
    dout: paddle.Tensor,
    lse: paddle.Tensor,
    flashmask_info: Optional[Union[FlashMaskInfoPaddle, paddle.Tensor]] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    seqused_q: Optional[paddle.Tensor] = None,
    seqused_k: Optional[paddle.Tensor] = None,
    learnable_sink: Optional[paddle.Tensor] = None,
    deterministic: bool = False,
    kv_postprocess_start: Optional[int] = None,
    kv_postprocess_end: Optional[int] = None,
    group=None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, Optional[paddle.Tensor]]:
    compute_capability = paddle.device.cuda.get_device_capability()[0]
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"
    assert cu_seqlens_q is None, "cu_seqlens_q must be None (varlen is not supported in flashmask)"
    assert cu_seqlens_k is None, "cu_seqlens_k must be None (varlen is not supported in flashmask)"
    assert seqused_q is None, "seqused_q must be None (varlen is not supported in flashmask)"
    assert seqused_k is None, "seqused_k must be None (varlen is not supported in flashmask)"

    num_head, head_dim = q.shape[-2:]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]

    # Large head dims (MLA-shaped 576/512 and friends) go through a separate SM100
    # backward kernel: dV (512 cols) + dK/dQ (576 cols each) + S/dP do not fit the
    # 512-col TMEM budget, so that kernel chunks the head_dim axes and drains every
    # output through the fp32 gmem accumulators. See flash_bwd_sm100_bigd.py.
    is_bigd_bwd = compute_capability == 10 and (head_dim > 256 or head_dim_v > 256)
    assert is_bigd_bwd or (head_dim <= 256 and head_dim_v <= 256), (
        f"backward does not support head_dim={head_dim}, head_dim_v={head_dim_v} "
        f"on sm_{compute_capability}0"
    )

    m_block_size = 128
    n_block_size = 128

    bigd_cfg = None
    if is_bigd_bwd:
        assert not deterministic, "deterministic reduction is not supported by big-headdim bwd"
        assert group is None or group.world_size <= 1, (
            "overlap is not supported by big-headdim bwd"
        )
        # The kernel solves its own tile config; the accumulator shapes and the
        # postprocess grid below are built from m_block_size / n_block_size, so they
        # have to agree with it.
        bigd_cfg = bigd_host_config(head_dim, head_dim_v)
        m_block_size = bigd_cfg["tile_m"]
        n_block_size = bigd_cfg["tile_n"]

    # SM100 d=256/dv=256 runs the 2-CTA backward, whose accumulators are folded
    # (64 rows per CTA). Folding is what halves their TMEM column cost and lets dK
    # and dV both stay resident -- no dK reduce, no dKV postprocess -- and it
    # requires a 64-row KV tile. Set here, BEFORE prepare_block_maxmin / the
    # block-list generation below, for the same reason as the SM90 case.
    if compute_capability == 10 and head_dim == 256 and head_dim_v == 256:
        n_block_size = 64

    # SM90: finalize n_block_size (the head_dim-dependent bwd N tile from
    # _tile_size_bwd_sm90, e.g. 64 for head_dim=256) BEFORE prepare_block_maxmin /
    # the block-list generation below, so they all agree with the actual kernel's N
    # tile (same class of bug as the fwd n_block_size fix: otherwise the flashmask
    # per-n_block max/min arrays are sized/scanned with kBlockN=128 while the kernel
    # and any block-list consumer expect the real n_block_size -> wrong blocks used
    # -> wrong dQ/dK/dV, or a shape mismatch if list-driven).
    if compute_capability == 9:
        n_block_size = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            False,
            sparse_block_size_q=None,
            flashmask=flashmask_info is not None,
            deterministic=deterministic,
        ).n_block_size

    cute_flashmask_info = None
    num_flashmask_tensors = 0

    if flashmask_info is not None and isinstance(flashmask_info, paddle.Tensor):
        flashmask_info = FlashMaskInfoPaddle(
            startend_row_indices=flashmask_info,
            is_causal=causal,
        )
    if flashmask_info is not None:
        assert isinstance(flashmask_info, FlashMaskInfoPaddle)
        # No valid_block_count here: unlike the forward, no backward kernel reads it.
        # It only ever fed a host-side density heuristic that chose between 2CTA and
        # 1CTA+split_dv for d192/dv128; the 2CTA path now skips fully-masked m blocks
        # itself, so the heuristic and the block-count scan behind it are both gone.
        prepare_block_maxmin(flashmask_info, kBlockN=n_block_size)
        cute_flashmask_info = to_cute_flashmask_info(flashmask_info)
        num_flashmask_tensors = 2 * flashmask_info.startend_row_indices.shape[-1]

    is_split_d_bwd = False
    is_split_dv_bwd = False

    # FM-4 backward consumes one split-AG segment at a time and gates each KV tile
    # on the producer's per-work completion bitmap.
    enable_overlap = group is not None and group.world_size > 1
    overlap_view_args = None
    overlap_bhsd_layout = None
    overlap_segment_idx = None
    if enable_overlap:
        if compute_capability != 10:
            raise NotImplementedError("FM-4 overlap bwd is only supported on SM100")
        overlap_runtime = _get_overlap_runtime()
        assert flashmask_info is not None, "overlap bwd requires flashmask_info (the post-AG mask)"
        assert cu_seqlens_q is None and cu_seqlens_k is None, "overlap bwd does not support varlen"
        assert kv_postprocess_start is None and kv_postprocess_end is None, (
            "overlap bwd owns the KV segment postprocess range"
        )
        assert k.dtype == paddle.bfloat16, "overlap SRBuffer is bf16"
        assert not causal, "overlap bwd does not support causal yet"
        startend_row_indices = flashmask_info.startend_row_indices
        overlap_runtime.ensure_initialized(
            k, v, group, mask_head=startend_row_indices.shape[1]
        )
        overlap_bhsd_layout = overlap_runtime.use_bhsd_layout()
        overlap_stream = overlap_runtime.current_stream_handle()
        overlap_ag_args = overlap_runtime.start_backward_ag(
            k, v, startend_row_indices, overlap_stream
        )
        overlap_view_args = overlap_ag_args.kv_view(0)
        segment_seqlen = overlap_ag_args.segment_seqlen
        full_seqlen_k = startend_row_indices.shape[2]
        assert full_seqlen_k == segment_seqlen * overlap_ag_args.num_segments
        assert segment_seqlen % n_block_size == 0
        segment_nblocks = segment_seqlen // n_block_size
        assert segment_nblocks % 4 == 0, (
            "FM-3 segment mask metadata requires a 4-block-aligned segment"
        )
        # Keep the full mask tensors. The SM100 loader applies the segment offset
        # while indexing, preserving the original batch/head stride without copies.
        cute_flashmask_info = to_cute_flashmask_info(flashmask_info)
        # All segment-local scheduler and semaphore shapes use this length. The
        # original local length is retained by k/v and by the final dK/dV outputs.
        seqlen_k = segment_seqlen

    if compute_capability == 9:
        sparse_q = None
        local = False
        cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            local,
            sparse_block_size_q=sparse_q,
            flashmask=cute_flashmask_info is not None,
            deterministic=deterministic,
        )
        m_block_size = cfg.m_block_size
        n_block_size = cfg.n_block_size
        num_stages_Q = cfg.num_stages_Q
        num_stages_dO = cfg.num_stages_dO
        num_stages_PdS = cfg.num_stages_PdS
        SdP_swapAB = cfg.SdP_swapAB
        dKV_swapAB = cfg.dKV_swapAB
        dQ_swapAB = cfg.dQ_swapAB
        AtomLayoutMSdP = cfg.AtomLayoutMSdP
        AtomLayoutNdKV = cfg.AtomLayoutNdKV
        AtomLayoutMdQ = cfg.AtomLayoutMdQ
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
        is_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is not None
        )
    else:
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1

        if is_bigd_bwd:
            # The big-headdim kernel solves its own (tile, chunk) config and always
            # runs 1 CTA per MMA; the split-d / split-dv flags are the other kernel's
            # halving scheme and do not apply.
            is_split_d_bwd = False
            is_split_dv_bwd = False
        else:
            # d192/dv128 included: it used to fall back to split_dv on sparse masks,
            # but the 2CTA path skips fully-masked m blocks itself now. d256/dv256
            # used to split both axes because dK did not fit in TMEM; the 2CTA folded
            # layout (n_block_size=64 above) keeps dK and dV resident instead.
            is_split_d_bwd = False
            is_split_dv_bwd = False

        need_large_cluster = (head_dim > 128) or (head_dim == 128 and flashmask_info is None)
        if is_bigd_bwd:
            cluster_size = 1
        elif not (is_split_d_bwd or is_split_dv_bwd):
            cluster_size = 2 if need_large_cluster else 1
        else:
            cluster_size = 1
        use_2cta_instrs = cluster_size == 2

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]

    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        seqlen_k = None
        total_k = k.shape[0]

    if enable_overlap:
        # k/v are local inputs, while the grad kernel consumes one gathered SRBuffer
        # segment at a time. The segment view supplies the scheduler and dK/dV shape.
        seqlen_k = overlap_view_args.shape[1]
        total_k = batch_size * seqlen_k
        assert num_head_kv == overlap_view_args.shape[2]
        assert head_dim == overlap_view_args.shape[3]
    elif cu_seqlens_k is None:
        assert k.shape == [batch_size, seqlen_k, num_head_kv, head_dim]
        assert v.shape == [batch_size, seqlen_k, num_head_kv, head_dim_v]
    else:
        assert k.shape == [total_k, num_head_kv, head_dim]
        assert v.shape == [total_k, num_head_kv, head_dim_v]
        assert cu_seqlens_k.shape == [
            batch_size + 1,
        ], "cu_seqlens_k must have shape (batch_size + 1,)"

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == [
            batch_size + 1,
        ], "cu_seqlens_q must have shape (batch_size + 1,)"

        assert out.shape == [total_q, num_head, head_dim_v]
        assert dout.shape == [total_q, num_head, head_dim_v]
        assert lse.shape == [num_head, total_q], "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == [batch_size, seqlen_q, num_head, head_dim_v]
        assert dout.shape == [batch_size, seqlen_q, num_head, head_dim_v]
        assert lse.shape == [batch_size, num_head, seqlen_q], (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [paddle.float16, paddle.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == paddle.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == paddle.float32, "lse must be float32"
    assert all(
        t is None or t.place.is_gpu_place()
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256 or is_bigd_bwd, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if compute_capability == 10:
        pack_gqa = False  # override for now

    place = q.place
    # TODO: check if this is the right rounding
    # Round head_dim to multiple of 64 for SM100 to ensure tiled_copy_2d compatibility
    # in postprocess (128 threads must divide tile_hdim/copy_elems evenly)
    hdim_round_to = 64 if compute_capability == 10 else 32
    head_dim_rounded = (head_dim + hdim_round_to - 1) // hdim_round_to * hdim_round_to
    head_dim_v_rounded = (head_dim_v + hdim_round_to - 1) // hdim_round_to * hdim_round_to

    # dq: dq_accum -> dq postprocess always writes the full m_block range, so empty_like
    # is safe on fixed-seqlen path and avoids a redundant bf16 fill (~150us in 4k bench).
    # dk/dv: only safe to skip the zero-fill when postprocess writes every row, i.e.
    # when dk_accum/dv_accum is in use (qhead_per_kvhead > 1 or is_split_d_bwd). In the
    # GQA-ratio==1 + not-split-d path the main bwd kernel writes dk/dv directly, and
    # FlashMask can skip whole n_blocks (no Q rows attend) — those rows would stay
    # garbage with empty_like and break correctness. Fall back to zeros_like there.
    kv_postprocess_full = (
        (qhead_per_kvhead > 1) or is_split_d_bwd or is_split_dv_bwd or is_bigd_bwd
    )
    fixed_seqlen = cu_seqlens_q is None and cu_seqlens_k is None
    if fixed_seqlen:
        dq = paddle.empty_like(q)
    else:
        dq = paddle.zeros_like(q)
    # Native RS writes the final local dK/dV directly into these tensors.
    if enable_overlap:
        dk = paddle.empty_like(k)
        dv = paddle.empty_like(v)
    elif fixed_seqlen and kv_postprocess_full:
        dk = paddle.empty_like(k)
        dv = paddle.empty_like(v)
    else:
        dk = paddle.zeros_like(k)
        dv = paddle.zeros_like(v)

    # ---- Compute shapes for fp32 accum workspaces ----
    if cu_seqlens_q is None:
        seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
        dq_accum_shape = [batch_size, num_head, seqlen_q_rounded * head_dim_rounded]
        dpsum_shape = [batch_size, num_head, seqlen_q_rounded]
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum_shape = [num_head, total_q_rounded_padded * head_dim_rounded]
        dpsum_shape = [num_head, total_q_rounded_padded]
    dpsum = paddle.empty(shape=dpsum_shape, dtype=paddle.float32)
    lse_log2 = paddle.empty(shape=dpsum_shape, dtype=paddle.float32)

    # The big-headdim kernel never writes dK / dV directly: both leave TMEM one
    # head_dim chunk at a time through the fp32 accumulators.
    need_kv_accum = (
        qhead_per_kvhead > 1 or is_split_d_bwd or is_split_dv_bwd or is_bigd_bwd
    )
    if need_kv_accum:
        if cu_seqlens_k is None:
            seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
            num_n_blocks = seqlen_k_rounded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                seqlen_k_rounded = seqlen_k_rounded + n_block_size
            dk_accum_shape = [batch_size, num_head_kv, seqlen_k_rounded * head_dim_rounded]
            dv_accum_shape = [batch_size, num_head_kv, seqlen_k_rounded * head_dim_v_rounded]
        else:
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * n_block_size - 1) // n_block_size * n_block_size
            )
            num_n_blocks = total_k_rounded_padded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                total_k_rounded_padded = total_k_rounded_padded + n_block_size
            dk_accum_shape = [num_head_kv, total_k_rounded_padded * head_dim_rounded]
            dv_accum_shape = [num_head_kv, total_k_rounded_padded * head_dim_v_rounded]

    kv_postprocess_enabled = kv_postprocess_start is not None or kv_postprocess_end is not None

    def _kv_postprocess_range():
        if not kv_postprocess_enabled:
            return 0, seqlen_k
        assert fixed_seqlen, "kv_postprocess range only supports fixed seqlen"
        start = 0 if kv_postprocess_start is None else int(kv_postprocess_start)
        end = seqlen_k if kv_postprocess_end is None else int(kv_postprocess_end)
        assert 0 <= start <= end <= seqlen_k, (
            f"invalid kv_postprocess range [{start}, {end}) for seqlen_k={seqlen_k}"
        )
        assert start % n_block_size == 0, (
            f"kv_postprocess_start must be aligned to n_block_size={n_block_size}, got {start}"
        )
        assert end % n_block_size == 0, (
            f"kv_postprocess_end must be aligned to n_block_size={n_block_size}, got {end}"
        )
        return start, end

    kv_post_start, kv_post_end = _kv_postprocess_range()


    # ---- Compute shapes for fp32 accum workspaces ----
    def _numel(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    zero_kv_accum_range = kv_postprocess_enabled and need_kv_accum and accum_zero_axis1_kv is not None
    zero_specs = []  # list of (key, shape, numel)
    if is_split_d_bwd:
        zero_specs.append(("dq_accum", dq_accum_shape, _numel(dq_accum_shape)))
    if need_kv_accum and not zero_kv_accum_range:
        zero_specs.append(("dk_accum", dk_accum_shape, _numel(dk_accum_shape)))
        zero_specs.append(("dv_accum", dv_accum_shape, _numel(dv_accum_shape)))

    _accum_buffers = {}
    if len(zero_specs) >= 2:
        _zero_total = sum(s[2] for s in zero_specs)
        _zero_big = paddle.zeros(shape=[_zero_total], dtype=paddle.float32)
        _off = 0
        for key, shape, numel in zero_specs:
            _accum_buffers[key] = _zero_big[_off : _off + numel].reshape(shape)
            _off += numel
        # Keep _zero_big alive in this frame so views remain valid.
    elif len(zero_specs) == 1:
        key, shape, _ = zero_specs[0]
        _accum_buffers[key] = paddle.zeros(shape=shape, dtype=paddle.float32)

    if is_split_d_bwd:
        dq_accum = _accum_buffers["dq_accum"]
    else:
        dq_accum = paddle.empty(shape=dq_accum_shape, dtype=paddle.float32)
    if need_kv_accum:
        if zero_kv_accum_range:
            # if start/end range is given, we can initialize only part of the accum buffer
            dk_accum = paddle.empty(shape=dk_accum_shape, dtype=paddle.float32)
            dv_accum = paddle.empty(shape=dv_accum_shape, dtype=paddle.float32)
            if is_split_d_bwd:
                dk_zero_hdim, dv_zero_hdim = head_dim // 2, head_dim_v // 2
                dk_split, dv_split = True, True
            elif is_split_dv_bwd:
                dk_zero_hdim, dv_zero_hdim = head_dim_rounded, head_dim_v // 2
                dk_split, dv_split = False, True
            else:
                dk_zero_hdim, dv_zero_hdim = head_dim_rounded, head_dim_v_rounded
                dk_split, dv_split = False, False
            accum_zero_axis1_kv(
                dk_accum,
                dv_accum,
                kv_post_start,
                kv_post_end - kv_post_start,
                dk_zero_hdim,
                dv_zero_hdim,
                dk_split,
                dv_split,
            )
        else:
            dk_accum = _accum_buffers["dk_accum"]
            dv_accum = _accum_buffers["dv_accum"]

    dtype = paddle2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, out, dout, dq, dk, dv)
    ]
    lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=lse.ndim - 1
    )
    dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (dq_accum, dpsum, lse_log2)
    ]
    if need_kv_accum:
        dk_accum_tensor, dv_accum_tensor = [
            from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (dk_accum, dv_accum)
        ]
    cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=t.ndim - 1)
        if t is not None
        else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    if deterministic:
        dQ_semaphore = paddle.zeros(
            shape=[batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size], dtype=paddle.int32
        )
    else:
        dQ_semaphore = None

    if deterministic and (qhead_per_kvhead > 1 or is_split_d_bwd or is_split_dv_bwd):
        dK_semaphore = paddle.zeros(
            shape=[batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2], dtype=paddle.int32
        )
        dV_semaphore = paddle.zeros(
            shape=[batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2], dtype=paddle.int32
        )
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Note(wusiming): paddle doesn’t expose the physics layout, so assert that the tensor is contiguous here
    if dQ_semaphore is not None:
        assert dQ_semaphore.is_contiguous()
    if dK_semaphore is not None:
        assert dK_semaphore.is_contiguous()
    if dV_semaphore is not None:
        assert dV_semaphore.is_contiguous()
    # Must match the compile-time fake semaphore layout (make_fake_tensor in
    # _make_fake_bwd_tensors: leading_dim = ndim-1, other strides sym_int64/dynamic),
    # exactly like every other bwd tensor above. Using convert_from_dlpack_leading_static
    # here (compact_shape_dynamic) mismatched the traced fake layout, so at runtime the
    # kernel read shape values as strides -> garbage semaphore address -> the dQ
    # deterministic wait_eq spun on uninitialized memory and hung for n_block >= 2.
    dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=t.ndim - 1)
        if t is not None
        else None
        for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
    ]
    current_stream = cuda.CUstream(paddle.device.current_stream().stream_base.cuda_stream)

    # Rebuild segment-local SRBuffer views inside the MLIR context. Split AG
    # publishes one completion flag per communication work item.
    if enable_overlap:
        overlap_view = overlap_view_args
        overlap_k_addr = cutlass.Int64(overlap_view.k_addr)
        overlap_v_addr = cutlass.Int64(overlap_view.v_addr)
        overlap_work_done_addr = cutlass.Int64(overlap_ag_args.work_done_addr)
        _ob, _os, _oh, _od = overlap_view.shape
        overlap_b = cutlass.Int32(_ob)
        overlap_s = cutlass.Int32(_os)
        overlap_h = cutlass.Int32(_oh)
        overlap_d = cutlass.Int32(_od)
        overlap_segment_idx = cutlass.Int32(0)
        overlap_comm_rpb = overlap_ag_args.comm_rpb
        overlap_dk_send_addr, overlap_dv_send_addr = overlap_ag_args.dkv_send_addrs(0)
        overlap_dk_addr = (
            None if need_kv_accum else cutlass.Int64(overlap_dk_send_addr)
        )
        overlap_dv_addr = (
            None if need_kv_accum else cutlass.Int64(overlap_dv_send_addr)
        )
    else:
        overlap_k_addr = None
        overlap_v_addr = None
        overlap_work_done_addr = None
        overlap_dk_addr = None
        overlap_dv_addr = None
        overlap_b = None
        overlap_s = None
        overlap_h = None
        overlap_d = None
        overlap_comm_rpb = None
        overlap_bhsd_layout = None

    compile_key_pre = (compute_capability, dtype, head_dim, head_dim_v, head_dim_rounded, m_block_size, num_threads)
    if compile_key_pre not in _flash_attn_bwd.compile_cache_pre:
        fa_bwd_pre = FlashAttentionBackwardPreprocess(
            dtype,
            head_dim,
            head_dim_v,
            m_block_size,
            dq_head_dim=head_dim_rounded,
        )
        # Compile with FA4-style fake tensors (all dims are sym_int, strides have
        # divisibility=8/4 → 128-bit alignment statically guaranteed).
        (
            f_mQ, f_mK, f_mV, f_mO, f_mdO, f_mdQ, f_mdK, f_mdV,
            f_mLSE, f_mLSElog2, f_mPdPsum, f_mdQaccum, f_mdKaccum, f_mdVaccum,
            _f_mdQ_semaphore, _f_mdK_semaphore, _f_mdV_semaphore,
        ) = _make_fake_bwd_tensors(dtype, has_gqa=qhead_per_kvhead > 1)
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            f_mO,
            f_mdO,
            f_mPdPsum,
            f_mLSE,
            f_mLSElog2,
            f_mdQaccum,
            None,  # mCuSeqlensQ
            None,  # mSeqUsedQ
            None,  # mdLSE
            current_stream,
        )
    _flash_attn_bwd.compile_cache_pre[compile_key_pre](
        o_tensor,
        do_tensor,
        dpsum_tensor,
        lse_tensor,
        lse_log2_tensor,
        dq_accum_tensor,
        cu_seqlens_q_tensor,
        seqused_q_tensor,
        None,  # mdLSE
        current_stream,
    )

    # Backward kernel: compute dk, dv, dq_accum.
    if compute_capability == 9:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            num_flashmask_tensors,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            deterministic,
        )
    else:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            num_flashmask_tensors,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            deterministic,
            is_split_d_bwd if compute_capability == 10 else False,
            is_split_dv_bwd if compute_capability == 10 else False,
            # overlap: an overlap grad kernel (K/V rebuilt from SRBuffer addr) and a
            # plain one for the same shapes are different compiled artifacts.
            enable_overlap,
            overlap_bhsd_layout,
            overlap_comm_rpb,
        )

    # SM100/SM110 uses default from function signature (384).
    if compute_capability not in [9, 12]:
        num_threads = 384

    if compile_key not in _flash_attn_bwd.compile_cache:
        fa_bwd_sm80 = FlashAttentionBackwardSm80(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            m_block_size,
            n_block_size,
            num_stages_Q,
            num_stages_dO,
            num_threads,
            pack_gqa,
            causal,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs=V_in_regs,
        )
        if compute_capability == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                is_local=False,
                deterministic=deterministic,
                tile_m=m_block_size,
                tile_n=n_block_size,
                Q_stage=num_stages_Q,
                dO_stage=num_stages_dO,
                PdS_stage=num_stages_PdS,
                SdP_swapAB=SdP_swapAB,
                dKV_swapAB=dKV_swapAB,
                dQ_swapAB=dQ_swapAB,
                AtomLayoutMSdP=AtomLayoutMSdP,
                AtomLayoutNdKV=AtomLayoutNdKV,
                AtomLayoutMdQ=AtomLayoutMdQ,
                num_threads=num_threads,
                V_in_regs=V_in_regs,
                # score_mod=score_mod,
                # score_mod_bwd=score_mod_bwd,
                # mask_mod=mask_mod,
                # has_aux_tensors=aux_tensors is not None,
                # subtile_factor=subtile_factor,
                # dQ_single_wg=dQ_single_wg,
            )

            # TODO: check @can_implement
            # Compile with FA4-style fake tensors (fully-symbolic dims with stride
            # divisibility hints). Pass the real cute_flashmask_info at compile so
            # the flashmask code path (const_expr enable flags) is generated; its
            # tensors are marked layout-dynamic so the kernel stays shape-generic.
            (
                f_mQ, f_mK, f_mV, f_mO_unused, f_mdO,
                f_mdQ_unused, f_mdK, f_mdV,
                f_mLSE_unused, f_mLSElog2, f_mPdPsum,
                f_mdQaccum, f_mdKaccum, f_mdVaccum,
                f_mdQ_semaphore, f_mdK_semaphore, f_mdV_semaphore,
            ) = _make_fake_bwd_tensors(
                dtype, has_gqa=qhead_per_kvhead > 1, deterministic=deterministic, cluster_size=1
            )
            _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
                fa_bwd_obj,
                f_mQ,
                f_mK,
                f_mV,
                f_mdO,
                f_mLSElog2,
                f_mPdPsum,
                f_mdQaccum,
                f_mdK if not need_kv_accum else f_mdKaccum,
                f_mdV if not need_kv_accum else f_mdVaccum,
                softmax_scale,
                mCuSeqlensQ=None,
                mCuSeqlensK=None,
                mSeqUsedQ=None,
                mSeqUsedK=None,
                mdQ_semaphore=f_mdQ_semaphore,
                mdK_semaphore=f_mdK_semaphore,
                mdV_semaphore=f_mdV_semaphore,
                flashmask_info=cute_flashmask_info,
                stream=current_stream,
            )
        else:
            if is_bigd_bwd:
                # Solves its own tile / chunk config from (head_dim, head_dim_v); the
                # remaining knobs are 1-CTA only for now. fm_bound_num has to come in
                # here because the startend_row_indices tensor reaches the kernel with a
                # fully dynamic layout, so its width is not visible at trace time.
                fa_bwd_obj = FlashAttentionBackwardSm100BigD.from_shape(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    qhead_per_kvhead=qhead_per_kvhead,
                    deterministic=deterministic,
                    fm_bound_num=(
                        0
                        if cute_flashmask_info is None
                        else flashmask_info.startend_row_indices.shape[-1]
                    ),
                )
            else:
                fa_bwd_obj = FlashAttentionBackwardSm100(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    qhead_per_kvhead=qhead_per_kvhead,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    deterministic=deterministic,
                    is_split_d=is_split_d_bwd,
                    is_split_dv=is_split_dv_bwd,
                )
            _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
                fa_bwd_obj,
                q_tensor,
                k_tensor,
                v_tensor,
                do_tensor,
                lse_log2_tensor,
                dpsum_tensor,
                dq_accum_tensor,
                dk_tensor if not need_kv_accum else dk_accum_tensor,
                dv_tensor if not need_kv_accum else dv_accum_tensor,
                softmax_scale,
                mCuSeqlensQ=cu_seqlens_q_tensor,
                mCuSeqlensK=cu_seqlens_k_tensor,
                mSeqUsedQ=seqused_q_tensor,
                mSeqUsedK=seqused_k_tensor,
                mdQ_semaphore=dQ_semaphore_tensor,
                mdK_semaphore=dK_semaphore_tensor,
                mdV_semaphore=dV_semaphore_tensor,
                flashmask_info=cute_flashmask_info,
                overlap_k_addr=overlap_k_addr,
                overlap_v_addr=overlap_v_addr,
                overlap_work_done_addr=overlap_work_done_addr,
                overlap_segment_idx=overlap_segment_idx,
                overlap_dk_addr=overlap_dk_addr,
                overlap_dv_addr=overlap_dv_addr,
                overlap_b=overlap_b,
                overlap_s=overlap_s,
                overlap_h=overlap_h,
                overlap_d=overlap_d,
                overlap_comm_rpb=overlap_comm_rpb,
                overlap_bhsd_layout=overlap_bhsd_layout,
                stream=current_stream,
            )
    if compute_capability == 9:
        _flash_attn_bwd.compile_cache[compile_key](
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not need_kv_accum else dk_accum_tensor,
            dv_tensor if not need_kv_accum else dv_accum_tensor,
            softmax_scale,
            mCuSeqlensQ=cu_seqlens_q_tensor,
            mCuSeqlensK=cu_seqlens_k_tensor,
            mSeqUsedQ=seqused_q_tensor,
            mSeqUsedK=seqused_k_tensor,
            mdQ_semaphore=dQ_semaphore_tensor,
            mdK_semaphore=dK_semaphore_tensor,
            mdV_semaphore=dV_semaphore_tensor,
            flashmask_info=cute_flashmask_info,
            stream=current_stream,
        )
    else:
        def _run_bwd_main(
            segment_flashmask_info,
            segment_k_addr,
            segment_v_addr,
            segment_dk_addr,
            segment_dv_addr,
            segment_idx=None,
        ):
            _flash_attn_bwd.compile_cache[compile_key](
                q_tensor,
                k_tensor,
                v_tensor,
                do_tensor,
                lse_log2_tensor,
                dpsum_tensor,
                dq_accum_tensor,
                dk_tensor if not need_kv_accum else dk_accum_tensor,
                dv_tensor if not need_kv_accum else dv_accum_tensor,
                softmax_scale,
                mCuSeqlensQ=cu_seqlens_q_tensor,
                mCuSeqlensK=cu_seqlens_k_tensor,
                mSeqUsedQ=seqused_q_tensor,
                mSeqUsedK=seqused_k_tensor,
                mdQ_semaphore=dQ_semaphore_tensor,
                mdK_semaphore=dK_semaphore_tensor,
                mdV_semaphore=dV_semaphore_tensor,
                flashmask_info=segment_flashmask_info,
                overlap_k_addr=segment_k_addr,
                overlap_v_addr=segment_v_addr,
                overlap_work_done_addr=overlap_work_done_addr,
                overlap_segment_idx=(
                    None if segment_idx is None else cutlass.Int32(segment_idx)
                ),
                overlap_dk_addr=segment_dk_addr,
                overlap_dv_addr=segment_dv_addr,
                overlap_b=overlap_b,
                overlap_s=overlap_s,
                overlap_h=overlap_h,
                overlap_d=overlap_d,
                stream=current_stream,
            )

        if not enable_overlap:
            _run_bwd_main(
                cute_flashmask_info,
                overlap_k_addr,
                overlap_v_addr,
                overlap_dk_addr,
                overlap_dv_addr,
            )

    num_threads = 256 if compute_capability == 9 else 128
    arch = compute_capability * 10

    # The dK/dV postprocess decodes the fp32 accum with its 2-CTA branch
    # (flash_bwd_postprocess.py:412+), which hard-codes row_groups = 2: it reads a
    # panel of (128 threads x ncol) values and interprets threads 0..63 as rows
    # 0..63 of the block and threads 64..127 as the SAME 64 rows at hdim + 128.
    # That is exactly what the folded 2CTA bwd kernel writes with a 64-row KV
    # tile, so the postprocess block must span TWO of its n_blocks -> 128 rows.
    # (seqlen_k_rounded is padded to a multiple of 2 * n_block_size whenever
    # cluster_size == 2, so the pairing never runs off the end.) It is also the
    # only workable choice mechanically: tcgen05.make_tmem_copy cannot slice a
    # 1-CTA accumulator with fewer than 128 rows -- at block_size=64 it becomes
    # (((16,4),hdim),1,1):(((65536,2097152),1),0,0) and every Ld32x32b Repetition
    # (8..128) is rejected.
    n_block_size_kv_post = (
        max(n_block_size, 128) if compute_capability == 10 else n_block_size
    )
    # ... and the same folding is why the dK/dV postprocess must use the 2-CTA
    # decode branch: it is the only one that reads a panel as "threads 0..63 ->
    # rows 0..63, threads 64..127 -> the same rows at hdim + hdim/2". The plain
    # branch assumes row == thread index, i.e. an unfolded 128-row writer.
    dKV_accum_folded = (
        compute_capability == 10 and use_2cta_instrs and n_block_size < 128
    )

    kv_postprocess_enabled = kv_postprocess_start is not None or kv_postprocess_end is not None

    def _kv_postprocess_range():
        if not kv_postprocess_enabled:
            return 0, seqlen_k
        assert fixed_seqlen, "kv_postprocess range only supports fixed seqlen"
        start = 0 if kv_postprocess_start is None else int(kv_postprocess_start)
        end = seqlen_k if kv_postprocess_end is None else int(kv_postprocess_end)
        assert 0 <= start <= end <= seqlen_k, (
            f"invalid kv_postprocess range [{start}, {end}) for seqlen_k={seqlen_k}"
        )
        assert start % n_block_size == 0, (
            f"kv_postprocess_start must be aligned to n_block_size={n_block_size}, got {start}"
        )
        return start, end

    kv_post_start, kv_post_end = _kv_postprocess_range()

    def _slice_kv_out(t):
        if not kv_postprocess_enabled:
            return t
        return t[:, kv_post_start:kv_post_end, :, :]

    def _slice_kv_accum(t, accum_hdim):
        if not kv_postprocess_enabled:
            return t
        return t[..., kv_post_start * accum_hdim:kv_post_end * accum_hdim]

    def _to_cute(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )

    def _kv_out_cute(t, tensor):
        return _to_cute(_slice_kv_out(t)) if kv_postprocess_enabled else tensor

    def _kv_accum_cute(t, tensor, accum_hdim):
        return _to_cute(_slice_kv_accum(t, accum_hdim)) if kv_postprocess_enabled else tensor

    def _postprocess_run(
        d_accum_t,
        d_out_t,
        scale,
        hd,
        block_size,
        atom_layout,
        swapAB,
        use_2cta,
        cluster,
        cu_seqlens_t,
        seqused_t,
        cache_tag,
        raw_output_addr=None,
        raw_b=None,
        raw_s=None,
        raw_h=None,
        raw_d=None,
        raw_storage_d=None,
    ):
        compile_key_post = (
            dtype,
            hd,
            arch,
            block_size,
            num_threads,
            atom_layout,
            swapAB,
            use_2cta,
            cluster,
            cache_tag,
            raw_output_addr is not None,
        )
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, hd, arch, block_size, num_threads, atom_layout, swapAB,
                use_2cta_instrs=use_2cta, cluster_size=cluster,
            )
 
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                d_accum_t, d_out_t, scale,
                cu_seqlens_t, seqused_t,
                raw_output_addr=raw_output_addr,
                raw_b=raw_b,
                raw_s=raw_s,
                raw_h=raw_h,
                raw_d=raw_d,
                raw_storage_d=raw_storage_d,
                stream=current_stream,
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            d_accum_t, d_out_t, scale,
            cu_seqlens_t, seqused_t,
            raw_output_addr=raw_output_addr,
            raw_b=raw_b,
            raw_s=raw_s,
            raw_h=raw_h,
            raw_d=raw_d,
            raw_storage_d=raw_storage_d,
            stream=current_stream,
        )

    if enable_overlap:
        raw_b = cutlass.Int32(batch_size)
        raw_s = cutlass.Int32(segment_seqlen)
        raw_h = cutlass.Int32(num_head_kv)
        raw_storage_d_k = cutlass.Int32(head_dim)
        raw_storage_d_v = cutlass.Int32(head_dim_v)

        def _raw_addr(addr, element_offset=0):
            return cutlass.Int64(int(addr) + 2 * element_offset)

        def _run_overlap_dkv_postprocess(dk_send_addr, dv_send_addr):
            if is_split_d_bwd:
                half_hdim = head_dim // 2
                half_hdim_v = head_dim_v // 2
                dk_accum_low, dk_accum_high = (
                    dk_accum[..., : dk_accum.shape[-1] // 2],
                    dk_accum[..., dk_accum.shape[-1] // 2 :],
                )
                dv_accum_low, dv_accum_high = (
                    dv_accum[..., : dv_accum.shape[-1] // 2],
                    dv_accum[..., dv_accum.shape[-1] // 2 :],
                )
                for accum_part, output_addr, hd in (
                    (dk_accum_low, _raw_addr(dk_send_addr), half_hdim),
                    (dk_accum_high, _raw_addr(dk_send_addr, half_hdim), half_hdim),
                ):
                    _postprocess_run(
                        _to_cute(_slice_kv_accum(accum_part, hd)), dk_tensor,
                        softmax_scale, hd, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                        False, 1, cu_seqlens_k_tensor, seqused_k_tensor, "ovl_dk_split",
                        output_addr, raw_b, raw_s, raw_h, cutlass.Int32(hd),
                        raw_storage_d_k,
                    )
                for accum_part, output_addr, hd in (
                    (dv_accum_low, _raw_addr(dv_send_addr), half_hdim_v),
                    (dv_accum_high, _raw_addr(dv_send_addr, half_hdim_v), half_hdim_v),
                ):
                    _postprocess_run(
                        _to_cute(_slice_kv_accum(accum_part, hd)), dv_tensor,
                        cutlass.Float32(1.0), hd, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                        False, 1, cu_seqlens_k_tensor, seqused_k_tensor, "ovl_dv_split",
                        output_addr, raw_b, raw_s, raw_h, cutlass.Int32(hd),
                        raw_storage_d_v,
                    )
            elif is_split_dv_bwd:
                half_hdim_v = head_dim_v // 2
                _postprocess_run(
                    _kv_accum_cute(dk_accum, dk_accum_tensor, head_dim_rounded),
                    _kv_out_cute(dk, dk_tensor), softmax_scale,
                    head_dim, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                    False, cluster_size, cu_seqlens_k_tensor, seqused_k_tensor, "ovl_dk",
                    _raw_addr(dk_send_addr), raw_b, raw_s, raw_h,
                    cutlass.Int32(head_dim), raw_storage_d_k,
                )
                dv_accum_low, dv_accum_high = dv_accum[..., : dv_accum.shape[-1] // 2], dv_accum[..., dv_accum.shape[-1] // 2 :]
                for accum_part, output_addr in (
                    (dv_accum_low, _raw_addr(dv_send_addr)),
                    (dv_accum_high, _raw_addr(dv_send_addr, half_hdim_v)),
                ):
                    _postprocess_run(
                        _to_cute(_slice_kv_accum(accum_part, half_hdim_v)), dv_tensor,
                        cutlass.Float32(1.0), half_hdim_v, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                        False, 1, cu_seqlens_k_tensor, seqused_k_tensor, "ovl_dv_split",
                        output_addr, raw_b, raw_s, raw_h, cutlass.Int32(half_hdim_v),
                        raw_storage_d_v,
                    )
            else:
                _postprocess_run(
                    _kv_accum_cute(dk_accum, dk_accum_tensor, head_dim_rounded),
                    _kv_out_cute(dk, dk_tensor), softmax_scale,
                    head_dim, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                    dKV_accum_folded, cluster_size, cu_seqlens_k_tensor, seqused_k_tensor, "ovl_dk",
                    _raw_addr(dk_send_addr), raw_b, raw_s, raw_h,
                    cutlass.Int32(head_dim), raw_storage_d_k,
                )
                _postprocess_run(
                    _kv_accum_cute(dv_accum, dv_accum_tensor, head_dim_v_rounded),
                    _kv_out_cute(dv, dv_tensor), cutlass.Float32(1.0),
                    head_dim_v, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                    dKV_accum_folded, cluster_size, cu_seqlens_k_tensor, seqused_k_tensor, "ovl_dv",
                    _raw_addr(dv_send_addr), raw_b, raw_s, raw_h,
                    cutlass.Int32(head_dim_v), raw_storage_d_v,
                )

        for segment_idx in range(overlap_ag_args.num_segments):
            segment_view = overlap_ag_args.kv_view(segment_idx)
            segment_dk_send_addr, segment_dv_send_addr = (
                overlap_ag_args.dkv_send_addrs(segment_idx)
            )
            segment_k_addr = cutlass.Int64(segment_view.k_addr)
            segment_v_addr = cutlass.Int64(segment_view.v_addr)
            segment_dk_addr = (
                None
                if need_kv_accum
                else cutlass.Int64(segment_dk_send_addr)
            )
            segment_dv_addr = (
                None
                if need_kv_accum
                else cutlass.Int64(segment_dv_send_addr)
            )

            if not need_kv_accum:
                overlap_runtime.wait_dkv_buffer(segment_idx, overlap_stream)
            _run_bwd_main(
                cute_flashmask_info,
                segment_k_addr,
                segment_v_addr,
                segment_dk_addr,
                segment_dv_addr,
                segment_idx=segment_idx,
            )
            if need_kv_accum:
                overlap_runtime.wait_dkv_buffer(segment_idx, overlap_stream)
                _run_overlap_dkv_postprocess(
                    segment_dk_send_addr, segment_dv_send_addr
                )
            overlap_runtime.run_backward_rs(dk, dv, segment_idx, overlap_stream)

            if segment_idx + 1 < overlap_ag_args.num_segments:
                if need_kv_accum:
                    dk_accum.zero_()
                    dv_accum.zero_()
                overlap_runtime.start_backward_segment(
                    segment_idx + 1, overlap_stream
                )
                if deterministic:
                    dQ_semaphore.zero_()
                    if need_kv_accum:
                        dK_semaphore.zero_()
                        dV_semaphore.zero_()
        overlap_runtime.wait_backward_rs(overlap_stream)

    if enable_overlap:
        if is_split_d_bwd:
            half_hdim = head_dim // 2
            dq_accum_low, dq_accum_high = dq_accum[..., : dq_accum.shape[-1] // 2], dq_accum[..., dq_accum.shape[-1] // 2 :]
            for accum_part, out_part in (
                (dq_accum_low, dq[..., :half_hdim]),
                (dq_accum_high, dq[..., half_hdim:]),
            ):
                _postprocess_run(
                    _to_cute(accum_part), _to_cute(out_part), softmax_scale,
                    half_hdim, m_block_size, AtomLayoutMdQ, dQ_swapAB,
                    False, 1, cu_seqlens_q_tensor, seqused_q_tensor, "ovl_dq_split",
                )
        else:
            _postprocess_run(
                dq_accum_tensor, dq_tensor, softmax_scale,
                head_dim, m_block_size, AtomLayoutMdQ, dQ_swapAB,
                use_2cta_instrs, 1, cu_seqlens_q_tensor, seqused_q_tensor, "ovl_dq",
            )
    elif is_bigd_bwd:
        # The big-headdim accumulators are blocked as
        #   [head_dim slice][row block][4-col group][row][4 cols]
        # so each slice is byte-identical to a head_dim=slice accumulator. The shared
        # postprocess cannot swallow head_dim 576 in one go (it stages a whole
        # tile x head_dim fp32 tile in SMEM and registers), so it runs once per slice,
        # writing a last-dim slice of the output -- the same trick the split-d path
        # uses, and legal because the postprocess writes dQ/dK/dV with plain gmem
        # copies rather than TMA.
        slice_d = bigd_cfg["accum_slice_d"]
        slice_dv = bigd_cfg["accum_slice_dv"]
        assert head_dim == head_dim_rounded and head_dim_v == head_dim_v_rounded, (
            "the big-headdim postprocess writes whole head_dim slices, so head_dim "
            f"must already be a multiple of 64 (got {head_dim} / {head_dim_v})"
        )
        for accum, out, scale, hd, hd_slice, block, seq_rounded, tag in (
            (dq_accum, dq, softmax_scale, head_dim_rounded, slice_d,
             m_block_size, seqlen_q_rounded, "bigd_dq"),
            (dk_accum, dk, softmax_scale, head_dim_rounded, slice_d,
             n_block_size, seqlen_k_rounded, "bigd_dk"),
            (dv_accum, dv, cutlass.Float32(1.0), head_dim_v_rounded, slice_dv,
             n_block_size, seqlen_k_rounded, "bigd_dv"),
        ):
            accum_slice_elems = seq_rounded * hd_slice
            for j in range(hd // hd_slice):
                _postprocess_run(
                    _to_cute(
                        accum[..., j * accum_slice_elems : (j + 1) * accum_slice_elems]
                    ),
                    _to_cute(out[..., j * hd_slice : (j + 1) * hd_slice]),
                    scale,
                    hd_slice,
                    block,
                    1,
                    False,
                    False,
                    1,
                    cu_seqlens_q_tensor if tag == "bigd_dq" else cu_seqlens_k_tensor,
                    seqused_q_tensor if tag == "bigd_dq" else seqused_k_tensor,
                    tag,
                )
    elif is_split_d_bwd:
        half_hdim = head_dim // 2
        half_hdim_v = head_dim_v // 2

        def _slice_accum(t):
            n = t.shape[-1] // 2
            return t[..., :n], t[..., n:]

        # dQ split [low | high] postprocess.
        # Pass non-contiguous views directly: kernel uses universal gmem copy (not TMA)
        # so strided last-dim-slice writes are fine. This avoids ~2-3 extra full-tensor
        # copies (contiguous() + concat + copy_) per gradient.
        dq_accum_low, dq_accum_high = _slice_accum(dq_accum)
        for accum_part, out_part in (
            (dq_accum_low, dq[..., :half_hdim]),
            (dq_accum_high, dq[..., half_hdim:]),
        ):
            _postprocess_run(
                _to_cute(accum_part), _to_cute(out_part), softmax_scale,
                half_hdim, m_block_size, AtomLayoutMdQ, dQ_swapAB,
                False, 1, cu_seqlens_q_tensor, seqused_q_tensor, "dq_split",
            )

        # dK split [low | high] postprocess
        dk_accum_low, dk_accum_high = _slice_accum(dk_accum)
        dk_accum_low = _slice_kv_accum(dk_accum_low, half_hdim)
        dk_accum_high = _slice_kv_accum(dk_accum_high, half_hdim)
        dk_post = _slice_kv_out(dk)
        for accum_part, out_part in (
            (dk_accum_low, dk_post[..., :half_hdim]),
            (dk_accum_high, dk_post[..., half_hdim:]),
        ):
            _postprocess_run(
                _to_cute(accum_part), _to_cute(out_part), softmax_scale,
                half_hdim, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                False, 1, cu_seqlens_k_tensor, seqused_k_tensor, "dk_split",
            )

        # dV split [low | high] postprocess
        dv_accum_low, dv_accum_high = _slice_accum(dv_accum)
        dv_accum_low = _slice_kv_accum(dv_accum_low, half_hdim_v)
        dv_accum_high = _slice_kv_accum(dv_accum_high, half_hdim_v)
        dv_post = _slice_kv_out(dv)
        for accum_part, out_part in (
            (dv_accum_low, dv_post[..., :half_hdim_v]),
            (dv_accum_high, dv_post[..., half_hdim_v:]),
        ):
            _postprocess_run(
                _to_cute(accum_part), _to_cute(out_part), cutlass.Float32(1.0),
                half_hdim_v, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                False, 1, cu_seqlens_k_tensor, seqused_k_tensor, "dv_split",
            )
    elif is_split_dv_bwd:
        half_hdim_v = head_dim_v // 2

        def _slice_accum(t):
            n = t.shape[-1] // 2
            return t[..., :n], t[..., n:]

        _postprocess_run(
            dq_accum_tensor, dq_tensor, softmax_scale,
            head_dim, m_block_size, AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs, 1, cu_seqlens_q_tensor, seqused_q_tensor, "dq",
        )

        _postprocess_run(
            _kv_accum_cute(dk_accum, dk_accum_tensor, head_dim_rounded),
            _kv_out_cute(dk, dk_tensor),
            softmax_scale,
            head_dim, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
            False, cluster_size, cu_seqlens_k_tensor, seqused_k_tensor, "dk",
        )

        # dV split [low | high] postprocess
        dv_accum_low, dv_accum_high = _slice_accum(dv_accum)
        dv_accum_low = _slice_kv_accum(dv_accum_low, half_hdim_v)
        dv_accum_high = _slice_kv_accum(dv_accum_high, half_hdim_v)
        dv_post = _slice_kv_out(dv)
        for accum_part, out_part in (
            (dv_accum_low, dv_post[..., :half_hdim_v]),
            (dv_accum_high, dv_post[..., half_hdim_v:]),
        ):
            _postprocess_run(
                _to_cute(accum_part), _to_cute(out_part), cutlass.Float32(1.0),
                half_hdim_v, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                False, 1, cu_seqlens_k_tensor, seqused_k_tensor, "dv_split",
            )
    else:
        # Postprocess kernel: convert dq_accum from float32 to dq in bf16/fp16
        _postprocess_run(
            dq_accum_tensor, dq_tensor, softmax_scale,
            head_dim, m_block_size, AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs, 1, cu_seqlens_q_tensor, seqused_q_tensor, "dq",
        )
        
        if qhead_per_kvhead > 1:
            _postprocess_run(
                _kv_accum_cute(dk_accum, dk_accum_tensor, head_dim_rounded),
                _kv_out_cute(dk, dk_tensor),
                softmax_scale,
                head_dim, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                dKV_accum_folded, cluster_size, cu_seqlens_k_tensor, seqused_k_tensor, "dk",
            )
            _postprocess_run(
                _kv_accum_cute(dv_accum, dv_accum_tensor, head_dim_v_rounded),
                _kv_out_cute(dv, dv_tensor),
                cutlass.Float32(1.0),
                head_dim_v, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                dKV_accum_folded, cluster_size, cu_seqlens_k_tensor, seqused_k_tensor, "dv",
            )

    # ---- learnable_sink gradient ----
    # dsink[h] = -sum_{b,s} exp2(sink[h]*log2e - lse_log2[b,h,s]) * delta[b,h,s]
    # where delta == dpsum and lse_log2 == lse * log2e are both already produced by
    # the preprocess kernel above. Padded rows have dpsum == 0 (lse_log2 is 0.0 there,
    # not +inf), so the product exp2(...) * dpsum == 0 and they contribute nothing. A
    # standalone cute-dsl reduction kernel (one block per head, no atomics ->
    # deterministic) consumes the existing preprocess outputs instead of launching
    # multiple Paddle ops.
    dsink = None
    if learnable_sink is not None:
        assert cu_seqlens_q is None, "learnable_sink gradient does not support varlen"
        sink_dtype = learnable_sink.dtype
        dsink = paddle.empty(shape=[num_head], dtype=paddle.float32)
        sink_tensor, dsink_tensor = [
            from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (learnable_sink, dsink)
        ]
        compile_key_dsink = (compute_capability, paddle2cute_dtype_map[sink_dtype], num_threads)
        if compile_key_dsink not in _flash_attn_bwd.compile_cache_dsink:
            fa_bwd_dsink = FlashAttentionBackwardDsink(num_threads=num_threads)
            _flash_attn_bwd.compile_cache_dsink[compile_key_dsink] = cute.compile(
                fa_bwd_dsink,
                dpsum_tensor,
                lse_log2_tensor,
                sink_tensor,
                dsink_tensor,
                current_stream,
            )
        _flash_attn_bwd.compile_cache_dsink[compile_key_dsink](
            dpsum_tensor,
            lse_log2_tensor,
            sink_tensor,
            dsink_tensor,
            current_stream,
        )
        dsink = dsink.astype(sink_dtype)

    return dq, dk, dv, dsink


_flash_attn_bwd.compile_cache_pre = {}
_flash_attn_bwd.compile_cache = {}
_flash_attn_bwd.compile_cache_post = {}
_flash_attn_bwd.compile_cache_dsink = {}


class FlashAttnFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[paddle.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        mask_mod: Optional[Callable] = None,
        full_block_cnt: Optional[paddle.Tensor] = None,
        full_block_idx: Optional[paddle.Tensor] = None,
        mask_block_cnt: Optional[paddle.Tensor] = None,
        mask_block_idx: Optional[paddle.Tensor] = None,
    ):
        # Only create block sparse tensors if at least one block sparse parameter is provided
        block_sparse_tensors = None
        if any(
            t is not None for t in [full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx]
        ):
            block_sparse_tensors = BlockSparseTensorsPaddle(
                full_block_cnt=full_block_cnt,
                full_block_idx=full_block_idx,
                mask_block_cnt=mask_block_cnt,
                mask_block_idx=mask_block_idx,
            )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensor()
        assert all(size is None for size in ctx.window_size), (
            "local attention backward is not supported"
        )
        dq, dk, dv, _ = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            deterministic=ctx.deterministic,
        )
        # TODO(wusiming): do we need to return None for other fwd inputs?
        return dq, dk, dv


class FlashAttnVarlenFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        cu_seqlens_q: Optional[paddle.Tensor],
        cu_seqlens_k: Optional[paddle.Tensor],
        seqused_q: Optional[paddle.Tensor] = None,
        seqused_k: Optional[paddle.Tensor] = None,
        page_table: Optional[paddle.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[paddle.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensor()
        assert seqused_q is None
        assert seqused_k is None
        assert ctx.softcap == 0.0
        dq, dk, dv, _ = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            deterministic=ctx.deterministic,
        )

        # TODO(wusiming): do we need to return None for other fwd inputs?
        return dq, dk, dv


def flash_attn_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[paddle.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[paddle.Tensor] = None,
    full_block_idx: Optional[paddle.Tensor] = None,
    mask_block_cnt: Optional[paddle.Tensor] = None,
    mask_block_idx: Optional[paddle.Tensor] = None,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        mask_mod,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt,
        mask_block_idx,
    )


def flash_attn_varlen_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    seqused_q: Optional[paddle.Tensor] = None,
    seqused_k: Optional[paddle.Tensor] = None,
    page_table: Optional[paddle.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[paddle.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
    )


def _flash_attn_fwd_combine(
    out_partial: paddle.Tensor,
    lse_partial: paddle.Tensor,
    out: paddle.Tensor,
    lse: Optional[paddle.Tensor] = None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    seqused: Optional[paddle.Tensor] = None,
    num_splits_dynamic_ptr: Optional[paddle.Tensor] = None,
    semaphore_to_reset: Optional[paddle.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    # Input validation
    assert out_partial.ndim in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.ndim in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [paddle.float16, paddle.bfloat16, paddle.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    assert lse_partial.dtype == paddle.float32, "lse_partial must be fp32"
    assert out_partial.place.is_gpu_place() and lse_partial.place.is_gpu_place(), (
        "tensors must be on CUDA device"
    )
    assert out_partial.strides[-1] == 1, "out_partial must be contiguous in the last dimension"
    assert lse_partial.strides[-2] == 1, "lse_partial must be contiguous in the seqlen dimension"
    assert lse_partial.shape == out_partial.shape[:-1]

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.ndim == 4

    # Validate output tensor shapes and types
    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == paddle.float32, "lse must be fp32"

    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            assert t.dtype == paddle.int32, f"{name} must be int32"
            assert t.place.is_gpu_place(), f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"

    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    m_block_size = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if m_block_size == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    # Convert to cute tensors (using kernel-formatted tensors)
    out_partial_tensor = from_dlpack(out_partial.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=4 if not is_varlen else 3
    )
    lse_partial_tensor = from_dlpack(lse_partial.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=lse_partial.ndim - 2
    )
    out_tensor = from_dlpack(out.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=3 if not is_varlen else 2
    )
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 2)
        if lse is not None
        else None
    )

    optional_tensors = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if t is not None
        else None
        for t in (cu_seqlens, seqused, num_splits_dynamic_ptr, semaphore_to_reset)
    ]
    cu_seqlens_tensor, seqused_tensor, num_splits_dynamic_tensor, semaphore_tensor = (
        optional_tensors
    )

    current_stream = cuda.CUstream(paddle.device.current_stream().stream_base.cuda_stream)

    # Create combine kernel configuration
    dtype = paddle2cute_dtype_map[out.dtype]
    dtype_partial = paddle2cute_dtype_map[out_partial.dtype]

    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
    )

    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        fa_combine = FlashAttentionForwardCombine(
            dtype=dtype,
            dtype_partial=dtype_partial,
            head_dim=head_dim,
            m_block_size=m_block_size,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
        )

        # Check if implementation is supported
        if not fa_combine.can_implement(
            dtype,
            dtype_partial,
            head_dim,
            m_block_size,
            k_block_size,
            log_max_splits,
            num_threads=256,
        ):
            raise RuntimeError(
                "FlashAttention combine kernel cannot be implemented with given parameters"
            )

        _flash_attn_fwd_combine.compile_cache[compile_key] = cute.compile(
            fa_combine,
            out_partial_tensor,
            lse_partial_tensor,
            out_tensor,
            lse_tensor,
            cu_seqlens_tensor,
            seqused_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
            current_stream,
        )

    _flash_attn_fwd_combine.compile_cache[compile_key](
        out_partial_tensor,
        lse_partial_tensor,
        out_tensor,
        lse_tensor,
        cu_seqlens_tensor,
        seqused_tensor,
        num_splits_dynamic_tensor,
        semaphore_tensor,
        current_stream,
    )


_flash_attn_fwd_combine.compile_cache = {}


def flash_attn_combine(
    out_partial: paddle.Tensor,
    lse_partial: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    out_dtype: Optional[paddle.dtype] = None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    seqused: Optional[paddle.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.ndim in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.ndim in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype == paddle.float32, "out_partial must be fp32 (from accumulation)"
    assert lse_partial.dtype == paddle.float32, "lse_partial must be fp32"

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.ndim == 4

    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == [num_splits, total_q, num_heads], (
            "lse_partial shape mismatch for varlen"
        )
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == [num_splits, batch_size, seqlen, num_heads], (
            "lse_partial shape mismatch"
        )

    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype

    # Create output if not provided
    place = out_partial.place
    if out is None:
        if is_varlen:
            out = paddle.zeros(shape=[total_q, num_heads, head_size], dtype=out_dtype)
        else:
            out = paddle.zeros(shape=[batch_size, seqlen, num_heads, head_size], dtype=out_dtype)

    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = paddle.full(shape=[num_heads, total_q], fill_value=float('-inf'), dtype=paddle.float32).transpose(0, 1)
        else:
            lse = paddle.full(
                shape=[batch_size, num_heads, seqlen], fill_value=float('-inf'), dtype=paddle.float32
            ).transpose(1, 2)
    else:
        lse = None

    _flash_attn_fwd_combine(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
    )
    return out, lse

class FlashMaskFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        causal: bool = False,
        softmax_scale: float | None = None,
        learnable_sink: paddle.Tensor | None = None,
        startend_row_indices: paddle.Tensor | None = None,
        block_mask: paddle.Tensor | None = None,
        group=None,
    ) -> paddle.Tensor | Tuple[paddle.Tensor, paddle.Tensor]:
        out, lse = _flash_attn_fwd(
            query,
            key,
            value,
            causal=causal,
            softmax_scale=softmax_scale,
            learnable_sink=learnable_sink,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            pack_gqa=False,
            group=group,
        )
        ctx.save_for_backward(query, key, value, startend_row_indices, out, lse, learnable_sink)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.group = group
        return [out, lse]

    @staticmethod
    def backward(ctx, dout, *args) -> Tuple[paddle.Tensor, ...]:
        query, key, value, startend_row_indices, out, lse, learnable_sink = ctx.saved_tensor()
        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=ctx.causal,
            )
        else:
            flashmask_info = None
        dq, dk, dv, dsink = _flash_attn_bwd(
            query,
            key,
            value,
            out,
            dout,
            lse,
            flashmask_info,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])["FLAGS_cudnn_deterministic"],
            learnable_sink=learnable_sink,
            group=ctx.group,
        )
        if learnable_sink is None:
            return dq, dk, dv
        return dq, dk, dv, dsink

# TODO(wusiming): should we align the parameters with those of paddle.nn.functional.flashmask_attention?
def flashmask_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: paddle.Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: paddle.Tensor | None = None,
    learnable_sink: paddle.Tensor | None = None,
    group=None,
):
    if _is_cutedsl_kernel_supported(query, key, value, startend_row_indices):
        assert dropout == 0.0, (
            "flashmask v4 does not support dropout"
        )
        # TODO(wusiming): support sliding window mask gen when giving a window_size
        assert window_size is None, (
            "flashmask v4 does not support generate sliding window mask automatically"
        )
        assert not return_seed_offset, (
            "flashmask v4 does not support return seed_offset"
        )
        assert fixed_seed_offset is None, (
            "flashmask v4 does not support setting seed_offset"
        )
        assert rng_name == "", (
            "flashmask v4 does not support setting rng_name"
        )
        assert training, (
            "flashmask v4 does not support setting training to False"
        )
        assert name is None, (
            "flashmask v4 does not support setting training name"
        )
        assert block_mask is None, (
            "flashmask v4 does not support block mask"
        )

        if startend_row_indices is not None:
            assert startend_row_indices.dtype == paddle.int32, (
                f"startend_row_indices.dtype must be paddle.int32, but got {startend_row_indices.dtype}"
            )
            assert len(startend_row_indices.shape) == 4, (
                f"startend_row_indices rank must be 4,but got {startend_row_indices.shape}"
            )
            assert startend_row_indices.shape[0] == key.shape[0], (
                f"startend_row_indices.shape[0] must be equal to batch_size, but got {startend_row_indices.shape[0]} and {key.shape[0]}"
            )
            assert startend_row_indices.shape[2] == key.shape[1], (
                f"startend_row_indices.shape[2] must be equal to seqlen_k, but got {startend_row_indices.shape[2]} and {key.shape[1]}"
            )
            assert startend_row_indices.shape[1] in [
                1,
                key.shape[2],
            ], (
                "startend_row_indices head_num must be equal to 1(broadcast) or head_num_k."
            )

            # Note(wusiming): has_end is not necessary, just for better code reasoning about
            if causal:
                if startend_row_indices.shape[-1] == 1:
                    has_end = False
                elif startend_row_indices.shape[-1] == 2:
                    has_end = True
                else:
                    raise ValueError(
                        f"Invalid shape of startend_row_indices, when causal is True, the last dimension should be either 1 or 2 but got {startend_row_indices.shape[-1]}"
                    )
            else:
                if startend_row_indices.shape[-1] == 2:
                    has_end = False
                elif startend_row_indices.shape[-1] == 4:
                    has_end = True
                else:
                    raise ValueError(
                        f"Invalid shape of startend_row_indices, when causal is False, the last dimension should be either 2 or 4 but got {startend_row_indices.shape[-1]}"
                    )

        # Note(wusiming): when softmax_scale is None, it will be set to 1.0 / math.sqrt(head_dim) in _flash_attn_fwd
        out, lse = FlashMaskFunc.apply(
            query,
            key,
            value,
            causal=causal,
            softmax_scale=softmax_scale,
            learnable_sink=learnable_sink,
            startend_row_indices=startend_row_indices,
            group=group,
        )
        if return_softmax_lse:
            return [out, lse]
        else:
            return out
    else:
        assert learnable_sink is None, (
            "learnable_sink is only supported on the flashmask v4 (cute) path"
        )
        original_flash_attn_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]
        if original_flash_attn_version == 4:
            paddle.set_flags({"FLAGS_flash_attn_version": 2})
            assert (
                not causal or (query.shape[1] == key.shape[1])
            ), (
                f"Fallback to flashmask v1 is not supported when using causal mask "
                f"and query/key sequence lengths differ (seqlen_q={query.shape[1]}, seqlen_k={key.shape[1]}). "
                "Please ensure seqlen_q equals seqlen_k or disable causal."
            )
        try:
            outputs = paddle.nn.functional.flashmask_attention(
                query=query,
                key=key,
                value=value,
                startend_row_indices=startend_row_indices,
                dropout=dropout,
                causal=causal,
                window_size=window_size,
                return_softmax_lse=return_softmax_lse,
                return_seed_offset=return_seed_offset,
                fixed_seed_offset=fixed_seed_offset,
                rng_name=rng_name,
                training=training,
                name=name,
                softmax_scale=softmax_scale,
                block_mask=block_mask,
            )
        finally:
            if original_flash_attn_version == 4:
                paddle.set_flags({"FLAGS_flash_attn_version": 4})
        return outputs

# Note(wusiming): do we need to align api to tridao?
def flash_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
    softmax_scale=None,
):
    if _is_cutedsl_kernel_supported(query, key, value):
        assert dropout == 0.0, (
            "flash attention 4 does not support dropout"
        )
        # Note(wusiming): return_softmax means return attn score, not lse
        assert not return_softmax, (
            "flash attention 4 does not support return_softmax"
        )
        assert fixed_seed_offset is None, (
            "flash attention 4 does not support setting seed_offset"
        )
        assert rng_name == "", (
            "flash attention 4 does not support setting rng_name"
        )
        assert training, (
            "flash attention 4 does not support setting training to False"
        )
        assert name is None, (
            "flash attention 4 does not support setting name"
        )

        # Note(wusiming): i dont think it is necessary to add a pylayer for flash_attention, just reuse flashmask
        out, lse = FlashMaskFunc.apply(
            query,
            key,
            value,
            causal=causal,
            softmax_scale=softmax_scale,
            startend_row_indices=None,
        )
        return out, None
    else:
        original_flash_attn_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]
        if original_flash_attn_version == 4:
            paddle.set_flags({"FLAGS_flash_attn_version": 2})
            assert (
                not causal or (query.shape[1] == key.shape[1])
            ), (
                f"Fallback to flash attention version 2 is not supported when using causal mask "
                f"and query/key sequence lengths differ (seqlen_q={query.shape[1]}, seqlen_k={key.shape[1]}). "
                "Please ensure seqlen_q equals seqlen_k or disable causal."
            )
        try:
            out, lse = paddle.nn.functional.flash_attention.flash_attention(
                query=query,
                key=key,
                value=value,
                dropout=dropout,
                causal=causal,
                return_softmax=return_softmax,
                fixed_seed_offset=fixed_seed_offset,
                rng_name=rng_name,
                training=training,
                name=name,
                softmax_scale=softmax_scale,
            )
        finally:
            if original_flash_attn_version == 4:
                paddle.set_flags({"FLAGS_flash_attn_version": 4})
        return out, lse
