"""SM100 backward for head dims larger than 256 (supported and measured: 512/512).

Why a separate file from `flash_bwd_sm100.py`: that kernel hand-unrolls the
low|high halves of its split axes in ~8 places in the load warp, and hardcodes
`tile_m == tile_n == 128`. Two halves is not enough once head_dim exceeds 512
(576/2 = 288 exceeds the UMMA N limit of 256, so that shape needs 3 chunks), so
every one of those sites would have to become a loop, which puts the
already-working d256/d192 configs at risk. Here the chunk count and `cta_group`
are parameters from the start, so the axis splits into as many chunks as the
shape needs.

Resource model (all numbers verified on B200, see the "column rule" below):

  TMEM columns of an accumulator = N * (rows_per_cta / 128), except that a
  cta_group=1 MMA with M=64 uses the 16-datapath interleave: it occupies lanes
  {0-15, 32-47, 64-79, 96-111} and still spans N columns.

    cta_group=1, M=128, N -> ((128,N)):((65536,1))                 -> N   cols
    cta_group=1, M= 64, N -> (((16,4),N)):(((65536,2097152),1))    -> N   cols
    cta_group=2, M=128, N -> ((64,(N/2,2))):((65536,(1,4194304)))  -> N/2 cols
    cta_group=2, M=256, N -> ((128,N)):((65536,1))                 -> N   cols

  (TMEM addresses put the lane in the high bits and the column in the low 16,
  so a stride of 65536 is one lane and a stride of 1 is one column.)

Phase 1 (the only path that runs today): cta_group=1, no swapAB, all three of dV/dK/dQ
are produced per m-iteration and drained by the reduce warps into the fp32 gmem
accumulators. Correct but accum-traffic bound.

Phase 2 (later): cta_group=2 + swapAB on dV/dK so both stay resident in TMEM for
the whole m loop, dropping the accum traffic from once per m-iteration to once
per n-block. The cta_group=2 arithmetic is already threaded through (column rules,
n_block_pair skip derivation, MMA M constraints) but the path is NOT enabled: `_launch`
passes no `cluster=` to .launch() and does not round the grid to the pair size, so a
cta_group=2 kernel would build multicast TMA atoms without a cluster to multicast to.
"""

import functools
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.utils.layout import LayoutEnum
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05

from flash_mask.cute import blackwell_helpers as sm100_utils
from flash_mask.cute import copy_utils, layout_utils, utils
from flash_mask.cute.tile_scheduler import SingleTileScheduler, TileSchedulerArguments


SM100_TMEM_CAPACITY_COLUMNS = 512
# cudaFuncAttributeMaxDynamicSharedMemorySize on SM100.
SM100_SMEM_CAPACITY_BYTES = 227 * 1024
# Slack for barrier storage, LSE / dPsum, the flashmask row indices and the
# per-buffer alignment padding that cute.struct adds on top of the tile bytes.
# Measured on 576/512: the real struct is 3076 B larger than the solver's tile bytes,
# so 6KB is ~2x headroom.
SM100_SMEM_RESERVE_BYTES = 6 * 1024
# UMMA N limit (cutlass/cute/nvgpu/tcgen05/mma.py).
UMMA_MAX_N = 256
# All accumulators that take part in a "T2R then packed-bf16 R2T" round trip must
# have M = 128 so the 32-datapath atoms apply: at M=64 only the 16-datapath atoms
# exist and their per-thread element order does not survive the bf16 packing
# (measured on B200: exact for a constant A operand, ~7% off for A = m or A = n).
UMMA_REQUIRED_M = 128
# Slots used to reduce flashmask bounds before deriving per-m-block skip ranges. Two are
# in use (max of the lower-tail starts, min of the ends); the rest are headroom for the
# 4-bound form, which needs four.
FLASHMASK_META_SLOTS = 8


def tmem_columns(n: int, m_total: int, cta_group: int) -> int:
    """Return per-CTA TMEM columns for an ``m_total x n`` accumulator.

    See the column rule in the module docstring.

    Args:
        n: MMA N mode.
        m_total: MMA M mode across the full CTA group.
        cta_group: Number of CTAs participating in the MMA.

    Returns:
        Number of TMEM columns occupied by each CTA.
    """
    rows_per_cta = m_total // cta_group
    assert rows_per_cta in (64, 128), f"unexpected rows per CTA: {rows_per_cta}"
    if cta_group == 1:
        # M=64 uses the 16-datapath interleave: half the lanes idle, N columns.
        return n
    return n if rows_per_cta == 128 else n // 2


MAX_OUT_SLOTS = 2


def tmem_plan(tile_m: int, tile_n: int, d_chunk: int, dv_chunk: int, cta_group: int):
    """Return the TMEM column plan for one configuration.

    Args:
        tile_m: Query tile height.
        tile_n: Key tile width.
        d_chunk: GEMM chunk width along head_dim.
        dv_chunk: GEMM chunk width along head_dim_v.
        cta_group: Number of CTAs participating in each MMA.

    Returns:
        Dictionary with the S / dP offsets, the output scratch slot width, the number
        of output slots and the total per-CTA column count.

    The single source of truth for the layout: ``solve_config`` uses it to decide
    feasibility and ``__init__`` uses it to place the regions. They used to carry two
    hand-written copies of the formula that disagreed on both terms that matter -- the
    solver charged the dQ accumulator ``tile_m`` columns where the layout charges it
    ``d_chunk``, and it did not account for the output slot count at all, so it could
    hand back a config the constructor then rejected.
    """
    s_cols = tmem_columns(tile_m, cta_group * tile_n, cta_group)
    dp_cols = tmem_columns(tile_m, cta_group * tile_n, cta_group)
    dv_cols = tmem_columns(dv_chunk, cta_group * tile_n, cta_group)
    dk_cols = tmem_columns(d_chunk, cta_group * tile_n, cta_group)
    dq_cols = tmem_columns(d_chunk, tile_m, cta_group)
    out_offset = s_cols + dp_cols
    slot_cols = max(dv_cols, dk_cols, dq_cols)
    num_out_slots = min(
        MAX_OUT_SLOTS, (SM100_TMEM_CAPACITY_COLUMNS - out_offset) // slot_cols
    )
    return dict(
        s_cols=s_cols,
        out_offset=out_offset,
        slot_cols=slot_cols,
        num_out_slots=num_out_slots,
        total=out_offset + max(num_out_slots, 1) * slot_cols,
    )



def accum_slice_candidates(hdim: int, chunk: int, max_slice: int = 192) -> list:
    """Return the legal accumulator slice widths for an axis, narrowest first.

    Args:
        hdim: Padded head dimension represented by the accumulator.
        chunk: GEMM chunk width; a slice must contain whole chunks.
        max_slice: Preferred maximum width accepted by postprocessing.

    Returns:
        Ascending list of legal slice widths, empty if the chunk admits none.

    The fp32 accumulators are blocked as [slice][row block][...] so that each slice
    is byte-identical to a `head_dim = slice` accumulator and can be handed to the
    shared FlashAttentionBackwardPostprocess as-is. That kernel stages a whole
    tile_m x head_dim fp32 tile in SMEM and holds it in registers, hence the cap.

    A slice must be a whole number of gemm chunks (so a chunk never straddles two
    slices) and a multiple of 64 (the postprocess rounds head_dim up to 64 and would
    otherwise read past the slice). A chunk wider than the cap raises it to the chunk
    width -- the existing d256 path already runs that postprocess at head_dim 256.
    """
    max_slice = max(max_slice, chunk)
    step = chunk * 64 // math.gcd(chunk, 64)
    return [w for w in range(step, min(hdim, max_slice) + 1, step) if hdim % w == 0]


def accum_slice_width(hdim: int, chunk: int, max_slice: int = 192) -> int:
    """Return the widest legal accumulator slice width for an axis.

    Args:
        hdim: Padded head dimension represented by the accumulator.
        chunk: GEMM chunk width; a slice must contain whole chunks.
        max_slice: Preferred maximum width accepted by postprocessing.

    Returns:
        Largest legal slice width not exceeding the effective maximum.
    """
    widths = accum_slice_candidates(hdim, chunk, max_slice)
    assert widths, (
        f"no legal accumulator slice for hdim={hdim}, chunk={chunk} "
        f"(nothing that is a whole number of chunks, a multiple of 64, divides "
        f"{hdim} and is <= {max(max_slice, chunk)})"
    )
    return widths[-1]


def _chunk_candidates(hdim: int) -> list:
    """Legal chunk widths for a headdim axis, widest first.

    <= UMMA_MAX_N because the chunk is an MMA N; a multiple of 32 because the
    reduce path tiles the fp32 workspace by dQ_reduce_ncol = 32; and a divisor of
    the (64-padded) headdim so the axis splits evenly.
    """
    return [
        c
        for c in range(min(UMMA_MAX_N, hdim), 0, -32)
        if hdim % c == 0
    ]


def solve_config(
    head_dim: int,
    head_dim_v: int,
    *,
    cta_group: int = 1,
    dtype_bytes: int = 2,
    # tile_m == tile_n == 128 only: the output drain maps one drain thread to one
    # accumulator row and asserts it (a 64-row m tile would need a second mapping).
    # 64 cannot buy occupancy either: sK / sV / sdS all scale with tile_n, which the UMMA
    # M=128 rule pins, so the best case is ~163KB, still 1 CTA/SM.
    tile_m_choices: tuple = (128,),
    smem_budget: int = SM100_SMEM_CAPACITY_BYTES - SM100_SMEM_RESERVE_BYTES,
    reduce_ncol: int = 32,
):
    """Pick a feasible tile and chunk configuration for a head shape.

    Args:
        head_dim: Query/key head dimension before padding.
        head_dim_v: Value head dimension before padding.
        cta_group: Number of CTAs participating in each MMA.
        dtype_bytes: Bytes per input element used by the SMEM model.
        tile_m_choices: Candidate query tile heights.
        smem_budget: Maximum modeled shared-memory bytes per CTA.
        reduce_ncol: Column width of each output T2R / reduce slice.

    Returns:
        Dictionary containing the selected tile, chunk, and resource values.

    Feasibility is the verified resource model:

      TMEM (columns, per CTA)  -- see tmem_plan(), which is what actually decides
        S/P and dP/dS live across the whole m iteration; the three outputs time-share
        a scratch region of num_out_slots x the widest output chunk.

      SMEM (bytes)
        sQ  tile_m * d_chunk      sK  tile_n * d_chunk * d_chunks_resident
        sV  tile_n * dv_chunk     sdO tile_m * dv_chunk
        sdS tile_n * tile_m

      Accumulator slicing -- a chunk width that admits no legal slice (see
      accum_slice_candidates) is rejected here rather than left to blow up in the
      constructor.

    Ranking is MEASURED, not modelled. `flush/work` (accumulator traffic per unit of
    work) used to be the primary key; the kernel turned out to sit at ~20% of HBM
    bandwidth and ~11% of the MMA peak, i.e. bound by latency rather than traffic, so
    that model does not describe the bottleneck. What the 512/512 sweep
    (b=16 s=4096 h=16) actually showed:

      dv_chunk == tile_n is a sweet spot, not "wider is better": dv 128 -> 64 costs
        10%, -> 32 costs 24%, and dv 256 (which forces d_chunk 32) is the worst config
        measured. dv_chunk is the dV gemm's N; away from tile_n it either adds MMA
        rounds or starves the output scratch.
      At equal SMEM, a wider d_chunk wins: d128 beat d64 by 8%.

    Hence the key: |dv_chunk - tile_n|, then d_chunk, then flush/work, then the LOWER
    K residency.

    That last key looks backwards -- more resident K chunks ought to help the S and dQ
    gemms -- but it is what keeps the ranking on measured ground: the sweep that produced
    the numbers above ran at d_chunks_resident=2, and nothing has measured what the
    leftover SMEM is worth as extra residency. That leftover is real (the accumulator
    staging buffer this model used to charge for is gone), so residency is the obvious
    next knob to sweep; until then the solver does not spend it on a guess.
    """
    pad = lambda x: int(math.ceil(x / 64) * 64)
    d, dv = pad(head_dim), pad(head_dim_v)
    # Rule 2: the K-side gemms have M = cta_group * tile_n, and that must be 128.
    tile_n = UMMA_REQUIRED_M // cta_group
    solutions = []
    for tile_m in tile_m_choices:
        for d_chunk in _chunk_candidates(d):
            for dv_chunk in _chunk_candidates(dv):
                num_d_chunks = d // d_chunk
                if not accum_slice_candidates(d, d_chunk):
                    continue
                if not accum_slice_candidates(dv, dv_chunk):
                    continue
                for resident in range(min(2, num_d_chunks), num_d_chunks + 1):
                    plan = tmem_plan(tile_m, tile_n, d_chunk, dv_chunk, cta_group)
                    if plan["num_out_slots"] < 1:
                        continue
                    if plan["total"] > SM100_TMEM_CAPACITY_COLUMNS:
                        continue
                    smem = dtype_bytes * (
                        tile_m * d_chunk                      # sQ (aliases sQt)
                        + tile_n * d_chunk * resident         # sK (aliases sKt)
                        + tile_n * dv_chunk                   # sV
                        + tile_m * dv_chunk                   # sdO (aliases sdOt)
                        + tile_n * tile_m                     # sdS
                    )
                    if smem > smem_budget:
                        continue
                    flush_per_work = (
                        tile_n * (d + dv) + tile_m * d
                    ) / (tile_m * tile_n)
                    solutions.append(
                        dict(
                            tile_m=tile_m,
                            tile_n=tile_n,
                            d_chunk=d_chunk,
                            dv_chunk=dv_chunk,
                            d_chunks_resident=resident,
                            dQ_reduce_ncol=reduce_ncol,
                            cta_group=cta_group,
                            tmem_columns=plan["total"],
                            smem_bytes=smem,
                            flush_per_work=flush_per_work,
                            # Ranking key: measured, not modelled. Prefer dv_chunk ==
                            # tile_n, then the wider d_chunk.
                            _tie=(
                                abs(dv_chunk - tile_n),
                                -d_chunk,
                                flush_per_work,
                                resident,
                            ),
                        )
                    )
    if not solutions:
        raise ValueError(
            f"no feasible config for head_dim={head_dim}, head_dim_v={head_dim_v}, "
            f"cta_group={cta_group}"
        )
    solutions.sort(key=lambda s: s["_tie"])
    return solutions[0]


CONFIG_KEYS = (
    "tile_m", "tile_n", "d_chunk", "dv_chunk", "d_chunks_resident", "dQ_reduce_ncol",
)

# Measured pins, keyed by (head_dim, head_dim_v, cta_group). These win over the solver's
# ranking, which is a *model*: the measured kernel sits at ~20% of HBM bandwidth and ~11%
# of the MMA peak, so it does not currently describe the bottleneck.
#
# 512/512 (B30Z / sm103, b1/s8192/h64/hkv1, causal document mask): 23.75ms against the
# 24.12ms of the config the solver ranked first at the time. What the pin buys is the
# halved slice count in the drain (48 -> 24 per m tile), which is why dQ_reduce_ncol has
# to come with the wider d_chunk -- d_chunk is the tile the drain slices, so a 64-column
# slice is only legal once the chunk is at least 64 wide.
_MEASURED_CONFIG = {
    (512, 512, 1): {
        "d_chunk": 128,
        "d_chunks_resident": 2,
        "dQ_reduce_ncol": 64,
    },
}


class FlashAttentionBackwardSm100BigD:
    """SM100 backward kernel specialized for head dimensions larger than 256."""

    @classmethod
    def from_shape(cls, head_dim: int, head_dim_v: int, cta_group: int = 1, **kwargs):
        """Build a kernel with the selected configuration for a head shape.

        Args:
            head_dim: Query/key head dimension.
            head_dim_v: Value head dimension.
            cta_group: Number of CTAs participating in each MMA.
            **kwargs: Explicit constructor overrides for selected configuration values.

        Returns:
            Configured ``FlashAttentionBackwardSm100BigD`` instance.
        """
        cfg = solve_config(head_dim, head_dim_v, cta_group=cta_group)
        pin = _MEASURED_CONFIG.get((head_dim, head_dim_v, cta_group))
        if pin is not None:
            cfg = {**cfg, **pin}
        for key in CONFIG_KEYS:
            kwargs.setdefault(key, cfg[key])
        kwargs.setdefault("cta_group", cta_group)
        return cls(head_dim, head_dim_v, **kwargs)

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        is_causal: bool = False,
        qhead_per_kvhead: int = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        d_chunk: int = 96,
        dv_chunk: int = 128,
        d_chunks_resident: int = 3,
        dQ_reduce_ncol: int = 32,
        cta_group: int = 1,
        swap_dKV: bool = False,
        deterministic: bool = False,
        fm_bound_num: int = 0,
    ):
        # head_dim is padded to a multiple of 64 to match head_dim_rounded in the
        # interface. Out-of-range columns need no predication here: the TMA zero-fills
        # them and the per-element mask below kills their contribution.
        hdim_multiple_of = 64
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.d_chunk = d_chunk
        self.dv_chunk = dv_chunk
        self.d_chunks_resident = d_chunks_resident
        # Width of one output T2R / reduce slice. Narrower slices mean more, smaller
        # vector atomics; 32 is the default and 64 is what 512/512 measured best at. Any
        # multiple of 4 keeps the byte layout the postprocess expects (its run is ordered
        # [4-column group][row][4 columns], so a slice of any multiple of 4 columns is
        # still a contiguous range).
        assert dQ_reduce_ncol % 4 == 0 and d_chunk % dQ_reduce_ncol == 0
        assert dv_chunk % dQ_reduce_ncol == 0
        self.dQ_reduce_ncol_cfg = dQ_reduce_ncol
        self.cta_group_size = cta_group
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        # Number of columns of startend_row_indices (0 = no flashmask). It has to be a
        # constructor arg: the tensor arrives with a fully dynamic layout, so the kernel
        # cannot read it off the shape at trace time.
        #
        # The legal widths depend on is_causal, and the dependency is load bearing in
        # two places, so an out-of-set width is a silently wrong answer rather than a
        # missed optimization:
        #   - the per-element mask derives `has_end` as (causal and 2) or (not causal
        #     and 4); a non-causal width of 1 takes the has_end=False path, which reads
        #     fm_row[1] -- past the end of a single-column tensor.
        #   - the m-block skip only reduces an `end` bound when the width is >= 2, so a
        #     non-causal width of 1 leaves min_end at INT32_MAX, which pushes m_lo past
        #     m_hi and makes num_iters 0: the whole n block computes nothing and dK / dV
        #     / dQ stay at their zero-initialised accumulator values.
        legal_bound_num = (0, 1, 2) if is_causal else (0, 2, 4)
        assert fm_bound_num in legal_bound_num, (
            f"startend_row_indices width {fm_bound_num} is not valid for "
            f"is_causal={is_causal} (legal widths: {legal_bound_num})"
        )
        self.fm_bound_num = fm_bound_num

        assert not deterministic, (
            "deterministic reduction is not supported by the big-headdim bwd: dK / dV / "
            "dQ are accumulated with red.global.add, whose order is not reproducible"
        )
        # cta_group=2 is NOT ENABLED and has never run: _launch builds multicast TMA
        # atoms from cluster_shape_mnk but passes no `cluster=` to .launch(), and the grid
        # is not rounded to a multiple of the pair size. The column rules in tmem_plan(),
        # the n_block_pair skip derivation and the tile_m constraint below are all in
        # place for it, but turning it on means fixing the launch and the grid first.
        assert cta_group in (1, 2)

        assert not swap_dKV, "swapAB (phase 2) is not implemented yet"
        # UMMA N limit. This is what rules out the d=576 -> 2x288 split that the
        # d=256 config uses.
        assert d_chunk <= 256 and dv_chunk <= 256, "chunk width exceeds the UMMA N limit"
        # The dQ / dK reduce paths tile the flat fp32 workspace by dQ_reduce_ncol
        # (32) and dK_reduce_ncol, so the chunk width must be a multiple of 32.
        assert d_chunk % 32 == 0 and dv_chunk % 32 == 0
        assert self.tile_hdim % d_chunk == 0, (
            f"head_dim {self.tile_hdim} must be a whole number of {d_chunk}-wide chunks"
        )
        assert self.tile_hdimv % dv_chunk == 0, (
            f"head_dim_v {self.tile_hdimv} must be a whole number of {dv_chunk}-wide chunks"
        )
        self.num_d_chunks = self.tile_hdim // d_chunk
        self.num_dv_chunks = self.tile_hdimv // dv_chunk
        assert 1 <= d_chunks_resident <= self.num_d_chunks
        # head_dim slicing of the fp32 accumulators, so the shared postprocess can be
        # called once per slice. Public: interface.py reads these to drive the
        # postprocess loop.
        self.accum_slice_d = accum_slice_width(self.tile_hdim, d_chunk)
        self.accum_slice_dv = accum_slice_width(self.tile_hdimv, dv_chunk)

        # ---------------------------------------------------------------- MMA tilers
        cg = self.cta_group_size
        # S^T = K @ Q^T          (contraction over d, accumulated over d chunks)
        self.mma_tiler_kq = (cg * tile_n, tile_m, d_chunk)
        # dP^T = V @ dO^T        (contraction over dv, accumulated over dv chunks)
        self.mma_tiler_vdo = (cg * tile_n, tile_m, dv_chunk)
        # dV_c = P^T @ dO_c      (output split over dv)
        self.mma_tiler_pdo = (cg * tile_n, dv_chunk, tile_m)
        # dK_c = dS^T @ Q_c      (output split over d)
        self.mma_tiler_dsq = (cg * tile_n, d_chunk, tile_m)
        # dQ_c = dS @ K_c        (output split over d)
        self.mma_tiler_dsk = (tile_m, d_chunk, tile_n * cg)
        if cg == 2:
            # A cta_group=2 MMA needs M in {128, 256}; dQ's M is tile_m.
            assert tile_m >= 128, "cta_group=2 requires tile_m >= 128 for the dQ gemm"

        self.acc_dtype = Float32
        self.cluster_shape_mn = (cg, 1)
        self.cta_group = tcgen05.CtaGroup.TWO if cg == 2 else tcgen05.CtaGroup.ONE

        # ---------------------------------------------------------------- TMEM layout
        # S/P and dP/dS live across the whole m-iteration (P is the A operand of
        # the dV gemm, dS of the dK/dQ gemms). The three outputs are produced and
        # immediately drained, so they time-share one scratch region whose width
        # is the widest of them.
        #
        # Output scratch slots. All three outputs share the same columns -- they never
        # overlap in time -- but with a single slot the mma warp cannot issue chunk
        # c+1's gemm until the drain warps have emptied chunk c, so the ~20 output
        # gemms per m tile are fully serialised against their own drain.
        #
        # A second slot breaks that: chunk c+1 goes to the other slot, so its gemm
        # overlaps chunk c's T2R + reduce. This is what DSA does for its dKV
        # (dsa_bwd_sm100.py:100-105 gives dKV two slots and aliases dKV2/3 onto
        # them) -- note it does NOT keep dKV resident either, it double buffers.
        #
        # True residency for dK/dV is a different thing and does not fit here: at
        # tile_n=128 one resident dV is tile_n * head_dim_v * 4B = 256KB, which is
        # the entire 512-column TMEM (512 cols x 128 lanes x 4B). Both dK and dV
        # resident needs 512KB, i.e. cta_group=2 (two CTAs' TMEM, and each
        # accumulator split in half along N). No column trick substitutes for that.
        #
        # The arithmetic lives in tmem_plan() so that solve_config's feasibility test
        # and this layout cannot drift apart.
        plan = tmem_plan(tile_m, tile_n, d_chunk, dv_chunk, cg)
        self.tmem_S_cols = plan["s_cols"]

        self.tmem_S_offset = 0
        # P and dS are bf16, so they need only half the columns of the f32
        # accumulator they overlay, and they sit in its UPPER half: S (resp. dP)
        # must be fully read into registers before P (resp. dS) clobbers it.
        self.tmem_s_to_p_offset = tile_m // 2
        self.tmem_P_offset = self.tmem_S_offset + self.tmem_s_to_p_offset
        self.tmem_dP_offset = self.tmem_S_offset + self.tmem_S_cols
        self.tmem_dS_offset = self.tmem_dP_offset + self.tmem_s_to_p_offset
        self.tmem_out_offset = plan["out_offset"]
        self.tmem_out_slot_cols = plan["slot_cols"]
        self.num_out_slots = plan["num_out_slots"]
        assert self.num_out_slots >= 1, (
            f"no room for an output slot: out_offset={self.tmem_out_offset}, "
            f"slot={self.tmem_out_slot_cols}"
        )
        # Chunk c uses slot c % num_out_slots, starting at tmem_out_offset plus
        # (c % num_out_slots) * tmem_out_slot_cols.
        self.tmem_total = plan["total"]
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS, (
            f"TMEM overflow: {self.tmem_total} > {SM100_TMEM_CAPACITY_COLUMNS} columns"
        )
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        # ------------------------------------------------------------ warps / barriers
        # The output drain (T2R of dV/dK/dQ -> SMEM staging -> cp.reduce into the
        # fp32 gmem accumulators) runs on its OWN warpgroup, not on the compute
        # warps. It used to sit at the tail of the compute warps' m iteration, which
        # serialised ~48 16KB slices against the next iteration's softmax / dS math
        # and kept three f32 output fragments (dV 128 + dK 64 + dQ 64 registers per
        # thread) live inside the compute warps' budget. DSA's sm100 bwd splits the
        # same way (4 compute + 8 reduce warps).
        self.drain_warp_ids = (0, 1, 2, 3)
        # The TMEM->register copies are partitioned over 128 threads (one
        # warpgroup): tcgen05.make_tmem_copy sizes its thread layout from the
        # accumulator, and for these shapes that is 4 warps. Dispatching 8 compute
        # warps at it made threads 128..255 read/write past their fragment. Warps 8-11
        # stay idle until they get a tile of their own (the second warpgroup).
        self.compute_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.threads_per_cta = cute.arch.WARP_SIZE * 16

        # setmaxnreg budget. The launch is 512 threads, so every warp starts at
        # 65536/512 = 128 registers; a warpgroup asking for less must *decrease* and
        # for more must *increase*, and the per-thread values summed over the four
        # warpgroups must stay within 512.
        #   drain 200 (warps 0-3)   compute 136 (4-7)   idle 24 (8-11)
        #   load / mma 88 (12-15)
        # The drain warpgroup got the registers the output fragments need now that
        # the drain moved there (dV 128 + dK 64 + dQ 64 f32 per thread, processed one
        # chunk at a time); compute keeps its 136 since its four S/P/dP/dS fragments
        # did not change. Warps 8-11 are idle at runtime (num_regs_empty), so the budget
        # below is what the kernel actually allocates -- reserve a second compute
        # warpgroup here once they get a real tile.
        self.num_regs_drain = 200
        self.num_regs_compute = 136
        self.num_regs_load = 88
        self.num_regs_mma = 88
        self.num_regs_empty = 24
        assert (
            self.num_regs_drain
            + self.num_regs_compute
            + self.num_regs_empty
            + max(self.num_regs_load, self.num_regs_mma)
            <= 512
        )

        self.buffer_align_bytes = 1024
        # Width, in m columns, of one S -> P -> dP -> dS round trip. The live register
        # set of that round trip is 5 * softmax_chunk_m f32 per thread (S, P, dP, dS
        # and the two packed R2T buffers, which are half-width), so this is what keeps
        # the compute warps out of local memory: at the full tile_m = 128 it was 640
        # f32 against 128 registers per thread, and ncu measured 65GB of local traffic
        # with 84% of the stall budget on L1TEX. 32 keeps it at 160.
        #
        # Must divide tile_m and be a multiple of 32: the packed bf16 P / dS region is
        # addressed in W // 32 * 16 f32-equivalent columns, and the R2T store rep is
        # W // 8.
        self.softmax_chunk_m = 32 if tile_m % 32 == 0 else tile_m
        assert self.tile_m % self.softmax_chunk_m == 0
        assert self.softmax_chunk_m % 32 == 0
        self.num_softmax_chunks = self.tile_m // self.softmax_chunk_m

    def _setup_attributes(self):
        # Phase 1 keeps the pipelines at their shallowest: Q and dO chunks are
        # re-fetched per use, K holds `d_chunks_resident` chunks for the whole
        # n-block. Deeper staging is a later perf knob.
        self.Q_stage = 1
        self.dO_stage = 1
        self.K_smem_stages = self.d_chunks_resident

        # Output reduce granularity: the T2R slice width, which also fixes the byte
        # layout of the fp32 accumulators and therefore has to stay something
        # FlashAttentionBackwardPostprocess can read back.
        self.dQ_reduce_ncol = self.dQ_reduce_ncol_cfg
        assert (self.d_chunk // self.cta_group_size) % self.dQ_reduce_ncol == 0

        # mbarrier slot offsets inside the single mbar_ptr MemRange: the pipeline slots
        # occupy [0, mbar_count() - 1) and the one scalar barrier sits at the end.
        self.mbar_tmem_dealloc_offset = self.mbar_count() - 1

    def _get_tiled_mma(self, ab_dtype):
        cg = self.cta_group
        mk = tcgen05.OperandMajorMode.K
        mn = tcgen05.OperandMajorMode.MN
        tmem_src = tcgen05.OperandSource.TMEM
        mma = lambda tiler, b_major, a_src=None: sm100_utils_basic.make_trivial_tiled_mma(
            ab_dtype, mk, b_major, self.acc_dtype, cg, tiler[:2], *( (a_src,) if a_src else () )
        )
        tiled_mma_S = mma(self.mma_tiler_kq, mk)
        tiled_mma_dP = mma(self.mma_tiler_vdo, mk)
        # A operands of the dV / dK gemms come from TMEM (P and dS), which is the
        # path flash_fwd_sm100 / flash_bwd_sm100 use. It requires the K-side
        # accumulators to be M=128 so that the 32-datapath T2R/R2T atoms apply:
        # with M=64 the 16-datapath atoms are forced, and their per-thread element
        # order does not survive the bf16 packing (measured: exact for a constant A
        # operand, ~7% off for A = m or A = n, i.e. a local permutation). Hence
        # the K-side MMA M mode is fixed at 128; tile_m independently remains 128
        # for the accumulator staging and dQ mapping used by this kernel.
        tiled_mma_dV = mma(self.mma_tiler_pdo, mn, tmem_src)
        tiled_mma_dK = mma(self.mma_tiler_dsq, mn, tmem_src)
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            ab_dtype, mn, mn, self.acc_dtype, cg, self.mma_tiler_dsk[:2]
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self, ab_dtype):
        mma_S, mma_dP, mma_dK, mma_dV, mma_dQ = self._get_tiled_mma(ab_dtype)
        la = sm100_utils_basic.make_smem_layout_a
        lb = sm100_utils_basic.make_smem_layout_b

        # S^T = K @ Q^T : all d chunks of K stay resident for the n-block, Q is
        # streamed one chunk at a time.
        self.sK_layout = la(mma_S, self.mma_tiler_kq, ab_dtype, self.K_smem_stages)
        self.sQ_layout = lb(mma_S, self.mma_tiler_kq, ab_dtype, self.Q_stage)
        # dP^T = V @ dO^T
        self.sV_layout = cute.slice_(
            la(mma_dP, self.mma_tiler_vdo, ab_dtype, 1), (None, None, None, 0)
        )
        self.sdOt_layout = lb(mma_dP, self.mma_tiler_vdo, ab_dtype, self.dO_stage)
        # dV_c = P^T @ dO_c   (P comes from TMEM)
        self.tP_layout = cute.slice_(
            la(mma_dV, self.mma_tiler_pdo, ab_dtype, 1), (None, None, None, 0)
        )
        self.sdO_layout = lb(mma_dV, self.mma_tiler_pdo, ab_dtype, self.dO_stage)
        # dK_c = dS^T @ Q_c   (dS comes from TMEM)
        self.tdS_layout = cute.slice_(
            la(mma_dK, self.mma_tiler_dsq, ab_dtype, 1), (None, None, None, 0)
        )
        # The SMEM staging of dS must come from a SMEM-sourced mma: make_smem_layout_a
        # on the TMEM-sourced tiled_mma_dK describes the TMEM operand (two bf16 per
        # 32-bit lane), so its element count is 2x and partition_D handed each thread
        # 256 elements against the 128 the T2R produced.
        mma_dK_smem = sm100_utils_basic.make_trivial_tiled_mma(
            ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
        )
        self.sdSt_layout = cute.slice_(
            la(mma_dK_smem, self.mma_tiler_dsq, ab_dtype, 1), (None, None, None, 0)
        )
        self.sQt_layout = lb(mma_dK, self.mma_tiler_dsq, ab_dtype, self.Q_stage)
        # dQ_c = dS @ K_c
        self.sdS_layout = cute.slice_(
            la(mma_dQ, self.mma_tiler_dsk, ab_dtype, 1), (None, None, None, 0)
        )
        self.sKt_layout = lb(mma_dQ, self.mma_tiler_dsk, ab_dtype, self.K_smem_stages)

        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )

    def mbar_count(self):
        """Number of Int64 mbarrier slots in SharedStorage.

        Exactly the layout the kernel builds (see the mbar_* offsets there), because a
        fixed fudge factor silently capped the config space: with 8 d chunks and 8 dv
        chunks the slots overflowed the MemRange and the config became unusable.

          Sin  full/empty    2 * num_d_chunks       S full          1
          dPin full/empty    2 * num_dv_chunks      dP full         1
          dVin full/empty    2 * num_dv_chunks      PdS full        1
          dKin full/empty    2 * num_d_chunks       dSsmem full     1
          dQin full/empty    2 * num_d_chunks       tmem dealloc    1
          out  full/empty    2 * (num_dv_chunks + 2 * num_d_chunks)

        tmem dealloc is last, so it is the slot mbar_tmem_dealloc_offset points at.
        """
        per_chunk = (
            2 * self.num_d_chunks       # Sin
            + 2 * self.num_dv_chunks    # dPin
            + 2 * self.num_dv_chunks    # dVin
            + 2 * self.num_d_chunks     # dKin
            + 2 * self.num_d_chunks     # dQin
            + 2 * (self.num_dv_chunks + 2 * self.num_d_chunks)  # out
        )
        scalars = 5  # S full, dP full, PdS full, dSsmem full, tmem dealloc
        return per_chunk + scalars

    def make_shared_storage(self, q_dtype, do_dtype, ds_dtype):
        """Build the shared-memory storage type for the selected configuration.

        Args:
            q_dtype: Element type used by Q, K, and V staging buffers.
            do_dtype: Element type used by dO staging buffers.
            ds_dtype: Element type used by dS staging buffers.

        Returns:
            Cute struct type describing the kernel's shared-memory allocation.

        Aliasing, mirroring the 1CTA layout of flash_bwd_sm100.py:
          - sQ also backs sQt (transposed view)
          - sdO also backs sdOt
          - sK also backs sKt

        The outputs have no SMEM buffer of their own: dV / dK / dQ leave TMEM through
        vector atomics straight out of the drain warps' registers.
        """
        sQ_alloc_bytes = max(
            cute.size_in_bytes(q_dtype, self.sQ_layout),
            cute.size_in_bytes(q_dtype, self.sQt_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(do_dtype, self.sdO_layout),
            cute.size_in_bytes(do_dtype, self.sdOt_layout),
        )
        mbar_count = self.mbar_count()
        align = self.buffer_align_bytes

        @cute.struct
        class SharedStorage:
            """Shared-memory buffers and barriers used by one BigD CTA."""

            mbar_ptr: cute.struct.MemRange[cutlass.Int64, mbar_count]
            tmem_holding_buf: cutlass.Int32
            sFM_max_min_ptr: cute.struct.MemRange[
                cutlass.Int32, FLASHMASK_META_SLOTS
            ]

            sQ: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes], align]
            sK: cute.struct.Align[
                cute.struct.MemRange[q_dtype, cute.cosize(self.sK_layout)], align
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[q_dtype, cute.cosize(self.sV_layout)], align
            ]
            sdO: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes], align]
            sdS: cute.struct.Align[
                cute.struct.MemRange[ds_dtype, cute.cosize(self.sdSt_layout)], 128
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sLSE_layout)], 128
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sdPsum_layout)], 128
            ]

        return SharedStorage

    # `__call__` matches FlashAttentionBackwardSm100's signature so interface.py can pick
    # between the two by shape alone.

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left=None,
        window_size_right=None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        blocksparse_tensors=None,
        flashmask_info=None,
        overlap_k_addr=None,
        overlap_v_addr=None,
        overlap_work_done_addr=None,
        overlap_segment_idx=None,
        overlap_dk_addr=None,
        overlap_dv_addr=None,
        overlap_b=None,
        overlap_s=None,
        overlap_h=None,
        overlap_d=None,
        overlap_comm_rpb: cutlass.Constexpr = None,
        overlap_bhsd_layout: cutlass.Constexpr = False,
        # Always keep stream as the last parameter.
        stream=None,
    ):
        # None of the features below are implemented; fail loudly rather than silently
        # wrong.

        assert all(x is None for x in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)), (
            "varlen is not supported by the big-headdim bwd"
        )
        assert window_size_left is None and window_size_right is None, (
            "local attention is not supported by the big-headdim bwd yet"
        )
        assert aux_tensors is None and blocksparse_tensors is None, (
            "aux tensors / block sparsity are not supported by the big-headdim bwd"
        )
        assert (flashmask_info is None) == (self.fm_bound_num == 0), (
            "fm_bound_num must match whether flashmask_info was passed"
        )
        if cutlass.const_expr(flashmask_info is not None):
            assert flashmask_info.is_causal == self.is_causal, (
                "flashmask_info.is_causal disagrees with the kernel's is_causal"
            )
        assert mdQ_semaphore is None and mdK_semaphore is None and mdV_semaphore is None, (
            "deterministic reduction is not supported by the big-headdim bwd yet"
        )
        assert overlap_k_addr is None and overlap_dk_addr is None, (
            "the FM-4 overlap path is not supported by the big-headdim bwd"
        )
        # mdK / mdV are the fp32 accumulators here (need_kv_accum is forced on for
        # this kernel: dK and dV always leave TMEM through gmem).
        self._launch(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            softmax_scale * math.log2(math.e),
            stream,
            mFM=(
                None
                if cutlass.const_expr(flashmask_info is None)
                else flashmask_info.startend_row_indices
            ),
        )

    def _launch(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdKaccum: cute.Tensor,
        mdVaccum: cute.Tensor,
        softmax_scale_log2: Float32,
        stream,
        mFM: Optional[cute.Tensor] = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.ds_dtype = mQ.element_type

        self._setup_attributes()
        self._setup_smem_layout(self.q_dtype)
        (
            tiled_mma_S,
            tiled_mma_dP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
        ) = self._get_tiled_mma(self.q_dtype)
        self.cluster_shape_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.shared_storage = self.make_shared_storage(
            self.q_dtype, self.do_dtype, self.ds_dtype
        )
        # solve_config's SMEM model is an estimate (it does not know about the barriers,
        # LSE / dPsum or cute.struct's alignment padding) and _MEASURED_CONFIG can pin a
        # config the estimate never approved, so check the real struct against the
        # hardware cap here rather than finding out as a launch failure.
        smem_bytes = self.shared_storage.size_in_bytes()
        assert smem_bytes <= SM100_SMEM_CAPACITY_BYTES, (
            f"shared storage is {smem_bytes} B, over the {SM100_SMEM_CAPACITY_BYTES} B "
            f"SM100 cap (tile_m={self.tile_m}, tile_n={self.tile_n}, "
            f"d_chunk={self.d_chunk}, dv_chunk={self.dv_chunk}, "
            f"d_chunks_resident={self.d_chunks_resident})"
        )

        layout_transpose = [1, 3, 2, 0]  # (b, s, h, d) -> (s, d, h, b)
        mQ, mK, mV, mdO = [
            layout_utils.select(t, mode=layout_transpose) for t in (mQ, mK, mV, mdO)
        ]
        # The dV / dK gemms take dO / Q as MN-major B operands, i.e. their smem
        # tile is (chunk, tile_m) -- the transpose of the (s, d) gmem order used by
        # the dP / S gemms. They therefore need their own (d, s, h, b) views (this
        # is why the existing kernel carries both mQ/mQt and mdO/mdOt). With
        # dv_chunk == tile_m the shapes coincide, so getting this wrong compiles
        # and silently reads the transpose.
        mdO_dV, mQ_dK, mK_dQ = [
            layout_utils.select(t, mode=[1, 0, 2, 3]) for t in (mdO, mQ, mK)
        ]

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            tiled_mma_S,
            self.cluster_shape_mnk.shape,
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            tiled_mma_S,
            self.cluster_shape_mnk.shape,
        )
        # dP^T = V @ dO^T: V is the A operand, dO^T the B operand, both chunked
        # over head_dim_v.
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            tiled_mma_dP,
            self.cluster_shape_mnk.shape,
        )
        tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mdO,
            cute.select(self.sdOt_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            tiled_mma_dP,
            self.cluster_shape_mnk.shape,
        )
        # dV_c = P^T @ dO_c and dK_c = dS^T @ Q_c need dO / Q in the other
        # orientation, so they get their own atoms (and are re-TMA'd into the same
        # buffers once dP / S are done with them).
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mdO_dV,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            tiled_mma_dV,
            self.cluster_shape_mnk.shape,
        )
        tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_dK,
            cute.select(self.sQt_layout, mode=[0, 1, 2]),
            self.mma_tiler_dsq,
            tiled_mma_dK,
            self.cluster_shape_mnk.shape,
        )
        # dQ_c = dS^T @ K_c needs K in the (d, s) orientation as its B operand.
        tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK_dQ,
            cute.select(self.sKt_layout, mode=[0, 1, 2]),
            self.mma_tiler_dsk,
            tiled_mma_dQ,
            self.cluster_shape_mnk.shape,
        )
        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(dtype, cute.select(layout, mode=[0, 1, 2]))
            for name, dtype, layout in [
                ("Q", self.q_dtype, self.sQ_layout),
                ("K", self.k_dtype, self.sK_layout),
                ("V", self.v_dtype, self.sV_layout),
                ("dOt", self.do_dtype, self.sdOt_layout),
                ("dO", self.do_dtype, self.sdO_layout),
                ("Qt", self.q_dtype, self.sQt_layout),
                ("Kt", self.k_dtype, self.sKt_layout),
            ]
        }

        num_n_block = cute.ceil_div(cute.size(mK.shape[0]), self.tile_n)
        grid_dim = (num_n_block, cute.size(mQ.shape[2]), cute.size(mK.shape[3]))
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dOt,
            tma_tensor_dO,
            tma_tensor_Qt,
            tma_tensor_Kt,
            mLSE,
            mdPsum,
            mFM,
            mdQaccum,
            mdKaccum,
            mdVaccum,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dOt,
            tma_atom_dO,
            tma_atom_Qt,
            tma_atom_Kt,
            tiled_mma_S,
            tiled_mma_dP,
            tiled_mma_dV,
            tiled_mma_dK,
            tiled_mma_dQ,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdOt_layout,
            self.sdO_layout,
            self.sQt_layout,
            self.tP_layout,
            self.tdS_layout,
            self.sdS_layout,
            self.sdSt_layout,
            self.sKt_layout,
            softmax_scale_log2,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdO_dV: cute.Tensor,
        mQ_dK: cute.Tensor,
        mK_dQ: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mFM: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        mdKaccum: cute.Tensor,
        mdVaccum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dOt: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: cute.CopyAtom,
        tma_atom_Kt: cute.CopyAtom,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        softmax_scale_log2: Float32,
    ):
        """Run the tiled BigD backward dataflow for one scheduled KV block.

        Tensor arguments provide global inputs and fp32 gradient accumulators; TMA
        atoms, MMA descriptors, and layouts define the compile-time transfer and
        compute mappings. The kernel writes dQ, dK, and dV through the accumulator
        tensors and has no Python return value.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        n_block, head_idx, batch_idx = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mbar_ptr = storage.mbar_ptr.data_ptr()
        # Every barrier below is used exactly ONCE per m iteration, which is what lets a
        # single phase bit per warp track all of them (see the m loops: iteration j waits
        # on parity j & 1).
        # K and Q for one d chunk arrive on the same barrier (same producer, same
        # consuming gemm), so one full/empty pair per d chunk.
        mbar_Sin_full = mbar_ptr + 0
        mbar_Sin_empty = mbar_Sin_full + self.num_d_chunks
        mbar_S_full = mbar_Sin_empty + self.num_d_chunks
        # V and dO^T for one dv chunk arrive on the same barrier (same producer,
        # consumed together by the dP gemm), so one full/empty pair per dv chunk.
        mbar_dPin_full = mbar_S_full + 1
        mbar_dPin_empty = mbar_dPin_full + self.num_dv_chunks
        mbar_dP_full = mbar_dPin_empty + self.num_dv_chunks
        # P/dS written back to TMEM by the compute warps (bf16, upper half of the
        # S/dP regions), then the three output gemms. Each edge gets its own
        # barrier so that the one-arrival-per-iteration rule above holds.
        mbar_PdS_full = mbar_dP_full + 1
        mbar_dVin_full = mbar_PdS_full + 1
        mbar_dVin_empty = mbar_dVin_full + self.num_dv_chunks
        mbar_dKin_full = mbar_dVin_empty + self.num_dv_chunks
        mbar_dKin_empty = mbar_dKin_full + self.num_d_chunks
        # dS also has to reach SMEM: the dQ gemm needs it in the (m, n) orientation,
        # which TMEM cannot provide (its A operand is the (n, m) accumulator). One
        # buffer, two views -- write through sdSt (n, m), read through sdS (m, n).
        mbar_dSsmem_full = mbar_dKin_empty + self.num_d_chunks
        mbar_dQin_full = mbar_dSsmem_full + 1
        mbar_dQin_empty = mbar_dQin_full + self.num_d_chunks
        mbar_out_full = mbar_dQin_empty + self.num_d_chunks
        num_out_chunks = self.num_dv_chunks + 2 * self.num_d_chunks
        mbar_out_empty = mbar_out_full + num_out_chunks
        mbar_tmem_dealloc = mbar_ptr + self.mbar_tmem_dealloc_offset
        # The pipeline slots above have to end exactly where tmem_dealloc starts, or
        # mbar_count() and this layout have drifted apart and some barrier is aliasing
        # another.
        pipeline_slots = (
            2 * self.num_d_chunks       # Sin full / empty
            + 1                         # S full
            + 2 * self.num_dv_chunks    # dPin full / empty
            + 1                         # dP full
            + 1                         # PdS full
            + 2 * self.num_dv_chunks    # dVin full / empty
            + 2 * self.num_d_chunks     # dKin full / empty
            + 1                         # dSsmem full
            + 2 * self.num_d_chunks     # dQin full / empty
            + 2 * num_out_chunks        # out full / empty
        )
        assert pipeline_slots == self.mbar_tmem_dealloc_offset, (
            f"kernel uses {pipeline_slots} pipeline barrier slots but mbar_count() "
            f"reserved {self.mbar_tmem_dealloc_offset}"
        )

        if warp_idx == 1:
            # *_in_empty and out_empty are arrived by the DRAIN warps (they own the
            # T2R and the SMEM operand release); PdS_full / dSsmem_full stay with the
            # compute warps; tmem_dealloc needs both, since compute reads S / dP out
            # of TMEM and the drain reads the output scratch.
            num_drain_threads = cutlass.const_expr(
                cute.arch.WARP_SIZE * len(self.drain_warp_ids)
            )
            num_compute_threads_init = cutlass.const_expr(
                cute.arch.WARP_SIZE * len(self.compute_warp_ids)
            )
            for c in cutlass.range_constexpr(self.num_d_chunks):
                cute.arch.mbarrier_init(mbar_Sin_full + c, 1)
                cute.arch.mbarrier_init(mbar_Sin_empty + c, 1)
                cute.arch.mbarrier_init(mbar_dKin_full + c, 1)
                cute.arch.mbarrier_init(mbar_dKin_empty + c, num_drain_threads)
            cute.arch.mbarrier_init(mbar_S_full, 1)
            for c in cutlass.range_constexpr(self.num_dv_chunks):
                cute.arch.mbarrier_init(mbar_dPin_full + c, 1)
                cute.arch.mbarrier_init(mbar_dPin_empty + c, 1)
                cute.arch.mbarrier_init(mbar_dVin_full + c, 1)
                cute.arch.mbarrier_init(mbar_dVin_empty + c, num_drain_threads)
            cute.arch.mbarrier_init(mbar_dP_full, 1)
            for c in cutlass.range_constexpr(self.num_d_chunks):
                cute.arch.mbarrier_init(mbar_dQin_full + c, 1)
                cute.arch.mbarrier_init(mbar_dQin_empty + c, num_drain_threads)
            cute.arch.mbarrier_init(mbar_dSsmem_full, num_compute_threads_init)
            # every compute thread arrives on PdS_full; every drain thread on out_empty
            cute.arch.mbarrier_init(mbar_PdS_full, num_compute_threads_init)
            for c in cutlass.range_constexpr(num_out_chunks):
                cute.arch.mbarrier_init(mbar_out_full + c, 1)
                cute.arch.mbarrier_init(mbar_out_empty + c, num_drain_threads)
            cute.arch.mbarrier_init(
                mbar_tmem_dealloc, num_compute_threads_init + num_drain_threads
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdOt = storage.sdO.get_tensor(
            sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype
        )
        # Second views of the same storage, used after their first consumer is
        # done: dV needs dO in the (dv_chunk, tile_m) orientation and dK needs Q^T,
        # so both are re-TMA'd into the buffer they already occupy (the existing
        # kernel does the same, see its "dO_low reload (for dV_low)").
        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        sQt = storage.sQ.get_tensor(
            sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype
        )

        # TMEM: S^T at column 0, dP^T right after it. We always allocate all 512
        # columns starting at 0, so a fake pointer at 0 addresses them (fwd trick).
        thr_mma_S = tiled_mma_S.get_slice(0)
        thr_mma_dP = tiled_mma_dP.get_slice(0)
        tStS_frag = thr_mma_S.make_fragment_C(
            thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])
        )
        tdPtdP_frag = thr_mma_dP.make_fragment_C(
            thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        )
        tmem_ptr = cute.make_ptr(
            Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS_frag.layout)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP_frag.layout)
        # The bf16 P / dS views of the upper halves of these two regions are built per
        # softmax chunk in the compute branch (tStP_c / tStdS_c); the MMA reaches them
        # through tA_addr, so there is no full-tile view here.
        # dS in SMEM: sdSt is the natural (n, m) orientation the compute warps
        # produce, sdS the (m, n) view the dQ gemm needs as its A operand. Same
        # bytes -- this is the dual-view trick from flash_bwd_sm100.py:1453.
        sdSt = storage.sdS.get_tensor(
            sdSt_layout.outer, swizzle=sdSt_layout.inner, dtype=self.ds_dtype
        )
        sdS = cute.make_tensor(
            cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer
        )
        sKt = storage.sK.get_tensor(
            sKt_layout.outer, swizzle=sKt_layout.inner, dtype=self.k_dtype
        )
        # Flashmask metadata reduced before deriving the m-block skip ranges.
        #
        # Built HERE, at the kernel's top level, and not inside the warp regions: the
        # DSL's AST pass treats `obj.method(...)` as a write to `obj` and threads that
        # object through every enclosing region, and a SharedStorage instance cannot be
        # flattened into a region argument ("unable to convert SharedStorage to
        # Numeric"). Every storage.*.get_tensor() call therefore has to stay out here.
        # The layout is built here rather than reused from a host-side attribute: a
        # layout created in the host-side jit function is an SSA value of *that*
        # region, and kernel regions are isolated ("'cute.make_view' op using value
        # defined outside the region"). Layouts either come in as kernel parameters
        # or are constructed inside the kernel.
        sFM_red = storage.sFM_max_min_ptr.get_tensor(
            cute.make_layout(FLASHMASK_META_SLOTS)
        )
        # A-operand views for the output gemms (address comes from tA_addr).
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        tP = cute.make_tensor(tmem_ptr, tP_layout.outer)
        tdS = cute.make_tensor(tmem_ptr, tdS_layout.outer)
        # One view per output scratch slot; output chunk c uses slot
        # c % num_out_slots. Same layout in every slot, so the T2R copy atoms built
        # from slot 0 apply to all of them.
        _slot_base = lambda s: (
            tmem_ptr + self.tmem_out_offset + s * self.tmem_out_slot_cols
        )
        tdVtdV_slots = tuple(
            cute.make_tensor(
                _slot_base(s),
                thr_mma_dV.make_fragment_C(
                    thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
                ).layout,
            )
            for s in range(self.num_out_slots)
        )
        tdQtdQ_slots = tuple(
            cute.make_tensor(
                _slot_base(s),
                thr_mma_dQ.make_fragment_C(
                    thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
                ).layout,
            )
            for s in range(self.num_out_slots)
        )
        tdKtdK_slots = tuple(
            cute.make_tensor(
                _slot_base(s),
                thr_mma_dK.make_fragment_C(
                    thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
                ).layout,
            )
            for s in range(self.num_out_slots)
        )

        # The m loop: every barrier below is used exactly once per m iteration, so a
        # single phase bit per warp tracks all of them (mbarrier phases flip on each
        # completion; iteration j waits phase j & 1).
        seqlen_q = cute.size(mQ.shape[0])
        seqlen_k = cute.size(mK.shape[0])
        num_m_block = cute.ceil_div(seqlen_q, self.tile_m)

        # GQA / MQA: the grid's y is a *query* head; K, V and the dK / dV
        # accumulators are indexed by its kv head. Several query heads therefore
        # add into the same dK / dV rows, which the atomic reduce already handles.
        head_idx_kv = head_idx // self.qhead_per_kvhead

        # ---------------------------------------------------------------- m skip range
        # An m block whose every element is masked contributes nothing (P == 0 => dV, and
        # dS = P * (dP - dPsum) == 0 => dK, dQ), so the m loop walks
        #     [m_lo, m_hi) minus one contiguous band [b_lo, b_hi)
        # instead of every block. The per-element mask further down stays the source of
        # truth, which makes these bounds only need to be CONSERVATIVE: skipping less
        # than possible costs time, skipping a block that still has an unmasked element
        # is a wrong answer.
        #
        # At cta_group=2 the two CTAs of a pair own adjacent key blocks but share every
        # collective operation (the gemms below take cta_group=2, the loads are
        # multicast), so they must run the SAME number of m iterations. The range is
        # therefore computed over the pair's whole key range -- both CTAs read the same
        # bounds and apply the same expressions, so they agree by construction and no
        # cross-CTA reduction is needed. cta_group=1 keeps its exact previous range.
        n_block_pair = (n_block // self.cta_group_size) * self.cta_group_size
        m_lo = Int32(0)
        m_hi = num_m_block
        b_lo = num_m_block
        b_hi = num_m_block
        if cutlass.const_expr(self.is_causal):
            # keep n_global <= m_global + seqlen_k - seqlen_q, so the first m block that
            # can hold an unmasked element is (n_lo + seqlen_q - seqlen_k) / tile_m --
            # the identical expression to block_info.py's get_m_block_min_max, evaluated
            # at the pair's lowest key (the least constraining of the two).
            m_lo = cutlass.max(
                Int32(0),
                (n_block_pair * self.tile_n + seqlen_q - seqlen_k) // self.tile_m,
            )
        if cutlass.const_expr(self.fm_bound_num > 0):
            fm_heads = cute.size(mFM.shape[1])
            fm_b = batch_idx if cute.size(mFM.shape[0]) > 1 else Int32(0)
            fm_h = head_idx // (cute.size(mQ.shape[2]) // fm_heads)
        if cutlass.const_expr(self.fm_bound_num in (1, 2)):
            # fm_bound_num == 4 (non-causal, both tails bounded) deliberately gets NO
            # skip: it would need four reduced scalars (max/min of both tails' starts and
            # ends) and the resulting iteration space is two bands rather than one, which
            # the seg1 / seg2 walk below cannot express. Falling through leaves
            # [0, num_m_block) and only costs time -- the per-element mask stays the
            # source of truth.
            #
            # Reduce this key pair's per-column bounds to two scalars: the largest
            # lower-tail start and the smallest end. One warp does it (32 lanes x 4
            # columns + a butterfly reduce, so every lane ends up with the result and can
            # write the same value), then the CTA barrier publishes it to the load / mma /
            # compute warps -- they must all see the SAME range or they deadlock, so this
            # is computed once and read, never recomputed per warp.
            assert self.tile_n % cute.arch.WARP_SIZE == 0
            if warp_idx == self.load_warp_id:
                lane = tidx % cute.arch.WARP_SIZE
                acc_ds = Int32(0)
                acc_end = Int32(2**31 - 1)
                pair_cols = self.cta_group_size * self.tile_n
                for j in cutlass.range_constexpr(pair_cols // cute.arch.WARP_SIZE):
                    col = n_block_pair * self.tile_n + j * cute.arch.WARP_SIZE + lane
                    # Columns past seqlen_k are masked for every m, so they must not
                    # constrain the skip: ds = 0 and end = INT32_MAX are the neutral
                    # elements of max(ds) and min(end). The same clamp covers a pair
                    # whose second key block falls entirely past seqlen_k.
                    in_range = col < seqlen_k
                    safe_col = cutlass.min(col, seqlen_k - 1)
                    acc_ds = cutlass.max(
                        acc_ds, mFM[fm_b, fm_h, safe_col, 0] if in_range else Int32(0)
                    )
                    if cutlass.const_expr(self.fm_bound_num >= 2):
                        acc_end = cutlass.min(
                            acc_end,
                            mFM[fm_b, fm_h, safe_col, 1]
                            if in_range
                            else Int32(2**31 - 1),
                        )
                acc_ds = utils.warp_reduce(acc_ds, lambda a, b: cutlass.max(a, b))
                acc_end = utils.warp_reduce(acc_end, lambda a, b: cutlass.min(a, b))
                sFM_red[0] = acc_ds
                sFM_red[1] = acc_end
            cute.arch.barrier()
            max_ds = sFM_red[0]
            min_end = sFM_red[1]
            # Which m blocks are fully masked, from the reference semantics
            # (generate_startend_row_indices.py:4-35), evaluated for ALL columns at once
            # via max(ds) / min(end):
            #   lower tail [ds, de) or [ds, seqlen_q)  -> block j masked if max_ds <= j*tm
            #   upper tail [0, ue)                     -> block j masked if (j+1)*tm <= min_ue
            first_masked = cute.ceil_div(max_ds, self.tile_m)
            if cutlass.const_expr(self.is_causal):
                if cutlass.const_expr(self.fm_bound_num == 2):
                    # band [ds, de): a middle range of m blocks drops out
                    b_lo = first_masked
                    b_hi = cutlass.max(b_lo, min_end // self.tile_m)
                else:
                    # band [ds, seqlen_q): everything from first_masked on drops out
                    m_hi = cutlass.min(m_hi, first_masked)
            else:
                # [0, ue) at the front and [ds, seqlen_q) at the back. Conservative:
                # different columns may be covered by different clauses, which two
                # scalars cannot express, so only the two ends are trimmed.
                m_lo = cutlass.max(m_lo, min_end // self.tile_m)
                m_hi = cutlass.min(m_hi, first_masked)
        # Iteration space: seg1 = below the band, seg2 = above it. b_hi >= b_lo is
        # guaranteed above, so the two segments can neither overlap nor leave a gap.
        seg1 = cutlass.max(Int32(0), cutlass.min(b_lo, m_hi) - m_lo)
        seg2_base = cutlass.max(m_lo, b_hi)
        num_iters = seg1 + cutlass.max(Int32(0), m_hi - seg2_base)

        # gmem tiles: mK/mQ/mV/mdO are (s, d, h, b) after the transpose.
        mK_cur = mK[None, None, head_idx_kv, batch_idx]
        mQ_cur = mQ[None, None, head_idx, batch_idx]
        mV_cur = mV[None, None, head_idx_kv, batch_idx]
        mdO_cur = mdO[None, None, head_idx, batch_idx]
        gK = cute.local_tile(
            mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, None)
        )
        gQ = cute.local_tile(
            mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (0, None)
        )
        gdOt = cute.local_tile(
            mdO_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (0, None)
        )
        # dO / Q^T for the dV / dK gemms come from the (d, s, h, b) views, so the
        # tile is (chunk, tile_m) and the chunk index lives in mode 0.
        gdO = cute.local_tile(
            mdO_dV[None, None, head_idx, batch_idx],
            cute.select(self.mma_tiler_pdo, mode=[1, 2]),
            (None, 0),
        )
        gQt = cute.local_tile(
            mQ_dK[None, None, head_idx, batch_idx],
            cute.select(self.mma_tiler_dsq, mode=[1, 2]),
            (None, 0),
        )
        # K^T for dQ: (d, s) view, tile (d_chunk, tile_n), chunk in mode 0.
        gKt = cute.local_tile(
            mK_dQ[None, None, head_idx_kv, batch_idx],
            cute.select(self.mma_tiler_dsk, mode=[1, 2]),
            (None, n_block),
        )
        tSgK = thr_mma_S.partition_A(gK)
        tSgQ = thr_mma_S.partition_B(gQ)
        tdPgdOt = thr_mma_dP.partition_B(gdOt)
        tdVgdO = thr_mma_dV.partition_B(gdO)
        tdKgQt = thr_mma_dK.partition_B(gQt)
        tdQgKt = thr_mma_dQ.partition_B(gKt)

        if warp_idx == self.load_warp_id:
            # The register budget is per-warp state, not per-iteration work: setting it
            # inside the m loop re-issues setmaxnreg on every iteration.
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            phase = Int32(0)
            for it in cutlass.range(num_iters, unroll=1):
                # The iteration counter drives the barrier phases; m_iter is the actual
                # block index, which skips the fully masked band (see the skip range).
                m_iter = m_lo + it if it < seg1 else seg2_base + (it - seg1)
                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), tSgK, sK
                )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), thr_mma_S.partition_B(
                        cute.local_tile(
                            mQ_cur,
                            cute.select(self.mma_tiler_kq, mode=[1, 2]),
                            (m_iter, None),
                        )
                    ), sQ
                )
                # K streams through d_chunks_resident buffers (3 chunks, 2 buffers at
                # tile_n=128) and Q through its single stage; both for chunk c land on
                # the same barrier, so chunk c waits for chunk c-1 to be consumed.
                for c in cutlass.range_constexpr(self.num_d_chunks):
                    # The dependency wraps across m iterations: chunk 0 of this
                    # iteration reuses the buffer that the previous iteration's last
                    # chunk was still reading. Without this the loads raced ahead and
                    # the MMA saw a half-overwritten buffer (illegal access).
                    #
                    # The wrap wait must be SKIPPED on the first iteration.
                    # mbarrier_wait spins on mbarrier.try_wait.parity, which only
                    # returns once the phase with that parity has *completed*: on a
                    # fresh barrier a wait for parity 1 blocks until two completions,
                    # so "phase ^ 1" on the first iteration (it == 0) is a deadlock,
                    # not a no-op.
                    if cutlass.const_expr(c > 0):
                        cute.arch.mbarrier_wait(mbar_Sin_empty + (c - 1), phase)
                    else:
                        if it > 0:
                            cute.arch.mbarrier_wait(
                                mbar_Sin_empty + (self.num_d_chunks - 1), phase ^ 1
                            )
                            # sQ and sK are each used TWICE per iteration through a
                            # second view: sQt (for dK) aliases sQ, sKt (for dQ)
                            # aliases sK. The release for the next iteration therefore
                            # has to come from the LAST consumer of the buffer, not
                            # from the S loop -- with only the Sin_empty gate the next
                            # iteration's K/Q loads overwrote the buffers while the
                            # dK / dQ gemms were still reading them.
                            cute.arch.mbarrier_wait(
                                mbar_dKin_empty + (self.num_d_chunks - 1), phase ^ 1
                            )
                            cute.arch.mbarrier_wait(
                                mbar_dQin_empty + (self.num_d_chunks - 1), phase ^ 1
                            )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_Sin_full + c,
                            self.tma_copy_bytes["K"] + self.tma_copy_bytes["Q"],
                        )
                    load_K(c, c % self.d_chunks_resident, tma_bar_ptr=mbar_Sin_full + c)
                    load_Q(c, 0, tma_bar_ptr=mbar_Sin_full + c)
                # V + dO^T: one dv chunk at a time, both into their single stage.
                # sV has its stage mode sliced off (rank 3), so cpasync.tma_partition
                # needs a gmem tile of matching rank: bake the chunk index into
                # local_tile and use single_stage=True, one copy fn per chunk (the
                # existing kernel does the same with load_V_low / load_V_high). sdOt
                # keeps its stage mode, so it uses the indexed form like Q.
                load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dOt, 0, cute.make_layout(1), thr_mma_dP.partition_B(
                        cute.local_tile(
                            mdO_cur,
                            cute.select(self.mma_tiler_vdo, mode=[1, 2]),
                            (m_iter, None),
                        )
                    ), sdOt
                )
                for c in cutlass.range_constexpr(self.num_dv_chunks):
                    gV_c = cute.local_tile(
                        mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, c)
                    )
                    load_V_c, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        thr_mma_dP.partition_A(gV_c),
                        sV,
                        single_stage=True,
                    )
                    # Same cross-iteration wrap as the K/Q loop, and again skipped on
                    # the first iteration (a parity-1 wait on a fresh barrier blocks).
                    if cutlass.const_expr(c > 0):
                        cute.arch.mbarrier_wait(mbar_dPin_empty + (c - 1), phase)
                    else:
                        if it > 0:
                            cute.arch.mbarrier_wait(
                                mbar_dPin_empty + (self.num_dv_chunks - 1), phase ^ 1
                            )
                            # sdO is the same buffer as sdOt (the dV view aliases the
                            # dP view), so the next iteration's V/dO^T loads must also
                            # wait for the dV gemm to be done with it.
                            cute.arch.mbarrier_wait(
                                mbar_dVin_empty + (self.num_dv_chunks - 1), phase ^ 1
                            )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_dPin_full + c,
                            self.tma_copy_bytes["V"] + self.tma_copy_bytes["dOt"],
                        )
                    load_V_c(tma_bar_ptr=mbar_dPin_full + c)
                    load_dOt(c, 0, tma_bar_ptr=mbar_dPin_full + c)

                # dO reload in the dV orientation. The buffer is the one dP just
                # finished with, so gate the first reload on the last dP consumption.
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), thr_mma_dV.partition_B(
                        cute.local_tile(
                            mdO_dV[None, None, head_idx, batch_idx],
                            cute.select(self.mma_tiler_pdo, mode=[1, 2]),
                            (None, m_iter),
                        )
                    ), sdO
                )
                cute.arch.mbarrier_wait(mbar_dPin_empty + (self.num_dv_chunks - 1), phase)
                for c in cutlass.range_constexpr(self.num_dv_chunks):
                    # Cross-iteration wrap, skipped on the first iteration.
                    if cutlass.const_expr(c > 0):
                        cute.arch.mbarrier_wait(mbar_dVin_empty + (c - 1), phase)
                    else:
                        if it > 0:
                            cute.arch.mbarrier_wait(
                                mbar_dVin_empty + (self.num_dv_chunks - 1), phase ^ 1
                            )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_dVin_full + c, self.tma_copy_bytes["dO"]
                        )
                    load_dO(c, 0, tma_bar_ptr=mbar_dVin_full + c)

                # Q^T reload for dK, into the sQ buffer that the S gemm is done with.
                load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Qt, 0, cute.make_layout(1), thr_mma_dK.partition_B(
                        cute.local_tile(
                            mQ_dK[None, None, head_idx, batch_idx],
                            cute.select(self.mma_tiler_dsq, mode=[1, 2]),
                            (None, m_iter),
                        )
                    ), sQt
                )
                cute.arch.mbarrier_wait(mbar_Sin_empty + (self.num_d_chunks - 1), phase)
                for c in cutlass.range_constexpr(self.num_d_chunks):
                    # Cross-iteration wrap, skipped on the first iteration.
                    if cutlass.const_expr(c > 0):
                        cute.arch.mbarrier_wait(mbar_dKin_empty + (c - 1), phase)
                    else:
                        if it > 0:
                            cute.arch.mbarrier_wait(
                                mbar_dKin_empty + (self.num_d_chunks - 1), phase ^ 1
                            )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_dKin_full + c, self.tma_copy_bytes["Qt"]
                        )
                    load_Qt(c, 0, tma_bar_ptr=mbar_dKin_full + c)

                # K^T for the dQ gemm: the S gemm is long done with sK, so re-stream it
                # through the same d_chunks_resident buffers in the transposed layout.
                #
                # Measured (do not retry): sKt is NOT an aliasable view of sK, so this
                # re-TMA cannot be hoisted out of the m loop even though the n-block's
                # K bytes never change. At d_chunk=64 the two layouts are
                #   sK  ((128,16),1,4,8):((64,1),0,16,8192)
                #   sKt ((64,16),1,8,8):((1,64),0,1024,8192)
                # i.e. genuinely transposed strides (same swizzle S<3,4,3>). The S gemm
                # contracts K over d, the dQ gemm contracts it over n, so one SMEM copy
                # cannot serve both. Same for sQ / sQt and sdO / sdOt. Removing the
                # double fetch would need either a second 128KB SMEM buffer (no room) or
                # an in-SMEM transpose. Its cost is TMA issue + barrier latency rather
                # than HBM traffic: all 64 query heads of an n-block re-read the same
                # K bytes, so L2 serves them.
                load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Kt, 0, cute.make_layout(1), tdQgKt, sKt
                )
                cute.arch.mbarrier_wait(mbar_Sin_empty + (self.num_d_chunks - 1), phase)
                for c in cutlass.range_constexpr(self.num_d_chunks):
                    if cutlass.const_expr(c >= self.d_chunks_resident):
                        cute.arch.mbarrier_wait(
                            mbar_dQin_empty + (c - self.d_chunks_resident), phase
                        )
                    else:
                        # wraps into the previous m iteration (see the K/Q loop), so it
                        # is skipped on the first iteration
                        if it > 0:
                            cute.arch.mbarrier_wait(
                                mbar_dQin_empty
                                + (c - self.d_chunks_resident + self.num_d_chunks),
                                phase ^ 1,
                            )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar_dQin_full + c, self.tma_copy_bytes["Kt"]
                        )
                    load_Kt(c, c % self.d_chunks_resident, tma_bar_ptr=mbar_dQin_full + c)

                phase ^= 1
        elif warp_idx == self.mma_warp_id:
            # TMEM is allocated ONCE per CTA, not once per m iteration. Allocating
            # inside the loop asked for a second 512-column region while the first was
            # still held (and after relinquish_tmem_alloc_permit, which forbids any
            # further allocation) -- the second allocation returned an address the
            # tcgen05 MMAs then wrote through, which is the illegal access that showed
            # up as cudaErrorLaunchFailure once num_m_block > 1.
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            cute.arch.alloc_tmem(Int32(self.tmem_alloc_cols), storage.tmem_holding_buf)
            cute.arch.sync_warp()
            cute.arch.relinquish_tmem_alloc_permit()

            phase = Int32(0)
            for it in cutlass.range(num_iters, unroll=1):
                # The iteration counter drives the barrier phases; m_iter is the actual
                # block index, which skips the fully masked band (see the skip range).
                m_iter = m_lo + it if it < seg1 else seg2_base + (it - seg1)
                tSrK = tiled_mma_S.make_fragment_A(sK)
                tSrQ = tiled_mma_S.make_fragment_B(sQ)
                for c in cutlass.range_constexpr(self.num_d_chunks):
                    cute.arch.mbarrier_wait(mbar_Sin_full + c, phase)
                    sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_S,
                        tStS,
                        tSrK,
                        tSrQ,
                        sA=sK,
                        sB=sQ,
                        A_idx=c % self.d_chunks_resident,
                        B_idx=0,
                        zero_init=(c == 0),
                        cta_group=self.cta_group_size,
                    )
                    with cute.arch.elect_one():
                        # One commit per gemm: two back-to-back tcgen05.commit calls do
                        # not reliably attach both barriers to the same MMA, which
                        # deadlocked the K stream. K and Q for a chunk therefore share
                        # one full/empty pair (waiting on chunk c-1 is stricter than K's
                        # real need of c-2, which is harmless).
                        tcgen05.commit(mbar_Sin_empty + c)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar_S_full)

                # dP^T = sum_c V_c @ dO_c^T  (contraction over head_dim_v)
                tdPrV = tiled_mma_dP.make_fragment_A(sV)
                tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
                for c in cutlass.range_constexpr(self.num_dv_chunks):
                    cute.arch.mbarrier_wait(mbar_dPin_full + c, phase)
                    sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dP,
                        tdPtdP,
                        tdPrV,
                        tdPrdOt,
                        sA=sV,
                        sB=sdOt,
                        # sV had its stage mode sliced off, so its A fragment is rank 3
                        # and takes no index; sdOt kept its (single) stage mode, so its
                        # B fragment must be indexed or gemm_ptx_partial's crd2idx on a
                        # rank-4 layout fails.
                        A_idx=None,
                        B_idx=0,
                        zero_init=(c == 0),
                        cta_group=self.cta_group_size,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_dPin_empty + c)
                with cute.arch.elect_one():
                    tcgen05.commit(mbar_dP_full)

                # The compute warps read S and dP out, then write P and dS back as
                # bf16 into the upper halves of those same regions.
                cute.arch.mbarrier_wait(mbar_PdS_full, phase)

                # Output chunk c writes scratch slot c % num_out_slots, so the slot's
                # previous user is chunk c - num_out_slots -- in this iteration if that
                # index is still >= 0, otherwise in the previous one (the chunk sequence
                # runs continuously across m iterations, hence the phase flip). With
                # num_out_slots == 1 this is exactly the old "wait for c - 1" rule.
                def wait_out_slot_free(oc):
                    prev = cutlass.const_expr(oc - self.num_out_slots)
                    if cutlass.const_expr(prev >= 0):
                        cute.arch.mbarrier_wait(mbar_out_empty + prev, phase)
                    else:
                        if it > 0:
                            cute.arch.mbarrier_wait(
                                mbar_out_empty + (num_out_chunks + prev), phase ^ 1
                            )

                # dV_c = P^T @ dO_c   (A = P from TMEM)
                tdVrP = tiled_mma_dV.make_fragment_A(tP)
                tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
                for c in cutlass.range_constexpr(self.num_dv_chunks):
                    cute.arch.mbarrier_wait(mbar_dVin_full + c, phase)
                    wait_out_slot_free(c)
                    sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dV,
                        tdVtdV_slots[c % self.num_out_slots],
                        tdVrP,
                        tdVrdO,
                        sA=None,
                        sB=sdO,
                        A_idx=None,
                        B_idx=0,
                        zero_init=True,
                        tA_addr=self.tmem_P_offset,
                        cta_group=self.cta_group_size,
                    )
                    with cute.arch.elect_one():
                        # Exactly one commit per gemm: back-to-back commits in a single
                        # elect_one did not reliably arm both barriers (that deadlocked
                        # the K stream). The matching *_in_empty signals are arrived by
                        # the compute warps, which only get there after out_full fired.
                        tcgen05.commit(mbar_out_full + c)

                # dK_c = dS^T @ Q_c   (A = dS from TMEM)
                tdKrdS = tiled_mma_dK.make_fragment_A(tdS)
                tdKrQt = tiled_mma_dK.make_fragment_B(sQt)
                for c in cutlass.range_constexpr(self.num_d_chunks):
                    out_c = self.num_dv_chunks + c
                    cute.arch.mbarrier_wait(mbar_dKin_full + c, phase)
                    wait_out_slot_free(out_c)
                    sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dK,
                        tdKtdK_slots[out_c % self.num_out_slots],
                        tdKrdS,
                        tdKrQt,
                        sA=None,
                        sB=sQt,
                        A_idx=None,
                        B_idx=0,
                        zero_init=True,
                        tA_addr=self.tmem_dS_offset,
                        cta_group=self.cta_group_size,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_out_full + out_c)

                # dQ_c = dS^T @ K_c  (A = dS from SMEM in the (m, n) view; M = tile_m =
                # 128, so the 32-datapath T2R applies to its accumulator too)
                cute.arch.mbarrier_wait(mbar_dSsmem_full, phase)
                tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
                tdQrKt = tiled_mma_dQ.make_fragment_B(sKt)
                for c in cutlass.range_constexpr(self.num_d_chunks):
                    out_c = self.num_dv_chunks + self.num_d_chunks + c
                    cute.arch.mbarrier_wait(mbar_dQin_full + c, phase)
                    wait_out_slot_free(out_c)
                    sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dQ,
                        tdQtdQ_slots[out_c % self.num_out_slots],
                        tdQrdS,
                        tdQrKt,
                        sA=sdS,
                        sB=sKt,
                        A_idx=None,
                        B_idx=c % self.d_chunks_resident,
                        zero_init=True,
                        cta_group=self.cta_group_size,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_out_full + out_c)

                phase ^= 1
            tmem_ptr_real = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
            )
            cute.arch.mbarrier_wait(mbar_tmem_dealloc, 0)
            cute.arch.dealloc_tmem(tmem_ptr_real, Int32(self.tmem_alloc_cols))

        elif warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            compute_tidx = tidx - self.compute_warp_ids[0] * cute.arch.WARP_SIZE
            phase = Int32(0)
            for it in cutlass.range(num_iters, unroll=1):
                # The iteration counter drives the barrier phases; m_iter is the actual
                # block index, which skips the fully masked band (see the skip range).
                m_iter = m_lo + it if it < seg1 else seg2_base + (it - seg1)
                # The S -> P -> dP -> dS round trip runs in chunks of
                # softmax_chunk_m columns instead of over the whole tile at once.
                #
                # Why (ncu on the 23.75ms config, b1/s8192/h64/d512):
                #   l1tex local ld / st            34.06 GB / 31.25 GB
                #   launch__registers_per_thread   128  (Block Limit Registers = 1)
                #   tensor pipe utilisation        0.25 % of peak
                #   warp cycles per issued instr   53.95, of which 45.1 (83.7%)
                #                                  stalled on an L1TEX scoreboard
                # One fragment is tile_n * tile_m / 128 = 128 f32 per thread and the
                # round trip keeps four of them live, plus the two packed R2T buffers:
                # 640 f32 against 128 registers, and the CTA already owns the whole
                # register file, so more registers cannot be had. 65GB of local
                # traffic and 84% of the stall budget were the spill. Chunking makes
                # the live set 5 * W instead of 5 * tile_m. DSA's bwd sits at 32 f32
                # per thread here (tile 64x64, split into two Rep(4) LDTMs).
                #
                # Everything below is built once: every chunk has the same shape and
                # the same thread mapping, only the TMEM column offset and the m
                # coordinate base move.
                W = cutlass.const_expr(self.softmax_chunk_m)
                # 32-datapath T2R. This is why the config uses tile_n=128: with
                # M = cta_group * tile_n = 128 these atoms apply and their value mode is
                # a pure column run, which is what makes the packed bf16 R2T store below
                # line up element-for-element (load rep W/4, store rep W/8 -- the packed
                # view is half as wide in f32 columns). At M=64 only the 16-datapath
                # atoms are available and that correspondence breaks: measured exact for
                # a constant A operand but ~7% off for A = m or A = n.
                mma_S_chunk = sm100_utils_basic.make_trivial_tiled_mma(
                    self.q_dtype,
                    tcgen05.OperandMajorMode.K,
                    tcgen05.OperandMajorMode.K,
                    self.acc_dtype,
                    self.cta_group,
                    (self.mma_tiler_kq[0], W),
                )
                thr_mma_S_c = mma_S_chunk.get_slice(0)
                tS_chunk_layout = thr_mma_S_c.make_fragment_C(
                    thr_mma_S_c.partition_shape_C((self.mma_tiler_kq[0], W))
                ).layout
                # dP gets its own chunk layout: the full-tile views this replaced took
                # P's shape from tStS (mma_tiler_kq, the S gemm) and dS's from tdPtdP
                # (mma_tiler_vdo, the dP gemm). They are two different MMAs, so do not
                # assume one layout describes both.
                mma_dP_chunk = sm100_utils_basic.make_trivial_tiled_mma(
                    self.q_dtype,
                    tcgen05.OperandMajorMode.K,
                    tcgen05.OperandMajorMode.K,
                    self.acc_dtype,
                    self.cta_group,
                    (self.mma_tiler_vdo[0], W),
                )
                thr_mma_dP_c = mma_dP_chunk.get_slice(0)
                tdP_chunk_layout = thr_mma_dP_c.make_fragment_C(
                    thr_mma_dP_c.partition_shape_C((self.mma_tiler_vdo[0], W))
                ).layout
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(W // 4)), Float32
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    tmem_load_atom, cute.make_tensor(tStS.iterator, tS_chunk_layout)
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(compute_tidx)
                # partition_D takes a tensor already shaped like the accumulator, i.e.
                # partitioned by the MMA first -- a flat (tile_n, W) identity
                # tensor does not match the tiler and fails op creation.
                cS = thr_mma_S_c.partition_C(
                    cute.make_identity_tensor((self.mma_tiler_kq[0], W))
                )
                tScS = thr_copy_t2r.partition_D(cS)
                # Four separate f32 fragments, on purpose. Tried and REVERTED: aliasing
                # them in pairs (P overwriting S, dS overwriting dP) to cut the live
                # count in half made BW 23.75 -> 40.47 ms (+70%) on b1/s8192/h64/d512.
                # make_fragment lowers to an alloca, and reading and writing the same
                # alloca inside one unrolled loop defeats its promotion to registers, so
                # the whole buffer lands in local memory. Left separate, ptxas sees that
                # S dies after the P loop (and dP after the dS loop) and reuses those
                # registers itself. Shrinking the fragments is the fix; aliasing is not.
                tSrS = cute.make_fragment(tScS.shape, Float32)
                tSrP = cute.make_fragment(tScS.shape, Float32)
                tSrdP = cute.make_fragment(tScS.shape, Float32)
                tSrdS = cute.make_fragment(tScS.shape, Float32)
                frag_len = cutlass.const_expr(cute.size(tSrS))

                # R2T machinery: P and dS go back into the upper halves of the S / dP
                # regions as bf16, where they are the A operands of the dV / dK gemms.
                # With M = cta_group * tile_n = 128 the 32-datapath atoms apply, so the
                # store is the proven fwd pattern: an f32 register fragment whose bytes
                # are viewed as bf16, and a store rep that is half the load rep (the
                # packed view is half as wide in f32 columns).
                tile_p_like_f32_c = cutlass.const_expr(W // 32 * self.q_dtype.width)
                tmem_store_atom = cute.make_copy_atom(
                    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(W // 8)), Float32
                )
                tP_chunk_layout = cute.composition(
                    tS_chunk_layout,
                    cute.make_layout((self.tile_n, tile_p_like_f32_c)),
                )
                tdS_chunk_layout = cute.composition(
                    tdP_chunk_layout,
                    cute.make_layout((self.tile_n, tile_p_like_f32_c)),
                )
                cP = cute.make_identity_tensor((self.tile_n, tile_p_like_f32_c))
                thr_store_P = tcgen05.make_tmem_copy(
                    tmem_store_atom,
                    cute.make_tensor(
                        tStS.iterator + self.tmem_s_to_p_offset, tP_chunk_layout
                    ),
                ).get_slice(compute_tidx)
                thr_store_dS = tcgen05.make_tmem_copy(
                    tmem_store_atom,
                    cute.make_tensor(
                        tdPtdP.iterator + self.tmem_s_to_p_offset, tdS_chunk_layout
                    ),
                ).get_slice(compute_tidx)
                tSrP_r2t_f32 = cute.make_fragment(
                    thr_store_P.partition_S(cP).shape, Float32
                )
                tSrdS_r2t_f32 = cute.make_fragment(
                    thr_store_dS.partition_S(cP).shape, Float32
                )
                tSrP_r2t = cute.make_tensor(
                    cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS.layout
                )
                tSrdS_r2t = cute.make_tensor(
                    cute.recast_ptr(tSrdS_r2t_f32.iterator, dtype=self.ds_dtype), tSrS.layout
                )
                # Two bf16 per f32 lane: the packed fragment must hold exactly the
                # elements the T2R load produced, or part of P / dS stays uninitialized
                # and the gemm returns NaN.
                assert 2 * cute.size(tSrP_r2t_f32) == frag_len, (
                    "packed R2T fragment holds %d bf16, T2R produced %d"
                    % (2 * cute.size(tSrP_r2t_f32), frag_len)
                )
                # dS -> SMEM in its natural (n, m) orientation; the dQ gemm reads the
                # same bytes through the transposed sdS view.
                #
                # Vectorised r2s, built from the T2R copy's destination TV layout -- the
                # recipe the postprocess uses (flash_bwd_postprocess.py:438-447 and
                # :519-524): get_smem_store_op picks the widest legal store for the
                # (layout, dtype) pair and make_tiled_copy re-tiles it onto exactly the
                # thread/value mapping the T2R produced, so the register fragment can go
                # out as 16-byte stores instead of W scalar ones. NB: make_tiled_copy_D
                # is NOT the tool here -- it hands each thread twice the elements and
                # does not expose layout_tv; that mistake cost four rounds.
                #
                # If this breaks, the symptom is precise: dS feeds dK through TMEM and dQ
                # through SMEM, so dQ wrong while dK stays right means the store mapping
                # is off.
                smem_store_atom = sm100_utils_basic.get_smem_store_op(
                    LayoutEnum.ROW_MAJOR,
                    self.ds_dtype,
                    Float32,
                    tiled_copy_t2r,
                )
                thr_r2s_dS = cute.make_tiled_copy(
                    smem_store_atom,
                    layout_tv=tiled_copy_t2r.layout_dst_tv_tiled,
                    tiler_mn=tiled_copy_t2r.tiler_mn,
                ).get_slice(compute_tidx)
                # partition_C needs a plain (tile_n, m) view: handed the raw A-operand
                # layout it broadcasts instead of slicing (measured: a stride-0 mode),
                # and partition_D then hands every thread the whole tile. Composing the
                # swizzled layout down to (n, m) keeps the A-operand addresses.
                sdSt_nm = cute.make_tensor(
                    sdSt.iterator,
                    cute.composition(
                        sdSt.layout, cute.make_layout((self.tile_n, self.tile_m))
                    ),
                )

                mLSE_cur = cute.local_tile(
                    mLSE[batch_idx, head_idx, None], (self.tile_m,), (m_iter,)
                )
                mdPsum_cur = cute.local_tile(
                    mdPsum[batch_idx, head_idx, None], (self.tile_m,), (m_iter,)
                )
                # Masks, applied to P rather than to S: P == 0 kills the element's
                # contribution to dV, and dS = P * (dP - dPsum) == 0 kills it for dK and
                # dQ too. Doing it on P also means LSE / dPsum garbage in the padded rows
                # cannot turn into NaN (a select on the result, not on the exponent).
                #
                # S^T is (n, m) = (key, query): mode 0 of tScS is the key index and is
                # thread-derived (one thread owns one whole key row, for every chunk),
                # mode 1 is the query index and is a per-element compile-time constant.
                # So the key-side predicates are loop- and chunk-invariant and only the
                # query side is per element.
                # (Measured on B200 -- note this is the OPPOSITE of what mask.py's
                # apply_mask_sm100_transposed assumes, which is why that helper is not
                # used here.)
                #
                # TMA already zero-fills the out-of-range Q / K / dO tiles; what it
                # cannot do is stop exp2(0 - lse_pad) from being a nonzero P.
                n_global = n_block * self.tile_n + tScS[0][0]
                n_oob = n_global >= seqlen_k
                m_base = m_iter * self.tile_m
                if cutlass.const_expr(self.is_causal):
                    # Bottom-right aligned, the FA convention (mask.py's
                    # causal_row_offset = 1 + seqlen_k - n_block*tile_n - seqlen_q - ...):
                    # keep n_global <= m_global + seqlen_k - seqlen_q.
                    causal_m_min = n_global - (seqlen_k - seqlen_q)
                if cutlass.const_expr(self.fm_bound_num > 0):
                    # flashmask: startend_row_indices is [b, h_fm, seqlen_k, bound_num]
                    # and gives, PER KEY COLUMN, bounds on the QUERY rows to mask. The
                    # key column is this thread's row of S^T, so the bounds are read
                    # once per thread (mask.py re-reads them per element because there
                    # the key is the per-element coordinate).
                    #
                    # Semantics, verbatim from the reference
                    # (test_flashmask/generate_startend_row_indices.py:4-35):
                    #   has_end = (causal and bound_num == 2) or (not causal and == 4)
                    #   lower tail: mask rows [idx0, idx1) if has_end else [idx0, seqlen_q)
                    #   causal    : plus the causal mask (handled above)
                    #   not causal: plus the upper tail, [idx2, idx3) if has_end
                    #               else [0, idx1)
                    # fm_b / fm_h come from the kernel's top level (the skip range needs
                    # them too).
                    # Threads whose key is past seqlen_k are masked by n_oob anyway, but
                    # the read itself still has to stay in bounds.
                    fm_row = mFM[
                        fm_b, fm_h, cutlass.min(n_global, seqlen_k - 1), None
                    ]
                    has_end = cutlass.const_expr(
                        (self.is_causal and self.fm_bound_num == 2)
                        or (not self.is_causal and self.fm_bound_num == 4)
                    )
                    fm_ds = fm_row[0]
                    fm_de = fm_row[1] if cutlass.const_expr(has_end) else seqlen_q
                    if cutlass.const_expr(not self.is_causal):
                        if cutlass.const_expr(has_end):
                            fm_us, fm_ue = fm_row[2], fm_row[3]
                        else:
                            fm_us, fm_ue = Int32(0), fm_row[1]

                # S^T, then P = exp2(S * scale_log2 - lse[m]). S^T is (n, m) = (key,
                # query), and softmax normalizes over keys, so LSE and dPsum are
                # indexed by the *m* coordinate of each element.
                cute.arch.mbarrier_wait(mbar_S_full, phase)
                for cmi in cutlass.range_constexpr(self.num_softmax_chunks):
                    # DESCENDING chunk order, and it has to be: P is stored into the
                    # UPPER HALF of the S region (tmem_s_to_p_offset = tile_m // 2,
                    # bf16 needs half the columns) and dS likewise into dP's, so a
                    # chunk's R2T store lands on columns another chunk may not have
                    # read yet. Ascending order was measured wrong exactly this way:
                    # P of chunk 0 goes to f32 columns [64, 80), which is inside
                    # chunk 2's S range [64, 96) -- chunks 0 and 1 came out right and
                    # 2 and 3 read clobbered S / dP (dq cosine 0.85, dkv 0.90).
                    # Descending is always safe: chunk cm stores into
                    # [tile_m/2 + cm*W/2, ...), and tile_m/2 + cm*W/2 >= cm*W for any
                    # cm*W <= tile_m, i.e. never below the columns already consumed.
                    cm = cutlass.const_expr(self.num_softmax_chunks - 1 - cmi)
                    # TMEM column offsets: the f32 accumulators advance by W columns per
                    # chunk, the packed bf16 P / dS regions by W // 2 (two bf16 per f32
                    # column).
                    col_f32 = cutlass.const_expr(cm * W)
                    col_packed = cutlass.const_expr(cm * (W // 2))
                    m_off = cutlass.const_expr(cm * W)
                    tStS_c = cute.make_tensor(tStS.iterator + col_f32, tS_chunk_layout)
                    tdPtdP_c = cute.make_tensor(
                        tdPtdP.iterator + col_f32, tdP_chunk_layout
                    )
                    tStP_c = cute.make_tensor(
                        tStS.iterator + self.tmem_s_to_p_offset + col_packed,
                        tP_chunk_layout,
                    )
                    tStdS_c = cute.make_tensor(
                        tdPtdP.iterator + self.tmem_s_to_p_offset + col_packed,
                        tdS_chunk_layout,
                    )

                    cute.copy(thr_copy_t2r, thr_copy_t2r.partition_S(tStS_c), tSrS)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(frag_len):
                        m_idx = tScS[i][1] + m_off
                        p = cute.math.exp2(
                            tSrS[i] * softmax_scale_log2 - mLSE_cur[m_idx], fastmath=True
                        )
                        m_global = m_base + m_idx
                        bad = n_oob or m_global >= seqlen_q
                        if cutlass.const_expr(self.is_causal):
                            bad = bad or m_global < causal_m_min
                        if cutlass.const_expr(self.fm_bound_num > 0):
                            bad = bad or (m_global >= fm_ds and m_global < fm_de)
                            if cutlass.const_expr(not self.is_causal):
                                bad = bad or (m_global >= fm_us and m_global < fm_ue)
                        tSrP[i] = 0.0 if bad else p

                    # dP^T, then dS^T = P * (dP - dPsum[m]). The wait sits after the
                    # FIRST PROCESSED chunk's P pass (cmi == 0, i.e. the highest cm),
                    # where it was before this loop existed, so that pass still
                    # overlaps the mma warp's dP gemms.
                    if cutlass.const_expr(cmi == 0):
                        cute.arch.mbarrier_wait(mbar_dP_full, phase)
                    cute.copy(thr_copy_t2r, thr_copy_t2r.partition_S(tdPtdP_c), tSrdP)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(frag_len):
                        m_idx = tScS[i][1] + m_off
                        tSrdS[i] = tSrP[i] * (tSrdP[i] - mdPsum_cur[m_idx])

                    # R2T of this chunk's P and dS, then its dS slice to SMEM.
                    for i in cutlass.range_constexpr(frag_len):
                        tSrP_r2t[i] = tSrP[i].to(self.q_dtype)
                        tSrdS_r2t[i] = tSrdS[i].to(self.ds_dtype)
                    cute.copy(thr_store_P, tSrP_r2t_f32, thr_store_P.partition_D(tStP_c))
                    cute.copy(
                        thr_store_dS, tSrdS_r2t_f32, thr_store_dS.partition_D(tStdS_c)
                    )
                    tdSsdS = thr_copy_t2r.partition_D(
                        thr_mma_S_c.partition_C(
                            cute.local_tile(sdSt_nm, (self.tile_n, W), (0, cm))
                        )
                    )
                    assert cute.size(tdSsdS) == frag_len, (
                        "dS r2s destination has %d elements per thread, T2R produced %d"
                        % (cute.size(tdSsdS), frag_len)
                    )
                    cute.copy(
                        thr_r2s_dS,
                        cute.make_tensor(tSrdS_r2t.iterator, tdSsdS.shape),
                        tdSsdS,
                    )
                # P and dS are complete for the whole tile: publish them to the mma
                # warp (TMEM) and to the dQ gemm (SMEM). One arrive each per m
                # iteration, which is what the barriers were initialised for.
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(mbar_PdS_full)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(mbar_dSsmem_full)

                # dV / dK / dQ are drained by the drain warpgroup (see the branch
                # below); the compute warps are done with this iteration once P / dS
                # are in TMEM and SMEM.

                phase ^= 1

            # TMEM is released once, after the whole m loop -- the mma warp holds the
            # single allocation for the CTA's lifetime. Both the compute and the drain
            # warpgroup have to be done with TMEM before it goes away.
            cute.arch.mbarrier_arrive(mbar_tmem_dealloc)

        elif warp_idx <= self.drain_warp_ids[-1]:
            # ---------------------------------------------------------- drain warps
            # dV / dK / dQ leave TMEM through a fp32 gmem accumulator: T2R one
            # dQ_reduce_ncol-column slice into registers, then red.global.add.v4.f32
            # it into the accumulator. The add is what makes the m loop (dK/dV) and
            # the n grid (dQ) accumulate without any cross-CTA handshake.
            #
            # This used to run at the tail of the compute warps' iteration. It is its
            # own warpgroup so that the slices of iteration i overlap the softmax /
            # dS math of iteration i+1.
            # The handshake is unchanged: mbar_out_full[c] from the mma warp, and
            # mbar_out_empty[c] / *_in_empty[c] back to it -- only their arrive counts
            # moved to this warpgroup.
            #
            # Accumulator element order is identical to what the SMEM-staged version
            # wrote (see the note at the atomics), so FlashAttentionBackwardPostprocess
            # and _unblock_accum are unchanged.
            #
            # The postprocess still has to be *called* per head_dim slice: its
            # 1CTA path stages a whole tile_m x head_dim fp32 tile in SMEM
            # (at head_dim 512 that is 128*512*4 = 256KB) and holds it in
            # registers, so it has to be driven as e.g. 4 x 128 with
            # raw_storage_d instead of once at the full head_dim.
            cute.arch.setmaxregister_increase(self.num_regs_drain)
            drain_tidx = tidx - self.drain_warp_ids[0] * cute.arch.WARP_SIZE
            num_drain_threads = cutlass.const_expr(
                cute.arch.WARP_SIZE * len(self.drain_warp_ids)
            )
            ncol = cutlass.const_expr(self.dQ_reduce_ncol)
            assert num_drain_threads == self.tile_m and self.tile_n == self.tile_m, (
                "the accumulator layout assumes one row per drain thread"
            )
            # gmem accumulator per (batch, head); the tile index is the key block for
            # dK / dV and the m iteration for dQ.
            mdVaccum_cur = mdVaccum[batch_idx, head_idx_kv, None]
            mdKaccum_cur = mdKaccum[batch_idx, head_idx_kv, None]
            mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
            # ONE T2R per dQ_reduce_ncol-column slice, not one per chunk.
            #
            # MEASURED, and this is what the whole drain hinges on: with the drain in and
            # out, local ld/st went 19.70 / 19.40 GB vs 0.33 / 0.03 GB and BW 18.97ms vs
            # 6.55ms. So all 39GB of local traffic and 65% of the runtime is this drain, and the
            # 39GB is exactly "every T2R fragment element written once and read once"
            # (12 chunks x 512B = 6KB per thread per m tile, x 128 threads x ~49k m
            # tiles = 37GB). The fragments were not living in registers at all.
            #
            # Chunk-wide T2R needed a (ncol, flen/ncol) re-view of the fragment to hand
            # one slice at a time to the staging copy, i.e. `make_tensor(frag.iterator,
            # ...)` -- and taking an alloca's address is enough to stop SROA promoting
            # it, whatever its size. Slicing the T2R itself removes both problems: the
            # fragment IS one slice (64 f32 instead of 128), and no address is taken.
            #
            # Byte order is unchanged, so FlashAttentionBackwardPostprocess and
            # _unblock_accum keep working: the old code took fragment elements
            # [s*ncol, (s+1)*ncol) for slice s, and the T2R value mode is a pure column
            # run (element i <-> column i), so that is the same data a T2R at column
            # offset s*ncol produces.
            #
            # All three outputs share this machinery: dV's, dK's and dQ's accumulators
            # all have M = 128 (cta_group * tile_n for the K-side pair, tile_m for dQ),
            # so a single (128, ncol) slice layout and copy atom covers them and only
            # the TMEM column offset differs.
            mma_out_slice = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                self.acc_dtype,
                self.cta_group,
                (self.mma_tiler_pdo[0], ncol),
            )
            thr_mma_out_slice = mma_out_slice.get_slice(0)
            out_slice_layout = thr_mma_out_slice.make_fragment_C(
                thr_mma_out_slice.partition_shape_C((self.mma_tiler_pdo[0], ncol))
            ).layout
            tmem_load_atom_out = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(ncol // 4)), Float32
            )
            thr_t2r_out = tcgen05.make_tmem_copy(
                tmem_load_atom_out,
                cute.make_tensor(tmem_ptr + self.tmem_out_offset, out_slice_layout),
            ).get_slice(drain_tidx)
            c_out_slice = thr_mma_out_slice.partition_C(
                cute.make_identity_tensor((self.mma_tiler_pdo[0], ncol))
            )
            shape_out_slice = thr_t2r_out.partition_D(c_out_slice).shape
            flen_slice = cutlass.const_expr(cute.size(shape_out_slice))
            assert (
                self.mma_tiler_pdo[0] == self.mma_tiler_dsq[0]
                and self.mma_tiler_pdo[0] == self.mma_tiler_dsk[0]
            ), (
                "the shared slice T2R assumes all three output accumulators have the "
                "same M: %d / %d / %d"
                % (self.mma_tiler_pdo[0], self.mma_tiler_dsq[0], self.mma_tiler_dsk[0])
            )
            assert num_drain_threads * flen_slice == self.tile_m * ncol, (
                "slice T2R gives %d elements per thread over %d threads, staging wants "
                "tile_m * ncol = %d" % (flen_slice, num_drain_threads, self.tile_m * ncol)
            )
            # gmem accumulator for this CTA's block. Two levels of blocking:
            #   [head_dim slice][row block][4-col group][row][4 cols]
            # The outer slice exists so the shared postprocess can be called once
            # per slice with head_dim = accum_slice: each slice is a contiguous
            # region that looks exactly like a head_dim=accum_slice accumulator.
            # dK / dV are indexed by the key block (the grid's x), dQ by the m
            # iteration.
            num_n_block = cute.ceil_div(seqlen_k, self.tile_n)
            phase = Int32(0)
            for it in cutlass.range(num_iters, unroll=1):
                # Same iteration mapping as the mma / compute warps: the counter
                # drives the barrier phases, m_iter is the actual block index (it
                # indexes dQ's accumulator, so it must match exactly).
                m_iter = m_lo + it if it < seg1 else seg2_base + (it - seg1)
                # Iterated with range_constexpr, not `for ... in outputs`: a bare
                # Python for over a tuple is rewritten by the DSL into a dynamic loop
                # region and then it tries to flatten the tuple's contents.
                # All three outputs live at the same scratch base; only the slot and
                # the slice offset move.
                out_base_ptr = tmem_ptr + self.tmem_out_offset
                outputs = (
                    (self.num_dv_chunks, self.dv_chunk, mdVaccum_cur,
                     self.accum_slice_dv, num_n_block, n_block, 0, mbar_dVin_empty),
                    (self.num_d_chunks, self.d_chunk, mdKaccum_cur,
                     self.accum_slice_d, num_n_block, n_block, self.num_dv_chunks,
                     mbar_dKin_empty),
                    (self.num_d_chunks, self.d_chunk, mdQaccum_cur,
                     self.accum_slice_d, num_m_block, m_iter,
                     self.num_dv_chunks + self.num_d_chunks, mbar_dQin_empty),
                )
                # Iterated with range_constexpr, not `for ... in outputs`: a bare
                # Python for over a tuple is rewritten by the DSL into a dynamic loop
                # region and then it tries to flatten the tuple's contents.
                for oi in cutlass.range_constexpr(len(outputs)):
                    (nchunks, chunk_w, maccum, hd_slice, num_blocks,
                     block_idx, base, in_bar) = outputs[oi]
                    chunks_per_slice = cutlass.const_expr(hd_slice // chunk_w)
                    # Offsets are plain pointer arithmetic instead of nested local_tile
                    # because the outer (per-slice) extent is dynamic.
                    slice_stride = num_blocks * (self.tile_m * hd_slice)
                    block_base = block_idx * (self.tile_m * hd_slice)
                    for c in cutlass.range_constexpr(nchunks):
                        out_c = base + c
                        chunk_base = cutlass.const_expr(
                            (c % chunks_per_slice) * (self.tile_m * chunk_w)
                        )
                        slice_idx = cutlass.const_expr(c // chunks_per_slice)
                        cute.arch.mbarrier_wait(mbar_out_full + out_c, phase)
                        elem_base = slice_idx * slice_stride + block_base + chunk_base
                        # Slot the mma warp wrote this chunk into.
                        slot_base = cutlass.const_expr(
                            (out_c % self.num_out_slots) * self.tmem_out_slot_cols
                        )
                        for s in cutlass.range_constexpr(chunk_w // ncol):
                            # One ncol-column slice per pass: T2R it, then reduce it into
                            # the fp32 gmem accumulator with red.global.add.v4.f32
                            # straight out of the registers.
                            #
                            # This replaced a SMEM staging round trip (a vectorised r2s
                            # into a staging buffer, two named barriers around it, one
                            # elected thread issuing a 32KB cp.reduce.async.bulk.add.f32).
                            # MEASURED split of the drain before that change:
                            #   full 15.0ms, no_reduce 9.65ms, none 6.68ms
                            # i.e. the gmem reduce was 5.35ms and the on-chip
                            # T2R + staging was 2.97ms -- and at the one staging slot the
                            # SMEM budget allowed at ncol=64 that staging was fully
                            # serialised: wait for every outstanding reduce to have read
                            # the slot, barrier, fill, fence, barrier, issue. Per m tile
                            # that is 24 slices x 2 whole-warpgroup barriers. DSA drains
                            # its dKV the same way this does now (dsa_bwd_sm100.py's
                            # scatter_dkv_atomic: float4 atomics from registers, no
                            # staging).
                            #
                            # Byte layout is preserved exactly, which is what keeps
                            # FlashAttentionBackwardPostprocess and _unblock_accum valid.
                            # The old path was: tiled_copy_1d(f32, 128 threads, 4 elems)
                            # put thread t's fragment element r*4+v at staging position
                            # r*512 + t*4 + v, and the bulk copy moved the buffer to gmem
                            # in order. So element r*4+v belongs at gmem offset
                            # r*(num_drain_threads*4) + t*4 + v -- which is what the 16
                            # vector atomics below write. Each is 16B aligned (t*4 f32)
                            # and a warp's 32 lanes cover 512 contiguous bytes.
                            frag = cute.make_fragment(shape_out_slice, Float32)
                            tmem_src = cute.make_tensor(
                                out_base_ptr + slot_base + s * ncol, out_slice_layout
                            )
                            cute.copy(
                                thr_t2r_out, thr_t2r_out.partition_S(tmem_src), frag
                            )
                            cute.arch.fence_view_async_tmem_load()
                            gbase = maccum.iterator + (
                                elem_base + s * (self.tile_m * ncol) + drain_tidx * 4
                            )
                            for r in cutlass.range_constexpr(flen_slice // 4):
                                copy_utils.atomic_add_fp32x4(
                                    frag[r * 4 + 0],
                                    frag[r * 4 + 1],
                                    frag[r * 4 + 2],
                                    frag[r * 4 + 3],
                                    gbase + r * (num_drain_threads * 4),
                                )
                        cute.arch.mbarrier_arrive(mbar_out_empty + out_c)
                        cute.arch.mbarrier_arrive(in_bar + c)

                phase ^= 1

            # Nothing to drain at the end any more: red.global.add.v4.f32 is a fire-and
            # -forget reduction with no bulk groups and no SMEM source to protect.
            cute.arch.mbarrier_arrive(mbar_tmem_dealloc)

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)


@functools.lru_cache(maxsize=None)
def bigd_host_config(head_dim: int, head_dim_v: int, cta_group: int = 1):
    """Return the BigD configuration values required by ``interface.py``.

    Args:
        head_dim: Query/key head dimension.
        head_dim_v: Value head dimension.
        cta_group: Number of CTAs participating in each MMA.

    Returns:
        Tile sizes and accumulator slice widths used by host-side allocation and
        postprocessing.

    tile_m / tile_n have to match the interface's m_block_size / n_block_size (they
    size the accumulators and drive the postprocess grid), and the accumulator slice
    widths drive the postprocess loop, so both are read from the kernel object rather
    than duplicated.
    """
    kernel = FlashAttentionBackwardSm100BigD.from_shape(
        head_dim, head_dim_v, cta_group=cta_group
    )
    return {
        "tile_m": kernel.tile_m,
        "tile_n": kernel.tile_n,
        "accum_slice_d": kernel.accum_slice_d,
        "accum_slice_dv": kernel.accum_slice_dv,
    }
