# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.


import cutlass
import cutlass.cute as cute
from typing import Optional, Type, Tuple, Callable, Sequence

from cutlass import Int32, Int64, const_expr


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def _dsl_narrows_slice_index() -> bool:
    """Whether the installed DSL lowers slice index math in 32 bits.

    4.5.0 loads the i64 strides of a dynamic layout with `ld.param.b32` and multiplies the
    coordinates with `mul.lo.s32` before sign-extending; 4.4.1 widens first (`cvt.u64.u32` +
    `mul.lo.s64`) and needs nothing from us.
    """
    try:
        version = tuple(int(p) for p in cutlass.__version__.split(".")[:3])
    except (AttributeError, ValueError):
        return True  # unknown build: prefer the explicit 64-bit path over a silent wrap
    return version >= (4, 5, 0)


def _needs_64b_offset(a: cute.Tensor, coord) -> bool:
    """Whether `a[coord]` is a gmem slice whose offset can overflow 32 bits.

    Everything we are not sure about is left to the DSL: element access, TMA coordinate
    tensors (no pointer iterator), nested modes, swizzled pointers (`make_ptr` cannot carry a
    swizzle), sub-byte types, and layouts whose strides are all static -- those are folded at
    compile time and stay correct in 4.5.0. Bailing out means the DSL's 32-bit path is used, so
    a >2**31-element tensor of a shape rejected here still wraps; add a case rather than assume
    it is handled.
    """
    if type(coord) is not tuple or not cute.has_underscore(coord):
        return False
    it = getattr(a, "iterator", None)
    if not all(hasattr(it, attr) for attr in ("toint", "memspace", "alignment")):
        return False
    if it.memspace not in (cute.AddressSpace.gmem, cute.AddressSpace.generic):
        return False
    if getattr(it.type, "is_swizzled", False):
        return False
    if a.element_type.width % 8 != 0:
        return False
    stride = a.layout.stride
    if len(coord) != len(stride):
        return False
    has_dynamic = False
    for c, s in zip(coord, stride):
        if c is None:
            continue
        if isinstance(c, tuple) or isinstance(s, tuple):
            return False
        if not isinstance(s, int):
            has_dynamic = True
    return has_dynamic


def install_slice_64b() -> None:
    """Route gmem slices through `slice_64b` on the DSL versions that narrow the index math.

    Slicing is where a (batch, head) coordinate meets a dynamic i64 stride, and the `t[b, h, None]`
    subscript syntax is the only spelling used for it here, so patching `_Tensor.__getitem__` keeps
    every call site -- present and future -- correct without a special API to remember. Note that
    `cute.slice_` and `cute.local_tile` reach `_cute_ir` directly and are *not* covered; they are
    only ever applied to layouts or to already-sliced tensors whose offsets stay small.
    No-op on 4.4.1, and idempotent.
    """
    from cutlass.cute.tensor import _Tensor  # DSL-private, but the only slicing entry point

    if not _dsl_narrows_slice_index() or getattr(_Tensor.__getitem__, "_slice_64b", False):
        return

    orig_getitem = _Tensor.__getitem__

    def __getitem__(self, coord, **kwargs):
        if _needs_64b_offset(self, coord):
            return slice_64b(self, coord, loc=kwargs.get("loc"), ip=kwargs.get("ip"))
        return orig_getitem(self, coord, **kwargs)

    __getitem__._slice_64b = True
    _Tensor.__getitem__ = __getitem__


def slice_64b(a: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Tensor:
    """`a[coord]` with the linear offset computed in 64 bits.

    nvidia-cutlass-dsl 4.5.0 narrows crd2idx to 32 bits (4.4.1 kept it in 64), so slicing
    a tensor that holds more than 2**31 elements -- dQaccum once batch*nheads*seqlen_q_
    rounded*head_dim_rounded crosses that -- wraps to a negative address and faults. Fold
    the integer coordinates into an Int64 byte offset ourselves; None keeps a mode, just
    like Tensor.__getitem__. `utils.elem_pointer_i64` does the same widening for a full
    coordinate, i.e. when no mode is kept.

    `install_slice_64b` makes plain slicing take this path, so calling it directly is only
    needed to state the intent explicitly.
    """
    offset = Int64(0)
    keep = []
    for i, c in enumerate(coord):
        if const_expr(c is None):
            keep.append(i)
        else:
            stride = a.layout.stride[i]
            assert not isinstance(stride, tuple), "cannot fold a nested mode into an offset"
            offset += Int64(c) * Int64(stride)
    # HACK: as in utils.elem_pointer_i64, we assume the offset does not change the alignment
    ptr = cute.make_ptr(
        a.element_type,
        a.iterator.toint() + offset * (a.element_type.width // 8),
        a.iterator.memspace,
        assumed_align=a.iterator.alignment,
        loc=loc,
        ip=ip,
    )
    return cute.make_tensor(ptr, cute.select(a.layout, mode=keep, loc=loc, ip=ip), loc=loc, ip=ip)



def convert_layout_acc_mn(acc_layout: cute.Layout, transpose: bool = False) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),  # MMA_N
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),  # MMA_M
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),  # MMA_N
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


def reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout, transpose=transpose))


@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For back to back gemm, convert layout of acc0 to gemm 1 accept layout.
    # For Sm80, as the mma instruction shape is 16x8x16, we need to convert from (4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
    # If N / 8 is odd, we'll convert to ((2, 2, 1), MMA_M, N / 8, MMA_N).
    # TODO: Sm90 FP8
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):  # Sm90
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        l = cute.logical_divide(
            acc_layout, ((None, None, div), None, None)
        )  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:  # Sm80
        # (4, MMA_M, MMA_N) -> (4, MMA_M, (2, MMA_N / 2))
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0], l.shape[2][0]),
                l.shape[1],
                l.shape[2][1],
            ),
            stride=(
                (l.stride[0], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))



def mma_partition_C_vec(
    sVec: cute.Tensor, thr_mma: cute.core.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = make_acc_tensor_mn_view(thr_mma.partition_C(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]

