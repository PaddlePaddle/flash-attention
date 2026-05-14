# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
"""
Block-sparsity utilities for FlexAttention — Paddle backend.

Mirrors torch/block_sparsity.py but uses paddle.Tensor instead of torch.Tensor.
"""

from typing import Callable, NamedTuple, Tuple

import cutlass.cute as cute
import paddle

from flash_mask.flash_attn_v4.paddle.cute_dsl_utils import get_broadcast_dims, to_cute_tensor

# Re-use the CuTe-tensor based BlockSparseTensors from the root module
from flash_mask.flash_attn_v4.block_sparsity import BlockSparseTensors


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class BlockSparseTensorsPaddle(NamedTuple):
    """Block-sparse tensor container for the Paddle backend."""
    mask_block_cnt: paddle.Tensor
    mask_block_idx: paddle.Tensor
    full_block_cnt: paddle.Tensor | None = None
    full_block_idx: paddle.Tensor | None = None
    cu_total_m_blocks: paddle.Tensor | None = None
    cu_block_idx_offsets: paddle.Tensor | None = None
    block_size: tuple[int, int] | None = None
    dq_write_order: paddle.Tensor | None = None
    dq_write_order_full: paddle.Tensor | None = None
    spt: bool | None = None


def _ordered_to_dense_simple(
    num_blocks: paddle.Tensor,
    indices: paddle.Tensor,
    num_cols: int,
) -> paddle.Tensor:
    """Convert ordered sparse representation to dense binary matrix.

    Args:
        num_blocks: [B, H, num_rows] count of valid entries per row
        indices: [B, H, num_rows, max_entries] column indices (valid entries packed left)
        num_cols: total number of columns

    Returns:
        dense: [B, H, num_rows, num_cols] binary int32 matrix
    """
    B, H, num_rows, max_entries = indices.shape
    dense = paddle.zeros([B, H, num_rows, num_cols + 1], dtype=paddle.int32)
    col_range = paddle.arange(max_entries)
    valid = col_range.reshape([1, 1, 1, max_entries]) < num_blocks.unsqueeze(-1)
    safe_indices = paddle.where(
        valid, indices.cast(paddle.int64), paddle.full_like(indices, num_cols, dtype=paddle.int64)
    )
    ones = paddle.ones_like(safe_indices, dtype=paddle.int32)
    dense = paddle.put_along_axis(dense, safe_indices, ones, axis=-1, reduce="assign")
    return dense[:, :, :, :num_cols]


def compute_dq_write_order(
    fwd_mask_cnt: paddle.Tensor,
    fwd_mask_idx: paddle.Tensor,
    fwd_full_cnt: paddle.Tensor | None,
    fwd_full_idx: paddle.Tensor | None,
    bwd_mask_cnt: paddle.Tensor,
    bwd_mask_idx: paddle.Tensor,
    bwd_full_cnt: paddle.Tensor | None,
    bwd_full_idx: paddle.Tensor | None,
    spt: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor | None]:
    """Compute dQ write-order metadata for deterministic block-sparse backward.

    For each (n_block, i) in the backward iteration, computes the semaphore
    lock value: the rank of n_block in the combined (partial + full) sorted
    contributor list for the target m_block.

    Lock values are assigned in ascending n_block order (or descending if spt=True)
    to guarantee deadlock-freedom with the CTA scheduling order.
    """
    B, H, num_m, max_kv_partial = fwd_mask_idx.shape
    _, _, num_n, max_q_partial = bwd_mask_idx.shape

    has_full = fwd_full_cnt is not None and fwd_full_idx is not None

    dense_partial = _ordered_to_dense_simple(fwd_mask_cnt, fwd_mask_idx, num_n)
    if has_full:
        dense_full = _ordered_to_dense_simple(fwd_full_cnt, fwd_full_idx, num_n)
        dense = paddle.clip(dense_partial + dense_full, max=1)
    else:
        dense = dense_partial

    cumsum = dense.cumsum(axis=-1)
    rank_table = (cumsum - dense).cast(paddle.int32)

    if spt:
        total_per_m = cumsum[:, :, :, -1:]
        rank_table = (total_per_m - 1 - rank_table).cast(paddle.int32)

    def _gather_write_order(bwd_idx, bwd_cnt):
        # Use paddle.take_along_axis equivalent via advanced indexing.
        # rank_table has shape [B, H, num_m, num_n]; we gather along the last two dims:
        #   out[b, h, n, q] = rank_table[b, h, bwd_idx[b, h, n, q], n]
        # Implement via take_along_axis on axis=2 then diagonalize on last axis.
        m_vals = paddle.clip(bwd_idx.cast(paddle.int64), 0, num_m - 1)
        # Expand rank_table on axis=2: [B,H,num_m,num_n] -> gather along num_m
        # Broadcast m_vals from [B,H,num_n,max_q] to take along axis=2:
        # We need to gather rank_table[:,:,m_vals[b,h,n,q],n] -> produce [B,H,num_n,max_q]
        max_q = bwd_idx.shape[-1]
        # index1: [B,H,num_n,max_q] for axis=2 gather; we'll produce tmp[b,h,n,q,n'] = rank_table[b,h,m_vals[b,h,n,q],n']
        # Then pick n' = n.
        m_idx_exp = m_vals.unsqueeze(-1).expand([B, H, num_n, max_q, num_n])
        rank_exp = rank_table.unsqueeze(2).expand([B, H, num_n, num_m, num_n])
        # gather along axis=3 (the num_m axis in rank_exp)
        tmp = paddle.take_along_axis(rank_exp, m_idx_exp, axis=3)
        # tmp shape: [B,H,num_n,max_q,num_n]; select n' = n
        n_range = paddle.arange(num_n).reshape([1, 1, num_n, 1, 1]).expand([B, H, num_n, max_q, 1])
        out = paddle.take_along_axis(tmp, n_range, axis=4).squeeze(-1)
        return out.cast(paddle.int32)

    dq_write_order = _gather_write_order(bwd_mask_idx, bwd_mask_cnt)

    dq_write_order_full = None
    if has_full and bwd_full_cnt is not None and bwd_full_idx is not None:
        dq_write_order_full = _gather_write_order(bwd_full_idx, bwd_full_cnt)

    return dq_write_order, dq_write_order_full


def compute_dq_write_order_from_block_mask(
    block_mask,
    spt: bool = False,
) -> tuple[paddle.Tensor, paddle.Tensor | None]:
    (
        _seq_q,
        _seq_k,
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        *_,
    ) = block_mask.as_tuple()
    return compute_dq_write_order(
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        spt=spt,
    )


def get_sparse_q_block_size(
    tensors: BlockSparseTensorsPaddle | None,
    seqlen_q: int,
) -> int | None:
    """Return the Q sparse block size, or None when sparsity is unset or ambiguous."""
    if tensors is None:
        return None
    if tensors.block_size is not None:
        return tensors.block_size[0]
    num_m_blocks = tensors.mask_block_idx.shape[2]
    min_block_size = ceildiv(seqlen_q, num_m_blocks)
    max_block_size = seqlen_q if num_m_blocks == 1 else (seqlen_q - 1) // (num_m_blocks - 1)
    if min_block_size != max_block_size:
        return None
    return min_block_size


def _expand_sparsity_tensor(
    tensor: paddle.Tensor,
    expected_shape: Tuple[int, ...],
    tensor_name: str,
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> paddle.Tensor:
    """Check if we need to expand the tensor to expected shape, and do so if possible."""
    needs_expand = list(tensor.shape) != list(expected_shape)
    if not needs_expand:
        return tensor
    can_expand = all(map(lambda cur, tgt: cur == tgt or cur == 1, tensor.shape, expected_shape))
    if not can_expand:
        context_clause = f" ({context})" if context else ""
        resolved_hint = hint() if callable(hint) else hint
        hint_clause = f" Hint: {resolved_hint}" if resolved_hint else ""
        raise ValueError(
            f"{tensor_name}{context_clause} with shape {tensor.shape} cannot be expanded to expected shape {expected_shape}."
            f"{hint_clause}"
        )
    return tensor.expand(list(expected_shape))


def _check_and_expand_block(
    name: str,
    cnt: paddle.Tensor | None,
    idx: paddle.Tensor | None,
    expected_count_shape: Tuple[int, ...],
    expected_index_shape: Tuple[int, ...],
    context: str | None,
    hint: str | Callable[[], str] | None,
) -> Tuple[paddle.Tensor | None, paddle.Tensor | None]:
    if (cnt is None) != (idx is None):
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must both be provided or both be None"
        )
    if cnt is None or idx is None:
        return None, None
    if cnt.dtype != paddle.int32 or idx.dtype != paddle.int32:
        raise ValueError(f"{name}_block tensors must have dtype paddle.int32")
    if str(cnt.place) != str(idx.place):
        raise ValueError(f"{name}_block_cnt and {name}_block_idx must be on the same device")
    if not cnt.place.is_gpu_place() or not idx.place.is_gpu_place():
        raise ValueError(f"{name}_block tensors must live on CUDA (GPU)")
    expanded_cnt = _expand_sparsity_tensor(
        cnt, expected_count_shape, f"{name}_block_cnt", context, hint
    )
    # [Note] Allow Compact block sparse indices
    if idx.ndim == 4 and idx.shape[3] <= expected_index_shape[3]:
        expected_index_shape = (*expected_index_shape[:3], idx.shape[3])
    expanded_idx = _expand_sparsity_tensor(
        idx, expected_index_shape, f"{name}_block_idx", context, hint
    )
    return expanded_cnt, expanded_idx


def _check_and_expand_metadata_tensor(
    name: str,
    tensor: paddle.Tensor | None,
    expected_shape: Tuple[int, ...],
    context: str | None,
    hint: str | Callable[[], str] | None,
    place,
) -> paddle.Tensor | None:
    if tensor is None:
        return None
    if tensor.dtype != paddle.int32:
        raise ValueError(f"{name} must have dtype paddle.int32")
    if str(tensor.place) != str(place):
        raise ValueError(f"{name} must be on the same device as block sparse tensors")
    if not tensor.place.is_gpu_place():
        raise ValueError(f"{name} must live on CUDA (GPU)")
    return _expand_sparsity_tensor(tensor, expected_shape, name, context, hint)


def get_block_sparse_expected_shapes(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Return (expected_count_shape, expected_index_shape) for block sparse normalization."""
    m_block_size_effective = q_stage * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, m_block_size_effective)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)
    return expected_count_shape, expected_index_shape


def infer_block_sparse_expected_shapes(
    tensors: BlockSparseTensorsPaddle,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    q_stage: int,
    context: str,
    sparse_block_size_q: int | None = None,
    sparse_block_size_kv: int | None = None,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], int]:
    base_m_block = q_stage * m_block_size
    base_n_block = n_block_size
    if sparse_block_size_kv is None:
        sparse_block_size_kv = base_n_block
    if sparse_block_size_kv != base_n_block:
        raise ValueError(f"Block sparse tensors{context} require BLOCK_SIZE_KV={base_n_block}.")
    if tensors.mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")
    num_m_blocks = tensors.mask_block_idx.shape[2]

    if sparse_block_size_q is None:
        sparse_block_size_q = get_sparse_q_block_size(tensors, seqlen_q)
        if sparse_block_size_q is None and base_m_block != 1:
            raise ValueError(
                f"Block sparse tensors{context} require explicit sparse_block_size[0] "
                f"to disambiguate block size for seqlen_q={seqlen_q} and num_m_blocks={num_m_blocks}."
            )
        if sparse_block_size_q is None:
            sparse_block_size_q = ceildiv(seqlen_q, num_m_blocks)

    if sparse_block_size_q % base_m_block != 0:
        raise ValueError(
            f"Block sparse tensors{context} have block size {sparse_block_size_q}, "
            f"which must be a multiple of {base_m_block}."
        )

    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, sparse_block_size_kv)
    q_subtile_factor = sparse_block_size_q // base_m_block
    expected_count_shape = (batch_size, num_head, expected_m_blocks)
    expected_index_shape = (batch_size, num_head, expected_m_blocks, expected_n_blocks)

    mask_block_cnt = tensors.mask_block_cnt
    mask_block_idx = tensors.mask_block_idx
    if mask_block_cnt is None or mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")
    if mask_block_cnt.ndim != 3 or mask_block_idx.ndim != 4:
        raise ValueError(
            f"Block sparse tensors{context} must have shapes (B, H, M) and (B, H, M, N)."
        )
    for dim_name, cur, tgt in (
        ("batch", mask_block_cnt.shape[0], expected_count_shape[0]),
        ("head", mask_block_cnt.shape[1], expected_count_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1.")
    for dim_name, cur, tgt in (
        ("batch", mask_block_idx.shape[0], expected_index_shape[0]),
        ("head", mask_block_idx.shape[1], expected_index_shape[1]),
    ):
        if cur != tgt and cur != 1:
            raise ValueError(f"Block sparse tensors{context} {dim_name} dim must be {tgt} or 1.")
    if mask_block_cnt.shape[2] != mask_block_idx.shape[2]:
        raise ValueError(f"Block sparse tensors{context} must share the same m-block dimension.")
    if mask_block_idx.shape[3] > expected_n_blocks:
        raise ValueError(
            f"Block sparse tensors{context} n-block dimension must be <= {expected_n_blocks}."
        )
    if expected_m_blocks != num_m_blocks:
        raise ValueError(
            f"Block sparse tensors{context} m-block dimension {num_m_blocks} does not match "
            f"sparse_block_size_q={sparse_block_size_q}. "
            f"Set BlockSparseTensorsPaddle.block_size to match the BlockMask BLOCK_SIZE."
        )
    return expected_count_shape, expected_index_shape, q_subtile_factor


def get_block_sparse_expected_shapes_bwd(
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    m_block_size: int,
    n_block_size: int,
    subtile_factor: int,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    sparse_block_size_q = subtile_factor * m_block_size
    expected_m_blocks = ceildiv(seqlen_q, sparse_block_size_q)
    expected_n_blocks = ceildiv(seqlen_k, n_block_size)
    expected_count_shape = (batch_size, num_head, expected_n_blocks)
    expected_index_shape = (batch_size, num_head, expected_n_blocks, expected_m_blocks)
    return expected_count_shape, expected_index_shape


def normalize_block_sparse_tensors(
    tensors: BlockSparseTensorsPaddle,
    *,
    expected_count_shape: Tuple[int, ...],
    expected_index_shape: Tuple[int, ...],
    context: str | None = None,
    hint: str | Callable[[], str] | None = None,
) -> BlockSparseTensorsPaddle:
    if tensors.mask_block_cnt is None or tensors.mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    mask_cnt, mask_idx = _check_and_expand_block(
        "mask",
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if mask_cnt is None or mask_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    full_cnt, full_idx = _check_and_expand_block(
        "full",
        tensors.full_block_cnt,
        tensors.full_block_idx,
        expected_count_shape,
        expected_index_shape,
        context,
        hint,
    )
    if full_cnt is not None and str(mask_cnt.place) != str(full_cnt.place):
        raise ValueError("All block sparse tensors must be on the same device")

    dq_write_order = _check_and_expand_metadata_tensor(
        "dq_write_order",
        tensors.dq_write_order,
        tuple(mask_idx.shape),
        context,
        hint,
        mask_cnt.place,
    )
    dq_write_order_full = _check_and_expand_metadata_tensor(
        "dq_write_order_full",
        tensors.dq_write_order_full,
        tuple(full_idx.shape) if full_idx is not None else expected_index_shape,
        context,
        hint,
        mask_cnt.place,
    )
    spt = tensors.spt
    if spt is not None and not isinstance(spt, bool):
        raise ValueError("spt must be a bool when provided")
    if spt is not None and dq_write_order is None:
        raise ValueError("spt requires dq_write_order to be provided")

    return BlockSparseTensorsPaddle(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        cu_total_m_blocks=tensors.cu_total_m_blocks,
        cu_block_idx_offsets=tensors.cu_block_idx_offsets,
        block_size=tensors.block_size,
        dq_write_order=dq_write_order,
        dq_write_order_full=dq_write_order_full,
        spt=spt,
    )


def is_block_sparsity_enabled(tensors: BlockSparseTensorsPaddle) -> bool:
    return any(t is not None for t in (tensors.full_block_cnt, tensors.mask_block_cnt))


def get_block_sparse_broadcast_pattern(
    tensors: BlockSparseTensorsPaddle,
) -> Tuple[Tuple[bool, ...], ...] | None:
    if not is_block_sparsity_enabled(tensors):
        return None

    patterns = []
    for tensor in (
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        tensors.full_block_cnt,
        tensors.full_block_idx,
        tensors.dq_write_order,
        tensors.dq_write_order_full,
    ):
        if tensor is not None:
            patterns.append(get_broadcast_dims(tensor))
        else:
            patterns.append(None)
    return tuple(patterns)


def normalize_block_sparse_config(
    tensors: BlockSparseTensorsPaddle,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    q_stage: int,
) -> tuple[BlockSparseTensorsPaddle, Tuple[Tuple[bool, ...], ...] | None, int]:
    """Validate the block-sparse config, infer expected shapes, and normalize.

    Handles both fixed-length (3D `[B, H, M]` / 4D `[B, H, M, N]`) and varlen
    (2D `[H, total_m_blocks]` / `[H, total_n_blocks]`) layouts. Varlen is
    detected by `tensors.cu_total_m_blocks is not None` and forces
    `q_subtile_factor == 1` (TODO: potentially remove this restriction).
    """
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = None, n_block_size
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity requires sparse_block_size[1]={n_block_size} to match tile_n."
        )
    if tensors.cu_total_m_blocks is not None:
        base_m_block = q_stage * m_block_size
        if sparse_block_size_q is not None and sparse_block_size_q != base_m_block:
            raise ValueError(
                f"Varlen block sparsity requires sparse_block_size[0]={base_m_block} "
                f"(= q_stage * tile_m); got {sparse_block_size_q}."
            )
        total_m_blocks = tensors.mask_block_cnt.shape[-1]
        total_n_blocks = tensors.mask_block_idx.shape[-1]
        expected_count_shape = (num_head, total_m_blocks)
        expected_index_shape = (num_head, total_n_blocks)
        q_subtile_factor = 1
    else:
        expected_count_shape, expected_index_shape, q_subtile_factor = (
            infer_block_sparse_expected_shapes(
                tensors,
                batch_size=batch_size,
                num_head=num_head,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                q_stage=q_stage,
                context="forward",
                sparse_block_size_q=sparse_block_size_q,
                sparse_block_size_kv=sparse_block_size_kv,
            )
        )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
    )
    return (
        normalized_tensors,
        get_block_sparse_broadcast_pattern(normalized_tensors),
        q_subtile_factor,
    )


def normalize_block_sparse_config_bwd(
    tensors: BlockSparseTensorsPaddle,
    *,
    batch_size: int,
    num_head: int,
    seqlen_q: int,
    seqlen_k: int,
    block_size: tuple[int, int],
    subtile_factor: int,
) -> tuple[BlockSparseTensorsPaddle, Tuple[Tuple[bool, ...], ...] | None]:
    m_block_size, n_block_size = block_size
    if tensors.block_size is None:
        sparse_block_size_q, sparse_block_size_kv = subtile_factor * m_block_size, n_block_size
    else:
        sparse_block_size_q, sparse_block_size_kv = tensors.block_size
    if sparse_block_size_q != subtile_factor * m_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size_q={subtile_factor * m_block_size} "
            f"for subtile_factor={subtile_factor}."
        )
    if sparse_block_size_kv != n_block_size:
        raise ValueError(
            f"Block sparsity expects sparse_block_size[1]={n_block_size} to match tile_n."
        )
    expected_count_shape, expected_index_shape = get_block_sparse_expected_shapes_bwd(
        batch_size,
        num_head,
        seqlen_q,
        seqlen_k,
        m_block_size,
        n_block_size,
        subtile_factor,
    )
    normalized_tensors = normalize_block_sparse_tensors(
        tensors,
        expected_count_shape=expected_count_shape,
        expected_index_shape=expected_index_shape,
        context="_flash_attn_bwd",
        hint=lambda: (
            f"Backward expects Q-direction block-sparse tensors (q_mask_cnt/q_mask_idx, "
            f"and optionally full_q_cnt/full_q_idx). Regenerate the backward BlockMask with "
            f"BLOCK_SIZE=({subtile_factor * m_block_size}, {n_block_size})."
        ),
    )
    return normalized_tensors, get_block_sparse_broadcast_pattern(normalized_tensors)


def to_cute_block_sparse_tensors(
    tensors: BlockSparseTensorsPaddle, enable_tvm_ffi: bool = True
) -> BlockSparseTensors | None:
    """Convert paddle block sparsity tensors to CuTe tensors, optionally for tvm ffi."""
    if not is_block_sparsity_enabled(tensors):
        return None
    mask_block_cnt_tensor, mask_block_idx_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        for t in (tensors.mask_block_cnt, tensors.mask_block_idx)
    ]
    full_block_cnt_tensor, full_block_idx_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (tensors.full_block_cnt, tensors.full_block_idx)
    ]
    cu_total_m_blocks_tensor, cu_block_idx_offsets_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=0, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (tensors.cu_total_m_blocks, tensors.cu_block_idx_offsets)
    ]
    dq_write_order_tensor, dq_write_order_full_tensor = [
        to_cute_tensor(t, assumed_align=4, leading_dim=-1, enable_tvm_ffi=enable_tvm_ffi)
        if t is not None
        else None
        for t in (tensors.dq_write_order, tensors.dq_write_order_full)
    ]

    return BlockSparseTensors(
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
        full_block_cnt_tensor,
        full_block_idx_tensor,
        cu_total_m_blocks_tensor,
        cu_block_idx_offsets_tensor,
        dq_write_order_tensor,
        dq_write_order_full_tensor,
    )


def fast_sampling(mask_mod):
    """Convenience decorator to mark mask_mod as safe for 5-point fast sampling."""
    mask_mod.use_fast_sampling = True
    return mask_mod
