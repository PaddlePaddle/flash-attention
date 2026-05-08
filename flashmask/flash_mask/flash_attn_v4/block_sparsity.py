# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""
Framework-neutral block-sparsity data structures.

BlockSparseTensors (cute.Tensor based) is defined here for use by root-level
CuTe DSL kernel files (e.g. flash_fwd.py, block_sparse_utils.py).

Framework-specific helpers are re-exported from the active backend's
block_sparsity module.
"""

from typing import NamedTuple

import cutlass.cute as cute

try:
    from flash_mask._backend import BACKEND as _backend_name
except ImportError:
    _backend_name = 'paddle'


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: cute.Tensor | None = None
    full_block_idx: cute.Tensor | None = None
    cu_total_m_blocks: cute.Tensor | None = None
    cu_block_idx_offsets: cute.Tensor | None = None
    dq_write_order: cute.Tensor | None = None
    dq_write_order_full: cute.Tensor | None = None

    def __new_from_mlir_values__(self, values):
        new_fields = []
        idx = 0
        for original in self:
            if original is None:
                new_fields.append(None)
            else:
                new_fields.append(values[idx])
                idx += 1
        return BlockSparseTensors(*new_fields)


if _backend_name == 'torch':
    from flash_mask.flash_attn_v4.torch.block_sparsity import (  # noqa: F401
        BlockSparseTensorsTorch,
        ceildiv,
        get_sparse_q_block_size,
        get_block_sparse_expected_shapes,
        get_block_sparse_expected_shapes_bwd,
        infer_block_sparse_expected_shapes,
        normalize_block_sparse_tensors,
        normalize_block_sparse_config,
        normalize_block_sparse_config_bwd,
        is_block_sparsity_enabled,
        get_block_sparse_broadcast_pattern,
        to_cute_block_sparse_tensors,
        fast_sampling,
        _expand_sparsity_tensor,
        _check_and_expand_block,
    )
else:
    from flash_mask.flash_attn_v4.paddle.block_sparsity import (  # noqa: F401
        BlockSparseTensorsPaddle,
        ceildiv,
        get_sparse_q_block_size,
        get_block_sparse_expected_shapes,
        get_block_sparse_expected_shapes_bwd,
        infer_block_sparse_expected_shapes,
        normalize_block_sparse_tensors,
        normalize_block_sparse_config,
        normalize_block_sparse_config_bwd,
        is_block_sparsity_enabled,
        get_block_sparse_broadcast_pattern,
        to_cute_block_sparse_tensors,
        fast_sampling,
        _expand_sparsity_tensor,
        _check_and_expand_block,
    )
