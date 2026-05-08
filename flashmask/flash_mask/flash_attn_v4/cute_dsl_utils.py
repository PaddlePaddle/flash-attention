# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
"""
Framework-neutral CuTe DSL utilities.

Re-exports everything from the active backend's cute_dsl_utils so that
shared modules (flash_fwd.py, block_sparsity.py, etc.) that do
  from flash_mask.flash_attn_v4.cute_dsl_utils import ...
continue to work without modification.
"""

try:
    from flash_mask._backend import BACKEND as _backend_name
except ImportError:
    _backend_name = 'paddle'

if _backend_name == 'torch':
    from flash_mask.flash_attn_v4.torch.cute_dsl_utils import (  # noqa: F401
        StaticTypes,
        get_max_active_clusters,
        get_device_capacity,
        assume_strides_aligned,
        assume_tensor_aligned,
        to_cute_tensor,
        to_cute_aux_tensor,
        get_aux_tensor_metadata,
        get_broadcast_dims,
        ParamsBase,
        ArgumentsBase,
        make_fake_tensor,
        torch2cute_dtype_map,
    )
else:
    from flash_mask.flash_attn_v4.paddle.cute_dsl_utils import (  # noqa: F401
        get_max_active_clusters,
        get_device_capacity,
        assume_strides_aligned,
        assume_tensor_aligned,
        to_cute_tensor,
        to_cute_aux_tensor,
        get_aux_tensor_metadata,
        get_broadcast_dims,
        ParamsBase,
        ArgumentsBase,
        make_fake_tensor,
    )
