# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""Flash Attention CUTE (CUDA Template Engine) implementation.

NOTE: To avoid circular imports (paddle.interface -> cache_utils -> this __init__),
we use lazy imports via __getattr__ instead of importing symbols at module level.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]

# Resolve backend from compile-time _backend.py (set by setup.py).
# Default to 'paddle' (matching setup.py default) if _backend.py is absent.
try:
    from flash_mask._backend import BACKEND as _backend_name
except ImportError:
    _backend_name = 'paddle'

_lazy_cache = {}

def __getattr__(name):
    if name in ("flash_attn_func", "flash_attn_varlen_func"):
        if name not in _lazy_cache:
            if _backend_name == 'torch':
                from flash_mask.flash_attn_v4.torch.interface import (
                    flash_attn_func as _func,
                    flash_attn_varlen_func as _varlen_func,
                )
            else:
                from flash_mask.flash_attn_v4.paddle.interface import (
                    flash_attn_func as _func,
                    flash_attn_varlen_func as _varlen_func,
                )
            _lazy_cache["flash_attn_func"] = _func
            _lazy_cache["flash_attn_varlen_func"] = _varlen_func
        return _lazy_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
