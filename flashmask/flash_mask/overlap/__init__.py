"""FM-4 Overlap package.

Importing this package never fails, even when the native bridge
(libfm4_overlap.so) was not built. ``overlap_runtime`` itself is import-safe;
the .so is only loaded lazily on first use via ``overlap_runtime._load()``.
"""

from . import overlap_runtime  # noqa: F401

__all__ = ["overlap_runtime"]
