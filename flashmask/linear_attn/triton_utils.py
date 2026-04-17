# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Adapted for PaddlePaddle

import os
from contextlib import contextmanager

import paddle
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution


@cache
def _is_package_installed(dist_name: str) -> bool:
    try:
        distribution(dist_name)
        return True
    except PackageNotFoundError:
        return False


# Pre-create Paddle triton driver in mixed torch environment
paddle_driver = None
if _is_package_installed("torch"):
    with paddle.use_compat_guard(enable=True, silent=True):
        from triton.runtime.driver import _create_driver
        paddle_driver = _create_driver()


# ---------------------------------------------------------------------------
# Driver probe: captures the active triton driver *during* kernel execution.
# Disabled by default (zero overhead). Tests enable it via enable_driver_probe().
# Set FLA_BENCHMARK=1 to keep probing disabled even if tests try to enable it.
# ---------------------------------------------------------------------------
_driver_probe_enabled: bool = False
_driver_probe_result: str = "not_probed"
_compat_wrapper_fastpath_depth: int = 0


def enable_driver_probe():
    """Enable driver probing during kernel launch (for tests)."""
    global _driver_probe_enabled, _driver_probe_result
    if os.environ.get("FLA_BENCHMARK", "0") == "1":
        return
    _driver_probe_enabled = True
    _driver_probe_result = "not_probed"


def disable_driver_probe():
    """Disable driver probing (restore zero overhead)."""
    global _driver_probe_enabled
    _driver_probe_enabled = False


def get_driver_probe_result() -> str:
    """Return the driver framework detected during the last kernel launch."""
    return _driver_probe_result


def _detect_driver_framework(active_driver) -> str:
    """Identify the framework behind a triton driver object."""
    fn = active_driver.get_current_stream
    # Check __module__ first (most reliable)
    mod = getattr(fn, '__module__', '') or ''
    if 'paddle' in mod:
        return 'paddle'
    if 'torch' in mod:
        return 'torch'
    # Fallback: check string representation
    fn_str = str(fn)
    if 'paddle' in fn_str or '_get_current_raw_stream' in fn_str:
        return 'paddle'
    if 'torch' in fn_str or '_cuda_getCurrentRawStream' in fn_str:
        return 'torch'
    return 'unknown'


def _probe_active_driver():
    """Snapshot the active triton driver framework (called inside swap guard)."""
    global _driver_probe_result
    try:
        from triton.runtime.driver import driver
        _driver_probe_result = _detect_driver_framework(driver.active)
    except Exception as e:
        _driver_probe_result = f'error({e})'


def _wrap_probe_only(fn):
    def wrapped_fn(*args, **kwargs):
        if _driver_probe_enabled:
            _probe_active_driver()
        return fn(*args, **kwargs)

    return wrapped_fn


def swap_driver_guard(fn):
    """Temporarily swap triton's active driver to Paddle driver."""
    from triton.runtime.driver import driver

    def wrapped_fn(*args, **kwargs):
        if paddle_driver is None or driver.active is paddle_driver:
            if _driver_probe_enabled:
                _probe_active_driver()
            return fn(*args, **kwargs)
        driver.set_active(paddle_driver)
        try:
            if _driver_probe_enabled:
                _probe_active_driver()
            return fn(*args, **kwargs)
        finally:
            driver.reset_active()

    return wrapped_fn


def _should_bypass_compat_kernel_wrapper() -> bool:
    if _compat_wrapper_fastpath_depth <= 0 or paddle_driver is None:
        return False
    try:
        from triton.runtime.driver import driver
    except Exception:
        return False
    return driver.active is paddle_driver


@contextmanager
def compat_kernel_wrapper_fastpath():
    """Allow compat-wrapped kernels to skip re-wrapping when Paddle driver is already active."""
    global _compat_wrapper_fastpath_depth
    _compat_wrapper_fastpath_depth += 1
    try:
        yield
    finally:
        _compat_wrapper_fastpath_depth -= 1


@contextmanager
def activate_paddle_driver():
    """Activate the Paddle Triton driver for a wider Python region when available."""
    if paddle_driver is None:
        yield
        return

    from triton.runtime.driver import driver

    if driver.active is paddle_driver:
        yield
        return

    driver.set_active(paddle_driver)
    try:
        yield
    finally:
        driver.reset_active()


def enable_compat_on_triton_kernel(triton_kernel):
    """
    Triton kernel compat decorator (ref: FastDeploy PR#6897).

    - No torch env: return original kernel (zero overhead, relies on global enable_compat)
    - Has torch env: wrap kernel to use Paddle driver on launch

    Usage:
        @enable_compat_on_triton_kernel  # outermost
        @triton.autotune(...)            # optional
        @triton.jit
        def my_kernel(...):
            ...
    """
    if not _is_package_installed("torch"):
        return triton_kernel

    class WrappedTritonKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, index):
            if _should_bypass_compat_kernel_wrapper():
                launcher = self.kernel[index]
                if _driver_probe_enabled:
                    return _wrap_probe_only(launcher)
                return launcher
            return swap_driver_guard(self.kernel[index])

        def __getattr__(self, name):
            return getattr(self.kernel, name)

    return WrappedTritonKernel(triton_kernel)
