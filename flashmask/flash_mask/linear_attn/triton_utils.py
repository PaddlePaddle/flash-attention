# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Original portions of this file are licensed under the MIT License.
# See the LICENSE-MIT file or the original project license for details.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
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
# Adapted for PaddlePaddle

import os
from contextlib import contextmanager


def _get_triton_python_include():
    """Return the Python include path Triton will pass to the C compiler."""
    import sysconfig

    scheme = sysconfig.get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    return sysconfig.get_paths(scheme=scheme).get("include")


def _ensure_python_include_path():
    """Ensure the Python include path used by Triton's C compiler is valid."""
    import sys
    import sysconfig

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    include_candidates = [
        _get_triton_python_include(),
        sysconfig.get_path("include"),
        sysconfig.get_path("platinclude"),
        sysconfig.get_config_var("INCLUDEPY"),
        sysconfig.get_config_var("CONFINCLUDEPY"),
        os.path.join(sys.prefix, "include", version),
        os.path.join(sys.base_prefix, "include", version),
        f"/usr/local/include/{version}",
        f"/usr/include/{version}",
    ]

    for include_dir in include_candidates:
        if include_dir and os.path.isfile(os.path.join(include_dir, "Python.h")):
            for env_name in ("C_INCLUDE_PATH", "CPATH"):
                existing = os.environ.get(env_name, "")
                existing_paths = existing.split(os.pathsep) if existing else []
                if include_dir not in existing_paths:
                    os.environ[env_name] = os.pathsep.join(
                        [include_dir, *existing_paths]
                    )
            return


_ensure_python_include_path()

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


_HAS_TORCH = _is_package_installed("torch")


# Patch triton nvidia backend helpers to not depend on torch.
# Triton 3.6.0 uses torch.cuda for driver activation and benchmark timing.
# Use CUDA/Paddle directly so linear attention works in Paddle-only envs.
import ctypes as _ctypes


class _PaddleCudaInterface:
    Event = staticmethod(paddle.device.cuda.Event)

    @staticmethod
    def synchronize():
        paddle.device.synchronize()

    @staticmethod
    def current_device():
        device = paddle.device.get_device()
        if isinstance(device, str) and ':' in device:
            return int(device.rsplit(':', 1)[1])
        return 0


_PADDLE_CUDA_INTERFACE = _PaddleCudaInterface()


def _triton_cuda_is_active():
    try:
        return _ctypes.CDLL("libcuda.so.1").cuInit(0) == 0
    except Exception:
        return False


def _triton_get_device_interface(self):
    return _PADDLE_CUDA_INTERFACE


def _triton_get_active_paddle_device(self):
    return paddle.CUDAPlace(self.get_current_device())


def _triton_get_empty_cache_for_benchmark(self):
    return paddle.empty([64 * 1024 * 1024], dtype='int32')


def _triton_clear_cache(self, cache):
    if cache is not None:
        cache.zero_()


try:
    from triton.backends.nvidia import driver as _triton_nvidia_driver

    _triton_nvidia_driver.CudaDriver.is_active = staticmethod(_triton_cuda_is_active)
    if not _HAS_TORCH:
        _triton_nvidia_driver.CudaDriver.get_device_interface = _triton_get_device_interface
        _triton_nvidia_driver.CudaDriver.get_active_torch_device = _triton_get_active_paddle_device
        _triton_nvidia_driver.CudaDriver.get_empty_cache_for_benchmark = _triton_get_empty_cache_for_benchmark
        _triton_nvidia_driver.CudaDriver.clear_cache = _triton_clear_cache
except Exception:
    pass  # triton not installed or no nvidia backend, skip


# Pre-create Paddle triton driver (works with or without torch)
paddle_driver = None
try:
    with paddle.use_compat_guard(enable=True, silent=True):
        from triton.runtime.driver import _create_driver, driver as _triton_driver

        paddle_driver = _create_driver()
        _triton_driver._default = paddle_driver  # cache to global driver singleton
except Exception:
    pass


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
    if paddle_driver is None:
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
