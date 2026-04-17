# -*- coding: utf-8 -*-
# Test fixtures for Paddle migration tests

import logging
import os
import warnings
import importlib

import paddle
import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# assert_close: matches torch fla.utils.assert_close semantics
# ---------------------------------------------------------------------------

def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().pow(2).mean().sqrt().item()
    base = x.detach().flatten().pow(2).mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"
    logger.info(msg)
    if abs_atol <= err_atol:
        return
    assert not paddle.isnan(ref).any(), f"{prefix}: NaN detected in ref"
    assert not paddle.isnan(tri).any(), f"{prefix}: NaN detected in tri"
    if warning:
        if error_rate > ratio:
            warnings.warn(msg)
    else:
        assert error_rate < ratio, msg


class FrameworkTracker:
    """Track the framework backend of tensors and triton driver at runtime.

    Driver detection modes:
      - probe (default in tests): reads the result captured *during* kernel
        execution by the swap_driver_guard probe hook. This is accurate even
        in mixed torch+paddle environments.
      - snapshot: inspects the triton active driver at call time (outside
        kernel execution). Fast but shows the *default* driver, which is
        torch when torch is installed.

    Set ``FLA_BENCHMARK=1`` to disable all probing overhead. In that mode,
    ``detect_triton_driver()`` falls back to snapshot and the report omits
    the Triton Driver line entirely.
    """

    def __init__(self):
        self._benchmark = os.environ.get("FLA_BENCHMARK", "0") == "1"

    # ------ tensor framework ------

    @staticmethod
    def detect_tensor_framework(tensor) -> str:
        """Detect which framework a tensor belongs to."""
        module = type(tensor).__module__
        if 'paddle' in module:
            return 'paddle'
        elif 'torch' in module:
            return 'torch'
        return f'unknown({module})'

    # ------ triton driver ------

    def detect_triton_driver(self) -> str:
        """Return the triton driver detected during the last kernel launch.

        Uses the probe result captured inside swap_driver_guard when probing
        is enabled; falls back to a snapshot of the current active driver
        otherwise.
        """
        if self._benchmark:
            return self._detect_triton_driver_snapshot()
        from linear_attn.triton_utils import get_driver_probe_result
        result = get_driver_probe_result()
        if result == "not_probed":
            return self._detect_triton_driver_snapshot()
        return result

    @staticmethod
    def _detect_triton_driver_snapshot() -> str:
        """Inspect the current triton active driver (outside kernel execution)."""
        try:
            from triton.runtime.driver import driver
            from linear_attn.triton_utils import _detect_driver_framework
            return _detect_driver_framework(driver.active)
        except Exception as e:
            return f'error({e})'

    # ------ autograd ------

    @staticmethod
    def detect_autograd_framework(tensor) -> str:
        """Detect the autograd backend of a tensor."""
        import paddle
        if isinstance(tensor, paddle.Tensor):
            if not tensor.stop_gradient:
                return 'paddle'
            return 'paddle (no_grad)'
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                if tensor.requires_grad:
                    return 'torch'
                return 'torch (no_grad)'
        except ImportError:
            pass
        return 'unknown'

    # ------ report ------

    def report(self, tensors: dict, label: str = ""):
        """Generate a framework detection report."""
        lines = []
        if label:
            lines.append(f"\n{'='*60}")
            lines.append(f"  Framework Detection Report: {label}")
            lines.append(f"{'='*60}")

        if not self._benchmark:
            lines.append(f"  Triton Driver: {self.detect_triton_driver()}")
            lines.append(f"  {'─'*56}")

        for name, tensor in tensors.items():
            fw = self.detect_tensor_framework(tensor)
            ag = self.detect_autograd_framework(tensor)
            lines.append(f"  {name:20s} | framework: {fw:8s} | autograd: {ag}")

        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)


@pytest.fixture(autouse=True)
def _driver_probe_lifecycle():
    """Enable driver probing before each test, disable after."""
    if os.environ.get("FLA_BENCHMARK", "0") == "1":
        yield
        return
    from linear_attn.triton_utils import enable_driver_probe, disable_driver_probe
    enable_driver_probe()
    yield
    disable_driver_probe()


@pytest.fixture(autouse=True)
def _linear_attn_cache_isolation():
    modules = []
    for name in (
        'linear_attn.ops.common.chunk_o',
        'linear_attn.ops.kda.wy_fast',
    ):
        try:
            modules.append(importlib.import_module(name))
        except Exception:
            continue
    for mod in modules:
        for attr in ('_const_tiling', '_chunk_o_launch_meta', '_wy_tiling', '_wy_launch_meta'):
            fn = getattr(mod, attr, None)
            if fn is not None and hasattr(fn, 'cache_clear'):
                fn.cache_clear()
    yield
    for mod in modules:
        for attr in ('_const_tiling', '_chunk_o_launch_meta', '_wy_tiling', '_wy_launch_meta'):
            fn = getattr(mod, attr, None)
            if fn is not None and hasattr(fn, 'cache_clear'):
                fn.cache_clear()


@pytest.fixture
def framework_tracker():
    return FrameworkTracker()
