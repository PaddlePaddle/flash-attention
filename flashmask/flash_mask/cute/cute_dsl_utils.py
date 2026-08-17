# Copyright (c) 2025, Tri Dao.

import os
import pathlib
from typing import Tuple
from functools import partial, lru_cache
from dataclasses import dataclass, fields

import paddle

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import JitArgument
from cutlass.cutlass_dsl import NumericMeta
from typing import Optional

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


def _dsl_version_tuple():
    """Leading numeric components of cutlass.__version__, e.g. "4.6.0.dev0" -> (4, 6, 0)."""
    parts = []
    for tok in str(getattr(cutlass, "__version__", "")).split("."):
        if not tok.isdigit():
            break
        parts.append(int(tok))
    return tuple(parts)


def _patch_dsl_user_op_frame_walk():
    """Skip the per-op Python stack walk in cutlass-dsl 4.5.0 - 4.6.0.

    `dsl_user_op` calls `_find_user_frame` on every MLIR op it builds so that
    locations point at user code instead of cutlass internals. It is meant to be
    gated by `_ENABLE_FRAME_FILTERING` (driven by the `generate_line_info`
    compile option, which defaults to False), but the wrapper never reads that
    flag, so the walk always runs. The frame it lands on lives in a large user
    file (flash_fwd_sm100.py is ~256KB), which makes the following
    `inspect.getframeinfo()` far more expensive and roughly doubles JIT trace
    time. Upstream wires the flag up in 4.6.1, so only patch the affected range.

    Replacing the walk with the identity gives the immediate caller frame, i.e.
    what 4.4.x used. Locations then point into cutlass internals, which only
    affects error messages and IR dumps.
    """
    if not ((4, 5, 0) <= _dsl_version_tuple() < (4, 6, 1)):
        return
    from cutlass.base_dsl._mlir_helpers import op as _op

    if getattr(_op, "_find_user_frame", None) is None:
        return
    # Resolved via LOAD_GLOBAL inside the wrapper, so this also affects the ops
    # that were already decorated at import time.
    _op._find_user_frame = lambda start_frame: start_frame


_patch_dsl_user_op_frame_walk()


paddle2cute_dtype_map = {
    paddle.float16: cutlass.Float16,
    paddle.bfloat16: cutlass.BFloat16,
    paddle.float32: cutlass.Float32,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: paddle.device = None) -> Tuple[int, int]:
    return paddle.cuda.get_device_capability(device)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ArgumentsBase(JitArgument):
    def __c_pointers__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        c_ptrs = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        types, self._values_pos = [], []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__get_mlir_types__"):
                obj_types = obj.__get_mlir_types__()
                types.extend(obj_types)
                self._values_pos.append(len(obj_types))
            else:
                self._values_pos.append(0)
        return types

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


def load_cubin_module_data_patched(cubin_data, filepath):
    pathlib.Path(filepath).write_bytes(cubin_data)
    return load_cubin_module_data_og(cubin_data)


def cute_compile_patched(*args, **kwargs):
    """A patched version of cute.compile that dump the SASS to a file if CUTE_CUBIN_PATH is set."""
    cubin_path = os.getenv("CUTE_CUBIN_PATH", None)
    if cubin_path is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = partial(
            load_cubin_module_data_patched, filepath=cubin_path
        )
    output = cute_compile_og(*args, **kwargs)
    if cubin_path is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = load_cubin_module_data_og
        if extract is not None:
            sass = extract(cubin_path, None)
            pathlib.Path(cubin_path).with_suffix(".annotated.sass").write_text(sass)
    return output


def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
    """Build a fake cute tensor with sym_int strides. Used to compile kernels
    without allocating real GPU tensors."""
    if dtype is None:
        return None
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )
