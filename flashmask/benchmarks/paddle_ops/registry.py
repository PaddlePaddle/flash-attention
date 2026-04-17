from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import paddle
import paddle.nn.functional as F

logger = logging.getLogger(__name__)


def shape_BTHD(B, T, H, D, **kw):
    return (B, T, H, D)


def shape_BTH(B, T, H, D, **kw):
    return (B, T, H)


logsigmoid = F.log_sigmoid


def sigmoid_transform(t):
    return t.sigmoid()


@dataclass
class TensorSpec:
    shape_fn: Callable
    requires_grad: bool = True
    dtype: Any = 'default'
    transform: Callable | None = None


@dataclass
class OpConfig:
    name: str
    import_path: str
    inputs: dict[str, TensorSpec]
    func_name: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    output_is_tuple: bool = True
    skip_backward: bool = False
    category: str = ''


_REGISTRY: dict[str, OpConfig] = {}


def register_op(config: OpConfig) -> None:
    _REGISTRY[config.name] = config


SHAPE_CONFIGS = {
    'B1_T8192_H96_D128': {'B': 1, 'T': 8192, 'H': 96, 'D': 128},
    'B2_T16384_H16_D128': {'B': 2, 'T': 16384, 'H': 16, 'D': 128},
    'B4_T2048_H16_D128': {'B': 4, 'T': 2048, 'H': 16, 'D': 128},
    'B4_T4096_H64_D128': {'B': 4, 'T': 4096, 'H': 64, 'D': 128},
    'B8_T2048_H32_D256': {'B': 8, 'T': 2048, 'H': 32, 'D': 256},
    'B8_T1024_H8_D64': {'B': 8, 'T': 1024, 'H': 8, 'D': 64},
}


def get_op(name: str) -> OpConfig:
    if name not in _REGISTRY:
        raise KeyError(f"Op '{name}' not registered. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_ops() -> list[str]:
    return sorted(_REGISTRY.keys())


def _resolve_dtype(dtype):
    if dtype == 'default':
        return paddle.bfloat16
    if dtype == 'float32':
        return paddle.float32
    if dtype == 'int64':
        return paddle.int64
    return dtype


def _set_device(device: str | None):
    if device is None:
        return
    current = paddle.get_device()
    if current != device:
        paddle.device.set_device(device)


def generate_inputs(
    config: OpConfig,
    B: int,
    T: int,
    H: int,
    D: int,
    dtype=paddle.bfloat16,
    device: str | None = None,
) -> dict[str, paddle.Tensor]:
    _set_device(device)
    inputs: dict[str, paddle.Tensor] = {}
    for param_name, spec in config.inputs.items():
        shape = spec.shape_fn(B, T, H, D)
        tensor_dtype = dtype if spec.dtype == 'default' else _resolve_dtype(spec.dtype)
        if tensor_dtype == paddle.int64:
            tensor = paddle.randint(0, 10, shape=shape, dtype=tensor_dtype)
        else:
            tensor = paddle.randn(shape, dtype=tensor_dtype)
        if spec.transform is not None:
            tensor = spec.transform(tensor)
        if spec.requires_grad and paddle.is_floating_point(tensor):
            tensor.stop_gradient = False
        inputs[param_name] = tensor
    return inputs


_simple_qkv = {
    'q': TensorSpec(shape_BTHD),
    'k': TensorSpec(shape_BTHD),
    'v': TensorSpec(shape_BTHD),
}

register_op(OpConfig(
    name='chunk_gdn',
    import_path='linear_attn.ops.gated_delta_rule',
    func_name='chunk_gated_delta_rule',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    category='gate_beta',
))

register_op(OpConfig(
    name='chunk_kda',
    import_path='linear_attn.ops.kda',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTHD, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True, 'safe_gate': True, 'lower_bound': -5},
    category='gate_beta',
))

register_op(OpConfig(
    name='recurrent_gdn',
    import_path='linear_attn.ops.gated_delta_rule',
    func_name='fused_recurrent_gated_delta_rule',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTH, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True},
    skip_backward=True,
    category='gate_beta',
))

register_op(OpConfig(
    name='recurrent_kda',
    import_path='linear_attn.ops.kda',
    func_name='fused_recurrent_kda',
    inputs={
        **_simple_qkv,
        'g': TensorSpec(shape_BTHD, transform=logsigmoid),
        'beta': TensorSpec(shape_BTH, transform=sigmoid_transform),
    },
    extra_kwargs={'use_qk_l2norm_in_kernel': True, 'safe_gate': True, 'lower_bound': -5},
    skip_backward=True,
    category='gate_beta',
))
