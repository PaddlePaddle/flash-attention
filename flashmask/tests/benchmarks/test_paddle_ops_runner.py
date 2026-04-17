# -*- coding: utf-8 -*-

import importlib
from types import SimpleNamespace

import paddle
import torch


def _load_torch_benchmark_run_module():
    return importlib.import_module('benchmarks.ops.run')


def _load_paddle_benchmark_run_module():
    return importlib.import_module('benchmarks.paddle_ops.run')


def test_paddle_benchmark_registry_registers_gdn_and_kda():
    from benchmarks.paddle_ops.registry import get_op, list_ops

    ops = list_ops()

    assert 'chunk_gdn' in ops
    assert 'recurrent_gdn' in ops
    assert 'chunk_kda' in ops
    assert 'recurrent_kda' in ops
    assert get_op('chunk_gdn').import_path == 'linear_attn.ops.gated_delta_rule'
    assert get_op('recurrent_gdn').func_name == 'fused_recurrent_gated_delta_rule'
    assert get_op('chunk_kda').import_path == 'linear_attn.ops.kda'
    assert get_op('recurrent_kda').func_name == 'fused_recurrent_kda'


def test_paddle_benchmark_registry_generates_expected_tensor_shapes():
    from benchmarks.paddle_ops.registry import generate_inputs, get_op

    gdn_inputs = generate_inputs(get_op('chunk_gdn'), B=2, T=16, H=4, D=32, dtype=paddle.bfloat16, device='gpu')
    kda_inputs = generate_inputs(get_op('chunk_kda'), B=2, T=16, H=4, D=32, dtype=paddle.bfloat16, device='gpu')

    assert list(gdn_inputs['q'].shape) == [2, 16, 4, 32]
    assert list(gdn_inputs['g'].shape) == [2, 16, 4]
    assert list(gdn_inputs['beta'].shape) == [2, 16, 4]
    assert isinstance(gdn_inputs['q'], paddle.Tensor)

    assert list(kda_inputs['q'].shape) == [2, 16, 4, 32]
    assert list(kda_inputs['g'].shape) == [2, 16, 4, 32]
    assert list(kda_inputs['beta'].shape) == [2, 16, 4]
    assert isinstance(kda_inputs['g'], paddle.Tensor)


def test_paddle_benchmark_runner_lists_registered_ops(capsys):
    from benchmarks.paddle_ops import run

    run.main(['--list'])

    captured = capsys.readouterr()
    assert 'chunk_gdn' in captured.out
    assert 'chunk_kda' in captured.out


def test_torch_benchmark_runner_skip_backward_warms_up_forward_only(monkeypatch):
    module = _load_torch_benchmark_run_module()
    import triton

    config = SimpleNamespace(
        name='recurrent_stub',
        import_path='unused',
        inputs={},
        skip_backward=True,
        output_is_tuple=True,
        extra_kwargs={},
        dim_constraints=None,
    )

    monkeypatch.setattr(module, 'get_op', lambda name: config)
    monkeypatch.setattr(module, 'generate_inputs', lambda *args, **kwargs: {'x': torch.ones(1)})
    monkeypatch.setattr(module, '_warmup_autotune', lambda fn, n=None: fn())

    def fake_op(**kwargs):
        return (torch.ones(1),)

    def fake_do_bench(fn, quantiles, **kwargs):
        fn()
        return (1.0, 0.9, 1.1)

    monkeypatch.setattr(module, '_import_op', lambda cfg: fake_op)
    monkeypatch.setattr(triton.testing, 'do_bench', fake_do_bench)

    results = module.benchmark_op('recurrent_stub', {'smoke': {'B': 1, 'T': 2, 'H': 3, 'D': 4}})

    assert [row['mode'] for row in results] == ['fwd']


def test_paddle_benchmark_runner_skip_backward_warms_up_forward_only(monkeypatch):
    module = _load_paddle_benchmark_run_module()
    import triton

    config = module.OpConfig(
        name='recurrent_stub',
        import_path='unused',
        inputs={},
        skip_backward=True,
    )

    monkeypatch.setattr(module, 'get_op', lambda name: config)
    monkeypatch.setattr(module, 'generate_inputs', lambda *args, **kwargs: {'x': paddle.ones([1])})
    monkeypatch.setattr(module, '_warmup_autotune', lambda fn, n=None: fn())

    class FakeLoss:
        def backward(self):
            raise AssertionError('backward should not run during warmup for skip_backward ops')

    def fake_op(**kwargs):
        return (paddle.ones([1]),)

    def fake_do_bench(fn, quantiles, **kwargs):
        fn()
        return (1.0, 0.9, 1.1)

    monkeypatch.setattr(module, '_import_op', lambda cfg: fake_op)
    monkeypatch.setattr(module.paddle, 'sum', lambda tensor: FakeLoss())
    monkeypatch.setattr(triton.testing, 'do_bench', fake_do_bench)

    results = module.benchmark_op('recurrent_stub', {'smoke': {'B': 1, 'T': 2, 'H': 3, 'D': 4}})

    assert [row['mode'] for row in results] == ['fwd']


def test_paddle_benchmark_runner_uses_native_backward_api(monkeypatch):
    module = _load_paddle_benchmark_run_module()

    calls = []

    def fake_backward(tensors, grads):
        calls.append((tensors, grads))

    monkeypatch.setattr(module.paddle.autograd, 'backward', fake_backward)

    tensor = paddle.ones([2], dtype='float32')
    grad = paddle.ones([2], dtype='float32')

    module._backward(tensor, grad)

    assert len(calls) == 1
    assert calls[0][0] == [tensor]
    assert calls[0][1] == [grad]


def test_paddle_benchmark_runner_clears_gradients_by_setting_none():
    module = _load_paddle_benchmark_run_module()

    x = paddle.randn([2, 3], dtype='float32')
    x.stop_gradient = False
    y = (x * x).sum()
    y.backward()

    assert x.grad is not None

    module._clear_gradients({'x': x})

    assert x.grad is None
