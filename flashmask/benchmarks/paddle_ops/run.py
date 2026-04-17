from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import socket
import sys
from contextlib import contextmanager

import paddle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from registry import SHAPE_CONFIGS, OpConfig, generate_inputs, get_op, list_ops  # noqa: E402

logger = logging.getLogger(__name__)


@contextmanager
def _activate_paddle_driver():
    try:
        from linear_attn.triton_utils import paddle_driver
        from triton.runtime.driver import driver
    except Exception:
        yield
        return

    if paddle_driver is None:
        yield
        return

    driver.set_active(paddle_driver)
    try:
        yield
    finally:
        driver.reset_active()


def _import_op(config: OpConfig):
    mod = importlib.import_module(config.import_path)
    attr = config.func_name or config.name
    fn = getattr(mod, attr, None)
    if fn is None:
        raise ImportError(
            f"Cannot find '{attr}' in module '{config.import_path}'. "
            f"Available: {[x for x in dir(mod) if not x.startswith('_')]}"
        )
    return fn


def _get_machine_info() -> dict:
    info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'paddle_version': paddle.__version__,
    }
    try:
        import triton
        info['triton_version'] = triton.__version__
    except Exception:
        info['triton_version'] = 'N/A'

    if paddle.device.is_compiled_with_cuda():
        info['gpu_name'] = paddle.device.cuda.get_device_name()
        info['gpu_count'] = paddle.device.cuda.device_count()
    else:
        info['gpu_name'] = 'N/A'
        info['gpu_count'] = 0
    return info


def _warmup_iters() -> int:
    return max(1, int(os.environ.get('FLA_BENCH_OP_WARMUP_ITERS', '5')))


def _do_bench_kw():
    warmup_ms = int(os.environ.get('FLA_BENCH_WARMUP_MS', '25'))
    rep_ms = int(os.environ.get('FLA_BENCH_REP_MS', '100'))
    return {'warmup': max(1, warmup_ms), 'rep': max(1, rep_ms)}


def _synchronize():
    if paddle.device.is_compiled_with_cuda():
        paddle.device.synchronize()


def _clear_gradients(inputs: dict[str, paddle.Tensor]):
    for tensor in inputs.values():
        if isinstance(tensor, paddle.Tensor) and not tensor.stop_gradient:
            tensor.clear_gradient(False)


def _backward(tensor: paddle.Tensor, grad: paddle.Tensor):
    paddle.autograd.backward([tensor], [grad])


def _warmup_autotune(fn, n: int | None = None):
    if n is None:
        n = _warmup_iters()
    with _activate_paddle_driver():
        for _ in range(n):
            fn()
    _synchronize()


def benchmark_op(
    op_name: str,
    shapes: dict[str, dict[str, int]],
    modes: list[str] | None = None,
) -> list[dict]:
    import triton

    if modes is None:
        modes = ['fwd', 'fwdbwd']

    config = get_op(op_name)
    op_fn = _import_op(config)
    if config.skip_backward and 'fwdbwd' in modes:
        modes = [mode for mode in modes if mode != 'fwdbwd']

    dtype = paddle.bfloat16
    device = 'gpu'

    print(f"\n  [{op_name}] Warming up {len(shapes)} shape(s)...")
    failed_shapes = set()
    for shape_name, shape_dict in shapes.items():
        B, T, H, D = shape_dict['B'], shape_dict['T'], shape_dict['H'], shape_dict['D']
        try:
            inputs = generate_inputs(config, B, T, H, D, dtype=dtype, device=device)
            with _activate_paddle_driver():
                out = op_fn(**inputs, **config.extra_kwargs)
            out_tensor = out[0] if config.output_is_tuple else out
            do = paddle.randn(out_tensor.shape, dtype=out_tensor.dtype)

            def _fwd_fn(inputs=inputs):
                return op_fn(**inputs, **config.extra_kwargs)

            def _fwdbwd_fn(inputs=inputs, do=do):
                _clear_gradients(inputs)
                result = op_fn(**inputs, **config.extra_kwargs)
                tensor = result[0] if config.output_is_tuple else result
                _backward(tensor, do)

            warmup_fn = _fwdbwd_fn if 'fwdbwd' in modes else _fwd_fn
            _warmup_autotune(warmup_fn)
        except Exception as error:
            logger.warning(f"Warmup failed for {op_name} @ {shape_name}: {error}")
            failed_shapes.add(shape_name)

    valid_shapes = {name: cfg for name, cfg in shapes.items() if name not in failed_shapes}
    print(f"  [{op_name}] Warmup done.")

    results = []
    for shape_name, shape_dict in valid_shapes.items():
        B, T, H, D = shape_dict['B'], shape_dict['T'], shape_dict['H'], shape_dict['D']
        try:
            inputs = generate_inputs(config, B, T, H, D, dtype=dtype, device=device)
            with _activate_paddle_driver():
                out = op_fn(**inputs, **config.extra_kwargs)
            out_tensor = out[0] if config.output_is_tuple else out
            do = paddle.randn(out_tensor.shape, dtype=out_tensor.dtype)
        except Exception as error:
            logger.warning(f"Input generation failed for {op_name} @ {shape_name}: {error}")
            continue

        for mode in modes:
            if mode == 'fwd':
                def fn(inputs=inputs):
                    return op_fn(**inputs, **config.extra_kwargs)
            else:
                def fn(inputs=inputs, do=do):
                    _clear_gradients(inputs)
                    result = op_fn(**inputs, **config.extra_kwargs)
                    tensor = result[0] if config.output_is_tuple else result
                    _backward(tensor, do)

            try:
                with _activate_paddle_driver():
                    ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8], **_do_bench_kw())
            except Exception as error:
                logger.warning(f"Bench failed for {op_name} {mode} @ {shape_name}: {error}")
                continue

            results.append({
                'op': op_name,
                'mode': mode,
                'B': B,
                'T': T,
                'H': H,
                'D': D,
                'median_ms': ms[0],
                'p20_ms': ms[1],
                'p80_ms': ms[2],
            })

    return results


def print_results_table(results: list[dict], machine_info: dict | None = None):
    if not results:
        print("\n  No results to display.")
        return

    width = 92
    print(f"\n{'=' * width}")
    if machine_info:
        gpu = machine_info.get('gpu_name', 'N/A')
        paddle_version = machine_info.get('paddle_version', 'N/A')
        print(f"  Machine: {gpu} | Paddle {paddle_version}")
    print(f"{'=' * width}")
    print(f"  {'op':<18s} {'mode':<7s} {'B':>4s} {'T':>6s} {'H':>4s} {'D':>4s} {'median(ms)':>12s} {'p20(ms)':>12s} {'p80(ms)':>12s}")
    print(f"  {'-' * 18} {'-' * 7} {'-' * 4} {'-' * 6} {'-' * 4} {'-' * 4} {'-' * 12} {'-' * 12} {'-' * 12}")
    for result in results:
        print(
            f"  {result['op']:<18s} {result['mode']:<7s} {result['B']:>4d} {result['T']:>6d} "
            f"{result['H']:>4d} {result['D']:>4d} {result['median_ms']:>12.3f} "
            f"{result['p20_ms']:>12.3f} {result['p80_ms']:>12.3f}"
        )
    print(f"{'=' * width}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Paddle benchmark runner for flash-linear-attention ops')
    parser.add_argument('--op', nargs='+', default=None, help='Op name(s) to benchmark, or "all"')
    parser.add_argument(
        '--custom-shapes',
        default=None,
        help='JSON string to override default shapes, e.g. \'{"my": {"B":1,"T":2048,"H":16,"D":128}}\'',
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        default=['fwd', 'fwdbwd'],
        choices=['fwd', 'fwdbwd'],
        help='Benchmark modes (default: fwd fwdbwd)',
    )
    parser.add_argument('--json', dest='json_file', default=None, help='Output file path for JSON results')
    parser.add_argument('--list', action='store_true', help='List all registered ops and exit')
    args = parser.parse_args(argv)

    if args.list:
        ops = list_ops()
        print(f"Registered ops ({len(ops)}):")
        for name in ops:
            cfg = get_op(name)
            print(f"  {name:18s}  [{cfg.category}]  {cfg.import_path}")
        return []

    if args.op is None:
        parser.error('--op is required unless --list')

    op_names = list_ops() if args.op == ['all'] else args.op
    shape_configs = json.loads(args.custom_shapes) if args.custom_shapes else SHAPE_CONFIGS

    machine_info = _get_machine_info()
    print(f"Machine: {machine_info.get('gpu_name', 'N/A')} | Paddle {machine_info.get('paddle_version', 'N/A')}")
    print(f"Shapes: {len(shape_configs)} configs")
    print(f"Ops: {op_names}")

    all_results = []
    for op_name in op_names:
        try:
            all_results.extend(benchmark_op(op_name, shape_configs, modes=args.modes))
        except Exception as error:
            logger.error(f"Failed to benchmark {op_name}: {error}")

    mode_order = {'fwd': 0, 'fwdbwd': 1}
    all_results.sort(key=lambda result: (mode_order.get(result['mode'], 9), result['B'], result['T'], result['H'], result['D'], result['op']))
    print_results_table(all_results, machine_info)

    if args.json_file:
        output = {'machine_info': machine_info, 'results': all_results}
        with open(args.json_file, 'w') as handle:
            json.dump(output, handle, indent=2)
        print(f"\nResults saved to {args.json_file}")

    return all_results


if __name__ == '__main__':
    main()
