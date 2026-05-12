# linear_attn

Triton-based GDN (Gated Delta Networks) and KDA (Kimi Delta Attention) operators with chunk-wise and fused-recurrent execution modes.

## Dependencies

- PaddlePaddle-GPU (GPU required)
- triton
- pytest (for tests)
- einops (for GDN tests)

## Environment Setup

```bash
# Install the flash_mask package (includes flash_mask.linear_attn)
cd /path/to/flash-attention/flashmask
FLASHMASK_BUILD=fla pip install -e . --no-build-isolation
```

`FLASHMASK_BUILD` supports `fa3`, `fa4`, `fla`, `all`, and combinations such as `fa3+fla` or `fa3, fla`. Use `fla` when you only need the linear-attention GDN/KDA operators. Import operators from the installed package namespace, for example `from flash_mask.linear_attn.ops.kda import chunk_kda`. The legacy top-level `linear_attn` module is not installed as a separate package.

## Running Tests

Test files are located at `flashmask/tests/linear_attn/`:

| File | Description |
|------|-------------|
| `test_gated_delta.py` | GDN operator correctness tests |
| `test_kda.py` | KDA operator correctness tests |

Each test compares the Triton-optimized implementation against a naive Python reference, checking both forward output and backward gradients.

```bash
cd /path/to/flash-attention/flashmask

# Run all tests
pytest tests/linear_attn/test_gated_delta.py tests/linear_attn/test_kda.py -v

# Run GDN tests only
pytest tests/linear_attn/test_gated_delta.py -v

# Run KDA tests only
pytest tests/linear_attn/test_kda.py -v

# Run a single test function
pytest tests/linear_attn/test_gated_delta.py::test_fused_recurrent -v
pytest tests/linear_attn/test_kda.py::test_chunk -v

# Filter by parametrized id
pytest tests/linear_attn/test_gated_delta.py -k "test_fused_recurrent and B1-T63" -v
```

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_TEST_CHUNK_VARLEN=1` | unset | Skip varlen (variable-length sequence) tests |
| `FLA_BENCHMARK=1` | `0` | Disable driver probing overhead |

```bash
SKIP_TEST_CHUNK_VARLEN=1 pytest tests/linear_attn/test_gated_delta.py tests/linear_attn/test_kda.py -v
```

## Running Benchmarks

The benchmark framework has been moved to the `test_flashmask/` repository and supports 4 operators:

| Operator | Description | Modes |
|----------|-------------|-------|
| `chunk_gdn` | GDN chunk-level | fwd / fwdbwd |
| `chunk_kda` | KDA chunk-level | fwd / fwdbwd |
| `recurrent_gdn` | GDN fused recurrent | fwd only |
| `recurrent_kda` | KDA fused recurrent | fwd only |

```bash
cd /path/to/test_flashmask

# List registered operators
python benchmark_linear_attention_run.py --list

# Run all benchmarks
python benchmark_linear_attention_run.py --op all

# Run specific operators
python benchmark_linear_attention_run.py --op chunk_gdn
python benchmark_linear_attention_run.py --op chunk_kda recurrent_kda

# Forward only
python benchmark_linear_attention_run.py --op chunk_gdn --modes fwd

# Custom shapes
python benchmark_linear_attention_run.py --op chunk_gdn \
  --custom-shapes '{"smoke":{"B":1,"T":64,"H":2,"D":32}}'

# Save results as JSON
python benchmark_linear_attention_run.py --op all --json results.json
```

### Default Shape Configs

| Config Name | B | T | H | D |
|-------------|---|---|---|---|
| B1_T8192_H96_D128 | 1 | 8192 | 96 | 128 |
| B2_T16384_H16_D128 | 2 | 16384 | 16 | 128 |
| B4_T2048_H16_D128 | 4 | 2048 | 16 | 128 |
| B4_T4096_H64_D128 | 4 | 4096 | 64 | 128 |
| B8_T2048_H32_D256 | 8 | 2048 | 32 | 256 |
| B8_T1024_H8_D64 | 8 | 1024 | 8 | 64 |

### Benchmark Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLA_BENCH_OP_WARMUP_ITERS` | `5` | Number of warmup iterations |
| `FLA_BENCH_WARMUP_MS` | `100` | `do_bench` warmup time (ms) |
| `FLA_BENCH_REP_MS` | `500` | `do_bench` repeat measurement time (ms) |

## Known Limitations

- Context Parallel (CP) is NOT supported.
- `fused_recurrent_gdn` / `fused_recurrent_kda` are forward-only. Use `chunk_gdn` / `chunk_kda` for training workloads that require gradients.

## Known Issue: GDN Backward Precision on Hopper GPUs with Triton >= 3.4.0

The upstream fla-org/flash-linear-attention project has identified a backward precision issue in the gated `chunk_bwd_dqkwg` kernel when running on Hopper-class GPUs (H20, H100, GB200, etc.) with Triton >= 3.4.0 ([upstream PR #827](https://github.com/fla-org/flash-linear-attention/pull/827)). The upstream fix introduces a TileLang-based kernel as an alternative backend.

**Current status in this fork:**
- On NVIDIA H800 (Hopper) with the current Triton version, this issue has **not** been observed in practice.
- If you plan to deploy on other Hopper GPUs (H20, GB200, etc.), or upgrade Triton to >= 3.4.0, you may encounter this backward precision regression.
- The TileLang backend has **not** been integrated into this Paddle port yet.

**Action needed:** When targeting Hopper GPUs other than H800 or upgrading Triton, consider integrating the TileLang backend from the upstream fix (`pip install tilelang` + dispatch logic in `fla/ops/common/chunk_o.py`).
