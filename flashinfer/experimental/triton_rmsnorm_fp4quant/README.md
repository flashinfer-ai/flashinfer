# Experimental Triton RMSNorm + NVFP4 on SM120

This backend fuses FP32 RMSNorm, per-16 E4M3 scale generation, and native E2M1
packing in one row program. It targets SM120 (RTX 5090), with four warps per row
and no shape autotuning. The existing CuTe implementation remains the default.

Owner: @Micdiane. Tracking issue: [#5209](https://github.com/flashinfer-ai/flashinfer/issues/5209).
Proposed graduation target: 0.7.1, with lifecycle review by 2026-10-12, subject
to maintainer agreement. [RESULTS.md](RESULTS.md) separates prototype evidence
from the adapted revision's completed SM120 validation and paired timings.

## Use

With FlashInfer installed from this branch and Triton installed:

```python
import torch
from flashinfer.norm import rmsnorm_fp4quant

x = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
global_scale = torch.tensor([32.0], device="cuda")
q, sf = rmsnorm_fp4quant(
    x, w, global_scale=global_scale,
    is_sf_swizzled_layout=True, backend="triton",
)
# q: float4_e2m1fn_x2 [128, 2048], even element in the low nibble
# sf: float8_e4m3fn, padded 128x4 swizzled storage
# Reuse the same output storage, including during CUDA Graph capture:
rmsnorm_fp4quant(
    x, w, q, sf, global_scale=global_scale,
    is_sf_swizzled_layout=True, backend="triton",
)
```

Naming the backend explicitly opts in and emits `ExperimentalWarning` once.
No environment variable is needed. Neither the default nor `backend="auto"`
selects this backend, even when experimental automatic selection is enabled.
Triton is imported only after explicit selection. This backend is JIT-only.

Supported inputs are contiguous BF16 tensors of rank two or three, BF16
weight `[K]`, `64 <= K <= 8192`, and `K % 16 == 0`. Only NVFP4 (`block_size=16`,
`scale_format=None` or `"e4m3"`) is supported. Both row-major (default) and
128x4 swizzled scale layouts are supported. Empty batches return empty outputs.
Provided output buffers must have the expected shape, dtype and device, be
contiguous, and not overlap inputs or each other. Swizzled padding is left
unspecified and must not be read as valid scales.

`enable_pdl=None` and `False` use ordinary stream ordering; `True` is rejected.
FP16, MXFP4, other architectures, and wider hidden dimensions are not supported
by this experimental backend. Existing CuTe calls retain their supported modes.

## Arithmetic and implementation

For each row, the kernel loads X into FP32 values, reduces the sum of squares,
and computes `y = (x * rsqrt(mean(x*x) + eps)) * weight`. It retains the row
through quantization instead of explicitly staging X in shared memory and
issuing a second set of global X loads as the current CuTe kernel does.
That is an implementation difference, not a measured claim about HBM traffic:
cache hits, register allocation and instruction scheduling also affect timing.

For each 16-value group, `sf = E4M3(min(max(abs(y))/6 * global_scale, 448))`,
then `q = E2M1(y * global_scale / sf)`. Zero scales produce zero magnitudes.
The corresponding dequantization is `q * sf / global_scale`. A supplied FP32
`global_scale[1]` is read from the device at execution time, so replaying a graph
uses its updated value. `None` means one without allocating a scale tensor.
Inputs must be finite, FP32 intermediate arithmetic must be representable,
and supplied global scales must be finite and positive; the hot path does not
synchronize to inspect device values.

CuTe's BF16 path rounds `x * weight` to BF16 before applying the RMS coefficient.
This kernel keeps that arithmetic in FP32, so bitwise equivalence is not the
contract. Tests use an independent CPU FP64 RMSNorm reference, E2M1 grid
enumeration with ties-to-even, and dequantization error checks.

The packing helper uses `cvt.rn.satfinite.e2m1x2.f32` via Triton's inline PTX.
Row offsets are widened to 64 bits before multiplication. There is no separate
GEMM, transpose, or BF16 intermediate materialization inside this kernel.

## Reproduce

On an allocated SM120 GPU, from the repository root:

```bash
python -m pytest tests/experimental/test_triton_rmsnorm_fp4quant.py -q
compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest tests/experimental/test_triton_rmsnorm_fp4quant.py -q
python benchmarks/bench_triton_rmsnorm_fp4quant.py --output forward.jsonl
python benchmarks/bench_triton_rmsnorm_fp4quant.py --reverse --output reverse.jsonl
# Diagnose public-output allocation, host overhead, and layout separately:
python benchmarks/bench_triton_rmsnorm_fp4quant.py --smoke \
  --allocation public --output public.jsonl
python benchmarks/bench_triton_rmsnorm_fp4quant.py --smoke \
  --timing eager --layout row-major --output eager-row-major.jsonl
```

Each output file must be new. The benchmark checks numerical error before
measuring all three methods: CuTe default PDL, CuTe with PDL disabled, and
Triton. It uses `flashinfer.testing.bench_gpu_time`, resident inputs, five
randomized rounds, and an unchanged shape matrix. Graph timing amortizes
launch overhead over 50 calls and returns 20 replay samples per round.
`--timing cupti` opts into hardware profiling when CUPTI is installed.
Eager CUDA-event timing includes host launch gaps and must not be combined
with graph results. Global scales one and 32 are separate benchmark strata.

Performance claims and tested versions belong in the PR's results table;
prototype timings must not be presented as timings of an adapted kernel.
