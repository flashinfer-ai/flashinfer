# FROST low-latency FP8 projections (experimental)

This backend implements the existing `flashinfer.bmm_fp8` operation. It has no
LightLM, checkpoint, attention, or recurrent-state dependency. The BMM batch
dimension is one; M activation rows can belong to independent requests.

```python
import torch
from flashinfer import bmm_fp8

a = torch.randn(1, 2, 1024, device="cuda").to(torch.float8_e4m3fn)
b = torch.randn(1, 1024, 1024, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
sa = torch.ones(1, device="cuda")
sb = torch.ones(1, device="cuda")
y = bmm_fp8(a, b, sa, sb, torch.bfloat16, backend="frost-low-latency")
```

The backend is explicit-only, including when
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`; it does not change the default
backend or participate in autotuning. It emits the standard experimental
warning. Compatibility and long-term support are not guaranteed.

Admission: SM100/SM107, E4M3 inputs, scalar device FP32 scales, BF16 output, BMM batch
one, `1 <= M <= 64`, positive K divisible by 512, positive N divisible by 8.
A is contiguous row-major, B is column-major, and both must be 16-byte aligned.
Optional `out` is contiguous on the same device. This is a correctness support
range, not a claim of a performance win throughout that range.

Four warps per CTA compute eight output channels for one activation row. Each
warp reuses vectorized activation loads across two channels. E4M3 values are
expanded into packed BF16 pairs; their finite products are exact in BF16 (at
most eight significant bits, with exponent range fitting BF16). All sums and
warp reductions are FP32, followed by scalar scaling and one BF16 output cast.
Reduction order can differ from tensor-core GEMMs. This is a standalone SIMT
projection, with no PDL, split-K workspace, or inter-CTA synchronization.

Kernel specializations cache only geometry/device metadata. Tensor addresses,
scale values, and the current stream are rebound each call. Warm the operation
before CUDA graph capture. Disk compilation artifacts use FlashInfer's normal
source/compiler-sensitive CuTe DSL cache.

The likely application is small-M projections where tensor-core setup and
partition reduction are expensive. Larger M permits tensor-core weight reuse
that this kernel does not provide. Measure both cache-resident and evicted
weights on the deployment GPU; do not dispatch on model identity.

Validation and crossover sweep:

```bash
pytest tests/experimental/frost_low_latency/test_fp8.py -q
python benchmarks/bench_frost_low_latency_fp8.py --out fp8.json --cupti
```

The benchmark compares explicit cuBLAS, CUTLASS, cuDNN, the existing
`mm_fp8` TRT-LLM and CuTe DSL low-latency backends, and this backend.
TRT-LLM weight packing and fixed-scale multiplication occur outside timing.
It warms/autotunes each provider, preallocates outputs, measures CUDA graph replay with alternating
order, checks outputs after replay, and reports unavailable providers rather
than silently replacing them. Hot graph timings amortize submission across 100 calls per graph.
With `--cupti`, the benchmark also reports GPU spans from the first kernel start
to the last kernel end with warm and evicted weights; eviction is outside the
measured interval. These are distinct
timing methods and should not be mixed in a speedup ratio. Numbers
are component timings, not serving speedups.

Owner: @YangXu1990uiuc. Graduation target: the first release on or after
2026-11-04, subject to measured dispatch coverage and maintainer review.
Tracking issue: [#6189](https://github.com/flashinfer-ai/flashinfer/issues/6189).
