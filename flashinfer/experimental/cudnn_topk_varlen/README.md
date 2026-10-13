# Experimental cuDNN variable-length Top-K backend

This backend adapts the optional cuDNN Frontend `IndexerTopKVarlen` prepared
API to `flashinfer.top_k_varlen`. It copies no cuDNN device code into FlashInfer.
The stable FlashInfer API and its existing backends remain unchanged by default.
The integrated API has been checked on GB300 (SM103) and Rubin (SM107).
Automatic selection is restricted to the measured shapes below; this is not
full-model or serving validation.

Explicit opt-in is `backend="cudnn"`. Automatic selection additionally requires
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`; leave that variable unset to
measure the existing automatic backend selection. Both uses emit the normal
FlashInfer experimental-backend warning. Record the variable and the selected
backend (`top_k_varlen.suitable_auto_backends[0]` immediately after an auto call)
when measuring the integration.

Automatic eligibility is the explicit `CUDNN_TOPK_AUTO_SHAPES` constant in
`support.py`: SM103 has `(T,N)=(512,16384),(512,32768),(512,131072)`; SM107 has
`(512,16384),(512,32768),(256,131072)`. These calls must use BF16 contiguous scores,
contiguous Int32 lengths, K=512, next_n=1, compress_ratio=1, indices-only output
and no hint. Other calls retain the existing ranking. The optional frontend
is not imported by this route when the gate is off or the call is ineligible.

The explicit backend supports T=1..512, N=1..262144, K in {512,1024,2048},
positive next_n dividing T and a positive Int32 compress_ratio. Scores are BF16;
lengths and output indices are Int32. Contiguous, 16-byte-aligned CUDA storage
on SM103/SM107 is required. Output buffers may use the public API's flat or
2-D form. Return values and hints are unsupported by this backend. Unresolved
negative/conjugate views and output/input aliases are rejected. Ties may select
any valid equal-valued index; output ordering is unspecified. NaNs within an
eligible row prefix are outside the cuDNN backend's input contract; values
outside that prefix are ignored. Length arithmetic is signed Int64 and clamped
to the score width, with -1 padding for short rows.

Install a cuDNN Frontend build exposing `IndexerTopKVarlen`
([upstream PR #1298](https://github.com/NVIDIA/cudnn-frontend/pull/1298)) and
`nvidia-cutlass-dsl >= 4.7` (SM107 requires >= 4.8). Older/missing frontends do
not enter auto selection. Compilation and launch failures are propagated.
Run each geometry eagerly before CUDA graph capture. The adapter keeps
metadata/code-only plans for the process lifetime, including their compiled
module owners so that existing graphs survive later preparations. It retains
no tensor, pointer, stream or GPU scratch. Each call binds current tensors
and the current stream; cuDNN records the allocations' stream use. Independent graphs may share the compiled plan
while using independent inputs and outputs.

Run `python examples/experimental/cudnn_topk_varlen.py` and
`pytest tests/experimental/test_cudnn_topk_varlen.py -q` on an intended GPU.
The example checks a captured replay after both scores and lengths change.

Owner: **@Anerudhan**. Tracking issue: [#5732](https://github.com/flashinfer-ai/flashinfer/issues/5732).
The graduation or removal review is due **2026-10-28**, targeting the first
regular FlashInfer release after that review, contingent on a frontend release
and broader measured coverage. This backend is JIT-only and has no AOT registration.

## Integrated measurements

The same `top_k_varlen(..., backend="auto")` call was measured with the
experimental gate disabled and enabled. All rows below use BF16, K=512,
next_n=1 and compress_ratio=1. Latencies are in microseconds, summarized as
the geometric mean of six medians (full/ragged lengths, three fresh seeds).
Each median contains nine timing samples.

| GPU | T | N | Warm: existing → cuDNN | Warm speedup | Cold: existing → cuDNN | Cold speedup |
|---|---:|---:|---:|---:|---:|---:|
| GB300 | 512 | 16384 | 28.34 → 22.36 | 1.27× | 36.27 → 30.01 | 1.21× |
| GB300 | 512 | 32768 | 43.50 → 23.97 | 1.82× | 51.26 → 32.54 | 1.58× |
| GB300 | 512 | 131072 | 108.02 → 56.73 | 1.90× | 115.18 → 65.86 | 1.75× |
| Rubin | 512 | 16384 | 18.99 → 14.65 | 1.30× | 24.41 → 20.60 | 1.19× |
| Rubin | 512 | 32768 | 28.49 → 18.64 | 1.53× | 35.14 → 25.36 | 1.39× |
| Rubin | 256 | 131072 | 50.85 → 28.19 | 1.80× | 57.41 → 36.25 | 1.58× |

Every individual profile exceeded 1.05× in both cache regimes. The disabled
route executed `radix` at N≤32768 and `radix_filter` at N=131072; the enabled
route executed the public cuDNN plan. Each GPU passed 40 API tests and 36
comparison profiles, with input-preservation and semantic output checks after
every timed block. There were no candidate or baseline correctness exemptions
in these integrated comparisons.

The GB300 environment used CUDA 13.3 and PyTorch 2.13 from the NVIDIA 26.06
container; Rubin used CUDA 13.5.7 and PyTorch 2.14. Both used CuTe DSL 4.8.0,
TVM-FFI 0.1.14.post1, and cuDNN Frontend commit
`21343ed785c563d42d9e1a78bf1aa93855165abb`.
See [the reproduction script and timing method](benchmark.md). Cold measurements
apply capacity eviction before each timed call; they do not measure hardware
cache-miss rates. Compilation is outside timing. These measurements exclude
framework remapping, logits production, attention, and model serving.
