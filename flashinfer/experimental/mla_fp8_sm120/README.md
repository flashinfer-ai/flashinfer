# SM120 FP8 MLA research backend

Source-only development implementation for GLM-4.7-Flash absorbed MLA on SM120.
Both QK and PV use E4M3 tensor-core MMA with FP32 accumulators and softmax;
partial and final outputs use BF16. Supports decode, causal/noncausal prefill,
and extend prefill with ragged, paged KV. This backend is experimental.

The internal `NativeMLA` runner is exercised through the standalone harness in
[benchmarks/mla_fp8_sm120](../../../benchmarks/mla_fp8_sm120/README.md).
It is not registered with `BatchMLAPagedAttentionWrapper`, automatic dispatch,
trace-apply, or AOT. Explicitly running the research harness is the opt-in;
`FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` does not select this backend.
No public core API has been added on this development branch.

From the repository root:

```bash
python benchmarks/mla_fp8_sm120/native_fp8/example.py
python tests/experimental/mla_fp8_sm120/check.py --output /tmp/mla-check.json
python benchmarks/mla_fp8_sm120/native_fp8/bench.py --output /tmp/mla-bench.json
```

The scripts wait for five consecutive seconds of GPU and memory-controller
utilization at or below 2%, without stopping any process. The benchmark rejects
rounds with observed external activity. The low-level runner does not manage
GPU availability; callers must coordinate shared devices themselves.

Contract and limitations:

- SM120 only; CUDA 12.8+ with `nvcc`, PyTorch, Triton and FlashInfer headers.
  Tested CUDA 12.9. The actual kernel target is `sm_120a` (byte `ldmatrix` transpose).
- Q is contiguous `[T, H, 576]`; KV is contiguous `[pages, page_size, 576]`.
  Output is contiguous `[T, H, 512]`. Pages may have 1, 16, 32, 64 or 128 tokens.
- `run()` accepts BF16 Q and includes Q quantization. `run_prequantized()`
  accepts E4M3 Q. Both require E4M3 KV and FP32 scales: one per Q row/head and
  one per KV token. The quantizer assumes finite BF16 inputs; externally
  provided scales must be finite and positive. All 576 dimensions are quantized.
- Default softmax scale is **1/16**, derived from GLM's original QK dimension
  192+64, not the absorbed dimension 512+64.
- Metadata must be valid nonnegative int32 ragged offsets and physical page
  indices. The caller owns page validity/lifetime and cache updates. No paged
  KV scatter or integration into a serving allocator is implemented.
- Planning happens in the constructor outside graph capture. A runner owns its
  metadata, output and quantization buffers and needs an exclusive contiguous
  uint8 CUDA workspace of at least 128 MiB. Runners may reuse the workspace
  sequentially. Output buffers are overwritten on each invocation. A runner
  may be replayed on its planning stream; synchronize explicitly before moving
  it to another stream. It is not safe for concurrent runs.
- Tiles and persistent workers are explicit tuning parameters; no universal
  heuristic is claimed. At most 440 workers and 16,384 planner work entries.
- Synthetic errors and operator speedups are not model task accuracy or tokens/s.
  Short full prefill currently loses to BF16; see the measured size breakdown.

The native CUDA algorithm reuses FlashInfer's paged persistent work format,
`MLAParams`, `cp.async`, swizzle, FP8 MMA, and stable `state_t` merge algebra.
The local scheduler parameterizes query/KV tiles and worker count; the merge
parallelizes split reads across warps. FP32 KV scales enter both QK and PV.

This research ABI uses ctypes to keep the installed baseline separate while
iterating. Its compiler writes only to a locked user JIT cache, fingerprints
checkout FlashInfer headers and compiler flags, and respects
`FLASHINFER_DISABLE_JIT=1`. It intentionally does not yet use the core TVM-FFI
`JitSpec` module: conversion and stable wrapper routing are graduation work.
See [TRACKING.md](TRACKING.md) for the development scope and remaining work.
