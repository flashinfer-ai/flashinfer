# Nemotron MXFP8 local experts

Experimental GB300/SM103 operator for hidden size 768, intermediate size 4096,
128 local experts, 512 global experts and top-k 10. Owner: helenn.
It owns local kernels and workspace; the caller owns NVLS gather/reduce-scatter.
There is no distributed initialization, MCore dependency or automatic dispatch.

```python
from flashinfer.fused_moe.nemotron_mxfp8 import (
    pack_nemotron_mxfp8_weights,
    prepare_nemotron_mxfp8_local_experts,
)

# Select the CUDA device first. Pack once per layer, before graph capture.
weights = pack_nemotron_mxfp8_weights(w1, s1, w2, s2)
plan = prepare_nemotron_mxfp8_local_experts(
    weights,
    max_tokens=4 * local_capacity,
    local_expert_offset=128 * ep_rank,
)
plan.run(gathered_x, global_ids, probabilities, out=nvls_fp32_partial)
```

## Contract

Canonical weights are contiguous E4M3 `[128,4096,768]` / `[128,768,4096]`.
Scales are unswizzled E8M0 bytes `[128,4096,24]` / `[128,768,128]`.
Packing changes layout without requantization. Plans use exact gathered capacity
`T`, with BF16 input `[T,768]`, int64 global IDs `[T,10]`, FP32 probabilities
`[T,10]` and caller-provided FP32 output `[T,768]`. Tensors must be contiguous,
16-byte aligned, non-overlapping and on the selected CUDA device.

IDs must be -1 (padding) or in `[0,512)`. Routes outside the selected local
expert range contribute zero. Routing values are not scanned on the host.
Invalid contribution slots may contain stale data; assembly masks their loads.
`contribution_view()` borrows BF16 `[T,10,768]` storage for diagnostics.

The plan runs on the current stream. Warm before capture; keep plans, packed
weights and graph inputs/outputs alive until graphs are destroyed and queued
work completes. Separate buckets require separate plans. Packed weights may be
shared but must not be mutated. A plan cannot run concurrently on multiple
streams. `close()` releases storage without synchronizing.

## Computation

Fused input quantization/routing feeds fused FC1, hardware-tanh clamped squared
ReLU, MXFP8 activation quantization and FC2. The activation is
`(16 * tanh(max(z, 0) / 16))**2`. Detached cleanup resets workspace after compute;
FP32 assembly applies routing weights in slot order, with multiply/add separate.
No intermediate BF16 top-k reduction is performed.

Defaults preserve the research candidate: 152 workers, group hint 1024,
M128/N256/K128 tiles, one-CTA clusters, lookup scheduling, flag batch 1,
data padding 64 and scale padding 128. Pipeline stage calculation is unchanged.
`hardware_tanh=False` is a numerical attribution control.

## Validation and provenance

Run `python tests/experimental/nemotron_mxfp8/test_plan_contract.py` for CPU contracts,
`python tests/experimental/nemotron_mxfp8/test_gpu.py` for SM103 reference/replay tests,
and `python examples/experimental/nemotron_local_experts.py` for a synthetic example.
The example is not a representative performance benchmark.

Adapted from research commit `52e7da3f4791f059b5840d4ec210c7556b70c421`, using
kernel-team snapshot `50d583aa0` and FlashInfer's existing shared primitives.
Unchanged MXFP8 kernel/epilogue methods and the FC2 destination descriptor are
reused directly; this package contains the Nemotron-specific overrides.
The vendor `src/` drop is unchanged. Research timing/debug paths are excluded. Private kernels are specialized to the
fixed local-expert configuration: no standalone FC1/FC2 diagnostics, alternate
geometry, quantized combine output or in-kernel top-k reduction. The fused
activation, lookup scheduler, pipeline sizing and detached cleanup are retained.

Synthetic reference and graph checks do not establish checkpoint numerical
qualification. Deployment experiments have matched research token outputs and
reached about 17.4k tokens/s at 768 requests on GB300 EP4, but intermittent
prefill slowdowns remain under investigation. Performance preservation and
independent checkpoint qualification are required before claiming readiness.
Experimental submission also requires a tracking issue and graduation target.
