# Experimental SM110 GQA decode

This package provides an explicit opt-in FP16 decode path for NVIDIA SM110
(Jetson AGX Thor). It is specialized for 32 query heads, 8 key/value heads,
head dimension 128, and one query token per request. The current kernel
therefore has a fixed 4:1 query-to-KV head ratio.

The backend is reached only through explicit experimental APIs. It is
not registered for AOT packaging or automatic decode routing. The experimental
lifecycle is owned by `@yyihuang` and tracked in issue #5051.

The input contract is:

- `q`: contiguous FP16 `[batch, 32, 128]`
- `kv`: contiguous FP16 `[batch, 2, 8, capacity, 128]`, with K at index 0 and
  V at index 1
- `sequence_lengths`: contiguous CUDA int32 `[batch]`; every value must be in
  the inclusive range `[1, capacity]`
- `out`: optional caller-owned contiguous FP16 `[batch, 32, 128]`

Capacities through 64 use a 256-thread short-prefix kernel. Larger capacities
use a 384-thread pipelined kernel. Both routes use SM110 `tcgen05` tensor-core
instructions, TMA, and tensor memory, so they require compute capability 11.0
and CUDA 13.0 or newer.

```python
import torch
from flashinfer import sm110_gqa_decode

q = torch.randn(1, 32, 128, dtype=torch.float16, device="cuda")
kv = torch.randn(1, 2, 8, 1024, 128, dtype=torch.float16, device="cuda")
sequence_lengths = torch.tensor([1024], dtype=torch.int32, device="cuda")
out = sm110_gqa_decode(q, kv, sequence_lengths)
```

The CUDA source closure is generated and SHA-256 attested by `manifest.json`.
The JIT loader verifies every source digest before compiling the module.
See [RESULTS.md](RESULTS.md) for SM110 correctness and cold-L2 CUPTI results.

This API is experimental because its tensor layout and fixed head geometry are
serving-workload specific. Graduation requires broader workload validation,
stable packaging coverage, and an agreed integration point with FlashInfer's
decode APIs.

## Prepared sustained decode

Use `prepare_sm110_gqa_decode` and `launch_sm110_gqa_decode_prepared` to reuse
caller-owned output and workspace across ordinary launches or CUDA Graph replays:

```python
from flashinfer import prepare_sm110_gqa_decode, launch_sm110_gqa_decode_prepared

output = torch.empty_like(q)
prepared = prepare_sm110_gqa_decode(
    {"Q": q, "KV": kv, "O": output, "sequence_lengths": sequence_lengths}
)
result = launch_sm110_gqa_decode_prepared(prepared)  # result is output
```

Preparation validates tensor metadata, selects a manifest route and compiles
outside Graph capture. Every length must remain in `[1, capacity]` at launch;
GPU length values may change, but this precondition is not checked with a host
read. Unlike the convenience API, the prepared API requires caller output that
does not alias any input storage, and finite positive `q_scale` (default 1).
Launch uses the current PyTorch stream and returns asynchronously. Keep the
prepared object and tensors alive until all work and captured Graphs finish.
Order tensor updates before launch, and allocate separate prepared instances
for concurrently executing streams or Graphs because workspace is mutable.

Defaults select original short through capacity64, N32 direct output for
B4/capacity256, N64 KV-last S10 for B1/capacity1024, and N32 ring3 S10 for
B1/capacity4096. Other shapes use original long. Explicit `num_splits=1`
selects original long above capacity64; the specialized long routes also
support explicit `num_splits=10`. Unsupported splits are rejected.

See [PARTIAL_RESIDENCY.md](PARTIAL_RESIDENCY.md) for the frozen paired study,
per-condition results and measurement limits. Run
`examples/experimental/sm110_gqa_decode_prepared.py --graph` for a prepared
Graph example. Existing `sm110_gqa_decode` behavior is unchanged.
