# Experimental SM110 GQA decode

This package provides an explicit opt-in FP16 decode path for NVIDIA SM110
(Jetson AGX Thor). It is specialized for 32 query heads, 8 key/value heads,
head dimension 128, and one query token per request. The current kernel
therefore has a fixed 4:1 query-to-KV head ratio.

The backend is reached only through the explicit `sm110_gqa_decode` API. It is
not registered for AOT packaging or automatic decode routing. The experimental
lifecycle is owned by `@yyihuang` and tracked in issue #5051.

The input contract is:

- `q`: contiguous FP16 `[batch, 32, 128]`
- `kv`: contiguous FP16 `[batch, 2, 8, capacity, 128]`, with K at index 0 and
  V at index 1
- `sequence_lengths`: contiguous CUDA int32 `[batch]`; each value selects a
  valid prefix no larger than `capacity`
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
