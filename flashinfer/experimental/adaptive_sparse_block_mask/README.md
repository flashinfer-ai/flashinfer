# Experimental adaptive sparse block-mask selection

This package contains a JIT-only CUDA implementation of the adaptive
threshold policy used by Tencent hpc-ops Stem TPD.  It turns BF16 proxy
attention scores into a per-head `torch.bool` block mask while preserving a
fixed prefix and causal diagonal window.

The public entry point is
`flashinfer.sparse.adaptive_sparse_block_mask`. Calling it is the explicit
experimental opt-in and emits `ExperimentalWarning`; it is never selected by
an automatic backend and is not registered in AOT.

Supported contract:

- `block_logits`: contiguous CUDA BF16 `[B, H, Qb, Kb]`;
- length tensors: contiguous CUDA int32 `[B]`, expressed in tokens;
- `1 <= Kb <= 32768`;
- CUDA architectures supported by FlashInfer from SM90 onward;
- ties at the top-k boundary may select more than the nominal budget.

The port is derived from hpc-ops commit
`2a2e26562433a8ba4b504858f1c938eb7612c901`. The experimental owner is
[`@slhslh`](https://github.com/slhslh). Lifecycle and graduation work is
tracked in [flashinfer-ai/flashinfer#5366](https://github.com/flashinfer-ai/flashinfer/issues/5366).

Example:

```python
import torch
from flashinfer.sparse import adaptive_sparse_block_mask

scores = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.bfloat16)
lengths = torch.tensor([64 * 128], device="cuda", dtype=torch.int32)
mask = adaptive_sparse_block_mask(scores, lengths, lengths, lengths, alpha=0.5)
```

Graduation requires stable naming, SM90 and SM100 benchmark coverage, CUDA
Graph coverage with caller-owned output, trace integration, and demonstrated
end-to-end benefit when connected to a FlashInfer sparse-attention backend.
Measured B200 results and reproducer commands are recorded in
[RESULTS.md](RESULTS.md).
