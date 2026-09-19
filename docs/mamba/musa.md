# MUSA Mamba2 / SSD provider

The MUSA provider is selected by the caller; it does not change the public
FlashInfer API. vLLM-MUSA selects it with `--mamba-backend flashinfer` and the
MUSA SSD dispatch flag. The provider is implemented in `flashinfer/mamba` and
uses Triton-MUSA kernels.

## Native paths

`ssd_combined_fwd_varlen` is native when the packed inputs are half or BF16,
`dt` is rank 2, `A` is rank 1, softplus is enabled, the lower `dt_limit` is
non-negative, and the chunk size is a power of two. The packed metadata is
`cu_seqlens`, `cu_chunk_seqlens`, `last_chunk_indices`, and `seq_idx`. It can
return per-chunk intermediate states for Mamba cache mode `all`.

Ordinary one-token `selective_state_update` is native for FP16 state and the
rank/layout combination used by Nemotron Mamba2. The stochastic path supports
dstate 64, 128, and 256, Philox rounds 5 or 10, and preserves the full 64-bit
seed/counter. The output uses the unrounded state; stochastic rounding is only
used when writing the cache.

## Reference fallbacks

The MUSA implementation intentionally falls back to the reference recurrence
for unsupported dtypes/layouts, checkpoint materialization, arbitrary
token-level `seq_idx` transitions, fixed-length APIs, MTP/speculative decode,
quantized states, and non-one-token SSU modes. A fallback is a functional
compatibility path, not a native performance claim.

Do not flatten a fixed `[B,S,H,D]` call into the varlen API unless its state,
output layout, sequence transitions, and checkpoint semantics have been
validated independently. The fixed API and checkpoint paths have caller-owned
layouts that are not equivalent to packed chunk states.

## Validation contract

MUSA changes require differential output/state tests against the reference
implementation, exact stochastic cache-bit tests against the CPU Philox oracle,
and a compiled vLLM smoke with prefill PIECEWISE and decode FULL graphs. A
kernel microbenchmark is not sufficient evidence for serving speed; the
FlashInfer and default Triton backends must use identical input/output token
counts, warmup, timing, and graph settings.
