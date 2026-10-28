# Experimental paged FP8 MQA TopK

Owner: @rosenrodt. A tracking issue and target release for graduation still need
to be supplied by the owner. Graduate when the API stabilizes and decode
performance is good: no further breaking public API changes, and B148/Q1 decode
reaches parity with or beats Full + GVRv2 on the MR-table profiles.
The default incubation review cadence is four weeks.

`flashinfer.FP8PagedMQATopKWrapper` is an experimental API for both its explicit
`strategy="full"` and `strategy="selective"` modes. Calling its constructor opts
in and emits `ExperimentalWarning` once per process. No environment variable is
required; the wrapper does not automatically select another backend. Its API
may change or disappear without deprecation. The existing public FP8 and FP4
paged scorers remain stable and are not changed by this designation.

## Support and numerical contract

- Datacenter Blackwell with CuTe DSL: SM100 needs CUDA 12.8 or newer;
  SM103 needs CUDA 12.9 or newer. Experimental helpers are JIT-only.
- FP8 queries and fused paged FP8 KV, FP32 weights, page sizes 32/64/128.
- Selective TopK 1..2048, compression ratio 1 or 4; full TopK 512/1024/2048,
  compression ratio 1. The caller supplies already-compressed KV for ratio 4.
- Returned int32 indices are request-local, with `-1` suffix padding for short
  or empty rows. Equivalent indices may be chosen at score ties.
- `math_mode="relu_strict"` is the default. Selective ranks final NaN scores as
  negative infinity; full retains the public scorer's existing NaN behavior.
- Selective publication is bounded and exact repair completes overflowing or
  underfilled rows. D128 uses sparse gather4 SAMPLE where its layout bounds
  permit, with device-computed positions and gated boundary-bin refinement.
  Other dimensions and unsupported gather4 layouts use whole-page sampling.

## Serving lifecycle

Construct after the indexer cache is allocated, before graph capture. Capacities
and the cache layout are fixed for the wrapper's lifetime; reallocation requires
a new wrapper. A layout exemplar is inspected but not retained. The alternative
`num_pages`/`device` form describes contiguous storage.

Construction rejects active CUDA graph capture before backend loading and
argument validation. Both strategies raise `RuntimeError` in that case; the
full strategy reports `construct the wrapper outside capture`. This early guard
keeps definition loading outside live caller captures.

```python
from flashinfer import FP8PagedMQATopKWrapper

wrapper = FP8PagedMQATopKWrapper(
    strategy="selective", num_heads=64, head_dim=128, page_size=128,
    top_k=2048, compression_ratio=1,
    max_context_length=1048576, max_chunk_size=8192, max_batch_size=1,
    kv_layout=kv_fused,
)
# Optional warmup; one eager run also prepares the layout.
wrapper.prepare_pool(kv_fused)
wrapper.plan(qo_indptr, kv_indptr, block_table, num_query_rows=q.shape[0])
indices = wrapper.run(q, weights, kv_fused)
# Inspect error and repair_error after synchronization outside graph capture.
```

Call `plan(qo_indptr, kv_indptr, block_table, *, num_query_rows=None)` once per
forward step and `run(q, weights, kv_fused, *, out=None)` for every indexer layer.
The page table needs at least
`ceil((max_context_length // compression_ratio) / page_size)` columns.
Selective planning and equal-layout pool binding are capturable after warmup;
full planning reads prefixes on the host and must run outside capture. Full
keeps a score buffer and GVR workspace per prepared pool and invalidates those
pool graphs on a new plan. Selective shares its scratch across layer pools.

`input_buffers()` exposes fixed-address staging views. `error` is a device
validation bitmask; `repair_error` is an OR flag, not a count; `active_shape`
contains active row/request counts. Use `clear_pools()` outside capture before
rebinding same-shape pools. Keep wrappers alive while their caller graphs replay,
and release wrappers outside capture.

## Coverage and limitations

Tests and the runnable lifecycle/oracle cases live in
`tests/experimental/attn_scores/`. The in-tree benchmark is
`benchmarks/bench_fp8_mqa_selective_logits.py`.

From the repository root on a supported GPU:

```bash
python tests/experimental/attn_scores/example.py
python -m pytest -q tests/experimental/attn_scores
```

The sparse loader has measured decode overhead at small contexts. Real-input
captures cover H4/H32/H64 prefill, not H8 or decode; MXFP4 evaluation is
threshold-policy-only. Multi-GPU paths and installed-cache-only execution are
not covered by this feature's single-GPU qualification. CuTe kernels still need
their supported compiler/runtime during JIT compilation. See the PR's measured
tables for the performance scope, rather than assuming selective is faster for
every shape.
