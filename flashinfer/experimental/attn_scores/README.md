# Experimental FP8 paged MQA indexer TopK

`FP8PagedMQATopKWrapper` computes paged FP8 MQA indexer scores and returns
request-local int32 TopK indices, with `-1` padding for short rows. It serves
DeepSeek-style sparse attention prefill and decode for serving frameworks
such as vLLM and TRT-LLM.

- **Selective logits** (`strategy="selective"`, Indexer-throwaway) estimates
  thresholds from sampled scores, publishes bounded candidate scores and
  indices, then verifies and repairs incomplete or overflowing rows exactly.
  The complete score matrix is never written.
- **Full logits** (`strategy="full"`) scores every key and runs **GVRv2** over
  stripes of a shared, budgeted score arena.

## Status and graduation

Both the API and backend are experimental. Construction is explicit opt-in
and emits `ExperimentalWarning` once per process; no environment variable is
required. There is no automatic routing, autotuning or trace-apply selection.
The interface may change or disappear without deprecation.

The plan/run/prepare contract and capacity/workspace options are settling.
Decode performance is not consistently at parity with **Full logits** +
**GVRv2** on the documented profiles. The implementation is runtime-JIT only
and is not registered for AOT packages.

Graduate when the public API has no further breaking changes and B148/Q1
decode matches or beats **Full logits** + **GVRv2** on the documented profiles.
The default lifecycle review cadence is four weeks, following the
[experimental policy](../README.md).

- Owner: @yweng0828 @rosenrodt
- Target release: TBD.
- Tracking issue: flashinfer-ai/flashinfer#8109

## Support

| Strategy | TopK | Compression ratio | Page size (tokens) |
| --- | --- | --- | --- |
| **Selective logits** | Every integer from 1 to 2048 | 1 or 4 | 32, 64 or 128 |
| **Full logits** | 512, 1024 or 2048 | 1 | 32, 64 or 128 |

Both require CuTe DSL and B200 (SM100, CUDA >= 12.8) or B300 (SM103,
CUDA >= 12.9). Queries are FP8 E4M3 `[rows, heads, head_dim]`, head weights
are FP32 `[rows, heads]`, and fused KV is contiguous uint8
`[pages, page_size, 1, head_dim + 4]`, storing FP8 keys and FP32 scales.
Head dimensions must be divisible by 32; scorer specialization and resource
limits are checked during preparation. Strict ReLU is the default.

Ratio 4 takes caller-compressed KV with original-token lengths and returns
compressed-slot indices. **Selective logits** ranks NaN scores as negative
infinity; **Full logits** follows the public scorer's NaN behavior. Equal
scores may select equivalent indices. Multi-device execution is not qualified.

## API and serving lifecycle

Construct after allocating the indexer cache, outside capture. Plan once per
step and run per layer. This example assumes two equal-layout layer pools,
FP8 queries, FP32 weights, device int32 prefixes and a device int32 page table:

```python
import torch
from flashinfer import FP8PagedMQATopKWrapper

wrapper = FP8PagedMQATopKWrapper(
    strategy="selective", num_heads=8, head_dim=128,
    page_size=128, top_k=512, compression_ratio=1,
    max_context_length=65536, max_chunk_size=148, max_batch_size=148,
    kv_layout=kv_layer0,
)
rows = q0.shape[0]
out0 = torch.empty((rows, 512), dtype=torch.int32, device=q0.device)
out1 = torch.empty_like(out0)

wrapper.plan(qo_indptr, kv_indptr, block_table, num_query_rows=rows)
wrapper.run(q0, weights0, kv_layer0, out=out0)
wrapper.run(q1, weights1, kv_layer1, out=out1)

# Read status after execution, outside graph capture.
if wrapper.error.item() or wrapper.repair_error.item():
    raise RuntimeError("step validation or exact repair failed")
```

- Unhinted `plan()` refreshes device metadata without host reads or
  allocation. `next_n=1/2/4` declares uniform query rows per request.
  `host_length_hints=(query_prefix_cpu, kv_bounds_cpu)` is eager-only and is
  rejected under capture. It sizes eager **Full logits** stripes; selective
  sampling still validates device lengths before using a hinted bypass.
- Either strategy prepares a missing pool binding on its first eager
  `run()`. Optional `prepare_pool()` moves compilation and warmup out of that
  call. Before capture, run or prepare every layer pool for each
  `(num_query_rows, request count, next_n)` bucket.
- In-capacity replans preserve buffer addresses and prepared bindings.
  `input_buffers()` exposes fixed-address staging. Separate `out=` tensors
  preserve layer results. A host planning error requires a valid replan.
- The wrapper owns no parent graph. The caller captures plan/run as needed;
  under caller capture, selective exact repair uses a CUDA conditional node
  whose body the library records. Eager **Selective logits** synchronizes one
  repair-needed flag per layer to skip empty repair. Captured **Full logits**
  uses an unhinted capacity-fixed layout.
- `max_workspace_size` bounds only the full FP32 score arena (default
  512 MiB), excluding **GVRv2**, metadata, staging and outputs. Its minimum is
  `8 * max(8192, top_k, padded_seq_len(max_context_length))` bytes for one
  query pair. Selective validates the option but ignores it for scratch sizing.
- Keep wrappers and pools alive during replay, and release wrappers outside
  capture. Use one wrapper per concurrent stream. Discard caller graphs before
  `clear_pools()`; changing capacity requires a new wrapper.

Correctness and capture tests: `tests/experimental/attn_scores/`.
Runnable example: `python tests/experimental/attn_scores/example.py`.
Benchmark: `python benchmarks/bench_fp8_mqa_selective_logits.py --help`.
