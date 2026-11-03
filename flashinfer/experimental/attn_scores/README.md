# Experimental MQA TopK for prefill

`flashinfer.FP8MQATopKWrapper` supports `strategy="full"` (FP8/FP32 logits,
`backend="v1"`) and `strategy="selective"` with `backend="v1"` or `"v2"`.
Requests concatenate K and scales in prefix order. These continuous routes do
not gather keys. `FP8PagedMQATopKWrapper` is a thin adapter over the same wrapper:
Full, selective v1 and selective v2 all gather paged K/scales into fixed
contiguous buffers before delegation. The gather copies capacity, including
unused live-context capacity, on every replay. Plan and prepare pools eagerly
before capture; warmed replay uses fixed buffers without allocation or compilation.

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

```python
import torch
from flashinfer import FP8MQATopKWrapper

op = FP8MQATopKWrapper(
    backend="v2", num_heads=64, head_dim=128, top_k=2048,
    max_context_length=131072, max_chunk_size=8192, max_batch_size=1,
    device="cuda", qkv_dtype="fp8", logits_dtype=torch.bfloat16,
)
# q: FP8[8192,64,128], weights: FP32[8192,64]
# k: FP8[131072,128], k_scale: FP32[131072]
qo = torch.tensor([0, 8192], dtype=torch.int32, device="cuda")
kv = torch.tensor([0, 131072], dtype=torch.int32, device="cuda")
op.plan(qo, kv)
indices = op.run(q, weights, k, k_scale=k_scale)
```

## Input and output contract

`plan(qo_indptr, kv_indptr, *, num_query_rows=None)` accepts contiguous CUDA
int32 `[B+1]` prefixes. Both start at zero. KV prefixes and
`max_context_length` count original tokens. For compression ratio `r` (1 or 4),
request b stores `floor(KV_b / r)` consecutive keys; the causal end of query i is
`floor((KV_b - Q_b + i + 1) / r)`. Request storage has no inter-request padding.
Q prefixes may leave an inactive suffix within the supplied query extent.

`num_keys` optionally specifies total K storage capacity. Its default is
`max_batch_size * floor(max_context_length / compression_ratio)`. Each bound K
and scale tensor must retain that shape, dtype, device and contiguous layout;
base addresses must be 16-byte aligned. Live prefixes may describe fewer keys
than storage. V2 MXFP4 storage must have a multiple of 256 rows to avoid an
internal constructor padding copy; pad storage once and keep the live length
in `plan`. FP8 accepts partial tiles.

| Backend | Q/K | Heads, D | Logits | TopK |
|---|---|---|---|---|
| v1 | FP8 E4M3 | Existing SM100/SM103 v1 specializations, including H4/H8/H32/H64, D128 | FP32 | 1–2048 |
| v2 | FP8 E4M3 or packed MXFP4 | H4/H32/H64, D128 | FP16/BF16/FP32 | 512/1024/2048 |

FP8 K is `[num_keys,D]`, with FP32 `k_scale[num_keys]`. MXFP4 K is uint8
`[num_keys,D/2]`, with UE8M0 bytes `k_scale[num_keys,4]`; Q uses the same packed
width, and `q_scale` is uint8 `[Q,H,4]` or packed int32 `[Q,H]`. Weights remain
FP32 `[Q,H]`.

V1 supports ragged batches. V2 remains single-request prefill, with at most 1M
stored keys and query capacity divisible by `max(256/H, 8 if H == 32 else 1)`.
Its six optional `model_config` presets retain the existing head/TopK contracts.
The four v2 kernels and their selection/recovery policies are unchanged.

Output is int32 `[supplied_Q,top_k]`, containing request-local indices. Short
rows have a -1 suffix; inactive rows contain only -1. Tied indices need not be
stable. The returned workspace is overwritten by the next run; consume it or
pass `out=`. Outputs must not overlap inputs or wrapper-owned buffers.

## Planning and capture

Construct outside CUDA graph capture. `plan` publishes a step once; `run`
consumes Q/weights and directly binds K/scales for each layer. `input_buffers()`
exposes stable Q, weights and prefix staging (plus q_scale for MXFP4), so a
producer can avoid Q/weight staging copies. It contains no page map.

Call `prepare_pool(k, k_scale=...)` or run eagerly before capture. Prepare each
v2 K/scale address pair before capturing it. V2 bindings share scratch and only
replace kernel arguments; they do not allocate a separate candidate workspace
per layer. Input updates at the same addresses are consumed on every replay.
Keep the wrapper and caller graph inputs alive until the graphs are discarded.
Use one wrapper per concurrent stream; its calls must be serialized.
`clear_pools()` synchronizes and releases bindings outside capture. Discard
caller graphs first, then prepare and recapture.

The device `error` word resets at each plan: bit 1 is an invalid supplied
extent, bit 2 an invalid prefix, and bit 4 an exceeded context/query/key capacity.
Invalid steps fail closed. The continuous API has no page IDs; the paged
adapter additionally uses bit 8 for invalid page IDs. `repair_error` is
OR-accumulated until the next plan. V2 retains per-layer numerical checks for
`relu_fast` and Qwen constants (bit 16); removing paging does not remove those
contracts. `relu_strict` remains the default. NaN scores rank as negative
infinity.

## Validation and benchmarks

`tests/experimental/attn_scores/test_paged_delegate.py` covers continuous
Full and v1 ragged offsets, compression, scale A/B/A binding, invalid-prefix
recovery, and caller-owned graph replay, plus paged delegation for all routes.
`test_indexer_topk_v2.py` covers direct v2 inputs/logits types, mutable payloads,
numerical contracts and recovery boundaries. `test_attn_scores.py` covers the
shared scoring, metadata and lifecycle contracts. Selection checks use independent
FP32 scores with boundary ties allowed; index output order is unspecified.
The v1 continuous API exports a `fi_trace` reference and checker; v2 trace
export remains unsupported. Historical paged selective fixtures exercise the
private shared engine and do not describe the public prefill API.

`benchmarks/bench_real_input_data.py` passes capture K/scales directly to both
selective backends. Its historical composed `full_gvr2` comparison prepares
paged storage for the native scorer and GVRv2; it is separate from the public
continuous Full strategy. Optional baseline/source comparison arms require `--reference-root`;
FlashInfer-only arms require only `--data-root`. Compare the same arithmetic mode, logits dtype,
input crop and timing scope; API timings include staging and validation.


Captured-input benchmarks accept `--data-root /path/to/captures`. Supply `.pt`
dictionaries containing `q`, `k`, `weights`, `starts`, `ends`, optional `q_scale`
and `k_scale`, and optional `metadata`. Lossless `.igemm` archives are also
supported when their portable `indexer_inputs/`, `format.json` and
`metadata_profiles.json` accompany the data. Tensor bytes are preserved; API
scope retains the most recent keys and translates causal bounds. Standalone
arms use FP32 logits and an independent all-row FP32 oracle; kernel scope and
lower-precision reference comparisons require the optional comparison suite.

```bash
python benchmarks/bench_real_input_data.py --data-root /path/to/captures \
  --model DS-V3.2 --scope api --length-axis original --lengths 8192 \
  --qkv fp8 --logits fp32 --arms full_gvr2 v1 v2_generic \
  --samples-per-config 1 --output results/captures
```
