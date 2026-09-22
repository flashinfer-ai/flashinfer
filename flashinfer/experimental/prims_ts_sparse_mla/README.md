# Experimental native sparse MLA

`BatchSparseMLADecodePagedTSWrapper` implements Dqk = Dv = 512 sparse
attention on SM100/SM103 through the existing Prims-TS MLA kernel families.
Q and KV have matching native BF16 or E4M3 types; output is BF16. Calling
this API explicitly opts into experimental functionality. CuTe DSL 4.7 is
required. There is no packed-FP8 cache conversion, projection, RoPE or indexer.
Owner: @PerkzZheng; lifecycle and graduation are tracked in
[issue #5432](https://github.com/flashinfer-ai/flashinfer/issues/5432).

## Interface

```python
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.testing.sparse_mla_metadata import prepare_sparse_mla_metadata

w = BatchSparseMLADecodePagedTSWrapper()
w.plan(
    q.device,
    B,
    H,
    max_topk=K,
    max_seq_len_q=SQ,
    q_data_type=q.dtype,
    has_sinks=True,
    return_lse=True,
)
# Testing adapter; applications can provide equivalent metadata themselves.
meta = prepare_sparse_mla_metadata(w, q, kv, indices, lengths, sinks=sinks)
out, lse = w.run(q, kv, meta, sinks=sinks)
```

For two sources, pass `max_extra_topk` to `plan`, and `extra_kv_cache`,
`extra_indices`, and `extra_lengths` to preparation. Supply the extra pool as
`run(q, kv, meta, extra_kv_cache=extra, ...)`. Both lists participate in one
softmax; duplicates retain their multiplicity. Neither source is required to
be SWA. A single selected pool supports no-SWA attention.

Queries are `[B,SQ,H,512]`, or `[total_q,H,512]` with `packed_query=True`
and `qo_indptr`. Native pools are `[pages,page_size,512]` or singleton-head
NHD/HND tensors. Page padding is supported when row strides are multiples of
512 elements. Prepared indices address storage rows, including physical page
strides, so the attention kernel always uses page size one.

Sinks are FP32 `[H]` denominator mass. LSE uses natural logarithms and excludes
the sink. Empty rows return zero output and negative-infinity LSE. FP8 Q and
KV descales are positive per-tensor scalars or live CUDA FP32 scalar tensors.
Independent KV descales use two attention passes and a stable joint merge;
shared scales use one pass. BF16 Q/KV descales must be one. Supply the model's
`softmax_scale` explicitly when it differs from `512**-0.5`. Selected Q/KV
values and effective scale products must be finite.

## Metadata and ownership

`SparseMLAPreparedMetadata` holds caller-owned CUDA buffers:

| Field | Shape and type | Meaning |
|---|---|---|
| `indices`, `extra_indices` | int32 `[rows, capacity]` | Storage rows in each source; `-1` is invalid |
| `lengths`, `extra_lengths` | int32 `[rows]` | Active index prefixes |
| `routes` | int32 `[passes, rows, route_capacity]` | Bit 31 selects the extra source; low 31 bits encode its row; `0x7fffffff` is invalid |
| `execution_lengths`, `valid_counts` | int32 `[passes, rows]` | Padded execution span and valid entry count |
| `scale_params` | FP32 `[passes, 2 + H + rows]` | QK scale, PV scale, sinks, counts |

Each source's execution span is rounded to 128. Route capacity is at least
256; empty rows execute one masked slot. Independent scales require two
passes. The testing preparer emits all representations, including those used
by combined-route schedules. Packed routes, counts and scale headers must
agree with source indices and `run` arguments; this is a caller obligation.

Preparation adapts vLLM's Apache-licensed sparse-index mapping helpers and
adds the Prims-TS metadata layout. It translates block-table indices, compacts
holes stably and fills invalid tails. `prepare_causal_indices` can construct
SWA windows or HCA compressed prefixes from absolute positions. Native Q/KV
values are never repacked. Model-specific distributed cache layouts are not
implemented by this single-device testing adapter.

Warm up before CUDA Graph capture. Keep all inputs, metadata, outputs and
workspace alive at stable addresses; capture with preallocated outputs and
`validate=False`. Update metadata with `prepare_sparse_mla_metadata(...,
out=meta)` when indices, lengths, sinks or scales change. A wrapper/workspace
supports one in-flight call; use separate instances for concurrent streams.
Input, output and scratch overlap is unsupported and unchecked.

`get_prims_ts_sparse_mla_decode_workspace_size` takes the same arguments as
`plan` and returns the scratch size without compilation or allocation.
`batch_sparse_mla_decode_with_paged_kv_cache` is the eager plan/run helper.
There are no separate prepared methods or modes in the public API.

## Kernel and dispatch

The shared MLA kernels add D512/no-RoPE geometry and TMA `gather4` for all
1-CTA swap-AB, 1-CTA keep-AB and 2-CTA families. Gather warps cache row offsets
in registers through delayed V consumption; contiguous groups can use bulk
TMA. Source boundaries are 128-aligned so a gather quadruple uses one source.

Host-only policy ranks schedules using query/head work, selected capacity,
dtype and SM count. Split-KV and V partitioning fill otherwise idle SMs;
larger query tiles share gather work when there is sufficient parallelism.
KV reuse retains K through PV when one CTA owns the full V dimension and its
shared-memory budget admits enough stages. It is available through M64 but
is not always selected: fewer head tiles or reduced buffering can cost more
than the saved loads. BF16 BK64 reuse permits two live KV tiles where BK128
would exceed shared memory. Runtime request lengths remain device values.

BF16 supports FlashMLA's six-log2 deferred-max bound. FP8 uses exact maxima
and probability scale 448. `assume_valid_prefix=True` promises no holes before
each length; direct FP8 2-CTA kernels can then derive validity analytically.
`run(validate=True)` checks that promise. Generic hole masking remains supported.
The FP8 2-CTA path balances softmax/correction registers at 144/144 and uses
two K and two V stages, reducing spills and shared-memory pressure.

## Validation and reproduction

```bash
pytest tests/experimental/prims_ts_sparse_mla/ tests/attention/test_prims_ts_schedule_verification.py tests/trace/test_prims_ts_sparse_mla_trace.py
python tests/trace/sparse_mla_example.py
python benchmarks/bench_attention_ts_sparse_mla.py --indices 38,39,449,479 --output /tmp/sparse-mla.json
```

The model suite contains 480 configurations: BF16/FP8, H=8/16/32/64/128,
top-k=512/1024/2048, raw context 32768; prefill B=1/SQ=8192 and decode
B=1/4/16/64/256, SQ=1/4/8. Top-k uses FlashMLA-test-style random scores,
masked to valid lengths, followed by sorted top-k and random physical pages.
Both backends receive the same Q/KV/indices and are checked against one oracle.
The FP64 oracle includes sinks and scales. FP8 uses a probability-quantization
and BF16-output-rounding bound; comparator failures remain invalid comparisons.

Timing uses CUDA Graphs and a 4xL2 eviction before every measured invocation.
Metadata preparation and eviction time are excluded for both backends;
attention reductions, finishing and counter resets remain timed. Backend order
alternates between replays. Omit `--indices` to run all 480 configurations;
`--compression-ratio 128` exercises HCA and `--no-swa` uses one selected source.
The TRT DSV4 ABI requires an invalid 128-entry first segment for that comparison.
The optional JSON report records shape, precision, environment, seed, timing
quantiles and accuracy failures. Speedup is TRT latency divided by Prims-TS
latency. Both backends share the same actual input tensors.

The focused tests cover automatic dispatch across small/large grids, packed
queries, one/two sources, both native dtypes, live graph metadata/scales,
padded pages, empty rows and caller-owned outputs/workspace. Separate checks
cover metadata mapping/causality and the softmax score-update boundary. The
short direct-loader regression tests late masked tiles and omitted packed
metadata. No test locks down exact performance-policy selections.
