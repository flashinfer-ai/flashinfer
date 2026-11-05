# SM120 NVFP4 decode buckets

An experimental paged-attention backend for mixed KV lengths on SM120. It
balances 256-token KV tiles across requests while reusing native XQA's packed
NVFP4 loader, BF16 QK/PV arithmetic, FP32 softmax and accumulation, and last-CTA
partial merge. A plan can be prepared once and reused across compatible
attention layers. Native XQA dispatch and defaults are unchanged.

Both the API and backend are experimental. Calling the public functions below
is the explicit opt-in and emits an `ExperimentalWarning` once per function;
no environment variable is required. This feature does not participate in
`backend="auto"`, autotuning, trace-apply or AOT packages. Callers choose when to
use it and retain their native fallback.

- Owner: [@Nicko-F](https://github.com/Nicko-F) (proposed for admission).
- Tracking issue: [#8757](https://github.com/flashinfer-ai/flashinfer/issues/8757).
- Proposed graduation target: 0.8.0, with the first lifecycle review within four
  weeks of admission. The owner and target are proposals for maintainer review.

## Supported configurations

| Input | Contract |
| --- | --- |
| GPU | SM120; validated on RTX 5090 |
| Batch | 1–32 requests |
| Query/output | Contiguous BF16, head dimension 256; `[B,H*R,256]` for T=1 or `[B,T,H*R,256]` for T>1 |
| Query length T | Uniform 1–5 tokens per request, with a causal suffix |
| KV heads H | Positive, subject to device grid and checked workspace/index bounds; correctness anchors cover H=1/2/4/8 |
| Q/KV head ratio R | 4, 6 or 8 |
| K/V cache | NHD packed NVFP4 `uint8 [pages,P,H,128]` |
| K/V block scales | E4M3 `float8_e4m3fn [pages,P,H,16]` |
| Page size P | 16, 32 or 64 tokens |
| Metadata | Contiguous CUDA int32 page table `[B,width]` and lengths `[B]` |
| Query tile | 16 rows (default), or explicit 32 rows for T>1 |
| Effective scales | Independent finite `bmm1_scale` for QK and `bmm2_scale` for V/output |

Packed heads are contiguous; native page/token/head byte strides are retained,
including separate or interleaved K/V storage. Q, cache, scales, output and
scratch require 16-byte aligned addresses. Output must not share storage with
query, cache, metadata or workspace tensors. Unsupported cases stay with the
caller's native path. Sinks, sliding windows, soft caps, ragged queries,
arbitrary tree masks and FP8 output are outside this backend's contract.

Lengths include the complete T-token suffix. A length is zero for a padded
request or at least T, and cannot exceed `width * P`. Query row t attends to
keys before `L - T + t + 1`. Every page ID reached by a live length must address
the supplied cache. These device-value conditions are caller preconditions;
the hot path does not copy metadata back to the host. Zero-length requests
write zero output. No model, TP degree or configured context size is required.

## API and layer reuse

Use the public entry points in `flashinfer.xqa`:

```python
from flashinfer.xqa import create_xqa_decode_bucket_plan, xqa_decode_with_buckets

# seq_lens and page_table are the caller's CUDA int32 metadata.
# Allocate and JIT before compilation or CUDA Graph capture.
plan = create_xqa_decode_bucket_plan(
    seq_lens,
    page_table_width=page_table.shape[1],
    page_size=32,
    num_kv_heads=4,
    head_group_ratio=6,
    q_len_per_req=1,
    query_tile=16,
    mode="persistent",
)
workspace = plan.create_workspace()

# Update seq_lens in place for each forward, then prepare once.
plan.prepare()
for q, k, v, ks, vs, out in compatible_attention_layers:
    xqa_decode_with_buckets(
        q, k, v, ks, vs, page_table, plan, workspace,
        bmm1_scale=qk_scale, bmm2_scale=v_scale, out=out,
    )
```

The plan retains the length tensor and fixed descriptor addresses. Compatible
layers share request order, lengths and plan geometry; their Q/cache/output
remain independent. Sequential layers may reuse a workspace. Concurrent
computes require independent workspaces and outputs. Keep plans, workspaces and
referenced tensors alive until every asynchronous consumer and captured graph
has completed. Order plan refresh and workspace reuse on one stream, or with
explicit CUDA events across streams.

`mode="persistent"` assigns global interval buckets and can execute several
request tasks per CTA. `mode="aligned"` uses request-local partitions and one
task per CTA. They are explicit alternatives: balanced and skewed workloads can
prefer different modes. `query_tile=32` similarly selects an explicit native
M32 specialization; M16 remains the default.

For B=1, or when every potential tile fits the device CTA budget, `plan.direct`
is true. Compute derives task boundaries from live lengths, descriptors are
empty, and `prepare()` is a no-op. Otherwise `prepare()` runs one GPU assignment
kernel, independently of layer count. Selection uses static table capacity,
so a reserved table wider than its live prefix can change launch geometry.

The default `planner="gpu"` reads device lengths. The optional host planner uses
caller-owned CPU int32 lengths in the same request order:

```python
plan = create_xqa_decode_bucket_plan(
    seq_lens,
    page_table_width=page_table.shape[1],
    planner="host",
    host_seq_lens=host_seq_lens,
)
# After updating both CPU and GPU length buffers:
plan.prepare()  # C++ fill and one asynchronous H2D, outside capture
compute_graph.replay()
```

The host planner uses one pinned staging buffer and waits for its previous H2D
before refilling it. CPU/device length agreement remains the caller's
responsibility. Host planning was slower than GPU planning in the measured
completed-wall comparisons; it is an integration option, not the default.

### CUDA Graphs and torch.compile

Warm the actual public compute call before capture or compilation, and supply
`out` to avoid output allocation. Capture GPU `plan.prepare()` once followed by
compatible compute calls, or prepare before replaying a compute-only graph.
Host preparation must stay outside capture and compiled consumers. Its fill,
H2D and dependency costs belong in completed-wall measurements.

On PyTorch 2.14.0+cu130, fixed-shape `torch.compile(fullgraph=True)` and changed
input CUDA Graph replay passed for direct, persistent, aligned, host-prepared
and two-layer shared-plan configurations. Local `torch.library` operators
explicitly declare descriptor/workspace/output mutations. This qualification
covers warmed calls; it does not cover cold JIT, the first experimental warning,
autograd or dynamic-shape compilation. Compute `opcheck` schema validation hit
a PyTorch Float8 `allclose` limitation; fake-tensor checks, byte checks of
read-only inputs and real compiled numerical replay passed. See the
[compile results](../../../docs/benchmark_protocol.md#compile-and-shared-plan-results).

## Performance

RTX 5090, one attention layer, hot CUDA Graph timing, five warmups and twenty
samples per arm. Native XQA receives identical operands and keeps its default
dispatch. Native values below average the before/after medians; candidate
windows include GPU assignment when a plan needs it. Allocation and JIT are
excluded. P/H/R/T denotes page size, KV heads, Q/KV ratio and query length.

| Configuration | KV lengths | Native (us) | Bucket full window (us) | Native / bucket |
| --- | --- | ---: | ---: | ---: |
| P32/H1/R6/T1, M16 persistent | 131072 + 31 × 4096 | 618.459 | 64.474 | 9.59x |
| P32/H1/R6/T1, M16 persistent | 32 × 4096 | 32.323 | 38.467 | 0.84x |
| P32/H4/R6/T3, M32 persistent | 65536 + 7 × 4096 | 475.357 | 111.582 | 4.26x |
| P32/H4/R6/T3, M32 aligned | 65536 + 7 × 4096 | 475.357 | 173.432 | 2.74x |
| P32/H4/R6/T3, M32 aligned | 32 × 4096 | 158.789 | 168.824 | 0.94x |

The 22-case M16 suite improved 17 full-window medians and regressed five,
with ratios from 0.56x to 9.59x. The M32 ownership follow-up shows why both modes
remain explicit: aligned improves balanced 4K over persistent, while persistent
is faster on the T3 skew case. These are attention-module results, not model or
serving latency. Complete tables, source revisions, resource reports and timing
boundaries are in [the benchmark protocol and results](../../../docs/benchmark_protocol.md).

## Run the example and tests

Use a normal FlashInfer source installation and CUDA toolchain with SM120
support. Sources and caches follow FlashInfer package/workspace discovery.

```bash
python examples/xqa_decode_buckets.py --lengths 65536,4096,4096,4096 --benchmark
pytest tests/experimental/test_xqa_decode_buckets.py -q
pytest tests/experimental/test_xqa_decode_buckets_compile.py -q

# Reproduce the representative suite or the explicit M32 mode comparison.
python examples/xqa_decode_buckets.py --suite tencent22 --layers 1 --output-json results.json
python examples/xqa_decode_buckets.py --suite query_modes4 --layers 1 --output-json modes.json
```

The example also supports `--page-size`, `--num-kv-heads`, `--head-group-ratio`,
`--q-len-per-req`, `--query-tile`, `--capacity-length`, `--compare-planners` and
`--compare-compile`. Suite JSON contains samples, errors, counters and resources;
`--resume` continues completed cases only when the saved configuration matches.
Tests include changed lengths/page IDs/Q/K/V, graph replay, zero/restore,
independent workspaces, CPU/GPU descriptor agreement and causal future-KV
perturbations against a dequantized reference.

## Resource bounds

With `G = H * ceil(T*R/query_tile)`, the per-group CTA capacity is
`C = max(B, min(ceil(SMs/G), B * ceil(width*P/256)))`. Prepared descriptor
capacity is `C+B-1`, shared across groups, and scratch has at most
`G * (C+B-1)` native slots. The factory checks int32 counts, uint32 native
head/cache offsets, scratch byte extent and the device grid-Y limit before use.
Table width is bounded by `(INT32_MAX-255)//P`; live lengths need not equal
that capacity.

Observed M16/M32 shared memory is 70,272/86,656 bytes per CTA, with one resident
CTA per SM in the qualified variants. Registers depend on specialization;
M32 R6/P32 persistent/aligned use 219/228 registers with a 112-byte stack.
Retained compiler qualification reports zero spill loads/stores. The plan's
`resources` tuple reports registers, shared bytes, local bytes and resident CTAs
for the actual compiled kernel; local-byte counts alone do not establish spills.

