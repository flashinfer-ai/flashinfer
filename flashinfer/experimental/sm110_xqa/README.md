# Experimental SM110 XQA attention

This opt-in module provides native `tcgen05` attention for NVIDIA Thor GPUs
with the exact SM110a target. Call `flashinfer.sm110_xqa.prepare`
for prepared replay or `attention` for a single invocation. It uses frozen
CUDA sources, FlashInfer's JIT compiler and native TVM-FFI stream handling.
Both entry points use FlashInfer's standard experimental API decorator:
calling either is explicit opt-in and emits `ExperimentalWarning` once per
API. Backend implementation and JIT sources remain under
`flashinfer.experimental.sm110_xqa`. Experimental APIs may change or be
removed without deprecation.

Native compilation and correctness checks have run with CUDA 13.4 on physical
SM110 hardware. See [RESULTS.md](RESULTS.md) for execution-validation status,
sanitizer outcomes and performance results. The interface supports two
attention families:

| Family | Q | KV | Other metadata |
| --- | --- | --- | --- |
| Decode, D128 | FP16 `[B,Hq,128]` or `[B,1,Hq,128]` | FP16 `[B,2,Hkv,C,128]` | int32 lengths `[B]`; Hq/Hkv = 4, 8, or 16 |
| Tree, D512 | FP16 `[B,Q,Hq,512]` | FP16/E4M3 `[B,2,Hkv,C,512]` | int32 lengths `[B]`, packed mask; Hq/Hkv = 2, 4, 8, or 16 |
| Paged tree | Same Q, or packed Q below | FP16/E4M3 `[numPages,128,Hkv,512]` | int32 page table `[B,2,maxPages]` |
| Packed tree | FP16 `[totalQ,Hq,512]` | Either tree layout | int32 prefix offsets `[B+1]`, explicit maximum query length |

All tensors must be contiguous CUDA tensors on one device. The two components
of the page table address K and V independently. The draft mask contains
int32 packed words, with bit 1 permitting attention. It covers only the final
draft-token block; all earlier tokens are visible. Its shape is
`[B,Q,ceil(Q/32)]`, or `[totalQ,ceil(maxQ/32)]` for packed queries. Packed rows
retain the maximum-length mask stride even when a request is shorter.

Device metadata must satisfy these preconditions: sequence lengths fit cache
capacity; each tree length includes that request's draft block; page indices
address the pool; query offsets are nondecreasing, start at zero, end at
totalQ, and each request length is at most maxQ. Preparation validates tensor
metadata without copying device contents to the CPU.

```python
from flashinfer.sm110_xqa import prepare

plan = prepare(
    q, kv, sequence_lengths,
    mask=mask,                         # omitted for D128 decode
    out=output,                       # optional caller-owned tensor
    page_table=page_table,             # None for contiguous KV
    page_size=128,                    # 0 for contiguous KV
    k_scale=k_scale,
    v_scale=v_scale,
)
output = plan.run()
```

`sm_scale` defaults to `1/sqrt(D)`. `k_scale` and `v_scale` independently
dequantize FP8 storage; decode currently requires both to be 1. Output is
FP16 and may not alias any input. D128 accepts a runtime `partition_tokens`
argument that is a positive multiple of 64; omitting it selects the frozen
manifest's default. The manifest also identifies whether split results use a
separate merge kernel or a fused last-CTA merge. This is a frozen physical
choice, not a change to attention semantics.
If the frozen producer caches merge statistics, preparation selects a
separate specialization with that cache disabled for exactly one partition.
Multiple partitions use the cache-enabled specialization, including its
generic device loop when there are more than four partitions. The selected
physical route is available as `plan.route`; replay does not redo dispatch.

D128 workspace is `(partial, statistics, counters)`: FP32 tensors of shapes
`[B*Hq,partitions,128]` and `[B*Hq,partitions,2]`, plus int32 counters
`[B,Hkv]`, where `partitions=ceil(C/partition_tokens)`. It is available as
`plan.workspace`; a compatible, disjoint workspace tuple can be passed to
`prepare`. D128 Q, KV, output and partial-output base addresses must be
16-byte aligned for vector memory accesses. The stats-cache producer requires
an 8-byte-aligned statistics address for vector loads; the single-partition
specialization and producers without stats cache require 4-byte alignment.
Native launch bindings reject insufficient alignment before submission.
Preparation initializes counters
to zero once. The fused kernel
wraps them back to zero at the end of each completed replay; a single
partition writes output directly. Workspace must remain private to ordered
launches: do not launch two plans sharing it concurrently or re-prepare it
while an earlier launch is running.

Prepare before timing or graph capture. `plan.run()` submits the complete
kernel sequence on the caller's current stream and performs no tensor
allocation. `attention(...)` is a convenience wrapper that prepares and runs
once. Standard PyTorch stream dependencies apply to inputs and workspace
initialization. A launch on another stream must wait for preparation and any
previous replay using the same workspace; output consumers must wait for the
launch stream. No host synchronization or stream selection is hidden in the
prepared plan. For example:

```python
preparation_stream = torch.cuda.current_stream(q.device)
plan = prepare(q, kv, sequence_lengths, partition_tokens=256)  # D128
execution_stream = torch.cuda.Stream(device=q.device)
execution_stream.wait_stream(preparation_stream)
with torch.cuda.stream(execution_stream):
    output = plan.run()
preparation_stream.wait_stream(execution_stream)
```

Neither API substitutes a kernel for another GPU architecture.

The module requires CUDA 13.0 or newer and physical capability 11.0; the tested
toolchain is CUDA 13.4. The frozen source manifest under `csrc/sm110_xqa/`
specifies compiler flags and route geometry for six base routes, plus the
single-partition specialization when required. Its `validation_status` field
is an immutable source-generation record captured at freeze. Subsequent
execution validation is documented separately in [RESULTS.md](RESULTS.md).
The frozen D512 route declares a 128- or 256-column output tile and four or
eight cooperating copy warps. Grid geometry comes from that manifest; native
bindings use the traced thread block, shared memory and tensor-memory layout
of the same physical candidate.
Each route also records whether it stages raw FP8 bytes asynchronously before
widening to FP16. This physical option applies only to E4M3 cache routes; FP16
routes always record it as disabled. Raw prefetch is enabled only when that
FP8 staging path is active. The manifest also records the complete-prefix
mask fast path and the half-warp fused decode merge selection. These physical
choices do not change the public tensor API or dequantization scales.

Validation commands from a FlashInfer checkout:

```bash
python -m pytest tests/experimental/sm110_xqa/test_sm110_xqa_jit.py -q
python -m pytest tests/experimental/sm110_xqa/test_sm110_xqa.py -q
python benchmarks/bench_sm110_xqa.py --warmup-ms 250 --output sm110-xqa-native.json
```

The GPU suite retains the complete 44-case numerical ledger and adds runtime
partition/repeated-counter checks, caller-provided workspace initialization,
two cross-stream graph-replay cases, output-alias rejection and misaligned
stats-cache workspace rejection (54 cases in
total). Runtime cases cover one, two, three, four and more than four partitions
and verify the selected specialization. Run it against each candidate manifest before freezing the physical
choice. Numerical checks use an FP32
oracle and `atol=rtol=1e-2`, including quantized cache cases. Seven performance
rows are specified in `benchmarks/sm110_xqa_shapes.json`. Consult
[RESULTS.md](RESULTS.md) for the measured results and validation status.
Performance inputs use seed-0 uniform [-1,1] values for D512 and seed-0 normal
(0,1) values for D128, as recorded separately in that ledger.

The standalone benchmark uses `flashinfer.testing.bench_gpu_time` with CUPTI,
cold L2, a default 250 ms warmup target and 256 measured iterations. The
`--warmup-ms` option passes `dry_run_time_ms` to the upstream timer and leaves
`dry_run_iters` unspecified; the timer estimates a count from that duration.
An explicit `--warmup-iters` is mutually exclusive with duration warmup.
The output records the selected policy. It measures the complete prepared replay,
including both stages when a separate merge is selected. CUPTI dependency
preflight and an exception on the timer's fallback warning prevent CUDA-event
fallback. The upstream helper uses events for its preliminary warmup estimate;
the reported samples use CUPTI's full activity span. This benchmark reports
native latency and does not expose an ordered activity audit or establish a
paired source/export performance comparison.
