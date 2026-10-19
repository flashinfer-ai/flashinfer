# Experimental SM110 XQA attention

This opt-in module provides native attention for NVIDIA Thor GPUs with the
exact SM110a target in five physical kernel families: `tcgen05` (tensor memory,
D128 decode and D512 tree), `register_mma` (the register `mma.sync` XQA
schedule, D512 tree only, selected with `kernel="register_mma"`),
`register_mma_split` (the same schedule split over the two KV halves by two
eight-warp groups with an in-CTA merge, frozen for FP16 page128 KV only,
selected with `kernel="register_mma_split"`; `kernel="register_mma_auto"`
picks it for FP16 page128 KV and `register_mma` elsewhere), `tmem` (the
tcgen05/TMEM-accumulator D512 tree schedule with Q and K/V fetched once per
thread-block cluster by TMA multicast, every D512 cache mode, GQA ratios
2/4/8/16, selected with `kernel="tmem"`) and `pair` (the same
schedule as two `cta_group::2` CTA pairs in a `(4, 1, 1)` cluster that issue
one MMA stream for the two 128-row Q tiles of a KV head; E4M3 KV with an even
Q-tile count per head, GQA ratios 2/4/8/16, selected with `kernel="pair"`;
`kernel="auto"` picks `pair` for E4M3 KV with an even Q-tile count and `tmem`
otherwise). Call
`flashinfer.sm110_xqa.prepare` for prepared replay or `attention` for a single
invocation. It uses frozen
CUDA sources, FlashInfer's JIT compiler and native TVM-FFI stream handling.
Both entry points use FlashInfer's standard experimental API decorator:
calling either is explicit opt-in and emits `ExperimentalWarning` once per
API. Backend implementation and JIT sources remain under
`flashinfer.experimental.sm110_xqa`. Experimental APIs may change or be
removed without deprecation.

Native compilation and correctness checks have run with CUDA 13.4 on physical
SM110 hardware. The interface supports two attention families:

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
default recorded in `jit.FROZEN`. The frozen decode producer merges its
partitions in the last CTA; that is a frozen physical choice, not a change to
attention semantics.
Because the frozen producer caches merge statistics, preparation selects a
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
toolchain is CUDA 13.4. `jit.py` is the physical record of the delivered
sources, written by the Cake export that generates them: `MODULES` lists the
generated programs (the four `register_mma` D512 tree routes `tree_*_mma`, the
`register_mma_split` FP16 page128 tree route `tree_fp16_paged_mma_split`, the
`tmem` D512 tree routes `tree_*_tmem` in their two cluster forms and the two
`pair` E4M3 tree routes `tree_fp8_*_pair`), each with its kernel and binding
translation units, compiler flags, launch geometry and the argument plan of
its `run` entry; `FROZEN` lists the routes kept from the earlier frozen tree
(the four `tcgen05` D512 tree routes and the D128 decode producer with its
single-partition specialization) with their compiler flags, content hash and
launch facts; `ROUTES` maps `<route>__<cluster form>` to the serving program.
The `tcgen05` D512 route runs a 64-row Q tile per CTA over a 256-column output
tile; native bindings use the traced thread block, shared memory and
tensor-memory layout of the same physical candidate.
The `register_mma` tree routes launch one 32-row Q tile per CTA over all 512
output columns with eight QK warps and eight PV warps (512 threads, grid
`(1, Hkv * ceil(Q * ratio / 32), B)`), the same tensor layouts, mask contract,
dequantization scales and tolerances as the `tcgen05` tree routes, and no
tensor memory. The `register_mma_split` route keeps that Q tile, grid, mask
contract, scales and tolerances, and runs two eight-warp groups over the two
halves of the KV sequence (each with its own K/V ring) that merge their
unnormalised partials and row statistics in shared memory before one group
writes the output; it is frozen for FP16 page128 KV, the one cache mode where
it is faster than `register_mma` on both validated Thor nodes. The `tmem`
tree routes launch one 128-row Q tile times one 256-column output half per CTA
(512 threads, grid `(2, Hkv * ceil(Q * ratio / 128), B)`) with S, P and O
accumulated in tensor memory on `tcgen05.mma`; the two output-half CTAs of a Q
tile and the two Q tiles of a KV head form a `(2, 2, 1)` thread-block cluster in
which the Q tile is fetched once per output-half pair and every K/V tile once
per cluster by TMA multicast (a `(2, 1, 1)` Q-multicast form serves heads with
an odd number of Q tiles). The binding encodes the Q and KV tensor maps on the
host from the caller's tensors and launches with the cluster attribute; the
same tensor layouts, mask contract, dequantization scales and tolerances apply.
Each `tmem` program is one kernel template instantiated for the GQA ratios 2,
4, 8 and 16 (the 128-row Q tile is ratio heads x 128 / ratio tokens), one
program per cache mode and cluster form; the host selects the program from the
Q-tile parity and the binding dispatches on the `head_group_size` argument and
encodes Q with the ratio's box. The `pair` tree routes
(`tree_fp8_{contiguous,paged}_pair`) run the same tcgen05/TMEM schedule as two
`cta_group::2` CTA pairs in a `(4, 1, 1)` cluster per two Q tiles of a KV head
(grid `(4, Hkv * ceil(Q * ratio / 128) / 2, B)`, one template instantiation
per GQA ratio): the pair
leader issues one M256 MMA stream for both Q tiles, each CTA holds one token
half of every K chunk and one column half of every V chunk, and the raw E4M3 V
rows are multicast to the pair. They serve E4M3 KV whose heads have an even
number of 128-row Q tiles; `kernel="pair"` rejects FP16 KV and odd Q-tile
counts. The families are selected explicitly; `register_mma_auto` chooses
between the two register families and `auto` selects `pair` for E4M3 KV with
an even Q-tile count per head and `tmem` otherwise. D128 decode has only the
`tcgen05` family.
The generated binding of every D512 program binds the KV cache as bytes (the
E4M3 routes widen them in the kernel), the draft mask as unsigned 32-bit words
and the pointer arguments the kernels never dereference (packed-query offsets
for uniform queries, attention sinks, semaphores, scratch) as null device
addresses; no placeholder tensors are allocated.

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
and verify the selected specialization. Numerical checks use an FP32
oracle and `atol=rtol=1e-2`, including quantized cache cases. The eighteen
performance rows are specified in `benchmarks/sm110_xqa_shapes.json`.
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
