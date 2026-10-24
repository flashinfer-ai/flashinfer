# Cake W8A8 MXFP8 MegaMoE EP16 (experimental, SM103)

Experimental **backend** behind the stable `MoEEpMegaLayer` API. Select it by
name: `kernel_name="sm103_mxfp8_mxfp8_bf16_cake"` through
`Sm103_Mxfp8_Mxfp8_Bf16_Cake_MegaMoeConfig`. Constructing a layer with it emits
one `ExperimentalWarning`. It does not take part in automatic backend selection,
autotuning or trace-apply, and it is not registered for AOT (JIT only).

## What it is

One decode forward is a fixed sequence of five Cake-generated CUDA programs on
the current stream (route `w8a8_megamoe_ep16_staged_v4`), each launched by a
hand-written tvm-ffi binding, with no host memset or copy between them:

| # | Program | Grid | Role |
|---|---|---|---|
| 1 | `publish_push` | tokens | quantize bf16 tokens to block-32 MXFP8 in-kernel; zero this rank's combine read window; reset the routing counters, route map and overflow flag; push each token's 3232-byte slot (row, scales, ids, weights) into the inbox of every rank that owns one of its experts and flag it |
| 2 | `local_pack` | 16 * tokens | wait on the local inbox flags; pack the rows this rank owns per expert (overflow accounting) |
| 3 | `fc1` | persistent grid | SwapAB FC1 + SwiGLU + MXFP8 requant |
| 4 | `fc2` | persistent grid | SwapAB FC2, route rows pushed back to their source rank; the last CTA adds +1 to every rank's `fc2_completion` counter |
| 5 | `combine` | 3 * tokens | wait on the local `fc2_completion` counter; source-order FP32 accumulation of the eight route rows, one CTA per 1024-column slice, bf16 out |

There is no input barrier (the push replaces the pull dispatch and its barrier,
see "Push dispatch" below) and no partial barrier (the FC2 completion counters
replace it, see "FC2 completion counters" below).

The persistent grid is the largest `g` in `[ceil(3 * sm_count / 4), sm_count]`
that divides both full-envelope tile counts (1280 FC1 tiles, 768 FC2 tiles), so
every CTA runs whole waves of both programs; the even SM count when no such
divisor exists. On 148- and 152-SM Blackwell parts it is 128
(`cake_plan.persistent_grid_for_sm_count`).

Weights are quantized once at layer construction by the route's own
quantization program (`preprocess_weights`), so the kernel-layout bytes match
the measured route exactly. Communication uses plain CUDA over NVSHMEM
peer-mapped symmetric buffers; the layer bootstraps NVSHMEM over the EP process
group (`runtime_requirements = {torch_dist, nvshmem}`).

## Contract

| Item | Value |
|---|---|
| Compute capability | exactly 10.3 (`sm_103a`); `10.3a` must be a FlashInfer build target |
| EP geometry | world size 16, 512 experts (32 per rank), hidden 3072, intermediate 5120, top-8 |
| Tokens | `max_tokens_per_rank` in 1..64; bf16 `hidden_states`, int64 `topk_ids` (global ids in `[0, 512)` or `-1` for a masked slot, which contributes zero), float32 `topk_weights`; `quantize_input=True` only |
| Routes per expert | at most 32 per forward (device-side overflow flag, see below) |
| Weights | `UnquantizedMoEWeights` bf16 `w13 [32, 10240, 3072]` (gate then up), `w2 [32, 3072, 5120]`; pre-quantized weights raise `NotImplementedError` |
| Stream | the current torch stream (`BootstrapConfig.stream` must be 0) |
| Workspace | pooled per (device, EP group, `max_tokens_per_rank`); all layers of a model share one |

Every rank publishes the same rank-uniform token bound: `compile_tokens_per_rank`
when passed to `MoEEpMegaLayer.stage_inputs`, else the workspace capacity. Rows
between a rank's live token count and the bound are padded (`-1` routes, zero
activations and weights) and never returned. Ragged per-rank token counts are
therefore supported; the padded rows still cost push and pack work.

FC2 writes a route's term only when `local_pack` packed the route, and `combine`
sums all eight slots of a token, so `publish_push` zeroes this rank's combine
read window of the term buffer (rows `[rank * bound, (rank + 1) * bound)` of the
`[1024, 8, 3072]` bf16 buffer) in-kernel before it releases its inbox flags (a
remote `fc2` writes a token's window only after that rank's `local_pack`
consumed the token's flag): a masked `-1` slot of a live token contributes zero
rather than a previous forward's term. The host performs no memset; the forward
is the five launches and nothing else.

### Push dispatch

There is no input barrier and no pull. `publish_push` (one CTA per staged token)
quantizes the token, zeroes its eight rows of this rank's combine read window,
resets the routing state and bulk-copies the token's 3232-byte slot
(`row_fp8[0, 3072) | scales[3072, 96) | ids_i32[3168, 32) | weights_f32[3200, 32)`)
into `push_inbox[destination][rank][token]` of every destination rank that owns
one of its eight experts (owner = `id // 32`); a destination that owns none
receives the 64-byte ids+weights tail only, so it can see that none of the
routes is its own. After the copies completed it stores `generation + 1` into
`push_flags[rank][token]` of every destination (release semantics). `local_pack`
(one CTA per (source rank, token)) waits on its LOCAL flag only with an untimed
system-scope acquire poll (`ld.acquire.sys.global.u32`; a late peer stalls this
rank, it never traps, exactly like the input barrier it replaces; the manifest
knob `PUSH_WAIT_TIMEOUT_CYCLES` is `"0"` in the shipped text), reads the LOCAL
slot through explicit coherent `ld.global` loads (the local inbox and flag
pointers carry no `__restrict__`, since peers write them while the kernel runs)
and packs the owned routes per expert exactly as the pull dispatch did (overflow
accounting included); the last CTA of the grid advances the rank-local
`push_generation[0]`. The inbox (`uint8 [16, 64, 3232]`, 3,309,568 bytes per
rank) and the flags (`uint32 [16, 64]`, 4096 bytes) are symmetric buffers zeroed
once at workspace allocation; `push_generation` (`uint32 [2]`) is rank-local and
zeroed once; the host never resets any of them. That zeroing must be complete
on every rank before any rank's first `publish_push`: `_allocate_workspace`
ends with a device synchronize and a process-group barrier (the rendezvous the
manifest's `initialization_rule` requires). A slot that is not pushed in a
forward (a token at or beyond the bound) keeps an older flag that never equals
the expected generation, so every rank must execute the same number of forwards
(one eager warmup, then eager forwards or graph replays in lockstep). A rank
overwrites a peer's inbox slot in forward `g + 1` only after its `combine(g)`
passed, which waited until every rank's `fc2(g)` had signalled completion; a
rank signals only at the exit of its `fc2(g)`, after its `local_pack(g)`
consumed the slot (manifest `route_contract.push_dispatch.reuse_rule`). The
peer tables are by-value 32-slot aggregates of which entries `[0, 16)` are
used, and on every rank a table's own entry is the local pointer bound beside
it (`push_inbox_peers[pg_rank] == push_inbox`, `push_flags_peers[pg_rank] ==
push_flags`, likewise `remote_terms_peers` / `fc2_completion_peers` of `fc2` and
`fc2_completion_peers` of `combine`): the programs poll or read a buffer through
the local pointer while the peers write it through the table, so a mismatch
would be a silent hang. `_allocate_workspace` asserts that symmetric self entry
once per workspace (`cake_symmetric.check_self_entry`) and every binding
re-checks it per launch (the flag pad is bound as a table only and has no
local pointer). `local_pack` is bound with `tokens_per_rank` = this forward's
bound, `active_rows_per_expert = 32` and `scale_tile_k128 = 1`, the constants
the exported FC1 program expects (`cake_plan.PUSH_*`, manifest
`route_contract.push_dispatch`).

### FC2 completion counters

There is no partial barrier. Every rank owns one symmetric `fc2_completion`
buffer, `uint32 [2]` (8 bytes; `cake_plan.FC2_COMPLETION_*`, manifest
`route_contract.fc2_completion`), zeroed once at workspace allocation together
with the inbox flags and `push_generation`, and never written by the host
afterwards. Word 0 is the done counter: after a `fc2` CTA stored its last tile's
routes, lane 0 of its first epilogue warp adds +1 to word 1 (this rank's CTA
arrival counter, `atom.acq_rel.gpu`); the last arriver returns word 1 to 0 and
its warp issues one warp-wide `red.release.sys.global.add.u32 +1` in which lane
`d` targets word 0 of rank `d` (itself included): 16 adds in one instruction.
No separate `fence.release.sys` precedes the reduction: its `.release`
qualifier is the signal's one system-scope release, cumulative over lane 0's
acquire on the arrival word and so over every CTA's route stores, so the
signal is exactly one system fence per rank per forward (`red.release.sys`
lowers to one `MEMBAR.ALL.SYS` + one `REDG`), nothing per tile or route and no
explicit fence in front of the reduction. Word 0 therefore grows by exactly 16
per forward and is never reset; between forwards every rank's counter reads
`[16 * forwards, 0]`.

`combine` is `fc2`'s programmatic dependent: every thread executes
`griddepcontrol.wait`, then thread 0 of each CTA reads `generation =
push_generation[0]` (the value `local_pack` of this forward stored, `g + 1`
during forward `g`) and polls its LOCAL word 0 untimed with
`ld.acquire.sys.global.u32` until `observed - 16 * generation < 2^31` unsigned
(a wrap-safe `>=`; a late peer stalls this rank, it never traps), then
`fence.acquire.sys` and `bar.sync`; all threads then read the eight term rows
with coherent `ld.global.v4.b32` loads through the plain (no `__restrict__`)
`local_terms` pointer and accumulate them in FP32 in route order, the same order
as before, so the output is bit-identical to the barrier route. No add of
forward `g + 1` can land before `combine(g)` passed (a rank's `fc2(g + 1)`
follows its `local_pack(g + 1)`, which waits for every source's flags of
`g + 1`, released after that source's `combine(g)`), so the target is reached
exactly and the modular compare stays valid across the u32 wrap (every 2^28
forwards); no host reset is ever needed (one would require a full barrier
first). The manifest knob `FC2_COMPLETION_COUNTERS` is `"1"` in the shipped
text (`"0"` would be the previous six-program route with a partial barrier and
the plain combine; the shipped programs do not contain it). The `fc2` binding
receives the counter and its 16-entry peer table as its two trailing device
arguments (`fc2_completion`, `fc2_completion_peers`); the `combine` binding
receives the same pair after `push_generation`, `tokens_per_rank`,
`source_rank` and the process-group arguments.

The SwapAB programs use static full-envelope tile maps: every forward streams
the weight tiles of all 32 local experts (32 x 40 FC1 tiles, 32 x 24 FC2
tiles) regardless of routing, and experts with zero routes produce no rows.
Latency therefore does not shrink with sparse routing; the contract rows keep
every local expert active on the critical-path ranks, so they carry no extra
cost.

### Routing overflow

The SwapAB programs compute one 32-row tile per expert and forward
(`active_rows_per_expert = 32`, the N32 route tile). `local_pack` packs a route
only while its expert has fewer than 32 routes; every further route to that
expert is not packed or computed and increments a device-side `overflow` counter
instead. `publish_push` zeroes the counter at the beginning of every forward, and the
forward never validates routing on the host (no per-call synchronization). The
backend exposes the counter:

```python
from flashinfer.experimental.cake_w8a8_megamoe_ep16 import routing_overflow

flag = routing_overflow(layer)   # int32 [1] device tensor of the layer's last forward
if flag.item():                  # the caller's explicit synchronization
    ...                          # that forward's output is not valid; the next forward resets the flag
```

### CUDA graphs

Warm up on all ranks first, then capture the bucketed call:

```python
layer.warmup()
with torch.cuda.graph(graph):
    layer.stage_inputs(t, compile_tokens_per_rank=bound)   # same bound on every rank
    out = layer.compute_staged(output=None)                 # workspace view
```

Capture without an explicit bound raises `MoEEpConfigError`.

## Binding interface

Each program has one hand-written tvm-ffi binding,
`csrc/cake_w8a8_megamoe_ep16/cake_w8a8_megamoe_ep16_<stage>_binding.cu`, exporting
exactly one function named `<stage>`; the Python side calls `module.<stage>(*args)`
with the program's device arguments in the manifest's physical order followed by
`grid_x` (all grids are one-dimensional; block size and dynamic shared memory are
fixed in the binding):

| Manifest kind | Binding parameter | Python argument |
|---|---|---|
| `buffer`, `tma` | `tvm::ffi::TensorView` (the binding encodes the tensor map) | the tensor (contract shape and dtype) |
| `parameter`, `process_group_scalar` | `int64_t` (range-checked to i32) | `int` |
| `peer_table` | `tvm::ffi::Shape` of exactly 16 entries | `tvm_ffi.Shape` of 16 signed int64 address images (built once per buffer), slot r = rank r, all non-null |
| (all) | trailing `int64_t grid_x` | the stage grid |

The symmetric buffers travel as their local tensor followed by their peer table;
the binding copies the 16 entries into slots 0..15 of the programs' 32-slot device
table and zeroes the rest. Every program has a tensor argument, so every binding
selects its device and stream from the first tensor. Where a program indexes by
block without a guard (`publish_push`, `local_pack`, `combine`) the binding also
checks `grid_x` against the stage's grid formula; the persistent programs and
`quantize_weights` accept any positive grid. The `fc1`/`fc2` bindings opt their
kernel into its dynamic shared memory once per process and the `local_pack`
binding sets its shared-memory carveout once per process, so a process drives
one CUDA device (the EP16 deployment: one rank per GPU). The host test
`test_binding_sources_match_manifest_and_python_args` parses every binding's
export name, kernel declaration, block size, launch and function attributes and
`Run(...)` signature against the manifest and the Python argument tuples.

Programmatic dependent launch: the `fc1`, `fc2` and `combine` bindings (exactly
the programs whose manifest `launch.use_pdl` is true) launch with
`cudaLaunchAttributeProgrammaticStreamSerialization`; these are the route's
three PDL edges `local_pack -> fc1`, `fc1 -> fc2` and `fc2 -> combine`
(`cake_plan.PDL_EDGES`). `local_pack` executes `griddepcontrol.launch_dependents`
at entry (before it waits on its inbox flags), `fc1` and `fc2` when a CTA begins
its last tile (after their own wait); `fc1`/`fc2` prefill their first
expert-weight tile through the static tile map and `griddepcontrol.wait` before
they read the routed operands, and `combine` executes `griddepcontrol.wait` in
every thread before its first global read and only then polls the completion
counter, so a dependent never observes its primary's output early. `combine`
CTAs (128 threads, no shared memory, at most 192 per rank) block in the wait
beside the fully resident `fc2` grid without consuming issue slots, so the
overlap can neither starve nor deadlock it. `publish_push` and `local_pack`
launch without attributes (`publish_push` contains no `griddepcontrol`
instruction). The forward stays five launches; under CUDA-graph capture the
three attributed launches become programmatic graph edges (CUDA 12.3 or newer),
and the device programs are correct without the attribute (`griddepcontrol` is
then a no-op). `local_pack`
is the one program with a manifest `launch.preferred_shared_memory_carveout`
(100): its binding calls
`cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100)`
once per process, the same once-per-process pattern as the `fc1`/`fc2`
dynamic-shared-memory opt-in. No other program sets a carveout: `fc1`/`fc2`
already request 159 KiB and 177 KiB of dynamic shared memory through that opt-in,
`publish_push` uses 3328 bytes and the remaining programs use none.

Weights: `quantize_weights` runs twice at construction, mode 1 for `w13` (canonical
gate||up rows interleaved as up128|gate128 per 256-row chunk; one 1024-byte scale
tile per 256-row x 128-column weight tile) and mode 2 for `w2` (one 512-byte scale
tile per 128-row x 128-column weight tile). `cake_weights.pack_scales_cp` states
both layouts in pure Torch for the tests.

The FP8 payload of each weight is stored k-tile-major (the program's trailing
`ktile_major_rows` scalar, 256 for `w13` and 128 for `w2`): byte `(e, r, k)` of a
`[E, rows, K]` weight lives at
`(((e * (rows / R) + r / R) * (K / 128) + k / 128) * R + r % R) * 128 + k % 128`,
i.e. `[E][rows / R][K / 128][R][128 B]`, so every FC1/FC2 A tensor-map box
(`{128, R, 1, 1}` over the view `{128, R, K / 128, E * rows / R}` with byte
strides `{128, R * 128, R * K}`) is one contiguous `R x 128`-byte block. The host
tensors keep their contiguous `[E, rows, K]` shape; the scale carriers are
unchanged. `TransformedCakeW8a8MegaWeights` carries the layout tags
(`w13_layout = "ktile_major_r256_k128"`, `w2_layout = "ktile_major_r128_k128"`)
and `validate_transformed_mega_weights` rejects any other payload: the FC1/FC2
programs do not read a row-major payload. `cake_weights.reference_ktile_major_payload`
states the placement in pure Torch
(`q.view(E, rows // R, R, K // 128, 128).permute(0, 1, 3, 2, 4).contiguous().view(E, rows, K)`).

## Files

| File | Role |
|---|---|
| `cake_config.py` | the fixed-value config dataclass |
| `cake_backend.py` | `MegaKernelBackend` subclass, workspace, stage call interface |
| `cake_weights.py` | weight preprocessing through the quantization program; pure-Torch quantization rule and scale layouts for tests |
| `cake_symmetric.py` | NVSHMEM symmetric buffers (inbox, flags, term buffer, fc2 completion counter), peer tables, flag pad |
| `cake_staging.py` | input validation, rank-uniform bound, padding |
| `cake_plan.py` | geometry constants, grids, static tile maps, scale sizes |
| `cake_jit.py` | exact-`sm_103a` JIT spec (`cake_w8a8_megamoe_ep16_sm_103a_<digest>`) |
| `csrc/` | generated device programs (`*_kernel.cu`), hand-written bindings (`*_binding.cu`), manifest |

Core touch points: `flashinfer/moe_ep/__init__.py` exports the config, the
transformed-weights type and `preprocess_sm103_mxfp8_cake_mega_weights`;
`flashinfer/moe_ep/backends/mega/kernel/sm103/__init__.py` imports this package
so the kernel name registers. The package is lightweight at import time: tvm-ffi,
the JIT loader (`cake_jit`) and NVSHMEM are imported inside the functions that
use them, so importing `flashinfer.moe_ep` loads no kernel or JIT module.

## Tests, example, benchmark

* Host tests (no GPU): `pytest tests/experimental/test_cake_w8a8_megamoe_ep16.py -k "not test_ep16"`.
* 16-rank tests (four 4-GPU SM103 nodes, one NVLink domain):
  `torchrun --nnodes 4 --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint $HOST:29500 -m pytest tests/experimental/test_cake_w8a8_megamoe_ep16.py -k test_ep16`.
* Example: `examples/experimental/cake_w8a8_megamoe_ep16.py` (torchrun, 16 ranks).
* Benchmark: `benchmarks/moe_ep/bench_cake_w8a8_megamoe_ep16.py` compares this
  backend with `sm100_mxfp8_mxfp8_bf16_cutedsl` (inference and GenPhase) in one
  process over the six contract rows.

## Limitations

* `sm_103a` only; EP16 decode geometry only; no pre-quantized weights; no knobs.
* Capture needs an explicit `compile_tokens_per_rank`.
* Numerics: BF16-rounded route terms before the FP32 combine. The 16-rank test
  gates the output against the pure-Torch block-32 MXFP8 oracle at
  `atol = rtol = 0.1` and relative L2 error `<= 0.02`.
* One CUDA device per process (the dynamic shared-memory opt-in of `fc1`/`fc2`
  and the shared-memory carveout of `local_pack` run once per process).
* Every rank must execute the same number of forwards (the inbox flags and the
  fc2 completion counter are generation-counted and never reset).
