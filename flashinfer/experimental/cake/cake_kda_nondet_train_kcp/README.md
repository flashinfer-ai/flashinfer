# Cake non-deterministic context-parallel (KCP) packed KDA chunked training backend (SM100 / SM103 / SM107)

Backend of the context-parallel route of the packed Kimi Delta Attention
(KDA) chunked training forward and backward, the non-deterministic
counterpart of `flashinfer.experimental.cake.cake_kda_det_train_kcp`: the
packed batch is split over the ranks of a `torch.distributed` process group
and the per-document state recurrences cross the rank boundaries exactly.
The public entry points are in `flashinfer/cake_kda_nondet_train_kcp.py`; each
rank runs the chunk stages of `flashinfer.experimental.cake.cake_kda_nondet_train`
(the non-deterministic single-rank package) at its own shape.  The entry
points of `flashinfer.cake_kda_nondet_train_kcp` are a stable top-level API;
the modules of this package are their backend.

### Numerical mode

* The package is the mode: no entry point of this package or of the facade
  `flashinfer.cake_kda_nondet_train_kcp` takes a mode keyword, and none runs
  the deterministic package.  A `CakeKDADetKcpWorkspace` or a saved set of the
  deterministic package passed to an entry point here raises `ValueError`
  naming the mode.
* The KCP programs (the K1 / K2 state walks, the K3 transfer walk, the K4
  affine merge, `kcp_fwd_o`, `kcp_dav`) and the exchanges are copies of the
  deterministic package's, shipped under this package's library prefix.  Every
  output tile has one writer and the cross-rank merge runs in the
  recurrence's own order, so they have no reduction order to relax.
* The rank-local chunk stages take the records of the non-deterministic
  single-rank package: the forward (path `fwd_nosave`) runs copies of the
  deterministic programs; the backward (path `replay`, in both backward modes)
  runs that package's `S1` pair on the rows its registry lists: the fused
  gradient leaves the `q * dq` term of `dg` to the GLM intra program, which
  adds it from the FP32 `dq` it already holds.  The pair is selected together
  per rank; the single-rank plan refuses a mixed pair.

| arch | KCP backward (replay path) | 9 to 15 heads |
|---|---|---|
| `sm_100a` | `S1`: `dqkg_int_davq_v24` + `glm_intra_v21` (+ `gate_epilogue`) | deterministic pair |
| `sm_103a` | up to 8 heads `S1`: `dqkg_int_davq_v24` + `glm_intra_v22`; from 16 heads the deterministic pair (`dqkg_int_davq` + `glm_intra_v14`) | deterministic pair |
| `sm_107a` | provisional `S1`: `dqkg_int_davq_v24` (from 16 heads `dqkg_int_davq_nohint_v24`) + `glm_intra_v23` | deterministic pair |

* The schedule has a fixed order and no atomic adds.  At the same topology
  repeated calls give bitwise the same results, and the saved-mode and the
  checkpoint-replay backwards run the same programs (bitwise equal, as in the
  deterministic package).  `out`, `dq`, `dk`, `dv` and `dbeta` are bitwise the
  deterministic package's; `dg` differs from it by FP32 rounding on the `S1`
  rows (the same FP64-referenced accuracy budget).  No launch is unordered,
  so no entry raises an alert or a warning under
  `torch.use_deterministic_algorithms(True)` (also with `warn_only=True`);
  the entries never read that setting and never run the deterministic
  package.
* A partition, a context and the byte queries hold no mode: they describe the
  same plan in both packages, and a context of either package serves the calls
  of both (each package keeps its own launch tables on it).  A saved set is
  read by the backward of the package that wrote it (`CakeKDANondetKcpSaved`
  here), and a workspace (`CakeKDANondetKcpWorkspace`) serves this package
  only.
* The raw gate mode (`use_gate_in_kernel=True` with `A_log`, `dt_bias` and
  `lower_bound`, FlashInfer #7816) runs on this package too ("Raw gate mode"
  below): the rank's local stages take the single-rank package's raw records,
  the deterministic programs on every row (no `S1` row runs in this mode), so
  every output, the raw gate's gradient and the per-rank parameter partial
  sums are bitwise those of the deterministic package, with the in-kernel
  q / k L2 norm and without it.
* The in-kernel q / k L2 norm (`use_qk_l2norm_in_kernel=True`, #7817) runs on
  this package too ("In-kernel Q / K L2 norm" below): the rank's local stages
  take the single-rank package's records of the mode, copies of the
  deterministic programs (no `S1` row runs in this mode, in either gate mode),
  so every output, the saved rstd and every gradient are bitwise those of the
  deterministic package.
* These bitwise statements hold for the light saved set (`saved_rows="none"`,
  and `"auto"` / `"tape"` wherever the checkpoint hand-off does not engage) and
  for the hand-off's tape set (`"tape"`).  A saved set that keeps a rows set of
  `saved_rows` ("Saved rows" below; `"auto"` keeps the `"wy"` set where the
  hand-off engages) skips the backward's intra program, and with `"wy"` its WY
  program: the forward's rows stand in for the backward's recompute, so the
  gradients are held to the light path's FP64 error budget, not bitwise.

### Per-output determinism and accuracy

Every output repeats bit for bit: two calls on the same inputs and the same topology return bitwise-equal tensors.  Errors against an FP64 reference recurrence (GB300, BF16 inputs, 10 rows at CP8, 9 at CP16, 3 at CP32 and 2 at CP64 with 1M-token documents, including adversarial inputs):

| output | determinism | rel-L2 vs FP64, CP8 (x1e-3) | rel-L2 vs FP64, CP16 (x1e-3) | rel-L2 vs FP64, CP32 (x1e-3) | rel-L2 vs FP64, CP64 (x1e-3) | ratio to det@2e989556, CP8 | ratio to det@2e989556, CP16 | ratio to det@2e989556, CP32 | ratio to det@2e989556, CP64 | ratio to single-rank, CP8 | ratio to single-rank, CP16 | ratio to single-rank, CP32 | ratio to single-rank, CP64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| out | repeats bit for bit | 4.13-6.47 | 4.12-5.57 | 4.13 | 4.13 | 1.000 | 1.000 | 1.000 | 1.000 | 0.840-1.000 | 0.924-1.000 | 1.000 | 1.000 |
| dq | repeats bit for bit | 3.40-6.14 | 3.40-5.00 | 3.40 | 3.40 | 1.000 | 1.000 | 1.000 | 1.000 | 0.809-1.000 | 0.906-1.000 | 1.000 | 1.000 |
| dk | repeats bit for bit | 3.50-6.18 | 3.50-5.90 | 3.50 | 3.50 | 0.996-1.000 | 0.996-1.000 | 1.000 | 1.000 | 0.846-1.000 | 0.927-1.000 | 1.000 | 1.000 |
| dv | repeats bit for bit | 3.74-5.93 | 3.72-5.00 | 3.79 | 3.79 | 0.980-0.999 | 0.981-1.000 | 0.992 | 0.992 | 0.839-1.000 | 0.926-1.000 | 1.000 | 1.000 |
| dg | repeats bit for bit | 3.53-6.51 | 3.53-6.09 | 3.53-3.94 | 3.53-3.94 | 0.998-1.000 | 0.998-1.000 | 1.000 | 1.000 | 0.736-1.000 | 0.867-1.001 | 1.000 | 1.000 |
| dbeta | repeats bit for bit | 3.75-10.68 | 3.74-5.55 | 3.79 | 3.79 | 0.986-0.999 | 0.985-1.000 | 0.992 | 0.992 | 0.839-1.001 | 0.927-1.000 | 1.000 | 1.000 |
| dA_log (raw-gate mode) | repeats bit for bit | 3.23-5.37 | 2.67-5.16 | 3.38 | 3.38 | 0.957-0.998 | 0.998-1.137 | 1.005 | 1.005 | 0.839-1.243 | 1.000-1.407 | 1.000 | 1.000 |
| ddt_bias (raw-gate mode) | repeats bit for bit | 3.57-5.58 | 3.56-5.51 | 4.96 | 4.96 | 0.997-1.000 | 0.993-1.001 | 1.000 | 1.000 | 0.519-1.023 | 0.730-1.025 | 1.000 | 1.000 |

| output | worst-element ratio to det@2e989556, CP8 | worst-element ratio to det@2e989556, CP16 | worst-element ratio to det@2e989556, CP32 | worst-element ratio to det@2e989556, CP64 |
| --- | --- | --- | --- | --- |
| out | 1.000 | 1.000 | 1.000 | 1.000 |
| dq | 1.000 | 1.000 | 1.000 | 1.000 |
| dk | 0.959-1.064 | 0.922-1.000 | 1.000 | 1.000 |
| dv | 0.974-1.003 | 0.802-1.000 | 1.000 | 1.000 |
| dg | 0.905-1.003 | 0.901-1.021 | 1.000 | 1.000 |
| dbeta | 0.909-1.154 | 0.886-1.030 | 1.000 | 1.000 |
| dA_log (raw-gate mode) | 0.915-1.005 | 0.993-1.024 | 0.987 | 0.987 |
| ddt_bias (raw-gate mode) | 0.996-1.152 | 0.961-1.001 | 0.998 | 0.998 |

rel-L2 = ||x - ref|| / ||ref||; worst element = max|x - ref| / max|ref|.  Each cell is the range over the measured rows of the median over 5 seeds.  Ratio to det@2e989556 = median over seeds of this package / median over seeds of the deterministic package at FlashInfer 2e989556 on the same inputs (bound 1.1).  Ratio to single-rank = worst seed of this package / the single-rank package on the same inputs, rel-L2 (bound 1.10 for `out`, `dq`, `dk`, `dv`, `dg` and `dbeta`; reported for `dA_log` and `ddt_bias`, which hold one value per head).

Three adversarial cells are above 1.1 at 5 seeds and within it over 25 seeds: CP8 `dbeta` worst element 1.154 at 5 seeds, 0.932 at 25 seeds; CP8 `ddt_bias` worst element 1.152 at 5 seeds, 1.026 at 25 seeds; CP16 `dA_log` rel-L2 1.137 at 5 seeds, 1.022 at 25 seeds.  The worst-element metric follows a single element per tensor, and `dA_log` holds one value per head (16 here); the deterministic package at 2e989556 itself varies 1.6-2.5x across seeds on these three rows.

Determinism check: two calls on the same inputs and topology gave bitwise-equal outputs on every measured row and seed (out 24/24 rows, dq 24/24 rows, dk 24/24 rows, dv 24/24 rows, dg 24/24 rows, dbeta 24/24 rows, dA_log 14/14 rows, ddt_bias 14/14 rows).

### Names

| deterministic package (`flashinfer.cake_kda_det_train_kcp`) | this package (`flashinfer.cake_kda_nondet_train_kcp`) |
|---|---|
| `cake_kda_det_train_kcp_forward` | `cake_kda_nondet_train_kcp_forward` |
| `cake_kda_det_train_kcp_backward` | `cake_kda_nondet_train_kcp_backward` |
| `cake_kda_det_kcp` | `cake_kda_nondet_kcp` |
| `cake_kda_det_train_kcp_partition` | `cake_kda_nondet_train_kcp_partition` |
| `cake_kda_det_train_kcp_build_context` | `cake_kda_nondet_train_kcp_build_context` |
| `cake_kda_det_train_kcp_workspace_bytes` | `cake_kda_nondet_train_kcp_workspace_bytes` |
| `cake_kda_det_train_kcp_saved_bytes` | `cake_kda_nondet_train_kcp_saved_bytes` |
| `cake_kda_det_train_kcp_comm_bytes` | `cake_kda_nondet_train_kcp_comm_bytes` |
| `cake_kda_det_train_kcp_reduce_gate_param_grads` | `cake_kda_nondet_train_kcp_reduce_gate_param_grads` |
| (none: no checkpoint hand-off) | `cake_kda_nondet_train_kcp_handoff_bytes` |
| (none: no checkpoint hand-off) | `cake_kda_nondet_train_kcp_handoff_stats` |
| `CakeKDADetKcpRecomputeHandle` | `CakeKDANondetKcpRecomputeHandle` |
| `CakeKDADetKcpWorkspace` | `CakeKDANondetKcpWorkspace` |
| `CakeKDADetKcpSaved` | `CakeKDANondetKcpSaved` |
| `CakeKDADetKcpOptions` | `CakeKDANondetKcpOptions` |
| `CakeKDADetKcpCall` | `CakeKDANondetKcpCall` |
| `CakeKDADetKcpContext` | `CakeKDANondetKcpContext` |
| `CakeKDADetKcpPartition` | `CakeKDANondetKcpPartition` |
| `CakeKDADetKcpFragment` | `CakeKDANondetKcpFragment` |
| `CakeKDADetKcpFunction` | `CakeKDANondetKcpFunction` |

## Usage

The packed batch of one training step may be split over the ranks of a
`torch.distributed` process group.  `cake_kda_nondet_train_kcp_partition` cuts the documents into
per-rank fragments (`"contiguous"`, `"zigzag"` or `"zigzag_per_document"`
layouts, an explicit `fragments` table, or `cut_points` with the
`"zigzag_per_document"` layout); `cake_kda_nondet_train_kcp_build_context` gives every rank the
device tables of its fragments, the collectives that carry the per-document
state recurrences and the convolution's history rows across the rank
boundaries, and the segment plan of its fragment walks.  The per-rank entry
points take the rank's shard of the operands (the tokens of its fragments, in
the partition's order) and return that shard's outputs and gradients:

```python
from flashinfer.cake_kda_nondet_train_kcp import (cake_kda_nondet_train_kcp_partition, cake_kda_nondet_train_kcp_build_context, cake_kda_nondet_kcp,
                                           cake_kda_nondet_train_kcp_forward, cake_kda_nondet_train_kcp_backward,
                                           cake_kda_nondet_train_kcp_workspace_bytes, cake_kda_nondet_train_kcp_saved_bytes, cake_kda_nondet_train_kcp_comm_bytes)
from flashinfer.cake_causal_conv1d_det_train_kcp import cake_causal_conv1d_det_kcp

part = cake_kda_nondet_train_kcp_partition(cu_seqlens_global_cpu, world_size)      # the same object on every rank
ctx = cake_kda_nondet_train_kcp_build_context(part, group, rank, device)           # this rank's tables and collectives
y_local = cake_causal_conv1d_det_kcp(x_local, conv_weight, ctx)    # the convolution before the KDA step
out_local = cake_kda_nondet_kcp(q_local, k_local, v_local, g_local, beta_local, ctx)   # autograd
out_local, saved = cake_kda_nondet_train_kcp_forward(q_local, k_local, v_local, g_local, beta_local, ctx,
                                     save_mode="light", boundary_mode="saved")
dq, dk, dv, dbeta, dg = cake_kda_nondet_train_kcp_backward(q_local, k_local, v_local, g_local, beta_local, ctx,
                                          dout_local, saved=saved)
```

The context is a plain frozen object, not a dataclass: the pre-forward hooks
of FSDP2 (`fully_shard`) walk every dataclass, dict, list and tuple of a
module's arguments, and see the context as one leaf. Per-layer metadata that
carries it costs no walk time per fragment.

CUDA-graph replay: the three KDA entry points -- `cake_kda_nondet_kcp`,
`cake_kda_nondet_train_kcp_forward` and `cake_kda_nondet_train_kcp_backward` -- take `graph="replay"`
(default `"off"`, the eager path); the convolution halo takes `graph=` only
through the backend module's function, the `cake_causal_conv1d_det_kcp` facade
has no such parameter. The first call of a
validity key runs eagerly, the second is captured into a CUDA graph and every
later call of the key replays it: the same launches and collectives, bitwise
the eager result, with no host work between them. The key is the call's kind,
its `CakeKDANondetKcpContext` / workspace / backend objects, every option and the shape,
stride, dtype and data pointer of every operand and saved tensor.
Requirements: the operands of a layer must keep their data pointers, shapes,
strides and dtypes across steps (static activation buffers), the same context,
workspace and backend objects must be reused, and a layer's outputs and saved
set must be consumed before its next call (a replay overwrites the graph's
static tensors). Any other call runs eagerly without an error (`GRAPH_STATS`
and `LAST_CALL[kind]["graph_event"]` say what happened; a capture failure
disables that key for the process), `exchange="ring"`, `debug_sequence` and
calls without a caller-owned workspace are never captured. Memory: every
cached key pins a private memory pool -- 0.7-9.0 GiB per rank per key on the
A/B rows (the per-layer forward and backward of CP4-CP16 at the B1 /
B3 batch sizes), at most `GRAPH_CACHE_MAX = 64` keys (the oldest key is
evicted). Call
`graph_cache_clear()` (both modules) before
`torch.distributed.destroy_process_group()` and whenever a layer's operands
are gone for good. A replayed call keeps the tape programs at one CTA per SM:
the device cache-config bracket of the eager path cannot be issued inside a
capture, so right after each tape launch is captured the host sets the
preferred shared-memory carveout as a launch attribute of the kernel node it
just added (`cudaStreamGetCaptureInfo` + `cudaGraphKernelNodeSetAttribute`,
host-only `ctypes` calls into the runtime that is already loaded) and the
replayed graph carries the cap itself. Without those entry points, or when the
runtime cannot name the captured node, the capture continues without the cap
after one warning (`LAST_CALL[kind]["cache_config"] == "unavailable"`); the
results are bitwise identical either way. The previous replay form stays
reachable for A/B records through the private test knob `graph_cap`
(after `cake_kcp_plan.enable_test_knobs()`); `graph="off"` is the shipped
default. The A/B
record was taken with pointer-stable harness inputs; its gain is an upper
bound of what a framework sees.

Limits of a partition: at most 65536 fragments per rank (the layout table's
document search), at most 64 segments per fragment and no walk chain longer
than 256 chunk steps (the segment plan's bounds); the byte queries
`cake_kda_nondet_train_kcp_workspace_bytes(ctx, H=...)`, `cake_kda_nondet_train_kcp_saved_bytes(ctx, H=...)` and
`cake_kda_nondet_train_kcp_comm_bytes(ctx, H=..., C=...)` size the rank's workspace, saved set and
one step's traffic before any allocation.  Every rank of the group calls the
same entry points in the same order; the collectives are issued inside them.

Rank agreement: the ranks of `ctx.group` must agree on what sets the count,
order and split sizes of the collectives and the form of the recurrence.  The
first call of each key on a context checks that, right before its first
collective (the forward's `c1` or first ring hop; a saved-mode backward's `b1`
or first ring hop; a replay-mode backward's replayed `c1`): every rank
all-gathers a 16-byte BLAKE2b digest of the inputs below
(`all_gather_into_tensor` of `int64[2]` per rank, on the compute stream) and a
device-side assert (`torch._assert_async`) checks that every rank's digest
equals rank 0's.  A disagreement now trips that assert at the first call
(`CUDA error: device-side assert triggered` at the rank's next CUDA call, or
from the NCCL watchdog) instead of hanging in, or exchanging wrong bytes
through, the state exchange.  No host synchronization: the digest goes up in
one pinned non-blocking copy.  Later calls of the key issue nothing; neither
does a group of one rank, a call without an initialized process group, a
CUDA-graph capture or replay (`graph="replay"` runs a key's first call
eagerly, so the key is checked before its capture), nor the recompute of a
call with a recompute handle (it exchanges nothing; its original forward
checked the key).  The check is not counted in `LAST_CALL[kind]["n_collectives"]`
or `["sequence"]` (the exchange's collectives); `LAST_CALL[kind]["rank_check"]`
holds the digest of the call that issued it and `RANK_CHECK_STATS["issued"]`
counts the checks of the process.

| must agree on every rank (in the digest, `cake_kcp_plan.RANK_CHECK_FIELDS`) | rank-local (may differ; not in the digest) |
|---|---|
| the partition: world size, layout, global `cu_seqlens`, every rank's fragments | the rank and its operands' tokens, strides and data pointers |
| the context's `conv_kernel_size` | the SM count and the plan rule, hence the segment count `P` (the splits use the crossing counts and `H` only) |
| `H` (the value-head count of the operands) | the walk geometry and form, the device cache-config bracket |
| the resolved exchange (`a2a` / `ring`) and the resolved `save_transfer` (with a handle: its `checkpoint_keep` tier) | the checkpoint hand-off decision (`checkpoint_handoff`) |
| the boundary mode: a saved set (`boundary_mode="saved"` with a saved set) or the replay mode | `graph` (a captured call replays the collectives of its eager form) |
| `debug_sequence` | the workspace, the schedule test knobs, `spec3_probe` |
| whether a recompute handle is passed | |
| the gate mode (`use_gate_in_kernel`) and `use_qk_l2norm_in_kernel` | |

Not covered: the check runs once per key, so a rank that calls a key it has
not checked while another rank calls one it has (the ranks then no longer run
the same entry points in the same order) still hangs; the checkpoint nesting
level of a recompute handle is not part of the digest; and a refusal raised on
one rank before the first collective (an operand check, the L2 norm's
base-address alignment) leaves the other ranks waiting in that collective.

Workspace of a rank (`CakeKDANondetKcpWorkspace`, caller-held, reused across calls and
layers): the local arena of the token-side stages plus the boundary pools.
The backward's b1 / b2 send pools reuse the storage of the forward's c1 / c2
send pools (a call waits its sends before a backward writes either), and the
ring exchange's hop pools are allocated by the first `exchange="ring"` call.
`cake_kda_nondet_train_kcp_workspace_bytes` is the number of bytes an a2a workspace allocates
(`CakeKDANondetKcpWorkspace.bytes` after its first backward); the saved set and the
convolution's halo rows are per-call tensors outside it.  A workspace that
has served forwards only (`cake_kda_nondet_train_kcp_workspace_bytes(..., backward=False)`) holds
its local arena in the KCP forward layout (`fwd_kcp`: 2432 B per internal
token-head against at least 2560 for the existing backend's forward layout;
no segment state blocks, the internal `Akk` rows under the `h` tape) and leaves
the c1 receive, transfer staging and carry pools to their first reader.  A
forward with a saved set writes a fresh carry plane in place of the carry
pool and, with `save_transfer=True` (the a2a default), fresh saved-set
tensors in place of the other two; with `save_transfer=False` it allocates
those two, so pass the forward's `save_transfer` to the byte query.  A
forward without a saved set allocates the rest
(`cake_kcp_plan.kcp_arena_layout(ctx, H=H, backward=False,
save_transfer=...)["lazy"]` bytes more).  Under `exchange="ring"` such a
workspace never allocates the c1 receive pool.  The first backward on such a
workspace releases its forward arena before it allocates the backward one;
outputs, saved sets and gradients are bitwise those of a backward workspace.

One workspace serves every prepared pack of a rank whose calls run in stream
order, so the workspace memory does not grow with the number of prepared
packs: its pools hold the largest need of the packs and its local arena the
largest layout, and a smaller pack runs on a view of that arena under its own
plan.  `ws.reserve([ctx_a, ctx_b, ...], H)` sizes it for a list of contexts
before the first call (host work and allocations only; `ring=True` also
sizes the ring exchange's hop pools); the calls of those
packs then allocate nothing, keep every pointer (so graph keys stay valid) and
never re-size it -- a forward never turns a backward arena into a forward-only
one.  `cake_kda_nondet_train_kcp_workspace_bytes([ctx_a, ctx_b, ...], H=H)` is its size: each pool at
its largest need, the largest local arena and, where the intra record writes
one, each pack's scratch table set (KiB scale; none at 64 heads).  No content
crosses calls, so outputs, saved sets and gradients are bitwise those of one
workspace per pack.  The workspace holds no context alive.  It records the
stream of its last call; a call on another stream first waits for that
stream's queued work on the device (one `wait_stream`, no host wait) and
marks the workspace's storages as used on its own stream (`record_stream`),
so a pool or arena that a later call replaces, or a dropped workspace, is
reused by the caching allocator only after every stream that used it has
run its queued work.  Calls on one stream do neither.
`CakeKDANondetKcpWorkspace(ctx, H, allocate=False)` allocates nothing until its first call
and sizes that call for the backward, so a first forward does not build a
forward-only arena that the first backward replaces.

Saved set of a rank (`CakeKDANondetKcpSaved`, `boundary_mode="saved"`): the fresh `S_in`
(the initial states of its fragments), with `save_transfer` the owner's c1
receive tensor and `M_seg`, on a rank whose fragments are split into `P > 1`
segments the forward's carry plane (the saved-mode backward replays its tape
from it instead of re-walking the segments and re-merging the carries; the
private test knob `saved_carry=0` selects the re-walk), and the call's
scalars; the int32 tables are rebuilt from the context.  The token-row `Aqk` /
`Akk` of the single-rank saved set are not part of it.  By default the backward
recomputes the internal rows in both modes (saved set and checkpoint replay run
the same local programs), so `cake_kda_nondet_train_kcp_saved_bytes` counts the tables, `S_in`, the
`save_transfer` tensors and the carry plane (and the convolution's history
rows with `with_conv`), plus the kept rows of `saved_rows` below.

Saved rows (`saved_rows`, keyword-only, on `cake_kda_nondet_kcp` and
`cake_kda_nondet_train_kcp_forward`; `"auto"` default, `"none"`, `"intra"`,
`"wy"`, `"tape"`): `"intra"` keeps the forward's internal `Aqk` / `Akk` rows (`aqk_int` /
`akk_int`, BF16 `[R, H, 64]`), `"wy"` also its WY rows `w` / `u` (BF16
`[R, H, 128]`), in the saved set (`CakeKDANondetKcpSaved.rows` /
`.saved_rows`).  The backward then runs no intra program, and with `"wy"` no WY
program.  It still runs the prep and replays the K1 tape from `S_in` and the
carry plane.  The forward runs the same launches and collectives; only the
destinations of those rows move to fresh tensors.  With the in-kernel Q / K L2
norm the backward first runs the separate normalization from the saved rstd
where the skipped intra program was the first q / k reader (as the checkpoint
hand-off does).  The rows cost `R * H * 256` / `R * H * 768` bytes
(`cake_kda_nondet_train_kcp_handoff_bytes(ctx, H=..., saved_rows=...)`;
`cake_kda_nondet_train_kcp_saved_bytes(..., saved_rows=...)` counts them, the carry plane
stays).  For `"wy"` that is about 792 MiB per layer at 16384 tokens per rank,
H = 64, CP4 (R = 16896), linear in the tokens, held from the forward to the
node's backward.  The choice is rank-local and moves no collective.  The
gradients are held to the light path's FP64 error budget, not bitwise those of
the light set.  `cake_kda_nondet_kcp` decides once per call, in the caller's
frame:

| `saved_rows` | where | kept beside the light set |
| --- | --- | --- |
| `"auto"` | a checkpoint pass the hand-off engages on (the hand-off's rule, below) | the `"wy"` rows set instead of the tape set |
| `"auto"` | anywhere else | nothing |
| `"tape"` | a checkpoint pass the hand-off engages on | the tape set (the earlier `"auto"`) |
| `"tape"` | anywhere else | nothing |
| `"none"` | everywhere | nothing (`checkpoint_handoff="on"` raises `ValueError` on every call, in a checkpoint pass or not) |
| `"intra"` / `"wy"` | a checkpoint pass the hand-off engages on | that rows set instead of the tape set |
| `"intra"` / `"wy"` | a checkpoint pass the hand-off refuses (`checkpoint_handoff="off"` included) | nothing |
| `"intra"` / `"wy"` | no checkpoint, grad on, an input that requires grad, no saved-tensor hooks, not inside a backward | that rows set |
| `"intra"` / `"wy"` | under a caller's saved-tensor hooks (`save_on_cpu`, an offload) | nothing (counted in `SAVED_ROWS_STATS["refused_hooks"]`) |
| any | no grad, no input requires grad, `boundary_mode="replay"`, or a rank without tokens | nothing |

An input is any of `q`, `k`, `v`, `g`, `beta`, `A_log` and `dt_bias`.
Outside a checkpoint pass `checkpoint_handoff="off"` does not stop the rows.

The functional forward keeps no rows under `"auto"` / `"tape"` and keeps the
named set under `"intra"` / `"wy"` (a saving forward of a rank with tokens).
`"intra"` / `"wy"` are refused with `spec3_probe`.  `LAST_CALL[...]["saved_rows"]`
and `["saved_rows_bytes"]` report the set a pass kept or read;
`SAVED_ROWS_STATS` (a dict of the module
`flashinfer.experimental.cake.cake_kda_nondet_train_kcp.cake_kcp`; the facade
does not export it) counts the forwards that kept one and the backwards that
read one, once per eager call and once per graph capture (never in the
hand-off's counters).  Under `graph="replay"` the rows
are part of the backward's validity key and live in the captured forward's
static saved set.  Outside a checkpoint the default keeps no rows: there the
rows forms are opt-in (they add their bytes per layer until each backward).

Walk-launch geometry: the walk kernels' CTA lists are issued
longest-segment-first and the two tape programs (`fwdh_tape`, `dhu_tape`) run
at one CTA per SM. Because the JIT binding exposes no per-function carveout
attribute, the host sets the device-wide cache-config preference around each
tape launch (`cudaDeviceSetCacheConfig`: PreferL1 before, PreferNone after)
and restores it before the next launch; a launch issued by another thread of
the same process during that bracket would take the PreferL1 occupancy for
itself (occupancy only -- results never depend on the geometry). The earlier
geometry stays reachable for A/B records through the private test knob
`walk_geometry` (after `cake_kcp_plan.enable_test_knobs()`).

Output backward: the K2 adjoint walks run their forms' `fuse_dv1` builds,
which form the first value gradient `dv1 = Aqk^T dO` in FP32 inside the step
from the internal `Aqk` rows (`aqk_int`), add `kg bf16(dh)` and round once.
So `kcp_dav` launches only where its `dA_qk` has a reader: where the
single-rank plan dispatches its `dav` stage on the replay path
(`cake_kcp_plan.kcp_dav_needed`: the serial route below 16 heads, H 9-15),
whose fused gradient stores no `dA_qk`.  Everywhere else the fused gradient
stores every `dA_qk` row and the `dav` stage launches nothing (still counted
in the call's launches, like a fused-away gate epilogue).
`LAST_CALL["bwd"]["kcp_dav"]` records `"launched"` or `"skipped"`.  The
private test knob `kcp_dav="always"` launches it on every backward of the
replay layout; the outputs and gradients are bitwise those of `"auto"`.

Backward tape order (default `"under_b1"`; private test knob
`bwd_tape_order`): the saved-mode backward issues its K1 tape replay after
the b1 exchange and before its wait, instead of before the `dav` stage.  It
applies only where the backward skips `kcp_dav` (the K1 tape replay's one
earlier reader of `v_new`), at `P = 1` with the a2a exchange, the
`wait_first` schedule, no hand-off tape set and the replay arena (a lean
arena only with a kept `u`, `saved_rows="wy"`); every other backward runs
`"first"` (the replay before the `dav` stage), which the knob value
`"first"` forces for A/B records.
`LAST_CALL["bwd"]["bwd_tape_order"]` and `["bwd_tape_order_refusal"]` record
the order run and why.  The launches and the collectives are the same, so the
gradients are bitwise those of `"first"`.

Backward K3 stream (default `"side"`; private test knob `bwd_k3_stream`): the
saved-mode backward issues its dm-form K3 (the maps re-sent in b1 when the
saved set keeps no c1 receive tensor, e.g. under a recompute handle's
`"states"` tier) on the summary pair's side stream, from right after the last
writer of its operands, beside the K1 tape replay, the `dav` stage and K2
`_offset`, and joins it back before the b1 issue.  It applies at `P = 1` with
the a2a exchange on the trimmed host path where the `walk_streams` rule gives
two streams for the K3 list.  Every other backward, and the first K3 of a table
set over its list on a device, runs `"main"` (K3 on the call's stream after K2
`_offset`); the knob value `"main"` forces that stream for A/B records.
`LAST_CALL["bwd"]["bwd_k3_stream"]` and `["bwd_k3_stream_refusal"]` record the
stream run and why.  The gradients are bitwise those of `"main"`.

Host path: the eager KCP calls reuse the call-independent host work -- the
validated options, the workspace's last tables and pool views, the local
stages' launch templates and the table set's own device tables -- keyed on
object identity plus host integers and bounded (`ID_CACHE_MAX = 256` entries
per table-set cache; the conv halo module keeps at most `HOST_CACHE_MAX = 64`
contexts). The launch templates live on the workspace's local arena and
reference the table set only weakly, so a training loop that builds a new
context and workspace per step releases everything with them. The kernels,
their arguments and their order are unchanged, so the results are bitwise
those of the previous host path, which stays reachable for A/B records
through the private test knob `host_trim` (after
`cake_kcp_plan.enable_test_knobs()`; the conv module's
`_set_test_knobs({"host_trim": 0})`).

Default workspace: a call with `workspace=None` runs on a workspace the
package keeps per key (device, current stream, backend, modes, direction,
inference mode, the rank's token / fragment / head counts and the size of
every pool). Its host objects stay between calls -- the pool tensors, the
local arena object with its layout views and launch templates, the pool
views and the `ensure` records -- so a repeated call skips the workspace build,
the arena build and the views, as a caller-held workspace does. Its device
memory lives for the call alone: the call allocates the same bytes as a new
workspace through the caching allocator on its stream
(`untyped_storage().resize_(n)` of the kept storages: the local arena at the
call's start, each pool at its first reader, the rest of its pools right
before its first collective, the lazily allocated pools only where the call
reads them, so no kept pool is first allocated once the call's c1 / b1
exchange or first ring hop is issued) and frees them at its end, so the
package keeps host objects only. The exceptions are the read-only identity
tile per device and, on the rows whose intra record writes the chunk tables
(the segmented route, `cake_kcp_plan.local_scratch_table_bytes`), a
KiB-scale scratch table buffer per kept workspace. The bytes a call requests
(`torch.cuda.memory_stats()["requested_bytes.all.peak"]`) peak no higher than
on a new workspace, each form measured after a call of its own form. The
caching allocator's block-size high-water (`torch.cuda.max_memory_allocated`)
can differ either way: the pools are allocated in another order, and the
block each request gets depends on what the allocator has cached (on GB300,
from 3.4 MiB lower to 1.4 MiB higher per rank in the tests). At most
`DEFAULT_WS_KEYS = 16` keys with `DEFAULT_WS_PER_KEY = 2` workspaces each
are kept (least recently used key dropped); `cake_kcp.default_workspace_cache_clear()`
and `cake_kcp.set_backend` drop them (and the kept-set pools, "Kept-set pool"
below) and `cake_kcp.DEFAULT_WS_STATS` /
`LAST_CALL[kind]["default_ws"]` record hits, misses, bypasses and evictions.
Calls that cycle through more keys, such as a new stream per step (the
stream is part of the key), miss on every call (`DEFAULT_WS_STATS["evict"]`
grows); run them on a fixed stream or pass a workspace. A CUDA-graph capture, a non-CUDA device
and a context without the context-held table cache take a new workspace per
call, as before. The kernels, their arguments other than data pointers and
their order are unchanged, so the results are bitwise those of a new
workspace per call, which stays reachable for A/B records through the
private test knob `default_ws` (`0`). Under
`torch.use_deterministic_algorithms(True)` with
`torch.utils.deterministic.fill_uninitialized_memory` on (its default), the
storage a kept workspace allocates again is not filled, as a caller-held
workspace's storage is not filled again per call; a new workspace per call
(`default_ws=0`) is filled when it is built, so compare the two forms'
host time with deterministic algorithms off.

Host synchronization: no KCP call and no pack preparation copies a table from
pageable memory or waits for queued device work, with or without a
caller-owned workspace. `cake_kda_nondet_train_kcp_build_context` copies the context's tables in one
pinned non-blocking copy on `stream=` (default: the current stream), so pack
N+1 can be prepared while pack N runs; `heads=H` derives the head plans there
as well (host work; their device tables are copied only when read, and the
byte queries are host-only). The launch tables of a `(ctx, H)` live on the
context, shared by every workspace and by `workspace=None` calls: each is
copied once, by the first call that reads it, in one asynchronous pinned copy
on the call's stream. A context prepared on another stream is ordered on the
stream of each call by `ctx.wait_ready()` (event wait, no host wait; the KDA
and conv halo entry points call it on every call). A thread other than the
compute thread must pass `stream=` (the current stream is per thread). Copies
are refused under CUDA-graph capture (`graph="replay"` runs the first call of
a key eagerly). Not covered: `workspace=None` calls allocate the same
device bytes per call through the caching allocator (the package keeps host
objects only, see "Default workspace" above), so a caching-allocator miss
can call `cudaMalloc` and an out-of-memory retry synchronizes (reuse a
workspace under memory pressure), the debug paths
(`spec3_probe`, `debug_sequence`) and `TORCH_NCCL_BLOCKING_WAIT=1`. The
previous copies stay reachable for A/B records through the context test knobs
`upload="blocking"` and `table_cache="workspace"` (the latter only with the
former). Eager KCP calls are host-bound at high CP: bind each rank's threads
to the cores of its GPU's socket (for example `srun --cpu-bind=cores` with as
many threads as bound cores), since unbound torch threads of several ranks
oversubscribe the CPUs.

Options of the KDA entry points: `save_transfer` keeps the owner's c1 receive
tensor (and `M_seg`) for the backward instead of re-sending it in the backward
exchange -- the default with the all-to-all exchange (an earlier revision).
`exchange="ring"` (available when every document spans at most three ranks)
receives no such tensor, so `save_transfer` resolves to `False` there
automatically; an explicit `save_transfer=True` together with the ring is a
`ValueError`. `CakeKDANondetKcpSaved.save_transfer` always reports the value that was used.
`spec3_probe` sets the debug probe of the backend's first saved-mode backward,
which compares the backward's prep and WY rows with the forward's: `"off"` (the
default) skips it, `"record"` stores the comparison in
`LAST_CALL["bwd"]["spec3"]`, `"warn"` warns on a difference, `"assert"` raises
`KcpPreconditionError`.  A probing backward keeps five token-row planes (about
`ctx.R * H * 1536` bytes: 1.5 KiB per head for every row of the rank's
chunk-padded token layout) allocated through its prep and WY programs, so its
peak memory is higher by up to that amount.
The backward reads the value of its own call (pass it to
`cake_kda_nondet_train_kcp_backward`; the forward's is not kept in `CakeKDANondetKcpSaved`).  Pass the
same value on every rank: the probe is a rank-local check, so `"assert"`
serves single-process debugging.  Under `graph="replay"` the probing
backward runs eagerly; a rank without tokens does not hold back its capture.
No environment variable selects it.

Checkpoint recompute without the state exchange (issue #7815): under
`torch.utils.checkpoint` the backward re-runs the forward of the region, and
with it the forward's c1 / c2 exchanges (and the convolution's halo exchange).
A recompute handle keeps what the original forward received instead:

```python
from flashinfer.cake_kda_nondet_train_kcp import CakeKDANondetKcpRecomputeHandle, cake_kda_nondet_kcp
from flashinfer.cake_causal_conv1d_det_train_kcp import cake_causal_conv1d_det_kcp

def layer(xq, xk, xv, g, beta, hq, hk, hv):
    q = cake_causal_conv1d_det_kcp(xq, wq, ctx, recompute_handle=hq)   # "conv" entry of hq
    k = cake_causal_conv1d_det_kcp(xk, wk, ctx, recompute_handle=hk)
    v = cake_causal_conv1d_det_kcp(xv, wv, ctx, recompute_handle=hv)
    ...                                                          # split heads, normalize q / k
    return cake_kda_nondet_kcp(q, k, v, g, beta, ctx, recompute_handle=hq)   # "kda" entry of hq

handles = [CakeKDANondetKcpRecomputeHandle() for _ in range(3)]               # per layer and micro-step
out = torch.utils.checkpoint.checkpoint(layer, xq, xk, xv, g, beta, *handles, use_reentrant=False)
```

The original forward exchanges as always and stores its incoming states
`S_in` (the saved set's own tensor, key `"kda"`; one entry per key, so one
handle serves the KDA call and one convolution call); the recompute inside the
backward binds them and launches only the rank-local programs -- no
collective, no wait -- so the outputs and gradients are bitwise those of the
checkpoint without a handle (`use_reentrant=True` and `False`, `retain_graph`,
nested regions).  `checkpoint_keep="states"` (the default) keeps `S_in` only
and runs the call with `save_transfer=False` (the backward re-sends the maps
in b1; bitwise the default form); `checkpoint_keep="transfer"` also keeps the
owner's c1 receive tensor (`save_transfer=True`; not with the ring exchange).
An explicit `save_transfer` that contradicts the tier raises `ValueError`, in
the entry point and in both byte queries.  The handle retains
`cake_kda_nondet_train_kcp_saved_bytes(ctx, H=..., checkpoint_keep=..., with_conv=False)` bytes for
the KDA entry until it is dropped (`with_conv` adds one convolution's history
rows); `cake_kda_nondet_train_kcp_comm_bytes(ctx, ..., checkpoint_keep=...)` reports the step's
traffic (`B1 + B2`; the recompute moves nothing when the layer's convolutions
carry handles too -- a convolution without one still sends its `halo_fwd`
rows in the recompute).  Rules: create the handles outside the outermost checkpoint
region (a handle created inside a region is empty in that region's recompute,
which then exchanges -- correct, not communication-free); a forward outside
the backward that finds an entry raises (a handle reused across micro-steps);
a recompute whose entry was stored by another context or call configuration
raises; `graph="replay"`, CUDA-graph capture, `debug_sequence`, `spec3_probe`
and `boundary_mode="replay"` are refused with a handle.  The exchange is a
collective: every rank passes a handle for the same call or none, at the same
nesting level.  The handle's presence is part of the rank-agreement digest
("Rank agreement" above): a rank that passes one while another does not trips
the device-side assert at the first original forward of that key.  A mismatch
of the nesting level is not checked: it hangs or corrupts data.  The phase
comes from `torch._C._current_graph_task_id()`; a torch without it refuses the
handle.

Checkpoint hand-off (`cake_kda_nondet_kcp(..., checkpoint_handoff="auto" | "on" |
"off")`; under the default `saved_rows="auto"` an engaged call keeps the rows
form below, with `saved_rows="tape"` the tape set described here): under
`torch.utils.checkpoint` the forward keeps its tape set --
the `h` tape, `v_new`, `w` and the internal `Aqk` / `Akk` rows its own
launches write -- in the saved set, and the backward reads it instead of
re-running the intra program, WY, the K1 tape walk and, at `P > 1`, the
offset re-walk and the carry merges.  Non-reentrant checkpoint keeps the set
in both passes (the original forward's goes at its return); reentrant
checkpoint runs its first pass under no_grad, so only its recompute keeps
one.  An engaged forward runs the tape walk and `kcp_fwd_o` where the light
forward runs the fused walk `kcp_fwdh_o` (the backward reads the `h` /
`v_new` tapes the tape walk writes); every other launch and every collective
are unchanged.  On the
devices of `KCP_HANDOFF_VERIFIED` the outputs and gradients are bitwise those
of `"off"` (the hand-off rows of
`tests/experimental/test_cake_kda_nondet_train_kcp_handoff.py`); `"on"` on another
device has no such evidence.

`"auto"` (the default) engages where `cake_kcp_plan.kcp_handoff_auto_refusal`
admits the call, `"on"` wherever the call supports it, `"off"` never:

```text
(architecture, SM count, toolkit) in KCP_HANDOFF_VERIFIED
    = (sm_100a, 148, 13.4) B200, (sm_103a, 148, 13.4) B300,
      (sm_103a, 152, 13.4) GB300, (sm_107a, 212, 13.4) R200,
      (sm_100a, 152, 13.4) GB200
    (raw gate mode / in-kernel Q / K L2 norm: in KCP_HANDOFF_VERIFIED_MODES[(gate, l2n)])
and H in KCP_HANDOFF_HEADS = {4, 8, 16, 32, 64}
and H * (total packed tokens // world size) >= KCP_HANDOFF_MIN_HEAD_TOKENS = 393216
    (and >= KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY[key] for a key with its own bound:
     786432 on GB200)
```

The rule comes from the `ckpt` step (time with the hand-off off / on) at
261392 packed tokens in 4 documents, BF16, head dim 128.  On the rows the rule
admits, measured with `"auto"` on this version on all four keys with the
outputs and gradients bitwise those of `"off"`: H4 at CP2 (H * T / W = 522784)
1.058-1.106; H16 and H64 1.073-1.147 (and 1.034 on one B200 CP8 H16 row whose
timing drifted 5.20 %, over the benchmark's 2 % limit); H8 and H32, which stay
in `KCP_HANDOFF_HEADS` on this evidence, 1.041-1.146.  Per key, H8: B200
1.093-1.128, B300 1.082-1.122, GB300 1.041-1.124, R200 1.078-1.124; H32: B200
1.117-1.146, B300 1.118-1.139, GB300 1.129-1.137, R200 1.127-1.134.  The H4
rows the rule refuses were measured with `"on"` forced on the hand-off's first
version: H4 at CP4 (261392) 1.033-1.043 on B200, 1.011-1.030 on B300 and
1.025-1.034 on R200 but 0.978-0.989 on GB300; H4 at CP8 (130696) 0.985-1.002
on B200 and 0.979-1.010 on B300.  That run measured every H4 row at CP2 and
every H16 and H64 row 5 % or more faster.  H8 at CP8 (261392) is refused as
`"work"` on the 8-GPU parts.  The bound admits every measured row that gains
5 % or more and refuses every measured loss.  A new device, toolkit or head
count needs these rows and the hand-off rows before it enters
`KCP_HANDOFF_VERIFIED` or `KCP_HANDOFF_HEADS`.  These rows predate the fused
forward walk, which the light path's forwards run and an engaged forward does
not.  H, the batch's total and the
world size are the same on every rank, so the ranks of a group of one device
kind decide alike; a rank that decides otherwise (a rank without tokens never
keeps a set) still runs the same collectives, because the launches the
hand-off skips are rank-local.

In raw gate mode, with the in-kernel Q / K L2 norm and with both, `"auto"`
reads the mode's entry of `KCP_HANDOFF_VERIFIED_MODES` instead of
`KCP_HANDOFF_VERIFIED` (the head and work bounds are the same): GB300
(sm_103a, 152, 13.4), R200 (sm_107a, 212, 13.4) and GB200 (sm_100a, 152,
13.4) in all three modes.  These modes run other local records (the raw and
`_l2n` intra / prep twins), so a key of `KCP_HANDOFF_VERIFIED` is no
evidence for them.  On these keys the mode rows of
`tests/experimental/test_cake_kda_nondet_train_kcp_handoff.py` (each mode at
2 and 4 ranks, H 4 / 16 / 64, a rank without tokens) measured every output
and gradient bitwise those of `"off"`.  The GB300 evidence also has the
`ckpt` step at 4 ranks of 16384 tokens, H = 64, BF16 (time with `"off"` /
with `"auto"`, slowest rank, median of 40 interleaved samples; every output
and gradient bitwise that of `"off"` on every rank, the hand-off engaged on
every rank):

| mode, layout | no handle | handle, `"states"` | handle, `"transfer"` |
| --- | --- | --- | --- |
| raw gate + L2 norm, 4 documents (24576, 16384, 14336, 10240), zigzag per document | 9.803 / 9.224 ms = 1.063 | 9.484 / 8.912 ms = 1.064 | 9.167 / 8.597 ms = 1.066 |
| raw gate + L2 norm, 1 document of 65536, contiguous | 9.505 / 8.958 ms = 1.061 | 9.725 / 9.191 ms = 1.058 | 9.431 / 8.896 ms = 1.060 |
| L2 norm, the 4 documents | 8.778 / 8.309 ms = 1.056 | 8.459 / 7.997 ms = 1.058 | 8.144 / 7.682 ms = 1.060 |
| raw gate, the 4 documents | 8.884 / 8.320 ms = 1.068 | 8.574 / 8.009 ms = 1.071 | 8.246 / 7.685 ms = 1.073 |

On R200 the same four rows, each with no handle and with a handle of either
tier, gain 1.054-1.074 with the hand-off forced on (every output and
gradient bitwise that of `"off"`, the hand-off engaged on every rank).

The set takes `cake_kda_nondet_train_kcp_handoff_bytes` = 1320 MiB per rank on
the 4-document rows and 1290 MiB on the one-document row (`NC_pad` and `R`
are padded beyond the 256 chunks of 16384 tokens); the measured peak rises
by +1320 and +1282 MiB.

The rule's own rows in these modes on GB300 (the `ckpt` step at 261392 packed
tokens in 4 documents, zigzag per document, BF16, through
`benchmarks/bench_cake_kda_nondet_train_kcp.py` with the hand-off forced on:
arms `det`, `nd`, `control` (an A/A copy of `nd`) and `nd_off`, median of 3
groups of 8 samples; time with `"off"` / with the hand-off in ms; every output
and gradient bitwise that of `"off"`, the hand-off engaged on every rank,
every A/A ratio within 0.987-1.002 and every drift below 2 %):

| mode | CP | H4 | H8 | H16 | H32 | H64 |
| --- | --- | --- | --- | --- | --- | --- |
| raw gate + L2 norm | 2 | 6.496 / 6.094 = 1.066 | 10.115 / 9.441 = 1.071 | 18.787 / 17.428 = 1.078 | 34.324 / 32.104 = 1.069 | 66.958 / 62.546 = 1.071 |
| raw gate + L2 norm | 4 | 5.321 / 5.537 = 0.961 | 6.504 / 6.139 = 1.059 | 10.215 / 9.509 = 1.074 | 18.054 / 16.888 = 1.069 | 34.528 / 32.263 = 1.070 |
| L2 norm | 2 | 6.059 / 5.780 = 1.048 | 9.136 / 8.605 = 1.062 | 17.047 / 15.947 = 1.069 | 30.520 / 28.739 = 1.062 | 59.552 / 56.001 = 1.063 |
| L2 norm | 4 | 4.902 / 5.122 = 0.957 | 6.040 / 5.789 = 1.043 | 9.285 / 8.730 = 1.064 | 16.152 / 15.207 = 1.062 | 30.801 / 28.962 = 1.063 |
| raw gate | 2 | 5.980 / 5.767 = 1.037 | 9.005 / 8.370 = 1.076 | 16.774 / 15.443 = 1.086 | 31.156 / 28.928 = 1.077 | 60.616 / 56.191 = 1.079 |
| raw gate | 4 | 4.935 / 5.132 = 0.961 | 5.968 / 5.774 = 1.034 | 9.151 / 8.498 = 1.077 | 16.505 / 15.355 = 1.075 | 31.624 / 29.378 = 1.076 (a) |

(a) measured on another GB300 node, at the next revision, whose changes do not
reach this call (the in-kernel Q / K L2 norm's refusal, documentation, tests).

Every row the shared bounds admit gains 3.4-8.6 %, on the segmented route
(H4, H8 and, on sm_103a, H16; with the L2 norm the hand-off also launches the
separate normalization there where the intra program normalizes first) as on
the serial route.  The only losses are H4 at CP4 (0.957-0.961, H * T / W =
261392), which the work bound refuses ("work"), as on the activated gate.

The same rows on R200 (H 4 / 8 / 16 / 64; time with `"off"` / with the
hand-off in ms; every output and gradient bitwise that of `"off"` and the
hand-off engaged on every rank in every run; each admitted cell from a run
with its A/A ratio within 0.985-1.011 and drift below 2 %, two such runs
where a cell lists two):

| mode | CP | H4 | H8 | H16 | H64 |
| --- | --- | --- | --- | --- | --- |
| raw gate + L2 norm | 2 | 3.757 / 3.561 = 1.055 | 6.615 / 6.234 = 1.061 | 12.414 / 11.640 = 1.066 | 43.584 / 40.767 = 1.069 |
| raw gate + L2 norm | 4 | 1.021, 1.029, 0.984 (b) | 3.869 / 3.631 = 1.066, 3.828 / 3.675 = 1.042 | 6.711 / 6.303 = 1.065 | 22.497 / 21.040 = 1.069 |
| L2 norm | 2 | 3.486 / 3.330 = 1.047 | 6.163 / 5.820 = 1.059 | 11.509 / 10.789 = 1.067 | 41.033 / 38.742 = 1.059 |
| L2 norm | 4 | 2.846 / 2.798 = 1.017 | 3.630 / 3.417 = 1.062 | 6.210 / 5.847 = 1.062 | 21.072 / 19.908 = 1.058 |
| raw gate | 2 | 3.450 / 3.236 = 1.066 | 6.088 / 5.662 = 1.075 | 11.374 / 10.489 = 1.084 | 38.844 / 36.026 = 1.078 |
| raw gate | 4 | 2.888 / 2.766 = 1.044 | 3.635 / 3.337 = 1.089 | 6.134 / 5.695 = 1.077 | 20.056 / 18.583 = 1.079 |

(b) three runs, none within both limits (drift 2.56 % and 2.53 %, A/A 0.969):
the work bound refuses this row.

Every row the shared bounds admit gains 4.2-8.9 % on R200; the H4 rows at CP4,
which the work bound refuses, measure 0.984-1.044.  B200 and B300 have no
rows in these modes yet, so `"auto"` keeps the light set on them (`"on"`
engages, without that evidence).

The same rows on GB200, in the activated gate too (every output and gradient
bitwise that of `"off"` and the hand-off engaged on every rank in every run;
each cell from a run with its A/A ratio within 0.987-1.001 and drift below
2 %, two such runs where a cell lists two):

| mode | CP | H4 | H8 | H16 | H64 |
| --- | --- | --- | --- | --- | --- |
| raw gate + L2 norm | 2 | 6.599 / 6.199 = 1.065 | 11.034 / 10.333 = 1.068, 10.341 / 9.632 = 1.074 | 18.445 / 17.198 = 1.073 | 68.707 / 64.236 = 1.070 |
| raw gate + L2 norm | 4 | 5.168 / 5.359 = 0.964 | 6.549 / 6.157 = 1.064 | 9.972 / 9.357 = 1.066, 9.967 / 9.351 = 1.066 | 35.393 / 33.059 = 1.071 |
| L2 norm | 2 | 6.186 / 5.946 = 1.040 | 9.368 / 8.810 = 1.063 | 16.690 / 15.657 = 1.066 | 62.230 / 58.565 = 1.063 |
| L2 norm | 4 | 4.948 / 5.167 = 0.958 | 0.988, 1.060, 1.023 (b) | 9.044 / 8.535 = 1.060 | 32.397 / 30.493 = 1.062 |
| raw gate | 2 | 6.050 / 5.786 = 1.046 | 9.204 / 8.514 = 1.081 | 16.746 / 15.524 = 1.079 | 61.909 / 57.423 = 1.078 |
| raw gate | 4 | 5.009 / 5.165 = 0.970 | 6.907 / 6.878 = 1.004, 7.460 / 7.360 = 1.014 | 10.514 / 9.914 = 1.060 | 32.170 / 29.857 = 1.077 |
| activated gate | 2 | 5.745 / 5.631 = 1.020 | 8.628 / 8.113 = 1.064 | 15.865 / 14.861 = 1.068 | 58.812 / 55.159 = 1.066 |
| activated gate | 4 | 4.841 / 5.002 = 0.968 | 5.634 / 5.506 = 1.023, 5.647 / 5.511 = 1.025 | 1.059 (b) | 30.268 / 28.338 = 1.068 |

(b) no run within both limits.  H8 at CP4 with the L2 norm: three runs,
drift 3.12 %, 2.37 % and 2.24 %, A/A 0.993-0.995; the bound refuses this
row.  H16 at CP4 on the activated gate: one run, drift 2.51 %, A/A 0.997;
the bound admits this row.

GB200 keeps the work bound from these rows: `KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY`
gives it 786432 in place of 393216, in every mode.  At H * T / W = 522784 the
hand-off does not gain reliably there: H8 at CP4 measures 1.004-1.064 in runs
within both limits (1.004 on the raw gate, no such run with the L2 norm
alone) and down to 0.988 in runs over the drift limit, while H4 at CP2 (the
same work) gains 1.020-1.065, so no bound on the work separates the two.
786432 lies between 522784 and 1045568: every row it admits (H8 at CP2, H16
and H64 at CP2 and CP4) gains 1.060-1.081 in runs within both limits (and
1.059 in the activated gate's one H16 CP4 run), and it refuses H4 and H8 at
CP4 and H4 at CP2.  The H64 step at 16384 tokens per rank (1048576) stays
admitted.  The key's bound is read with the key, after the shared bounds, and
the original forward records the decision the recompute replays.

Memory: the set costs `cake_kda_nondet_train_kcp_handoff_bytes(ctx, H=...)` =
`NC_pad * H * 32768 + R * H * 768` bytes (1280 B per internal token-head) on
each rank and lives from the recompute to the end of the node's backward
(non-reentrant: also inside the original forward call).  An engaged call
raises the checkpoint peak by at most that much: the saved set grows by
`cake_kda_nondet_train_kcp_saved_bytes(..., checkpoint_handoff=True) - cake_kda_nondet_train_kcp_saved_bytes(...)`,
the tape set less the carry plane at `P > 1`.  Measured with `"on"` forced on
the hand-off's first version: +152 MiB per rank at CP8 H4 to +10196 MiB at CP2
H64 (T = 261392); with `"auto"` on this version, +625.5 MiB at CP8 H16
to +10196 MiB at CP2 H64.
`cake_kda_nondet_train_kcp_saved_bytes(..., checkpoint_handoff=True)` is the saved set of an
engaged call (no carry plane; the flag is the decision, a `bool`, not the
mode); the facade `flashinfer.cake_kda_nondet_train_kcp`
exports `cake_kda_nondet_train_kcp_saved_bytes(..., checkpoint_handoff=...)`,
`cake_kda_nondet_train_kcp_handoff_bytes` and
`cake_kda_nondet_train_kcp_handoff_stats()` (the forwards that kept a set, the backwards that read
one and the refused requests by reason).  To opt out, pass
`checkpoint_handoff="off"`; the deterministic package has no hand-off.

Rows form: with `saved_rows="intra"` / `"wy"` an engaged call keeps that rows
set ("Saved rows" above) instead of the tape set.  The forward keeps its fused
walk and the carry plane; the backward skips the intra program (and WY) and
still replays the K1 tape.  It costs
`cake_kda_nondet_train_kcp_handoff_bytes(ctx, H=..., saved_rows=...)` (256 /
768 B per internal token-head), counts in `SAVED_ROWS_STATS` (not in the
hand-off's `"save"` / `"bound"`), combines with a recompute handle (whose
recompute issues no collective), and its gradients are held to the light
path's FP64 error budget, not bitwise those of `"off"`.  `saved_rows="auto"`
(the default) keeps the `"wy"` rows form (`cake_kcp_plan.KCP_HANDOFF_AUTO_ROWS`)
where the hand-off engages: on GB300 at CP4, H 64, 16384 tokens per rank it
is 0.21-0.33 ms faster per checkpointed layer than the tape set and keeps
528 MiB less per layer and rank (768 instead of 1280 B per internal
token-head).  `saved_rows="tape"` keeps the tape set.

Kept-set pool: in a checkpoint pass of a `workspace=None` call the kept set
(the tape set or a rows set) comes from a package-private `torch.cuda.MemPool`
per device, each tensor's block cut for its leading extent rounded up to 16
steps per power of two (at most 1/16 more), so packs of nearby sizes reuse its
blocks; once a block of that size exists, a tensor takes its exact shape from
it.  Such a call allocates its arena at its start and frees it at its end; in
the caching allocator's default pool the set, which lives from the recompute
to the node's backward, would take the rest of the forward arena's freed
block, the backward arena a new segment, and the reserved memory would grow
over the steps.  A caller-held workspace's arena stays allocated, so its calls
keep the default pool.  Reserved memory per rank on GB300 (CP4, H 64, packs of
65536 tokens, max over ranks, MiB; first step after `torch.cuda.empty_cache()`
and the steady state of the next 3 steps; outputs and gradients bitwise
unchanged):

| gate | recompute | first step: before / pool | steady: before / pool | steady, `checkpoint_handoff="off"` |
| --- | --- | --- | --- | --- |
| `g` | plain | 35370 / 32340 | 36570 / 32340 | 31524 |
| `g` | handle | 31986 / 32340 | 36570 / 32340 | 31524 |
| `g`, L2 norm | plain | 35370 / 32188 | 36568 / 32342 | 31526 |
| `g`, L2 norm | handle | 31986 / 32188 | 36568 / 32188 | 31372 |
| raw, L2 norm | plain | 30540 / 30556 | 39718 / 30556 | 29740 |
| raw, L2 norm | handle | 30540 / 30556 | 39558 / 30556 | 29740 |

The pool costs one kept set over `"off"` (816 MiB here); the allocated peak
stays within 3 MiB of the default pool's, and the host time of a kept set's
allocation within 2 us of it (measured on GB300).  It keeps its blocks reserved
between calls, and `torch.cuda.empty_cache()` does not release the blocks of a
live pool: measured after the `workspace=None` steps, caller-held calls in the
same process reserve those 816 MiB more.  A process that stops making
`workspace=None` checkpointed calls drops the pools with
`cake_kcp.default_workspace_cache_clear()`, after which `empty_cache()`
releases them.

Under `saved_rows="auto"` the call keeps the light set outside a checkpoint
pass, with `graph="replay"`, `spec3_probe`, `boundary_mode="replay"`, inside a
CUDA-graph capture, under `torch.compile` tracing and under a Python
dispatch mode (a selective-checkpoint policy).  A non-reentrant checkpoint
decides once per call.  A request the call's own arguments refuse
(`boundary_mode`, `graph`, `spec3_probe`, a tensor off CUDA and, under
`"auto"`, the head count, the batch's total and the world size) is refused
alike in both passes, which pass the same arguments, and records nothing.
Under `"auto"` that refusal also reads `KCP_HANDOFF_HEADS` and
`KCP_HANDOFF_MIN_HEAD_TOKENS` at call time: neither may change between a
checkpointed forward and its backward (change them only between steps), or
one pass can refuse a call the other admits and the recorded decisions shift.
For every other call the original forward reads these states, the backend
and the device key and records its decision on the checkpoint frame, and
the recompute replays it in call order without reading its own state, so
both passes save the same tensors (torch raises `CheckpointError` otherwise)
even when a capture, `torch.compile` or a dispatch mode is active in one
pass only.  It combines with a recompute handle.

One halo exchange for a layer's q / k / v convolutions:
`cake_causal_conv1d_det_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx)` returns the
three outputs bitwise equal to three `cake_causal_conv1d_det_kcp` calls (outputs,
`dx` and `dweight`), but its forward and its backward each issue one halo
collective instead of three: the history rows of the three tensors travel side
by side in one `all_to_all_single` and are copied out after the wait, and each
tensor then runs its own kernels. With `recompute_handle=h` one handle serves
the call (its `"conv"` entry holds the three history pools) and may also carry
the layer's `"kda"` entry, so the checkpointed layer above needs one handle:

```python
def layer(xq, xk, xv, g, beta, h):
    q, k, v = cake_causal_conv1d_det_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx, recompute_handle=h)
    ...
    return cake_kda_nondet_kcp(q, k, v, g, beta, ctx, recompute_handle=h)

out = torch.utils.checkpoint.checkpoint(layer, xq, xk, xv, g, beta, CakeKDANondetKcpRecomputeHandle(),
                                        use_reentrant=False)
```

Per layer the halo collectives drop from 3 / 3 / 3 (original forward /
recompute / backward, without handles) or 3 / 0 / 3 (with handles) to 1 / 1 / 1
or 1 / 0 / 1.

Raw gate mode on the KCP route (issue #7816): the three KDA entry points take
the single-rank keywords `use_gate_in_kernel=True, A_log=, dt_bias=,
lower_bound=` with the same operand rules (`g` is the rank's raw BF16 gate,
any view under the stride rule of the single-rank package's raw gate mode,
never copied; `A_log` / `dt_bias` are the full parameters, the same tensors
on every rank).

```python
from flashinfer.cake_kda_nondet_train_kcp import CakeKDANondetKcpWorkspace, cake_kda_nondet_train_kcp_reduce_gate_param_grads

ws = CakeKDANondetKcpWorkspace(use_gate_in_kernel=True)                        # the workspace's mode, fixed
gate = dict(use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0)
out_local, saved = cake_kda_nondet_train_kcp_forward(q_local, k_local, v_local, raw_local, beta_local, ctx,
                                     workspace=ws, **gate)
dq, dk, dv, dbeta, draw, dA_log_r, ddt_bias_r = cake_kda_nondet_train_kcp_backward(
    q_local, k_local, v_local, raw_local, beta_local, ctx, dout_local, saved=saved, workspace=ws,
    **gate)
dA_log, ddt_bias = cake_kda_nondet_train_kcp_reduce_gate_param_grads(ctx, dA_log_r, ddt_bias_r)   # optional, see below
```

* The local stages run the raw records of the rank's plan
  (`cake_backend.plan(..., gate="raw")` of the single-rank package: copies of
  the deterministic raw programs; the intra and prep programs activate the
  gate, the raw GLM intra program writes `draw` and the per-chunk partials,
  one `param_finalize` launch reduces them; no gate epilogue, no `S1` pair).
  The `gk` rows the K1 / K2 / K3 walks read are the explicit path's bits for
  the same activated gate, so every KCP program, collective and exchanged
  byte is the explicit path's; no full-size FP32 gate or gate gradient exists
  on the route.  Every output, `draw` and the parameter partial sums are
  bitwise those of `cake_kda_det_train_kcp_backward` on the same rank (with
  the default saved set).
* `dA_log` / `ddt_bias` of a rank are its partial sums: the gradients through
  its own tokens, reduced on the rank in an order fixed by its local chunks
  (bitwise repeatable, the same under every exchange, schedule, graph replay
  and recompute).  The full gradient is their sum over `ctx.group`; a
  training framework sums them as it sums the gradients of any parameter
  replicated across the context-parallel ranks.  An NCCL all-reduce gives
  that sum to FP32 rounding but does not fix its summation order, so it is
  not bitwise reproducible; `cake_kda_nondet_train_kcp_reduce_gate_param_grads(ctx, dA_log,
  ddt_bias)` is: one `all_gather_into_tensor` of the `H + 128 H` values, then
  an FP32 sum in rank order on every rank (the same bits everywhere, and the
  same bits as the deterministic package's `cake_kda_det_train_kcp_reduce_gate_param_grads`).
* The mode is fixed per workspace (`CakeKDANondetKcpWorkspace(use_gate_in_kernel=True)`:
  its local arena is the replay arena planned in raw mode, whose backward
  layouts carry no gate epilogue buffers; the KCP pools do not change;
  `local_arena="lean"` is refused in raw mode), recorded in the saved set
  (`CakeKDANondetKcpSaved.gate`, `.lower_bound`) and in a recompute handle's fingerprint;
  a call in the other mode, a backward with another `lower_bound` and a
  recompute of another mode are refused with `ValueError`.
  `cake_kda_nondet_train_kcp_workspace_bytes(ctx, H=..., use_gate_in_kernel=True)` sizes it;
  `cake_kda_nondet_train_kcp_saved_bytes` and `cake_kda_nondet_train_kcp_comm_bytes` do not depend on the mode.
* The checkpoint hand-off engages in raw mode as on the activated gate (the
  same tape set, the same once-per-call decision; the forward's raw intra
  program writes the set, the backward's raw prep, gradient programs and
  `param_finalize` read it).  On the devices of
  `KCP_HANDOFF_VERIFIED_MODES[("raw", False)]` (with the flag below,
  `[("raw", True)]`) the outputs, the raw gate's gradient and the per-rank
  `dA_log` / `ddt_bias` partial sums (so their cross-rank sums) are bitwise
  those of `"off"`, and `"auto"` admits a raw call there only; `"on"` on
  another device has no such evidence.
* The backward allocates the raw gate's BF16 gradient, the partials (two FP32
  `[NC_pad, H, 128, 2]` buffers of the rank's chunks, for the duration of the
  call) and the two parameter gradients; a rank without tokens returns zero
  parameter gradients.

In-kernel Q / K L2 norm on the KCP route (issue #7817): the three KDA entry
points, `CakeKDANondetKcpWorkspace` and the workspace and saved byte queries take the
keyword-only `use_qk_l2norm_in_kernel=False`.  With `True` the local stages
normalize `q` and `k` per token and per 128-channel head (FP32 sum of squares,
`r = 1 / sqrt(ss + 1e-6)`, `x_hat = bf16(x * r)`; `v` is not normalized) and
the GLM intra programs fuse the FP32 pullback into the BF16 `dq` / `dk`, so
`q_local` / `k_local` are the rank's raw rows and the gradients are theirs.

```python
ws = CakeKDANondetKcpWorkspace(use_qk_l2norm_in_kernel=True)                   # the workspace's mode, fixed
out_local, saved = cake_kda_nondet_train_kcp_forward(q_raw, k_raw, v_local, g_local, beta_local, ctx,
                                     workspace=ws, use_qk_l2norm_in_kernel=True)
dq_raw, dk_raw, dv, dbeta, dg = cake_kda_nondet_train_kcp_backward(
    q_raw, k_raw, v_local, g_local, beta_local, ctx, dout_local, saved=saved, workspace=ws,
    use_qk_l2norm_in_kernel=True)
```

* `q_raw` / `k_raw` may be strided views, such as columns of a packed
  projection or convolution output `[S_local, (2 H + HV) * 128]`: innermost
  stride 1, head stride 128 elements, token stride a multiple of 8 elements
  and at least `H * 128` (no overlapping rows), base address 16-byte aligned
  (the single-rank rule).  They are never copied; any other view raises
  `ValueError`.  Without the flag `q` / `k` must stay contiguous.  `v` keeps
  its rules.  Every rank checks the metadata (shape, dtype, device, the three
  strides), a rank with zero or one token too, so all ranks refuse a view
  alike.  The base-address alignment is the one per-rank check: it is a
  property of each rank's own storage, checked where the rank has tokens.
* The normalization is token-local: the halo, the context, the KCP programs,
  every collective and its byte count are those of the flag-off route.
* The local stages run the single-rank package's records of the mode, copies
  of the deterministic package's programs; its plan selects no `S1` pair in
  the mode.  Every output, the saved rstd and every gradient are bitwise
  those of `cake_kda_det_train_kcp_forward` / `cake_kda_det_train_kcp_backward` on the same rank,
  in both gate modes (`use_gate_in_kernel=True` combines with the flag), with
  the default saved set.
* The raw `q` / `k` are not members of the saved set: the autograd node
  keeps the caller's tensors, as without the flag.  The saved set adds the
  FP32 rstd `CakeKDANondetKcpSaved.qk_rstd` `[2, S_local, H]` (plane 0 q, plane 1 k;
  empty on a rank without tokens): `cake_kda_nondet_train_kcp_saved_bytes(...,
  use_qk_l2norm_in_kernel=True)` is `8 * S_local * H` bytes more.  The
  normalized q / k are never saved: the backward rebuilds them bitwise from
  the raw rows and the rstd, and a checkpoint recompute writes its rstd
  afresh, so a recompute handle's entry and bytes do not change.  The
  backward keeps the rebuilt rows in its local arena, in the bytes of two
  internal buffers that it moves into the bytes of its own outputs (dq / dk
  and the gate gradient) until the GLM intra program writes those, so it
  needs no memory beyond the flag-off backward's.  Where an output is too
  small for its buffer (more than `2 * S_local` internal rows; in raw gate
  mode more than `S_local` for the gate gradient), that row set takes a
  per-call buffer (`256 * S_local * H` bytes each) instead.
* The mode is fixed per workspace (the replay local arena of that mode;
  `local_arena="lean"` is refused with it), recorded in the saved set
  (`CakeKDANondetKcpSaved.qk_l2norm`), the graph validity key and a recompute handle's
  fingerprint; a call, a saved set or a recompute of the other mode is
  refused with `ValueError`.  `spec3_probe` is refused with the flag.
* The checkpoint hand-off engages with the flag as without it, in both gate
  modes: the same tape set (the saved set keeps its rstd), the same
  decision.  The backward skips the intra program; where that program is the
  replay's first q / k reader (the fused form on the segmented route at most
  256 documents: `cake_kcp_plan.qk_l2norm_intra_first`), the backward first
  runs the separate normalization record `qk_l2norm_apply` from the saved
  rstd, which stores the same `bf16(x * r)` rows; elsewhere the prep's
  `_l2n` twin normalizes from the saved rstd as in the light backward.  On
  the devices of `KCP_HANDOFF_VERIFIED_MODES[("g", True)]` (raw gate mode:
  `[("raw", True)]`) the outputs and the gradients of the raw `q` / `k` are
  bitwise those of `"off"`, and `"auto"` admits such a call there only;
  `"on"` on another device has no such evidence (the mode's forward and
  replay intra records differ per architecture).  Where the device's
  architecture has no exported `qk_l2norm_apply` library, both `"auto"` and
  `"on"` keep the light set (counted as `refused_backend`): the ranks whose
  plan needs that record would otherwise raise before `b1`.
* With `use_qk_l2norm_in_kernel=False` (the default) every launch, saved
  tensor, key, fingerprint and byte is unchanged.
* The local programs of the mode are the single-rank backend's (its plan mode
  `l2n="fused"`: the first q / k reader normalizes, the GLM intra program
  pulls back; a saved-mode backward reads the saved rstd).  Until the
  single-rank registry selects a record of the mode for every such stage of
  the forward and replay paths over a fixed grid of problem sizes (at most
  and above 256 documents, few and many heads, the architecture's SM
  counts), each delivered and exported for the device's architecture, with
  exactly one normalizing first reader on the replay, and its forward and
  replay layouts carry the normalized slot views `qn_int` / `kn_int`, a
  call, a `CakeKDANondetKcpWorkspace` and the byte queries with the flag on raise
  `NotImplementedError` before any workspace build, plan, launch or
  collective.  The check reads rank-independent inputs only (the
  architecture of the context's device, the registry's selection over that
  grid and its exports, the flags and gate mode, the static layout tables),
  so every rank, one without tokens or with a misaligned view too, raises
  the same error with the same message.  The saved byte query takes the
  call's `use_gate_in_kernel` for this check.  With the flag off the byte
  queries make no CUDA call for it.

The segment plan splits every fragment into `P` segments: above the 256-chunk
chain cap the equal-segment wave rule on ranks with at most three fragments,
below it on one-fragment ranks the
largest `P` that runs one CTA per SM (`floor(SM / H)`), on every other rank
one CTA-per-SM occupancy (`ceil(SM / (n_frag H))`) -- except that a rank
with two or three fragments below the cap whose `P = 1` walks run one wave
(`n_frag H <= SM`) takes `P = 1` when the longest-processing-time makespan of
its actual fragment chunk counts at `P = 1` is not above the makespan at the
occupancy `P` (such a rank runs no segment composition and no backward
re-walk; at H = 64 on 152 SMs equal two-fragment ranks tie, unequal ones
keep `P > 1`);
`ctx.describe(H)["p_rule"]` records the rule (`"wave_fill1_p1tie"`). The
previous rules stay selectable through the private test knob `p_rule`, as
does the fused form of the carry merges (`merge_fuse="carry"`: the two carry
families of a phase in one launch; off by default).

The files of this package:

* `cake_kcp_jit.py` -- the `MODULES` registry of the sixteen shipped programs (copies of the deterministic
  package's records under the `cake_kda_train_` library prefix, the same twenty hex digits; the
  fragment walks of the forward and backward state recurrences in their
  `offset` and `tape` forms, the BF16x3 transfer walk (two CTAs per segment
  and head, each over one 64-column half of the state), the affine
  merge in one record per register-blocked build, and the two chunk stages of the
  context-parallel route, `kcp_fwd_o` (the output from the K1 tapes) and
  `kcp_dav` (`dA_qk` where the single-rank plan dispatches its `dav`
  stage; the K2 adjoint walks form `dv1` from `aqk_int`), and the fused
  forward walk `kcp_fwdh_o` (the forward state walk with the chunk output inside it: the
  same bits as the tape walk followed by `kcp_fwd_o`, no tape written; in its
  full-width and two-CTA records), which the forward that keeps no tape
  launches in place of those two; selection by `always`, by
  `walk_form`, by `walk_heads` (the `WALK_HEADS_RULE` bands of the launch's
  head count: the plain or the overlap-schedule full-width adjoint offset
  walk), or by `merge_rule` (the `MERGE_RULE` bands of the merge
  launch's 128-column chain blocks per architecture), grids by `kcp_segments_heads`,
  `kcp_chains`, `kcp_tile_items` or `kcp_pairs`), the logical vocabulary of
  the KCP arguments, `run_program(name, args, device_index=)`, the lowering
  of the host backend's walk and merge requests to the registry's launches,
  and `run_module` for the two chunk stages.  Its `TEST_MODULES` are
  test-only programs of `build.test.json` (below).
* `cake_kcp_plan.py` -- the partition, the segment plan and the context's
  tables and collectives, with the byte queries.
* `cake_kcp.py` -- the per-rank launch sequences, the saved set and workspace
  of a rank, the autograd `Function` and the three entry points.
* `_single_rank.py` -- the name of the single-rank package this package
  requires (see "Vendoring") and the import of its modules.
* `build.test.json` -- a `cake-build/1` manifest of the same form as
  `build.json` holding the test-only programs of `cake_kcp_jit.TEST_MODULES`
  (the TF32 hi/lo transfer walk, its static `precision_mode` 1 and 2 forms,
  and the full-width BF16x3 walk).  `build.json` never lists them and the
  shipped route never loads them: `cake_kcp_jit.load_test` JIT-builds an
  entry only after `cake_kcp_plan.enable_test_knobs()`, which also admits
  the test knobs (`precision_mode`, `transfer_walk`) of the package's own
  tests; without the file those tests skip.  The package reads no
  environment variable.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and the manifest of the sixteen shipped
  programs (schema `cake-build/1`, one entry per `<library>_<arch>`);
  `aot/` is never tracked (`.gitignore`).
* `csrc/cake_ffi_checks.h` -- the TVM-FFI helper header the bindings include.
* `csrc/cake_kda_train_<digest>_kcp_{kernel,binding}.cu` of the twenty-one
  context-parallel programs (the chunk stage `kcp_fwd_o` among them) and
  `csrc/cake_kda_train_<digest>_{kernel,binding}.cu` of the chunk stage
  `kcp_dav`, rendered like the single-rank ones (digests in `build.json` and
  `build.test.json`); the units of the deterministic package's programs under
  this package's names.

## Vendoring

This package needs its single-rank sibling `cake_kda_nondet_train`: it runs
that package's chunk stages and reads its registry.  `build.json` and
`build.test.json` declare it as `common.requires.packages:
["cake_kda_nondet_train"]` (directory names).  `_single_rank.py` finds the
sibling by name from this package's parent, so vendor the two directories
together, either as siblings under one parent package or both as top-level
packages.  Without the sibling, importing `cake_kcp` or `cake_kcp_jit` raises
`ModuleNotFoundError` naming `cake_kda_nondet_train`.  Importing them needs
no `flashinfer`: `flashinfer` is imported only on the JIT path.

## Status

The registry holds sixteen shipped programs (`cake_kcp_jit.MODULES`, in
`build.json`) and six test-only programs (`cake_kcp_jit.TEST_MODULES`, in
`build.test.json`) for `sm_100a`, `sm_103a` and `sm_107a`.

The local `S1` rows of `sm_107a` are provisional (their records can return
to the deterministic pair without a change here).

Tests: `tests/experimental/test_cake_kda_nondet_train_kcp_build.py` (the KCP
registries and manifests, no device needed),
`tests/experimental/test_cake_kda_nondet_train_kcp.py` (the per-rank path),
`tests/experimental/test_cake_kda_nondet_train_kcp_dist.py` (multi-GPU rows, with
the comparison against the deterministic package),
`tests/experimental/test_cake_kda_nondet_train_kcp_standalone.py` (the two
vendored layouts without `flashinfer`),
`tests/experimental/test_cake_kda_nondet_train_kcp_api.py` (the public entry
points, the names and the import isolation),
`tests/experimental/test_cake_kda_nondet_train_kcp_handoff.py` (the checkpoint
hand-off: multi-GPU rows bitwise against `"off"`, the admission rule, the
once-per-call decision, the byte queries; `saved_rows`: the rows sets, the
decision of the autograd entry and multi-GPU rows against the FP64 reference
within the light path's error),
`tests/experimental/test_cake_kda_nondet_train_kcp_raw_gate.py` (the raw gate mode:
the keywords, the workspace and its byte query, the saved set, the reduction
helper, the local stages' records; multi-GPU rows against the single-rank raw
path, the FP64 reference and, bitwise, the deterministic package's raw KCP
route),
`tests/experimental/test_cake_kda_nondet_train_kcp_l2n.py` (the in-kernel Q / K L2
norm: the keywords, the strided views, the workspace, the byte queries and
the saved set; multi-GPU rows against the single-rank path, the FP64
reference and, bitwise, the deterministic package's KCP route in the mode),
`tests/experimental/test_cake_kda_nondet_train_kcp_bwd_order.py` (the test
knobs `bwd_tape_order` / `bwd_k3_stream`: their rules, the dAqk-coverage
allowlist against the plans' fused-gradient records, `kcp_dav_needed` against
every plan, the issue order; multi-GPU rows bitwise against the forced
`"first"` / `"main"` forms and with the dAqk rows NaN-filled after the `dav`
stage),
`tests/experimental/test_cake_kda_nondet_train_kcp_adjoint.py` (the K2
adjoint walks' fused `dv1`: `kcp_dav="always"` bitwise `"auto"` and the
NaN-filled dAqk dead where `kcp_dav` is skipped, per gate mode, norm, head
count and path; the walk forms bitwise across the 63 / 64-head rule),
`_fsdp_walk.py`, `_lean.py`, `_shared_ws.py` and `_upload.py`; benchmark
`benchmarks/bench_cake_kda_nondet_train_kcp.py` (`torchrun`, one process per
rank; the deterministic and the non-deterministic package interleaved).
