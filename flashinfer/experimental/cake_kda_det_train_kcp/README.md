# Cake deterministic context-parallel (KCP) packed KDA chunked training backend (SM100 / SM103 / SM107)

Experimental backend of the context-parallel route of the packed Kimi Delta
Attention (KDA) chunked training forward and backward: the packed batch is
split over the ranks of a `torch.distributed` process group and the
per-document state recurrences cross the rank boundaries exactly.  The public
entry points are in `flashinfer/cake_kda_det_train_kcp.py`; each rank runs the
chunk stages of `flashinfer.experimental.cake_kda_det_train` at its own shape.
Both the API and the backend are experimental: no compatibility guarantee.

The path is deterministic: the same inputs, hardware, plan and process group
give bitwise the same outputs and gradients.

### Names

The names before the det rename stay bound to the same objects:
`flashinfer.kda_train` (`chunk_kda_kcp`, `chunk_kda_kcp_fwd`,
`chunk_kda_kcp_bwd`, `kcp_partition`, `build_kcp_context`,
`kcp_workspace_bytes`, `kcp_saved_bytes`, `kcp_comm_bytes`,
`KcpRecomputeHandle`, `KcpContext`, `KcpPartition`, `KcpOptions`,
`KcpSaved`, `KcpWorkspace`, `kcp_reduce_gate_param_grads`), the modules
`flashinfer.experimental.cake_kda_train.cake_kcp` / `cake_kcp_plan` /
`cake_kcp_jit` (the same module objects) and the previous names inside
`cake_kcp` and `cake_kcp_plan`.

| previous name | det name |
|---|---|
| `chunk_kda_kcp_fwd` | `chunk_kda_det_fwd_kcp` |
| `chunk_kda_kcp_bwd` | `chunk_kda_det_bwd_kcp` |
| `chunk_kda_kcp` | `chunk_kda_det_kcp` |
| `kcp_partition` | `partition_det_kcp` |
| `build_kcp_context` | `build_context_det_kcp` |
| `kcp_workspace_bytes` | `kda_det_workspace_bytes_kcp` |
| `kcp_saved_bytes` | `kda_det_saved_bytes_kcp` |
| `kcp_comm_bytes` | `kda_det_comm_bytes_kcp` |
| `kcp_reduce_gate_param_grads` | `reduce_gate_param_grads_det_kcp` |
| `KcpRecomputeHandle` | `RecomputeHandleDetKcp` |
| `KcpWorkspace` | `WorkspaceDetKcp` |
| `KcpSaved` | `SavedDetKcp` |
| `KcpOptions` | `OptionsDetKcp` |
| `KcpCall` | `CallDetKcp` |
| `KcpContext` | `ContextDetKcp` |
| `KcpPartition` | `PartitionDetKcp` |
| `KcpFragment` | `FragmentDetKcp` |
| `ChunkKDAKcpFunction` | `ChunkKDADetFunctionKcp` |

## Usage

The packed batch of one training step may be split over the ranks of a
`torch.distributed` process group.  `partition_det_kcp` cuts the documents into
per-rank fragments (`"contiguous"`, `"zigzag"` or `"zigzag_per_document"`
layouts, an explicit `fragments` table, or `cut_points` with the
`"zigzag_per_document"` layout); `build_context_det_kcp` gives every rank the
device tables of its fragments, the collectives that carry the per-document
state recurrences and the convolution's history rows across the rank
boundaries, and the segment plan of its fragment walks.  The per-rank entry
points take the rank's shard of the operands (the tokens of its fragments, in
the partition's order) and return that shard's outputs and gradients:

```python
from flashinfer.cake_kda_det_train_kcp import (partition_det_kcp, build_context_det_kcp, chunk_kda_det_kcp,
                                  chunk_kda_det_fwd_kcp, chunk_kda_det_bwd_kcp,
                                  kda_det_workspace_bytes_kcp, kda_det_saved_bytes_kcp, kda_det_comm_bytes_kcp)
from flashinfer.cake_causal_conv1d_train import cake_causal_conv1d_kcp

part = partition_det_kcp(cu_seqlens_global_cpu, world_size)          # the same object on every rank
ctx = build_context_det_kcp(part, group, rank, device)               # this rank's tables and collectives
y_local = cake_causal_conv1d_kcp(x_local, conv_weight, ctx)      # the convolution before the KDA step
out_local = chunk_kda_det_kcp(q_local, k_local, v_local, g_local, beta_local, ctx)   # autograd
out_local, saved = chunk_kda_det_fwd_kcp(q_local, k_local, v_local, g_local, beta_local, ctx,
                                     save_mode="light", boundary_mode="saved")
dq, dk, dv, dbeta, dg = chunk_kda_det_bwd_kcp(q_local, k_local, v_local, g_local, beta_local, ctx,
                                          dout_local, saved=saved)
```

The context is a plain frozen object, not a dataclass: the pre-forward hooks
of FSDP2 (`fully_shard`) walk every dataclass, dict, list and tuple of a
module's arguments, and see the context as one leaf. Per-layer metadata that
carries it costs no walk time per fragment.

CUDA-graph replay: the three KDA entry points -- `chunk_kda_det_kcp`,
`chunk_kda_det_fwd_kcp` and `chunk_kda_det_bwd_kcp` -- take `graph="replay"`
(default `"off"`, the eager path); the convolution halo takes `graph=` only
through the backend module's function, the `cake_causal_conv1d_kcp` facade
has no such parameter. The first call of a
validity key runs eagerly, the second is captured into a CUDA graph and every
later call of the key replays it: the same launches and collectives, bitwise
the eager result, with no host work between them. The key is the call's kind,
its `ContextDetKcp` / workspace / backend objects, every option and the shape,
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
`kda_det_workspace_bytes_kcp(ctx, H=...)`, `kda_det_saved_bytes_kcp(ctx, H=...)` and
`kda_det_comm_bytes_kcp(ctx, H=..., C=...)` size the rank's workspace, saved set and
one step's traffic before any allocation.  Every rank of the group calls the
same entry points in the same order; the collectives are issued inside them.

Workspace of a rank (`WorkspaceDetKcp`, caller-held, reused across calls and
layers): the local arena of the token-side stages plus the boundary pools.
The backward's b1 / b2 send pools reuse the storage of the forward's c1 / c2
send pools (a call waits its sends before a backward writes either), and the
ring exchange's hop pools are allocated by the first `exchange="ring"` call.
`kda_det_workspace_bytes_kcp` is the number of bytes an a2a workspace allocates
(`WorkspaceDetKcp.bytes` after its first backward); the saved set and the
convolution's halo rows are per-call tensors outside it.  A workspace that
has served forwards only (`kda_det_workspace_bytes_kcp(..., backward=False)`) holds
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
one.  `kda_det_workspace_bytes_kcp([ctx_a, ctx_b, ...], H=H)` is its size: each pool at
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
`WorkspaceDetKcp(ctx, H, allocate=False)` allocates nothing until its first call
and sizes that call for the backward, so a first forward does not build a
forward-only arena that the first backward replaces.

Saved set of a rank (`SavedDetKcp`, `boundary_mode="saved"`): the fresh `S_in`
(the initial states of its fragments), with `save_transfer` the owner's c1
receive tensor and `M_seg`, on a rank whose fragments are split into `P > 1`
segments the forward's carry plane (the saved-mode backward replays its tape
from it instead of re-walking the segments and re-merging the carries; the
private test knob `saved_carry=0` selects the re-walk), and the call's
scalars; the int32 tables are rebuilt from the context.  The token-row `Aqk` /
`Akk` of the single-rank saved set are not part of it: the backward recomputes
the internal rows in both modes (saved set and checkpoint replay run the same
local programs), so `kda_det_saved_bytes_kcp` counts the tables, `S_in`, the
`save_transfer` tensors and the carry plane (and the convolution's history
rows with `with_conv`).

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

Host synchronization: no KCP call and no pack preparation copies a table from
pageable memory or waits for queued device work, with or without a
caller-owned workspace. `build_context_det_kcp` copies the context's tables in one
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
a key eagerly). Not covered: `workspace=None` calls allocate their pools per
call (a caching-allocator miss can call `cudaMalloc`, an out-of-memory retry
synchronizes; reuse a workspace under memory pressure), the debug paths
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
`ValueError`. `SavedDetKcp.save_transfer` always reports the value that was used.
`spec3_probe` sets the debug probe of the backend's first saved-mode backward,
which compares the backward's prep and WY rows with the forward's: `"off"` (the
default) skips it, `"record"` stores the comparison in
`LAST_CALL["bwd"]["spec3"]`, `"warn"` warns on a difference, `"assert"` raises
`KcpPreconditionError`.  A probing backward keeps five token-row planes (about
`ctx.R * H * 1536` bytes: 1.5 KiB per head for every row of the rank's
chunk-padded token layout) allocated through its prep and WY programs, so its
peak memory is higher by up to that amount.
The backward reads the value of its own call (pass it to
`chunk_kda_det_bwd_kcp`; the forward's is not kept in `SavedDetKcp`).  Pass the
same value on every rank: the probe is a rank-local check, so `"assert"`
serves single-process debugging.  Under `graph="replay"` the probing
backward runs eagerly; a rank without tokens does not hold back its capture.
No environment variable selects it.

Checkpoint recompute without the state exchange (issue #7815): under
`torch.utils.checkpoint` the backward re-runs the forward of the region, and
with it the forward's c1 / c2 exchanges (and the convolution's halo exchange).
A recompute handle keeps what the original forward received instead:

```python
from flashinfer.cake_kda_det_train_kcp import RecomputeHandleDetKcp, chunk_kda_det_kcp
from flashinfer.cake_causal_conv1d_train import cake_causal_conv1d_kcp

def layer(xq, xk, xv, g, beta, hq, hk, hv):
    q = cake_causal_conv1d_kcp(xq, wq, ctx, recompute_handle=hq)   # "conv" entry of hq
    k = cake_causal_conv1d_kcp(xk, wk, ctx, recompute_handle=hk)
    v = cake_causal_conv1d_kcp(xv, wv, ctx, recompute_handle=hv)
    ...                                                          # split heads, normalize q / k
    return chunk_kda_det_kcp(q, k, v, g, beta, ctx, recompute_handle=hq)   # "kda" entry of hq

handles = [RecomputeHandleDetKcp() for _ in range(3)]               # per layer and micro-step
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
`kda_det_saved_bytes_kcp(ctx, H=..., checkpoint_keep=..., with_conv=False)` bytes for
the KDA entry until it is dropped (`with_conv` adds one convolution's history
rows); `kda_det_comm_bytes_kcp(ctx, ..., checkpoint_keep=...)` reports the step's
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
nesting level -- a mismatch hangs or corrupts data, and no rank checks the
others.  The phase comes from `torch._C._current_graph_task_id()`; a torch
without it refuses the handle.

One halo exchange for a layer's q / k / v convolutions:
`cake_causal_conv1d_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx)` returns the
three outputs bitwise equal to three `cake_causal_conv1d_kcp` calls (outputs,
`dx` and `dweight`), but its forward and its backward each issue one halo
collective instead of three: the history rows of the three tensors travel side
by side in one `all_to_all_single` and are copied out after the wait, and each
tensor then runs its own kernels. With `recompute_handle=h` one handle serves
the call (its `"conv"` entry holds the three history pools) and may also carry
the layer's `"kda"` entry, so the checkpointed layer above needs one handle:

```python
def layer(xq, xk, xv, g, beta, h):
    q, k, v = cake_causal_conv1d_kcp_multi((xq, xk, xv), (wq, wk, wv), ctx, recompute_handle=h)
    ...
    return chunk_kda_det_kcp(q, k, v, g, beta, ctx, recompute_handle=h)

out = torch.utils.checkpoint.checkpoint(layer, xq, xk, xv, g, beta, RecomputeHandleDetKcp(),
                                        use_reentrant=False)
```

Per layer the halo collectives drop from 3 / 3 / 3 (original forward /
recompute / backward, without handles) or 3 / 0 / 3 (with handles) to 1 / 1 / 1
or 1 / 0 / 1.

Raw gate mode on the KCP route (issue #7816): the three KDA entry points take
the single-rank keywords `use_gate_in_kernel=True, A_log=, dt_bias=,
lower_bound=` with the same operand rules (`g` is the rank's raw BF16 gate,
any view under the stride rule of "Raw gate mode" in the single-rank README,
`flashinfer/experimental/cake_kda_det_train/README.md`, never copied;
`A_log` / `dt_bias` are the full parameters, the same tensors on every rank).

```python
from flashinfer.cake_kda_det_train_kcp import WorkspaceDetKcp, reduce_gate_param_grads_det_kcp

ws = WorkspaceDetKcp(use_gate_in_kernel=True)                     # the workspace's mode, fixed
out_local, saved = chunk_kda_det_fwd_kcp(q_local, k_local, v_local, raw_local, beta_local, ctx,
                                         workspace=ws, use_gate_in_kernel=True, A_log=A_log,
                                         dt_bias=dt_bias, lower_bound=-5.0)
dq, dk, dv, dbeta, draw, dA_log_r, ddt_bias_r = chunk_kda_det_bwd_kcp(
    q_local, k_local, v_local, raw_local, beta_local, ctx, dout_local, saved=saved, workspace=ws,
    use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0)
dA_log, ddt_bias = reduce_gate_param_grads_det_kcp(ctx, dA_log_r, ddt_bias_r)   # optional, see below
```

* The local stages run the raw records of the rank's plan (the intra and prep
  programs activate the gate, the raw GLM intra program writes `draw` and the
  per-chunk partials, one `param_finalize` launch reduces them; no gate
  epilogue).  The `gk` rows the K1 / K2 / K3 walks read are the explicit
  path's bits for the same activated gate, so every KCP program, collective
  and exchanged byte is the explicit path's; no full-size FP32 gate or gate
  gradient exists on the route.
* `dA_log` / `ddt_bias` of a rank are its partial sums: the gradients through
  its own tokens, reduced on the rank in an order fixed by its local chunks
  (bitwise repeatable, the same under every exchange, schedule, graph replay
  and recompute).  The full gradient is their sum over `ctx.group`; a
  training framework sums them as it sums the gradients of any parameter
  replicated across the context-parallel ranks.  An NCCL all-reduce gives
  that sum to FP32 rounding but does not fix its summation order (it changes
  with the algorithm, topology and message size), so it is not bitwise
  reproducible; `reduce_gate_param_grads_det_kcp(ctx, dA_log, ddt_bias)` is:
  one `all_gather_into_tensor` of the `H + 128 H` values, then an FP32 sum in
  rank order on every rank (the same bits everywhere).
* The mode is fixed per workspace (`WorkspaceDetKcp(use_gate_in_kernel=True)`:
  its local arena is the replay arena planned in raw mode, whose backward
  layouts carry no gate epilogue buffers; the KCP pools do not change;
  `local_arena="lean"` is refused in raw mode), recorded in the saved set
  (`SavedDetKcp.gate`, `.lower_bound`) and in a recompute handle's
  fingerprint; a call in the other mode, a backward with another
  `lower_bound` and a recompute of another mode are refused with
  `ValueError`.  `kda_det_workspace_bytes_kcp(ctx, H=..., use_gate_in_kernel=True)`
  sizes it; `kda_det_saved_bytes_kcp` and `kda_det_comm_bytes_kcp` do not
  depend on the mode.
* The backward allocates the raw gate's BF16 gradient, the partials (two FP32
  `[NC_pad, H, 128, 2]` buffers of the rank's chunks, for the duration of the
  call) and the two parameter gradients; a rank without tokens returns zero
  parameter gradients.

In-kernel Q / K L2 norm on the KCP route (issue #7817): the three KDA entry
points, `WorkspaceDetKcp` and the workspace and saved byte queries take the
keyword-only `use_qk_l2norm_in_kernel=False`.  With `True` the local stages
normalize `q` and `k` per token and per 128-channel head (FP32 sum of squares,
`r = 1 / sqrt(ss + 1e-6)`, `x_hat = bf16(x * r)`; `v` is not normalized) and
the GLM intra programs fuse the FP32 pullback into the BF16 `dq` / `dk`, so
`q_local` / `k_local` are the rank's raw rows and the gradients are theirs.

```python
ws = WorkspaceDetKcp(use_qk_l2norm_in_kernel=True)                # the workspace's mode, fixed
out_local, saved = chunk_kda_det_fwd_kcp(q_raw, k_raw, v_local, g_local, beta_local, ctx,
                                         workspace=ws, use_qk_l2norm_in_kernel=True)
dq_raw, dk_raw, dv, dbeta, dg = chunk_kda_det_bwd_kcp(
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
* The saved set keeps the raw `q` / `k` (the caller's tensors) and adds the
  FP32 rstd `SavedDetKcp.qk_rstd` `[2, S_local, H]` (plane 0 q, plane 1 k;
  empty on a rank without tokens): `kda_det_saved_bytes_kcp(...,
  use_qk_l2norm_in_kernel=True)` is `8 * S_local * H` bytes more.  The
  normalized q / k are never saved: the backward rebuilds them bitwise from
  the raw rows and the rstd into a per-call buffer (`512 * S_local * H` bytes
  for the duration of the backward), and a checkpoint recompute writes its
  rstd afresh, so a recompute handle's entry and bytes do not change.
* The mode is fixed per workspace (the replay local arena of that mode;
  `local_arena="lean"` is refused with it), recorded in the saved set
  (`SavedDetKcp.qk_l2norm`), the graph validity key and a recompute handle's
  fingerprint; a call, a saved set or a recompute of the other mode is
  refused with `ValueError`.
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
  call, a `WorkspaceDetKcp` and the byte queries with the flag on raise
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

* `cake_kcp_jit.py` -- the `MODULES` registry of the eleven shipped programs (the
  fragment walks of the forward and backward state recurrences in their
  `offset` and `tape` forms, the BF16x3 transfer walk (two CTAs per segment
  and head, each over one 64-column half of the state), the affine
  merge at `cols_per_cta` 8 and 16, and the two chunk stages of the
  context-parallel route, `kcp_fwd_o` (the output from the K1 tapes) and
  `kcp_dav` (`dA_qk` and the `dv1` K2 reads); selection by `always` or by
  `columns` on the merge's `ncols`, grids by `kcp_segments_heads`,
  `kcp_chains`, `kcp_tile_items` or `kcp_pairs`), the logical vocabulary of
  the KCP arguments, `run_program(name, args, device_index=)`, the lowering
  of the host backend's walk and merge requests to the registry's launches,
  and `run_module` for the two chunk stages.  Its `TEST_MODULES` are
  test-only programs of `build.test.json` (below).
* `cake_kcp_plan.py` -- the partition, the segment plan and the context's
  tables and collectives, with the byte queries.
* `cake_kcp.py` -- the per-rank launch sequences, the saved set and workspace
  of a rank, the autograd `Function` and the three entry points.
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
  package (byte-identical copies) and the manifest of the eleven shipped
  programs (schema `cake-build/1`, one entry per `<library>_<arch>`);
  `aot/` is never tracked (`.gitignore`).
* `csrc/cake_ffi_checks.h` -- the TVM-FFI helper header the bindings include.
* `csrc/cake_kda_det_train_<digest>_kcp_{kernel,binding}.cu` of the fifteen
  context-parallel-only programs and `csrc/cake_kda_det_train_<digest>_{kernel,binding}.cu`
  of the two chunk stages (`kcp_fwd_o`, `kcp_dav`), rendered like the
  single-rank ones (digests in `build.json` and `build.test.json`).

## Status

The registry holds eleven shipped programs (`cake_kcp_jit.MODULES`, in
`build.json`) and six test-only programs (`cake_kcp_jit.TEST_MODULES`, in
`build.test.json`) for `sm_100a`, `sm_103a` and `sm_107a`.

Tests: `tests/experimental/test_cake_kda_det_train_kcp_build.py` (the KCP
registries and manifests, no device needed),
`tests/experimental/test_cake_kda_det_train_kcp.py` (the per-rank path),
`tests/experimental/test_cake_kda_det_train_kcp_dist.py` (multi-GPU rows) and
`tests/experimental/test_cake_kda_det_train_kcp_api.py` (the public entry
points and the previous names); benchmark
`benchmarks/bench_cake_kda_det_train_kcp.py` (`torchrun`, one process per
rank).
