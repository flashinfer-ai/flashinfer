# Cake deterministic packed GDN context-parallel (KCP) training backend (SM100 / SM103 / SM107)

Experimental backend for context-parallel training of packed GDN (scalar-gated
DeltaNet) layers with native grouped Q/K heads: forward, backward and
checkpoint replay over a process group whose ranks each hold fragments of the
documents of one packed batch, with precise (FP32 / TF32x3-class) cross-rank
passing of the per-document recurrent state.  Tokens never move between ranks;
only affine summaries of the state do.  Tracking issue:
flashinfer-ai/flashinfer#7082.

The facade `flashinfer.cake_gdn_det_train_kcp` is a stable FlashInfer API:
its entry points are `@flashinfer_api` functions.  This package, under
`flashinfer.experimental.cake`, is its backend.
SM100 (B200 / GB200) and SM103 (B300 / GB300) are the acceptance architectures
of the kernels; SM107 (R200) is compiled from the same sources with an nvcc
that emits `compute_107a`.  The feature does not participate in automatic
backend selection, autotuning or trace-apply.  Its libraries are loaded
prebuilt from `aot/` when the package ships them and JIT-built by FlashInfer
from `build.json` otherwise (see "Layout of this package").  The package
imports nothing from the KDA training package and reads no environment
variable.

> Status: scaffolding.  The numbers of this file (shapes, bytes, limits) are
> the contract; measured figures are added with the kernels.

## Mode and dispatch rules

- This package (`flashinfer.experimental.cake.cake_gdn_det_train_kcp`) and its
  facade `flashinfer.cake_gdn_det_train_kcp` are the deterministic (`det`)
  mode only.  Every public name carries the mode word (FlashInfer #8928):
  `cake_gdn_det_kcp` (the autograd entry), `CakeGDNDetKcp<Noun>` (the types,
  the autograd class `CakeGDNDetKcpFunction`) and
  `cake_gdn_det_train_kcp_<verb>` (the other functions).  No entry takes a
  `deterministic` keyword: passing one (any value) is a `TypeError`, like
  every unknown keyword.
- `torch.use_deterministic_algorithms(True)` changes nothing here: every launch
  has a fixed order, so no alert is raised and outputs and gradients are
  bitwise reproducible run to run at the same topology.  No entry routes to
  another mode.
- `cake_gdn_det_train_kcp_backward` (and the autograd backward) reads the
  deterministic forward's `CakeGDNDetKcpSaved` and a caller-held
  `CakeGDNDetKcpWorkspace` only.  It refuses a saved set or a workspace of the
  non-deterministic mode with `ValueError` before any launch.
- The non-deterministic mode is a separate package with its own partition,
  context and options types.  Its plan is its own: the `b1` layout differs
  from this package's, and the head plan and the `P` rule are each package's
  own rules, so a context of one mode is not passed to the other.
- This package's `__init__.py` exports the mode-word names only.  The
  constants and backend controls (`SCALE`, `LAST_CALL`, `PROGRAMS`,
  `KCP_SAVED_VERSION`, `set_backend`, `get_backend`, `graph_cache_clear`,
  `make_options`, ...) are read from their modules (`cake_kcp`,
  `cake_options`).

## Public entry points (`flashinfer/cake_gdn_det_train_kcp.py`)

```python
from flashinfer.cake_gdn_det_train_kcp import (
    cake_gdn_det_kcp, CakeGDNDetKcpFunction,
    cake_gdn_det_train_kcp_partition, cake_gdn_det_train_kcp_build_context,
    cake_gdn_det_train_kcp_forward, cake_gdn_det_train_kcp_backward, cake_gdn_det_train_kcp_packed_qkv_views,
    cake_gdn_det_train_kcp_workspace_bytes, cake_gdn_det_train_kcp_saved_bytes, cake_gdn_det_train_kcp_comm_bytes,
    is_cake_gdn_det_train_kcp_supported,
    CakeGDNDetKcpPartition, CakeGDNDetKcpContext, CakeGDNDetKcpOptions, CakeGDNDetKcpRecomputeHandle,
    CakeGDNDetKcpSaved, CakeGDNDetKcpWorkspace,
)

part = cake_gdn_det_train_kcp_partition(cu_seqlens_global_cpu, world_size, layout="zigzag_per_document")   # once per batch layout
ctx = cake_gdn_det_train_kcp_build_context(part, group, rank, device, conv_kernel_size=4)                 # tables and collectives of this rank

out = cake_gdn_det_kcp(q, k, v, g, beta, ctx)                                             # autograd
out, saved = cake_gdn_det_train_kcp_forward(q, k, v, g, beta, ctx, save_for_backward=True)
dq, dk, dv, dbeta, dg = cake_gdn_det_train_kcp_backward(q, k, v, g, beta, ctx, dout, saved=saved)
dq, dk, dv, dbeta, dg = cake_gdn_det_train_kcp_backward(q, k, v, g, beta, ctx, dout)                 # saved=None: checkpoint replay

ws = cake_gdn_det_train_kcp_workspace_bytes(ctx, H_qk=16, HV=48)                                      # per rank, pure CPU: a fresh workspace's first call
```

These seventeen names are the whole public surface (the `__all__` of
`flashinfer/cake_gdn_det_train_kcp.py`).  Every switch is a keyword argument
of the public call (`CakeGDNDetKcpOptions` fields); no module-level global and no
environment variable selects anything.

### Names

The package moved from `flashinfer.experimental.cake_gdn_det_train_kcp` and
its names took the mode-word scheme of FlashInfer #8928.  The former names and
the former path were removed with no alias (they fail at import), as were the
earlier no-det names (`flashinfer.gdn_train`,
`flashinfer.experimental.cake_gdn_train`, `chunk_gdn_kcp`, `KcpContext`,
`kcp_partition`, ...).

| former name | name |
|---|---|
| `flashinfer.experimental.cake_gdn_det_train_kcp` | `flashinfer.experimental.cake.cake_gdn_det_train_kcp` |
| `chunk_gdn_det_kcp` | `cake_gdn_det_kcp` |
| `ChunkGDNDetFunctionKcp` | `CakeGDNDetKcpFunction` |
| `chunk_gdn_det_fwd_kcp` / `chunk_gdn_det_bwd_kcp` | `cake_gdn_det_train_kcp_forward` / `cake_gdn_det_train_kcp_backward` |
| `partition_det_kcp` / `build_context_det_kcp` | `cake_gdn_det_train_kcp_partition` / `cake_gdn_det_train_kcp_build_context` |
| `packed_qkv_views_det_kcp` / `PackedQKVSplitDetKcp` | `cake_gdn_det_train_kcp_packed_qkv_views` / `CakeGDNDetKcpPackedQKVSplitFunction` |
| `gdn_det_workspace_bytes_kcp` / `gdn_det_saved_bytes_kcp` / `gdn_det_comm_bytes_kcp` | `cake_gdn_det_train_kcp_workspace_bytes` / `cake_gdn_det_train_kcp_saved_bytes` / `cake_gdn_det_train_kcp_comm_bytes` |
| `gdn_det_workspace_layout_kcp` / `gdn_det_saved_layout_kcp` | `cake_gdn_det_train_kcp_workspace_layout` / `cake_gdn_det_train_kcp_saved_layout` (`cake_bytes`) |
| `is_gdn_det_train_kcp_supported` | `is_cake_gdn_det_train_kcp_supported` |
| `OptionsDetKcp`, `ContextDetKcp`, `PartitionDetKcp`, `FragmentDetKcp` | `CakeGDNDetKcpOptions`, `CakeGDNDetKcpContext`, `CakeGDNDetKcpPartition`, `CakeGDNDetKcpFragment` |
| `WorkspaceDetKcp`, `SavedDetKcp`, `RecomputeHandleDetKcp`, `CallDetKcp` | `CakeGDNDetKcpWorkspace`, `CakeGDNDetKcpSaved`, `CakeGDNDetKcpRecomputeHandle`, `CakeGDNDetKcpCall` |

The constants (`SCALE`, `DEFAULT_SCALE`, `LAST_CALL`, ...), the `LAST_CALL`
keys `"fwd"` / `"bwd"`, the program names and the layout keys are data
identities and keep their names.

The generated libraries are named `cake_gdn_det_train_<20 hex>` for the
per-chunk programs and `cake_gdn_det_train_<20 hex>_kcp` for the
context-parallel-only programs (`kcp_*`); the kernel symbol of a library is
`kernel_<library>`.  Library names have no compatibility aliases.

### Contract (per rank; `T_r` = this rank's token count, any value including 0)

| operand | shape | dtype | notes |
|---|---|---|---|
| `q`, `k` | `[1, T_r, H_qk, 128]` | BF16 | grouped Q/K heads; `HV % H_qk == 0`; production `(H_qk, HV) = (16, 48)`, `(1, 3)` and `(4, 4)` are legal |
| `v`, `out`, `dout` | `[1, T_r, HV, 128]` | BF16 | one value head per state |
| `g` | `[1, T_r, HV]` | FP32 | scalar natural-log decay per token and value head; never expanded to 128 channels |
| `beta` | `[1, T_r, HV]` | FP32 | after the sigmoid; no BF16 cast |
| `dq`, `dk` | like `q`, `k` | BF16 | the `HV / H_qk` value heads of a Q/K head are reduced in FP32 in a fixed order and rounded once |
| `dv` | like `v` | BF16 | |
| `dg`, `dbeta` | like `g`, `beta` | FP32 | |

* The leading batch axis of 1 is optional on every operand; the outputs take
  the rank of the inputs.
* Strided views are accepted (`layout="tma"`, the default): the packed QKV
  row of `(2 * H_qk + HV) * 128` elements is read in place through its token
  stride (`q` at element 0, `k` at `H_qk * 128`, `v` at `2 * H_qk * 128`).
  `layout="copy"` makes a contiguous copy first; its time is part of the
  measured module when reported.  Strided and contiguous inputs give bitwise
  the same results.
* When `q` / `k` / `v` are head views of one packed row, take them with
  `cake_gdn_det_train_kcp_packed_qkv_views` for full backward speed: plain `split` views of
  the row give the same gradients, but the backward is about 1.1 ms (33-36 %)
  slower at `(16, 48)` and 32,768 tokens (**Packed Q/K/V** below).
* `scale` defaults to `128 ** -0.5`.
* The host reads no tensor contents (no `.item()`, `.tolist()`, no implicit
  synchronization): every table is built on the CPU from the partition
  metadata, uploaded once per `(HV, SM count, P)` key at first use
  (`ctx.prepare(H_qk, HV)`, also called by the first forward) and kept
  device-resident on `ctx`; the launch grids follow from `(T_r, H_qk, HV, SM
  count, architecture)` alone.  Tables are int32 except the affine merge's
  element offsets, which are int64.  Steady state is CUDA-graph capturable.
* Repeated calls on the same inputs give bitwise the same outputs; the
  backward from the saved set and the checkpoint replay give bitwise the same
  gradients; `torch.utils.checkpoint` direct and indirect backward agree
  bitwise.  Gradient accumulation (`.backward()` twice) accumulates.
* No cooperative launch anywhere in the package; every program is a plain
  stream-ordered launch.
* Host launch path: each launch binds torch's current stream for tvm-ffi
  through `tvm_ffi.use_raw_stream` with that stream's device and raw handle
  (the stream `tvm_ffi.use_torch_stream()` binds, without building a
  `torch.cuda.Stream` per launch).  The registry's answers (in-kernel L2 norm
  and fused-output records) and the chunk table's token count are derived once
  and reused.  An eager call sizes its workspace pools and ticket slots after
  its first launch is issued, so that host work overlaps the launch (a
  backward whose first program is persistent, #6 on saved or kept tapes, sizes
  the ticket slots alone before it); a captured call sizes them first.  Each
  launch key's record, grid and test flag are resolved once per context plan,
  and the plan's pool tables with their kernel addressing, its walk, merge-chain
  and zero-block device tables, split lists and pool sizes are derived at first
  use; `cake_gdn_det_kcp` validates its keyword options once per keyword set.
  Launches, arguments and bits are unchanged.
* On `sm_100a` and `sm_103a`, with a zigzag or zigzag-per-document partition
  of at most 8 ranks, the summaries' offset walk runs on a side stream beside
  the transfer walk (fork / join with CUDA events; the same launches,
  arguments and bitwise the same outputs).  `sm_107a` keeps one stream: on a
  212-SM R200 the CP4 zigzag rows ran 1.2-2.5 % slower paired, and there is
  no 216-SM measurement.  The private lever `test_knobs={"walk_pair": 0}` (or
  `1`) forces one stream (or the pair on a zigzag partition).
* A forward with `save_mode="light"` and `boundary_mode="saved"` that
  launches the output program and hands no tapes to a backward runs the
  output walk `kcp_fwdh_out` (#2+5 under "Kernel structure") at each of its
  tape walks in place of the #2 tape walk and #5: each chunk's output comes
  from the walk's own `h` snapshot and `v_new` operand, and the forward
  allocates neither tape.  Its workspace peak is #1's.  Under
  `graph="off"` the split form's tape walk holds the `h` / `v_new` tapes in
  place of #1's `akk` / `gcum` (30,528 B per padded row more at `HV = 48`), less
  the c1 send pool and its `M_f^T` tiles on a rank whose group exchanges
  and, in a forward without a saved set, less the arena `qk_rstd` (each
  released before that walk): the split form peaks above #1 by that
  difference when it is positive, else at #1 too.  Under `graph="replay"`
  nothing is released, and the split form peaks above #1 by the `h` /
  `v_new` tapes (36,864 B per padded row at `HV = 48`).
  `cake_gdn_det_train_kcp_workspace_layout(...)` reports both as `forward` and
  `forward_split`.  The output and the saved set are
  bitwise those of the split form.  `save_mode="full"`,
  `boundary_mode="replay"`, a forward that hands its tapes to the kept-tape
  backward, a re-run that leaves #5 out and the backward's replayed forward
  keep the split form.  The private lever `test_knobs={"fused_o": 0}` forces
  the split form.

### Partition and context

`cake_gdn_det_train_kcp_partition(cu_seqlens_global_cpu, world_size, layout, fragments=None)`
splits the packed batch over `world_size` ranks: `"contiguous"` (equal
contiguous slices), `"zigzag"` (the whole buffer in `2 * world_size` pieces,
rank `r` holding pieces `r` and `2W - 1 - r`), `"zigzag_per_document"` (the
zigzag applied inside every document) or explicit `fragments`.  Cut points
need not be multiples of the chunk length (64); ranks may hold unequal token
counts; a fragment may be empty, one token long or cross a chunk boundary; a
document may span any number of ranks including the whole group.

`cake_gdn_det_train_kcp_build_context(part, group, rank, device, conv_kernel_size=4)` returns
the rank's `CakeGDNDetKcpContext`: the fragment table, the state-ownership plan (head
groups over `HV`, padded when `HV` does not divide by the group size), the
causal-convolution halo plan (`conv_kernel_size - 1` history rows per
crossing-in fragment) and the int32 device tables of the kernels.  The same
context serves every call of the same batch layout and head pair.

### Options (`CakeGDNDetKcpOptions`, keyword-only; defaults in bold)

| option | values | meaning |
|---|---|---|
| `save_mode` | **`"light"`**, `"full"` | `"light"` keeps the boundary states (with `save_transfer=True` also the received summary pairs and `M_seg`) and no tape: the backward re-runs the chunk preparation and replays the intra-rank tapes; `"full"` also keeps `M_seg` and every forward tape (the ten tapes `h`, `v_new`, `w`, `u`, `Aqk`, `Akk`, `gamma`, `gcum`, `rowscale`, `qscale`: 74,307 B per padded token row at `HV = 48`; with `use_qk_l2norm_in_kernel` also the normalized rows `qn` / `kn`, 8,192 B per token at `H_qk = 16`, so its backward launches no `qk_apply`) |
| `boundary_mode` | **`"saved"`**, `"replay"` | whether the cross-rank boundary states (`S_in`, the owner's received summaries) are kept in the saved set or recomputed (exchange included) in the backward; `"replay"` keeps no tape and no `qk_rstd` either: `save_mode="full"` there runs the same backward as `"light"` (the whole-forward replay) |
| `exchange` | **`"a2a"`**, `"ring"` | the collective schedule; `"ring"` only when every document crosses at most 3 ranks (`n_d_max - 1 <= 2`) and `save_transfer=False` |
| `save_transfer` | **`True`**, `False` | keep the received summary pairs for the backward (`False` re-sends the transposed transfers) |
| `precision_mode` | **`0`** (TF32 hi/lo), `1`, `2` (test-only) | the state-transfer products of the transfer walk: `0` is the shipped two-MMA hi/lo form (the TF32x3 class); `1` is its three-MMA twin (bitwise the same results) and `2` plain TF32 (the negative control), both test-only programs of `build.test.json` |
| `transfer_walk` | **`"tf32_hilo"`** | the transfer-walk program; the vocabulary also names `"bf16x3"` (BF16x3 state products), which this package does not build: the validator refuses it as not built |
| `layout` | **`"tma"`**, `"copy"` | strided input read in place, or a timed contiguous copy |
| `graph` | **`"off"`**, `"replay"` | CUDA-graph replay of the steady state (a static pool per key) |
| `reuse_recompute_tapes` | **`"auto"`**, `True`, `False` | under `torch.utils.checkpoint(use_reentrant=False)` the light backward binds the forward tapes torch's recompute just wrote instead of replaying #1 and the #2 tape walk (bitwise the same gradients, the retention unchanged; a no-op outside that backward; `False` restores the replay form). `"auto"` keeps the tapes only when the GDN node is the last operation of the checkpointed region that saves a tensor (the same checkpoint-frame proof as `recompute_output="auto"`): then its backward is the region's first and the kept tapes die before any other backward runs; with a later saver in the region (an output projection, an MLP) the kept set would live through those operations' backward and raise the region's backward peak by the full tape set (`cake_gdn_det_train_kcp_saved_bytes(save_mode="full")` minus the saved states), so `"auto"` takes the replay form there. Measured on one GB300 (torch 2.14.1), 261,987 tokens, `(H_qk, HV) = (2, 6)`: a region of GDN + a 768x768 output projection + a 768->2x2048->768 SwiGLU has a backward peak of 9250.54 MiB with the kept tapes (`True`) and 6927.31 MiB with the replay form (`False` and `"auto"`), at 21.29 vs 22.70 ms per training step of the region; a region the GDN node closes keeps the tapes under `"auto"` (five 240k-262k-token (16, 48) packs: training step and steady peak unchanged against `True`). `True` always keeps the tapes, `False` never. With `graph="replay"`, the one mode that excludes the kept tapes, the default is `False` and an explicit `True` / `"auto"` is refused. One live kept set: in a checkpoint segment with several GDN layers only the last layer skips the replay, the earlier layers replay as before |
| `recompute_output` | **`"auto"`**, `"always"` | under `torch.utils.checkpoint(use_reentrant=False)` torch re-runs the forward inside `.backward()` only to re-save its tensors and discards that run's output; when the GDN node is the last operation of the checkpointed region that saves a tensor, `"auto"` does not launch the output program (#5) of that re-run. The proof is read from torch's checkpoint frame (the node is executable, the remaining saved-tensor holders are exactly this node's, early stop is on, and the torch release -- `torch.__version__` without its local `+cuNNN` build tag (`cake_kcp.torch_release`) -- is one of the verified releases in `cake_kcp.RECOMPUTE_OUTPUT_TORCH_VERSIONS`); any other state -- a later saver in the region such as the next layer or a norm on the output, a non-executable node, a reentrant checkpoint, `graph="replay"`, another torch build -- launches it as before, and `cake_kcp.LAST_CALL["fwd"]["fwd_o_reason"]` records why. Gradients and retained bytes are the same either way; `"always"` is the unconditional launch |
| `segments` | **`None`**, an int | how many intra-rank segments a fragment is walked as (`P`): `None` takes the context's plan, an int overrides it within `[1, min(64, chunks of the rank's longest fragment)]` (the byte queries' `P=`). The plan (test knob `p_rule` names its policy; test knob `walk_seg` an explicit override) is a function of metadata and the SM count only. The shipped policy costs the three per-rank programs over every rank's fragment table with a crossing-aware model and switches the whole row away from the per-rank default only when an alternative is cheaper by at least its margin (2 % for the `occ` program; 0 % for the `one` program, which has no segment composition and takes a model tie); on a row with at most 12 value heads whose fragments are all at most 160 chunks long every rank then runs `P = 1` (no segment composition), so the same topology gives the same `P` on every rank |
| `offset_class` | **`"fla"`**, `"fp32"` | the class of the forward offset `B_f`: the walk from zero with the reference rounding points, or an FP32 walk (`B_f` only) |
| `use_qk_l2norm_in_kernel` | **`False`**, `True` | the per-token L2 norm of every 128-channel Q / K head inside the programs (FLA's `use_qk_l2norm_in_kernel`; see "In-kernel Q / K L2 norm" below); a literal keyword of the three entry points as well |
| `test_knobs` | **`None`** | names the test-only values a test selects (`precision_mode` 1 or 2) and private levers such as `fwdh_vh` (forces the #2 value width) and `fused_o` (`0` forces the split tape walk and #5); without it every test-only value is refused |

The validator refuses every test-only value not named in `test_knobs`,
`exchange="ring"` without both of its conditions, `HV % H_qk != 0` and any
option value outside these tables, before any launch.  The public API cannot
select a test-only program: the test-only libraries live in `build.test.json`,
which the shipped route never reads.

### Saved set, replay and byte queries

With `s = HV * 128 * 128 * 4` bytes (3 MiB at `HV = 48`), `s_owner = Hg *
128 * 128 * 4` on a rank that owns heads (`Hg = ceil(HV / W)`), `s_pad =
n_owners * s_owner` (`= s` when `HV % Hg == 0`), `n_in(r)` the rank's
fragments with a predecessor on another rank, `n_out(r)` those with a
successor and `N_in` / `N_out` the group's totals (the per-token figures are
bytes per padded token row, `R = 64 * NC_pad` rows per rank, at `HV = 48`):

| item | `save_mode="full"` | `save_mode="light"` | bytes per rank |
|---|---|---|---|
| inputs `q, k, v, g, beta` (references) | yes | yes | 0 extra |
| `S_in` (incoming boundary states) FP32 | yes | yes | `n_in * s_pad` |
| received summary pairs (owner, `save_transfer=True`) | yes | yes | `2 * s_owner * N_out` |
| `M_seg` (segment transfers, `P > 1`) | yes | `save_transfer=True` only (else rebuilt by the replay) | `P * n_frag * s` |
| convolution history rows (the module convolution's; `with_conv=True` in the byte queries, no GDN call saves them) | yes | yes | `<= 3 * n_in * C * 2` |
| `gamma`, `gcum`, `Akk` (chunk-preparation outputs) | yes | **no** (the chunk preparation re-runs in the backward) | about 6,339 B per token |
| `rowscale`, `qscale` (chunk-preparation FP32 row factors) | yes | **no** (the chunk preparation re-runs in the backward) | 384 B per token |
| `h` tape, `v_new`, `w`, `u`, `Aqk` | yes | **no** (replayed in the backward) | 67,584 B per token |
| `qk_rstd` (`use_qk_l2norm_in_kernel=True`; `boundary_mode="saved"` only) FP32 | yes | yes | `8 * T_r * H_qk` |
| `qn`, `kn` (`use_qk_l2norm_in_kernel=True`: the normalized rows `fwd_prep_l2n` wrote) BF16 | yes | **no** (the replay's `fwd_prep_l2n` writes them again) | `2 * 256 * T_r * H_qk` (8,192 B per token at `H_qk = 16`) |

With `boundary_mode="replay"` the saved set keeps no boundary state, no
`M_seg`, no tape and no `qk_rstd` in either `save_mode`: the backward replays
the whole forward, and `save_mode="full"` runs the same backward as
`"light"`.

Everything is attached through the autograd `save_for_backward` only, so a
`torch.utils.checkpoint(use_reentrant=False)` forward retains 0 B per entry
(measured after `ctx.prepare` or one warm-up call, with `graph="off"`; the
`graph="replay"` static pool is outside that measurement).

Two replays exist, and they differ:

* The **light replay** is the backward of `save_mode="light"` with
  `boundary_mode="saved"`: it re-runs the chunk preparation and the tape
  walk from the saved `S_in` (segment-start states from `M_seg` when
  `P > 1`); no exchange, no transfer walk, no merge and no output program
  run again.  It is the only replay that starts from the saved `S_in`.
* The **checkpoint replay** (`torch.utils.checkpoint(use_reentrant=False)`
  around the module, or `boundary_mode="replay"`) retains nothing, so the
  recompute re-runs the whole module forward -- convolution halo, chunk
  preparation, offset and transfer walks, exchange, merge, tape walk and
  output -- and the backward then proceeds as in the saved form (around a
  `"light"` call with `boundary_mode="saved"`, with the light replay on top
  unless `reuse_recompute_tapes` keeps the recompute's tapes for it; with
  `recompute_output="auto"` the re-run may leave the output program out).
  Under `boundary_mode="replay"` the backward itself replays the whole forward
  in either `save_mode`; it is the minimum-memory option.

Both replays give bitwise the gradients of the saved-set backward.

A **recompute handle** removes the exchange from the checkpoint replay.  Pass
a `CakeGDNDetKcpRecomputeHandle` to the checkpointed function and to `cake_gdn_det_kcp`:

```python
def layer(x, handle):
    ...                                   # convolution, split, gates
    return cake_gdn_det_kcp(q, k, v, g, beta, ctx, recompute_handle=handle)

y = torch.utils.checkpoint.checkpoint(layer, x, CakeGDNDetKcpRecomputeHandle(), use_reentrant=False)
```

The original forward exchanges as always and stores this rank's incoming
states `S_in` (the saved set's own tensor) in the handle.  Torch's recompute
inside `.backward()` reads them back and runs the light replay's launches
(chunk preparation, at `P > 1` the local segment walks and carry merges, the
tape walk from `S_in`, then the output program): no collective, no summary
walk over the crossing-out fragments, no owner merge.  The gradients
are bitwise those of the same checkpoint without a handle; the backward still
exchanges.  The handle retains `S_in` --
`cake_gdn_det_train_kcp_saved_bytes(ctx, ..., save_mode="light", save_transfer=False,
with_conv=False)` -- until the checkpoint frame that holds it is freed.
Create one handle per layer and micro-step, outside the outermost checkpoint
region (a handle created inside a region is empty again in that region's
recompute, which then exchanges as usual).  The handle needs
`boundary_mode="saved"` and `graph="off"` and implies `save_transfer=False`.
A forward outside the backward that finds an entry raises (a handle reused
across micro-steps).  Other operations keep their own boundary rows in a
handle under their own key (the convolution's halo rows under `"conv"`).  A
handle holds one entry per key: it serves one GDN call and at most one
convolution call; a layer with separate Q / K / V convolutions uses one handle
per convolution call, and the GDN call shares one of them.

**Packed Q/K/V.**  `cake_gdn_det_train_kcp_packed_qkv_views(qkv, H_qk=..., HV=...)` splits a packed
`[T, (2 H_qk + HV) * 128]` row (the convolution's output) into `q` / `k` /
`v` head views without a copy; `cake_gdn_det_kcp` takes them as they are (token
stride `(2 H_qk + HV) * 128`).  When `q` / `k` / `v` are such views, the
backward writes `dq` / `dk` / `dv` as the same views of one fresh packed
row, and the split's backward returns that row as the gradient of `qkv`: no
concatenation and no copy between the GDN backward and the convolution's
backward.  The gradient values are bitwise those of separate operands
(`LAST_CALL["bwd"]["grad_layout"]` is `"packed"` or `"separate"`).

Callers whose `q` / `k` / `v` are head views of one packed row get the full
backward speed only through `cake_gdn_det_train_kcp_packed_qkv_views`.  Its conditions: `qkv`
is `[..., T, (2 H_qk + HV) * 128]` with a dense channel axis
(`qkv.stride(-1) == 1`), and `q` / `k` / `v` are its three head views in that
order.  Plain views of the same row
(`qkv.split((H_qk * 128, H_qk * 128, HV * 128), -1)`, then
`.view(T, heads, 128)`) give bitwise the same results, and the backward still
writes one packed row, but torch's split backward then concatenates `dq` /
`dk` / `dv` into a new row.  Measured cost of the plain
views (the #8367 repro: `(H_qk, HV) = (16, 48)`, 32,768 tokens in five
documents, one-rank context, forward and backward to the packed row; CUDA-event
ms, median of 60 = 3 rounds of 20 after 5 warm-up calls; PyTorch 2.14.1+cu130,
CUDA 13.0 runtime; both GPUs 152 SMs):

| GPU | in-kernel L2 norm | forward split / packed | backward split / packed | backward split - packed |
|---|---|---|---|---|
| GB200 (sm_100a) | off | 1.470 / 1.490 | 4.097 / 3.006 | +1.091 (1.363x) |
| GB200 (sm_100a) | on | 1.658 / 1.676 | 4.388 / 3.296 | +1.092 (1.331x) |
| GB300 (sm_103a) | off | 1.440 / 1.452 | 4.021 / 2.948 | +1.074 (1.364x) |
| GB300 (sm_103a) | on | 1.602 / 1.619 | 4.308 / 3.239 | +1.070 (1.330x) |

The three views are the outputs of one custom autograd node.  Hooks and
`retain_grad` on them fire as on any autograd output.  Its backward returns
the incoming packed row itself when `dq` / `dk` / `dv` are the head views of
one row of `qkv`'s shape and dtype (`PACKED_SPLIT_STATS["aliased"]`), and
concatenates them otherwise (`PACKED_SPLIT_STATS["concatenated"]`): a missing
gradient counts as zeros, and a hook that returns a new gradient also makes it
concatenate.  Autograd refuses an in-place change to the views while `qkv`
requires grad.  Plain `split` views keep torch's own autograd path and hooks.

```
cake_gdn_det_train_kcp_comm_bytes(ctx, *, H_qk, HV, C=None, save_transfer=True, boundary_mode="saved")
    -> this rank's {c1, c2, b1, b2, halo_fwd, halo_bwd: {send, recv}; B1, B2, B3 (the boundary totals); b1_d, b1_m,
    b1_design, b1_replay; padding; the crossing counts and state bytes}
    c1 send = 2 * s_pad * n_out(r)      c2 recv = s_pad * n_in(r)      b2 recv = s_pad * n_out(r)
    b1 send = s_pad * n_in(r), plus s_pad * n_io(r) with the re-sent maps (boundary_mode="saved",
    save_transfer=False; n_io(r): the rank's fragments crossing in and out, whose maps the owner's adjoint chain reads)
    on an owner: c1 recv = 2 * s_owner * N_out      c2 send = s_owner * N_in      b2 send = s_owner * N_out
    b1 recv = s_owner * N_in, plus s_owner * N_io with the re-sent maps      halo = rows(r) * C * 2 forward, * 4 backward
cake_gdn_det_train_kcp_saved_bytes(ctx, *, H_qk, HV, save_mode="light", boundary_mode="saved", save_transfer=True, with_conv=True,
    C=None, P=None, use_qk_l2norm_in_kernel=False) -> per rank (the table above)
cake_gdn_det_train_kcp_workspace_bytes(ctx, *, H_qk, HV, save_mode="light", boundary_mode="saved", save_transfer=True, backward=True,
    with_conv=True, C=None, P=None, group_reduce="fused", reuse_recompute_tapes=None, use_qk_l2norm_in_kernel=False,
    fused_walks=1, graph="off", save_for_backward=True)
    -> per rank: the most the caller-held workspace binds in the forward or the backward.  Covered: exchange="a2a"
    under the shipped wait_first schedule, graph "off" or "replay", with or without a saved set, on a workspace
    created empty (CakeGDNDetKcpWorkspace()): its first forward and the backward that follows on the arena that forward left
    (save_for_backward=False: with backward=True the replay a saved=None backward runs, with backward=False a
    forward-only call).  A backward leaves its tapes bound, so every later backward on the same workspace (another
    layer, the next step) binds the graph="replay" figure (measured for the replay form; at most that figure right
    after the fused forward of a save_for_backward=False call, light with a saved boundary, which released h /
    v_new: its replayed forward binds them again only after the c1 release) and every later forward more than the
    first (the backward's tapes stay bound; under graph="replay" its b1 / b2 pools too) and at most that figure
    (equal to it under graph="replay"): size a reused workspace with graph="replay".
    CakeGDNDetKcpWorkspace(ctx, H_qk, HV, device) sizes every pool up front at the plan's P with the fused walks (the ring
    pools, c2_send, the c1_recv / stage_m pools a saving call replaces, the b1 / b2 pools; for a call with a smaller
    segments override or fused_walks=0 also the plan's larger staging, carry and M_f^T pools) and binds them from #1
    on.  Not covered: exchange="ring" (also binds ring_out / ring_in, max(n_in(r), n_out(r)) states each, and its
    forward -- also the forward a replay backward re-runs -- keeps the c2 send pool and the c1 send pool with the
    M_f^T tiles through the tape walk: up to fwd.c1_send + fwd.c1_send_mf more, plus the c2 send pool (bwd.c2_send)
    in the forward), the schedule="overlap" test knob and the kept form's recompute under a CakeGDNDetKcpRecomputeHandle (a
    handle forces save_transfer=False; with the query given save_transfer=False too, its c1 send pool stays bound
    through its tape walk, up to fwd.c1_send + fwd.c1_send_mf above bwd.phase.recompute_forward; the replay form's
    figure bounds it).  The figures plan a workspace: no call checks a workspace against them (a caller-held
    CakeGDNDetKcpWorkspace grows any pool or tape a call needs).
    graph="off": the forward is the larger of #1 (the pools, the eight tapes #1 writes -- none with save_mode="full"
    and a saved set, whose tapes are saved-set tensors -- and the arena qk_rstd of a forward without a saved set) and
    the tape walk (akk / gcum gone; the split form, also `forward_split` of the layout, adds h / v_new; with a
    crossing fragment in the group the c1 send pool is gone); the eager forward's c2 send plane is a fresh tensor.
    S_in is never a workspace term: a call without a saved set also holds its fresh n_in(r) * s_pad bytes from the
    c1 release to the end of the forward, or to #8 of the replay backward (under graph="replay" to the end of the
    backward).  The backward is the largest of its phases (the exchange, the adjoint tape walk, the end state;
    without a saved set also the replayed forward before its c1 send pool goes: the eight tapes #1 re-writes, with
    h / v_new only after a split forward).  The default (`None`, i.e. `"auto"`) reports the replay form, in which the
    ten forward tapes are workspace (no forward tape with save_mode="full" and a saved set) and whose workspace
    bounds the kept form's; `reuse_recompute_tapes=True` gives the kept-tape figure (light save_mode, saved set), in
    which they are the recompute forward's hand-off tensors outside the workspace, with its M_seg^T staging plane at
    P > 1 without save_transfer (`cake_gdn_det_train_kcp_workspace_layout` lists them as `bwd.kept_tapes` and the total as
    `backward_transient`, which that plane can put above the replay form's).  graph="replay": nothing is released,
    so the forward is every forward pool and tape it writes (no tape with save_mode="full" and a saved set) and the
    backward every backward pool, the ten light tapes (none with save_mode="full" and a saved set) and every
    backward tape; with use_qk_l2norm_in_kernel both add qn / kn (none with save_mode="full" and a saved set, whose
    rows are saved-set tensors) and the arena qk_rstd where #1 writes it (a forward without a saved set, a backward
    that re-runs #1).
    At P > 1 the c1 send pool also holds the M_f^T tiles (`c1_send_mf` = s * n_out(r)) and, with the fused walks, the
    carry plane max(n_in(r), n_out(r)) more slots (`fused_walks=0` is the test knob's split form); the b1 pools
    take the larger split form (the d_f plane, or the d_f plane + the n_io(r) re-sent maps); a backward on a caller-held workspace also sizes the forward exchange pools
    (c1_send, c2_send, c1_recv unless saved); with_conv=True (default) adds the layer convolution's halo rows, which
    the entry points do not allocate (with_conv=False: what a call binds); boundary_mode="replay" gives the same
    figures for both save_mode values
```

`H_qk` enters the halo and strided-view terms; the exchange terms depend on
`HV` alone.

### In-kernel Q / K L2 norm (`use_qk_l2norm_in_kernel=True`)

`q` / `k` are the raw rows (the convolution's output views pass through);
per token `t` and Q / K head `h`, with `x = fp32(q[t, h, :])`:

```
ss    = 16 lanes x 8 channels: fma.rn chains (lane c holds channels 8c..8c+7), then the xor tree m = 8, 4, 2, 1
r     = div.rn(1, sqrt.rn(ss + 1e-6))                                    FP32, saved as rstd
q_hat = cvt.rn.bf16(mul.rn(x, r))
y     = mul.rn(x, r)  (FP32, from the raw row and the saved r)
dot   = 32-channel quadrant trees (pairs c <-> c^m, m = 16..1), then ((S0 + S1) + S2) + S3 of mul.rn(dq_hat, y)
dq    = cvt.rn.bf16(mul.rn(r, sub.rn(dq_hat, mul.rn(y, dot))))           dq_hat = the FP32 group sum of #8
```

FTZ off everywhere; `cake_l2n.py` is the pure-torch reference of these
orders (bitwise the device programs; the `fma.rn` of `ss` is emulated
exactly in FP64).  Programs: `fwd_prep_l2n` (#1 that first writes `qn` /
`kn` and `r`, then runs #1's body on them; its forward form, record
`fwd_prep_l2n_fwd`, stores no `akk` / `g_cum` and runs for every forward
whose `akk` / `g_cum` reach no backward), `bwd_dqkg_l2n` (#8 on `qn` /
`kn` with the pullback before the one BF16 rounding; `group_reduce="host"`
runs the unchanged #8 and the reference pullback on the host, per block of
2048 tokens) and `qk_apply` (`qn = bf16(q * r)` from the saved `r`).  Every
other reader (#2, #3, #5, #2+5, #7) is the unchanged program bound to `qn` / `kn`.

| path | `qn` / `kn` from | `r` the pullback reads | dropped |
|---|---|---|---|
| forward | `fwd_prep_l2n` | -- | `kn` before `out`, `qn` after #5 (a fused forward: both after its output walks; a hand-off forward hands both over with its tapes); the arena `r` after #1 (the saved `r` is a fresh tensor) |
| light backward | the replay's `fwd_prep_l2n` | saved | after #8 |
| kept recompute tapes | the recompute forward's `fwd_prep_l2n` (handed over with the tapes; no `qk_apply`) | saved | after #8 |
| `save_mode="full"` | the saved set (the forward's `fwd_prep_l2n` rows; no `qk_apply`) | saved | after #8 (unbound; the saved set holds them) |
| `save_mode="full"`, test knob `full_qk="apply"` | `qk_apply` after #6 | saved | after #8 |
| `boundary_mode="replay"` | the replayed forward's `fwd_prep_l2n` | arena | after #8 |
| recompute handle | `fwd_prep_l2n` (the recompute is a forward) | -- | as the forward |
| `graph="replay"` | as eager | as eager | never (the graph bakes the arena) |

The saved set gains `qk_rstd` (FP32 `[2, T_r, H_qk]`, `8 T_r H_qk` bytes;
`KCP_SAVED_VERSION` 2); with `save_mode="full"` it also keeps the normalized
rows (`2 x 256 T_r H_qk` bytes, fresh tensors `fwd_prep_l2n` writes; the
same bits `qk_apply` rebuilds from the saved `r`), so the full backward
launches nothing before #6.  Otherwise the rows are never saved: workspace in
the forward's tape walk and every backward phase of the replay forms; in the
kept-tape backward they are hand-off tensors, `bwd.kept_tapes`.  The byte queries take `use_qk_l2norm_in_kernel=`;
`LAST_CALL[...]` records `qk_l2norm`, `qk_source`, `rstd_source` and
`arena_tapes`.  The recompute handle's entry key includes the flag.  Until
the registry renders the three programs the kernel backend refuses the
option before any launch (`cake_kcp.l2n_records_shipped()`); the fake
backend emulates them.

### Exchange

Per forward: one `all_to_all_single` carrying the `[B_f | M_f^T]` summary
pair of every crossing-out fragment to the owner of its heads (`c1`), the
owner's FP32 affine chains along each document (fixed order), one
`all_to_all_single` returning the incoming state `S_in` of every crossing-in
fragment (`c2`).  Per backward: the adjoint `d_f` to the owners (`b1`), the
transposed chains, and the boundary adjoints back (`b2`).  The convolution
halo exchanges at most `conv_kernel_size - 1` history rows per crossing-in
fragment forward and their gradient rows backward.  All collectives run on
the caller's NCCL group, asynchronously, waited at first consumption.

### Precision policy

The FP32 state is carried in tensor memory across a fragment; the per-chunk
tapes are BF16 at the rounding points of the reference
`chunk_gated_delta_rule`: the block inverse `A` (FP32, stored BF16 as `Akk`),
the WY operands `bf16(A)`, `bf16(beta * e^G * k)`, `bf16(beta * v)`, `w` and
`u`, the `h` tape (the same BF16 snapshot is the operand of `W * h`), the
`v_new` tape, the K-side operand `bf16(rowscale * v)`, the output, the `dh`
tape (the operand of `K * dh`) and the `dv_new` tape.  Every product of an
FP32 state with a transfer `M` -- summary composition, cross-rank merge,
adjoint transport -- is FP32 FFMA or TF32 hi/lo with FP32 accumulation (the
transfer update is accumulated from zero and added onto the state with one
FP32 round-to-nearest); `g` and `beta` are FP32 until the operand tile is
formed; `dq` / `dk` reduce their value heads in FP32 in a fixed order and
round once; `dg` / `dbeta` are FP32 throughout.

Differences from the reference, recorded for the precision budget: `Aqk` is a
BF16 tape with `scale` inside the rounding (the reference rounds `A` with
`scale` outside) and the backward consumes that tape; `do_g =
bf16(scale * e^G * do)` is one extra BF16 rounding on the `dh` q-term and
the `dq` h-term; the backward gate `rowscale * dv` is applied in FP32 to the
accumulator (one fewer rounding than the reference's BF16 operand); the `dh`
accumulation order is the template's.  The precision gates compare every
tensor against an FP64 reference within 1.10 x the larger of the CP = 1 error
and the reference's own error.

## Kernel structure of one training step

Nine new programs and one reused program compose a step (the affine merge is
the KDA KCP package's gate-free program, rendered under this package's prefix
with the `_kcp` suffix; its 20-hex digests equal that package's):

| # | program | role |
|---|---|---|
| 1 | `fwd_prep` | per chunk: cumulative gate, `gamma`, the WY representation (`w`, `u`), `Aqk`, `Akk` |
| 2 | `kcp_fwdh` (`offset` / `tape` forms, each at value width `vh` 128 or 64) | the forward state walk over the rank's segments: from zero for the summary offset `B_f`, from `S_in` for the `h` / `v_new` tapes; the `vh` 64 form (two CTAs per segment and head, the `_vh64` records) runs when `n_walk × HV ≤ SM count` or when its two-CTA grid finishes within the longest walked segment (LPT over two CTAs per SM) and writes bitwise-identical results |
| 3 | `kcp_transfer_walk` | the transfer `M_f` of a segment, TF32 hi/lo on the tensor cores (`precision_mode` 0) on eight compute warps (a column half each); a CTA whose first 32 chunk decays multiply to at most 2^-64 stops once its carried transition is exactly zero (every later step keeps it zero; bitwise the full walk); its `precision_mode` 1 / 2 forms are four-warp test-only programs of `build.test.json` with bitwise the same results |
| 4 | `kcp_affine_merge` (`c8` / `c16`) | FP32 affine chains over state (`S = M S + B`) and map (compose) tiles; transposed for the adjoint |
| 5 | `fwd_o` | the output from the `h` tape, `v_new` and `Aqk`; persistent, the tiles beyond the grid drawn in units from a ticket word |
| 2+5 | `kcp_fwdh_out` (value width `vh` 128 or 64) | the output walk: #2's tape walk from `S_in` that writes each chunk's output from its `h` snapshot, its `v_new` operand, `q`, `Aqk` and `qscale` instead of the `h` / `v_new` tapes; the forward of a light, saved-boundary call that launches #5 runs it in place of the tape walk and #5 (see "Contract"); the `vh` 64 form (two CTAs per segment and head, the `_vh64` record) runs under #2's rule (`n_walk × HV ≤ SM count`, or its two-CTA grid finishes within the longest walked segment); bitwise the split form's output |
| 6 | `bwd_prep` | `do_g`, `dv_intra`; persistent, the items beyond the grid drawn from a ticket word |
| 7 | `kcp_dhu` (`offset` / `tape` forms, each at value width `vh` 128 or 64) | the adjoint state walk: the boundary adjoint from `dS_out`, the `dh` / `dv_new` tapes; the `vh` 64 form (two CTAs per segment and head, the `_vh64` records) writes bitwise-identical results and runs for the offset walk always, for the tape walk on `sm_100a` / `sm_103a` always and on `sm_107a` when `2 × n_walk × HV ≤ 4 × SM count`; on `sm_100a`, a walk whose longest segment has at least 400 chunks and whose two-CTA grid ends later than the one-CTA grid (LPT over two CTAs per SM) takes the `sm_107a` rule for both forms |
| 8 | `bwd_dqkg` | `dq`, `dk` (grouped FP32 reduction), `dv`, the `dg` / `dbeta` partials; persistent, the (chunk, Q/K head) groups drawn from a ticket word |
| 9 | `gate_epilogue` | the reverse cumulative sum of the `dg` partials and the `dbeta` finalize, FP32 fixed order |

Tensor cores are used through `tcgen05` only; the FP32 chains and the gate
epilogue run on CUDA cores.  Eighteen shipped libraries result (the two walk
forms of #2 and of #7 and the output walk, each at two value widths, the two
merge widths), each for `sm_100a`, `sm_103a` and `sm_107a`.

The persistent programs (#5, #6, #8) launch no cooperative grid: a CTA that
finishes its static tile draws the remaining work through an `atom.inc` on a
32-bit ticket word (the `ticket` argument), which every launch leaves at zero
again.  The host keeps one such word per launch site in the caller's
workspace, 128-byte aligned, zeroed once when the workspace is first sized
(an eager call: after its first launch, or before a persistent first program)
and never read back; a kernel resident on a side stream therefore delays one CTA,
not the whole grid.

## Layout of this package

| file | content |
|---|---|
| `build.json` | the build manifest (schema `cake-build/1`): one entry per `<program>_<arch>` library with its sources, digests and nvcc flags; the only record of what the package compiles |
| `build.test.json` | the test-only programs in the same form; never read by the shipped route |
| `cake_build.py` | the manifest loader (byte-identical across the Cake packages): prebuilt `aot/` library first, FlashInfer JIT otherwise |
| `cake_jit.py` | the registry: the records of every program (library stem, architectures, selection and grid rules, argument plan) |
| `cake_backend.py` | device -> architecture map, library loading, the generic launch of one record |
| `cake_kcp_plan.py`, `cake_tables.py`, `cake_bytes.py` | partition, context, ownership and halo plans, the CPU tables, the byte queries |
| `cake_kcp.py` | the launch sequences, the exchange schedule, the saved set and the autograd Function |
| `cake_l2n.py` | the pure-torch FP32 reference of the in-kernel Q / K L2 norm and its pullback (the fake backend's, the host reduction's and the tests' oracle) |
| `csrc/` | the generated device sources and tvm-ffi bindings (`<library>_kernel.cu`, `<library>_binding.cu`) and the binding checks header |
| `aot/` | prebuilt libraries and their `build_info.json` sidecar, when a builder fills it (git-ignored) |

## Known limits

* NVLink domain only: the ranks of the group must share one NVLink domain
  (one node, or NVLink-connected nodes).
* `exchange="ring"` needs `n_d_max - 1 <= 2` and `save_transfer=False`.
* `HV % H_qk == 0`; head dimension 128; chunk length 64.
* Plan caps per rank: at most `MAX_FRAGS = 65,536` fragments, `MAX_SEGMENTS = 64`
  segments per fragment and `MAX_CHAIN_STEPS = 256` chunk steps per walk
  chain under the per-rank programs (the shipped row policy may walk a whole
  fragment as one chain when its cost model prefers that); a partition beyond
  a cap is refused at context build.
* The output walk serves the light, saved-boundary forward only: under
  `save_mode="full"` or `boundary_mode="replay"` and in a forward that hands
  its tapes to the kept-tape backward the output program runs after the tape
  walk.  Under the default `recompute_output="auto"` a hand-off forward that is
  the checkpoint recompute closing its region launches no output program (with
  the default `reuse_recompute_tapes="auto"` every hand-off forward is one), and
  the backward's replay of the forward runs no output program.
* The causal-convolution halo backward is the conv package's program; its
  launch form is that package's.

## Status

Scaffolding.  Kernels, measured figures and the per-architecture validation
tables are added with the kernels.
