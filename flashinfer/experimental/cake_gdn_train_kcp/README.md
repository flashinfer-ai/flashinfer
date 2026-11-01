# Cake packed GDN context-parallel (KCP) training backend, non-deterministic (SM100 / SM103 / SM107)

Experimental backend for context-parallel training of packed GDN (scalar-gated
DeltaNet) layers with native grouped Q/K heads: forward, backward and
checkpoint replay over a process group whose ranks each hold fragments of the
documents of one packed batch, with precise (FP32 / TF32x3-class) cross-rank
passing of the per-document recurrent state.  Tokens never move between ranks;
only affine summaries of the state do.  The deterministic twin of this package is
`flashinfer.experimental.cake_gdn_det_train_kcp`: the same contract and the
same accuracy budgets, bitwise reproducible.  This package's results stay
within the deterministic route's accuracy budgets and are not bitwise equal
to its results; today they are run-to-run reproducible for a fixed plan and
partition, and later versions may use unordered reductions
(`deterministic=False` is the opt-in).

Both the API and the backend are experimental: no compatibility guarantee.
SM100 (B200 / GB200) and SM103 (B300 / GB300) are the acceptance architectures
of the kernels; SM107 (R200) is compiled from the same sources with an nvcc
that emits `compute_107a`.  The feature does not participate in automatic
backend selection, autotuning or trace-apply.  Its libraries are loaded
prebuilt from `aot/` when the package ships them and JIT-built by FlashInfer
from `build.json` otherwise (see "Layout of this package").  The package
imports nothing else from FlashInfer (neither the KDA training package nor its
deterministic twin: it ships its own copies of the programs it shares with
the twin) and reads no environment variable.

The numbers of this file (shapes, bytes, limits) are the contract.

## Public entry points (`flashinfer/cake_gdn_train_kcp.py`)

```python
from flashinfer.cake_gdn_train_kcp import (
    partition_kcp, build_context_kcp, PartitionKcp, ContextKcp, OptionsKcp,
    RecomputeHandleKcp, chunk_gdn_kcp, chunk_gdn_fwd_kcp, chunk_gdn_bwd_kcp,
    gdn_workspace_bytes_kcp, gdn_saved_bytes_kcp, gdn_comm_bytes_kcp,
    is_gdn_train_kcp_supported,
)

part = partition_kcp(cu_seqlens_global_cpu, world_size, layout="zigzag_per_document")   # once per batch layout
ctx = build_context_kcp(part, group, rank, device, conv_kernel_size=4)                 # tables and collectives of this rank

out = chunk_gdn_kcp(q, k, v, g, beta, ctx)                                             # autograd
out, saved = chunk_gdn_fwd_kcp(q, k, v, g, beta, ctx, save_for_backward=True)
dq, dk, dv, dbeta, dg = chunk_gdn_bwd_kcp(q, k, v, g, beta, ctx, dout, saved=saved)
dq, dk, dv, dbeta, dg = chunk_gdn_bwd_kcp(q, k, v, g, beta, ctx, dout)                 # saved=None: checkpoint replay

ws = gdn_workspace_bytes_kcp(ctx, H_qk=16, HV=48)                                      # per rank, pure CPU
```

These thirteen names are the whole public surface (the `__all__` of
`flashinfer/cake_gdn_train_kcp.py`; this package's `__init__.py` exports
them beside its backend controls).  Every switch is a keyword argument of the
public call (`OptionsKcp` fields); no module-level global and no environment
variable selects anything.

### Names

Every name of this package is the deterministic twin's name without `det`:

| deterministic (`cake_gdn_det_train_kcp`) | this package (`cake_gdn_train_kcp`) |
|---|---|
| `flashinfer.cake_gdn_det_train_kcp` | `flashinfer.cake_gdn_train_kcp` |
| `flashinfer.experimental.cake_gdn_det_train_kcp` | `flashinfer.experimental.cake_gdn_train_kcp` |
| `partition_det_kcp` | `partition_kcp` |
| `build_context_det_kcp` | `build_context_kcp` |
| `chunk_gdn_det_kcp` | `chunk_gdn_kcp` |
| `chunk_gdn_det_fwd_kcp` | `chunk_gdn_fwd_kcp` |
| `chunk_gdn_det_bwd_kcp` | `chunk_gdn_bwd_kcp` |
| `gdn_det_workspace_bytes_kcp` | `gdn_workspace_bytes_kcp` |
| `gdn_det_saved_bytes_kcp` | `gdn_saved_bytes_kcp` |
| `gdn_det_comm_bytes_kcp` | `gdn_comm_bytes_kcp` |
| `is_gdn_det_train_kcp_supported` | `is_gdn_train_kcp_supported` |
| `OptionsDetKcp`, `ContextDetKcp`, `PartitionDetKcp`, `RecomputeHandleDetKcp`, `SavedDetKcp`, `WorkspaceDetKcp`, `CallDetKcp`, `FragmentDetKcp` | `OptionsKcp`, `ContextKcp`, `PartitionKcp`, `RecomputeHandleKcp`, `SavedKcp`, `WorkspaceKcp`, `CallKcp`, `FragmentKcp` |
| `ChunkGDNDetFunctionKcp` | `ChunkGDNFunctionKcp` |

The legacy names of the deterministic package keep meaning the deterministic
objects: `flashinfer.gdn_train.chunk_gdn_kcp`, `KcpContext`, ... and the
legacy package path `flashinfer.experimental.cake_gdn_train` (with its
`chunk_gdn_kcp`) are aliases of `cake_gdn_det_train_kcp` objects, never of
this package's.  The name `chunk_gdn_kcp` of this package is a different
object from the legacy alias of the same spelling.

The generated libraries are named `cake_gdn_train_<20 hex>` for the
per-chunk programs and `cake_gdn_train_<20 hex>_kcp` for the
context-parallel-only programs (`kcp_*`); the kernel symbol of a library is
`kernel_<library>`.  No library name equals one of the deterministic
package's libraries, past or present (`cake_gdn_det_train_<20 hex>[_kcp]`,
and the `cake_gdn_train_<20 hex>` libraries the deterministic package shipped
under its pre-rename name, whose context-parallel programs carried no `_kcp`).

### Numerical mode

The packed GDN training routes come in four combinations:

| | single rank (local) | context-parallel (KCP) |
|---|---|---|
| deterministic | `flashinfer.experimental.cake_gdn_det_train` | `flashinfer.cake_gdn_det_train_kcp` -> `flashinfer.experimental.cake_gdn_det_train_kcp` |
| non-deterministic | `flashinfer.experimental.cake.cake_gdn_train` | `flashinfer.cake_gdn_train_kcp` with `deterministic=False` -> this package |

* The facade `flashinfer.cake_gdn_train_kcp` takes `deterministic: bool = True`
  on every entry: `True` runs the deterministic package's objects, `False`
  runs this package.  With `torch.use_deterministic_algorithms(True)`,
  `deterministic=False` runs the deterministic package and warns once per
  process.
* This package's compute entries (`chunk_gdn_kcp`, `chunk_gdn_fwd_kcp`,
  `chunk_gdn_bwd_kcp`) take `deterministic: bool = True` as well and raise
  `ValueError` unless it is `False`: the package holds no deterministic
  program and never falls back.
* Faster schedule: the default segment plan of this package (`fill_small`)
  walks fewer intra-rank segments than the deterministic package's plan where
  segment merges cost more than they save: at `HV <= 12`, when no fragment
  exceeds 160 chunks, every rank walks each fragment as one chain (`P = 1`: no
  segment merges, compose chain or staging planes); elsewhere a rank below the
  256-step chain cap takes the one-wave fill `P` when that is smaller.  It
  never raises a rank's `P`, needs no host synchronization, and the plan stays
  a pure function of the fragment tables, `HV` and the SM count (the same `P`
  on every rank).  A context built with `graph_replay=True` (for calls with
  `graph="replay"`) keeps the one-wave fill but not the `P = 1` rows
  (`fill_down`): graph replay removes the launch cost that the fewer merges
  save, and the longer walks then cost more.
* Reproducibility: the results are not bitwise equal to the deterministic
  package's (the different segment plan and the fixed-order levers of this
  package reassociate FP32 sums).  Every path of
  this version is fixed-order, so for a fixed plan and partition the results
  are run-to-run reproducible today; later versions may use unordered
  reductions.  Only a path that launches an unordered reduction raises
  PyTorch's alert for an operation without a deterministic implementation
  under `torch.use_deterministic_algorithms(True)`; this version has none,
  so no entry of this package raises it.
* The mode is not stored in the context, the partition or the options: both
  routes build the same plan objects.  The autograd node records its
  forward's mode and its backward refuses another; `chunk_gdn_bwd_kcp`
  refuses a saved set that is not this package's (`SavedKcp`), and the
  facade's deterministic backward refuses this package's saved set.
* Accuracy: every output stays within the deterministic route's budgets
  against the FP64 reference (`rel-RMS <= 1e-2`, `max-abs <= 2e-2` of
  `max|ref|`; the per-tensor KCP budgets of the deterministic tests).  The
  precision classes of the summaries, merges and adjoints are those of the
  deterministic route (FP32 / TF32x3; never plain TF32).

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
  measured module when reported.  Strided and contiguous inputs give the
  same results within the accuracy budgets.
* `scale` defaults to `128 ** -0.5`.
* The host reads no tensor contents (no `.item()`, `.tolist()`, no implicit
  synchronization): every table is built on the CPU from the partition
  metadata, uploaded once per `(HV, SM count, P)` key at first use
  (`ctx.prepare(H_qk, HV)`, also called by the first forward) and kept
  device-resident on `ctx`; the launch grids follow from `(T_r, H_qk, HV, SM
  count, architecture)` alone.  Tables are int32 except the affine merge's
  element offsets, which are int64.  Steady state is CUDA-graph capturable.
* Repeated calls on the same inputs, the backward from the saved set, the
  checkpoint replay and the `torch.utils.checkpoint` direct and indirect
  backward agree within the accuracy budgets (bitwise today, every path
  being fixed-order; not guaranteed).
  Gradient accumulation (`.backward()` twice) accumulates.
* No cooperative launch anywhere in the package; every program is a plain
  stream-ordered launch.
* Host launch path: the record, grid and merge columns of a launch are
  resolved once per launch key and kept with the context's tables; each
  forward or backward body, eager or captured, binds the caller's current
  stream for tvm-ffi once (`tvm_ffi.use_torch_stream()`), and a launch enters
  its own context only outside such a body or after a stream switch inside it.

### Partition and context

`partition_kcp(cu_seqlens_global_cpu, world_size, layout, fragments=None)`
splits the packed batch over `world_size` ranks: `"contiguous"` (equal
contiguous slices), `"zigzag"` (the whole buffer in `2 * world_size` pieces,
rank `r` holding pieces `r` and `2W - 1 - r`), `"zigzag_per_document"` (the
zigzag applied inside every document) or explicit `fragments`.  Cut points
need not be multiples of the chunk length (64); ranks may hold unequal token
counts; a fragment may be empty, one token long or cross a chunk boundary; a
document may span any number of ranks including the whole group.

`build_context_kcp(part, group, rank, device, conv_kernel_size=4)` returns
the rank's `ContextKcp`: the fragment table, the state-ownership plan (head
groups over `HV`, padded when `HV` does not divide by the group size), the
causal-convolution halo plan (`conv_kernel_size - 1` history rows per
crossing-in fragment) and the int32 device tables of the kernels.  The same
context serves every call of the same batch layout and head pair.
`build_context_kcp(..., graph_replay=True)` selects the segment plan for
calls with `graph="replay"` (see "Numerical mode"); such a context serves
eager calls too.

### Options (`OptionsKcp`, keyword-only; defaults in bold)

| option | values | meaning |
|---|---|---|
| `save_mode` | **`"light"`**, `"full"` | `"light"` keeps the boundary states and the cheap per-chunk tables and replays the intra-rank tapes in the backward; `"full"` keeps the tapes (`h`, `v_new`, `w`, `u`, `Aqk`: 67,584 B per token at `HV = 48`) |
| `boundary_mode` | **`"saved"`**, `"replay"` | whether the cross-rank boundary states (`S_in`, the owner's received summaries) are kept in the saved set or recomputed (exchange included) in the backward |
| `exchange` | **`"a2a"`**, `"ring"` | the collective schedule; `"ring"` only when every document crosses at most 3 ranks (`n_d_max - 1 <= 2`) and `save_transfer=False` |
| `save_transfer` | **`True`**, `False` | keep the received summary pairs for the backward (`False` re-sends the transposed transfers) |
| `precision_mode` | **`0`** (TF32 hi/lo), `1`, `2` (test-only) | the state-transfer products of the transfer walk: `0` is the shipped two-MMA hi/lo form (the TF32x3 class); `1` is its three-MMA twin (bitwise the same results) and `2` plain TF32 (the negative control), both test-only programs of `build.test.json` |
| `transfer_walk` | **`"tf32_hilo"`** | the transfer-walk program; the vocabulary also names `"bf16x3"` (BF16x3 state products), which this package does not build: the validator refuses it as not built |
| `layout` | **`"tma"`**, `"copy"` | strided input read in place, or a timed contiguous copy |
| `graph` | **`"off"`**, `"replay"` | CUDA-graph replay of the steady state (a static pool per key) |
| `reuse_recompute_tapes` | **`"auto"`**, `True`, `False` | under `torch.utils.checkpoint(use_reentrant=False)` the light backward binds the forward tapes torch's recompute just wrote instead of replaying #1 and the #2 tape walk (bitwise the same gradients, the retention unchanged; a no-op outside that backward; `False` restores the replay form). `"auto"` keeps the tapes only when the GDN node is the last operation of the checkpointed region that saves a tensor (the same checkpoint-frame proof as `recompute_output="auto"`): then its backward is the region's first and the kept tapes die before any other backward runs; with a later saver in the region (an output projection, an MLP) the kept set would live through those operations' backward and raise the region's backward peak by the full tape set (`gdn_saved_bytes_kcp(save_mode="full")` minus the saved states), so `"auto"` takes the replay form there. Measured on one GB300 (torch 2.14.1), 261,987 tokens, `(H_qk, HV) = (2, 6)`: a region of GDN + a 768x768 output projection + a 768->2x2048->768 SwiGLU has a backward peak of 9250.54 MiB with the kept tapes (`True`) and 6927.31 MiB with the replay form (`False` and `"auto"`), at 21.29 vs 22.70 ms per training step of the region; a region the GDN node closes keeps the tapes under `"auto"` (five 240k-262k-token (16, 48) packs: training step and steady peak unchanged against `True`). `True` always keeps the tapes, `False` never. With `graph="replay"`, the one mode that excludes the kept tapes, the default is `False` and an explicit `True` / `"auto"` is refused. One live kept set: in a checkpoint segment with several GDN layers only the last layer skips the replay, the earlier layers replay as before |
| `recompute_output` | **`"auto"`**, `"always"` | under `torch.utils.checkpoint(use_reentrant=False)` torch re-runs the forward inside `.backward()` only to re-save its tensors and discards that run's output; when the GDN node is the last operation of the checkpointed region that saves a tensor, `"auto"` does not launch the output program (#5) of that re-run. The proof is read from torch's checkpoint frame (the node is executable, the remaining saved-tensor holders are exactly this node's, early stop is on, and the torch release -- `torch.__version__` without its local `+cuNNN` build tag (`cake_kcp.torch_release`) -- is one of the verified releases in `cake_kcp.RECOMPUTE_OUTPUT_TORCH_VERSIONS`); any other state -- a later saver in the region such as the next layer or a norm on the output, a non-executable node, a reentrant checkpoint, `graph="replay"`, another torch build -- launches it as before, and `cake_kcp.LAST_CALL["fwd"]["fwd_o_reason"]` records why. Gradients and retained bytes are the same either way; `"always"` is the unconditional launch |
| `p_rule`, `walk_seg` | plan policy / explicit override | how many intra-rank segments a fragment is walked as (`P`); a function of metadata and the SM count only. The shipped policy `fill_small` starts from the deterministic twin's `row_cost_lpt` plan, which costs the three per-rank programs over every rank's fragment table with a crossing-aware model and switches the whole row away from the per-rank default only when an alternative is cheaper by at least its margin (2 % for the `occ` program; 0 % for the `one` program, which has no segment composition and takes a model tie), then lowers `P` as described under "Numerical mode" (faster schedule); the same topology gives the same `P` on every rank |
| `offset_class` | **`"fla"`**, `"fp32"` | the class of the forward offset `B_f`: the walk from zero with the reference rounding points, or an FP32 walk (`B_f` only) |
| `test_knobs` | **`None`** | names the test-only values a test selects (`precision_mode` 1 or 2) and private levers such as `fwdh_vh` (forces the #2 value width); without it every test-only value is refused |

The validator refuses every test-only value not named in `test_knobs`,
`exchange="ring"` without both of its conditions, `HV % H_qk != 0`, a
workspace smaller than `gdn_workspace_bytes_kcp`, and any option value
outside these tables, before any launch.  The public API cannot select a
test-only program: the test-only libraries live in `build.test.json`, which
the shipped route never reads.

### Saved set, replay and byte queries

With `s = HV * 128 * 128 * 4` bytes (3 MiB at `HV = 48`), `n_in(r)` the
rank's fragments with a predecessor on another rank and `n_out(r)` those with
a successor:

| item | `save_mode="full"` | `save_mode="light"` | bytes per rank |
|---|---|---|---|
| inputs `q, k, v, g, beta` (references) | yes | yes | 0 extra |
| `S_in` (incoming boundary states) FP32 | yes | yes | `n_in * s` |
| received summary pairs (owner, `save_transfer=True`) | yes | yes | `2 * s_owner * n_owned_fragments` |
| `M_seg` (segment transfers, `P > 1`) | yes | yes | `(P - 1) * n_frag * s` |
| convolution history rows | yes | yes | `<= 3 * n_in * C * 2` |
| `gamma`, `gcum`, `Akk` | yes | yes | about 6,339 B per token |
| `h` tape, `v_new`, `w`, `u`, `Aqk` | yes | **no** (replayed in the backward) | 67,584 B per token |

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
  output -- and the backward then proceeds as in the saved form (with the
  light replay on top in `"light"`).  `boundary_mode="replay"` is the
  minimum-memory option.

Both replays give bitwise the gradients of the saved-set backward.

A **recompute handle** removes the exchange from the checkpoint replay.  Pass
a `RecomputeHandleKcp` to the checkpointed function and to `chunk_gdn_kcp`:

```python
def layer(x, handle):
    ...                                   # convolution, split, gates
    return chunk_gdn_kcp(q, k, v, g, beta, ctx, recompute_handle=handle)

y = torch.utils.checkpoint.checkpoint(layer, x, RecomputeHandleKcp(), use_reentrant=False)
```

The original forward exchanges as always and stores this rank's incoming
states `S_in` (the saved set's own tensor) in the handle.  Torch's recompute
inside `.backward()` reads them back and runs the light replay's launches
(chunk preparation, at `P > 1` the local segment walks and carry merges, the
tape walk from `S_in`, then the output program): no collective, no summary
walk over the crossing-out fragments, no owner merge.  The gradients
are bitwise those of the same checkpoint without a handle; the backward still
exchanges.  The handle retains `S_in` --
`gdn_saved_bytes_kcp(ctx, ..., save_mode="light", save_transfer=False,
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

```
gdn_comm_bytes_kcp(ctx, *, H_qk, HV, C) -> {rank: {c1_send, c1_recv, c2_recv, b1_send, b2_recv, halo_fwd, halo_bwd}}
    c1_send = 2 * s * n_out(r)      c2_recv = s * n_in(r)      b2_recv = s * n_out(r)
    b1_send = s * n_in(r) * (1 if save_transfer else 2)
    c1_recv / b1_recv on an owner = 2 * s_owner * (fragments it owns)      halo = rows(r) * C * 2
gdn_saved_bytes_kcp(ctx, *, H_qk, HV, save_mode, save_transfer=True, P=None) -> per rank (the table above)
gdn_workspace_bytes_kcp(ctx, *, H_qk, HV, P=None, reuse_recompute_tapes=None) -> per rank: the most the caller-held
    workspace binds in the forward (the tape walk: pools + the eight live tapes) or the backward (its phases: the
    exchange, the adjoint tape walk, the end state); the default (`None`, i.e. `"auto"`) reports the replay form, the
    bound of both forms, in which the ten forward tapes are workspace; `reuse_recompute_tapes=True` gives the kept-tape
    figure, in which they are the recompute forward's hand-off tensors outside the workspace
    (`gdn_workspace_layout_kcp` lists them as `bwd.kept_tapes` and the total as `backward_transient`)
```

`H_qk` enters the halo and strided-view terms; the exchange terms depend on
`HV` alone.

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

Eight new programs and one reused program compose a step (the affine merge is
the KDA KCP package's gate-free program, rendered under this package's prefix
with the `_kcp` suffix; its 20-hex digests equal that package's):

| # | program | role |
|---|---|---|
| 1 | `fwd_prep` | per chunk: cumulative gate, `gamma`, the WY representation (`w`, `u`), `Aqk`, `Akk` |
| 2 | `kcp_fwdh` (`offset` / `tape` forms, each at value width `vh` 128 or 64) | the forward state walk over the rank's segments: from zero for the summary offset `B_f`, from `S_in` for the `h` / `v_new` tapes; the `vh` 64 form (two CTAs per segment and head, the `_vh64` records) runs when `n_walk × HV ≤ SM count` and writes bitwise-identical results |
| 3 | `kcp_transfer_walk` | the transfer `M_f` of a segment, TF32 hi/lo on the tensor cores (`precision_mode` 0) on eight compute warps (a column half each); its `precision_mode` 1 / 2 forms are four-warp test-only programs of `build.test.json` with bitwise the same results |
| 4 | `kcp_affine_merge` (`c8` / `c16`) | FP32 affine chains over state (`S = M S + B`) and map (compose) tiles; transposed for the adjoint |
| 5 | `fwd_o` | the output from the `h` tape, `v_new` and `Aqk`; persistent, the tiles beyond the grid drawn in units from a ticket word |
| 6 | `bwd_prep` | `do_g`, `dv_intra`; persistent, the items beyond the grid drawn from a ticket word |
| 7 | `kcp_dhu` (`offset` / `tape` forms, each at value width `vh` 128 or 64) | the adjoint state walk: the boundary adjoint from `dS_out`, the `dh` / `dv_new` tapes; the `vh` 64 form (two CTAs per segment and head, the `_vh64` records) writes bitwise-identical results |
| 8 | `bwd_dqkg` | `dq`, `dk` (grouped FP32 reduction), `dv`, the `dg` / `dbeta` partials; persistent, the (chunk, Q/K head) groups drawn from a ticket word |
| 9 | `gate_epilogue` | the reverse cumulative sum of the `dg` partials and the `dbeta` finalize, FP32 fixed order |

Tensor cores are used through `tcgen05` only; the FP32 chains and the gate
epilogue run on CUDA cores.  Sixteen shipped libraries result (the two walk
forms of #2 and of #7, each at two value widths, the two merge widths), each
for `sm_100a`, `sm_103a` and `sm_107a`.

The persistent programs (#5, #6, #8) launch no cooperative grid: a CTA that
finishes its static tile draws the remaining work through an `atom.inc` on a
32-bit ticket word (the `ticket` argument), which every launch leaves at zero
again.  The host keeps one such word per launch site in the caller's
workspace, 128-byte aligned, zeroed once when the workspace is allocated and
never read back; a kernel resident on a side stream therefore delays one CTA,
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
* The fused tape-walk-plus-output form is not part of this release: the
  output program runs after the tape walk.
* The causal-convolution halo backward is the conv package's program; its
  launch form is that package's.

## Status

Experimental.  The kernels are the deterministic route's programs, rendered
from the same Cake sources under this package's names; `build.json`'s
`generator` block names the producing revision.  This package differs from
its deterministic twin in the default segment plan (`fill_small`, see
"Numerical mode").  The tests are `tests/experimental/test_cake_gdn_train_kcp*.py`
(API, single GPU and multi-GPU rows).
