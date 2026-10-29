# Cake packed KDA chunked training backend (SM100 / SM103 / SM107)

Experimental backend for the packed Kimi Delta Attention (KDA) chunked
training forward and backward in the GLM-5.3-Flash geometry: head dimension
128 for `q`, `k`, `v` and the gate, chunk length 64, per-document zero initial
state, `H` heads per call (1 to 64), one packed batch of documents described by
`cu_seqlens`.  Tracking issue: flashinfer-ai/flashinfer#6485.

Both the API and the backend are experimental: no compatibility guarantee.
SM100 (B200 / GB200) and SM103 (B300 / GB300) are the acceptance architectures
of the kernels; SM107 (Rubin R200) is compiled from the same sources with an
nvcc that emits `compute_107a`.  The feature does not participate in automatic
backend selection, autotuning or trace-apply.  Its libraries are loaded
prebuilt from `aot/` when the package ships them and JIT-built by FlashInfer
from `build.json` otherwise (see "Layout of this package").

## Public entry points (`flashinfer/kda_train.py`)

```python
from flashinfer.kda_train import chunk_kda, chunk_kda_fwd, chunk_kda_bwd, kda_train_workspace, kda_train_arena

out = chunk_kda(q, k, v, g, beta, cu_seqlens, scale=None, workspace=None, save_mode=None, arena=None)  # autograd
out, saved = chunk_kda_fwd(q, k, v, g, beta, cu_seqlens, scale=None,
                           save_for_backward=True, workspace=None, save_mode=None, arena=None)
dq, dk, dv, dbeta, dg = chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, dout,
                                      saved=saved, scale=None, workspace=None, arena=None)
dq, dk, dv, dbeta, dg = chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, dout)         # checkpoint replay
ws = kda_train_workspace(S, num_docs, H, device, backward=True, save_mode="light")  # caller-owned
ws = kda_train_workspace(S, num_docs, H, device, arena=kda_train_arena(tail_passes=4))  # lower-memory arena
```

### Contract

| operand | shape | dtype | notes |
|---|---|---|---|
| `q`, `k`, `v` | `[1, S, H, 128]` | BF16 | contiguous; `q` / `k` L2-normalized by the caller (eps 1e-6) and stored as BF16 |
| `g` | `[1, S, H, 128]` | FP32 | preactivated natural-log decay, `g = -5 * sigmoid(exp(A_log) * (raw_gate + dt_bias))` |
| `beta` | `[1, S, H]` | BF16 | after the sigmoid |
| `cu_seqlens` | `[num_docs + 1]` | int32 | device prefix sums, first 0, last `S`; zero-length documents allowed |
| `out`, `dout` | `[1, S, H, 128]` | BF16 | laid out like `v` |
| `dq`, `dk`, `dv`, `dbeta` | like the inputs | BF16 | gradients at the inputs' storage boundary (no L2-norm backward) |
| `dg` | like `g` | FP32 | |

* The leading batch axis of 1 is optional on every operand; the outputs take
  the rank of the inputs.  Every operand must be contiguous: a strided view is
  refused with `ValueError`, never copied.
* `scale` defaults to `128 ** -0.5`.  Each document starts from a zero state;
  no final state is produced.
* The host reads no tensor contents (no `.item()`, `.tolist()` or implicit
  synchronization): the chunk tables are built on the device from `cu_seqlens`,
  the chunk-shaped kernels launch at the host bound `NC_pad` chunks (the empty
  pad chunks are no-ops) and the persistent kernels read the real chunk count
  from the device table.  Which variant of a stage runs and how large its grid
  is follow from `S`, `num_docs`, `H`, the device's SM count and its
  architecture alone.
* Repeated calls on the same inputs give bitwise the same outputs: the kernels
  reduce in a fixed order and use no atomic adds.  The backward from the saved
  set (with or without the saved segment states) and the checkpoint replay
  (`saved=None`) give bitwise the same gradients.
* Limits: `H * 128 * 64 < 2**31` (`H <= 262143`; the public contract covers
  `H` in 1..64); `R * H * 64 < 2**31` over the padded rows `R = 64 * NC_pad` of
  the intra-chunk buffers (at `H = 64` that is `S` up to about 524k tokens; the
  contract's shapes use at most half of that range); `num_docs <= 65536` (the
  layout table's 16-step document search); `S >= 1`, `num_docs >= 1`.  Any `S`,
  including `S < 64` and document lengths that are not multiples of 64.  The
  host refuses a problem outside these limits before any launch.

### Saved set (`KdaChunkSaved`)

`chunk_kda_fwd(..., save_for_backward=True)` returns a frozen dataclass:

| field | content |
|---|---|
| `version` | `4` (`save_mode="light"`, the default) or `2` (`save_mode="tape"`); versions 1 and 3 (the light sets before the fold's `qg` was saved) are no longer produced or accepted |
| `Aqk`, `Akk` | BF16 `[S, H, 64]` on the token rows: row `t` holds the 64 chunk-local columns of its chunk (zeros above the diagonal and past the chunk end) |
| `tables` | `(chunk_bos, chunk_len, seq_chunk_start, seq_order[, seg_table])`: int32 views of one device buffer (`NC_pad`, `NC_pad`, `num_docs + 1`, `num_docs` and, on the segmented route, the block table's `NSEG + 2 * num_docs + 1` entries), or `None` (the backward rebuilds them from `cu_seqlens`) |
| `S`, `H`, `num_docs`, `scale` | the problem the set belongs to |
| `h`, `v_new` | version 2 only: the forward's state tape BF16 `[NC_pad, H, 128, 128]` and `v_new` BF16 `[R, H, 128]` on the internal layout |
| `seg_m` | version 2 on the segmented route: the segment transfers FP32 `[NSEG, H, 128, 128]` the backward's reverse carry reads; `None` on the serial route |
| `hseg` | version 4 on the segmented route when `S >= 384 * NSEG`: the carried initial state of every segment, FP32 `[NSEG, H, 128, 128]` (the backward's recurrence recompute walks from it: pass 2 alone); `None` otherwise |
| `seg_state` | version 4, idem: FP32 `[2, NSEG, H, 128, 128]`, `[0]` the segment offsets, `[1]` the segment transfers the backward's reverse carry reads; `None` otherwise |
| `qg` | version 4: the gated queries of the GLM contract on the internal rows, BF16 `[R, H, 128]` (`R = 64 * NC_pad`), written by the saving forward's intra program; the light backward's prep program (`bwd_prep_g8_noqg*`) recomputes them no more |
| `w`, `u` | version 4 on the rows whose light backward reads the WY representation from the set: the paired serial 16-head rows of `sm_100a` (the serial route's 32-column form at 16 heads) and the segmented 16-head rows of `sm_103a` / `sm_107a` above 32 (document, head) chains (three documents and more); the forward's WY representation, BF16 `[R, H, 128]` each, written by the forward's WY program (`wy_fwd_rows` on the paired serial rows and the overlap rows, `wy_fwd` on the other segmented rows); the light backward launches no `wy` program there on either light path and its prep writes no `k * beta` (`bwd_prep_g8_noqg_nokb`); `None` elsewhere |

`kda_train_saved_bytes(S, H) = 256 * S * H` bytes (`Aqk` and `Akk`);
`kda_train_saved_bytes(S, H, num_docs, save_mode=...)` adds the tables and
the mode's extra tensors: for `"light"` (version 4) the fold's `qg`, `256 *
R * H` bytes on the internal rows (`R = 64 * NC_pad`), the segment-boundary
states, `3 * NSEG * H * 65536` bytes (`hseg` and the `seg_state` pair, FP32
`[128, 128]` per segment and head) on the segmented route (`H <= 8`, and the
16-head rows of `sm_103a` / `sm_107a`) when `S >= 384 * NSEG`, and `w` / `u`,
`512 * R * H` bytes, on the rows whose light backward reads them from the
set (the paired serial 16-head rows of `sm_100a`; the segmented 16-head rows
of `sm_103a` / `sm_107a` above 32 (document, head) chains); for
`"tape"` the version-2 tensors
(768 B more per internal token-head plus the segment transfers).  The size
rule (`cake_backend.seg_states_saved(S, NSEG)`,
`SEG_SAVE_MIN_TOKENS_PER_SEG = 384`) saves the states exactly when they are
no larger than the `Aqk` / `Akk` set (`3 * NSEG * H * 65536 <= 2 * S * H *
256`).  With `NSEG = num_docs * P` and `P = clamp(ceil(4 * sm_count /
(num_docs * H)), 1, 64)`, at most `sm_count // H` (at most 38 with `num_docs *
H <= 4`; 37 on `sm_100a`), `NSEG * H` is at most `4 * sm_count` (that much for `num_docs * H >=
10` unless the per-SM cap binds, which brings it down to `num_docs *
sm_count`), so the states cost at most about `12 * sm_count * 64 KiB` (115 MiB
on a 152-SM device) independent of `S`, and the threshold is about 58k tokens
at `H = 4` (four documents), 29k at `H = 8`,
`384 * 38 * num_docs` tokens with at most four chains (14592 for one document
at `H = 1`, 58368 for four; `384 * 37 * num_docs` on `sm_100a`: 14208 and
56832) and `384 * 64 * num_docs` tokens where `P` is
capped at 64 (196608 for eight documents at `H = 1`); the contract's
performance shapes (`S` around 240k-260k tokens) all save, a pack of a few
thousand tokens does not.  The byte queries read `sm_count` (the current
device's when `None`) and the architecture (`arch`, else the current
device's; without a device the architecture-independent count of 38 and, for
the version-4 tensors, the first exported architecture's prep records).  Every
tensor of the set is allocated by the call that writes it: a saved set can go
to a backward on any workspace of the same problem.  A caller may keep only
`Aqk` / `Akk` (`tables=None`).  With `save_for_backward=False` the token-row
`Aqk` / `Akk` are not written (the `fwd_nosave` path: the forward's programs
read the internal copies) and the backward must replay (`saved=None`:
the table where the plan launches one, prep and intra recompute them).  From a version-2 set the backward
skips its recurrence recompute (WY still runs for `w`); from a version-4 set
with the states its recurrence recompute is pass 2 alone from `hseg` and the
reverse carry reads `seg_state[1]`; from a version-4 set without them or on
the replay path it recomputes the states, `h` and `v_new` in its workspace
(a set carrying `w` / `u` spares the WY launch too).
The gradients are bitwise the same on every path; the backward writes nothing
into the saved set.

### Workspace (`KdaWorkspace`)

A workspace is one arena (a `uint8` tensor) holding the internal-layout
intermediates of one `(S, num_docs, H)` problem; `chunk_kda_fwd` /
`chunk_kda_bwd` allocate only their outputs and the saved set when given one.
Without one, each call allocates its arena when it starts and frees it when it
returns, so no device memory outlives the call.  The host side of that
workspace (the plan, which `cake_backend.plan` caches per `(S, num_docs, H,
sm_count, arch)`, and the layout views) is kept per device, stream and arena
option, so a repeated problem costs the host about what a caller-owned
workspace costs.  The arena is not
initialized: every element a launch reads is written earlier in the same
call, so the content at allocation never reaches a result (`zero_init=True`
zero-fills it for diagnostics).  The autograd node of `chunk_kda` holds a
caller's workspace only weakly: the backward runs in it while the caller
keeps it, and a workspace dropped after the forward (a per-call one, or any
workspace of a forward under non-reentrant `torch.utils.checkpoint`) is freed
at once, the backward then allocating its own; after a checkpointed forward
the node keeps no arena and no saved tensor alive.  Buffers whose
lifetimes within one call are disjoint share a slot: the lifetime tables
`cake_backend.SLOTS[mode]` (`fwd`, `bwd` for the saved-set paths, `replay`,
and their `*_tape` forms) list every occupant with its `[first write, last
read]` launch step, and `cake_backend.lifetime_table(mode)` renders them; a
backward workspace's arena is the larger of its saved-set and replay layouts.
`kda_train_workspace_bytes(S, num_docs, H, backward=True, save_mode="light",
sm_count=None)` is the allocation (`sm_count` enters through the segmented
route's segment count; the current device's when `None`);
`cake_backend.memory_report(S, num_docs, H, save_mode)` gives the arena, the
saved set, the outputs and the per-boundary totals.  The backward holds no
chunk-layout copies of `q` / `k` / `v`: its kernels read the token rows.

Outputs and the saved set are fresh tensors of every call; nothing is kept in
the arena between calls (the pad rows every consumer reads are written in the
same call by their producer).  `backward=False` sizes the arena for the
forward layout only; `save_mode="tape"` sizes it for the tape layouts (smaller
arenas: `h` and `v_new` live in the saved set) and makes `"tape"` the default
of the calls on it.

### Lower-memory arena (`KdaArena`)

The lower-memory arena option, off by default, shrinks the arena of a
backward workspace; the gradients stay bitwise the same.
`kda_train_arena(lean=False, tail_passes=1, batch_tokens=None,
max_docs_per_batch=None)` in `flashinfer/kda_train.py` returns a `KdaArena`, a
frozen dataclass with those four fields (`tail_passes >= 1`; `tail_passes > 1`
implies `lean=True`; the two batching fields are described below), which
`kda_train_workspace(...)`, `kda_train_workspace_bytes(...)`,
`chunk_kda_fwd`, `chunk_kda_bwd` and `chunk_kda` take as the keyword-only
`arena=`.  `None` takes the workspace's option (off without one); an option
different from a given workspace's is refused with `ValueError`.  A workspace
keeps its option as `arena_option` (its `arena` stays the `uint8` tensor).
The autograd node of `chunk_kda` records the option, so a backward that
allocates its own workspace (after a per-call forward, or a forward under
non-reentrant `torch.utils.checkpoint`) uses the same one.  The forward's
layout is the same under every option and smaller than the lean backward
layouts (2560 B per internal token-head against at least 2820 in the light
form), so the option changes nothing about the forward or the saved set.

```python
arena = kda_train_arena(lean=True, tail_passes=4)                # lean layouts, four tail passes
nbytes = kda_train_workspace_bytes(S, num_docs, H, arena=arena)  # the arena it allocates
ws = kda_train_workspace(S, num_docs, H, device, arena=arena)    # caller-owned, reused
out = chunk_kda(q, k, v, g, beta, cu_seqlens, workspace=ws)      # the default arena's gradients, bitwise
```

`lean=True` sizes a backward workspace for the saved-set paths alone: the
layouts `bwd_lean` / `bwd_tape_lean` (`cake_backend.LEAN_MODES`) take the
place of the larger of the saved-set and replay layouts and drop the replay
prep's internal `dout` copy (`do_int`).  Such a workspace serves no checkpoint
replay: `chunk_kda_bwd` with `saved=None` raises `ValueError` on it.
`torch.utils.checkpoint` is unaffected, since the node's backward always reads
the recomputed saved set, never the replay path.  A lean arena also refuses
the rows whose backward dispatches the separate dA / dv program `dav` (9 to 15
heads) with `ValueError` when a backward workspace is built and in the byte
queries; it serves the contract's head counts 1, 2, 4, 8, 16, 32 and 64.

`tail_passes = B` runs the chunk-parallel tail stages of the backward (the
fused gradient `dqkg`, the GLM intra program and, on `sm_100a`,
the gate epilogue `gate_epilogue`: the last launch steps of the backward's
lifetime tables) once per range of `ceil(NC_pad / B)` chunks.  Each pass takes views of the
whole-sequence buffers (`gk`, `v_new`, `dv2`, `beta_int`, the `h` / `dh`
tapes) offset to the range's first chunk, the range's slices of the chunk
tables and the per-pass buffers (`dq_f`, `dk_f`, `dg_f`, `db_f`, `dAkk`,
`dAqk` and, on `sm_100a`, the gate epilogue's `dg_int`, `db_int`), which are
sized for one range; the token tensors
(`q`, `k`, `v`, `dout`, the saved `Aqk` / `Akk`, the gradients) are passed
whole, every kernel reaching them through the absolute `chunk_bos` offsets.
The same kernels address the same bytes as in one pass, so the gradients are
bitwise those of the default arena; the cost is `B - 1` more launches per tail
stage and the speed lost to their smaller grids.

Sizes are in bytes per internal token-head (one of the `R = 64 * NC_pad`
padded rows times one head), plus, on the segmented route, FP32 state blocks
of `block = NSEG * H * 64 KiB` each.  A default backward workspace takes the
larger of the saved-set layout (4360; tape 3592) and the replay layout (4488;
tape 3720); one lean pass gives
`4108 * R * H + max(0, 3 * block - 256 * R * H)` bytes (tape 3340), the three
state blocks sharing a slot of 256 B per internal token-head as in the
default layouts.  From two passes on the whole-sequence set is 2820 (tape
2052) and the per-pass set, 2568 B per internal token-head of one range, is
packed first-fit into the two whole-sequence slots that are dead before the
tail (`qg` / `kg`, 512, and `w`, 256); the rest goes to an overflow slot
shared with the state blocks.  Two passes give
`2820 * R * H + max(3 * block, 516 * R * H)` (3336; tape 2568), four or more
`2820 * R * H + 3 * block + slack` (tape: 2052 in place of 2820), where
`slack` is the ceil slack of the last range (the per-pass buffers are sized
for `R_b = 64 * ceil(NC_pad / tail_passes)` rows; it is zero when
`tail_passes` divides `NC_pad`); passes above four buy nothing more.  On
`sm_103a` / `sm_107a`, where the GLM intra program hosts the gate epilogue, its inputs
`dg_int` (512 B) and `db_int` (4 B) leave the lean layouts: one
pass gives 4104 (tape 3336), the per-pass set is 2052 and two passes give
`2820 * R * H + max(3 * block, 258 * R * H)` (3078; tape 2310); the default
layouts are unchanged (the two buffers share slots there).  Against the
default's 4488 the light constants are about 8 % (one pass), 26 % (two
passes; 31 % on `sm_103a` / `sm_107a`) and 37 % (four or more) smaller.  The table
gives the constants (`sm_100a`; `sm_103a` / `sm_107a` where they differ);
the state blocks and the slack come on top where the formulas say so:

| arena | light: B per internal token-head | tape: B per internal token-head | checkpoint replay (`saved=None`) |
|---|---|---|---|
| default (`KdaArena()`) | 4488 | 3720 | yes |
| `lean=True`, one pass | 4108; 4104 | 3340; 3336 | no |
| `tail_passes=2` | 3336; 3078 | 2568; 2310 | no |
| `tail_passes >= 4` | 2820 | 2052 | no |

The byte queries take the option too: `kda_train_workspace_bytes(..., arena=)`
and `cake_backend.workspace_bytes(..., arena=)` give the configured arena for
any `tail_passes`, and `cake_backend.memory_report(..., arena=)` adds the
fields `lean`, `tail_passes`, `arena_bwd_default`, `arena_bwd_configured`,
`arena_fwd_configured` and `arena_floor` (`2820 * R * H + 3 * block`; 2052 in
place of 2820 for the tape forms) and recomputes its totals with the
configured backward arena (the replay total keeps the default's).
`cake_backend.lifetime_table(mode, tail_passes, plan=)` renders the lean
layouts: their per-pass rows carry `unit="R_b*H"`, and from two passes on a
`plan` is required, since the packing depends on the problem's geometry.  The
context-parallel (KCP) entry points do not take the option in this version.

#### Document groups (`batch_tokens`)

`kda_train_arena(batch_tokens=B, max_docs_per_batch=M)` (`M` optional and
needs `B`) runs the forward and the backward over groups of whole, contiguous
documents of at most `B` tokens and `M` documents each, one group after
another, in an arena sized for the largest group instead of the whole pack.
`cake_backend.doc_batches(doc_lengths, B, M)` is the partition: greedy from the
first document; a zero-length document joins the group before it; a document
longer than `B` is refused, so the budget's floor is the longest document.  The
calls then take `doc_lengths`, the documents' token counts as host integers in
`cu_seqlens` order (`len == num_docs`, `sum == S`): the host reads no tensor,
so `cu_seqlens` cannot supply them, and a device-side comparison trips an
asynchronous assertion when the two disagree.  `kda_train_workspace(...,
arena=, doc_lengths=)` and `kda_train_workspace_bytes(..., arena=,
doc_lengths=)` size the arena for the exact groups; without `doc_lengths` the
arena takes the bound of every group, `64 * even(ceil(min(B, S) / 64) +
num_docs)` internal rows and the whole problem's segment count, at most
`num_docs` chunks and `(num_docs - N_b) * P` state blocks above the largest
group's.  The autograd node records `doc_lengths` with the option, so a
backward that allocates its own workspace is sized for the exact groups too.

Each group runs the whole problem's launch list through a view of the arena
(`KdaWorkspace.batch_view`): the group's slice of `cu_seqlens`; the group's own
tables (the segmented route's forward intra program writes them, the row's
table program otherwise, one more launch per group on the backward, whose
saved tables describe the whole problem); the whole token tensors and
gradients (every program reaches the tokens through the absolute `chunk_bos`
offsets of its tables); the group's slices of a saved `h` / `v_new` tape; and
its state blocks, recomputed in the view on the `light` and `replay` paths,
taken from the saved `hseg` / `seg_state` / `seg_m` by chain rank on
`light_states` and `tape`, and written back when the forward saves them.  The
host ranks the chains from `doc_lengths` by the tables' rule (chunk count,
most first, ties in document order), so a group whose blocks are one run of
the saved ones reads views of them and its forward carry writes the saved
`hseg` directly (the offsets / transfers are contiguous copies); a group whose
blocks are not one run gathers and scatters them by index.  The records, the
segment count per document (`P`, with the
non-proportional block table) and the saved set are the whole problem's; the
saving forward therefore writes the whole problem's tables once with the
saved-set paths' table program (`table_v3` / `table_v2`) before its groups.
The views share the arena, the `eye` / `flags` buffer and one epoch counter
(the overlap pair's release flags are stamped with the epoch).  The results
are bitwise the whole problem's with one caveat: the paired WY programs
(`wy_fwd`, `wy_bwd`) load the two chunks of a pair into one operand and
accumulate `0 * partner` for the other chunk's rows, so an exactly-zero entry
of `u` / `w` in a chunk whose pair partner changed (a group starting at an odd
chunk, or a pad partner) may change sign; the package's tests compare bit
patterns on rows built for this case.  The option refuses the rows whose block
table gives the documents proportional segment counts (`num_docs * H <= 4` on
the segmented route: a group of fewer documents would change every document's
segments) and, on a lean workspace, the checkpoint replay (`saved=None`); a
batched workspace that is not lean runs the replay group by group.  With
`tail_passes = B` the arena is sized for the largest group (over `doc_lengths`,
or the bound of every group) at `B` passes, and each group runs its tail in the
fewest passes `b <= B` whose lean layout at the group's geometry fits that arena
(`b = 1`: the one-pass lean layout): each tail stage launches once per range of
`ceil(NC_pad / b)` chunks instead of `ceil(NC_pad / B)`, with the same gradients
bit for bit and the same arena bytes.  `b` is the `tail_passes` of the group
view the backward runs: `KdaWorkspace.batch_view(group, tape=True)` (at the
group's `NC_pad_tape`) where the call reads slices of a saved tape or of the
saved `qg` / `w` / `u`, else `batch_view(group)`.  The costs
per group are a table launch, the gathers of a group whose blocks are not one
run (at most three state blocks per segment of the group, within
`batch_scratch` of `memory_report`) and the smaller grids; the serial route's
chains of different groups run one group after another.

```python
arena = kda_train_arena(lean=True, tail_passes=4, batch_tokens=120_000)
ws = kda_train_workspace(S, num_docs, H, device, arena=arena, doc_lengths=lengths)
out = chunk_kda(q, k, v, g, beta, cu_seqlens, workspace=ws, doc_lengths=lengths)
```

`cake_backend.memory_report(..., arena=, doc_lengths=)` adds `batch_tokens`,
`max_docs_per_batch`, `arena_bwd_bound_batched` (the bound of every group),
`batches` (`(d0, d1, S, N)` per group), `largest_batch_rows`, `batch_scratch`
and `arena_floor_batched` (one document per group under the lean four-pass
layouts: `2820 * R_min * H + 3 * P * H * 64 KiB`, `R_min` the internal rows of
the longest document; 2052 for the tape forms).  `cake_backend.arena_for_budget(
plan, budget, doc_lengths=)` and `KdaWorkspace.for_budget(S, num_docs, H,
device, arena_budget_bytes=, doc_lengths=)` return the first option that fits a
byte budget: the default, lean, two tail passes, four, then four passes over
groups halving from half the pack down to one document per group (the rows
that refuse an option skip it).

## Kernel structure of one training step

The host resolves one plan per `(S, num_docs, H, sm_count, arch)`: which
record of each stage runs on each path and with which grid.  The paths are
`fwd` (the forward with `save_for_backward`: its programs read the saved
`Aqk` / `Akk` token rows where a form for them ships and the intra program
writes no internal copies there), `fwd_nosave` (the forward without a saved
set: the internal-row programs), and four backward paths: `light` (from a version-4 saved
set without the segment-boundary states: every serial problem, and a
segmented pack below the size rule), `light_states` (from a version-4 set that
carries them: the segmented route only), `tape` (from a version-2 set) and
`replay` (`saved=None`).  The state recurrence has two routes, chosen by the
head count and the architecture: the serial walk over each document's chunk
chain for `H > 8` (the "64-column form" with two CTAs per document and head
when `num_docs * H * 4 > 4 * sm_count`, else the "32-column form" with four),
except the 16-head rows of `sm_103a` / `sm_107a`, which take the segmented
route (`cake_jit.SEGMENTS["plan_heads_by_arch"]`), and the segmented route
for `H <= 8` (and those rows): every document is cut into `P` chunk-aligned
segments (`NSEG = num_docs * P` pseudo-sequences described by the block table
`seg_table`) with `P = clamp(ceil(4 * sm_count / (num_docs * H)), 1, 64)`, at
most `sm_count // H`, and at most 38 (37 on `sm_100a`) when `num_docs * H <= 4`
(the "few-chain" rows, which also take proportional segment lengths: `seg_arg =
-num_docs`); `cake_jit.SEGMENTS` holds the constants (`small_segments_by_arch`
the per-architecture few-chain counts).

Forward (`fwd`): the chunk tables from `cu_seqlens` -- on the segmented route
up to 256 documents no table launch: the intra program `fwd_intra_v14`, launched
first, computes the layout in its prologue and writes the chunk tables and the
block table itself; above 256 documents `table` (one CTA) followed by
`seg_table` (the block table, one CTA); on the serial route `table_v2` (one CTA
per 256 chunks) up to 256 documents, else `table`; the intra program
(`fwd_intra_v14` on the segmented route up to 256 documents, `fwd_intra`
elsewhere: the intra-chunk
`Aqk` / `Akk` and, with `save_for_backward`, their token-row copies, and the
gated keys / queries of the GLM contract on the chunk layout, the chunk-final
gate row and the `v_new` pad rows: no separate prep launch); then the WY
representation and the recurrence, which also writes the output:

* serial, `H >= 32` (`sm_107a`: `H == 32` only): `fwdh_wy_rows` (on `fwd`,
  from the saved `Akk` / `Aqk` token rows) or `fwdh_wy` (on `fwd_nosave`)
  forms the WY representation itself, walks the chain and writes `out` (three
  launches);
* serial otherwise: `wy_fwd` (persistent grid of `min(NC_pad / 2 * H,
  sm_count)` CTAs over the chunk pairs; forms the gated values `v * beta` from
  the token rows) then `fwdh_o_v1` (64 columns) or `fwdh_o_v2` (32 columns),
  the documents longest first through the layout table's `seq_order` (four);
  on `fwd` the 32-column rows launch `wy_fwd_rows` (the backward's WY program
  on the saved `Akk`) and `fwdh_o_v2_rows` (`fwdh_v35`: the same recurrence
  reading the saved `Aqk`) and the intra program writes no internal `Aqk` /
  `Akk` copies; the 64-column rows do the same on `sm_107a` (`wy_fwd_rows` and
  `fwdh_o_v1_rows`: `fwdh_v35`'s 64-column build) and keep `wy_fwd` / `fwdh_o_v1`
  on both paths on `sm_100a` / `sm_103a`;
* segmented: `wy_fwd`, `fwdh_v3` pass 1 (every segment walked from zero and
  from the identity: the segment offsets and transfers, FP32 `[128, 128]`
  each; grid `2 * NSEG * H`), `seg_carry` (chains them within each document
  into the carried initial state of every segment; two CTAs per document and
  head, TF32 tensor-core products with FP32 accumulation; with at most two
  heads the two-level carry `seg_carry_v8` where its host rule admits the
  problem: the segments of every document in groups
  of `gsize`, each group's chain and the chain over the groups in one launch,
  the CTAs handing over through flag words in the identity buffer under a
  fresh epoch; `cake_jit.carry_choice` picks the group size minimising the
  critical path whose grid fits the device with every CTA resident and gains
  over the chain by more than the two hand-offs) and `fwdh_p3_o`
  (pass 2 from the carried states, writing the `h` / `v_new` tapes and `out`;
  grid `NSEG * H`) (five; seven above 256 documents, where `table` and
  `seg_table` run first).  On the rows whose pass-2 grid is at least 3.0
  waves of the device on `sm_100a` / `sm_103a` and 2.3 waves on `sm_107a`
  (`NSEG * H >= waves * sm_count`, and at most 8192 segment-heads) the carry
  and pass 2 are the overlap pair: `seg_carry_ov` releases one flag per
  segment, head and carry half as each carried state is stored, and
  `fwdh_p3_o_ov` (`fwdh_p3_o_ov_id` above two heads: the chain-major CTA
  order) is launched with the programmatic-stream-serialization attribute,
  starts while the carry runs and waits on its segment's flags (the flags live
  in the identity buffer, stamped with a per-call epoch; the same launch count,
  bitwise the same results).  On `fwd` those rows launch the token-row twins:
  `wy_fwd_rows` (the backward's WY program on the saved `Akk`) and
  `fwdh_p3_o_ov_rows` / `fwdh_p3_o_ov_id_rows` (the saved `Aqk`), and the intra
  program writes no internal `Aqk` / `Akk` copies.

With `save_mode="light"` and `S >= 384 * NSEG` the offsets / transfers and the
carried states go into the saved set's `seg_state` / `hseg` instead of the
workspace (the same launches).

Backward: the table programs (when the saved set carries none: `table_v3` on
the segmented route up to 256 documents; on the replay where the intra program
does not write them), the prep program (the gated keys / values of the GLM
contract and the chunk-final gate row on the chunk layout; no copies of `q` /
`k` / `v`, which the gradient kernels read on the token rows):
`bwd_prep_g8_noqg` on the light paths (the gated queries come from the saved
set), `bwd_prep_g8_noqg_nokb` on the light paths of the rows whose WY
representation is saved too (the paired serial 16-head rows of `sm_100a`, the
segmented 16-head rows of `sm_103a` / `sm_107a` above 32 (document, head)
chains: no `k * beta` either), `bwd_prep_g8`
on `tape` and `bwd_prep_g8_replay` on the replay (it also keeps a copy of
`dout` on the internal rows), on the replay the intra program (the internal
`Aqk` / `Akk`; on the segmented rows up to 256 documents `fwd_intra_v14` runs
before the prep and writes the tables the prep reads), then the recompute of
the states:

* serial 64-column form with `H >= 64` (`light` / `replay`): `fwdh_wy_bwd`
  forms the WY representation and walks the chain in one launch (no `wy`
  launch), and the adjoint `dhu_v1_wt` reads its transposed `w` tile;
* serial otherwise: `wy_bwd` (the saved `Akk` token rows; `wy_bwd_replay` the
  internal rows; no `wy` launch on the light path of the paired serial
  16-head rows of `sm_100a`, whose set carries `w` / `u`: the recompute and
  the adjoint read the saved tensors) then `fwdh_v1` (64
  columns; `fwdh_v1_sm107` on `sm_107a`), `fwdh_v2` (32 columns) or
  `fwdh_v2h` (its wider-tile build from 16 heads on `sm_100a` / `sm_103a`;
  `sm_107a` keeps `fwdh_v2`, its 16-head rows being segmented);
* segmented `light`: `wy_bwd`, `fwdh_v3` pass 1, `seg_carry`, `fwdh_p3`
  (pass 2); `light_states` from the saved `hseg`: pass 2 alone -- `fwdh_p3_wy`
  (the WY representation fused in; no `wy` launch) with at most 32 (document,
  head) chains, `fwdh_p3` after `wy_bwd` above; on the 16-head rows of
  `sm_103a` / `sm_107a` above 32 chains the set carries `w` / `u`: no `wy`
  launch on either light path, pass 1 / pass 2 and the adjoint read the saved
  tensors (`fwdh_v3`, `fwdh_p3`, `dhu_v3*_f`; the rows with at most 32 chains
  keep `fwdh_p3_wy` and the `_wt` adjoint forms);
* `tape`: `wy_bwd` only (the tapes hold the states);

then `dav_serial` (`dA_qk` alone; persistent grid over the chunk pairs) on
the serial rows below 16 heads, and no `dav` launch elsewhere: the fused
gradient hosts `dA_qk` (`dqkg_davq`) on the serial rows from 16 heads and on
every segmented row, whose adjoint recurrence accumulates `dv` (on `sm_107a`
from 16 heads the fused gradient is its `_nohint` build: the same kernel with
the L2 prefetch and evict-first hints off); the adjoint recurrence
`dhu` (the state gradient `dh` and `dv2`: `dhu_v1` / `dhu_v1_wt` or `dhu_v2t`
on the serial forms; on the segmented route two launches around a reverse
`seg_carry` (`seg_carry_v8` on its rows), every form accumulating `dv1` itself: on the few-chain rows
`dhu_v3s_f` (two CTAs per segment and head; `dhu_v3s_f_wt` reads the
transposed `w` of `fwdh_p3_wy`), above four chains `dhu_v3_f` with TMA tapes
when `NSEG * H < 4 * sm_count` (`<=` at `H <= 4`) -- on `sm_103a` below that
bound `dhu_v3p_f`, the same recurrence prefetching the next chunk's tiles into
L2 in the launches that write no tapes -- else the register-tape
build `dhu_v3r_f` (their `_wt` forms behind `fwdh_p3_wy`, their `_replay`
forms on the replay)); `dqkg` (fused dq / dk / dg / dbeta partials and
`dA_kk`; `_davq` forms host `dA_qk`; `_int` forms read the internal rows on
the replay; persistent grid of `min(NC_pad * H, sm_count)` CTAs);
then the GLM intra program: `glm_intra_v14` on `sm_103a` and `glm_intra_v18`
on `sm_107a` (the intra-chunk terms, the GLM gate chain and the gate epilogue's
token-row `dg` / `dbeta` in one launch: no `gate_epilogue` stage there;
`glm_intra_v18` is the same program with every named-barrier slot an
immediate, so two of its CTAs share an SM on `sm_107a`) or `glm_intra` and
`gate_epilogue` (the token-row `dg` and `dbeta`) on `sm_100a`.

Launches per call (148 SMs; the table programs of a saved-set path count only
when the set carries no tables), on `sm_100a`:

| problem | `fwd` / `fwd_nosave` | `light` | `light_states` | `tape` | `replay` |
|---|---|---|---|---|---|
| 1 document, 16 heads (serial, 32-column form; the saved `w` / `u` spare the WY launch) | 4 | 6 | - | 6 | 9 |
| 4 documents, 16 heads (serial, 32-column form; the saved `w` / `u` spare the WY launch) | 4 | 6 | - | 6 | 9 |
| 4 documents, 64 heads (serial, 64-column form; fused forward and recompute) | 3 | 6 | - | 6 | 8 |
| 4 documents, 2 heads (segmented, 8 chains, `P = 64`; `seg_carry_v8`) | 5 | 11 | 8 | 8 | 12 |
| 1 document, 4 heads (segmented, 4 chains, `P = 37`) | 5 | 11 | 8 | 8 | 12 |

Every backward path of `sm_100a` launches the gate epilogue (the last launch
of the counts above).  On `sm_103a` the GLM intra program hosts it, one
launch fewer on every backward path, and the 16-head rows are segmented
(`P = 9`): 1 document, 16 heads (16 chains) 5 / 10 / 7 / 7 / 11; 4 documents,
16 heads (64 chains: the set carries `w` / `u`, no `wy` launch on the light
paths) 5 / 9 / 7 / 7 / 11; 64 heads 3 / 5 / - / 5 / 7; 2 heads and 4 heads
5 / 10 / 7 / 7 / 11 (`P = 38` at 4 heads).  On `sm_107a` the gate epilogue
launches as on `sm_100a`: 1 document, 16 heads (segmented, `P = 9`) 5 / 11 /
8 / 8 / 12; 4 documents, 16 heads 5 / 10 / 8 / 8 / 12; 64 heads 4 / 6 / - /
6 / 8 (its 64-column saving forward reads the saved token rows through
`wy_fwd_rows` + `fwdh_o_v1_rows`); 2 heads and 4 heads 5 / 11 / 8 / 8 / 12.

The selection and grid rules are data in `cake_jit.MODULES` (`select_rule`,
`grid_rule`) and are evaluated by `cake_jit.select_record` / `launch_grid` /
`segment_plan` / `segment_args`; `cake_backend.plan(S, num_docs, H, sm_count,
arch)` resolves them once per problem into `Plan.records[path][stage]` (cached
on its arguments; `arch=None` takes the current CUDA device's architecture, or
the first exported one when no exported device is present), and
`Plan.launch_count(path)` is the launch count of each path (`"bwd"`: the
backward from the plan's own light saved set).

### Programs by stage

67 records naming 65 programs from 39 kernels (`fwdh_v3` and `fwdh_p3` are
one program, the two passes of the segmented walk; `bwd_prep_g8_noqg` and
`bwd_prep_g8_noqg_nokb` are two keyword forms of one kernel; four records
ship on some of the architectures only: `glm_intra_v14`, `glm_intra_v18`, `glm_intra` and
`gate_epilogue`; `wy_bwd` and `wy_fwd_rows`
are one program, the WY representation from the saved `Akk` on the backward
and on the saving forward's overlap and paired serial rows; `fwdh_p3_o_ov`,
`fwdh_p3_o_ov_id` and their `_rows` twins are the only programs launched with
the programmatic-stream-serialization attribute).  Rules: "serial" `H > 8`
except the 16-head rows of `sm_103a` / `sm_107a`, "segmented" `H <= 8` and
those rows; "64-column form" `num_docs * H * 4 > 4 * sm_count`,
"32-column form" otherwise; "few chains" `num_docs * H <= 4`.

| stage | records | dispatch |
|---|---|---|
| `table` | `table_v3`, `table_v2`, `table` | up to 256 documents one CTA per 256 chunks: `table_v2` on the serial route (forward, replay, and a saved-set path whose set carries no tables), `table_v3` on the segmented route (the block table too) on a saved-set path whose set carries no tables alone (the forward and the replay there launch no table program: `fwd_intra_v14` writes the tables); above, `table` (one CTA) on every path that builds the tables |
| `seg_table` | `seg_table` | segmented route above 256 documents, after `table` (below, `fwd_intra_v14` or `table_v3` writes the block table) |
| `fwd_intra` | `fwd_intra_v14`, `fwd_intra` | forward (the GLM prep's outputs too: `write_prep = 1`), and the replay (the intra-chunk rows alone); `fwd_intra_v14` on the segmented route up to 256 documents (the first launch of its call: it computes the chunk layout from `cu_seqlens` and writes the chunk tables and the block table), `fwd_intra` on the serial route and above 256 documents |
| `wy_fwd` | `wy_fwd`, `wy_fwd_rows` | forward, unless the fused recurrence (`fwdh_wy*`) runs; `wy_fwd_rows` (the saved `Akk` token rows, the `wy_bwd` program) on the overlap rows and the 32-column serial rows of `fwd` (on `sm_107a` its 64-column serial rows too), `wy_fwd` elsewhere |
| `wy` | `wy_bwd`, `wy_bwd_replay` | backward (`_replay` on the replay), unless a fused recompute forms the WY representation (`fwdh_wy_bwd*`, `fwdh_p3_wy`) or the light set carries `w` / `u` (the paired serial 16-head rows of `sm_100a`, the segmented 16-head rows of `sm_103a` / `sm_107a` above 32 chains: neither light path) |
| `fwdh` | `fwdh_wy`, `fwdh_wy_rows`, `fwdh_o_v1`, `fwdh_o_v1_rows`, `fwdh_o_v2`, `fwdh_o_v2_rows`, `fwdh_v3`, `fwdh_wy_bwd`, `fwdh_wy_bwd_replay`, `fwdh_v1`, `fwdh_v1_sm107`, `fwdh_v2`, `fwdh_v2h` | forward: serial `fwdh_wy_rows` (`fwd`: the saved token rows) / `fwdh_wy` (`fwd_nosave`) from 32 heads (`sm_107a`: at 32 only), else by form `fwdh_o_v1` (64 columns: `fwd_nosave`, and `fwd` on `sm_100a` / `sm_103a`) / `fwdh_o_v1_rows` (64 columns on `fwd` on `sm_107a`: `fwdh_v35` reading the saved `Aqk` token rows), `fwdh_o_v2` (32 columns, `fwd_nosave`) / `fwdh_o_v2_rows` (32 columns on `fwd`: `fwdh_v35` reading the saved `Aqk` token rows); segmented `fwdh_v3` pass 1 (both forward paths, `light`, replay). Backward recompute (`light`, replay): `fwdh_wy_bwd*` on the 64-column form from 64 heads, `fwdh_v1` (`fwdh_v1_sm107` on `sm_107a`) below, `fwdh_v2` / `fwdh_v2h` (from 16 heads on `sm_100a` / `sm_103a`; `sm_107a` keeps `fwdh_v2`, its 16-head rows being segmented) on the 32-column form |
| `seg_carry` | `seg_carry`, `seg_carry_ov`, `seg_carry_v8` | segmented route: between the forward passes (where pass 1 runs) and, in reverse, between the adjoint passes; `seg_carry_ov` (the release flags) on the forward paths of the rows with `NSEG * H >= 3.0 * sm_count` (`sm_100a` / `sm_103a`) or `>= 2.3 * sm_count` (`sm_107a`), at most 8192 segment-heads; on the other launches `seg_carry_v8` (the two-level carry; grid rule `carry_groups`) with at most two heads where `cake_jit.carry_choice` admits the problem, `seg_carry` elsewhere |
| `fwdh_p3` | `fwdh_p3_o`, `fwdh_p3_o_ov`, `fwdh_p3_o_ov_id`, `fwdh_p3_o_ov_rows`, `fwdh_p3_o_ov_id_rows`, `fwdh_p3_wy`, `fwdh_p3` | segmented pass 2: `fwdh_p3_o` on the forward paths (the output fused), `fwdh_p3_o_ov` / `fwdh_p3_o_ov_id` (PDL launch; the release order up to 2 heads, the chain-major order above) where `seg_carry_ov` runs on `fwd_nosave` and their token-row twins `fwdh_p3_o_ov_rows` / `fwdh_p3_o_ov_id_rows` (the saved `Aqk`) on `fwd`, `fwdh_p3_wy` on `light_states` with at most 32 chains (the WY representation fused), `fwdh_p3` otherwise (`light`, replay, `light_states` above 32 chains) |
| `bwd_prep` | `bwd_prep_g8_noqg`, `bwd_prep_g8_noqg_nokb`, `bwd_prep_g8`, `bwd_prep_g8_replay` | backward; one CTA per chunk and head: `_noqg` on the light paths (the saved `qg`), `_noqg_nokb` on the light paths of the rows whose set carries `w` / `u` (the paired serial 16-head rows of `sm_100a`, the segmented 16-head rows of `sm_103a` / `sm_107a` above 32 chains: no `k * beta`), `bwd_prep_g8` on `tape`, `_replay` on the replay (it keeps the internal `dout` copy) |
| `dav` | `dav_serial`, `dav_serial_replay` | serial rows below 16 heads (`dA_qk` alone, no `dv1` output); the `_replay` form on the replay; none elsewhere (the fused gradient hosts `dA_qk`) |
| `dhu` | `dhu_v1`, `dhu_v1_replay`, `dhu_v1_wt`, `dhu_v1_wt_replay`, `dhu_v2t`, `dhu_v2t_replay`, `dhu_v3_f`, `dhu_v3_f_wt`, `dhu_v3_f_replay`, `dhu_v3p_f`, `dhu_v3p_f_wt`, `dhu_v3p_f_replay`, `dhu_v3r_f`, `dhu_v3r_f_wt`, `dhu_v3r_f_replay`, `dhu_v3s_f`, `dhu_v3s_f_wt`, `dhu_v3s_f_replay` | serial 64-column form: `dhu_v1_wt*` behind the fused recompute (from 64 heads on `light` / replay), `dhu_v1*` otherwise; 32-column form: `dhu_v2t*`; segmented, two launches each, every form accumulating `dv1`: few chains `dhu_v3s_f` (`light`, `tape`), `dhu_v3s_f_wt` (`light_states`), `dhu_v3s_f_replay`; above four chains `dhu_v3_f*` (TMA tapes) when `NSEG * H < 4 * sm_count` (`<=` at `H <= 4`) -- on `sm_103a` `dhu_v3p_f*` (the form prefetching the next chunk's tiles) below `4 * sm_count` and `dhu_v3_f*` at the boundary alone -- else `dhu_v3r_f*`: the plain form on `light` / `tape` (and `light_states` above 32 chains), `_wt` on `light_states` with at most 32 chains, `_replay` on the replay |
| `dqkg` | `dqkg`, `dqkg_int`, `dqkg_davq`, `dqkg_int_davq`, `dqkg_davq_nohint`, `dqkg_int_davq_nohint` | backward; `_davq` from 16 serial heads and on every segmented row (hosts `dA_qk`), `_int` on the replay; on `sm_107a` from 16 heads the `_nohint` builds of the two `_davq` forms (the L2 prefetch and evict-first hints off; the hinted builds elsewhere); reads `q` / `k` / `v` on the token rows |
| `glm_intra` | `glm_intra_v14`, `glm_intra_v18`, `glm_intra` | backward; reads `k` / `q` on the token rows; `glm_intra_v14` on `sm_103a` and `glm_intra_v18` on `sm_107a` (the gate epilogue fused: they write the token-row `dg` / `dbeta` themselves; `glm_intra_v18` names every named-barrier slot with an immediate, so two CTAs share an SM on `sm_107a`), `glm_intra` on `sm_100a` |
| `gate_epilogue` | `gate_epilogue` | backward on `sm_100a` (the GLM intra programs of `sm_103a` / `sm_107a` host it) |

## Context-parallel training (KCP)

The packed batch of one training step may be split over the ranks of a
`torch.distributed` process group.  `kcp_partition` cuts the documents into
per-rank fragments (`"contiguous"`, `"zigzag"` or `"zigzag_per_document"`
layouts, an explicit `fragments` table, or `cut_points` with the
`"zigzag_per_document"` layout); `build_kcp_context` gives every rank the
device tables of its fragments, the collectives that carry the per-document
state recurrences and the convolution's history rows across the rank
boundaries, and the segment plan of its fragment walks.  The per-rank entry
points take the rank's shard of the operands (the tokens of its fragments, in
the partition's order) and return that shard's outputs and gradients:

```python
from flashinfer.kda_train import (kcp_partition, build_kcp_context, chunk_kda_kcp,
                                  chunk_kda_kcp_fwd, chunk_kda_kcp_bwd,
                                  kcp_workspace_bytes, kcp_saved_bytes, kcp_comm_bytes)
from flashinfer.cake_causal_conv1d_train import cake_causal_conv1d_kcp

part = kcp_partition(cu_seqlens_global_cpu, world_size)          # the same object on every rank
ctx = build_kcp_context(part, group, rank, device)               # this rank's tables and collectives
y_local = cake_causal_conv1d_kcp(x_local, conv_weight, ctx)      # the convolution before the KDA step
out_local = chunk_kda_kcp(q_local, k_local, v_local, g_local, beta_local, ctx)   # autograd
out_local, saved = chunk_kda_kcp_fwd(q_local, k_local, v_local, g_local, beta_local, ctx,
                                     save_mode="light", boundary_mode="saved")
dq, dk, dv, dbeta, dg = chunk_kda_kcp_bwd(q_local, k_local, v_local, g_local, beta_local, ctx,
                                          dout_local, saved=saved)
```

CUDA-graph replay: the three KDA entry points -- `chunk_kda_kcp`,
`chunk_kda_kcp_fwd` and `chunk_kda_kcp_bwd` -- take `graph="replay"`
(default `"off"`, the eager path); the convolution halo takes `graph=` only
through the backend module's function, the `cake_causal_conv1d_kcp` facade
has no such parameter. The first call of a
validity key runs eagerly, the second is captured into a CUDA graph and every
later call of the key replays it: the same launches and collectives, bitwise
the eager result, with no host work between them. The key is the call's kind,
its `KcpContext` / workspace / backend objects, every option and the shape,
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
`kcp_workspace_bytes(ctx, H=...)`, `kcp_saved_bytes(ctx, H=...)` and
`kcp_comm_bytes(ctx, H=..., C=...)` size the rank's workspace, saved set and
one step's traffic before any allocation.  Every rank of the group calls the
same entry points in the same order; the collectives are issued inside them.

Workspace of a rank (`KcpWorkspace`, caller-held, reused across calls and
layers): the local arena of the token-side stages plus the boundary pools.
The backward's b1 / b2 send pools reuse the storage of the forward's c1 / c2
send pools (a call waits its sends before a backward writes either), and the
ring exchange's hop pools are allocated by the first `exchange="ring"` call.
`kcp_workspace_bytes` is the number of bytes an a2a workspace allocates
(`KcpWorkspace.bytes` after its first backward); the saved set and the
convolution's halo rows are per-call tensors outside it.

Saved set of a rank (`KcpSaved`, `boundary_mode="saved"`): the fresh `S_in`
(the initial states of its fragments), with `save_transfer` the owner's c1
receive tensor and `M_seg`, on a rank whose fragments are split into `P > 1`
segments the forward's carry plane (the saved-mode backward replays its tape
from it instead of re-walking the segments and re-merging the carries; the
private test knob `saved_carry=0` selects the re-walk), and the call's
scalars; the int32 tables are rebuilt from the context.  The token-row `Aqk` /
`Akk` of the single-rank saved set are not part of it: the backward recomputes
the internal rows in both modes (saved set and checkpoint replay run the same
local programs), so `kcp_saved_bytes` counts the tables, `S_in`, the
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

Options of the KDA entry points: `save_transfer` keeps the owner's c1 receive
tensor (and `M_seg`) for the backward instead of re-sending it in the backward
exchange -- the default with the all-to-all exchange (an earlier revision).
`exchange="ring"` (available when every document spans at most three ranks)
receives no such tensor, so `save_transfer` resolves to `False` there
automatically; an explicit `save_transfer=True` together with the ring is a
`ValueError`. `KcpSaved.save_transfer` always reports the value that was used.
`spec3_probe` sets the debug probe of the backend's first saved-mode backward,
which compares the backward's prep and WY rows with the forward's: `"off"` (the
default) skips it, `"record"` stores the comparison in
`LAST_CALL["bwd"]["spec3"]`, `"warn"` warns on a difference, `"assert"` raises
`KcpPreconditionError`.  A probing backward keeps five token-row planes (about
`ctx.R * H * 1536` bytes: 1.5 KiB per head for every row of the rank's
chunk-padded token layout) allocated through its prep and WY programs, so its
peak memory is higher by up to that amount.
The backward reads the value of its own call (pass it to
`chunk_kda_kcp_bwd`; the forward's is not kept in `KcpSaved`).  Pass the
same value on every rank: the probe is a rank-local check, so `"assert"`
serves single-process debugging.  Under `graph="replay"` the probing
backward runs eagerly; a rank without tokens does not hold back its capture.
No environment variable selects it.

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

The KCP kernels live beside the single-rank programs:

* `cake_kcp_jit.py` -- the `MODULES` registry of the nine KCP programs (the
  fragment walks of the forward and backward state recurrences in their
  `offset` and `tape` forms, the BF16x3 transfer walk (two CTAs per segment
  and head, each over one 64-column half of the state), the affine
  merge at `cols_per_cta` 8 and 16, and the two chunk stages of the
  context-parallel route, `kcp_fwd_o` (the output from the K1 tapes) and
  `kcp_dav` (`dA_qk` and the `dv1` K2 reads); selection by `always` or by
  `columns` on the merge's `ncols`, grids by `kcp_segments_heads`,
  `kcp_chains`, `kcp_sm_bounded` or `kcp_pairs`), the logical vocabulary of
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
* `csrc/cake_kda_train_<digest>_{kernel,binding}.cu` of the thirteen KCP
  programs, rendered like the single-rank ones (digests in `build.json` and
  `build.test.json`).

## Layout of this package

* `cake_kcp_jit.py`, `cake_kcp_plan.py`, `cake_kcp.py` and `build.test.json`
  -- the context-parallel training path (see above).
* `cake_jit.py` -- the `MODULES` registry (one record per program: its
  `library` stem in `build.json`, the stage it fills and its variant, the
  selection and grid rules, the kernel-parameter -> logical-argument map and
  the argument plan of its `run` entry; the derived leaves `kernel_symbol`,
  `launch`, `arg_plan` and `logical` are rewritten by the export), the paths,
  the stage order, the optional stages and the segmented route's constants,
  the logical vocabulary, the loader glue (`library_name`, `load`, `specs_for_arch`) and
  `run_module(name, values, grid=, device_index=)`.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`): one entry per `<library>_<arch>` (the 65 programs on the
  architectures each ships on: 61 on all three, `glm_intra_v14` on `sm_103a`,
  `glm_intra_v18` on `sm_107a`, `glm_intra` and `gate_epilogue` on `sm_100a`) with its two translation units, digests, nvcc flags
  (`--use_fast_math`), closure digest and `aot/<name>.so` path.
  `cake_build.load_library(name)` loads the prebuilt library when `aot/` holds
  it and otherwise JIT-builds the entry through FlashInfer (`gen_jit_spec`,
  sources digest-verified first; `sm_107a` needs an nvcc that lists
  `compute_107`).  `aot/` is never tracked (`.gitignore`).
* `cake_backend.py` -- operand checks, the host plan, the arena workspace and
  its lifetime tables, the saved set, the forward and backward launch
  sequences, the autograd `Function` and the byte queries.
* `csrc/cake_ffi_checks.h` -- the TVM-FFI helper header the bindings include,
  shipped by the export so the package compiles without FlashInfer's `csrc/`.
* `csrc/cake_kda_train_<digest>_{kernel,binding}.cu` -- the generated device
  and binding translation units of the 65 programs, shared by the
  architectures a program ships on (per-architecture lines under `#if
  __CUDA_ARCH__`) and compiled once per architecture.  The sources are identity-checked by the
  digests in `build.json`.

## Status

The registry holds the 67 records (65 programs) for `sm_100a`, `sm_103a` and
`sm_107a` (three of them on a subset of the architectures: see the programs by
stage), exported from the kernel snapshot named in the pull request.

The KCP path adds nine shipped programs (`cake_kcp_jit.MODULES`) to
`build.json` for the same three architectures and four test-only programs to
`build.test.json`.

Tests: `tests/experimental/test_cake_kda_train.py` (the backend suite: imports
only this package through `tests/test_helpers/cake_package`; the registry and
plan tests run everywhere, the kernel tests skip without a compute capability
10.0 / 10.3 / 10.7 device) against the exact FP64 per-document recurrence of
`tests/test_helpers/cake_kda_train_reference.py`, and
`tests/experimental/test_cake_kda_train_api.py` (the public entry points).
Benchmark: `benchmarks/bench_cake_kda_train.py`.  KCP:
`tests/experimental/test_cake_kda_train_kcp_build.py` (the KCP registries and
manifests, no device needed) and `tests/experimental/test_cake_kda_train_kcp.py`
(the per-rank path); benchmark `benchmarks/bench_cake_kda_train_kcp.py`
(`torchrun`, one process per rank).
