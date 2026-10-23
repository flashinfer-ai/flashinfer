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
from flashinfer.kda_train import chunk_kda, chunk_kda_fwd, chunk_kda_bwd, kda_train_workspace

out = chunk_kda(q, k, v, g, beta, cu_seqlens, scale=None, workspace=None, save_mode=None)  # autograd
out, saved = chunk_kda_fwd(q, k, v, g, beta, cu_seqlens, scale=None,
                           save_for_backward=True, workspace=None, save_mode=None)
dq, dk, dv, dbeta, dg = chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, dout,
                                      saved=saved, scale=None, workspace=None)
dq, dk, dv, dbeta, dg = chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, dout)         # checkpoint replay
ws = kda_train_workspace(S, num_docs, H, device, backward=True, save_mode="light")  # caller-owned
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
| `version` | `3` (`save_mode="light"`, the default) or `2` (`save_mode="tape"`); version 1 (the light set without the segment states) is no longer produced or accepted |
| `Aqk`, `Akk` | BF16 `[S, H, 64]` on the token rows: row `t` holds the 64 chunk-local columns of its chunk (zeros above the diagonal and past the chunk end) |
| `tables` | `(chunk_bos, chunk_len, seq_chunk_start, seq_order[, seg_table])`: int32 views of one device buffer (`NC_pad`, `NC_pad`, `num_docs + 1`, `num_docs` and, on the segmented route, the block table's `NSEG + 2 * num_docs + 1` entries), or `None` (the backward rebuilds them from `cu_seqlens`) |
| `S`, `H`, `num_docs`, `scale` | the problem the set belongs to |
| `h`, `v_new` | version 2 only: the forward's state tape BF16 `[NC_pad, H, 128, 128]` and `v_new` BF16 `[R, H, 128]` on the internal layout |
| `seg_m` | version 2 on the segmented route: the segment transfers FP32 `[NSEG, H, 128, 128]` the backward's reverse carry reads; `None` on the serial route |
| `hseg` | version 3 on the segmented route when `S >= 384 * NSEG`: the carried initial state of every segment, FP32 `[NSEG, H, 128, 128]` (the backward's recurrence recompute walks from it: pass 2 alone); `None` otherwise |
| `seg_state` | version 3, idem: FP32 `[2, NSEG, H, 128, 128]`, `[0]` the segment offsets, `[1]` the segment transfers the backward's reverse carry reads; `None` otherwise |

`kda_train_saved_bytes(S, H) = 256 * S * H` bytes (`Aqk` and `Akk`);
`kda_train_saved_bytes(S, H, num_docs, save_mode=...)` adds the tables and
the mode's extra tensors: for `"light"` (version 3) the segment-boundary
states, `3 * NSEG * H * 65536` bytes (`hseg` and the `seg_state` pair, FP32
`[128, 128]` per segment and head) on the segmented route (`H <= 8`) when
`S >= 384 * NSEG`, nothing otherwise; for `"tape"` the version-2 tensors
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
device's when `None`) and the current device's architecture (the
architecture-independent count of 38 without a device).  Every
tensor of the set is allocated by the call that writes it: a saved set can go
to a backward on any workspace of the same problem.  A caller may keep only
`Aqk` / `Akk` (`tables=None`).  With `save_for_backward=False` the token-row
`Aqk` / `Akk` are not written (the `fwd_nosave` path: the forward's programs
read the internal copies) and the backward must replay (`saved=None`:
the table where the plan launches one, prep and intra recompute them).  From a version-2 set the backward
skips its recurrence recompute (WY still runs for `w`); from a version-3 set
with the states its recurrence recompute is pass 2 alone from `hseg` and the
reverse carry reads `seg_state[1]`; from a version-3 set without them or on
the replay path it recomputes the states, `h` and `v_new` in its workspace.
The gradients are bitwise the same on every path; the backward writes nothing
into the saved set.

### Workspace (`KdaWorkspace`)

A workspace is one arena (a `uint8` tensor) holding the internal-layout
intermediates of one `(S, num_docs, H)` problem; `chunk_kda_fwd` /
`chunk_kda_bwd` allocate only their outputs and the saved set when given one,
and without one each call allocates a workspace of its own.  Buffers whose
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

## Kernel structure of one training step

The host resolves one plan per `(S, num_docs, H, sm_count, arch)`: which
record of each stage runs on each path and with which grid.  The paths are
`fwd` (the forward with `save_for_backward`: its programs read the saved
`Aqk` / `Akk` token rows where a form for them ships and the intra program
writes no internal copies there), `fwd_nosave` (the forward without a saved
set: the internal-row programs), and four backward paths: `light` (from a version-3 saved
set without the segment-boundary states: every serial problem, and a
segmented pack below the size rule), `light_states` (from a version-3 set that
carries them: the segmented route only), `tape` (from a version-2 set) and
`replay` (`saved=None`).  The state recurrence has two routes, chosen by the
head count alone: the serial walk over each document's chunk chain for
`H > 8` (the "64-column form" with two CTAs per document and head when
`num_docs * H * 4 > 4 * sm_count`, else the "32-column form" with four), and
the segmented route for `H <= 8`: every document is cut into `P` chunk-aligned
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
  `Akk` copies; the 64-column rows keep `wy_fwd` / `fwdh_o_v1` on both paths;
* segmented: `wy_fwd`, `fwdh_v3` pass 1 (every segment walked from zero and
  from the identity: the segment offsets and transfers, FP32 `[128, 128]`
  each; grid `2 * NSEG * H`), `seg_carry` (chains them within each document
  into the carried initial state of every segment; two CTAs per document and
  head, TF32 tensor-core products with FP32 accumulation) and `fwdh_p3_o`
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
does not write them), `bwd_prep_g8` (the gated keys / values of the GLM contract and the
chunk-final gate row on the chunk layout; no copies of `q` / `k` / `v`, which
the gradient kernels read on the token rows; the replay form also keeps a copy
of `dout` on the internal rows), on the replay the intra program (the internal
`Aqk` / `Akk`; on the segmented rows up to 256 documents `fwd_intra_v14` runs
before the prep and writes the tables the prep reads), then the recompute of
the states:

* serial 64-column form with `H >= 64` (`light` / `replay`): `fwdh_wy_bwd`
  forms the WY representation and walks the chain in one launch (no `wy`
  launch), and the adjoint `dhu_v1_wt` reads its transposed `w` tile;
* serial otherwise: `wy_bwd` (the saved `Akk` token rows; `wy_bwd_replay` the
  internal rows) then `fwdh_v1` (64 columns; `fwdh_v1_sm107` on `sm_107a`),
  `fwdh_v2` (32 columns) or `fwdh_v2h` (its wider-tile build from 16 heads;
  on `sm_107a` at 16 heads only);
* segmented `light`: `wy_bwd`, `fwdh_v3` pass 1, `seg_carry`, `fwdh_p3`
  (pass 2); `light_states` from the saved `hseg`: pass 2 alone -- `fwdh_p3_wy`
  (the WY representation fused in; no `wy` launch) with at most 32 (document,
  head) chains, `fwdh_p3` after `wy_bwd` above;
* `tape`: `wy_bwd` only (the tapes hold the states);

then `dav_serial` (`dA_qk` alone; persistent grid over the chunk pairs) on
the serial rows below 16 heads, and no `dav` launch elsewhere: the fused
gradient hosts `dA_qk` (`dqkg_davq`) on the serial rows from 16 heads and on
every segmented row, whose adjoint recurrence accumulates `dv`; the adjoint recurrence
`dhu` (the state gradient `dh` and `dv2`: `dhu_v1` / `dhu_v1_wt` or `dhu_v2t`
on the serial forms; on the segmented route two launches around a reverse
`seg_carry`, every form accumulating `dv1` itself: on the few-chain rows
`dhu_v3s_f` (two CTAs per segment and head; `dhu_v3s_f_wt` reads the
transposed `w` of `fwdh_p3_wy`), above four chains `dhu_v3_f` with TMA tapes
when `NSEG * H < 4 * sm_count` (`<=` at `H <= 4`) -- on `sm_103a` below that
bound `dhu_v3p_f`, the same recurrence prefetching the next chunk's tiles into
L2 in the launches that write no tapes -- else the register-tape
build `dhu_v3r_f` (their `_wt` forms behind `fwdh_p3_wy`, their `_replay`
forms on the replay)); `dqkg` (fused dq / dk / dg / dbeta partials and
`dA_kk`; `_davq` forms host `dA_qk`; `_int` forms read the internal rows on
the replay; persistent grid of `min(NC_pad * H, sm_count)` CTAs);
`glm_intra` (the intra-chunk terms and the GLM gate chain) and
`gate_epilogue` (the token-row `dg` and `dbeta`).

Launches per call (148 SMs; the table programs of a saved-set path count only
when the set carries no tables):

| problem | `fwd` / `fwd_nosave` | `light` | `light_states` | `tape` | `replay` |
|---|---|---|---|---|---|
| 1 document, 16 heads (serial, 32-column form) | 4 | 7 | - | 6 | 9 |
| 4 documents, 64 heads (serial, 64-column form; fused forward and recompute) | 3 | 6 | - | 6 | 8 |
| 4 documents, 2 heads (segmented, 8 chains) | 5 | 11 | 8 | 8 | 12 |
| 1 document, 4 heads (segmented, 4 chains) | 5 | 11 | 8 | 8 | 12 |

The selection and grid rules are data in `cake_jit.MODULES` (`select_rule`,
`grid_rule`) and are evaluated by `cake_jit.select_record` / `launch_grid` /
`segment_plan` / `segment_args`; `cake_backend.plan(S, num_docs, H, sm_count,
arch)` resolves them once per problem into `Plan.records[path][stage]`, and
`Plan.launch_count(path)` is the launch count of each path (`"bwd"`: the
backward from the plan's own light saved set).

### Programs by stage

59 records naming 57 programs from 35 kernels (`fwdh_v3` and `fwdh_p3` are
one program, the two passes of the segmented walk; `wy_bwd` and `wy_fwd_rows`
are one program, the WY representation from the saved `Akk` on the backward
and on the saving forward's overlap and paired serial rows; `fwdh_p3_o_ov`,
`fwdh_p3_o_ov_id` and their `_rows` twins are the only programs launched with
the programmatic-stream-serialization attribute).  Rules: "serial" `H > 8`,
"segmented" `H <= 8`; "64-column form" `num_docs * H * 4 > 4 * sm_count`,
"32-column form" otherwise; "few chains" `num_docs * H <= 4`.

| stage | records | dispatch |
|---|---|---|
| `table` | `table_v3`, `table_v2`, `table` | up to 256 documents one CTA per 256 chunks: `table_v2` on the serial route (forward, replay, and a saved-set path whose set carries no tables), `table_v3` on the segmented route (the block table too) on a saved-set path whose set carries no tables alone (the forward and the replay there launch no table program: `fwd_intra_v14` writes the tables); above, `table` (one CTA) on every path that builds the tables |
| `seg_table` | `seg_table` | segmented route above 256 documents, after `table` (below, `fwd_intra_v14` or `table_v3` writes the block table) |
| `fwd_intra` | `fwd_intra_v14`, `fwd_intra` | forward (the GLM prep's outputs too: `write_prep = 1`), and the replay (the intra-chunk rows alone); `fwd_intra_v14` on the segmented route up to 256 documents (the first launch of its call: it computes the chunk layout from `cu_seqlens` and writes the chunk tables and the block table), `fwd_intra` on the serial route and above 256 documents |
| `wy_fwd` | `wy_fwd`, `wy_fwd_rows` | forward, unless the fused recurrence (`fwdh_wy*`) runs; `wy_fwd_rows` (the saved `Akk` token rows, the `wy_bwd` program) on the overlap rows and the 32-column serial rows of `fwd`, `wy_fwd` elsewhere |
| `wy` | `wy_bwd`, `wy_bwd_replay` | backward (`_replay` on the replay), unless a fused recompute forms the WY representation (`fwdh_wy_bwd*`, `fwdh_p3_wy`) |
| `fwdh` | `fwdh_wy`, `fwdh_wy_rows`, `fwdh_o_v1`, `fwdh_o_v2`, `fwdh_o_v2_rows`, `fwdh_v3`, `fwdh_wy_bwd`, `fwdh_wy_bwd_replay`, `fwdh_v1`, `fwdh_v1_sm107`, `fwdh_v2`, `fwdh_v2h` | forward: serial `fwdh_wy_rows` (`fwd`: the saved token rows) / `fwdh_wy` (`fwd_nosave`) from 32 heads (`sm_107a`: at 32 only), else by form `fwdh_o_v1` (64 columns, both forward paths), `fwdh_o_v2` (32 columns, `fwd_nosave`) / `fwdh_o_v2_rows` (32 columns on `fwd`: `fwdh_v35` reading the saved `Aqk` token rows); segmented `fwdh_v3` pass 1 (both forward paths, `light`, replay). Backward recompute (`light`, replay): `fwdh_wy_bwd*` on the 64-column form from 64 heads, `fwdh_v1` (`fwdh_v1_sm107` on `sm_107a`) below, `fwdh_v2` / `fwdh_v2h` (from 16 heads; `sm_107a`: at 16 only) on the 32-column form |
| `seg_carry` | `seg_carry`, `seg_carry_ov` | segmented route: between the forward passes (where pass 1 runs) and, in reverse, between the adjoint passes; `seg_carry_ov` (the release flags) on the forward paths of the rows with `NSEG * H >= 3.0 * sm_count` (`sm_100a` / `sm_103a`) or `>= 2.3 * sm_count` (`sm_107a`), at most 8192 segment-heads |
| `fwdh_p3` | `fwdh_p3_o`, `fwdh_p3_o_ov`, `fwdh_p3_o_ov_id`, `fwdh_p3_o_ov_rows`, `fwdh_p3_o_ov_id_rows`, `fwdh_p3_wy`, `fwdh_p3` | segmented pass 2: `fwdh_p3_o` on the forward paths (the output fused), `fwdh_p3_o_ov` / `fwdh_p3_o_ov_id` (PDL launch; the release order up to 2 heads, the chain-major order above) where `seg_carry_ov` runs on `fwd_nosave` and their token-row twins `fwdh_p3_o_ov_rows` / `fwdh_p3_o_ov_id_rows` (the saved `Aqk`) on `fwd`, `fwdh_p3_wy` on `light_states` with at most 32 chains (the WY representation fused), `fwdh_p3` otherwise (`light`, replay, `light_states` above 32 chains) |
| `bwd_prep` | `bwd_prep_g8`, `bwd_prep_g8_replay` | backward; one CTA per chunk and head; the replay form keeps the internal `dout` copy |
| `dav` | `dav_serial`, `dav_serial_replay` | serial rows below 16 heads (`dA_qk` alone, no `dv1` output); the `_replay` form on the replay; none elsewhere (the fused gradient hosts `dA_qk`) |
| `dhu` | `dhu_v1`, `dhu_v1_replay`, `dhu_v1_wt`, `dhu_v1_wt_replay`, `dhu_v2t`, `dhu_v2t_replay`, `dhu_v3_f`, `dhu_v3_f_wt`, `dhu_v3_f_replay`, `dhu_v3p_f`, `dhu_v3p_f_wt`, `dhu_v3p_f_replay`, `dhu_v3r_f`, `dhu_v3r_f_wt`, `dhu_v3r_f_replay`, `dhu_v3s_f`, `dhu_v3s_f_wt`, `dhu_v3s_f_replay` | serial 64-column form: `dhu_v1_wt*` behind the fused recompute (from 64 heads on `light` / replay), `dhu_v1*` otherwise; 32-column form: `dhu_v2t*`; segmented, two launches each, every form accumulating `dv1`: few chains `dhu_v3s_f` (`light`, `tape`), `dhu_v3s_f_wt` (`light_states`), `dhu_v3s_f_replay`; above four chains `dhu_v3_f*` (TMA tapes) when `NSEG * H < 4 * sm_count` (`<=` at `H <= 4`) -- on `sm_103a` `dhu_v3p_f*` (the form prefetching the next chunk's tiles) below `4 * sm_count` and `dhu_v3_f*` at the boundary alone -- else `dhu_v3r_f*`: the plain form on `light` / `tape` (and `light_states` above 32 chains), `_wt` on `light_states` with at most 32 chains, `_replay` on the replay |
| `dqkg` | `dqkg`, `dqkg_int`, `dqkg_davq`, `dqkg_int_davq` | backward; `_davq` from 16 serial heads and on every segmented row (hosts `dA_qk`), `_int` on the replay; reads `q` / `k` / `v` on the token rows |
| `glm_intra` | `glm_intra` | backward; reads `k` / `q` on the token rows |
| `gate_epilogue` | `gate_epilogue` | backward |

## Context-parallel training (KCP)

The packed batch of one training step may be split over the ranks of a
`torch.distributed` process group.  `kcp_partition` cuts the documents into
per-rank fragments (`"contiguous"`, `"zigzag"` or `"zigzag_per_document"`
layouts, or explicit cut points); `build_kcp_context` gives every rank the
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

CUDA-graph replay: every KCP entry point -- `chunk_kda_kcp`,
`chunk_kda_kcp_fwd`, `chunk_kda_kcp_bwd` and `cake_causal_conv1d_kcp` -- takes
`graph="replay"` (default `"off"`, the eager path). The first call of a
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
the A/B rows (the per-layer forward and backward of CP4-CP16 at the B1 /
B3 batch sizes), at most `GRAPH_CACHE_MAX = 64` keys (LRU). Call
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
(`CAKE_KCP_TEST_KNOBS=1`); `graph="off"` is the shipped default. The A/B
record was taken with pointer-stable harness inputs; its gain is an upper
bound of what a framework sees.

Limits of a partition: at most 65536 fragments per rank (the layout table's
document search), at most 64 segments per fragment and no walk chain longer
than 256 chunk steps (the segment plan's bounds); the byte queries
`kcp_workspace_bytes(ctx, H=...)`, `kcp_saved_bytes(ctx, H=...)` and
`kcp_comm_bytes(ctx, H=..., C=...)` size the rank's workspace, saved set and
one step's traffic before any allocation.  Every rank of the group calls the
same entry points in the same order; the collectives are issued inside them.

Saved set of a rank (`KcpSaved`, `boundary_mode="saved"`): the fresh `S_in`
(the initial states of its fragments), with `save_transfer` the owner's c1
receive tensor and `M_seg`, and the call's scalars; the int32 tables are
rebuilt from the context.  The token-row `Aqk` / `Akk` of the single-rank
saved set are not part of it: the backward recomputes the internal rows in
both modes (saved set and checkpoint replay run the same local programs), so
`kcp_saved_bytes` counts the tables, `S_in` and the `save_transfer` tensors
(and the convolution's history rows with `with_conv`).

Walk-launch geometry: the walk kernels' CTA lists are issued
longest-segment-first and the two tape programs (`fwdh_tape`, `dhu_tape`) run
at one CTA per SM. Because the JIT binding exposes no per-function carveout
attribute, the host sets the device-wide cache-config preference around each
tape launch (`cudaDeviceSetCacheConfig`: PreferL1 before, PreferNone after)
and restores it before the next launch; a launch issued by another thread of
the same process during that bracket would take the PreferL1 occupancy for
itself (occupancy only -- results never depend on the geometry). The earlier
geometry stays reachable for A/B records through the private test knob
`walk_geometry` (`CAKE_KCP_TEST_KNOBS=1`).

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
through the private test knob `host_trim` (`CAKE_KCP_TEST_KNOBS=1`; the conv
module's `_set_test_knobs({"host_trim": 0})`).

Options of the KDA entry points: `save_transfer` keeps the owner's c1 receive
tensor (and `M_seg`) for the backward instead of re-sending it in the backward
exchange -- the default with the all-to-all exchange (an earlier revision).
`exchange="ring"` (available when every document spans at most three ranks)
receives no such tensor, so `save_transfer` resolves to `False` there
automatically; an explicit `save_transfer=True` together with the ring is a
`ValueError`. `KcpSaved.save_transfer` always reports the value that was used.

The segment plan splits every fragment into `P` segments: above the 256-chunk
chain cap the equal-segment wave rule, below it on one-fragment ranks the
largest `P` that runs one CTA per SM (`floor(SM / H)`), on every other rank
one CTA-per-SM occupancy (`ceil(SM / (n_frag H))`) -- except that a rank
with two or three fragments below the cap whose `P = 1` walks run one wave
(`n_frag H <= SM`) takes `P = 1` when the longest-processing-time makespan of
its actual fragment chunk counts at `P = 1` is not above the makespan at the
occupancy `P` (such a rank runs no segment composition and no backward
re-walk; equal two-fragment ranks tie, unequal ones keep `P > 1`);
`ctx.describe(H)["p_rule"]` records the rule (`"wave_fill1_p1tie"`). The
previous rules stay selectable through the private test knob `p_rule`, as
does the fused form of the carry merges (`merge_fuse="carry"`: the two carry
families of a phase in one launch; off by default).

The KCP kernels live beside the single-rank programs:

* `cake_kcp_jit.py` -- the `MODULES` registry of the seven KCP programs (the
  fragment walks of the forward and backward state recurrences in their
  `offset` and `tape` forms, the TF32 hi/lo transfer walk and the affine
  merge at `cols_per_cta` 8 and 16; selection by `always` or by `columns` on
  the merge's `ncols`, grids by `kcp_segments_heads` or `kcp_chains`), the
  logical vocabulary of the KCP arguments, and `run_program(name, args,
  device_index=)`, the lowering of the host backend's walk and merge requests
  to the registry's launches.  Its `TEST_MODULES` are test-only programs of
  `build.test.json` (below).
* `cake_kcp_plan.py` -- the partition, the segment plan and the context's
  tables and collectives, with the byte queries.
* `cake_kcp.py` -- the per-rank launch sequences, the saved set and workspace
  of a rank, the autograd `Function` and the three entry points.
* `build.test.json` -- a `cake-build/1` manifest of the same form as
  `build.json` holding the test-only programs of `cake_kcp_jit.TEST_MODULES`
  (the transfer walk's static `precision_mode` 1 and 2 forms and the BF16x3
  candidate at both value widths).  `build.json` never lists them and the
  shipped route never loads them: `cake_kcp_jit.load_test` JIT-builds an
  entry only under the environment variable `CAKE_KCP_TEST_KNOBS=1`, which
  also admits the test knobs (`precision_mode`, `transfer_walk`) of the
  package's own tests; without the file those tests skip.
* `csrc/cake_kda_train_<digest>_{kernel,binding}.cu` of the eleven KCP
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
  `cake-build/1`): one entry per `<library>_<arch>` (57 programs x 3
  architectures) with its two translation units, digests, nvcc flags
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
  and binding translation units of the 57 programs, shared by the three
  architectures (per-architecture lines under `#if __CUDA_ARCH__`) and
  compiled once per architecture.  The sources are identity-checked by the
  digests in `build.json`.

## Status

The registry holds the 59 records (57 programs) for `sm_100a`, `sm_103a` and
`sm_107a`,
exported from the kernel snapshot named in the pull request.

The KCP path adds seven shipped programs (`cake_kcp_jit.MODULES`) to
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
