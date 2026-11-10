# Cake packed KDA chunked training, non-deterministic package (SM100 / SM103 / SM107)

Backend for the packed Kimi Delta Attention (KDA) chunked training forward
and backward in the GLM-5.3-Flash geometry, the non-deterministic counterpart
of the deterministic package `flashinfer.experimental.cake.cake_kda_det_train`.
The public API is the facade `flashinfer.cake_kda_nondet_train`; this package
is its backend, internal to FlashInfer.  The feature does not participate in
automatic backend selection, autotuning or trace-apply.

The operands, shapes, dtypes, saved set (versions 4 and 2), workspace, arena
options, document groups (`batch_tokens`), launch sequences and byte queries
are the deterministic package's; its README
(`flashinfer/experimental/cake/cake_kda_det_train/README.md`) describes them.  This
package imports nothing from it and ships its own copies of every program.
This README describes what differs, including the checkpoint hand-off of
`cake_kda_nondet`, which the deterministic package does not have.

## Entry points and numerical mode

The public entry points are the facade functions of `flashinfer.cake_kda_nondet_train`:

```python
from flashinfer.cake_kda_nondet_train import cake_kda_nondet, cake_kda_nondet_train_forward, cake_kda_nondet_train_backward, cake_kda_nondet_train_workspace

out = cake_kda_nondet(q, k, v, g, beta, cu_seqlens)                                   # autograd
ws = cake_kda_nondet_train_workspace(S, num_docs, H, device)                          # caller-owned
out, saved = cake_kda_nondet_train_forward(q, k, v, g, beta, cu_seqlens, workspace=ws)
dq, dk, dv, dbeta, dg = cake_kda_nondet_train_backward(q, k, v, g, beta, cu_seqlens, dout, saved=saved,
                                                       workspace=ws)

# raw gate mode: the kernels activate the gate (FLA's use_gate_in_kernel keywords)
raw_kw = dict(use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0)
out = cake_kda_nondet(q, k, v, raw, beta, cu_seqlens, **raw_kw)                      # grads to raw, A_log, dt_bias
ws = cake_kda_nondet_train_workspace(S, num_docs, H, device, use_gate_in_kernel=True)
out, saved = cake_kda_nondet_train_forward(q, k, v, raw, beta, cu_seqlens, workspace=ws, **raw_kw)
dq, dk, dv, dbeta, draw, dA_log, ddt_bias = cake_kda_nondet_train_backward(
    q, k, v, raw, beta, cu_seqlens, dout, saved=saved, workspace=ws, **raw_kw)
```

The facade functions (`cake_kda_nondet_train_forward`, `cake_kda_nondet_train_backward`, `cake_kda_nondet`,
`cake_kda_nondet_train_arena`, `cake_kda_nondet_train_workspace`, `cake_kda_nondet_train_workspace_bytes`,
`cake_kda_nondet_train_saved_bytes`, `is_cake_kda_nondet_train_supported`) take the
deterministic facade's arguments and no mode argument; `cake_kda_nondet` adds
`checkpoint_handoff` (see "Checkpoint hand-off").  They always run this
package.

`CakeKDANondetFunction.apply` takes `checkpoint_handoff` (`None`, `"auto"`, `"on"`
or `"off"`) as the positional argument after `doc_lengths`, before the raw
gate arguments.  A `bool` in that slot raises `TypeError`: an `apply`
written with the deterministic Function's raw gate arguments after
`doc_lengths` fails instead of running with shifted arguments.

- The autograd node's backward is this package's.
- Both packages write the saved set in the same format (the forward programs
  are the same), but each backward reads its own package's set only.  Every
  entry of this package refuses a `workspace`, a `saved` set or an `arena`
  option that is not this package's type (`CakeKDANondetWorkspace`,
  `CakeKDANondetChunkSaved`, `CakeKDANondetArena`) with `ValueError` naming the
  mode, before any other check: the deterministic package's
  `CakeKDADetWorkspace`, `CakeKDADetChunkSaved` and `CakeKDADetArena` included.
  The deterministic package refuses this package's objects in the same way.
- No launch of this package is unordered (every reduction has a fixed order),
  so `torch.use_deterministic_algorithms(True)`, also with
  `warn_only=True`, raises no alert and no warning here, and the entries never
  run the deterministic package.  A future launch with an unordered reduction
  must raise PyTorch's non-determinism alert immediately before that launch.
- This package runs the explicit FP32 gate and the raw gate mode
  (`use_gate_in_kernel=True` with `A_log`, `dt_bias` and `lower_bound`,
  FlashInfer #7816) under the deterministic package's contract for that mode
  (its README, "Raw gate mode"): the same operands, stride rule and refusals,
  the 7-tuple `(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)` of `cake_kda_nondet_train_backward`,
  gradients to `g`, `A_log` and `dt_bias` from `cake_kda_nondet`, and one
  `param_finalize` launch per backward.  The mode is part of the plan
  (`cake_backend.plan(..., gate="raw")`), of the workspace
  (`CakeKDANondetWorkspace.for_problem` / `for_budget` and `workspace_bytes` with
  `gate="raw"`; a call in the other mode is refused with `ValueError`, in both
  directions) and of the saved set.
- The saved set carries `gate` / `lower_bound` (the deterministic package's
  field names); a backward in the other gate mode or with another
  `lower_bound` is refused.
- This package runs the in-kernel q / k L2 norm (`use_qk_l2norm_in_kernel=True`,
  raw `q` / `k` normalized inside the kernels, FlashInfer #7817) in either gate
  mode under the deterministic package's contract for it: the same keyword and
  modes (`cake_jit.L2N_MODES`: `"off"`, `"fused"`, and the `"fallback"` test
  route), operands, saved `qk_rstd`, workspace and byte queries.  The mode is
  part of the plan, the workspace and the saved set, as in the deterministic
  package.  The registry copies the deterministic package's records of the
  mode (see "Registry"), so an l2n call launches the deterministic programs,
  and on the tape backward possibly the tape WY twin (see "Tape WY twin"),
  which writes the WY program's `w` bit for bit.

## What changes: the `S1` backward pairs

The deterministic backward forms the gate gradient `dg` in two programs.  The
fused dq / dk / dg program (`dqkg` stage) reads `q` for one term only, the
`q * dq` term of `dg`.  The GLM intra program (`glm_intra` stage) adds the
remaining terms and forms the reverse chunk-local suffix.  In an `S1` pair:

- the fused program (`dqkg_*_v24`) builds the deterministic `_davq` kernel
  without the `q * dq` term and without its `q` loads (16 KiB per chunk and
  head; no `q_tma` parameter);
- the GLM intra program (`glm_intra_v21` on `sm_100a`, before the unchanged
  gate epilogue; `glm_intra_v22` on `sm_103a`; `glm_intra_v23` on `sm_107a`)
  adds the term with one more FMA from the FP32 `dq` it already holds.

The two programs of a pair must agree: an `S1` fused program before a
deterministic GLM intra program drops the term, and the reverse pairing adds
it twice.  `cake_jit.S1_RECORDS` lists the `S1` records, and the host plan
refuses a plan whose two records disagree on a path.

Dispatch ("saved" = the saved-set paths `light`, `light_states` and `tape`;
"replay" = `saved=None`).  The serial rows of 9 to 15 heads keep the
deterministic pair on every architecture and path.

| arch | saved | replay |
| --- | --- | --- |
| `sm_100a` | `H <= 8`: deterministic pair; `H >= 16`: `dqkg_davq_v24` + `glm_intra_v21` | `dqkg_int_davq_v24` + `glm_intra_v21` |
| `sm_103a` | `dqkg_davq_v24` + `glm_intra_v22` | `H <= 8`: `dqkg_int_davq_v24` + `glm_intra_v22`; `H >= 16`: deterministic pair |
| `sm_107a` (provisional) | `dqkg_davq_v24` (from 16 heads `dqkg_davq_nohint_v24`) + `glm_intra_v23` | `dqkg_int_davq_v24` (from 16 heads `dqkg_int_davq_nohint_v24`) + `glm_intra_v23` |

The `_nohint` forms are built with the L2 prefetch and evict-first hints off,
as the deterministic package uses on `sm_107a` from 16 heads.  The `sm_107a`
rows are provisional; the `cake_jit` docstring lists the edits that return
them to the deterministic pair.  If the `sm_107a` `S1` rows are rejected, the
deterministic copies `dqkg_davq` / `dqkg_int_davq` get the `sm_107a` rows
back, and the deterministic `dqkg_davq_nohint` / `dqkg_int_davq_nohint`
records ship again (their programs ship on `sm_107a` already, for the raw gate
mode).

The raw gate mode has no `S1` pair: its backward runs the deterministic
package's raw-mode pair on every row (`dqkg_raw`, `dqkg_int_raw`,
`dqkg_davq_raw`, `dqkg_int_davq_raw`, `dqkg_davq_nohint_raw`,
`dqkg_int_davq_nohint_raw` with `glm_intra_raw`, which adds no `q * dq` term;
see "Registry").  Its parameter gradients are the deterministic package's
too: `glm_intra_raw` writes per-chunk FP32 `(hi, lo)` partials of `dA_log` /
`ddt_bias` and `param_finalize` merges them in a fixed order (the global chunk
order), on every architecture.

The in-kernel q / k L2 norm has no `S1` pair either: no GLM intra program of
its modes (`glm_intra_l2n`, `glm_intra_raw_l2n` and the fallback route's
`glm_intra_l2n_capture` / `glm_intra_raw_l2n_capture`) adds the `q * dq` term.
Its backward runs the deterministic fused gradient on every row: in g mode the
`_l2n` records of the `dqkg` stage (`dqkg_l2n`, `dqkg_int_l2n`,
`dqkg_davq_l2n`, `dqkg_int_davq_l2n`, `dqkg_davq_nohint_l2n`,
`dqkg_int_davq_nohint_l2n`), in raw mode the raw records above.  Its raw
backward finishes the parameter gradients with `param_finalize_l2n`, the
deterministic `param_finalize`.

## Numerical contract

- `out` and the saved set are bitwise the deterministic package's (the same
  forward programs).
- `dq`, `dk`, `dv` and `dbeta` are bitwise the deterministic package's on every
  row: the `S1` programs compute them with the same instructions on the same
  operands.
- `dg` (explicit gate) has the same exact value and the same number and
  class of FP32 roundings; only their order changes, so `dg` differs from the deterministic
  package's by FP32 rounding.  The accuracy budget against the FP64 reference
  is the deterministic package's.
- Repeated calls on the same inputs and path give bitwise the same outputs
  (fixed order, no atomic adds).  The checkpoint replay and a saved-set
  backward take different pairs on the `sm_100a` rows up to 8 heads and the
  `sm_103a` rows from 16 heads, so their `dg` differ there by FP32 rounding;
  every other gradient stays bitwise the same.  A torch checkpoint re-runs the
  saving forward and takes the saved-set pair, so checkpointed and direct
  training agree bitwise.
- In the raw gate mode every output (`out`, the saved set, `dq`, `dk`, `dv`,
  `dbeta`, the raw gate's `dg`, `dA_log` and `ddt_bias`) is bitwise the
  deterministic package's raw mode on every path (saved set, replay, batched
  groups, arena options, autograd and `torch.utils.checkpoint`) and
  architecture: the same programs run in the same order.
- With the in-kernel q / k L2 norm, in either gate mode, every output (the
  outputs above, the raw gate mode's `dA_log` / `ddt_bias` included, and the
  saved `qk_rstd`) is bitwise the deterministic package's in the same mode,
  on the same paths and architectures: the same programs run in the same
  order (the raw gate mode's parameter reduce is the deterministic
  `param_finalize`, as `param_finalize_l2n`; the tape path's WY twin below
  writes the same `w`).

## Checkpoint hand-off

`cake_kda_nondet(..., checkpoint_handoff="auto")` applies under
`torch.utils.checkpoint` (reentrant or not).  The forward that torch re-runs
inside the backward also writes the `h` / `v_new` tapes and keeps the
forward's `w` and the segment transfers in a per-device record.  The node's
backward takes the record and runs the tape path from it: no recurrence
recompute, and no `wy` launch where the record carries `w`.  `out` and the
saved set do not change.

| `checkpoint_handoff` | engages |
| --- | --- |
| `"auto"` (default) | at the head counts of `cake_backend.HANDOFF_HEADS` (2, 4, 8, 12, 16, 32, 64) on the devices of `cake_backend.HANDOFF_VERIFIED` (architecture, SM count, Cake build toolkit): `sm_100a` and `sm_103a` with 148 or 152 SMs, `sm_107a` with 212 SMs, toolkit 13.4; on a problem of `S` tokens in `N` documents at `H` heads with `H * S >= cake_backend.HANDOFF_MIN_HEAD_TOKENS` (196608) and `S >= cake_backend.HANDOFF_MIN_DOC_TOKENS * N` (8192 tokens per document on average) |
| `"on"` | every call it supports (tests and benchmarks) |
| `"off"` | never |

- On those devices and head counts the hand-off suite measured the
  gradients bitwise equal to `"off"` on the tested rows: every plan family
  its packs reach, under both checkpoint modes.  Other head counts select
  record combinations the suite has not run.  At one head the step is
  host-bound and the hand-off gains nothing.
- The size bounds: a record adds a fixed host cost to each checkpointed
  call, and saves kernel time that grows with the token-heads and the
  document length (mostly the recurrence recompute, which walks each
  document's chunks in order).  On a smaller problem the step is host-bound,
  and the hand-off makes it slower.  Measured at 12 heads on B200 and B300
  (x86 hosts), checkpointed step det / ND: `S = 32768` in one or four
  documents and `S = 16384` in one document 1.04-1.14 (admitted);
  `S = 16384` in four documents and `S = 8192` in one or four documents
  0.92-0.98 with the hand-off on (refused now; about 0.98 at `S = 8192` in
  four documents with it off).  The other head counts and devices take the
  same bounds until rows of their own are measured.
- `N` counts every document of `cu_seqlens`, zero-length ones too (the
  check reads its shape only, with no device sync): a pack padded to a fixed
  document count with empty documents can fall under the mean-length bound
  although its real documents are long; pass `checkpoint_handoff="on"` there.
- Elsewhere `"auto"` runs as `"off"`, also when the toolkit cannot be read.
  A new device, toolkit or registry needs the suite on that device before it
  enters `HANDOFF_VERIFIED`.
- The keyword is the only switch: the package reads no environment variable.
- Row records: the registry maps each forward intra record to a twin
  (`cake_jit.HANDOFF_TWINS`: `fwd_intra_v19` for `fwd_intra_v14`,
  `fwd_intra_v20` for `fwd_intra`).  The twin is the same program and also
  writes every `gk` row and the internal beta.  A recompute that launches
  the twin keeps `gk`, `kg`, `beta_int`, and `kb` where it keeps no `w`, in
  a row record.  Its backward launches no prep, and one fill kernel zeroes
  the `dv2` pad rows.  A record without these rows is a tape record.
- The recompute asks for a row record only on the devices of
  `cake_backend.HANDOFF_ROWS_VERIFIED` (architecture, SM count, full toolkit
  version such as `13.4.92`), at the `HANDOFF_HEADS` head counts, for a call
  without a `workspace`, when the twin, the intra record and the light prep
  come from one build (all prebuilt, or all built by the JIT) and the twin
  loads.  The twin's rows equal the prep's only where one toolkit compiles
  both to the same instructions, so the key includes the patch version.
  The set holds `sm_100a` and `sm_103a` at 148 and 152 SMs and `sm_107a` at
  212 SMs with toolkit `13.4.92` (B200, GB200, B300, GB300, R200).  On each,
  the hand-off suite with row records, the twin-row test, the P8 64-head gate
  and the hand-off parity rows passed.  Elsewhere every record is a tape
  record.
- Memory: a tape record holds `memory_report(...)["handoff_bytes"]` from the
  recompute to its node's backward; a row record holds
  `["handoff_bytes_rows"]` (772 B more per internal token-head, and 256 B
  more where it keeps `kb`).  A recompute that asks for a tape record is
  admitted within two tape records of the call's shape
  (`["handoff_budget_bytes"]`); one that asks for a row record within the
  larger of that and one row record.  A row record that does not fit becomes
  a request for a tape record.  A tape record that does not fit first turns
  the live row records of the device into tape records (they drop their
  rows), and is refused if it still does not fit.  The backward that takes a record
  allocates only the layout it runs (`["arena_bwd_tape"]`,
  `["arena_bwd_handoff"]`: the row record's is 516 B per internal
  token-head smaller).  With tape records the measured peak of a
  checkpointed step rose about 4 % on the rows whose record allocates its
  own `w` (the forward launches a WY program and the light set carries no
  `w`), and did not change elsewhere.  A call without a `workspace`
  allocates its backward arena in the backward.  In a loop of checkpointed
  steps near the memory limit, other allocations can split the cached arena,
  and the backward then runs out of memory, with the hand-off on or off.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` avoids it.
- The record is stamped with the inputs, their versions, the plan and the
  stream.  Any mismatch, an arena option, `doc_lengths`, CUDA-graph capture,
  `torch.compile`, compiled autograd, a Python dispatch mode, the raw gate
  mode or the in-kernel q / k L2 norm runs the light path (those modes write
  and take no record, and `memory_report` reports no hand-off bytes for
  them);
  `save_mode="tape"` runs the tape path without the hand-off.
  Repeated misses turn the hand-off off on the device.

## Tape WY twin

The tape backward (a `save_mode="tape"` saved set, or a hand-off record
without the forward's `w`) takes `v_new` from the tape, so nothing it launches
reads the `u` that its WY record `wy_bwd` also writes.  The registry maps
`wy_bwd` to a twin (`cake_jit.TAPE_TWINS`: `wy_bwd_v9`, the same program
without the `u` half; it reads neither `v` nor `beta` and writes the same `w`).
No rule selects the twin.

- The tape backward launches the twin in place of `wy_bwd`, on its grid, on
  the devices of `cake_backend.TAPE_WY_VERIFIED` (architecture, SM count, full
  toolkit version), when the twin and `wy_bwd` come from one build (all
  prebuilt, or all built by the JIT) and the twin loads.  Elsewhere, and after
  a failed load, it launches `wy_bwd`.
- The set holds `sm_100a` at 148 and 152 SMs, `sm_103a` at 148 SMs and
  `sm_107a` at 212 SMs with toolkit `13.4.92` (B200, GB200, B300, R200).  On
  each, the hand-off suite, the tape-mode tests with the twin and the hand-off
  parity rows passed, bitwise.
- The launch count, the arena and the gradients do not change.

## Names

| | deterministic package | this package |
| --- | --- | --- |
| facade | `flashinfer.cake_kda_det_train` | `flashinfer.cake_kda_nondet_train` |
| backend package | `flashinfer.experimental.cake.cake_kda_det_train` | `flashinfer.experimental.cake.cake_kda_nondet_train` |
| compute entries | `cake_kda_det_train_forward`, `cake_kda_det_train_backward`, `cake_kda_det` | `cake_kda_nondet_train_forward`, `cake_kda_nondet_train_backward`, `cake_kda_nondet` |
| helpers | `cake_kda_det_train_arena`, `cake_kda_det_train_workspace(_bytes)`, `cake_kda_det_train_saved_bytes`, `is_cake_kda_det_train_supported` | `cake_kda_nondet_train_arena`, `cake_kda_nondet_train_workspace(_bytes)`, `cake_kda_nondet_train_saved_bytes`, `is_cake_kda_nondet_train_supported` |
| types | `CakeKDADetFunction`, `CakeKDADetChunkSaved`, `CakeKDADetWorkspace`, `CakeKDADetArena` | `CakeKDANondetFunction`, `CakeKDANondetChunkSaved`, `CakeKDANondetWorkspace`, `CakeKDANondetArena` |
| context-parallel facade | `flashinfer.cake_kda_det_train_kcp` | `flashinfer.cake_kda_nondet_train_kcp` |
| libraries | `cake_kda_det_train_<digest>` | `cake_kda_train_<digest>` |

No name of one package is an alias of the other's, and neither package takes
a mode argument.  The libraries are internal build artifacts and keep their
stems; the copies of unchanged programs keep their digest under this
package's prefix.

## Registry

118 records naming 105 programs.  The records of the
deterministic package are copied under this package's prefix, with these
changes:

| record | change |
| --- | --- |
| `dqkg_davq_v24`, `dqkg_int_davq_v24` | new: the saved-set / replay `S1` fused programs (all three architectures) |
| `dqkg_davq_nohint_v24`, `dqkg_int_davq_nohint_v24` | new: their `_nohint` forms (`sm_107a`) |
| `glm_intra_v21`, `glm_intra_v22`, `glm_intra_v23` | new: the `S1` GLM intra programs (`sm_100a`, `sm_103a`, `sm_107a`) |
| `dqkg_davq`, `dqkg_int_davq` | the same programs; rules narrowed to the deterministic-pair rows of the explicit gate (`dqkg_davq`: `sm_100a`; `dqkg_int_davq`: `sm_103a`); the programs are built on all three architectures for the raw class |
| `glm_intra`, `glm_intra_v14`, `glm_intra_v18` | the same programs; rules narrowed to the deterministic-pair rows |
| `dqkg_davq_nohint`, `dqkg_int_davq_nohint` | not shipped (`sm_107a` runs the `S1` pairs from 16 heads); their programs ship as `dqkg_davq_nohint_raw` / `dqkg_int_davq_nohint_raw` |
| `fwd_intra_raw`, `fwd_intra_v14_raw`, `bwd_prep_g8_raw`, `bwd_prep_g8_replay_raw`, `bwd_prep_g8_noqg_raw`, `bwd_prep_g8_noqg_nokb_raw`, `glm_intra_raw`, `param_finalize` | the raw gate mode's records, unchanged (all three architectures) |
| `dqkg_raw`, `dqkg_int_raw`, `dqkg_davq_raw`, `dqkg_int_davq_raw`, `dqkg_davq_nohint_raw`, `dqkg_int_davq_nohint_raw` | new: the raw gate mode's `dqkg` class (`gate: "raw"`), the deterministic package's `dqkg` records of the same names without the suffix and their rules, so the raw mode selects the deterministic fused gradient on every row (on the deterministic records' architectures: the `_nohint` forms on `sm_107a`, the others on all three) |
| `fwd_intra_l2n`, `fwd_intra_v14_l2n`, `fwd_intra_v37_l2n`, `bwd_prep_g8_l2n`, `bwd_prep_g8_replay_l2n`, `bwd_prep_g8_replay_saved_l2n`, `bwd_prep_g8_noqg_l2n`, `bwd_prep_g8_noqg_nokb_l2n`, `glm_intra_l2n`, `glm_intra_v30_l2n`, `glm_intra_l2n_capture`, `qk_l2norm`, `qk_l2norm_apply`, `qk_pullback`, `fwd_intra_raw_l2n`, `fwd_intra_v14_raw_l2n`, `bwd_prep_g8_raw_l2n`, `bwd_prep_g8_replay_raw_l2n`, `bwd_prep_g8_noqg_raw_l2n`, `bwd_prep_g8_noqg_nokb_raw_l2n`, `glm_intra_raw_l2n`, `glm_intra_raw_l2n_capture` | the in-kernel Q/K L2 norm's records, unchanged (`fwd_intra_v37_l2n` and `glm_intra_v30_l2n` on `sm_103a`, the saved-rstd twin `bwd_prep_g8_replay_saved_l2n` on `sm_100a` / `sm_103a`, the others on all three architectures; no path selects the twin) |
| `dqkg_l2n`, `dqkg_int_l2n`, `dqkg_davq_l2n`, `dqkg_int_davq_l2n`, `dqkg_davq_nohint_l2n`, `dqkg_int_davq_nohint_l2n` | new: the in-kernel Q/K L2 norm's `dqkg` records of the g class (`l2n: ["fused", "fallback"]`), the deterministic package's `dqkg` records of the same names without the suffix and their rules, so an l2n mode selects the deterministic pair on every row; each shares its program with the raw class record of the same build |
| `param_finalize_l2n` | new: the in-kernel Q/K L2 norm's `param_finalize` record of the raw class (`l2n: ["fused", "fallback"]`), the deterministic package's `param_finalize` record and its rule on all three architectures, so the raw gate mode's l2n backward finishes the deterministic GLM intra programs' partials as the deterministic package does; it shares its program with `param_finalize` |
| `fwd_intra_v19`, `fwd_intra_v20` | new: the checkpoint hand-off's intra twins of `fwd_intra_v14` and `fwd_intra` (all three architectures); no path selects them |
| `wy_bwd_v9` | new: the tape WY twin of `wy_bwd`, writing `w` alone (all three architectures); no path selects it |

The stages of `cake_jit.GATE_STAGES` select among the records of the call's
gate mode (a record's `gate`, `"g"` when absent), and the stages of
`cake_jit.L2N_STAGES` (the deterministic package's, `dqkg` and
`param_finalize`) among the records of the call's in-kernel Q/K L2 norm
mode (a record's `l2n`, `["off"]` when absent).  The copies of the plain
records keep the deterministic records' `l2n` modes; the `S1` records and
the hand-off twins serve `"off"` alone; the raw class's
`dqkg` records serve every mode, as does the tape WY twin (its stage `wy` is
no l2n stage).

`build.json` has 288 entries: 95 on `sm_100a`, 96 on `sm_103a` and 97 on
`sm_107a`.

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry, `S1_RECORDS`, `HANDOFF_TWINS`,
  `TAPE_TWINS`, the paths, the stage order, the optional stages and the
  segmented route's constants, the logical vocabulary, the loader glue and
  `run_module`.  The derived leaves `kernel_symbol`, `launch`, `arg_plan` and
  `logical` are rewritten by the export.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`).  `aot/` is never tracked (`.gitignore`).
* `cake_backend.py` -- operand checks, the host plan and its pairing check,
  the arena workspace, the saved set, the forward and backward launch
  sequences, the autograd `Function`, the checkpoint hand-off, the tape WY
  twin's rule and the byte queries.
* `csrc/` -- the generated device and binding translation units and
  `cake_ffi_checks.h`, identity-checked by the digests in `build.json`.

## Vendoring

The package imports only itself, `torch` and `tvm_ffi`. It imports
`flashinfer` only on the JIT path, for a library that is not prebuilt under
`aot/` (`cake_build.py`, and the checkpoint hand-off's toolkit probe in
`cake_backend.py`). The context-parallel package `cake_kda_nondet_train_kcp`
requires this package as a sibling directory.

## Status

Tests:

* `tests/experimental/test_cake_kda_nondet_train.py`: the backend suite against the
  FP64 reference of `tests/experimental/test_helpers/cake_kda_det_train_reference.py`,
  the comparison with the deterministic package, the hand-off twins'
  records and rows, and the tape WY twin's record, rule and tape backward.
* `tests/experimental/test_cake_kda_nondet_train_l2n.py`: the in-kernel q / k
  L2 norm against the deterministic package's.
* `tests/experimental/test_cake_kda_nondet_train_api.py`: the public entry
  points, their names, the refusal of the deterministic package's objects and
  `torch.use_deterministic_algorithms(True)`.
* `tests/experimental/test_cake_kda_nondet_train_handoff.py`: the checkpoint
  hand-off with tape and row records, with and without the tape WY twin.

Benchmark: `benchmarks/bench_cake_kda_nondet_train.py`.  The context-parallel path
has its own package, `flashinfer/experimental/cake/cake_kda_nondet_train_kcp`
(facade `flashinfer.cake_kda_nondet_train_kcp`).
