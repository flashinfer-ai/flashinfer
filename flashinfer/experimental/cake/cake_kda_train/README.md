# Cake packed KDA chunked training, non-deterministic package (SM100 / SM103 / SM107)

Experimental backend for the packed Kimi Delta Attention (KDA) chunked
training forward and backward in the GLM-5.3-Flash geometry, the
`deterministic=False` counterpart of the deterministic package
`flashinfer.experimental.cake_kda_det_train`.  Both the API and the backend
are experimental: no compatibility guarantee.  The feature does not
participate in automatic backend selection, autotuning or trace-apply.

The operands, shapes, dtypes, saved set (versions 4 and 2), workspace, arena
options, document groups (`batch_tokens`), launch sequences and byte queries
are the deterministic package's; its README
(`flashinfer/experimental/cake_kda_det_train/README.md`) describes them.  This
package imports nothing from it and ships its own copies of every program.
This README describes what differs, including the checkpoint hand-off of
`chunk_kda`, which the deterministic package does not have.

## Entry points and numerical mode

The public entry points are the facade functions of `flashinfer.cake_kda_train`:

```python
from flashinfer.cake_kda_train import chunk_kda, chunk_kda_fwd, chunk_kda_bwd, kda_train_workspace

out = chunk_kda(q, k, v, g, beta, cu_seqlens, deterministic=False)                  # autograd
ws = kda_train_workspace(S, num_docs, H, device, deterministic=False)               # caller-owned
out, saved = chunk_kda_fwd(q, k, v, g, beta, cu_seqlens, workspace=ws, deterministic=False)
dq, dk, dv, dbeta, dg = chunk_kda_bwd(q, k, v, g, beta, cu_seqlens, dout, saved=saved,
                                      workspace=ws, deterministic=False)
```

Every facade function (`chunk_kda_fwd`, `chunk_kda_bwd`, `chunk_kda`,
`kda_train_arena`, `kda_train_workspace`, `kda_train_workspace_bytes`,
`kda_train_saved_bytes`, `is_kda_train_supported`) takes
`deterministic: bool = True` and raises `TypeError` for a value that is not a
`bool`:

| `deterministic` | `torch.are_deterministic_algorithms_enabled()` | runs |
| --- | --- | --- |
| `True` (default) | any | the deterministic package, unchanged bits |
| `False` | `False` | this package |
| `False` | `True` | the deterministic package, one `UserWarning` per process |

This package's `chunk_kda_fwd`, `chunk_kda_bwd` and `chunk_kda`, its
workspace `KdaWorkspace` (the constructor, `for_problem` and `for_budget`) and
its autograd node `ChunkKDAFunction.apply` (a trailing argument) take
`deterministic: bool = True` too and raise `ValueError` unless it is `False`:
a call that only swaps the import path of the deterministic package fails
instead of changing mode.

`ChunkKDAFunction.apply` takes `checkpoint_handoff` (`None`, `"auto"`, `"on"`
or `"off"`; see below) as the positional argument after `doc_lengths`, before
`deterministic`.  This changes the positional signature: an `apply` written
for the previous one passes its `bool` in that slot and raises `TypeError`.

- The mode of `chunk_kda` is fixed at its forward call: the autograd node's
  backward is the forward's package.
- Both packages write the saved set in the same format (the forward programs
  are the same), so either backward reads it.  This package's saved set
  carries `deterministic=False`.
- A workspace serves its own package.  This package refuses another
  package's workspace (`TypeError`), and a `deterministic=True` facade call
  refuses a `deterministic=False` workspace.  When torch's deterministic
  algorithms turn a `deterministic=False` call with such a workspace
  deterministic, the call runs on a workspace of its own with the same arena
  option.  The facade accepts an arena option of either package in both
  modes.
- The schedule below has no unordered reduction, so this package raises no
  PyTorch non-determinism alert.
- This package runs the explicit FP32 gate without the in-kernel q / k L2
  norm.  The raw gate mode (`use_gate_in_kernel=True` with `A_log`,
  `dt_bias` and `lower_bound`, FlashInfer #7816) runs on the deterministic
  package only: the facade forwards those keywords to it with
  `deterministic=True`, and this package's entry points raise `ValueError`
  for them (`KdaWorkspace.for_problem` and `workspace_bytes` for
  `gate="raw"`).  The off values (`use_gate_in_kernel=False`, the others
  `None`) are accepted, so a call that spells them out runs unchanged.  The
  saved set carries the deterministic package's `gate` / `lower_bound`
  fields (`"g"` / `None` here); the backward refuses a set of the raw gate
  mode or of the in-kernel L2 norm (#7817).

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
deterministic copies `dqkg_davq` / `dqkg_int_davq` get the `sm_107a`
architecture back, and the deterministic `dqkg_davq_nohint` /
`dqkg_int_davq_nohint` records ship again.

## Numerical contract

- `out` and the saved set are bitwise the deterministic package's (the same
  forward programs).
- `dq`, `dk`, `dv` and `dbeta` are bitwise the deterministic package's on every
  row: the `S1` programs compute them with the same instructions on the same
  operands.
- `dg` has the same exact value and the same number and class of FP32
  roundings; only their order changes, so `dg` differs from the deterministic
  package's by FP32 rounding.  The accuracy budget against the FP64 reference
  is the deterministic package's.
- Repeated calls on the same inputs and path give bitwise the same outputs
  (fixed order, no atomic adds).  The checkpoint replay and a saved-set
  backward take different pairs on the `sm_100a` rows up to 8 heads and the
  `sm_103a` rows from 16 heads, so their `dg` differ there by FP32 rounding;
  every other gradient stays bitwise the same.  A torch checkpoint re-runs the
  saving forward and takes the saved-set pair, so checkpointed and direct
  training agree bitwise.

## Checkpoint hand-off

`chunk_kda(..., checkpoint_handoff="auto")` applies under
`torch.utils.checkpoint` (reentrant or not).  The forward that torch re-runs
inside the backward also writes the `h` / `v_new` tapes and keeps the
forward's `w` and the segment transfers in a per-device record.  The node's
backward takes the record and runs the tape path from it: no recurrence
recompute, and no `wy` launch where the record carries `w`.  `out` and the
saved set do not change.

| `checkpoint_handoff` | engages |
| --- | --- |
| `"auto"` (default) | at the head counts of `cake_backend.HANDOFF_HEADS` (2, 4, 8, 12, 16, 32, 64) on the devices of `cake_backend.HANDOFF_VERIFIED` (architecture, SM count, Cake build toolkit): `sm_100a` and `sm_103a` with 148 or 152 SMs, `sm_107a` with 212 SMs, toolkit 13.4 |
| `"on"` | every call it supports (tests and benchmarks) |
| `"off"` | never |

- On those devices and head counts the hand-off suite measured the
  gradients bitwise equal to `"off"` on the tested rows: every plan family
  its packs reach, under both checkpoint modes.  Other head counts select
  record combinations the suite has not run.  At one head the step is
  host-bound and the hand-off gains nothing.
- Elsewhere `"auto"` runs as `"off"`, also when the toolkit cannot be read.
  A new device, toolkit or registry needs the suite on that device before it
  enters `HANDOFF_VERIFIED`.
- The environment variable `CAKE_KDA_CKPT_HANDOFF` (`off` / `on`) is read by
  `chunk_kda` at each call, not by `ChunkKDAFunction.apply`.  `off` turns
  every call off; `on` turns `"auto"` into `"on"`.  An explicit `"off"` stays
  off.
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
  `torch.compile`, compiled autograd or a Python dispatch mode runs the light
  path; `save_mode="tape"` runs the tape path without the hand-off.
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

| deterministic package (unchanged) | this package |
| --- | --- |
| `flashinfer.cake_kda_det_train` (facade) | `flashinfer.cake_kda_train` (facade, `deterministic=...`) |
| `flashinfer.experimental.cake_kda_det_train` | `flashinfer.experimental.cake.cake_kda_train` |
| `chunk_kda_det_fwd`, `chunk_kda_det_bwd`, `chunk_kda_det` | `chunk_kda_fwd`, `chunk_kda_bwd`, `chunk_kda` |
| `kda_det_train_arena`, `kda_det_train_workspace(_bytes)`, `kda_det_train_saved_bytes`, `is_kda_det_train_supported` | `kda_train_arena`, `kda_train_workspace(_bytes)`, `kda_train_saved_bytes`, `is_kda_train_supported` |
| `ChunkKDADetFunction`, `KdaDetChunkSaved`, `KdaDetWorkspace`, `KdaDetArena` | `ChunkKDAFunction`, `KdaChunkSaved`, `KdaWorkspace`, `KdaArena` |
| libraries `cake_kda_det_train_<digest>` | `cake_kda_train_<digest>` |

The previous names keep their deterministic meaning: `flashinfer.kda_train`
(`chunk_kda`, `chunk_kda_fwd`, `chunk_kda_bwd`, `kda_train_*`,
`is_kda_train_supported`) and the compatibility package
`flashinfer.experimental.cake_kda_train` bind the deterministic objects and
take no `deterministic` keyword.  The copies of unchanged programs keep their
digest under this package's prefix.

## Registry

75 records naming 73 programs from 46 kernels.  The records of the
deterministic package are copied under this package's prefix, with these
changes:

| record | change |
| --- | --- |
| `dqkg_davq_v24`, `dqkg_int_davq_v24` | new: the saved-set / replay `S1` fused programs (all three architectures) |
| `dqkg_davq_nohint_v24`, `dqkg_int_davq_nohint_v24` | new: their `_nohint` forms (`sm_107a`) |
| `glm_intra_v21`, `glm_intra_v22`, `glm_intra_v23` | new: the `S1` GLM intra programs (`sm_100a`, `sm_103a`, `sm_107a`) |
| `dqkg_davq`, `dqkg_int_davq` | the same programs; rules narrowed to the deterministic-pair rows, built only where those rows launch them (`dqkg_davq`: `sm_100a`; `dqkg_int_davq`: `sm_103a`) |
| `glm_intra`, `glm_intra_v14`, `glm_intra_v18` | the same programs; rules narrowed to the deterministic-pair rows |
| `dqkg_davq_nohint`, `dqkg_int_davq_nohint` | not shipped (`sm_107a` runs the `S1` pairs from 16 heads) |
| `fwd_intra_v19`, `fwd_intra_v20` | new: the checkpoint hand-off's intra twins of `fwd_intra_v14` and `fwd_intra` (all three architectures); no path selects them |
| `wy_bwd_v9` | new: the tape WY twin of `wy_bwd`, writing `w` alone (all three architectures); no path selects it |

`build.json` has 197 entries: 66 on `sm_100a`, 65 on `sm_103a` and 66 on
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

## Status

Tests:

* `tests/experimental/test_cake_kda_train.py`: the backend suite against the
  FP64 reference of `tests/test_helpers/cake_kda_det_train_reference.py`,
  the comparison with the deterministic package, the hand-off twins'
  records and rows, and the tape WY twin's record, rule and tape backward.
* `tests/experimental/test_cake_kda_train_api.py`: the facade, the mode
  keyword and the dispatch.
* `tests/experimental/test_cake_kda_train_handoff.py`: the checkpoint
  hand-off with tape and row records, with and without the tape WY twin.

Benchmark: `benchmarks/bench_cake_kda_train.py`.  The context-parallel path
is not part of this package.
