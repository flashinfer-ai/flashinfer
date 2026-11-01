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
This README describes what differs.

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

72 records naming 70 programs from 43 kernels.  The records of the
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

`build.json` has 188 entries: 63 on `sm_100a`, 62 on `sm_103a` and 63 on
`sm_107a`.

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry, `S1_RECORDS`, the paths, the stage
  order, the optional stages and the segmented route's constants, the logical
  vocabulary, the loader glue and `run_module`.  The derived leaves
  `kernel_symbol`, `launch`, `arg_plan` and `logical` are rewritten by the
  export.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`).  `aot/` is never tracked (`.gitignore`).
* `cake_backend.py` -- operand checks, the host plan and its pairing check,
  the arena workspace, the saved set, the forward and backward launch
  sequences, the autograd `Function` and the byte queries.
* `csrc/` -- the generated device and binding translation units and
  `cake_ffi_checks.h`, identity-checked by the digests in `build.json`.

## Status

Tests: `tests/experimental/test_cake_kda_train.py` (the backend suite against
the FP64 reference of `tests/test_helpers/cake_kda_det_train_reference.py`,
and the comparison with the deterministic package) and
`tests/experimental/test_cake_kda_train_api.py` (the facade, the mode keyword
and the dispatch).  Benchmark: `benchmarks/bench_cake_kda_train.py`.  The
context-parallel path is not part of this package.
