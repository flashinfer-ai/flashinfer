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
  is follow from `S`, `num_docs`, `H` and the device's SM count alone.
* Repeated calls on the same inputs give bitwise the same outputs: the kernels
  reduce in a fixed order and use no atomic adds.  The backward from the saved
  set and the checkpoint replay (`saved=None`) give bitwise the same gradients.
* Limits: `H * 128 * 64 < 2**31` (`H <= 262143`; the public contract covers
  `H` in 1..64); `S >= 1`, `num_docs >= 1`.  Any `S`, including `S < 64` and
  document lengths that are not multiples of 64.

### Saved set (`KdaChunkSaved`)

`chunk_kda_fwd(..., save_for_backward=True)` returns a frozen dataclass:

| field | content |
|---|---|
| `version` | `1` (`save_mode="light"`, the default) or `2` (`save_mode="tape"`) |
| `Aqk`, `Akk` | BF16 `[S, H, 64]` on the token rows: row `t` holds the 64 chunk-local columns of its chunk (zeros above the diagonal and past the chunk end) |
| `tables` | `(chunk_bos, chunk_len, seq_chunk_start, seq_order[, seg_chunk_start])`: int32 views of one device buffer (`NC_pad`, `NC_pad`, `num_docs + 1`, `num_docs` and, on the segmented route, `NSEG + 1` entries), or `None` (the backward rebuilds them from `cu_seqlens`) |
| `S`, `H`, `num_docs`, `scale` | the problem the set belongs to |
| `h`, `v_new` | version 2 only: the forward's state tape BF16 `[NC_pad, H, 128, 128]` and `v_new` BF16 `[R, H, 128]` on the internal layout |
| `seg_m` | version 2 on the segmented route: the segment transfers FP32 `[NSEG, H, 128, 128]` the backward's reverse carry reads; `None` on the serial route |

`kda_train_saved_bytes(S, H) = 256 * S * H` bytes (`Aqk` and `Akk`);
`kda_train_saved_bytes(S, H, num_docs, save_mode=...)` adds the tables and,
for `"tape"`, the version-2 tensors (768 B more per internal token-head plus
the segment transfers).  Every tensor of the set is allocated by the call that
writes it: a saved set can go to a backward on any workspace of the same
problem.  A caller may keep only `Aqk` / `Akk` (`tables=None`).  With
`save_for_backward=False` the token-row `Aqk` / `Akk` are not written and the
backward must replay (`saved=None`: table, prep and intra recompute them).
From a version-2 set the backward skips its recurrence recompute (WY still
runs for `w`); from a version-1 set or on the replay path it recomputes `h`
and `v_new` in its workspace.

### Workspace (`KdaWorkspace`)

A workspace is one arena (a `uint8` tensor) holding the internal-layout
intermediates of one `(S, num_docs, H)` problem; `chunk_kda_fwd` /
`chunk_kda_bwd` allocate only their outputs and the saved set when given one,
and without one each call allocates a workspace of its own.  Buffers whose
lifetimes within one call are disjoint share a slot: the lifetime tables
`cake_backend.FWD_SLOTS` / `BWD_SLOTS` (and the `*_TAPE_SLOTS` of the tape
layouts) list every occupant with its `[first write, last read]` launch step,
and `cake_backend.lifetime_table(mode)` renders them.
`kda_train_workspace_bytes(S, num_docs, H, backward=True, save_mode="light",
sm_count=None)` is the allocation (`sm_count` enters through the segmented
route's segment count; the current device's when `None`);
`cake_backend.memory_report(S, num_docs, H, save_mode)` gives the arena, the
saved set, the outputs and the per-boundary totals.  For the contract's
largest row (`S = 261393`, 4 documents, `H = 64`) the backward arena is 82 GiB.

Outputs and the saved set are fresh tensors of every call; nothing is kept in
the arena between calls (the pad rows every consumer reads are written in the
same call by their producer).  `backward=False` sizes the arena for the
forward layout only; `save_mode="tape"` sizes it for the tape layouts (smaller
arenas: `h` and `v_new` live in the saved set) and makes `"tape"` the default
of the calls on it.

## Kernel structure of one training step

Forward (six launches; nine on the segmented route): `table` (chunk tables
from `cu_seqlens`, one CTA) and, on the segmented route, `seg_table` (the
segment table, one CTA), `fwd_prep` (`bulk` at >= 256 chunk-head CTAs, else
`split` with two CTAs per chunk: the gated keys / values of the GLM contract on
the chunk layout), `fwd_intra` (the intra-chunk `Aqk` / `Akk` and, with
`save_for_backward`, their token-row copies), `wy` (the WY representation `u`,
`w`; one CTA per chunk pair), the state recurrence `fwdh` and `fwd_o` (the
output; persistent grid of `min(NC_pad * H, sm_count)` CTAs).

The state recurrence has two routes, chosen by the head count alone.  With
`H > 8` the serial walk over each document's chunk chain runs: `fwdh_v2` with
four CTAs per document and head when `num_docs * H * 4 <= 4 * sm_count`, else
`fwdh_v1` with two.  With `H <= 8` the segmented route runs: every document is
cut into `P = clamp(ceil(4 * sm_count / (num_docs * H)), 1, 64)` chunk-aligned
segments (`NSEG = num_docs * P` pseudo-sequences, `seg_chunk_start`), `fwdh_v3`
pass 1 walks every segment from zero and from the identity (grid
`2 * NSEG * H`: the segment offsets and transfers, FP32 `[128, 128]` each),
`seg_carry` chains them within each document (one CTA per document and head;
TF32 tensor-core products, FP32 accumulation) into the carried initial state of
every segment, and `fwdh_v3` pass 2 walks from those states (grid `NSEG * H`)
writing the `h` / `v_new` tapes.

Backward from a version-1 saved set (eight launches; twelve on the segmented
route): `bwd_prep` (`g8` at >= 256 CTAs, `g8s` with two CTAs per chunk at
128..255, else `v2`; copies the saved `Aqk` / `Akk` onto the chunk layout),
`wy`, `fwdh` (as above), `dav` (`dA_qk` and the first part of `dv`; one CTA per
chunk pair), the adjoint recurrence `dhu` (the state gradient `dh` and `dv2`:
`dhu_v2` with four CTAs per document and head when `num_docs * H * 4 <=
sm_count`, else `dhu_v1`, for `H > 8`; for `H <= 8` `dhu_v3` pass 1 in reverse
from zero, `seg_carry` in reverse with the forward's transfers, `dhu_v3` pass 2
from the carried adjoint states), `dqkg` (fused dq / dk / dg / dbeta partials
and `dA_kk`; persistent grid of `min(NC_pad * H, sm_count)` CTAs), `glm_intra`
(the intra-chunk terms and the GLM gate chain) and `gate_epilogue` (the
token-row `dg` and `dbeta`).  One launch more (two on the segmented route) when
the saved set carries no tables; from a version-2 saved set `fwdh` is skipped
(seven / nine launches); on the checkpoint replay path (`saved=None`) `table`
(+ `seg_table`), `bwd_prep` with `copy_saved = 0` and `fwd_intra` precede the
rest (ten / fifteen launches).

The selection and grid rules are data in `cake_jit.MODULES` (`select_rule`,
`grid_rule`; `cake_jit.SEGMENTS` holds the segmented route's constants) and
are evaluated by `cake_jit.select_record` / `launch_grid` / `segment_plan`;
`cake_backend.plan(S, num_docs, H, sm_count)` resolves them once per problem,
and `Plan.launch_count(boundary)` is the launch count of each boundary.

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry (one record per program: its
  `library` stem in `build.json`, the stage it fills and its variant, the
  selection and grid rules, the kernel-parameter -> logical-argument map and
  the argument plan of its `run` entry; the derived leaves `kernel_symbol`,
  `launch`, `arg_plan` and `logical` are rewritten by the export), the stage
  order, the optional stages and the segmented route's constants, the logical
  vocabulary, the loader glue (`library_name`, `load`, `specs_for_arch`) and
  `run_module(name, values, grid=, device_index=)`.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`): one entry per `<library>_<arch>` (21 programs x 3
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
  and binding translation units of the 21 programs, shared by the three
  architectures (per-architecture lines under `#if __CUDA_ARCH__`) and
  compiled once per architecture.  The sources are identity-checked by the
  digests in `build.json`.

## Status

The registry holds the 21 programs for `sm_100a`, `sm_103a` and `sm_107a`,
exported from the kernel snapshot named in the pull request.

Tests: `tests/experimental/test_cake_kda_train.py` (the backend suite: imports
only this package through `tests/test_helpers/cake_package`; the registry and
plan tests run everywhere, the kernel tests skip without a compute capability
10.0 / 10.3 / 10.7 device) against the exact FP64 per-document recurrence of
`tests/test_helpers/cake_kda_train_reference.py`, and
`tests/experimental/test_cake_kda_train_api.py` (the public entry points).
Benchmark: `benchmarks/bench_cake_kda_train.py`.
