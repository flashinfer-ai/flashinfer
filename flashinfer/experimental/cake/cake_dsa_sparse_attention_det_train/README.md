# Cake DSA sparse-attention training backend, deterministic (SM100 / SM103 / SM107)

Backend package of `flashinfer.cake_dsa_sparse_attention_det_train`, the
deterministic package of the native 64-query-head DeepSeek Sparse Attention
training kernels (top-k sparse MLA with absorbed queries; 64 heads, 512 latent +
64 rope query/key dimensions, 512 value dimensions).  Tracking issue:
flashinfer-ai/flashinfer#5657.

The entry points of `flashinfer.cake_dsa_sparse_attention_det_train` are stable
APIs; this backend package is experimental (no compatibility guarantee for its
own helpers).  The libraries compile for `sm_100a`, `sm_103a` and `sm_107a`
from the same sources.  The feature does not participate in automatic backend
selection, autotuning or trace-apply.  Its libraries are loaded prebuilt from
`aot/` when the package ships them and JIT-built by FlashInfer from
`build.json` otherwise (see "Layout of this package").

## Mode and names (flashinfer-ai/flashinfer#8928)

The module name states the mode and no entry point takes a `deterministic=`
keyword.  This is the `det` package: every output repeats bit for bit for the
same inputs, device architecture and package version.  Bits are not promised
across package versions, across architectures, or against the
non-deterministic package `flashinfer.cake_dsa_sparse_attention_nondet_train`.
The result does not depend on the SM count, the launch grid, other work on the
device (side-stream kernels), CUDA-graph or eager execution, activation
checkpoint recompute or the saved set, repeated destination rows, or the order
of the tokens.  Nothing raises or warns under
`torch.use_deterministic_algorithms(True)` (no alert is registered), and no
entry falls back to another package.

| entry | autograd Function |
|---|---|
| `cake_dsa_sparse_attention_det` | `CakeDSASparseAttentionDetFunction` |
| `cake_dsa_sparse_attention_det_varlen` | `CakeDSASparseAttentionDetFunction` (over the compacted rows) |
| `cake_dsa_sparse_attention_det_nope` | `CakeDSASparseAttentionDetNopeFunction` |
| `cake_dsa_sparse_attention_det_nope_varlen` | `CakeDSASparseAttentionDetNopeVarlenFunction` |

The entries take the arguments of the non-deterministic entries.
`key_passes` and `dkv_direct` are accepted and validated as there and have no
effect: the backward's main stage always runs once over all keys, and the
accumulation into `dkv_acc` has one form.  The facade reads
`FLASHINFER_CAKE_DSA_CHECK_DST_MAP` and
`FLASHINFER_CAKE_DSA_TRAIN_BINDING_CACHE_CAPACITY` (as the non-deterministic
facade does, for this package's own cache); the backend reads no environment
variable.

### What repeats bit for bit

| output | how it is formed |
|---|---|
| `out`, `lse`, `o_lo` | one fixed-order forward kernel per token |
| derived `topk_length` (no caller lengths) | written by the forward kernel (last valid slot + 1) |
| varlen global rows and lengths | one index-preparation launch (or the forward's own conversion), fixed per row |
| `dq_latent`, `dq_rope` (NoPE `dq`) | summed in a fixed tile order, written once per row |
| `dkv_latent`, `dk_rope` (BF16 or `dkv_fp32`), NoPE `dkv` | integer sums on a per-row grid, one rounding into the output |
| `dkv_acc` columns `0:576` (NoPE `0:512`), with or without `dkv_dst_map` | integer sums on a per-row grid, then one FP32 addition per element |

## dK / dV reduction order

The backward launches every registered stage of
`cake_jit.BACKWARD_STAGES_OF[program]` on every call, in this order (the NoPE
program prefixes the names with `nope_`); no stage is chosen by a tensor value:

1. `bwd_delta`: `delta = rowsum(dout * (out + o_lo))` `[T, 64]` FP32 (the
   non-deterministic package's bits), the per-query bound terms `qstat [T, 4]`
   (largest magnitudes and norms of `dout`, `delta` and the query parts,
   summed over the heads in a fixed order), and the zero fill of the global
   words.
2. `bwd_norm`: the key norms `||kv_latent[k]||` and their maximum; it zeroes
   the row-bound words.
3. `bwd_bound`: per destination row, a fixed-point upper bound `B_row` on the
   row's `sum |dK|`, `sum |dV|` over every valid (query, key) pair that maps to
   it (latent and rope parts separately), and the row's pair count `N`.  While
   `T * topk < 2 ** 26`, one reduction per pair adds the pair count and a
   12-bit term to one word, and a pair whose 12-bit term is small adds its
   full-resolution term to a second word, so `B_row` keeps the full resolution
   (global word 10 records the packed launch for `bwd_grid`); a larger launch
   uses one word per part and a count word.
4. `bwd_grid`: the row's grid exponent `e = ceil(log2 B_row) - 30 - L` with
   `L = 12` (`cake_backend.GRID_SHIFT`) and the overflow counters' skip
   threshold `2 ** (30 - ceil(log2 N))`; it zeroes the accumulator row and
   counters of every row a pair touches.
5. `bwd_main`: dQ, and each pair's contribution `x` added to the int32
   accumulator of its destination row as `n = rni(x * 2 ** -e)`
   (round-to-nearest-even; the scaling by a power of two is exact).  Integer
   addition is associative, so the sum is the same bits in any order.  Per row
   four overflow counters bound `sum |n|`; a row whose counters stay below
   `2 ** 31` is proven to have left the int32 range in no order.
6. `bwd_redo_prep` / `bwd_redo`: a row whose counters do not prove the range
   moves to a grid at least `L + 1` bits coarser
   (`cake_backend.MIN_BUMP`), is zeroed and is recomputed by the main
   kernel's redo build, which visits every token and adds only the pairs of the
   flagged rows (no dQ).  Both stages are launched on every backward; the redo
   runs one persistent CTA per SM (`cake_backend.redo_launch_grid`) and
   returns at once when no row is flagged (no host synchronization).
7. `bwd_fold`: `v = float(n) * 2 ** e` (one rounding) into the fresh outputs
   (every row; a row no pair touches is `+0.0`), or `dkv_acc[r] = prev + v`
   (one IEEE FP32 addition with subnormals kept; an element whose integer is
   zero keeps its bits, so a planted `-0.0` stays `-0.0`; a row no pair touches
   is not written).

Repeated destination rows of `dkv_dst_map` add into the same integers, so they
combine exactly before the one FP32 addition.

Non-finite values: a pair whose contribution is NaN or infinite has no integer
image, so the counters flag its destination row, the redo flags it again, and
the fold writes every element of that row as NaN (fresh outputs and `dkv_acc`
alike); the other rows are unaffected.  The rows written as NaN:

| non-finite input | dK/dV rows written as NaN |
|---|---|
| `q_latent`, `q_rope`, `dout` or `out` of token `t` | every row token `t` selects |
| `kv_latent[k]` or `k_rope[k]` | the destination row of key `k` only |
| `lse[t]` = -inf or NaN | every row token `t` selects |
| `lse[t]` = +inf | none (the probabilities are 0) |

A finite input large enough (about `1e38`) to overflow an FP32 bound term
counts as non-finite.  Where the non-deterministic package writes NaN or
infinities element by element, this package writes NaN across the whole row.
`dq` follows the main kernel's arithmetic (a NaN input gives NaN in the rows
it reaches).

Known limitation: the redo does not compact the pairs of the flagged rows.
When any row is flagged, the redo costs about one more main-kernel pass over
all pairs (without dQ).

## Storage and the `o_lo` rule

The accumulator region of one backward holds, per destination row, the int32
accumulator (576 columns; 512 for NoPE; the row-bound words live in the row
until the grid stage reads them), 32 B of overflow counters, the grid words
(8 B; 4 B for NoPE) and the pair count (4 B).  `delta` and the global words
are never part of the region.  The rest lives in output storage that is dead
while it is used:

* `qstat` (`[T, 4]` FP32) and the key norms (4 B per key row) at the front of
  dQ (`dq_latent`; the NoPE `dq`; 64 KiB per query token) when they fit (`S`
  up to about `16380 T`) and dQ starts 16-byte aligned: `bwd_delta` and
  `bwd_norm` write them, `bwd_norm` and `bwd_bound` read them, and then the
  main stage writes every dQ element (`cake_backend.dq_scratch_layout`).
* With fresh rope dK/dV outputs, the redo mark and the redo list slot (4 B
  each per row) at the front of `dkv_latent`: `bwd_redo_prep` writes them,
  `bwd_redo` reads them, and the fold writes the output last
  (`cake_backend.redo_lists_fit`).  Otherwise (`dkv_acc`, NoPE) they are part
  of the region.
* After them, when they fit too, `delta` (`[T, 64]` FP32) and the global
  words: `bwd_delta` writes them (it reads neither output), the redo is their
  last reader (`cake_backend.output_scratch_layout`,
  `cake_backend.delta_fits_output`: `2 * align(R * 4) + align(T * 256) + 256`
  bytes within `dkv_latent`).
* Otherwise those parts stay in the workspace or the region.  The eager and
  autograd entries allocate per call only what stays there of `delta`, the
  global words and the region (no `topk_length` vector).

Peak memory against the non-deterministic package (same binding, allocated
bytes).  A prepared rope runner with fresh outputs holds the int32
accumulator where that package holds its FP32 accumulator (2304 B per row)
plus the counters, grid words and pair counts (44 B per destination row), and
no `delta`: it needs `align(R * 32) + align(R * 8) + align(R * 4) -
align(T * 256)` bytes more, which is less whenever `R <= ~5.8 T`.  A prepared
NoPE runner needs 48 B per destination row and the global words (one 256-byte
part) more: its fresh `dkv` is the integer accumulator through the fold, the
main stage writes dQ before the redo, and the redo reads `delta`, so no output
storage is dead while the counters, grid words, pair counts and redo lists
are live.  The NoPE autograd entry that adds into `dkv_acc` carves the region
from `o_lo` and allocates the global words (64 B) beside `delta` per call;
the global words may not alias `o_lo` (`bwd_delta` zeroes them while it reads
`o_lo`).

* The autograd entries use the storage of their saved `o_lo` (64 KiB per query
  token) for the region when it fits: `bwd_delta` is the only stage that reads
  `o_lo` (no later kernel reads `out` or `o_lo`).  The exact condition
  (`cake_backend.carve_fits`, 256-byte aligned parts) with every part in the
  region is
  `align(R * 2304) + align(R * 32) + align(R * 8) + 3 * align(R * 4) + align(S * 4) <= 65536 * T`
  for the rope program (`R` destination rows, `S` keys, `T` query tokens);
  for NoPE `align(R * 2048) + align(R * 32) + 4 * align(R * 4) +
  align(S * 4) <= 65536 * T`; the key norms and the redo lists drop out when
  they live in dQ or the output.  At `S = R` this is about `R <= 27.7 T`
  (rope) and `R <= 31.2 T` (NoPE).  A fresh NoPE `dkv` is its own integer
  accumulator (the fold rewrites it as FP32 in place), so only the per-row
  words go to the region.  A larger region is allocated per call.  Rows with
  more destination rows per query token than that therefore use more memory
  than the non-deterministic package, which adds into `dkv_acc` directly; for
  example a 4096-query backward chunk over 268757 keys mapped to 260611
  destination rows (a region of about 614 MB), a 2123-query segment over 67923
  keys (about 160 MB) and a NoPE 4096-query chunk over 269564 keys into
  `dkv_acc` (about 565 MB).
* After such a backward, `o_lo` no longer holds the forward's residual.  A
  second backward through the same graph (`retain_graph=True`) first runs the
  forward again into the saved `out`, `lse` and `o_lo` (same kernel, same
  operands: the same bits; derived lengths are derived again into a temporary).
  Under non-reentrant activation checkpointing the backward receives a
  recomputed `o_lo`.  A CUDA graph that captures forward and backward together
  replays correctly; a graph that holds only the backward and is replayed twice
  without a forward replay in between is not supported.
* The eager `backward` / `backward_nope` and the prepared runners take
  `o_lo_scratch` (default `False`: the region is allocated, or held in the
  runner's workspace; `dsa_train_workspace_size` / `dsa_nope_train_workspace_size`
  count it).  With `o_lo_scratch=True` a runner's `backward()` must follow its
  `forward()`; a second `backward()` raises outside graph capture.
* The kernels do not range-check `dkv_dst_map`; a value outside `[0, S_dst)`
  writes outside the region.  `check_dst_map=True` (or
  `FLASHINFER_CAKE_DSA_CHECK_DST_MAP`) validates the values on every backward
  at the cost of one device synchronization; the check is skipped during CUDA
  graph capture.

No entry synchronizes with the host unless a check that says so is enabled
(`check_dst_map`, the varlen `check_cu_seqlens`).

## NoPE (zero-rope) program

`forward_nope` / `backward_nope`, `prepare_dsa_nope_train` and the NoPE
autograd entries take `q [T, 64, 512]` and the shared latent `kv [S, 512]`
(K = V), a default softmax scale of 1/16, and return one combined
natural-layout FP32 `dkv = dK + dV [S, 512]` -- fresh, or added into the
caller's `dkv_acc`.  The index routes (in-kernel length derivation below
`NOPE_CONVERT_SLOT_LIMIT` slots, the index-preparation stage at or above it,
the packed varlen conversion) are those of the non-deterministic package; the
backward reads the saved global rows and lengths on every route.

## Layout of this package

* `cake_jit.py` -- the `MODULES` registry (one record per program, the rope
  `cake_dsa_h64_det_train` and the NoPE `cake_dsa_h64_det_nope_train`, each
  shared by every architecture it compiles for; per stage the `library` stem
  of its `build.json` entries, the FFI entry and the launch geometry that the
  export rewrites), the stage names of both programs and the loader glue
  (`library_name`, `load_cake_dsa_train_module`).
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`): one entry per `<library>_<arch>` with its translation units,
  digests, nvcc flags, closure digest and `aot/<name>.so` path.
  `cake_build.load_library(name)` loads the prebuilt library when `aot/` holds
  it and otherwise JIT-builds the entry through FlashInfer.  `aot/` is never
  tracked (`.gitignore`).
* `cake_launch.py` -- generated positional launchers and grid functions over
  the kernels' own argument names (`STAGE_ARGS` / `LAUNCH` / `GRID` keyed by
  stage name; `GRID[stage](num_queries, num_kv, topk, num_dst)`).
* `cake_backend.py` -- validation, the storage plan of the backward, varlen
  index offsetting, the prepared runners, the autograd Functions and the eager
  entry points.
* `csrc/cake_ffi_checks.h` -- the TVM-FFI helper header the bindings include,
  shipped by the export so the package compiles without FlashInfer's `csrc/`.
* `csrc/cake_dsa_h64_det_train/` -- generated kernel and binding translation
  units, one pair per library, compiled once per library and architecture
  (`.clang-format` disables formatting: the sources are identity-checked by
  the digests in `build.json`).  The forward units are those of the
  non-deterministic package's forward under this package's program names.

The package imports only itself, `torch` and `tvm_ffi`; `cake_build.py`
reaches FlashInfer only for the JIT fallback (flashinfer-ai/flashinfer#6553).

## Status

Tests: `tests/experimental/test_cake_dsa_sparse_attention_det_train.py` (the
backend suite: imports only this package; skips without a registered program
or a compute capability 10.0 / 10.3 / 10.7 device),
`tests/experimental/test_cake_dsa_sparse_attention_det_train_api.py` (the
public entry points and the environment variables of the facade) and the
CPU-only `tests/experimental/test_cake_dsa_sparse_attention_det_train_names.py`
and `tests/experimental/test_cake_dsa_sparse_attention_det_train_standalone.py`.
Benchmark: `benchmarks/bench_cake_dsa_sparse_attention_det_train.py`.
