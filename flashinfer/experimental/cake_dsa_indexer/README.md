# Cake DSA indexer top-k backend (SM100 / SM103 / SM107)

Experimental backend for the fused DeepSeek Sparse Attention indexer of a
training stack: scoring of every visible key plus the deterministic exact
top-k selection with aligned scores, in one device-side pipeline (GLM-5.2
geometry: 32 indexer heads, head dimension 128, top-k 2048).  Tracking issue:
flashinfer-ai/flashinfer#5676.

Both the API and the backend are experimental: no compatibility guarantee.
SM107 (Rubin) is the primary target of the request, SM100 (B200 / GB200) the
comparison architecture, SM103 (B300 / GB300) the third target.  The feature
does not participate in automatic backend selection, autotuning or
trace-apply; its libraries are loaded prebuilt from `aot/` when the package
ships them and JIT-built by FlashInfer from `build.json` otherwise (see
"Layout of this package").

## Public entry points (`flashinfer/dsa_indexer.py`)

```python
from flashinfer.dsa_indexer import dsa_indexer_topk, dsa_indexer_topk_workspace_size

indices, scores = dsa_indexer_topk(q, k, w, cu_seqlens_q, cu_seqlens_k,
                                   top_k=2048, softmax_scale=None, q_causal_offsets=None, ratio=1,
                                   max_seqlen_q=None, max_seqlen_k=None,
                                   workspace_buffer=None, indices=None, scores=None)
nbytes = dsa_indexer_topk_workspace_size(T, Tkv, S, top_k=2048, ratio=1, device=q.device)
```

| argument | shape and type | meaning |
|---|---|---|
| `q` | `[T, 32, 128]` BF16, contiguous | indexer queries after RoPE |
| `k` | `[Tkv, 128]` BF16 | packed indexer keys after normalization and RoPE; a row-strided view (`packed[:, :128]` of a `[Tkv, 704]` tensor) is read in place, row stride a multiple of 8 elements |
| `w` | `[T, 32]` FP32 | signed head weights, already scaled by `32 ** -0.5` |
| `cu_seqlens_q`, `cu_seqlens_k` | `[S + 1]` int32, device | independent query and key segment boundaries (not read on the host) |
| `q_causal_offsets` | optional `[S]` int64, device | per-segment query offsets |
| `top_k` | int, `1 <= top_k <= 4096` | output slots per query (default 2048) |
| `softmax_scale` | positive float | logit scale (default `128 ** -0.5`) |
| `ratio` | positive int | key compression ratio (normally 1) |
| `max_seqlen_q` | optional int | host mirror of the longest query segment; accepted, not read by any launch decision |
| `max_seqlen_k` | optional int | the caller's bound on every key segment (`max(cu_seqlens_k[1:] - cu_seqlens_k[:-1])`); sizes the rank finalize's bitmap pool and so selects its program variant (`rank_seg_window`) -- the results are bitwise identical with and without it; a value above `Tkv` or below `ceil(Tkv / S)` is rejected |
| `workspace_buffer` | optional uint8 tensor | at least `dsa_indexer_topk_workspace_size(T, Tkv, S, top_k=..., ratio=..., device=...)` bytes, 8-byte aligned |
| `indices`, `scores` (outputs) | `[T, top_k]` int32 and FP32 | selected segment-local ids in ascending order and their aligned scores; tail padding `-1` / `-inf` |

`T`, `Tkv`, the segment lengths and the last tile are arbitrary (the recorded
training calls have `T = 16231` and `16172`); nothing needs caller-side
padding.  `T = 0` returns empty outputs; a call without keys returns all
padding.

## Scoring and the reduction scheme

For query `t` and key `j` of the same segment

```
s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])
```

with FP32 products, accumulation and score arithmetic; no BF16 intermediate
rounding; no fast-math (IEEE round-to-nearest, no flush-to-zero).  Every
generated program evaluates it in one fixed sequence, so the score of a pair
is a function of `(q[t], k[j], w[t], softmax_scale)` alone and does not depend
on which program, CTA, tile or warp evaluates it:

1. **Dot product**: one tensor-core MMA of a fixed shape (128 keys x (queries
   of the unit x 32 heads) x K = 16) accumulates the 128 products of a (query,
   head, key) triple in eight K-steps over `d = 0 .. 127` in ascending order
   into an FP32 accumulator; the BF16 x BF16 products are exact in FP32.
2. **Per-key head reduction** (one thread per key row): `r_h = fl(x_h +
   |x_h|)` (exactly `2 relu(x_h)`), four interleaved FMA chains starting from
   `+0.0`, `c_m = fma(r_h, w'_h, c_m)` for `h = m, m + 4, ..., m + 28`
   (`m = 0 .. 3`), then `s = fl(fl(fl(c_0 + c_2) + fl(c_1 + c_3)) * 0.5)`.
3. **Weights**: `w'_h = fl(w_h * softmax_scale)` once per query block
   (`relu(scale * x) = scale * relu(x)` for `scale > 0`).

Every head term therefore sees at most 11 roundings after the MMA, and two
independent FP32 evaluations of one score differ by at most `gamma_n * A`
with `n = 2D + 4H + 4 = 388`, `gamma_n = n u / (1 - n u)`, `u = 2 ** -24` and
`A = softmax_scale * sum_h |w_h| sum_d |q_hd k_jd|` (the bound the tests
enforce against an independent FP64 / FP32 reference).

**Signed zeros.**  Every chain accumulator is initialised to `+0.0` and the
FMA order is fixed, and `fma(+0, w', +0) = +0` for either sign of `w'`, so a
row whose head terms are all zero yields `+0.0`; a `-0.0` score cannot arise
from this reduction (registry field `NUMERICS["zero_sign_policy"] =
"positive_accumulator"`).  Ordering rule: `+0.0` and `-0.0` compare equal, an
equal-score tie goes to the larger key id, and the written score keeps the
computed bits -- it is never canonicalised.  The reference in
`tests/test_helpers/cake_dsa_indexer_reference.py` follows the same reduction
model, so the bit-exact test cases compare like with like.

## Visibility

Let `u` be the query's segment-local position, `j` the key's segment-local
id and `Lk` the segment's key count.  Key `j` is visible exactly when

```
0 <= j < Lk   and   j < floor((offset + u + 1) / ratio)
```

with `offset = q_causal_offsets[s]` when supplied, else `Lk - Lq` for
`ratio == 1` (queries are the tail of their key prefix) and `0` for compressed
keys (`ratio > 1`).  The floor is a true floor for negative offsets.  Keys of
other segments or outside the visible prefix are never selected; negative
offsets, empty segments and rows without visible keys are defined (the row is
all padding).

## Selection, ties, output order, padding

* Every row selects the exact global top `min(top_k, visible)` keys of its
  visible prefix; the internal candidate gate is exact (certified in-kernel;
  a unit whose sampled threshold proves too aggressive is streamed again
  exactly), never a fixed candidate margin.
* Ranking: score descending, then key id descending (the larger id wins an
  equal-score tie); `+0.0 == -0.0`.
* Output: ascending id order with aligned scores; ids unique; tail slots
  `id = -1`, `score = -inf`.
* Identical inputs give identical ids and score bits across calls.  Neither
  the program the host dispatches for a geometry (unit geometry, key-range
  split, CTA pair, tile-loop unroll, unit order, sampled first threshold) nor
  the partition knobs of the backend (`grid_ctas`, `candidate_multiplier`,
  `check_period`, `sample_tiles_max`, `sample_shift_permille`, `far_sample`,
  `finalize_threads`) change a result (the tests compare bitwise).  `far_sample`
  (`0` or `1`; the architecture's record sets the default) set to `1` makes the
  sampled first threshold of the ascending-tile-order scan programs rank the far
  sample tiles only, leaving out the sample tile nearest the diagonal (an
  attention sink's local band); the descending-order and wide programs ignore it.

## Non-finite inputs

Finite inputs and finite scores are the contract's domain; `-inf` is reserved
for padding.  With NaN, infinite or FP32-overflowing inputs (NaN queries,
`+-inf` keys, BF16-max values whose products overflow FP32 -- the verified
case) the operator returns -- it must not hang -- with the output structure
intact: shapes and dtypes, unique ids inside the row's visible prefix in
ascending order, and the rows of finite segments of the same call meet the
finite contract exactly.  The score values and the relative order of NaN and
`+-inf` scores inside an affected row are not specified (the reference ranks
NaN lowest for its own bookkeeping only).  The kernels' loop bounds and
barriers do not depend on score comparisons, which is what lets any bit
pattern terminate.

## Host dispatch

One call runs two or three kernels: the persistent **scan** (scoring, exact
candidate gate, per-row selection; grid = `grid_ctas`, one CTA per SM), the
**merge** of the per-range selections when the dispatch splits the key ranges
(grid = `T`), and the **finalize** (ascending-id sort of every row in place,
by the CUB block radix sort or -- where the policy's `finalize_rank_for`
names a bitmap window for the call -- by the prefix-popcount rank scatter
program of that window; grid = `T`).  The scan exists in several physical programs and the host picks
one per call from host-known integers only -- `T`, `Tkv`, `S`, `ratio`,
`top_k`, the optional `max_seqlen_k` bound and the SM count (the device's compute capability and SM count
are read once per device index and reused); no tensor is read -- through the registry's
per-architecture `POLICY` record (`cake_policy.DispatchPolicy`,
`cake_backend.plan_dsa_indexer_topk`).  The estimate behind most rules is the
mean key tiles per work unit, `(Tkv - T / (2 ratio)) / S / 128`.

| decision | rule (record fields) |
|---|---|
| unit geometry | `block_q_wide` queries per unit first where the record sets `wide_first_rule`, the call is wide-eligible (at least `wide_rule[0]` wide rounds per CTA and `rounds x mean tiles >= wide_rule[1]`) and the mean unit tiles reach `wide_first_rule[0]`; else `block_q_l6` (three math warpgroups) when the mean unit tiles are at most `l6_rule[0]` and `top_k >= l6_rule[1]`; else `block_q_wide` when wide-eligible; else `block_q_narrow` |
| key-range split | `n_split = clamp(grid // units, 1, min(mean tiles // split_min_range_tiles, split_max))`; on architectures with `split_wave_rule` a two-way split when the units are long and the unsplit call would idle `split_wave_rule[1]` of the CTAs in its ragged last round |
| CTA pair | the two-CTA multicast program when the mean unit tiles lie in `pair_rule[0..1]`, `top_k >= pair_rule[2]`, the call is not split (grid made even) and the unit geometry is one of `pair_units` (the units whose pair program the architecture registers) |
| unit order | snake order when `snake_rule` names the program kind and the mean tiles / unit-cost spread reach its floors |
| tile-loop unroll | `tile_unroll_factor` when `tile_unroll_rule` names the program kind and `top_k` reaches its floor (yielding to the snake order where the rule says so) |
| sampled first threshold on buffer-fitting units | when the mean unit tiles are at most `sample_fit_max_mean_tiles`, the mean segment offset leaves fitting units and the program kind is one of `sample_fit_kinds` (the kinds whose sampled-fit program the architecture registers; the split programs, for example, run their plain form on every call) |
| sampled rank shift | `sample_shift_rule[1]` permille where the record sets `sample_shift_rule` and `top_k >= sample_shift_rule[0]`, else `sample_shift_permille`; an explicit `sample_shift_permille=` knob overrides both.  A launch scalar only: it sets how conservative the sampled first threshold is (a threshold that proves too aggressive is streamed again exactly), never the program, the capacity or an output bit; unsampled calls (`sample_tiles_max=0`) do not read it |
| candidate capacity | `candidate_multiplier` x `top_k` (`cand_mult_rule[1]` for long units), raised to `cand_cap_floor` slots (`cand_cap_floor_rule[1]` slots from `top_k >= cand_cap_floor_rule[0]` on, where the record sets the rule: on `sm_100a` the 16384-slot floor applies from `top_k = 1024` on and smaller `top_k` keep 8192; `sm_103a` keeps its 16384-slot floor on every `top_k`) and lowered to `cand_cap_ceiling` slots where the record sets one (the multiplier becomes `ceiling // top_k`, so the buffer stops growing with `top_k` once it holds the ceiling), rounded up to whole tiles; with the sampled first threshold off (`sample_tiles_max=0`) the `candidate_multiplier_unsampled` / `cand_mult_rule_unsampled` / `cand_cap_floor_unsampled` / `cand_cap_ceiling_unsampled` values apply instead; the default check period follows the capacity (`check_period_cap_divisor`, per-kind overrides) |
| finalize program | threads = the smallest power of two of sort slots holding `top_k` (`finalize_threads_fit` x `finalize_items`) for small `top_k`, else `finalize_threads_small` up to `finalize_top_k_small`, else `finalize_threads`.  Where the rank rules admit the call -- `rank_finalize` and `top_k >= rank_top_k_min` -- the row sort is a prefix-popcount rank finalize program sized for the call's **pool**: `Tkv`, or the caller's `max_seqlen_k` where `rank_seg_window` is set (lever RW; a bound above `Tkv` or below `ceil(Tkv / S)` is rejected whether or not the switch is set).  A pool above `rank_slab_window_max` takes the two-level (bucket + word) program of the smallest `rank_two_level_window_variants` entry holding it, with no segment bound, where `rank_two_level` is set and the window is within `rank_two_level_rule` (lever FR2; key `finalize_rank:t<threads>:w<window>:two`, staged where `rank_two_level_staged_rule` admits the window); a pool within the slab bound takes the smallest `rank_window_variants` entry holding it and one bitmap word (32 bits) per thread when `pool <= rank_rule[0]` and the segment -- mean `Tkv / S`, or `max_seqlen_k` itself under lever RW -- is at most `rank_rule[1]`, skipping the `rank_window_variants_seg_only` entries when the pool is `Tkv` (`finalize_rank:t<threads>:w<window>`, stage role `finalize_rank`; the kernel reads the segment boundaries on the device; where `rank_staged` is set and `rank_staged_rule` admits the window, its SMEM-staged, coalesced-I/O form `...:staged` -- lever FRS, same role and arguments); every other call sorts with the CUB block radix sort program (`finalize:t<threads>`, `(Tkv - 1).bit_length()` key bits).  Both produce the same ids and bits  Two forms of the staged program carry their own key tags: `:i16` (`rank_t16` / `rank_t16_slots`: the 4096-slot staged form runs 256 launched threads x 16 slots) and `:bulk` (`rank_bulk_io` / `rank_bulk_align_bytes`: `cp.async.bulk` row I/O when `top_k x 4` and the output addresses are multiples of the granule; unaligned caller-owned outputs run the registered plain twin, same results).  Where the record's `rank_persist_max_k` is above zero (SM107 only), the staged bulk program of rows with `top_k <= rank_persist_max_k` and 8-item threads carries `:persist` (the `:i16` form keeps the one-row program): one persistent grid of `min(rows, rank_persist_ctas_per_sm x SMs)` CTAs ranks the call's rows, each CTA landing its next row while it ranks the current one (`cake_policy.finalize_grid`); same results as the one-row program. |

The dispatched program must be registered for the device's architecture
(`cake_jit.PROGRAM_KEYS[arch]`); the backend raises `NotImplementedError`
naming the program key otherwise, never a different program.  The kernel
source's own dispatch functions are the reference for every rule; the export
that writes the registry verifies the record against them before it is frozen.

## Workspace and host behaviour

* **Workspace bound** (`dsa_indexer_topk_workspace_size(T, Tkv, S, top_k=,
  ratio=, device=)`): `grid_ctas x queries_per_unit x
  candidate_capacity(top_k) x 8` bytes of persistent candidate buffers for the
  dispatched unit geometry and capacity (for example about 74 MiB at `top_k =
  2048` with four queries per unit on 148 SMs at the 16384-slot capacity
  floor, 148 MiB with eight queries per unit; on `sm_103a`, whose short units
  take 16 x `top_k` slots up to the 32768-slot ceiling, 148 MiB with four
  queries per unit and 296 MiB with eight at `top_k = 2048`) plus `T x n_split x top_k x 8` bytes
  of staging when the dispatch splits the key ranges.  The bound depends on the
  call geometry; size a reused buffer for the largest bound over the
  geometries it serves.  On `sm_103a`, units of 512-1024 mean tiles with
  `top_k >= 512` and at least six wide rounds now take eight queries per unit
  (`wide_first_rule`): 152 MiB (159383552 B) at the 16384-slot floor and
  304 MiB (318767104 B) at `top_k = 4096` on 152 SMs, where the previous
  package needed 114 / 228 MiB (119537664 / 239075328 B); a reused buffer
  sized under the previous package for such a call must be re-sized (the
  runner rejects it with `workspace_buffer needs N bytes`), and
  `dsa_indexer_topk_workspace_size` follows the new plan automatically.  It is
  sized for the sampled default; a call with `sample_tiles_max=0` (the
  unsampled capacity values) needs at most as much.  On `sm_100a`, calls with
  `top_k` below `cand_cap_floor_rule[0]` (1024) take the 8192-slot floor: a
  `top_k = 512` call with six queries per unit (for example the ratio-4 pooled
  selection) needs 57 MiB (59768960 B) on 152 SMs, where the previous package
  needed 114 MiB (119537792 B), with identical ids and score bits; a
  `top_k = 256` call needs the same 57 MiB.  `sm_103a` sets no rule and keeps
  the 16384-slot floor on every `top_k` (114 MiB for that call).  Separate
  from the inputs and the two `[T, top_k]` outputs.
* **No host synchronization**: the segment boundaries and offsets are read on
  the device only; the host uses `T`, `Tkv`, `S`, `ratio`, `top_k`, the SM
  count and the strides.  The eager entry point allocates the outputs and the
  workspace (unless supplied) through the caching allocator; the prepared
  runner (`cake_backend.prepare_dsa_indexer_topk`) allocates nothing at launch
  and is CUDA-graph capturable.
* **Launch structure**: scan, optional merge, finalize (see above).  No
  `[T, Tkv]` matrix is materialized.

## Layout of this package

The kernel and launcher sources under `csrc/`, the registry `cake_jit.py` and
`build.json` were produced by the Cake exporter and are maintained by hand
pending exporter support: programs that differ only in their constants or lever
text share one translation unit, and each `build.json` entry selects its
program through its own `-D` definitions.  A changed kernel program is
re-rendered from the Cake kernel source, re-folded into its shared unit, and
every entry of that unit, expanded under its `-D` definitions, is checked token
for token against the program's own render before it ships; `build.json`
records every source's digest and a closure digest per entry (the identity rule
over program, architecture, kernel symbol, compile flags, source and shared
header digests for re-rendered entries; the original exporter's digests for the
others).  `csrc/.clang-format` disables formatting so the digests stay valid.

* `cake_jit.py` -- registry: one record per program (`PROGRAMS`: role, the
  architecture-free `library` stem, architectures), the program of every
  dispatch key per architecture (`PROGRAM_KEYS`), the per-architecture
  dispatch policy (`POLICY`), the argument plan per role, and the loader glue
  (`library_name(program, arch)` = the `build.json` entry `<library>_<arch>`,
  `load_program(program, arch)` through `cake_build.load_library`).  The
  launch geometry of a program (block, cluster, dynamic shared memory) is
  fixed in its binding.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  FlashInfer package and this package's build manifest (schema `cake-build/1`),
  the only record of what the package compiles: one entry per program and
  architecture with its package-relative sources, nvcc flags, closure digest,
  per-source SHA-256 and `aot/<name>.so` path, plus the common nvcc / link
  flags, the `tvm-ffi` requirement and the CCCL revision the finalize programs
  (CUB block radix sort) were compiled against.  `cake_build.load_library(name)`
  loads the prebuilt library when `aot/` holds it (checked against the
  builder's `aot/build_info.json` and the library's own `build_cuda_version()`)
  and otherwise JIT-builds the entry through FlashInfer from the same sources
  and flags; the prebuilt path never imports `flashinfer`.
* `csrc/cake_ffi_checks.h` -- the tvm-ffi / DLPack include set, the named
  tensor-argument checks and `cake_ffi_build_cuda_version()` the generated
  launchers use; the package compiles without FlashInfer's `csrc/`.
* `cake_policy.py` -- the dependency-free dispatch policy (program choice,
  capacity, check period, finalize program, workspace bound).
* `cake_backend.py` -- validation, planning, argument binding, the prepared
  runner and the eager entry point.
* `csrc/cake_dsa_indexer_topk/` -- generated kernel and binding translation
  units: one kernel per program or per group of programs that differ only in
  their constants (a scan lever-twin group, the rank-finalize programs of one
  window form, the CUB finalize programs) and one binding per launcher text
  (the 37 scan programs share 7 bindings, one per TMA box shape, block size
  and cluster form; the 25 rank-finalize programs share 2, the 5 CUB finalize
  programs 1; the merge program has its own).  A shared source names its
  kernel symbol and host-shim namespace through `CAKE_PROGRAM_KERNEL` /
  `CAKE_PROGRAM_SHIM` and its constants (the lever text, the window or
  thread-count constants; a finalize binding reads its block size and dynamic
  shared memory from `THREADS` / `SMEM_TOTAL`) through `#ifndef` defaults
  naming the program it was exported for; every other program's `build.json`
  entry passes its own values with `-D`.  The TMA descriptor encoders every
  scan binding carried verbatim live once in the shared launch helpers,
  selected by `CAKE_HOST_COMMON_PART`.  The sources come with the shared device
  preamble `cake_dsa_indexer_topk_device_common.cuh` and the shared launch helpers
  `cake_dsa_indexer_topk_host_common.cuh` delivered once; the preamble's helper
  functions are content-addressed files under `csrc/cake_device_helpers/`
  (`csrc/.clang-format` disables formatting so the digests stay valid).

## Tests and benchmark

Tests: `tests/experimental/test_cake_dsa_indexer.py` (the backend suite, which
imports the package only through `tests/test_helpers/cake_package.py`) and
`tests/experimental/test_cake_dsa_indexer_api.py` (the `flashinfer.dsa_indexer`
wrapper), with the independent
FP64 / FP32 reference in `tests/test_helpers/cake_dsa_indexer_reference.py`
(exact semantic cases incl. cutoff ties, all-equal scores, mixed signed zeros
at the boundary with bit checks, negative scores and a few winners over a
large tie; packed causality incl. explicit positive and negative offsets,
`ratio = 2`, empty and singleton segments, short rows, `top_k = 1` and
`4096`; random-input accuracy under the `gamma_n A` bound; near ties; both
recorded token counts; tile-boundary neighbours; changing shapes; bitwise
repeatability, partition-knob and program-dispatch invariance; strided `k`;
non-finite inputs; CUDA-graph capture; the workspace bound).  The host-only
tests (registry shape, dispatch policy, validation, binding, the reference
itself) run without a GPU.  Benchmark: `benchmarks/bench_cake_dsa_indexer.py`
(paired AB / BA / interleaved CUPTI spans of the complete operator against two
torch + FlashInfer compositions; both baseline arms are reconstructions of the
training stack's chunked scoring / coarse top-k / rescoring path written for
this benchmark, not the stack's own kernels, so their absolute numbers are
speed references only).
