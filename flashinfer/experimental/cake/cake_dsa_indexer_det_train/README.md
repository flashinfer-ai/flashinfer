# Cake DSA indexer top-k backend (SM100 / SM103 / SM107)

Experimental backend for the fused DSA (sparse attention) indexer of a
training stack: scoring of every visible key plus the deterministic exact
top-k selection with aligned scores, in one device-side pipeline (target
geometry: 32 indexer heads, head dimension 128, top-k 2048).  Tracking issue:
flashinfer-ai/flashinfer#5676.

The public entry points live in the module `flashinfer.cake_dsa_indexer_det_train`;
this backend package under `flashinfer/experimental/cake/` is their implementation
and carries no compatibility guarantee of its own.  The non-deterministic
counterpart is `flashinfer.cake_dsa_indexer_nondet_train`
(flashinfer-ai/flashinfer#8928), which leaves the slot order of each row
unspecified.
SM107 (Rubin) is the primary target of the request, SM100 (B200 / GB200) the
comparison architecture, SM103 (B300 / GB300) the third target.  The feature
does not participate in automatic backend selection, autotuning or
trace-apply; its libraries are loaded prebuilt from `aot/` when the package
ships them and JIT-built by FlashInfer from `build.json` otherwise (see
"Layout of this package").

## Public entry points (`flashinfer/cake_dsa_indexer_det_train.py`)

```python
from flashinfer.cake_dsa_indexer_det_train import (
    cake_dsa_indexer_det_train_topk,
    cake_dsa_indexer_det_train_topk_workspace_size,
)

indices, scores = cake_dsa_indexer_det_train_topk(q, k, w, cu_seqlens_q, cu_seqlens_k,
                                                  top_k=2048, softmax_scale=None, q_causal_offsets=None, ratio=1,
                                                  max_seqlen_q=None, max_seqlen_k=None,
                                                  workspace_buffer=None, indices=None, scores=None)
nbytes = cake_dsa_indexer_det_train_topk_workspace_size(T, Tkv, S, top_k=2048, ratio=1, device=q.device)

# the FP8 form (opt-in): float8_e4m3fn q / k, per-key FP32 k_scale, every other scale folded into w
indices, scores = cake_dsa_indexer_det_train_topk(q_fp8, k_fp8, w, cu_seqlens_q, cu_seqlens_k, k_scale=k_scale)
nbytes = cake_dsa_indexer_det_train_topk_workspace_size(T, Tkv, S, device=q.device, dtype=torch.float8_e4m3fn)
```

Every name carries the mode word `det`, and no entry point takes a
`deterministic=` keyword (flashinfer-ai/flashinfer#8928).  The previous names
(`flashinfer.dsa_indexer.dsa_indexer_det_topk(..., backend="cake")`,
`flashinfer.dsa_indexer.dsa_indexer_det_topk_workspace_size`, `dsa_indexer_topk`,
`dsa_indexer_topk_workspace_size`, the backend's `dsa_indexer_*` aliases) and
the package paths `flashinfer.experimental.cake_dsa_indexer_det` and
`flashinfer.experimental.cake_dsa_indexer` are removed, with no aliases;
`flashinfer.dsa_indexer.dsa_indexer_det_topk` keeps only its
`backend="sm120"` form.

| argument | shape and type | meaning |
|---|---|---|
| `q` | `[T, 32, 128]` BF16, contiguous | indexer queries after RoPE |
| `k` | `[Tkv, 128]` BF16 | packed indexer keys after normalization and RoPE; a row-strided view (`packed[:, :128]` of a `[Tkv, 704]` tensor) is read in place, row stride a multiple of 8 elements |
| `w` | `[T, 32]` FP32 | signed head weights, already scaled by `32 ** -0.5` |
| `cu_seqlens_q`, `cu_seqlens_k` | `[S + 1]` int32, device | independent query and key segment boundaries (not read on the host) |
| `q_causal_offsets` | optional `[S]` int64, device | per-segment query offsets |
| `top_k` | int, `1 <= top_k <= 4096` | output slots per query (default 2048) |
| `softmax_scale` | positive float | logit scale (default `128 ** -0.5`); `None` for the FP8 form |
| `k_scale` | optional `[Tkv]` FP32, contiguous | the FP8 form's per-key scales (see "FP8 form"); BF16 calls take none |
| `ratio` | positive int | key compression ratio (normally 1) |
| `max_seqlen_q` | optional int | host mirror of the longest query segment; accepted, not read by any launch decision |
| `max_seqlen_k` | optional int | the caller's bound on every key segment (`max(cu_seqlens_k[1:] - cu_seqlens_k[:-1])`); sizes the rank finalize's bitmap pool and so selects its program variant (`rank_seg_window`); it also shows the host whether every key segment of a packed call has the same length (`Tkv == S * max_seqlen_k`), and such a call runs its scan program without the longest-segment-first unit order (the `:lsf` copy) -- the results are bitwise identical with and without it; a value above `Tkv` or below `ceil(Tkv / S)` is rejected |
| `workspace_buffer` | optional uint8 tensor | at least `cake_dsa_indexer_det_train_topk_workspace_size(T, Tkv, S, top_k=..., ratio=..., device=...)` bytes, 8-byte aligned |
| `indices`, `scores` (outputs) | `[T, top_k]` int32 and FP32 | selected segment-local ids in ascending order and their aligned scores; tail padding `-1` / `-inf` |

`T`, `Tkv`, the segment lengths and the last tile are arbitrary (the recorded
training calls have `T = 16231` and `16172`); nothing needs caller-side
padding.  `T = 0` returns empty outputs; a call without keys returns all
padding.

## Outputs

The "repeats bit for bit" column holds for identical inputs and configuration:
the same device, the same SM count and the same arguments.  The bits may change
between package versions.

| output | contents | repeats bit for bit | not specified |
|---|---|---|---|
| `indices [T, top_k]` int32 | segment-local key ids: per row, the `min(top_k, visible)` selected ids in ascending order, then `-1` in every other slot | every slot: the selected ids, their order and the `-1` tail | nothing |
| `scores [T, top_k]` FP32 | aligned slot by slot with `indices`: the computed FP32 score of the id in the same slot (a selected `-0.0` stays `-0.0`); `-inf` in every padding slot; `None` with `return_scores=False` | every slot's bits | nothing |

Neither the program that the host dispatches nor the partition knobs of the
backend change these bits (see "Selection, ties, output order, padding").

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
`tests/experimental/test_helpers/cake_dsa_indexer_reference.py` follows the same reduction
model, so the bit-exact test cases compare like with like.

## FP8 form

The FP8 form is opt-in and selected by the input dtype: `q [T, 32, 128]` and
`k [Tkv, 128]` both `torch.float8_e4m3fn` (the same layout rules as BF16; a
row stride that is a multiple of 16 elements), `k_scale [Tkv]` FP32
(contiguous, on the device of `q`), `w [T, 32]` FP32 carrying every other
scale (the query scale, the softmax scale and the head weights), and
`softmax_scale=None` (the programs take 1.0).  Mixed q / k dtypes,
`float8_e5m2` and `q_scale` are rejected; a BF16 call keeps its default
behaviour and rejects `k_scale`.  The score is

```
s[t, j] = k_scale[j] * sum_h w[t, h] * relu(sum_d q[t, h, d] * k[j, d])
```

* **Dot product**: the e4m3 x e4m3 products are exact and accumulate in FP32
  on the tensor cores: four K = 32 steps over `d = 0 .. 127` on `sm_100a` /
  `sm_103a`, the order of DeepGEMM's `fp8_mqa_logits`, whose logits these
  programs reproduce bit for bit; two K = 64 steps on `sm_107a`, whose logits
  can differ from DeepGEMM's in the last bits.
* **Head reduction**: the sum over the heads of `w[t, h] * relu(logit)` and
  its product with `k_scale[j]` are FP32; the tests bound the score like the
  BF16 form (`gamma_n * A` against an FP64 score of the e4m3 inputs, `A`
  scaled by `|k_scale[j]|`).
* **Invariance**: every FP8 program of one architecture computes the same ids
  and score bits (program-dispatch invariance holds per form and
  architecture).  FP8 results are not BF16 results of the dequantized
  inputs, and `sm_107a` results can differ from `sm_100a` / `sm_103a` results
  in the last bits.
* **Programs**: `sm_100a` / `sm_103a` register the `narrow`, `l6`,
  `split_narrow` and `split_l6` scan programs, each with its longest-first
  segment order copy (`:lsf`), and also the unpaired `wide`
  program (8 queries per unit, tile unroll 2) with its `:lsf` copy;
  `sm_107a` registers `narrow`, `l6` with its snake unit order copy,
  `split_narrow` and the unpaired `wide` program (tile unroll 2), without
  `:lsf` copies.  Keys carry the form tag before the order suffix:
  `scan:<kind>:u<n>:s<0|1>:f1:fp8[:lsf]`, `s1` only for the `sm_107a` `l6`
  snake program.  The
  merge and finalize programs are the BF16 form's.  Launches are non-PDL and
  non-cooperative, like BF16.
* **Dispatch**: the architecture's FP8 record is its BF16 `POLICY` record with
  the fields of `cake_jit.POLICY_FP8[arch]` replaced
  (`cake_policy.fp8_policy_record`); `POLICY_FP8` names every architecture's
  FP8 `static_scan_kinds` itself.  On `sm_100a` and `sm_103a` it also sets
  `wide_wave_rule`, so the FP8 form chooses its wide unit by waves of the
  grid (the unit geometry rule below).
  `cake_policy.select_fp8_program` runs the BF16 decision procedure under it
  and maps the decided kind to a registered FP8 kind: its own where an FP8
  program of it is registered (`wide` on `sm_100a`, `sm_103a` and `sm_107a`),
  else its fallback (`FP8_KIND_FALLBACK`: `wide` / `pair_wide` / `pair_l6`
  -> `l6`, the largest FP8 unit below `wide`, `pair_narrow` -> `narrow`,
  `split_wide` -> `split_narrow`, and `split_l6` -> `split_narrow` where no
  FP8 `split_l6` program is registered); the program
  runs without a CTA pair, in the plain unit order (the snake order where
  the decision takes it, the snake copy of the FP8 program is registered
  and the call stays within the FP8 record's `snake_cap_rule`: the `l6`
  program on `sm_107a`, for one document within 40 waves of the persistent
  grid), in its sampled-fit form,
  with the decided tile unroll where an FP8 program of it is registered
  (else the smallest registered one) and in the longest-first order where
  the BF16 rule picks it and an FP8 copy exists (the FP8 record of `sm_107a`
  turns `lsf_programs` off: its static programs draw no tickets to order),
  or where the call is one
  document of the `wide` unit within the FP8 record's `lsf_single_rule`
  (`sm_103a`: from 64 mean unit tiles on; the copy walks the same units in
  the same order there and measured faster).  A fallback kind takes its
  own `sample_class_rule` class and `check_period_kind_overrides` period
  (`cake_policy.fallback_knobs`): an `l6` program run for a `wide` decision
  samples like an `l6` decision, while the FP8 `wide` program keeps the
  `wide` decision's sampling and check period; explicit knobs stay.
  On `sm_107a` every FP8 kind keeps the static unit enumeration
  (`static_scan_kinds`): no ticket tail, nothing zeroed.  An architecture
  without FP8 programs raises `NotImplementedError`.
* **Workspace**: `cake_dsa_indexer_det_train_topk_workspace_size(...,
  dtype=torch.float8_e4m3fn)` bounds the FP8 programs' workspace; the
  default `dtype=torch.bfloat16` returns the BF16 bound unchanged.  A plain
  `wide` decision (no CTA pair, no split) runs the FP8 `wide` unit, 8
  queries per unit, with 4/3 the scan entries of the `l6` fallback (the
  same grid and candidate capacity).  On `sm_100a` / `sm_103a` its FP8
  bound is the BF16 bound of the call; on `sm_107a`, where it ran the `l6`
  fallback before, the FP8 bound of such a call grows by up to 4/3 in scan
  entries (the bound is the larger of the sampled and unsampled plans).  A
  `pair_wide` decision runs the FP8 `l6` unit, 6 queries per unit rather
  than the narrow unit's 4, so its FP8 bound has 1.5x the scan entries of
  the narrow fallback.  An FP8 buffer sized for such a call under an
  earlier, smaller fallback (`narrow`, and for a `wide` decision also `l6`)
  is rejected (`workspace_buffer needs N bytes`) and must be sized again
  with this function.  On `sm_100a` and `sm_103a` the FP8 wave rule
  (`wide_wave_rule`) gives more calls the 8-query `wide` unit than the BF16
  rule does; their FP8 bound grows with the queries per unit (8 rather than
  the `narrow` unit's 4 or the `l6` unit's 6), and a buffer sized for such a
  call under the previous package is rejected the same way.
* **Prepared runners** bind `k_scale` with `q` / `k`: new values written into
  it are read by the next run.

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
program of that window; grid = `T`, or the persistent grid of the `:persist` program in the finalize row below).  The scan exists in several physical programs and the host picks
one per call from host-known integers only -- `T`, `Tkv`, `S`, `ratio`,
`top_k`, the optional `max_seqlen_k` bound and the SM count (the device's compute capability and SM count
are read once per device index and reused); no tensor is read -- through the registry's
per-architecture `POLICY` record (`cake_policy.DispatchPolicy`,
`cake_backend.plan_dsa_indexer_det_topk`).  The estimate behind most rules is the
mean key tiles per work unit, `(Tkv - T / (2 ratio)) / S / 128`.

| decision | rule (record fields) |
|---|---|
| unit geometry | where the record sets `wide_wave_rule` (the FP8 records of `sm_100a` and `sm_103a`), it replaces `wide_rule` and `wide_first_rule`: `block_q_wide` queries per unit when the unit the `l6_rule` leaves (`block_q_l6`, else `block_q_narrow`) needs at least `wide_wave_rule[1]` times the wide unit's waves of the grid (`ceil(ceil(T / block_q) / grid)`) against `block_q_l6`, from `wide_wave_rule[2]` mean unit tiles on (a fifteen-element rule, `[1.7, 1.15, 64.0, 2048, 1.1, 112.0, 3.5, 14, 1.26, 40.0, 9.0, 8.0, 55.0, 8, 60.0]` on `sm_103a`, raises that count by `wide_wave_rule[6]` per wide wave below `wide_wave_rule[7]` wide waves but on one document to at most `wide_wave_rule[11] / (W - 1) ** 2` (from two wide waves on, at most the larger of that and `wide_wave_rule[12]`, from `wide_wave_rule[13]` wide waves on of that and `wide_wave_rule[14]`) while `W` is at least `wide_wave_rule[8]`, to at least `wide_wave_rule[5]` on a packed call and to at least `wide_wave_rule[9] / (W - 1)` while the ratio `W` of the `block_q_l6` waves to the wide unit's stays below `wide_wave_rule[8]` (on a packed call then also to at least `wide_wave_rule[10] / (W - 1) ** 2`), and multiplies it by `(top_k / wide_wave_rule[3]) ** wide_wave_rule[4]` above `top_k = wide_wave_rule[3]`), or `wide_wave_rule[0]` times against `block_q_narrow`; else that unit.  Without it: `block_q_wide` queries per unit first where the record sets `wide_first_rule`, the call is wide-eligible (at least `wide_rule[0]` wide rounds per CTA and `rounds x mean tiles >= wide_rule[1]`) and the mean unit tiles reach `wide_first_rule[0]`; else `block_q_l6` (three math warpgroups) when the mean unit tiles are at most `l6_rule[0]` and `top_k >= l6_rule[1]`; else `block_q_wide` when wide-eligible; else `block_q_narrow` |
| key-range split | `n_split = clamp(grid // units, 1, min(mean tiles // split_min_range_tiles, split_max))`; on architectures with `split_wave_rule` a two-way split when the units are long and the unsplit call would idle `split_wave_rule[1]` of the CTAs in its ragged last round (where the record sets `split_wave_scope` (sm_107a), only for its unit geometries on calls of at most its segment count, and with its third item on sampled calls only); only for a unit geometry of `split_units` (the units whose split program the architecture registers): any other unit runs unsplit. A call this rule splits takes `split_resident_rule` (sm_100a, sm_103a) `min(grid // units, 49152 // top_k, 32, max(8, n_fit if n_fit x top_k <= 32768 else 32))` splits, `n_fit = ceil(mean unit keys / top_k)` in whole 4096-entry merge chunks, so the merge keeps its staged entries in shared memory; elsewhere `split_fit_rule` raises the count to `n_fit` where that fits `split_max` and the grid |
| CTA pair | the two-CTA multicast program when the mean unit tiles lie in `pair_rule[0..1]`, `top_k >= pair_rule[2]`, the call is not split (grid made even) and the unit geometry is one of `pair_units` (the units whose pair program the architecture registers) |
| unpaired wide unit | where the record sets `wide_pair_snake_only` (sm_107a), a sampled call keeps the CTA pair of the wide unit only where the pair program walks the snake order; elsewhere it runs the unpaired wide program when that program is registered |
| unpaired three-warpgroup preference | a call whose plan does not split takes the `block_q_l6` unit without a CTA pair: with the sampled first threshold off (`sample_tiles_max=0`), where the record sets `l6_solo_unsampled_max_queries` (sm_100a, sm_103a: 16384) only a call planned on the CTA-pair l6 unit of at most that many queries (other units keep their plan), else always where the record sets `l6_solo_rule` (sm_107a); on a sampled call, where the record sets `l6_solo_rule`, with at least `l6_solo_rule[1]` segments or at most `l6_solo_rule[0]` mean unit tiles; only where the l6 unit would not split either and the program it picks is registered (`PROGRAM_KEYS[arch]`, which the backend passes to `select_program`; without it the preference stays off), else the plan of the rows above.  Where the record sets `snake_l6_fallback_waves` (sm_107a: 310), a sampling-off call whose snake-order l6 program is not registered runs the plain-order l6 program instead when the planned program needs at most that many waves of its persistent grid (`ceil(ceil(T / block_q) / slots)`, slots = CTAs, or CTA pairs for a pair program) |
| unit order | snake order when `snake_rule` names the program kind and the mean tiles / unit-cost spread reach its floors; on a call of two or more segments whose kind `snake_packed_rule` names (sm_107a: `[["l6"], 48.0, None, 7]`), only from `snake_packed_rule[1]` mean unit tiles on with the sampled first threshold on and from `snake_packed_rule[2]` with it off (None: never), or within `snake_packed_rule[3]` waves of the persistent grid (`ceil(ceil(T / block_q) / slots)`); else the plain order; an FP8 call runs a snake program only within `snake_cap_rule` [max segments, max waves of the persistent grid] (the FP8 record of `sm_107a`: [1, 40]; None elsewhere) |
| segment order | the longest-first copy (`:lsf`) of the decided program where the record sets `lsf_programs` (every BF16 record; the FP8 records of `sm_100a` and `sm_103a`) and the call packs at least two key segments that are not all equal (`max_seqlen_k` decides; without it the copy runs); a program registered without its copy runs in the argument order: `sm_100a` and `sm_103a` register the copy of every BF16 scan program, `sm_107a` only those of its two `l6` ticket programs (`scan:l6:u1:s0:f0`, `scan:l6:u1:s0:f1`), which keep the argument order when the longest key segment is at most 5/4 of the shortest non-empty one (same results either way), and every row that asks the registry for a program looks up its key without the suffix, so the copy never changes the plan; an FP8 call of the `wide` unit on one segment also runs it from `lsf_single_rule[0]` mean unit tiles on (the FP8 record of `sm_103a`: [64.0]; None elsewhere) |
| tile-loop unroll | `tile_unroll_factor` when `tile_unroll_rule` names the program kind and `top_k` reaches its floor (yielding to the snake order where the rule says so) |
| sampled first threshold on buffer-fitting units | when the mean unit tiles are at most `sample_fit_max_mean_tiles`, the mean segment offset leaves fitting units and the program kind is one of `sample_fit_kinds` (the kinds whose sampled-fit program the architecture registers; the split programs, for example, run their plain form on every call) |
| sampled rank shift | `sample_shift_rule[1]` permille where the record sets `sample_shift_rule` and `top_k >= sample_shift_rule[0]`, else `sample_shift_permille`; where the record sets `sample_long_rule` (sm_107a: `[["l6"], 500]`), a call of more than `sample_dispatch_mean_tiles_max` mean unit tiles on a kind it names takes `sample_long_rule[1]`; otherwise, where the record sets `sample_class_rule` (sm_100a, sm_103a: [8, 500]; sm_107a: [32, 500]), a wide-unit call of two or more segments takes `sample_class_rule[1]`, and a call of two or more segments of at most 4096 queries per segment on average and at most `sample_dispatch_tiny_tiles_max` mean unit tiles samples `sample_class_rule[0]` tiles at that shift (where `sample_class_rule[0]` exceeds `sample_tiles_tiny_units`, only a call of more than `sample_class_rule[0]` mean unit tiles: on shorter units the larger sample covers whole units); an explicit `sample_shift_permille=` / `sample_tiles_max=` knob overrides these.  A launch scalar only: it sets how conservative the sampled first threshold is (a threshold that proves too aggressive is streamed again exactly), never the program, the capacity or an output bit; unsampled calls (`sample_tiles_max=0`) do not read it |
| candidate capacity | `candidate_multiplier` x `top_k` (`cand_mult_rule[1]` for long units), raised to `cand_cap_floor` slots (`cand_cap_floor_rule[1]` slots from `top_k >= cand_cap_floor_rule[0]` on, where the record sets the rule: on `sm_100a` the 16384-slot floor applies from `top_k = 1024` on and smaller `top_k` keep 8192; `sm_103a` keeps its 16384-slot floor on every `top_k`) and lowered to `cand_cap_ceiling` slots where the record sets one (the multiplier becomes `ceiling // top_k`, so the buffer stops growing with `top_k` once it holds the ceiling), rounded up to whole tiles; with the sampled first threshold off (`sample_tiles_max=0`) the `candidate_multiplier_unsampled` / `cand_mult_rule_unsampled` / `cand_cap_floor_unsampled` / `cand_cap_ceiling_unsampled` values apply instead; the default check period follows the capacity (`check_period_cap_divisor`, per-kind overrides) |
| finalize program | threads = the smallest power of two of sort slots holding `top_k` (`finalize_threads_fit` x `finalize_items`) for small `top_k`, else `finalize_threads_small` up to `finalize_top_k_small`, else `finalize_threads`.  Where the rank rules admit the call -- `rank_finalize` and `top_k >= rank_top_k_min` -- the row sort is a prefix-popcount rank finalize program sized for the call's **pool**: `Tkv`, or the caller's `max_seqlen_k` where `rank_seg_window` is set (lever RW; a bound above `Tkv` or below `ceil(Tkv / S)` is rejected whether or not the switch is set).  A pool above `rank_slab_window_max` takes the two-level (bucket + word) program of the smallest `rank_two_level_window_variants` entry holding it, with no segment bound, where `rank_two_level` is set and the window is within `rank_two_level_rule` (lever FR2; key `finalize_rank:t<threads>:w<window>:two`, staged where `rank_two_level_staged_rule` admits the window); a pool within the slab bound takes the smallest `rank_window_variants` entry holding it and one bitmap word (32 bits) per thread when `pool <= rank_rule[0]` and the segment -- mean `Tkv / S`, or `max_seqlen_k` itself under lever RW -- is at most `rank_rule[1]`, skipping the `rank_window_variants_seg_only` entries when the pool is `Tkv` (`finalize_rank:t<threads>:w<window>`, stage role `finalize_rank`; the kernel reads the segment boundaries on the device; where `rank_staged` is set and `rank_staged_rule` admits the window, its SMEM-staged, coalesced-I/O form `...:staged` -- lever FRS, same role and arguments); every other call sorts with the CUB block radix sort program (`finalize:t<threads>`, `(Tkv - 1).bit_length()` key bits).  Both produce the same ids and bits  Two forms of the staged program carry their own key tags: `:i16` (`rank_t16` / `rank_t16_slots`: the 4096-slot staged form runs 256 launched threads x 16 slots) and `:bulk` (`rank_bulk_io` / `rank_bulk_align_bytes`: `cp.async.bulk` row I/O when `top_k x 4` and the output addresses are multiples of the granule; unaligned caller-owned outputs run the registered plain twin, same results).  Where the record's `rank_persist_max_k` is above zero (SM107 only), the staged bulk program of rows with `top_k <= rank_persist_max_k` and 8-item threads carries `:persist` (the `:i16` form keeps the one-row program): one persistent grid of `min(rows, rank_persist_ctas_per_sm x (SMs - rank_persist_spare_sms))` CTAs (SM107: 8 x (SMs - 1)) ranks the call's rows, each CTA landing its next row while it ranks the current one (`cake_policy.finalize_grid`); an SM107 SM holds 4 of these 256-thread CTAs at a time, so the grid is two whole waves of the other SMs when another stream holds a block on one SM, not one wave plus a short second one (#7548); same results as the one-row program. |

The dispatched program must be registered for the device's architecture
(`cake_jit.PROGRAM_KEYS[arch]`); the backend raises `NotImplementedError`
naming the program key otherwise, never a different program.  The kernel
source's own dispatch functions are the reference for every rule; the export
that writes the registry verifies the record against them before it is frozen.

## Workspace and host behaviour

* **Workspace bound** (`cake_dsa_indexer_det_train_topk_workspace_size(T, Tkv, S, top_k=,
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
  `cake_dsa_indexer_det_train_topk_workspace_size` follows the new plan automatically.  On
  `sm_100a`, BF16 calls of 5.5 to 6 wide rounds per CTA (`wide_rule[0]`; the
  FP8 form decides by its `wide_wave_rule`) now take eight queries per unit as
  well: 148 MiB (155197568 B) at the 16384-slot floor at `top_k = 2048` on 148
  SMs, where the previous package needed 74 MiB (77602944 B), with identical
  ids and score bits; re-size a buffer reused across such calls the same way.  It is
  the larger of the sampled default's plan and the plan with the sampled first
  threshold off (`sample_tiles_max=0`): on sm_107a the unpaired
  three-warpgroup preference can give an unsampled call six queries per unit
  where the sampled call keeps four, so a buffer sized by the bound serves both
  modes.  On `sm_100a`, calls with
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
  workspace (unless supplied) through the caching allocator; the scan binding
  zeroes the ticket tail of a supplied workspace right before every scan
  launch, in the same host call (none on a static call:
  where the shape alone bounds the work units by the grid,
  `cake_policy.ticket_static_path`, the scan never touches the ticket word;
  none for a program of the `static_scan_kinds` of its form's record, which
  keeps the static unit enumeration on every call: on `sm_107a` every FP8
  program);
  the prepared runner (`cake_backend.prepare_dsa_indexer_det_topk`) allocates nothing at launch
  and is CUDA-graph capturable.
* **Launch structure**: scan, optional merge, finalize (see above).  No
  `[T, Tkv]` matrix is materialized.

## Layout of this package

The kernel and launcher sources under `csrc/`, the registry `cake_jit.py` and
`build.json` are produced by the Cake exporter: programs that differ only in a
few constants share one kernel translation unit, programs whose launchers differ
only in the kernel symbol, block size and dynamic shared memory share one
binding, and each `build.json` entry selects its program through its own `-D`
definitions.  `build.json` records every source's digest and a closure digest
per entry.  `csrc/.clang-format` disables formatting so the digests stay valid.

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
  flags and the `tvm-ffi` requirement.  `cake_build.load_library(name)`
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
* `csrc/cake_dsa_indexer_det_topk/` -- generated kernel and binding translation
  units: one kernel per program or per group of programs that differ only in
  their constants (a scan lever-twin group, the rank-finalize programs of one
  window form, the CUB finalize programs) and one binding per launcher text
  (the 76 scan programs share 9 bindings, the 25 rank-finalize programs 9, the
  5 CUB finalize programs 1; the merge program has its own).  A shared kernel
  source names its kernel `CAKE_PROGRAM_KERNEL` and reads its constants (the
  window, block size, tile unroll and snake order) from `CAKE_INDEXER_<KNOB>`; a
  shared binding names the kernel, its host-shim namespace, block size and
  dynamic shared memory through `CAKE_KERNEL_SYMBOL`, `CAKE_PROGRAM_SHIM`,
  `CAKE_BINDING_THREADS` and `CAKE_BINDING_SMEM`.  Each of these has an
  `#ifndef` default naming the program the file was exported for; every other
  program's `build.json` entry passes its own values with `-D`.

## Tests and benchmark

Tests: `tests/experimental/test_cake_dsa_indexer_det_train.py` (the backend suite, which
imports the package only through `tests/cake/test_helpers/cake_package.py`),
`tests/experimental/test_cake_dsa_indexer_det_train_api.py` (the
`flashinfer.cake_dsa_indexer_det_train` entry points) and
`tests/experimental/test_cake_dsa_indexer_det_train_names.py` (the #8928 naming
rules), with the independent
FP64 / FP32 reference in `tests/experimental/test_helpers/cake_dsa_indexer_reference.py`
(exact semantic cases incl. cutoff ties, all-equal scores, mixed signed zeros
at the boundary with bit checks, negative scores and a few winners over a
large tie; packed causality incl. explicit positive and negative offsets,
`ratio = 2`, empty and singleton segments, short rows, `top_k = 1` and
`4096`; random-input accuracy under the `gamma_n A` bound; near ties; both
recorded token counts; tile-boundary neighbours; changing shapes; bitwise
repeatability, partition-knob and program-dispatch invariance; strided `k`;
non-finite inputs; CUDA-graph capture; the workspace bound).  The host-only
tests (registry shape, dispatch policy, validation, binding, the reference
itself) run without a GPU.  The FP8 form has its own suite,
`tests/experimental/test_cake_dsa_indexer_det_train_fp8.py`: host-only key grammar,
FP8 record, kind mapping and reachability of every registered FP8 key,
validation and BF16 plan / workspace identity; on the GPU, accuracy against
an FP64 reference of the e4m3 inputs under the `gamma_n A` bound with exact
ids up to near ties, bitwise program-dispatch invariance over every
registered FP8 key, the prepared runner's `k_scale` binding, the per-form
ticket decision and, where `deep_gemm` imports on `sm_100a` / `sm_103a`, a
bitwise comparison of the logits with `deep_gemm.fp8_mqa_logits`.  Benchmark: `benchmarks/bench_cake_dsa_indexer_det_train.py`
(paired AB / BA / interleaved CUPTI spans of the complete operator against two
torch + FlashInfer compositions; both baseline arms are reconstructions of the
training stack's chunked scoring / coarse top-k / rescoring path written for
this benchmark, not the stack's own kernels, so their absolute numbers are
speed references only).  `--form fp8` times the FP8 form against the BF16
form on the same data dequantized to BF16 and, where they import, DeepGEMM's
`fp8_mqa_logits` followed by FlashInfer's exact deterministic top-k
(`--rows fp8`: the FP8 shape set).
