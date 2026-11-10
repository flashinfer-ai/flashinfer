# Cake DSA indexer top-k backend, non-deterministic (SM100 / SM103 / SM107)

Backend package of `flashinfer.cake_dsa_indexer_nondet_train`. It is the
non-deterministic counterpart of `flashinfer.cake_dsa_indexer_det_train`: the
fused DSA (sparse attention) indexer of a training stack.  One device-side
pipeline scores every visible key and selects the exact top-k with aligned
scores.  The target geometry is 32 indexer heads, head dimension 128 and
top-k 2048.  Tracking issue: flashinfer-ai/flashinfer#5676.

This package runs the deterministic package's BF16 scan programs and its plain
split merge unchanged.  It leaves out the ordering pass, which is the
deterministic package's finalize / rank finalize that sorts every row into
ascending id order.  The scores are the same, and each row holds the same
selected set in a different slot order.  The package implements the
set-equality semantics of flashinfer-ai/flashinfer#8946 (point 4): per row,
the set of selected `(id, score bits)` pairs repeats bit for bit, but the slot
order does not.

The entry points of `flashinfer.cake_dsa_indexer_nondet_train` are stable
APIs.  This backend package under `flashinfer/experimental/cake/` is
experimental and has no compatibility guarantee of its own.  SM107 (Rubin) is
the primary target of the request.  SM100 (B200 / GB200) is the comparison
architecture and SM103 (B300 / GB300) is the third target.  The feature does
not take part in automatic backend selection, autotuning or trace-apply.  Its
libraries are loaded prebuilt from `aot/` when the package ships them.
Otherwise FlashInfer JIT-builds them from `build.json` (see "Layout of this
package").

## Mode and names (flashinfer-ai/flashinfer#8928)

Every public name carries the mode word `nondet`, and no entry point takes a
`deterministic=` keyword.  The deterministic package is
`flashinfer.cake_dsa_indexer_det_train`.  It runs the same scan programs and
returns the same set per row in ascending id order.  The backend's own names
also carry the mode word (`prepare_dsa_indexer_nondet_topk`,
`DSAIndexerNondetRunner`, `plan_dsa_indexer_nondet_topk`,
`validate_dsa_indexer_nondet_inputs`).  The generated programs are named
`cake_dsa_indexer_nondet_topk_<20 hex>`.

**Deterministic algorithms.**  The guard applies under
`torch.use_deterministic_algorithms(True)`:

* The eager entry `cake_dsa_indexer_nondet_train_topk` raises PyTorch's
  `RuntimeError` only when the call launches (`T > 0` and `Tkv > 0`).  It
  raises after its argument checks and before it allocates or launches
  anything.  The argument checks cover shapes, dtypes, the outputs, the plan,
  the registry lookup of the call's programs, and the size and alignment of a
  caller-owned `workspace_buffer`.
* A call without queries or keys launches nothing and does not alert.
* With `warn_only=True` the call issues a `UserWarning` and then runs.
* A prepared runner (`cake_backend.prepare_dsa_indexer_nondet_topk`) checks
  the setting at the top of every `run()` that launches, before its first
  launch.
* Preparing a runner never alerts.  The workspace-size function
  `cake_dsa_indexer_nondet_train_topk_workspace_size` never alerts either.
* A CUDA graph replay runs no host code and does not alert.  The setting is
  read when a run is captured.
* Nothing falls back to the deterministic package.

## Public entry points (`flashinfer/cake_dsa_indexer_nondet_train.py`)

```python
from flashinfer.cake_dsa_indexer_nondet_train import (
    cake_dsa_indexer_nondet_train_topk,
    cake_dsa_indexer_nondet_train_topk_workspace_size,
)

indices, scores = cake_dsa_indexer_nondet_train_topk(q, k, w, cu_seqlens_q, cu_seqlens_k,
                                                     top_k=2048, softmax_scale=None, q_causal_offsets=None, ratio=1,
                                                     max_seqlen_q=None, max_seqlen_k=None,
                                                     workspace_buffer=None, indices=None, scores=None,
                                                     return_scores=True)
nbytes = cake_dsa_indexer_nondet_train_topk_workspace_size(num_queries, num_keys, num_segments,
                                                           top_k=2048, ratio=1, device=None, dtype=torch.bfloat16)

# indices only: None is returned in place of the scores (they are still computed internally)
indices, _ = cake_dsa_indexer_nondet_train_topk(q, k, w, cu_seqlens_q, cu_seqlens_k, return_scores=False)
```

Every argument after `cu_seqlens_k` (and after `num_segments`) is
keyword-only.

| argument | shape and type | meaning |
|---|---|---|
| `q` | `[T, 32, 128]` BF16, contiguous | indexer queries after RoPE |
| `k` | `[Tkv, 128]` BF16 | packed indexer keys after normalization and RoPE; a row-strided view (`packed[:, :128]` of a `[Tkv, 704]` tensor) is read in place, unit inner stride, row stride a multiple of 8 elements |
| `w` | `[T, 32]` FP32, contiguous | signed head weights, already scaled by `32 ** -0.5` |
| `cu_seqlens_q`, `cu_seqlens_k` | `[S + 1]` int32, contiguous, device | independent query and key segment boundaries (`cu_seqlens_q[-1] == T`, `cu_seqlens_k[-1] == Tkv`; not read on the host) |
| `q_causal_offsets` | optional `[S]` int64, contiguous, device | per-segment query offsets |
| `top_k` | int, `1 <= top_k <= 4096` | output slots per query (default 2048) |
| `softmax_scale` | positive finite float | logit scale (default `128 ** -0.5`) |
| `ratio` | int, `>= 1` | key compression ratio (normally 1) |
| `max_seqlen_q` | optional int | host mirror of the longest query segment; accepted, not read by any launch decision |
| `max_seqlen_k` | optional int | the caller's bound on every key segment (`max(cu_seqlens_k[1:] - cu_seqlens_k[:-1])`); read only by the segment-order rule (a packed call of equal key segments runs its scan without the longest-first order); the selected sets are identical with and without it; a value above `Tkv` or below `ceil(Tkv / S)` is rejected |
| `workspace_buffer` | optional uint8 tensor | at least `cake_dsa_indexer_nondet_train_topk_workspace_size(T, Tkv, S, top_k=..., ratio=..., device=...)` bytes, 8-byte aligned; allocated through the caching allocator when omitted |
| `indices`, `scores` (outputs) | `[T, top_k]` int32 and FP32, contiguous, on the device of `q` | caller-owned outputs, allocated when omitted; see "Outputs"; `scores` must be omitted with `return_scores=False` |
| `return_scores` | bool | `False`: `None` is returned in place of the scores (default `True`) |

This package offers the BF16 form only, and there is no `k_scale` argument.
Float8_e4m3fn `q` / `k` raise `ValueError`.  The message says that this
package offers the BF16 form only and that the float8_e4m3fn form is the
opt-in FP8 form of `flashinfer.cake_dsa_indexer_det_train`.
`cake_dsa_indexer_nondet_train_topk_workspace_size(..., dtype=torch.float8_e4m3fn)`
raises the same way.  Every other non-BF16 dtype raises `ValueError`.  A
device other than compute capability 10.0, 10.3 or 10.7 raises `ValueError`.
An architecture without registered programs raises `NotImplementedError`.

`T`, `Tkv`, the segment lengths and the last tile are arbitrary.  The recorded
training calls have `T = 16231` and `16172`.  Nothing needs caller-side
padding.  `T = 0` returns empty outputs, and a call without keys returns all
padding.

## Outputs

The "repeats" column holds for identical inputs and configuration: the same
device, the same SM count and the same arguments.

| output | contents | repeats bit for bit | not specified |
|---|---|---|---|
| `indices [T, top_k]` int32 | segment-local key ids: per row, the `min(top_k, visible)` selected ids, unique within the row, and `-1` in every other slot | the set of selected ids of every row, and with it the row's count of valid (non-`-1`) slots | the slot order of the selected ids and the positions of the `-1` slots; both may vary between calls |
| `scores [T, top_k]` FP32 | aligned slot by slot with `indices`: the score of the id in the same slot, with its computed FP32 bits (a selected `-0.0` stays `-0.0`); `-inf` in every padding slot; `None` with `return_scores=False` | each selected id's score bits, so the set of `(id, score bits)` pairs of every row | the slot order, which follows `indices` |

* A row holds the same ids and score bits that the deterministic package
  returns for it, because both packages run the same scan programs.  Only the
  slot order differs.  A caller that needs ascending id order can sort the row
  or use the deterministic package.  The following rearranges every row into
  the deterministic package's layout (ascending ids, padding last):

  ```python
  key = indices.masked_fill(indices < 0, torch.iinfo(torch.int32).max)
  order = key.argsort(dim=1)
  indices, scores = indices.gather(1, order), scores.gather(1, order)
  ```

* A row with no visible key is all padding (`-1` / `-inf`).

## Scoring and the reduction scheme

For query `t` and key `j` of the same segment

```
s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])
```

Products, accumulation and score arithmetic are FP32.  There is no BF16
intermediate rounding and no fast-math (IEEE round-to-nearest, no
flush-to-zero).  The scan programs are the deterministic package's BF16 scan
programs, so a pair's score bits are that package's bits.  Every program
evaluates the score in one fixed sequence.  The score of a pair therefore
depends only on `(q[t], k[j], w[t], softmax_scale)`, not on which program,
CTA, tile or warp evaluates it:

1. **Dot product**: one tensor-core MMA of a fixed shape (128 keys x (queries
   of the unit x 32 heads) x K = 16).  It accumulates the 128 products of a
   (query, head, key) triple into an FP32 accumulator in eight K-steps over
   `d = 0 .. 127`, in ascending order.  The BF16 x BF16 products are exact in
   FP32.
2. **Per-key head reduction** (one thread per key row): `r_h = fl(x_h +
   |x_h|)` (exactly `2 relu(x_h)`).  Four interleaved FMA chains start from
   `+0.0`: `c_m = fma(r_h, w'_h, c_m)` for `h = m, m + 4, ..., m + 28`
   (`m = 0 .. 3`).  Then `s = fl(fl(fl(c_0 + c_2) + fl(c_1 + c_3)) * 0.5)`.
3. **Weights**: `w'_h = fl(w_h * softmax_scale)` once per query block
   (`relu(scale * x) = scale * relu(x)` for `scale > 0`).

Every head term therefore sees at most 11 roundings after the MMA.  Two
independent FP32 evaluations of one score differ by at most `gamma_n * A`,
with `n = 2D + 4H + 4 = 388`, `gamma_n = n u / (1 - n u)`, `u = 2 ** -24` and
`A = softmax_scale * sum_h |w_h| sum_d |q_hd k_jd|`.  The deterministic
package's tests enforce this bound against an independent FP64 / FP32
reference, and this package's score bits are the same.

**Signed zeros.**  Every chain accumulator starts at `+0.0` and the FMA order
is fixed.  Because `fma(+0, w', +0) = +0` for either sign of `w'`, a row whose
head terms are all zero yields `+0.0`.  A `-0.0` score cannot come from this
reduction (registry field `NUMERICS["zero_sign_policy"] =
"positive_accumulator"`).  Ordering rule: `+0.0` and `-0.0` compare equal, and
an equal-score tie goes to the larger key id.  The written score keeps the
computed bits and is never canonicalized.  The reference in
`tests/experimental/test_helpers/cake_dsa_indexer_reference.py` follows the same reduction
model, so the bit-exact test cases compare like with like after each row is
canonicalized (see "Tests and benchmark").

## Visibility

Let `u` be the query's segment-local position, `j` the key's segment-local id
and `Lk` the segment's key count.  Key `j` is visible exactly when

```
0 <= j < Lk   and   j < floor((offset + u + 1) / ratio)
```

`offset` is `q_causal_offsets[s]` when supplied.  Otherwise it is `Lk - Lq`
for `ratio == 1` (queries are the tail of their key prefix) and `0` for
compressed keys (`ratio > 1`).  The floor is a true floor for negative
offsets.  Keys of other segments and keys outside the visible prefix are never
selected.  Negative offsets, empty segments and rows without visible keys are
defined: the row is all padding.

## Selection, ties, output order, padding

* Every row selects the exact global top `min(top_k, visible)` keys of its
  visible prefix.  The internal candidate gate is exact, never a fixed
  candidate margin: it is certified in-kernel, and a unit whose sampled
  threshold proves too aggressive is streamed again exactly.
* Ranking: score descending, then key id descending (the larger id wins an
  equal-score tie).  `+0.0 == -0.0`.
* Output: no specified slot order (see "Outputs").  Ids are unique.  Every slot
  that does not hold a selected entry holds `id = -1`, `score = -inf`.
* The following never change the selected set of a row, only its slot order:
  * the program the host dispatches for a geometry: unit geometry, key-range
    split, CTA pair, tile-loop unroll, unit order, segment order, descending
    walk and sampled first threshold;
  * the partition knobs of the backend: `grid_ctas`, `candidate_multiplier`,
    `check_period`, `sample_tiles_max`, `sample_shift_permille` and
    `far_sample`;
  * `max_seqlen_k`.
* `far_sample` is `0` or `1`, and the architecture's record sets the default.
  Set to `1`, it makes the sampled first threshold of the ascending-tile-order
  scan programs rank the far sample tiles only.  It leaves out the sample tile
  nearest the diagonal (an attention sink's local band).  The descending-order
  and wide programs ignore it.

## Non-finite inputs

Finite inputs and finite scores are the contract's domain, and `-inf` is
reserved for padding.  The operator may get NaN, infinite or FP32-overflowing
inputs: NaN queries, `+-inf` keys, or BF16-max values whose products overflow
FP32.  It still returns, and it must not hang.  The output structure stays
intact:

* shapes and dtypes;
* unique ids inside the row's visible prefix, in no specified order;
* the rows of finite segments of the same call meet the finite contract
  exactly.

The score values inside an affected row, and the relative order of its NaN
and `+-inf` scores, are not specified.  The reference ranks NaN lowest for its
own bookkeeping only.  The kernels' loop bounds and barriers do not depend on
score comparisons, so any bit pattern terminates.

## Host dispatch

One call runs one kernel or two:

* **Unsplit call**: only the persistent **scan** runs.  It does the scoring,
  the exact candidate gate and the per-row selection, and writes every row's
  selected entries and padding straight into `indices` / `scores`.  Its grid
  is `grid_ctas`: one CTA per SM by default, made even for the CTA-pair
  program.
* **Split call**: when the dispatch splits the key ranges, the scan writes the
  per-range selections to the staging area of the workspace.  The plain split
  **merge** then writes the final rows (grid = `T`).  One merge program serves
  the three architectures.

No kernel orders a row.  The deterministic package's finalize, rank finalize
and rank merge programs are not part of this package.

The scan exists in several physical programs, and the host picks one per call
from host-known integers only: `T`, `Tkv`, `S`, `ratio`, `top_k`, the optional
`max_seqlen_k` bound and the SM count.  No tensor is read.  The device's
compute capability and SM count are read once per device index and reused.
The choice goes through the registry's per-architecture `POLICY` record
(`cake_policy.DispatchPolicy`, `cake_policy.select_program`,
`cake_backend.plan_dsa_indexer_nondet_topk`).  These rules are the
deterministic package's rules without its finalize, rank and FP8 rules.  A
call therefore gets the same scan program, candidate capacity, check period,
split count, CTA pair and ticket plan that the deterministic package picks for
it.  One exception is on `sm_107a`: the sampled tile count and rank
shift follow this package's own `sample_class_rule` and `sample_long_rule`
(whether a call samples does not change).  Most rules use one estimate, the
mean key tiles per work unit: `(Tkv - T / (2 ratio)) / S / 128`.

| decision | rule (record fields) |
|---|---|
| unit geometry | `block_q_wide` queries per unit first where the record sets `wide_first_rule` (sm_103a: `[512.0]`), the call is wide-eligible (at least `wide_rule[0]` wide rounds per CTA and `rounds x mean tiles >= wide_rule[1]`, `rounds = ceil(T / block_q_wide) / grid`) and the mean unit tiles reach `wide_first_rule[0]`; else `block_q_l6` (three math warpgroups) when the mean unit tiles are at most `l6_rule[0]` and `top_k >= l6_rule[1]`; else `block_q_wide` when wide-eligible; else `block_q_narrow`.  The policy also evaluates `wide_wave_rule`, which replaces both wide rules where a record sets it; no record of this package sets it |
| key-range split | `n_split = max(1, min(grid // units, mean tiles // split_min_range_tiles, split_max))`.  Where `n_split` is 1, a record with `split_wave_rule` splits in two when the units are long (at least `split_wave_rule[0]` and `2 x split_min_range_tiles` mean tiles) and the split raises the share of busy CTAs over the call's rounds by at least `split_wave_rule[1]`.  Where the record sets `split_wave_scope` (sm_107a: `[["narrow"], 1, true]`), that two-way split applies only to its unit geometries, on calls of at most its segment count, and with its third item on sampled calls only.  Only a unit geometry of `split_units` (the units whose split program the architecture registers) splits; any other unit runs unsplit.  A call that this rule splits takes `split_resident_rule` (sm_100a, sm_103a): `min(grid // units, 49152 // top_k, 32, max(8, n_fit if n_fit x top_k <= 32768 else 32))` splits, at least 2.  Here `n_fit = ceil(mean unit keys / top_k)`, rounded up to whole 4096-entry merge chunks, so the merge keeps its staged entries in shared memory.  Elsewhere `split_fit_rule` raises the count to `n_fit` where that fits `split_max` and the grid |
| CTA pair | the two-CTA multicast program when the mean unit tiles lie in `pair_rule[0..1]`, `top_k >= pair_rule[2]`, the grid has at least two CTAs, the call is not split (grid made even) and the unit geometry is one of `pair_units` (the units whose pair program the architecture registers) |
| unpaired wide unit | where the record sets `wide_pair_snake_only` (sm_107a), a sampled call keeps the CTA pair of the wide unit only where the pair program walks the snake order; elsewhere it runs the unpaired wide program when that program is registered |
| unpaired three-warpgroup preference | a call whose plan does not split takes the `block_q_l6` unit without a CTA pair.  With the sampled first threshold off (`sample_tiles_max=0`), where the record sets `l6_solo_unsampled_max_queries` (sm_100a, sm_103a: 16384), this applies only to a call planned on the CTA-pair l6 unit of at most that many queries; other units keep their plan.  Otherwise it applies always where the record sets `l6_solo_rule` (sm_107a).  On a sampled call, where the record sets `l6_solo_rule`, it applies with at least `l6_solo_rule[1]` segments or at most `l6_solo_rule[0]` mean unit tiles.  In every case it applies only where the l6 unit would not split either and the program it picks is registered (`PROGRAM_KEYS[arch]`, which the backend passes to `select_program`; without it the preference stays off); else the plan of the rows above applies.  Where the record sets `snake_l6_fallback_waves` (sm_107a: 310), a sampling-off call whose snake-order l6 program is not registered runs the plain-order l6 program instead, when the planned program needs at most that many waves of its persistent grid (`ceil(ceil(T / block_q) / slots)`, slots = CTAs, or CTA pairs for a pair program) |
| unit order | snake order when `snake_rule` names the program kind and the mean tiles (the ratio-1 or the `ratio > 1` floor) and the unit-cost spread reach its floors.  On a call of two or more segments whose kind `snake_packed_rule` names (sm_107a: `[["l6"], 48.0, None, 7]`), the snake order applies only from `snake_packed_rule[1]` mean unit tiles on with the sampled first threshold on, from `snake_packed_rule[2]` with it off (None: never), or within `snake_packed_rule[3]` waves of the persistent grid.  Else the plain order |
| segment order | the longest-first copy (`:lsf`) of the decided program where the record sets `lsf_programs` (every architecture) and the call packs at least two key segments that are not all equal.  `max_seqlen_k` decides: `Tkv == S x max_seqlen_k` means equal segments; without it the copy runs.  A program registered without its copy runs in the argument order: `sm_100a` and `sm_103a` register the copy of every scan program, `sm_107a` only those of its two `l6` ticket programs (`scan:l6:u1:s0:f0`, `scan:l6:u1:s0:f1`), which keep the argument order when the longest key segment is at most 5/4 of the shortest non-empty one (same sets either way).  Every row that asks the registry for a program looks up its key without the suffix, so the copy never changes the plan |
| descending walk | the descending-walk twin of the decided program (key kind `<kind>_desc`: the same unit, split, pair, tile unroll and fit form) where `desc_walk_kinds` names the physical kind for the call's sampling mode and the twin is registered (sm_103a: the `wide` unit with the sampled first threshold off); the plan, the knobs and the workspace stay |
| tile-loop unroll | `tile_unroll_factor` when `tile_unroll_rule` names the program kind and `top_k` reaches its floor (yielding to the snake order where the rule says so) |
| sampled first threshold on buffer-fitting units | when the mean unit tiles are at most `sample_fit_max_mean_tiles`, the mean segment offset leaves fitting units and the program kind is one of `sample_fit_kinds` (the kinds whose sampled-fit program the architecture registers; the split programs, for example, run their plain form on every call) |
| sampled tiles and rank shift | tiles: `sample_tiles_tiny_units` up to `sample_dispatch_tiny_tiles_max` mean unit tiles, `sample_tiles_short_units` up to `sample_dispatch_mean_tiles_max`, else `sample_tiles_max`.  Shift: `sample_shift_rule[1]` permille where the record sets `sample_shift_rule` and `top_k >= sample_shift_rule[0]`, else `sample_shift_permille`.  Where the record sets `sample_class_rule` (sm_100a, sm_103a: `[8, 500]`; sm_107a: `[32, 500]`), the wide-unit programs take its shift on packed calls, and the tiny dispatch class (packed calls of at most 4096 queries per segment and at most `sample_dispatch_tiny_tiles_max` mean unit tiles) takes its shift, and its tile count where the mean unit holds more tiles than that count.  Where the record sets `sample_long_rule` (sm_107a: `[["l6"], 500]`), a call of more than `sample_dispatch_mean_tiles_max` mean unit tiles whose physical kind the rule names takes its shift.  On sm_107a these two rules keep every rank input of flashinfer-ai/flashinfer#7549 at or below its time with sampling off.  Explicit knobs override all of these.  Whether sampling is on enters the rules above.  The tile count and the shift are launch scalars only: they set how conservative the sampled first threshold is (a threshold that proves too aggressive is streamed again exactly), never the program, the capacity or the selected set.  Unsampled calls do not read the shift |
| candidate capacity | `candidate_multiplier` x `top_k` (`cand_mult_rule[1]` for long units), rounded up to whole tiles.  It is raised to `cand_cap_floor` slots, or to `cand_cap_floor_rule[1]` slots from `top_k >= cand_cap_floor_rule[0]` on where the record sets that rule.  On `sm_100a` the 16384-slot floor applies from `top_k = 1024` on and smaller `top_k` keep 8192; `sm_103a` keeps its 16384-slot floor on every `top_k`.  It is lowered to `cand_cap_ceiling` slots where the record sets one: the multiplier becomes `ceiling // top_k`, so the buffer stops growing with `top_k` once it holds the ceiling.  With the sampled first threshold off (`sample_tiles_max=0`), the `candidate_multiplier_unsampled` / `cand_mult_rule_unsampled` / `cand_cap_floor_unsampled` / `cand_cap_ceiling_unsampled` values apply instead.  An explicit `candidate_multiplier` knob replaces the multiplier |
| check period | default `min(check_period_max, (cand_cap - top_k) // 128, cand_cap // check_period_cap_divisor)`, at least 1, or the kind's `check_period_kind_overrides` value clamped to the exactness bound; an explicit `check_period` must lie in `[1, limit]` (the kernel's headroom rule) |

The dispatched programs must be registered for the device's architecture
(`cake_jit.PROGRAM_KEYS[arch]`).  This covers the scan and, on a split call,
the merge.  Both are looked up before anything is allocated.  Otherwise the
backend raises `NotImplementedError` naming the program key; it never runs a
different program.  The kernel source's own dispatch functions are the
reference for every rule.  The Cake exporter verifies the record against them
before the record is frozen.

## Workspace and host behaviour

* **Workspace bound**
  (`cake_dsa_indexer_nondet_train_topk_workspace_size(T, Tkv, S, top_k=,
  ratio=, device=)`).  It is the sum of three parts:
  * `grid_ctas x queries_per_unit x candidate_capacity(top_k) x 8` bytes of
    persistent candidate buffers for the dispatched unit geometry and
    capacity.  For example, this is about 74 MiB at `top_k = 2048` with four
    queries per unit on 148 SMs at the 16384-slot capacity floor (the floor
    `sm_100a` applies from `top_k = 1024` on), and 148 MiB with eight queries
    per unit.
  * `T x n_split x top_k x 8` bytes of staging when the dispatch splits the
    key ranges.
  * An 8320-byte tail that every plan carries: 16 int64 for the ticket word,
    then 1024 int64 for the longest-first segment order that the scan
    prologue writes.

  This is the deterministic package's bound for the same BF16 call.  The
  bound depends on the call geometry.  Size a reused buffer for the largest
  bound over the geometries it serves.  The bound is the larger of the
  sampled default's plan and the plan with the sampled first threshold off
  (`sample_tiles_max=0`), so a buffer sized by it serves both modes.  The
  backend's function of the same name also takes the `grid_ctas` and
  `candidate_multiplier` knobs; size a buffer for a call that sets them with
  the same values.  The workspace is separate from the inputs and the two
  `[T, top_k]` outputs.  A buffer below the call's bound is rejected
  (`workspace_buffer needs N bytes`), and so is a buffer that is not 8-byte
  aligned.  Computing the bound never consults
  `torch.use_deterministic_algorithms`.
* **No host synchronization**.  The segment boundaries and offsets are read
  on the device only.  The host uses `T`, `Tkv`, `S`, `ratio`, `top_k`,
  `max_seqlen_k`, the SM count and the strides.  The eager entry point
  allocates the outputs and the workspace (unless supplied) through the
  caching allocator.  A call without keys fills its outputs with padding (two
  fills) instead of launching.
* **Ticket tail**.  On all three architectures (`TICKET_TAIL_ARCHES`), the
  scan programs draw every work unit from a ticket word.  A record's
  `static_scan_kinds` would name the kinds whose programs keep the static
  unit enumeration and get no tail; no record names one.  So a call that
  shares the GPU with another stream slows with the SMs that stream holds,
  not by a whole extra wave of work units (flashinfer-ai/flashinfer#7548).
  Every completed scan launch leaves the word at 0.  How the word is reset
  depends on the workspace:
  * Backend-owned workspace: the tail is zeroed once, when the runner is
    prepared.  The eager entry prepares a one-shot runner, so this happens
    once per eager call.
  * Caller-owned `workspace_buffer`: the scan binding zeroes the tail with
    `cudaMemsetAsync` on the launch stream right before every scan launch, in
    the same host call.  A CUDA graph that captures a run carries that memset
    node.
  * Static call: nothing is zeroed.  Here the shape alone bounds the work
    units by the launched grid (`cake_policy.ticket_static_path`), so the
    scan never touches the ticket word.
  * On `sm_107a`, a scan program listed in `ticket_arg_arches` draws from a
    16-byte ticket block passed as an argument instead of the tail.  A
    prepared runner owns such a block, zeroed once at prepare.  The eager
    entry passes the first 16 bytes of the workspace's tail instead, and the
    scan binding zeroes its ticket word right before the launch.  The `:lsf`
    copies of such programs still keep the segment order in the tail; every
    launch writes it before it reads it, so it needs no zeroing.
  * A run that raised while launching marks the tail dirty, and the next run
    zeroes it first.
* **Prepared runner**.  `cake_backend.prepare_dsa_indexer_nondet_topk(...)`
  validates one binding, decides its programs and allocates the missing
  outputs and the workspace.  These are the backend's only allocations.  The
  returned `DSAIndexerNondetRunner` launches with no allocation and no host
  synchronization.  It can be captured into a CUDA graph by the caller, and a
  replay reads new values written into the bound tensors.  Prepare a new
  runner when a shape, dtype, stride or tensor binding changes.  Every
  `run()` that launches checks the deterministic-algorithms setting first
  (see "Mode and names").
* **Launch structure**: the scan, then the merge on a split call (see above).
  No `[T, Tkv]` matrix is materialized.

## Layout of this package

The Cake exporter produces the kernel and launcher sources under `csrc/`, the
registry `cake_jit.py` and `build.json`.  Every program is one of the
deterministic package's BF16 scan programs or its plain split merge.  It is
exported under this package's prefix as `cake_dsa_indexer_nondet_topk_<20 hex>`
and keeps the twenty-hex identity of the deterministic program it comes from.
Programs that differ only in a few constants share one kernel translation
unit.  Programs whose launchers differ only in the kernel symbol, the
host-shim namespace and the launch constants share one binding.  Each
`build.json` entry selects its program through its own `-D` definitions.
`build.json` records every source's digest and a closure digest per entry.
`csrc/.clang-format` disables formatting so the digests stay valid.

* `cake_jit.py` is the registry.  It holds:
  * the stage roles `scan` and `merge` only (`STAGES`) and their argument
    plans (`ARG_PLANS`, the deterministic package's, under
    `ABI = "dsa_indexer_v5"`).  The scan's `K_scale` slot holds the
    deterministic package's FP8 per-key scales; here it gets a placeholder
    that no program of this package reads;
  * one record per program (`PROGRAMS`: role, the architecture-free `library`
    stem, architectures, and `ticket_arg_arches` where a scan program draws
    from its ticket block);
  * the program of every dispatch key per architecture (`PROGRAM_KEYS`):
    `scan:<kind>:u<unroll>:s<snake>:f<sampled fit>[:lsf]` and `merge`, with
    no finalize, rank or FP8 keys;
  * the per-architecture dispatch policy (`POLICY`), `NUMERICS` and
    `TICKET_TAIL_ARCHES`;
  * the loader glue: `library_name(program, arch)` gives the `build.json`
    entry `<library>_<arch>`, and `load_program(program, arch)` loads it
    through `cake_build.load_library`.

  The launch geometry of a program (block, cluster, dynamic shared memory) is
  fixed in its binding.
* `cake_build.py` + `build.json`: `cake_build.py` is the loader that every
  Cake-exported FlashInfer package shares, byte for byte.  `build.json` is
  this package's build manifest (schema `cake-build/1`) and the only record of
  what the package compiles.  It has one entry per program and architecture
  with its package-relative sources, nvcc flags, closure digest, per-source
  SHA-256 and `aot/<name>.so` path.  It also has the common nvcc / link flags
  and the `tvm-ffi` requirement.  `cake_build.load_library(name)` loads the
  prebuilt library when `aot/` holds it, checked against the builder's
  `aot/build_info.json` and the library's own `build_cuda_version()`.
  Otherwise it JIT-builds the entry through FlashInfer from the same sources
  and flags.  The prebuilt path never imports `flashinfer`.
* `csrc/cake_ffi_checks.h` holds the tvm-ffi / DLPack include set, the named
  tensor-argument checks and `cake_ffi_build_cuda_version()`, which the
  generated launchers use.  The package compiles without FlashInfer's
  `csrc/`.
* `cake_policy.py` is the dependency-free dispatch policy: program choice,
  capacity, check period, split count, the ticket static path and the
  workspace bound.
* `cake_backend.py` holds validation, planning, the deterministic-algorithms
  guard, argument binding, the prepared runner and the eager entry point.
* `csrc/cake_dsa_indexer_nondet_topk/` holds the generated kernel and binding
  translation units:
  * one kernel per program, or per group of programs that differ only in
    their constants (a scan lever-twin group);
  * one binding per launcher text: the 59 scan programs share
    6 bindings, and the merge program has its own.

  A shared kernel source names its kernel `CAKE_PROGRAM_KERNEL` and reads its
  constants (tile unroll and snake order) from
  `CAKE_INDEXER_<KNOB>`.  A shared binding names the kernel, its host-shim
  namespace, block size and dynamic shared memory through
  `CAKE_KERNEL_SYMBOL`, `CAKE_PROGRAM_SHIM`, `CAKE_BINDING_THREADS` and
  `CAKE_BINDING_SMEM`.  Each of these has an `#ifndef` default naming the
  program the file was exported for.  Every other program's `build.json`
  entry passes its own values with `-D`.

## Tests and benchmark

There are four test files:

* `tests/experimental/test_cake_dsa_indexer_nondet_train.py`: the backend
  suite;
* `tests/experimental/test_cake_dsa_indexer_nondet_train_vs_det.py`: the
  comparison with the deterministic package;
* `tests/experimental/test_cake_dsa_indexer_nondet_train_api.py`: the
  `flashinfer.cake_dsa_indexer_nondet_train` entry points;
* `tests/experimental/test_cake_dsa_indexer_nondet_train_names.py`: the #8928
  naming rules.

They share the independent FP64 / FP32 reference in
`tests/experimental/test_helpers/cake_dsa_indexer_reference.py` with the deterministic
package.  Results are compared after each row is canonicalized: valid slots
sorted by id with their score bits, padding last.  The comparisons therefore
check sets, not slot order.  Coverage:

* registry and policy host tests: no ordering, rank or FP8 keys; a split plan
  runs the scan, then the merge; an unsplit plan runs the scan only;
* validation;
* accuracy against the reference for the exact semantic cases;
* packed causality;
* random inputs;
* near ties;
* the recorded token counts;
* the ratio-4 pooled top-512 selection;
* `top_k = 4096`;
* set repeatability over five calls (the pairs and the valid count);
* split versus unsplit set equality, a forced split whose rows hold padding
  and a forced split that cuts a tie group;
* non-finite inputs;
* the workspace bound, the ticket tail and the ticket word;
* CUDA graphs and the absence of host synchronization;
* the deterministic-algorithms guard.

The comparison file checks that every row equals the deterministic package's
row in another slot order (ids, score bits, valid count and padding), and
that the registry keys, policy records and host plans of BF16 calls are the
deterministic package's.

Benchmark: `benchmarks/bench_cake_dsa_indexer_nondet_train.py` times this
package against the deterministic package on the same rows.  It compares their
sets, not slot-by-slot bits, and fails on any set mismatch.
