# Cake DSA indexer top-k backend (SM100 / SM103 / SM107)

Experimental backend for the fused DeepSeek Sparse Attention indexer of a
training stack: scoring of every visible key plus the deterministic exact
top-k selection with aligned scores, in one device-side pipeline (GLM-5.2
geometry: 32 indexer heads, head dimension 128, top-k 2048).  Tracking issue:
flashinfer-ai/flashinfer#5676.

Both the API and the backend are experimental: no compatibility guarantee.
SM107 (Rubin) is the primary target of the request, SM100 (B200 / GB200) the
comparison architecture; SM103 (B300 / GB300) is compiled from the same
sources.  The feature is JIT-only and does not participate in automatic
backend selection, autotuning or trace-apply.

## Public entry points (`flashinfer/dsa_indexer.py`)

```python
from flashinfer.dsa_indexer import dsa_indexer_topk, dsa_indexer_topk_workspace_size

indices, scores = dsa_indexer_topk(q, k, w, cu_seqlens_q, cu_seqlens_k,
                                   top_k=2048, softmax_scale=None, q_causal_offsets=None, ratio=1,
                                   max_seqlen_q=None, max_seqlen_k=None,
                                   workspace_buffer=None, indices=None, scores=None)
nbytes = dsa_indexer_topk_workspace_size(top_k=2048, device=q.device)
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
| `max_seqlen_q`, `max_seqlen_k` | optional ints | host mirrors; accepted, not read by any launch decision |
| `workspace_buffer` | optional uint8 tensor | at least `dsa_indexer_topk_workspace_size(top_k, device)` bytes, 8-byte aligned |
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
rounding; no fast-math (IEEE round-to-nearest, no flush-to-zero).  The generated
program evaluates it in one fixed sequence, so the score of a pair is a function
of `(q[t], k[j], w[t], softmax_scale)` alone and does not depend on which CTA,
tile or warp evaluates it:

1. **Dot product**: one tensor-core MMA of a fixed shape (128 keys x 128 (4
   queries x 32 heads) x K = 16) accumulates the 128 products of a (query,
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

**Signed zeros.**  What the program computes: every chain accumulator is
initialised to `+0.0` and the FMA order is fixed, and `fma(+0, w', +0) = +0`
for either sign of `w'`, so a row whose head terms are all zero yields `+0.0`;
a `-0.0` score cannot arise from this reduction (registry field
`numerics["zero_sign_policy"] = "positive_accumulator"`).  Ordering rule:
`+0.0` and `-0.0` compare equal, an equal-score tie goes to the larger key id,
and the written score keeps the computed bits -- it is never canonicalised.
The reference in `tests/test_helpers/cake_dsa_indexer_reference.py` follows
the same reduction model, so the bit-exact test cases compare like with like.

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
* Identical inputs give identical ids and score bits across calls, and the
  internal partition knobs of the backend (`grid_ctas`, `candidate_multiplier`,
  `check_period`, `sample_tiles_max`, `sample_shift_permille`,
  `finalize_stage`) never change a result (the tests compare bitwise).

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

## Workspace and host behaviour

* **Workspace bound** (`dsa_indexer_topk_workspace_size`): `grid x
  queries_per_cta x candidate_capacity(top_k) x entry_bytes` with `grid` the
  device's SM count (the persistent scan grid), `candidate_capacity(top_k) =
  max(4 top_k, top_k + 128)` rounded up to 128 and 8-byte packed entries; for
  example about 37 MiB at `top_k = 2048` on 148 SMs (53 MiB on 212 SMs), 74 /
  106 MiB at `top_k = 4096`, 4.6 / 6.6 MiB at `top_k = 256`.  Independent of
  `T`, `Tkv` and `S`; separate from the inputs and the two `[T, top_k]`
  outputs.  The constants come from the registry record (`gate_policy`), not
  from host-side assumptions.
* **No host synchronization**: the segment boundaries and offsets are read on
  the device only; the host uses `T`, `S`, `top_k`, the SM count and the
  strides.  The eager entry point allocates the outputs and the workspace
  (unless supplied) through the caching allocator; the prepared runner
  (`cake_backend.prepare_dsa_indexer_topk`) allocates nothing at launch and
  is CUDA-graph capturable.
* **Launch structure**: two kernels -- the persistent fused scan (grid = SM
  count) and a per-row ascending-id finalize (grid = `T`; the 256-thread
  program up to `top_k = 2048`, the 512-thread program above).  No
  `[T, Tkv]` matrix is materialized.

## Layout of this package

* `cake_jit.py` -- `MODULES` registry (one record per architecture, filled by
  the generated-program export), stage names and the JIT specs.
* `cake_backend.py` -- validation, gate policy and workspace bound, argument
  binding, the prepared runner and the eager entry point.
* `csrc/cake_dsa_indexer_topk/<arch>/` -- generated kernel and binding
  translation units (`.clang-format` disables formatting: the sources are
  identity-checked by the registry's closure digests).

## Status

**Placeholder registry.**  `MODULES` is empty until the generated programs for
`sm_100a`, `sm_103a` and `sm_107a` are exported from the kernel snapshot named
in the pull request.  Importing the package and the public entry point works;
`cake_jit.select_module(arch)` raises `NotImplementedError` naming the issue,
`cake_backend.generated_program_available(device)` returns `False`, a call
raises the same `NotImplementedError`, and the operator tests skip.  The
host-only tests (registry shape, gate policy, validation, binding, the
reference itself) run without a GPU.

Tests: `tests/experimental/test_cake_dsa_indexer.py` with the independent
FP64 / FP32 reference in `tests/test_helpers/cake_dsa_indexer_reference.py`
(exact semantic cases incl. cutoff ties, all-equal scores, mixed signed zeros
at the boundary with bit checks, negative scores and a few winners over a
large tie; packed causality incl. explicit positive and negative offsets,
`ratio = 2`, empty and singleton segments, short rows, `top_k = 1` and
`4096`; random-input accuracy under the `gamma_n A` bound; near ties; both
recorded token counts; tile-boundary neighbours; changing shapes; bitwise
repeatability and partition-knob invariance; strided `k`; non-finite inputs;
CUDA-graph capture; the workspace bound).  Benchmark:
`benchmarks/bench_cake_dsa_indexer.py` (paired AB / BA / interleaved CUPTI
spans of the complete operator against two torch + FlashInfer compositions;
both baseline arms are reconstructions of the training stack's chunked
scoring / coarse top-k / rescoring path written for this benchmark, not the
stack's own kernels, so their absolute numbers are speed references only).
