# Cake dense projection GEMM backend (SM100 / SM103 / SM107)

Experimental backend for the GLM-5.2 training GEMMs: the BF16 dense projection
GEMM (K1 -- BF16 operands, FP32 accumulation, BF16 or FP32 output, strided
views, ragged token counts, batched head projections) and the FP32 router-gate
GEMM through split-BF16x3 tensor-core emulation with chunked FP32 promotion
(K2).  Tracking issue: flashinfer-ai/flashinfer#5677.

Both the API and the backend are experimental: no compatibility guarantee.
SM107 (Rubin R200) is the primary architecture; SM100 (B200 / GB200) and
SM103 (B300 / GB300) are compiled from the same sources.  The feature is
JIT-only, has no core entry point yet (importing this package is the
explicit opt-in) and does not participate in automatic backend selection,
autotuning or trace-apply.

## Entry points (`cake_backend.py`)

```python
from flashinfer.experimental.dense_projection_gemm import cake_backend as gemm

# K1: strided-view GEMM  out[l, m, n] = sum_k A[l, m, k] B[l, k, n]
gemm.dense_projection_gemm(A, B, out, transposed_out=False)
gemm.projection_forward(X, W, out=None)               # out[T, N] = X[T, K] @ W[N, K].T
gemm.projection_dgrad(G, W, out=None, out_dtype=None)  # out[T, K] = G[T, N] @ W[N, K]
gemm.projection_wgrad(G, X, out=None, out_dtype=None)  # out[N, K] = G[T, N].T @ X[T, K]
prepared = gemm.prepare_dense_projection_gemm(A, B, out, transposed_out=False)
prepared = gemm.prepare_projection_wgrad(G, X, out)
prepared.launch()   # allocation-free; prepared.template / .grid / .module_name / .plan

# K2: FP32 router GEMM through split-BF16x3 emulation
gemm.router_fp32_gemm(A, B, out, splits=None)
gemm.router_forward(X, W, out=None, splits=4)          # out[T, N] = X[T, K] @ W[N, K].T
gemm.router_dgrad(G, W, out=None)                      # out[T, K] = G[T, N] @ W[N, K]
gemm.router_wgrad(G, X, out=None, splits=11)           # out[N, K] = G[T, N].T @ X[T, K]
prepared = gemm.prepare_router_fp32_gemm(A, B, out, splits=None)
prepared.launch()   # re-splits B, launches, reduces the split-K partials; .splits
```

### K1 contract

* `A` BF16 `[L, M, K]` (or `[M, K]`) with unit stride on `m` or `k`; `B`
  BF16 `[L, K, N]` (or `[K, N]`) with unit stride on `k` or `n`.  The other
  matrix stride and the batch stride are multiples of 8 elements (16 bytes);
  every view starts at a 16-byte aligned address.
* `out` BF16 or FP32 `[L, M, N]` / `[M, N]` with unit stride on `n` and
  16-byte aligned rows and batches, or -- with `transposed_out` -- `[L, N, M]`
  / `[N, M]` with unit stride on `m` (element `(m, n)` lands at `out[l, n,
  m]`).  The caller's view is written in place; nothing is copied or
  re-laid-out.
* `M, K >= 1` and `N` a positive multiple of 8 are runtime values (no
  padding: TMA zero-fills out-of-bounds boxes, the epilogue masks rows `>= M`
  and columns `>= N`); `L` and the batch strides are arbitrary, so the
  token-major head views of the MLA projections (`[T, H, D]` permuted to
  `[H, T, D]`, `D` a slice of a wider slot) run without copies.
* The three training layouts map onto the operator: forward `X @ W.T`
  (`A` K-major, `B = W.T` K-major), input gradient `G @ W` (`B` MN-major),
  weight gradient `G.T @ X` (both MN-major, `K = T` ragged).  A weight
  gradient with fewer than 256 output rows runs the swapped `X.T @ G` with
  the transposed store (`projection_wgrad` / `prepare_projection_wgrad`).
* Numerics: BF16 x BF16 products exact in FP32, tensor-core FP32
  accumulation, one rounding at the store.  BF16 outputs match an FP64
  reference to `atol = rtol = 1e-2` elementwise; FP32 outputs carry the
  accumulation error (`rel_fro <= 5e-5`, `max_abs <= 1e-3 * max|ref|`).

### Host planning (mirror of the Cake launcher)

The host selects one traced kernel instance per (A layout, B layout, tile
width, tile height, output kind, epilogue path, raster group, TMA L2 eviction
hints); every
rule is a verbatim copy of the Cake kernel host's (each function names its
source line range).  Two device facts enter the plan: the SM count (CTA pairs:
stream-K split, raster group, wave working set) and the L2 size (hint gate);
`prepare_*` reads both from the device (`device_sm_count`, `device_l2_bytes`
= `torch.cuda.get_device_properties(...).L2_cache_size`, falling back to
`cuDeviceGetAttribute`), `plan_dense_projection_gemm` takes them as inputs:

| choice | rule |
|---|---|
| operand layout | `operand_view`: the contraction axis contiguous -> K-major, else the M/N axis contiguous -> MN-major |
| `BLOCK_N` | `default_block_n`: 128 when `N <= 128`, 192 when `N` is a multiple of 192 but not of 256 (`N = 576`: three exact tiles), else 256.  The menu is `BLOCK_N_CHOICES = (128, 160, 192, 224, 256)`: 160 / 224 (Cake round 9) are wave-quantization fits a rule or caller selects; their 80 / 112-column warp slices take the register epilogue |
| `CTA_ROWS` | `default_cta_rows`: 128.  The 256-row tall-tile family (`cta_rows=256`, symbol `_m256`) and the 64-row Layout-B family (`cta_rows=64`, Cake round 9: 128 x `BLOCK_N` pair tiles through the M=128 `cta_group::2` MMA, four TMEM buffers; symbol `_m64`) are rule or caller selections (`CTA_ROWS_CHOICES = (64, 128, 256)`).  A caller-forced `cta_rows` outside the row's rule (or default) family drops the rule's `stages` / `slots` / `epi`; its other knobs still apply |
| epilogue | `epi_mode`: a warp slice is `epi_cols = BLOCK_N / 2` columns (`BLOCK_N / 4` on 64-row tiles; always a multiple of 16).  Transposed output -> register stores (`reg`); row-major fp32 -> TMA stores (`tma`) when the slice is whole 32-column chunks (`epi="reg"` is an opt-in float4 register path; 160 / 224-wide and 64-row 192-wide slices have no TMA path); row-major bf16 -> `tma` when `K <= 1024` and the slice is whole 64-column chunks (`BLOCK_N = 256` at any height, 128 at 128 / 256 rows), else `reg`; `quad_store` (rule-selected, `_q`) is the bf16 `reg` variant for whole-64-column slices that transposes 32-byte row segments across lane quads so each lane issues one 256-bit store per 4-row group |
| staging slots | `epi_slots`: 1 with the TMA-store epilogue (2 only when `K <= K_TWO_SLOTS = 0` and the slice's chunk count is even), 0 otherwise |
| stages | `default_stages(slots, cta_rows, block_n, b_mn)`: 7 / 6 / 5 for 0 / 1 / 2 slots at `BLOCK_N = 160 / 192 / 224 / 256`, 9 / 8 / 6 at `BLOCK_N = 128`, 4 / 4 / 3 for tall (256-row) tiles; 64-row tiles take the deepest pipeline that fits the 227 KiB opt-in beside the staging (32 KiB per slot), at most 12 (9 / 8 / 6 at `BLOCK_N = 256`, 12 / 12 / 10 at 128; `b_mn` sizes the MN-major B panels).  A rule or caller `stages` is bounded by `smem_limit_for(arch)`: the 232448 B opt-in, or 334336 B on `sm_107a` (CUDA 13.4 oversized shared-memory mode, `SMEM_OVERSIZED`); the limit is not part of the instance key, so an instance has one symbol on every architecture |
| raster group | `default_group_m(a_mn, b_mn, m_tiles, pair_tiles, pairs)`: 16 CTA row tiles per cluster-launch-control raster group (a launch parameter since round 11, not an instance-key field) (even, so the two CTAs of a pair are the row halves of one 256-row tile) on every row; a 4-row group for one-to-two-wave weight-gradient rows measured no gain in the production configuration and is not applied.  A non-default group would carry the symbol suffix `_g<n>` |
| L2 hints | `default_hints(a_mn, b_mn, m_tiles, n_tiles, K, group_m, pairs, l2_bytes)`: no hint while one wave's operand panels fit the L2 (`wave_working_set(m_tiles, n_tiles, K, group_m, pairs) <= l2_bytes`); otherwise `A` streams `evict_first` when there is one column tile (`n_tiles == 1`); every other row gets no hint (reuse-ratio hints on the weight-gradient class measured 0..-4 percent in the production stream-K configuration).  Symbol suffix `_h<a><b>` (first letters): `_hen` = A evict_first, B none |
| L2 promotion, prefetch | `default_promo` = `none`, `default_pf` = 0 (measured, off); the symbol carries `_<promo>` / `_pf<n>` when set |
| template | `instance_symbol(instance_key(...))`, e.g. `dense_proj_gemm_kk_n256`, `dense_proj_gemm_kk_n128_hen`, `dense_proj_gemm_kn_n256_f32_tma1`, `dense_proj_gemm_nn_n256_f32_tma1`, `dense_proj_gemm_nn_n128_hen_t`, `dense_proj_gemm_kk_n256_m64`, `dense_proj_gemm_kk_n256_m256_ov_ht`, `dense_proj_gemm_nn_n160_m256_bz64`, `dense_proj_gemm_nn_n128_hen_t_skx`, `dense_proj_gemm_kn_n256_tma1_bg8`, `dense_proj_gemm_kn_n256_f32_v8_ef_s9` (round 13: `_pk` parked / `_ov` overlapped single-TMEM-buffer / `_ht` half-height-tail-wave tall epilogues; `htail` replaces the stream-K tail plan by 2 x tail whole-K half items when they fit the CTA pairs, so `ws_f32_elems = 0` and `sk_iters = sk_units * k_blocks`; `_bz64` = 32-column MN-major B panels (`b_swz` 64, smaller stages, deeper default pipeline), `_skx` = exact p-way K split of the tail tiles (`sk_exact`, pre-empts `sk_parts` / `htail`, p travels in `sk_iters`); `_bg8` = eight adjacent batch entries (heads) interleaved across the CTA pairs (`batch_group`, batched MLA rows only, 0 for a batch of one); `_ef` = fp32 v8 register stores with `L1::no_allocate.L2::evict_first` (`store_ef`), `_s9` = nine pipeline stages (oversized dynamic shared memory on sm_107a)) |
| grid | `(num_full + sk_units) * 2` CTAs: one cluster of two CTAs per `2 * CTA_ROWS` x `BLOCK_N` output tile plus the stream-K units |
| stream-K | `stream_k_plan(pair_tiles, k_blocks, sm_count // 2, sk="auto")`: only when the whole problem is one partial wave of CTA pairs whose halves fit the pairs (`pair_tiles <= pairs`, `2 * tail <= pairs`) and `k_blocks >= 16`, every tile is split into two K-aligned halves (`sk_units = 2 * pair_tiles`, `iters_per_unit = ceil(k_blocks / 2)`) with an in-kernel deterministic fixup; otherwise every work item is a whole tile (multi-wave tails stay data-parallel).  `sk=True` is the linearised-units policy (up to `pairs` units of >= 8 K steps), `sk="tiles"` the whole-tile diagnostic, `sk=False` off.  A measured `sk_parts = p` rule, or the caller knob `sk_parts`, instead splits every tail tile `p` ways when `p * tail` units fit the pairs and each keeps >= 8 K steps (`sk_parts_plan`).  The host asserts `tail_tiles * 16 <= 4096` slice counters |

`plan_dense_projection_gemm(A, B, out, sm_count=..., l2_bytes=...)` exposes
the plan (`GemmPlan`, including `wave_working_set_bytes`) without a device;
the prepared launch resolves the template
through `cake_jit.KERNELS[arch][template]`, owns its stream-K partial slabs
(`(sk_units + tail_tiles) * 2 * CTA_ROWS * BLOCK_N` fp32), its slice counters
(`max(8192, tail_tiles * 16)` u32: the lane-spread dummy counters above index
4096 must be addressable), and binds the
module's argument plan by keyword (`bind_launch`, fails closed on an unknown
name).  Given the same device the result is bitwise identical to the Cake
launcher's.

### K2 contract

* `A`, `B`, `out` FP32 2-D views: `A [M, K]` with unit stride on `m` or `k`
  (other stride a multiple of 4), `B [K, N]` with unit stride on `k` or `n`,
  `out [M, N]` with unit inner stride and 16-byte rows; `N` a multiple of 8.
* The operand layouts select the instance and the production split-K count:
  `kk` (forward, 4 splits), `kn` (input gradient, 1), both MN-major -> the
  swapped transposed-store weight gradient `nn_t` (11 splits; `out.T = B.T
  A.T`).  An MN-major `A` with a K-major `B` has no instance.
* Every `launch()` re-splits the second operand into the retained
  `[3, outer, inner]` BF16 stack in the operand's own layout (exact three-way
  split, no allocation), launches, and sums the split-K partials on the host
  in a fixed order.  Splits, chunk boundaries and the reduction order are
  fixed: bit-exact run to run.  Error against an FP64 reference: FP32-class
  (`rel_fro` about 2e-7 to 4e-7 on the GLM-5.2 router shapes).

## Layout of this package

* `cake_jit.py` -- `MODULES` (one record per generated program; `arches` lists
  the architectures its single source pair serves) and `KERNELS` (`arch -> template -> module name`), both
  filled by the generated-program export, and the JIT specs.
* `cake_backend.py` -- view validation, instance selection, stream-K
  planning, argument-plan binding, the prepared launches and the eager entry
  points.
* `csrc/cake_dense_projection_gemm/` -- generated kernel and binding
  translation units (`.clang-format` disables formatting: the sources are
  identity-checked by the registry's closure digests).

## Status

The registries hold the generated programs of the current export lock: the
Cake exporter (`exports/dense_projection_gemm` of the Cake repository) emits
one program per template, shared by `sm_100a` and `sm_107a` wherever the generated text is the same (the per-architecture
template lists are `KERNELS` in `cake_jit.py`; `K1_TEMPLATES` of the exporter
is the source of truth and is regenerated whenever `ROW_RULES` resolve a new
instance), and `sm_103a` is compiled from the same sources when a B300 route
is exercised.  An entry point raises `NotImplementedError` naming the missing
instance only for a template that no export has generated.

CUDA Graphs: prepare outside capture and replay the prepared launch
(`prepare_*` once, `launch` many times).  The eager wrappers
(`dense_projection_gemm`, `projection_wgrad`, `router_fp32_gemm`) prepare a
new launch (a new stream-K workspace and counters) on every call; capture a
prepared launch object rather than the eager wrappers.

Tests: `tests/experimental/test_cake_dense_projection_gemm.py` (host-side
planning tests run everywhere; the GPU tests skip without a registered
instance or a compute capability 10.0 / 10.3 / 10.7 device).  Benchmark:
`benchmarks/bench_cake_dense_projection_gemm.py` (the 11 GLM-5.2 projection
rows x fwd / dgrad / wgrad x T in {16231, 16172}, the router rows and the two
batched MLA rows, paired against `torch.matmul`).
