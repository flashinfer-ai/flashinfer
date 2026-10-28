# Cake dense projection GEMM backend (SM100 / SM107)

Experimental backend for the GLM-5.2 training GEMMs: the BF16 dense projection
GEMM (K1 -- BF16 operands, FP32 accumulation, BF16 or FP32 output, strided
views, ragged token counts, batched head projections) and the FP32 router-gate
GEMM through split-BF16x3 tensor-core emulation with chunked FP32 promotion
(K2).  Tracking issue: flashinfer-ai/flashinfer#5677.

Both the API and the backend are experimental: no compatibility guarantee.
SM107 (Rubin R200) is the primary architecture; SM100 (B200 / GB200) is built
from the same sources; SM103 (B300 / GB300) ships its own builds of the
same programs and, for three input-gradient rows, the eight-warp CTA form of
the GEMM (`cake_jit.KERNELS` lists the registered architectures).  A device whose architecture the registry does not
list raises `NotImplementedError` naming the missing instance.  The feature has
no core entry point yet (importing this package is
the explicit opt-in) and does not participate in automatic backend selection,
autotuning or trace-apply.  Its libraries are loaded prebuilt from `aot/` when
the package ships them and JIT-built by FlashInfer from `build.json` otherwise
(see "Layout of this package").

## Entry points (`cake_backend.py`)

```python
from flashinfer.experimental.dense_projection_gemm import cake_backend as gemm

# K1: strided-view GEMM  out[l, m, n] = sum_k A[l, m, k] B[l, k, n]
gemm.dense_projection_gemm(A, B, out, transposed_out=False)   # convenience path (cached plan)
gemm.projection_forward(X, W, out=None)               # out[T, N] = X[T, K] @ W[N, K].T
gemm.projection_dgrad(G, W, out=None, out_dtype=None)  # out[T, K] = G[T, N] @ W[N, K]
gemm.projection_wgrad(G, X, out=None, out_dtype=None)  # out[N, K] = G[T, N].T @ X[T, K]
prepared = gemm.prepare_dense_projection_gemm(A, B, out, transposed_out=False)
prepared = gemm.prepare_projection_wgrad(G, X, out)
prepared.launch()   # allocation-free; prepared.template / .grid / .module_name / .plan

# K2: FP32 router GEMM through split-BF16x3 emulation
gemm.router_fp32_gemm(A, B, out, splits=None)                  # convenience path (cached plan)
gemm.router_forward(X, W, out=None, splits=4)          # out[T, N] = X[T, K] @ W[N, K].T
gemm.router_dgrad(G, W, out=None)                      # out[T, K] = G[T, N] @ W[N, K]
gemm.router_wgrad(G, X, out=None, splits=6)            # out[N, K] = G[T, N].T @ X[T, K]
prepared = gemm.prepare_router_fp32_gemm(A, B, out, splits=None)
prepared.launch()   # re-splits B, launches, reduces the split-K partials; .splits
```

### Host paths

Both paths run the same planner once per binding and bind the same generated
module; they differ in who owns the plan and the workspace.

| | prepared path (`prepare_*` + `launch()`) | convenience path (`dense_projection_gemm`, `projection_*`, `router_*`) |
|---|---|---|
| planning | once, in `prepare_*` | once per cache key, on the first call |
| workspace (K1: stream-K slabs + slice counters; K2: BF16x3 stack, the torch-chain split's fp32 scratch, split-K partials) | owned by the prepared object | carved out of the workspace arena of the (device, stream): one growable buffer per kind shared by every cached entry of that device and stream, grown to the largest requirement seen, never per entry |
| per call | bind nothing: launch the bound views | rebind the caller's views (view objects only), resolve the workspace views of the arena (memoised until the arena grows), launch; `out` is allocated only when the caller passes none |
| device queries | none after `prepare_*` | none after the first call |
| cache | none | per process, per kind (K1 / K2), `CONVENIENCE_CACHE_SIZE` = 64 entries, least recently used first out; the arena of a (device, stream) is released with its last entry and by `clear_convenience_caches()` |
| memory held | the prepared object's workspace | bounded by the largest geometry seen per workspace kind, however many token counts the cache has planned (a ragged T adds entries, not workspace) |
| CUDA Graphs | capture `launch()` | do not capture (a later growth replaces, an eviction frees, the workspace a graph replays) |

The orientation of `projection_wgrad` / `prepare_projection_wgrad` follows the
row's measured rule when the rule table keys the row in exactly one orientation
(the swapped `X.T @ G` form with the transposed store, or the direct `G.T @
X`), else the launcher's padded-tile-work rule (N < 256 and N = 576 swap);
`wgrad_views(G, X, arch=..., out_f32=...)` exposes the decision.

The cache key is `(device index, raw stream, flags / knobs, geometry of A, B,
out)` where a view's geometry is its shape, strides, dtype, storage offset and
the 16-byte alignment residue of its first element -- never its address, so new
tensors of the same geometry hit (a training step with fresh activations) and
an entry never holds a caller's tensor.  Launches on different streams never
share a workspace (the stream is in the key and each (device, stream) has its
own arena); launches of one device and stream -- the only users of its arena --
are ordered by that stream, which is what makes one buffer per kind safe to
share: the K2 split rewrites the BF16x3 stack and its scratch in full before
every launch, the split-K partials are reduced into `out` right after the
launch, and growing a buffer replaces it while an earlier launch may still be
in flight on the old one -- safe because PyTorch's caching allocator is
stream-ordered (a freed block is handed out again only to a later allocation on
the stream it was freed on, or after a device-synchronising `cudaFree`).  The
stream-K slice counters are zero-initialised once and reused without a memset:
the kernel resets every counter it uses (the last arriver of a tail tile's slice
writes zero after the fixup), so the shared counter buffer is zero again when
each launch completes -- the same contract the Cake launcher's per-device
counters rely on.  Threads: planning and the arena bookkeeping (creation,
growth, reference counts, insertion, eviction, `clear_convenience_caches()`)
are serialised by a module lock (a miss re-checks the cache under it, so a key
is planned once; a growth keeps the larger buffer); the K2 hit path holds its
arena's lock from the carve of the workspace views through the split-K
reduction, because one router call is three stream operations (split, kernel,
reduction) on the shared BF16x3 stack and partials that two threads issuing on
one stream must not interleave; the K1 hit path (one self-contained kernel per
call) and the cache lookup take no lock.  The stream-ordering argument assumes
one issuer per stream at a time, as CUDA stream semantics do.  The device facts
a plan depends on (architecture tag, SM count, L2 size) are read once per
device index (`device_facts`).

### K1 contract

* `A` BF16 `[L, M, K]` (or `[M, K]`) with unit stride on `m` or `k`; `B`
  BF16 `[L, K, N]` (or `[K, N]`) with unit stride on `k` or `n`.  The other
  matrix stride and the batch stride are multiples of 8 elements (16 bytes);
  every view starts at a 16-byte aligned address.
* `out` BF16 or FP32 `[L, M, N]` / `[M, N]` with unit stride on `n` and
  16-byte aligned rows and batches, or -- with `transposed_out` -- `[L, N, M]`
  / `[N, M]` with unit stride on `m` (element `(m, n)` lands at `out[l, n,
  m]`).  The caller's view is written in place; nothing is copied or
  re-laid-out.  The row and batch strides the kernel receives (`ldo`, `out_l`,
  in elements; `rows` / `cols` = `M` / `N`, or `N` / `M` under `transposed_out`)
  satisfy `ldo >= cols`, `out_l >= cols` and one of two layouts: batch-major,
  `out_l >= rows * ldo` (each batch holds its matrix), or interleaved within a
  row, `ldo >= L * out_l` (a `[T, H, D]` projection written as `[H, T, D]`:
  `out_l = D`, `ldo = H * D`).  A batch of one passes its view's batch stride
  when that spans the matrix and the matrix span `rows * ldo` otherwise
  (`_out_strides`); any other layout is refused at launch.
* `M, K >= 1` and `N` a positive multiple of 8 are runtime values (no
  padding: TMA zero-fills out-of-bounds boxes, the epilogue masks rows `>= M`
  and columns `>= N`); `L` is arbitrary and the batch strides follow the
  layout rule above, so the token-major head views of the MLA projections
  (`[T, H, D]` permuted to `[H, T, D]`, `D` a slice of a wider slot) run
  without copies.
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
`prepare_*` and the convenience path read both once per device index
(`device_facts`: `device_sm_count`, `device_l2_bytes` =
`torch.cuda.get_device_properties(...).L2_cache_size`, falling back to
`cuDeviceGetAttribute`), `plan_dense_projection_gemm` takes them as inputs:

| choice | rule |
|---|---|
| operand layout | `operand_view`: the contraction axis contiguous -> K-major, else the M/N axis contiguous -> MN-major |
| `BLOCK_N` | `default_block_n`: 128 when `N <= 128`, 192 when `N` is a multiple of 192 but not of 256 (`N = 576`: three exact tiles), else 256.  The menu is `BLOCK_N_CHOICES = (128, 160, 192, 224, 256)`: 160 / 224 (Cake round 9) are wave-quantization fits a rule or caller selects; their 80 / 112-column warp slices take the register epilogue |
| `CTA_ROWS` | `default_cta_rows`: 128.  The 256-row tall-tile family (`cta_rows=256`, symbol `_m256`) and the 64-row Layout-B family (`cta_rows=64`, Cake round 9: 128 x `BLOCK_N` pair tiles through the M=128 `cta_group::2` MMA, four TMEM buffers; symbol `_m64`) are rule or caller selections (`CTA_ROWS_CHOICES = (64, 128, 256)`).  A caller-forced `cta_rows` outside the row's rule (or default) family drops the rule's `stages` / `slots` / `epi`; its other knobs still apply |
| epilogue | `epi_mode`: a warp slice is `epi_cols = BLOCK_N / 2` columns (`BLOCK_N / 4` on 64-row tiles; always a multiple of 16).  Transposed output -> register stores (`reg`); row-major fp32 -> TMA stores (`tma`) when the slice is whole 32-column chunks (`epi="reg"` is an opt-in float4 register path; 160 / 224-wide and 64-row 192-wide slices have no TMA path); row-major bf16 -> `tma` when `K <= 1024` and the slice is whole 64-column chunks (`BLOCK_N = 256` at any height, 128 at 128 / 256 rows), else `reg`; `quad_store` (rule-selected, `_q`) is the bf16 `reg` variant for whole-64-column slices that transposes 32-byte row segments across lane quads so each lane issues one 256-bit store per 4-row group |
| staging slots | `epi_slots`: 1 with the TMA-store epilogue (2 only when `K <= K_TWO_SLOTS = 0` and the slice's chunk count is even), 0 otherwise |
| stages | `default_stages(slots, cta_rows, block_n, b_mn)`: 7 / 6 / 5 for 0 / 1 / 2 slots at `BLOCK_N = 160 / 192 / 224 / 256`, 9 / 8 / 6 at `BLOCK_N = 128`, 4 / 4 / 3 for tall (256-row) tiles; 64-row tiles take the deepest pipeline that fits the 227 KiB opt-in beside the staging (32 KiB per slot), at most 12 (9 / 8 / 6 at `BLOCK_N = 256`, 12 / 12 / 10 at 128; `b_mn` sizes the MN-major B panels); the single-CTA form (`cta1`, round 19: its B stage streams the whole `BLOCK_N` columns per CTA, `b_stage_bytes(..., cta_group=1)`, so 48 KiB stages at `BLOCK_N = 256` and 32 KiB at 128) takes the deepest fit the same way (4 / 4 / 3 and 7 / 6 / 5).  A rule or caller `stages` is bounded by `smem_limit_for(arch)`: the 232448 B opt-in, or 334336 B on `sm_107a` (CUDA 13.4 oversized shared-memory mode, `SMEM_OVERSIZED`); the limit is not part of the instance key, so an instance has one symbol on every architecture |
| raster group | `default_group_m(a_mn, b_mn, m_tiles, pair_tiles, pairs)`: 16 CTA row tiles per cluster-launch-control raster group (a launch parameter since round 11, not an instance-key field) (even, so the two CTAs of a pair are the row halves of one 256-row tile) on every row; a 4-row group for one-to-two-wave weight-gradient rows measured no gain in the production configuration and is not applied.  A non-default group would carry the symbol suffix `_g<n>` |
| L2 hints | `default_hints(a_mn, b_mn, m_tiles, n_tiles, K, group_m, pairs, l2_bytes)`: no hint while one wave's operand panels fit the L2 (`wave_working_set(m_tiles, n_tiles, K, group_m, pairs) <= l2_bytes`); otherwise `A` streams `evict_first` when there is one column tile (`n_tiles == 1`); every other row gets no hint (reuse-ratio hints on the weight-gradient class measured 0..-4 percent in the production stream-K configuration).  Symbol suffix `_h<a><b>` (first letters): `_hen` = A evict_first, B none |
| L2 promotion, prefetch | `default_promo` = `none`, `default_pf` = 0 (measured, off); the symbol carries `_<promo>` / `_pf<n>` when set |
| CTA form | `form`: `""` (the ten-warp pipeline of every other program) or one of `W8_FORMS = ("w8r", "w8rs")`, the eight-warp CTA form of the same GEMM measured for three `sm_103a` input-gradient rows: four epilogue warps, a loader, a scheduler (`w8r`) or the loader's elected lane issuing the cluster-launch-control queries in its place (`w8rs`) and the MMA warp on CTA-pair 256 x 256 tiles over six 32 KiB stages, whole tiles only, the BF16 output through the TMA store, the raster quotients resolved by four fast divisors the host passes (`fd_tiles_l` / `fd_tiles_g` / `fd_group` / `fd_last`: launch arguments of these programs alone).  Symbols `_tma1_w8r` / `_tma1_w8rs` (the form token is the last knob suffix; `instance_key` field 37).  A rule's form yields where the plan is not whole-tile or a caller forces a pipeline, tile, output, epilogue, tail or raster knob (the row's ten-warp program runs the plan asked for; the rule's `pd` / `pro` / `lw` / `mg` / `sw` are its program there); where the form stands, those knobs are cleared from the key.  A caller-forced form is refused there (`instance_key`; the planner for stream-K units) |
| template | `instance_symbol(instance_key(...))`, e.g. `dense_proj_gemm_kk_n256`, `dense_proj_gemm_kk_n128_hen`, `dense_proj_gemm_kn_n256_f32_tma1`, `dense_proj_gemm_nn_n256_f32_tma1`, `dense_proj_gemm_nn_n128_hen_t`, `dense_proj_gemm_kk_n256_m64`, `dense_proj_gemm_kk_n256_m256_ov_ht`, `dense_proj_gemm_nn_n160_m256_bz64`, `dense_proj_gemm_nn_n128_hen_t_skx`, `dense_proj_gemm_kn_n256_tma1_bg8`, `dense_proj_gemm_kn_n256_f32_v8_ef_s9`, `dense_proj_gemm_nn_n256_m256_sks_sb2`, `dense_proj_gemm_kk_n128_hen_c1`, `dense_proj_gemm_kn_n256_tma1_w8r`, `dense_proj_gemm_kn_n256_tma1_w8rs` (round 13: `_pk` parked / `_ov` overlapped single-TMEM-buffer / `_ht` half-height-tail-wave tall epilogues; `htail` replaces the stream-K tail plan by 2 x tail whole-K half items when they fit the CTA pairs, so `ws_f32_elems = 0` and `sk_iters = sk_units * k_blocks`; `_bz64` = 32-column MN-major B panels (`b_swz` 64, smaller stages, deeper default pipeline), `_skx` = exact p-way K split of the tail tiles (`sk_exact`, pre-empts `sk_parts` / `htail`, p travels in `sk_iters`); `_sks` = synchronised stream-K tail (round 18: `sk_sync`, the tail tiles' main units over K steps [0, s) plus collector units over the leftovers, one fp32 slab per tail tile and a deterministic two-addend fixup; `sk_iters` = the main-unit count, `iters_per_unit` = s = ceil((q K + `sk_sync_m`) / (q + 1)); yields to `sk_exact`, pre-empts `sk_parts` / `htail`); `_bg8` = eight adjacent batch entries (heads) interleaved across the CTA pairs (`batch_group`, batched MLA rows only, 0 for a batch of one); `_ef` = fp32 v8 register stores with `L1::no_allocate.L2::evict_first` (`store_ef`), `_s9` = nine pipeline stages (oversized dynamic shared memory on sm_107a); round 19: `_sb<n>` = the slab path of the synchronised fixup (`sk_slab`: 1 early arrival, 2 + the bulk slab read through the dead mainloop stages, 3 + the fp32 output as the slab - the same `_sks` plan, forced to 0 without `sk_sync`; a rule's 3 yields to 2 when a caller forces the row-major fp32 register epilogue, a rule's 2 to 0 when the eight warp slabs do not fit a caller-forced pipeline), `_c1` = the single-CTA form (`cta1`: one `CTA_ROWS` x `BLOCK_N` tile per CTA with cluster dims (1, 1, 1), so `m_tiles` is not rounded to pairs, `pair_tiles` counts CTA tiles, `sm_pairs` is the SM count (`sm_units`), the grid is `num_full` CTAs and the form carries no tail policy; 128-row family only); `_skf` directly after `_skx` = the tail-first order of the exact split (`sk_first`: pair work items `[0, sk_units)` are the K parts of the tail tiles, then the whole tiles; the same plan, counters and slabs as `_skx`), `_lw` between `_or` and `_pro` = the load warp leaving the pair prelude before the post-allocation CTA rendezvous (`lw`; pair programs whose MMA warp allocates TMEM, `pro` bit 4), `_mg` directly after `_lw` = the whole-tile tile resolve (`mg`: the load warp and the MMA warp walk whole tiles, one segment per item, with the raster quotients resolved through corrected fp32 reciprocals; whole tiles only - no batch raster, column-grouped raster, half-height tail or stream-K form, and the plan carries no stream-K unit); `_sw<n>` directly after `_mg` = the paced output stores (`sw`: each epilogue warp of the bf16 TMA-store pd-2 drain delays its bulk tensor store issue, after the staging write and its fences, by a per-warp number of steps of `n` dependent u32 multiply-adds seeded by a clock sample, so each per-item eight-store burst becomes a seven-step ramp; the stores and bytes are unchanged; needs `slots` > 0, `pd` 2 and a bf16 output; 0 = off); `_w8r` / `_w8rs` directly after `_pro<n>` = the eight-warp CTA form (`form`, the CTA form row above)) |
| rules | `ROW_RULES[(arch, A MN-major, B MN-major, fp32 out, transposed out, batched, N, K, M)]` (the ragged token count never keys a rule: `M = None` for forward / input gradients, `K = None` for weight gradients) lays per-row knobs over the defaults above, merged over `ARCH_RULE_DEFAULTS` (`pro` 7 on `sm_100a`, `sm_103a` and `sm_107a`).  Rounds 2-33 mirror the Cake table byte for byte; the shapes tranche (`sm_100a` and `sm_103a`: forward / input gradient / weight gradient rows at K 4096-17408, N 32-17408, measured at T 16172-16384 on GB200 / GB300 against cuBLAS) and its kernel-level round 2 (the half-tail column-grouped 4096 squares, 12-column raster groups on the N 5120 rows, the load-warp thin input gradients with the `l2_256b` promotion of the N 512 rows on `sm_103a`, the tail-first 16-way exact split of the 4096 x 12288 weight gradients) extend it per architecture.  A rule's `sk_first` / `lw` yield where the planned program cannot carry them (an exact split the arithmetic does not admit, the single-CTA form, a `pro` without bit 4, a caller-forced tile family).  Three `sm_103a` input-gradient rules (4096 -> 512, 5120 -> 512, the kv_a 6144 -> 576) carry `mg` with `sk: False` (the whole-tile tile resolve measured faster than the row's program above the pair spread at both T on GB300); a rule's `mg` yields where the plan carries stream-K units or exceeds the kernel's cluster-tile bound (a caller `sk` / `sk_max_units` / `sk_parts` / exact or synchronised split), where a caller forces a raster or tail form the `_mg` program excludes (`batch_group` / `cgrp` / `htail` / `sk_first` / `stt`), the single-CTA form or another tile family, and a caller-forced `mg` is refused on such a plan (`sk_units` is 0 on every `_mg` program).  Two `sm_103a` input-gradient rules (5120 -> 512, 5120 -> 1024) carry `sw` 25 (the paced output stores measured faster than the row's program above the pair spread at both T on GB300); a rule's `sw` yields where the planned program is not the bf16 TMA-store pd-2 drain (a caller-forced register epilogue or `slots` 0, another `pd`, the fp32 output) or the single-CTA form, and a caller-forced `sw` is refused there.  Three `sm_103a` input-gradient rules (4096 -> 512 and the kv_a 6144 -> 576 with `w8rs`, 5120 -> 512 with `w8r`) carry `form`: the eight-warp CTA form measured faster than the row's program above the pair spread at both T on GB300, with raster groups of 12 row tiles on 5120 -> 512 and kv_a; the knobs after it stay the row's program where the form yields.  A rule's `htail` names the program (`_ht`); the half-tail fit (`0 < 2 * tail <= pairs`) selects only the plan (`plan.htail_plan`): where the half items do not fit the device's pairs (the K 5120 thin input gradients: 1280 tiles on 76 pairs, tail 64; the 4096 x 12288 projections: tail 56) the `_ht` program runs the row's plain whole-tile plan (never linearised stream-K units, whatever `sk` the row or the caller asked for), as measured.  `sm_103a` registers every `sm_100a` program and its lanes measured the shapes tranche alone: a row without an `sm_103a` rule plans with the `sm_100a` rule (`RULE_PARENT`; the same knobs, so the same program, kept where the `sm_103a` registry has it), a rule `sm_103a` measured itself always wins, and the weight-gradient orientation follows the same lookup; `sm_100a` and `sm_107a` inherit nothing |
| exact-wave gate | a rule's `BLOCK_N` 224 (`EXACT_WAVE_BLOCK_N`) was measured for the whole-wave fit of its tiles on the 76-pair devices (`exact_waves`: 152 tall tiles of the 4096 x 4096 weight gradient = two waves, 76 tiles of 4096 -> 1024 = one); on a device whose pairs do not fit them in whole waves and where the default tile needs fewer waves (74 pairs of a 148-SM part) the rule's `BLOCK_N` yields to the default tile (`plan.wave_fallback`; the rule's other knobs stay); a caller-forced `block_n` is never touched, and the plan is the launcher's where the fit is exact or both tiles need the same waves (152 and 160 SMs) |
| registry fallback | a plan whose template this architecture's registry lacks is redone without the batched small-M swap (`plan.swap_fallback`), then without the row's measured rule (`plan.rule_fallback`: the architecture default `pro` form), then lands on the nearest generated program of the row's layout / output class (`plan.knob_fallback`): the hand-ordered knob variants first (plain TMA L2 policy, the single-CTA form, the 64-row pair family, the tall tile, the register epilogue, BLOCK_N 128, each in the three `pro` forms), then every generated program of the class, nearest symbol first, with the knobs its symbol decodes to; a caller-forced knob always wins.  Only a class without any generated program on the architecture raises (`NotImplementedError` at binding) |
| grid | `(num_full + sk_units) * 2` CTAs: one cluster of two CTAs per `2 * CTA_ROWS` x `BLOCK_N` output tile plus the stream-K units; under `cta1` (round 19) `num_full` CTAs, one per `CTA_ROWS` x `BLOCK_N` tile, launched with the module's recorded cluster dims (1, 1, 1) |
| stream-K | `stream_k_plan(pair_tiles, k_blocks, sm_count // 2, sk="auto")`: only when the whole problem is one partial wave of CTA pairs whose halves fit the pairs (`pair_tiles <= pairs`, `2 * tail <= pairs`) and `k_blocks >= 16`, the tiles' linearised K steps are cut into units of `iters_per_unit = ceil(k_blocks / 2)` (two K-aligned halves per tile on an even `k_blocks`), `sk_units = ceil(pair_tiles * k_blocks / iters_per_unit)` so that every unit holds a K step, with an in-kernel deterministic fixup; otherwise every work item is a whole tile (multi-wave tails stay data-parallel).  `sk=True` is the linearised-units policy (up to `pairs` units of >= 8 K steps), `sk="tiles"` the whole-tile diagnostic, `sk=False` off.  A measured `sk_parts = p` rule, or the caller knob `sk_parts`, instead splits every tail tile `p` ways when `p * tail` units fit the pairs and each keeps >= 8 K steps (`sk_parts_plan`).  The host asserts `tail_tiles * 16 <= 4096` slice counters, and `check_stream_k_plan` refuses a plan outside the kernel's launch contract: a unit without a K step on the `pd` 2 / `sk_slab` 2 / `stt` programs (they hand off the work in an item's last segment, so such a unit never finishes), or synchronised collectors outside `[1, tail_tiles]` (`[1, ...)` on the other `sk_sync` programs) |

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
  swapped transposed-store weight gradient `nn_t` (6 splits; `out.T = B.T
  A.T`).  An MN-major `A` with a K-major `B` has no instance.
* Every `launch()` re-splits the second operand into the retained
  `[3, outer, inner]` BF16 stack in the operand's own layout (exact three-way
  split, no allocation: one launch of the fused split program
  `split_fp32_bf16x3` when this checkout registers it for the device, else a
  five-op torch chain through an fp32 scratch -- the same bits either way),
  launches, and sums the split-K partials on the host in a fixed order.  Splits, chunk boundaries and the reduction order are
  fixed: bit-exact run to run.  Error against an FP64 reference: FP32-class
  (`rel_fro` about 2e-7 to 4e-7 on the GLM-5.2 router shapes).

## Layout of this package

* `cake_jit.py` -- the registry (schema v2) and the loader glue.  Three
  literals filled by the generated-program export, each wrapped in
  `# fmt: off` / `# fmt: on` and rendered one record per line (the export
  replaces a literal from its head line up to the next `# fmt: on`):
  `ARG_PLANS` (the distinct argument plans -- the ordered `[kind, name]` pairs
  a generated entry takes -- keyed by the first 12 hex digits of the SHA-256
  of the plan's compact JSON, `arg_plan_key`; one plan serves every K1 program
  and one every K2 program), `MODULES` (one record per generated program:
  template, kernel symbol, the architecture-free `library` stem, FFI entry,
  `arg_plan` key -- `arg_plan_of` resolves it and still accepts an inline
  list -- launch geometry, `arches`) and `KERNELS` (`arch -> template ->
  module name`).  `ARCHES` names the registered architectures;
  `load_cake_dense_projection_gemm_module(name, arch)` loads the build
  `<library>_<arch>` through `cake_build`.
* `cake_build.py` + `build.json` -- the loader shared by every Cake-exported
  package (byte-identical copies) and this package's build manifest (schema
  `cake-build/1`, written by the export): one entry per `<library>_<arch>`
  with its translation units, digests, nvcc flags, closure digest and
  `aot/<name>.so` path.  `cake_build.load_library(name)` loads the prebuilt
  library when `aot/` holds it and otherwise JIT-builds the entry through
  FlashInfer (`gen_jit_spec`, sources digest-verified first);
  `cake_jit.specs_for_arch(arch)` lists the JIT specs of an architecture for
  `flashinfer.jit.build_jit_specs`.  `cake_build.jit_spec(name)` /
  `jit_specs(arch)` gate an entry on its `min_cuda` only (`sm_107a` entries
  need CUDA 13.0): a 13.0 toolkit whose `nvcc` does not emit `compute_107a`
  fails inside the JIT build with nvcc's own message.
* `cake_backend.py` -- view validation, instance selection, stream-K
  planning, argument-plan binding, the prepared launches and the eager entry
  points.
* `csrc/cake_dense_projection_gemm/` -- generated kernel and binding
  translation units (`.clang-format` disables formatting: the sources are
  identity-checked by the manifest's digests); `csrc/cake_ffi_checks.h` -- the
  tvm-ffi tensor checks the bindings include (shipped with the package, no
  FlashInfer header).

## Status

The registries hold the generated programs of the current export lock: the
Cake exporter (`exports/dense_projection_gemm` of the Cake repository) emits
one program per template, shared by `sm_100a` and `sm_107a` wherever the generated text is the same (the per-architecture
template lists are `KERNELS` in `cake_jit.py`; `K1_TEMPLATES` of the exporter
is the source of truth and is regenerated whenever `ROW_RULES` resolve a new
instance).  `build.json` ships the builds of the architectures `cake_jit.KERNELS`
lists (`sm_100a`, `sm_103a` and `sm_107a`); the eight-warp form ships for
`sm_103a` only (two programs); every program the measured
`sm_100a` / `sm_103a` rules name is registered, so the registry fallback only
serves a caller that forces a template no export generated.  An entry point raises
`NotImplementedError` naming the missing instance for a template that no
export has generated and for every template on a device whose architecture the
registry does not list.

CUDA Graphs: prepare outside capture and replay the prepared launch
(`prepare_*` once, `launch` many times).  The convenience entry points
(`dense_projection_gemm`, `projection_*`, `router_*`) reuse a cached planned
launch per (device, stream, geometry) key and the shared workspace arena of its
(device, stream); capture a prepared launch object rather than the convenience
entry points (see "Host paths").

Tests: `tests/experimental/test_cake_dense_projection_gemm.py` (host-side
planning tests run everywhere; the GPU tests skip without a registered
generated instance -- a device of an architecture `cake_jit.KERNELS` lists).  Benchmark:
`benchmarks/bench_cake_dense_projection_gemm.py` (the 11 GLM-5.2 projection
rows x fwd / dgrad / wgrad x T in {16231, 16172}, the router rows and the two
batched MLA rows, paired against `torch.matmul`; `--host-overhead` adds the
host time per call of the prepared launch and of the convenience entry point).
