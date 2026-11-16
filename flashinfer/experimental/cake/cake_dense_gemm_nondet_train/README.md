# Cake dense GEMM backend, non-deterministic (SM100 / SM103 / SM107)

Backend of the public module `flashinfer.cake_dense_gemm_nondet_train`: the GLM-5.2 training GEMMs -- the BF16 dense projection GEMM (K1:
BF16 operands, FP32 accumulation, BF16 or FP32 output, strided views, ragged token counts, batched head projections)
and the FP32 router-gate GEMM through split-BF16x3 tensor-core emulation (K2) -- with programs whose split-K
reductions add the partials in the order they finish.  Family issue: flashinfer-ai/flashinfer#5677.

The deterministic implementation is the module `flashinfer.cake_dense_gemm_det_train` (backend
`flashinfer.experimental.cake.cake_dense_gemm_det_train`; flashinfer-ai/flashinfer#5677, #7043; see its README for the view
contract, the host paths and the per-architecture programs).  This package keeps its operations, dtypes, accuracy
budgets and host paths.  It relaxes one guarantee only: an output may come from a program whose bits change from run
to run where that program is at least 3 % faster ([Acceptance rules](#acceptance-rules)).  With the shipped tables and
default arguments no output does: every output repeats bit for bit (the per-output table of
[Numerics](#numerics-architectures-and-parallelism)).  Only a caller's split count of three or more gives bits that
vary, and that launch raises PyTorch's deterministic-algorithms alert.

## Names

The module selects the mode: there is no mode argument.  Import the names from `flashinfer.cake_dense_gemm_nondet_train`; it loads this
backend on the first attribute access and returns the backend's own objects.

| operation | `flashinfer.cake_dense_gemm_nondet_train` | `flashinfer.cake_dense_gemm_det_train` |
|---|---|---|
| K1 GEMM (convenience path), plan, prepared launch | `cake_dense_gemm_nondet_train_projection`, `cake_dense_gemm_nondet_train_projection_plan`, `cake_dense_gemm_nondet_train_projection_prepare` | `cake_dense_gemm_det_train_projection`, `cake_dense_gemm_det_train_projection_plan`, `cake_dense_gemm_det_train_projection_prepare` |
| K1 forward, input gradient, weight gradient | `cake_dense_gemm_nondet_train_projection_forward`, `cake_dense_gemm_nondet_train_projection_dgrad`, `cake_dense_gemm_nondet_train_projection_wgrad`, `cake_dense_gemm_nondet_train_projection_wgrad_prepare` | `cake_dense_gemm_det_train_projection_forward`, `cake_dense_gemm_det_train_projection_dgrad`, `cake_dense_gemm_det_train_projection_wgrad`, `cake_dense_gemm_det_train_projection_wgrad_prepare` |
| K2 router GEMM (convenience path), plan, prepared launch | `cake_dense_gemm_nondet_train_router`, `cake_dense_gemm_nondet_train_router_plan`, `cake_dense_gemm_nondet_train_router_prepare` | `cake_dense_gemm_det_train_router`, `cake_dense_gemm_det_train_router_plan`, `cake_dense_gemm_det_train_router_prepare` |
| K2 forward, input gradient, weight gradient | `cake_dense_gemm_nondet_train_router_forward`, `cake_dense_gemm_nondet_train_router_dgrad`, `cake_dense_gemm_nondet_train_router_wgrad` | `cake_dense_gemm_det_train_router_forward`, `cake_dense_gemm_det_train_router_dgrad`, `cake_dense_gemm_det_train_router_wgrad` |
| plan and prepared-launch types | `CakeDenseGemmNondetProjectionPlan`, `CakeDenseGemmNondetProjectionLaunch`, `CakeDenseGemmNondetRouterPlan`, `CakeDenseGemmNondetRouterLaunch` | `CakeDenseGemmDetProjectionPlan`, `CakeDenseGemmDetProjectionLaunch`, `CakeDenseGemmDetRouterPlan`, `CakeDenseGemmDetRouterLaunch` |
| convenience caches and workspace arenas | `cake_dense_gemm_nondet_train_convenience_cache_clear` (both packages') | `cake_dense_gemm_det_train_convenience_cache_clear` |
| support query | `is_cake_dense_gemm_nondet_train_supported(device)` | `is_cake_dense_gemm_det_train_supported(device)` |
| constants | `ARCHES` | `ARCHES`, `SUPPORTED_COMPUTE_CAPABILITIES` |

*Table: the public names of the two modules, per operation.*

- `cake_dense_gemm_nondet_train_projection_plan` and `cake_dense_gemm_nondet_train_projection_prepare` take keyword-only knobs: `splits`, `chains` and `block_n` of
  this package's planner, and the public knobs of the deterministic planner of the same entry.  A knob the caller
  gives reaches the deterministic planner as given; a knob the caller does not give takes that planner's default.
- The convenience caches of both packages are bounded by one setting, the deterministic backend's
  `cake_backend.CONVENIENCE_CACHE_SIZE` (read on every insertion); neither public module has that name.
- On a row of a K1 program table, `cake_dense_gemm_nondet_train_projection_prepare` and
  `cake_dense_gemm_nondet_train_projection_wgrad_prepare` return a `cake_backend.CakeDenseGemmNondetProgramLaunch`
  (not a name of the public module; see [Optimized and prebuilt paths](#optimized-and-prebuilt-paths)).

## Mode and dispatch rules

A call runs the routes of `cake_backend.ROUTES` (table below).  No entry point consults the PyTorch
deterministic-algorithms setting to choose a program.  Under `torch.use_deterministic_algorithms(True)` a launch whose
result can vary run to run raises PyTorch's alert right before it launches:

| launch | rows | order of the partials | under `torch.use_deterministic_algorithms(True)` |
|---|---|---|---|
| K1 CTA-pair split-K program, three or more splits | a caller's `splits >= 3` in the arrival order (no row of the shipped tables) | the order the splits finish | `RuntimeError` before the launch (the program is not launched and `out` is untouched); with `warn_only=True` a `UserWarning`, then the launch |
| K1 CTA-pair split-K program, two splits | every `sm_100a` / `sm_103a` row, `sm_107a` `8192 -> 128` and the FP32 rows `4096 -> 32`, `5120 -> 32`, `5120 -> 48` | one add of two values (repeats bit for bit) | no alert |
| K1 one split, the chain programs (`_c2`, `_c3`, `_c4`, `_c8`) | the one-split rows; `sm_107a` FP32 `4096 / 5120 -> 512 / 1024` from 961 tokens (208, 212, 216 and more SMs) two chains at an even number of K blocks and four at an odd number (eight at `5120 -> 1024`; `K1_PARITY_RANGES_BY_SM`); 212-SM `sm_107a` FP32 from 16172 to 16384 tokens (`K1_ORDERED_RANGES_BY_SM`): `4096 -> 1024` and `5120 -> 1024` four chains at an odd number of K blocks and two at an even number, `5120 -> 512` two chains at an even number; 148-SM `sm_100a` BF16 `4096 -> 1024` two chains from 961 to 16384 tokens (`K1_ORDERED_RANGES_BY_SM`) | whole tiles, chains added in order | no alert |
| K1 CTA-pair split-K program in the mirror order (`_mirror`) | `K1_ORDERED_RANGES_BY_SM` (212-SM `sm_107a` FP32 `4096 -> 32` at 16172 to 16384 tokens, four splits) | each split of the first half with its mirror, then the pairs in split order (repeats bit for bit) | no alert |
| K1 one-wave forward `dense_proj_gemm_wave_*`, input-gradient programs `dense_proj_gemm_dgrad_*` and `dense_proj_gemm_nvjet_early_*` | `K1_FORWARD_PROGRAM_RULES_BY_SM`, `K1_DGRAD_PROGRAM_RULES_BY_SM`, `K1_DGRAD_NVJET_RULES_BY_SM` | whole K per output element, or the deterministic plan's order (the same bits) | no alert |
| K1 stream-K weight gradient `dense_proj_gemm_sk_*` | `K1_WGRAD_SK_RULES_BY_SM` (148-SM `sm_100a` `5120 -> 1024`; 152-SM `sm_103a` `5120 -> 1024` BF16 at 16172 tokens and FP32 at 16100, 16172, 16231, 16300 and 16384 tokens; 212-SM `sm_107a` `4096 -> 1024` BF16 at 16172 and 16384 tokens) | a split tile's partials added by one fixed reducer in unit order (repeats bit for bit) | no alert |
| K1 cluster program `dense_proj_gemm_dsm_*_s2` / `_s3` / `_s4` | `K1_DSM_RANGES_BY_SM` (148-SM and 152-SM `sm_100a`, 152-SM `sm_103a` and 212-SM `sm_107a` parts, FP32 out) and the four-CTA 64-wide tiles of `K1_DSM_ODD_BLOCK_RANGES_BY_SM` (at an odd number of 64-token K blocks) | cluster-rank order in distributed shared memory (no global atomics, no workspace) | no alert |
| K2 stream-K program, more than two partials per element | a caller's `splits >= 3` on `nn_t` (no window of the shipped tables: a `"waves"` partition of more than two partials per element runs the deterministic program instead) | the order the partials finish (TMA reduce-add) | `RuntimeError` before the split, the fill of `out` and the program; with `warn_only=True` a `UserWarning`, then the launch |
| K2 stream-K program, at most two partials per element | a forward's `splits` of 1 or 2; the `"waves"` windows on a part without a router table of its SM count, where every element takes at most two partials | `0 + a + b == 0 + b + a` | no alert |
| K2 SIMT programs (forward, input and weight gradient), `router_partials_sum` | their token windows | fixed: every element stored once | no alert |
| K2 K-order program `router_fp32_gemm_ts_*_ksum_*` | the `"ksum_S<n>"` windows (212-SM `sm_107a` hidden 6144 at 16172 to 16231 tokens) | a tile's span partials added in K order by the span that draws the tile's last ticket (repeats bit for bit) | no alert |
| the deterministic programs, the BF16x3 split | the rows routed to the deterministic program (below) | fixed | no alert |

*Table: the alert rule per launch.*

- The alert is checked on every launch, prepared and convenience path alike, never at plan or prepare time.  The
  forward and input-gradient entry points never alert.
- CUDA graphs: the alert is decided when the launch is captured; a replay does not check the setting again.
- The setting never changes which program a call runs, and no call goes to the deterministic package because of it.
  For a bitwise reproducible result, call `flashinfer.cake_dense_gemm_det_train`.
- The alert text starts with `flashinfer.cake_dense_gemm_nondet_train: <launch> does not have a deterministic implementation` and names
  `flashinfer.cake_dense_gemm_det_train` as the deterministic module.

On some rows this module runs the deterministic package's fixed-order programs:

| case | where | result |
|---|---|---|
| a layout routed `"det"` (`ROUTES`), or a row the planner keeps on the deterministic program (no verdict), outside the rows of the K1 program tables | `cake_dense_gemm_nondet_train_projection_plan`, `cake_dense_gemm_nondet_train_projection_prepare`, `cake_dense_gemm_nondet_train_projection_wgrad_prepare`, `cake_dense_gemm_nondet_train_router_plan`, `cake_dense_gemm_nondet_train_router_prepare` | the deterministic package's plan or prepared launch (`CakeDenseGemmDetProjectionPlan`, `CakeDenseGemmDetProjectionLaunch`, `CakeDenseGemmDetRouterPlan`, `CakeDenseGemmDetRouterLaunch`) |
| the same rows on the convenience path | the convenience entry points | the deterministic plan and program through this package's binding (the same bits, fewer host operations per call) |
| a row routed to the deterministic program whose launch views this package's binding cannot rebuild from the call's tensors (two operands that alias), and a batched forward that the deterministic package runs on its swapped small-token program | the K1 convenience entry points | the deterministic package's entry point of the same operation (the same program, the same bits) |
| tensors on different CUDA devices, or a cached entry without a binding | the K1 convenience path of a row routed to the deterministic program | the same |

*Table: when this module returns a deterministic object or calls a deterministic entry point.*

- A row that runs a program of this package never goes to the deterministic package: when its views cannot be
  bound, it takes this package's unbound launch.
- An output varies run to run only where the speed rule of [Acceptance rules](#acceptance-rules) allows it (a
  program at least 3 % faster), and only through a launch that raises the alert above.  With the shipped tables and
  default arguments every output repeats bit for bit (the "run-to-run bits" rows and the per-output table below); a
  caller's split count of three or more is the only launch whose bits vary.
- The tensors passed between the forward and the gradients carry no mode: recompute a checkpointed forward with the
  module that made the saved one (same package only).  Neither package has an autograd Function or a saved tape.
- This package imports the deterministic package at run time, as its sibling (see
  [Standalone build](#standalone-build-vendoring)).  It re-exports no name of the deterministic package.

| operation | route |
|---|---|
| K1 forward (`cake_dense_gemm_nondet_train_projection_forward`, `kk` layouts) | deterministic program (whole output tiles, or the deterministic plan's two-way split of K with the partials added in a fixed order; a recomputed forward reproduces its result).  The rows of `K1_FORWARD_RULES_BY_SM` (exact token counts on 148-SM `sm_100a` and 152-SM `sm_100a` / `sm_103a` parts, the K1 forward table below): the deterministic package's single-CTA program, every tile's whole K in one CTA -- other bits than the deterministic package's plan, the same on every run -- or, at 16172 to 16384 tokens of `4096 -> 512` / `5120 -> 512`, its program `kk_n256_ht_cg_pro15` (the deterministic plan's bits).  Before that table, the rows of `K1_FORWARD_PROGRAM_RULES_BY_SM` (`4096 -> 512` / `5120 -> 512` at 16172, 16231 and 16384 tokens on 148- and 152-SM `sm_100a` and 152-SM `sm_103a` parts; on 152-SM `sm_100a` and `sm_103a` parts also `4096 -> 1024` / `5120 -> 1024` at 16172, 16231 and 16384 tokens): this package's one-wave forward program `dense_proj_gemm_wave_lean_kk_n224_ef_e8_pf2_st` -- one CTA pair per item of 512 output features by 224 tokens over the whole K, one wave of the SM pairs at 512 features and at most two items per pair at 1024 (the deterministic plan's bits), with the start path of cuBLAS's nvjet kernels, the epilogue freeing the tensor memory, eight epilogue warps, an L2 prefetch of the X box two stages ahead and each item's last K stage run by token half (at 512 features 0.948 to 0.993 times the time of the same program without the prefetch and the token halves on B200, GB200 and GB300 parts; at 1024 features 0.946 to 0.990 times the time of the deterministic plan on GB200 and GB300 parts) |
| K1 input gradient (`cake_dense_gemm_nondet_train_projection_dgrad`, `kn` layouts) | deterministic program (whole output tiles, or the deterministic plan's two-way split of K with the partials added in a fixed order).  The rows of `K1_DGRAD_RULES_BY_SM` (exact rows and token counts of `d_out <= 1024` on 148-SM `sm_100a`, 152-SM `sm_100a` / `sm_103a` and 212-SM `sm_107a` parts): another program of the deterministic package (`kn_n128_f32_tma2_s4_c1`, `kn_n256_tma1_w8` / `_w8r` / `_w8rs`, `kn_n256_f32_v8_ef_pro7`; `sm_107a`: `kn_n128_f32_tma1_pd2_c1`, `kn_n256_tma2_pd2_lw_pro15`), measured faster with the deterministic plan's bits.  After that table, the BF16 rows of `K1_DGRAD_PROMO_RANGES_BY_SM` (`4096 / 5120 -> 512 / 1024` and `6144 -> 576` from 16172 to 16384 tokens on 148-SM `sm_100a` parts): the deterministic plan of the row with the TMA L2 promotion of 256 bytes (a launch parameter: the same program, grid and bits; on a B200 part 0.976 to 0.998 times the time of the deterministic plan, which promotes none).  Before that table, the FP32 rows of `K1_DGRAD_PROGRAM_RULES_BY_SM` (`4096 / 5120 -> 512 / 1024` at 16172, 16231 and 16384 tokens on 148- and 152-SM `sm_100a` and 152-SM `sm_103a` parts; on 148-SM `sm_100a` and 152-SM `sm_103a` parts `4096 / 5120 -> 512` and `6144 -> 576` at every token count from 16172 to 16384): this package's input-gradient program `dense_proj_gemm_dgrad_kn_n256_f32_ef[_tma]` -- a persistent CTA pair per SM pair over 256 x 256 tiles, the whole reduction per tile, row or TMA stores (the deterministic plan's bits); on 148-SM `sm_100a` parts from 16172 to 16384 tokens, `4096 -> 512`, `5120 -> 512` and `6144 -> 576` run its TMA form without the evict-first hint, `dense_proj_gemm_dgrad_kn_n256_f32_tma_lean_free` (on a B200 part the 512-feature rows 0.95 to 0.97 times the time of the evict-first TMA form at 16172 and 16384 tokens and 0.93 to 0.95 times at 16231 tokens, `6144 -> 576` 0.944 to 0.949 times the time of the deterministic plan at 16172, 16231, 16300 and 16384 tokens over 24 placements of its weight each; the same bits); on 152-SM `sm_103a` parts from 16172 to 16384 tokens all three rows run `dense_proj_gemm_dgrad_kn_n256_f32_tma_lean_free` (at 16172, 16231 and 16384 tokens on a GB300 part the 512-feature rows 0.95 to 0.96 times the time of the evict-first TMA form and `6144 -> 576` 0.94 to 0.95 times that of the deterministic plan, the same bits).  Between the two, the BF16 rows of `K1_DGRAD_NVJET_RULES_BY_SM` (`4096 -> 32` and `4096 -> 64` at 512 to 16384 tokens except `4096 -> 32` at 2048 tokens, `4096 / 5120 / 6144 / 8192 -> 16` to `128` at 16231 tokens, on 152-SM `sm_103a` and 148- and 152-SM `sm_100a` parts; `sm_100a` also leaves `4096 -> 32` at 14592 tokens and `4096 -> 64` at 4864 tokens to the other tables): the deterministic package's plan of `kn_n256_tma1_w8r` launched on this package's program `dense_proj_gemm_nvjet_early_kn_n256_lean` -- the same kernel parameters, host plan, cluster and bits, with the first item's first K step loaded and the next boxes prefetched into the L2 in the prologue and the start path of cuBLAS's nvjet kernels (1.01 to 1.17 times faster than `kn_n256_tma1_w8r`) |
| K1 weight gradient (`cake_dense_gemm_nondet_train_projection_wgrad`, `nn` layouts) | the rows of `K1_SPLIT_RULES`: split-K over the ragged token count -- every split writes its FP32 partial tile to a workspace slot; the last split of a tile to arrive (a per-tile ticket counter) adds the other slots in arrival order and rounds once into `out`, then returns the counter to zero.  Before every K1 table, the FP32 rows and token ranges of `K1_DSM_RANGES_BY_SM` (148-SM and 152-SM `sm_100a`, 152-SM `sm_103a` and 212-SM `sm_107a` parts, the cluster tables below): the cluster program `dense_proj_gemm_dsm_*` -- a tile's K split over the `S` CTAs of a thread-block cluster, each CTA's FP32 partial sent to the CTAs that own its columns through distributed shared memory, the `S` partials added in cluster-rank order and stored once (no workspace, no ticket: the same bits on every run).  On 212-SM `sm_107a` parts FP32 `4096 -> 48` and `4096 -> 64` run four CTAs of 64-wide tiles from 16172 to 16384 tokens.  At an odd number of 64-token K blocks inside a range of `K1_DSM_ODD_BLOCK_RANGES_BY_SM` (148-SM and 152-SM `sm_100a` and 152-SM `sm_103a` FP32 `4096 -> 48` and `4096 -> 64` from 16172 to 16384 tokens) such a row runs that table's program in place of its two-CTA cluster program: four CTAs of 64-wide tiles (every two-way split missed the accuracy check there).  The ranges of `K1_ORDERED_RANGES_BY_SM` (212-SM `sm_107a`: FP32 `4096 -> 32` at 16172 to 16384 tokens four splits of 64-wide tiles in the mirror order, FP32 `4096 -> 1024` and `5120 -> 1024` at 16172 to 16384 tokens one split of four chains at an odd number of 64-token K blocks and of two at an even number, `5120 -> 512` two chains at an even number; 148-SM `sm_100a`: BF16 `4096 -> 1024` at 961 to 16384 tokens one split of two chains of 256-wide tiles) come before every table but the cluster ranges, and before them the FP32 `sm_107a` rows of `K1_PARITY_RANGES_BY_SM` (`4096 / 5120 -> 512 / 1024` from 961 tokens on 208, 212, 216 and more SMs): one split of two accumulation chains at an even number of 64-token K blocks and of four at an odd number (eight at `5120 -> 1024`; 128-wide tiles at 512 features, the 192-wide row-major program at `4096 -> 1024`, 256-wide tiles at `5120 -> 1024`), or the deterministic program in that table's deterministic ranges.  Ahead of the cluster ranges and every K1 table, the rows of `K1_WGRAD_SK_RULES_BY_SM` (148-SM `sm_100a` `5120 -> 1024` BF16 and FP32 at 16172, 16231 and 16384 tokens; 152-SM `sm_103a` `5120 -> 1024` BF16 at 16172 tokens and FP32 at 16231 and 16384 tokens): the stream-K program `dense_proj_gemm_sk_nn_n256[_f32]_t` -- the output tiles laid end to end on one line of 64-token K blocks, one unit of the line per CTA pair of the part, a split tile summed by the unit that holds its start in unit order (the units wait for one another: every CTA pair of the launch must be resident at once).  The rows of `K1_WGRAD_DET_RULES_BY_SM` (148-SM `sm_100a` `5120 -> 1024` at 16172 to 16384 tokens; below `K1_WGRAD_SK_RULES_BY_SM` on its rows): the deterministic package's two-range program `nn_n192_bz64_hne(_f32_v8_ef)_sks_sb2_pro15`.  Every other row: deterministic program |
| K2 forward (`cake_dense_gemm_nondet_train_router_forward`, `kk`) | the token windows of its `ROUTER_RULES` rows (152-SM `sm_100a` / `sm_103a` parts, 256 experts, hidden 4096, 5120 and 6144, 1 to 192 tokens; `sm_100a` hidden 4096: 1 to 160) and, on 212-SM `sm_107a` parts, of its `ROUTER_RULES_BY_SM` rows (256 experts, hidden 4096, 5120 and 6144, 1 to 192 tokens): the exact-FP32 SIMT program `router_fp32_fwd_simt_c8_*` -- `K` split over the eight CTAs of a thread-block cluster, FFMA chains of the FP32 operands read in place (up to 16 tokens in a compensated form), the eight partials added in cluster-rank order through distributed shared memory, every element of `out` stored once.  Every other row, token count and part: deterministic program (its four splits); an explicit `splits` of 1 or 2 runs this package's stream-K program (at most two partials per element) |
| K2 input gradient (`cake_dense_gemm_nondet_train_router_dgrad`, `kn`) | the token windows of its `ROUTER_RULES` rows (on 212-SM `sm_107a` and 148-SM `sm_100a` / `sm_103a` parts `ROUTER_RULES_BY_SM`): the exact-FP32 SIMT program `router_fp32_dgrad_simt_*` -- one FFMA per product of the FP32 operands read in place, every element of `out` stored once.  Every other row, token count and part: deterministic program (one split) |
| K2 weight gradient (`cake_dense_gemm_nondet_train_router_wgrad`, `nn_t`) | the SIMT token windows of its rows in the router tables (`ROUTER_RULES`; on 212-SM `sm_107a` and 148-SM `sm_100a` / `sm_103a` parts `ROUTER_RULES_BY_SM`): the exact-FP32 SIMT program `router_fp32_gemm_simt_*` on the CUDA cores, every element of `out` stored once; on 148-SM parts at one token at hidden 4096 its compensated form.  The `"det_s<n>"` windows (`router_det_splits`): the deterministic program at `n` splits -- hidden 4096 16 splits at 16384 tokens on 148-SM `sm_100a` parts, hidden 6144 six splits from 16129 to 16384 tokens on 148- and 152-SM `sm_100a` / `sm_103a` parts, hidden 5120 12 splits at 16172 to 16384 tokens on 212-SM `sm_107a` parts.  The `"ksum_S<n>"` windows (`router_ksum_spec_splits`): the K-order program of the part (`ROUTER_KSUM_TEMPLATES_BY_SM`, `router_fp32_gemm_ts_r6p6_atomic_nn_t_ksum_ht_c2_nreg`) at `n` spans per tile -- each span stores its FP32 partial into a workspace slot, and the span that draws the tile's last ticket adds the slots in K order and stores the tile (the deterministic program's `n` partials in K order, the same bits on every run): hidden 6144 six spans at 16172 to 16231 tokens on 212-SM `sm_107a` parts.  Every other row: deterministic program (its split count of the row, `router_splits`: six; nine at hidden 5120 from 16129 to 16384 tokens on 148- and 152-SM `sm_100a` / `sm_103a` parts, its `ROUTER_SPLIT_RULES`).  The stream-K program (spans alike, TMA reduce-add into `out`) runs only for a caller's `splits` and in the `"waves"` windows on a part without a router table of its SM count where every element takes at most two partials (else the deterministic program); no row of the shipped tables runs the head/tail program |
| BF16x3 split of the K2 second operand (inside the K2 launches) | the deterministic stage (exact) |

*Table: the routing of this module (`cake_backend.ROUTES`), per operation and row.*

## Numerics, architectures and parallelism

| property | contract |
|---|---|
| operands and accumulation | BF16 operands with FP32 accumulation (K1); FP32 operands split exactly into three BF16 terms with FP32 accumulation (K2), except in a K2 SIMT window (forward, input and weight gradient): the FP32 operands as they are on the CUDA cores, one FFMA per product (in a compensated form per output a sum and its compensation: in the Dot2 form of the 148-SM tables' one-token window at hidden 4096 each product's rounding error kept, in the forward's compensated windows Kahan adds of short FMA chains; the slices -- in the forward the lanes and the cluster's CTAs -- merged by TwoSum with their compensations), no BF16x3 split.  No silent TF32 or BF16 for the router |
| accuracy | the deterministic package's budgets against FP64 of the exact operands, worst over repeated runs: BF16 out `atol = rtol = 1e-2`; FP32 out relative Frobenius error `<= 5e-5` and max error `<= 1e-3 * max abs(ref)`; router `<= 2e-6` relative Frobenius error and no row whose top-8 experts differ from FP64 (outside near ties of the 8th and 9th logits).  At the largest token counts some thin K1 rows miss the BF16 (FP32) budget with the deterministic program and with this package's program alike; there the row keeps its route.  The deterministic program: BF16 output of `sm_100a` `4096 -> 32`, `4096 -> 48`, `4096 -> 64`, `5120 -> 32`, `5120 -> 48` and of `sm_103a` `4096 -> 32`, `4096 -> 48`, `5120 -> 32`, `5120 -> 48`, `5120 -> 64`, `5120 -> 128` (from 45058, 49153, 65536 or 65537 tokens up by row: the K1 token tables below), and FP32 output of the rows of the odd-block K1 table at an odd number of K blocks from 98305 tokens up.  This package's program: BF16 output of `sm_100a` `4096 -> 64` from about 43009 to 65535 tokens and `5120 -> 32` at 47105 to 65536 tokens, of `sm_103a` `4096 -> 32` from about 49153 to 65535 tokens, and on 148-SM parts of `sm_103a` `5120 -> 48` from 49152 tokens, `5120 -> 64` at 47104 and from 49153 tokens and `5120 -> 128` from 65537 tokens |
| rounding | each output element is rounded once, from its complete FP32 sum |
| subnormals | kept: K1 adds the partials with IEEE round-to-nearest adds, K2 reduces with the TMA reduce-add; a K2 SIMT window (forward, input and weight gradient) accumulates with round-to-nearest FMAs and adds, merges the sums of a CTA's slices (and in a compensated form their compensations) in a fixed order through shared memory (the forward: the sums of a cluster's eight CTAs, through distributed shared memory) and stores every element of `out` once, with no reduce-add; nothing flushes to zero |
| `torch.use_deterministic_algorithms(True)` | PyTorch's alert right before an order-varying launch: the K1 CTA-pair program at three or more splits in the arrival order (a caller's `splits >= 3`), K2 stream-K with more than two partials per element (a caller's `splits >= 3` on `nn_t`); no default plan of the shipped tables is such a launch.  `RuntimeError`, or a `UserWarning` with `warn_only=True` and then the launch.  The alert is checked on every launch (prepared and convenience), never at plan or prepare time.  A launch captured into a CUDA graph is checked at capture, not at replay.  The two-split K1 rows, the mirror order, the K1 cluster program (`K1_DSM_RANGES_BY_SM` and `K1_DSM_ODD_BLOCK_RANGES_BY_SM`, its splits added in cluster-rank order), the K2 forward, the K2 SIMT, K-order and `"waves"` windows and every deterministic program run without an alert |
| run-to-run bits, K1 | every row of the tables repeats bit for bit: two splits (one add of two values), the cluster program (`K1_DSM_RANGES_BY_SM` and `K1_DSM_ODD_BLOCK_RANGES_BY_SM`, two to four splits added in cluster-rank order), one split and the chain programs (whole tiles, chains added in order), the mirror order, the stream-K program (unit order), the deterministic programs; a caller's `splits >= 3` in the arrival order varies in the last bits |
| run-to-run bits, K2 | forward: in a SIMT window repeats bit for bit (every element stored once, the lanes and the cluster's CTAs combined in a fixed order), else the deterministic program's bits; weight gradient: repeats bit for bit on every row of the shipped tables on every part (a SIMT window, a K-order window, a `"waves"` window's partition of at most two partials per element, or the deterministic program at the split count of a `"det_s<n>"` window or of the deterministic package); a stream-K partition of more than two partials per element (a caller's `splits >= 3` on `nn_t`) varies in the last bits; a SIMT window of the input gradient: repeats bit for bit (every element stored once, its slices merged in a fixed order); with an explicit forward `splits` of 1 or 2: repeats bit for bit (`0 + a + b == 0 + b + a`), so recomputed router logits and top-k match |
| architectures | `sm_100a`, `sm_103a`, `sm_107a`: one `-gencode=arch=compute_<arch>,code=sm_<arch>` build per program, minimum CUDA 12.8 / 12.9 / 13.4 (`build.json`) |
| streams and threads | planning takes a module lock; the convenience path keeps one workspace arena per (device, stream); prepared launches own their workspace |
| CUDA graphs | prepared launches allocate nothing on the device and capture into a CUDA graph (K2: the split, the memset of `out`, the program); no host synchronization.  A convenience-path call that this package launches from its arenas (its own programs, and the deterministic programs it launches through its binding, see [Workspace](#workspace)), captured into a CUDA graph after a warm-up call on the capturing stream (so the capture plans and allocates nothing), stays valid until `cake_dense_gemm_nondet_train_convenience_cache_clear()` or `use_deterministic_backend`, which invalidate it: the arenas keep every buffer a launch was bound to, through growths and cache evictions.  A call on a row that runs the deterministic program, which this package hands to the deterministic package's entry point of the same operation -- operands whose launch views the binding cannot rebuild from the call's tensors (for example two operands that alias), a batched forward on that package's swapped small-token program -- follows that package's rule (capture its prepared launch, not its convenience path).  The graphs captured on one stream share its arena (`torch.cuda.graph` without `stream=` captures every graph on one shared stream): replay them one at a time, and not concurrently with calls on that stream |
| other kernels on the device | the programs walk their work persistently over at most `sm_count // 2` CTA pairs and, except the stream-K weight gradient of `K1_WGRAD_SK_RULES_BY_SM`, wait on no CTA that is not resident; that program runs one unit on each 2-CTA cluster, and the unit that sums a split tile waits only for the lower units that hold the tile's earlier K blocks, so a launch whose clusters are not all resident at once still completes when the hardware starts lower clusters no later than higher ones (the observed behaviour of these parts, not a CUDA guarantee) (the forward's SIMT program waits only on the CTAs of its own cluster, which the hardware schedules together); every routed row was measured next to a kernel holding 8, 16 or 32 SMs on another stream (no more than 3 % slower than the deterministic plan there) |
| context parallelism (KCP) | not applicable: a local operator on the caller's CP-local shards; no collectives; the reduction of parameter gradients across ranks stays with the caller |
| autograd, checkpointing, gradient accumulation | none in either package: callers compose the forward and both gradients; `out` is overwritten, not accumulated into |

*Table: the numerical, architecture and parallelism matrix of this module.*

| output (default arguments, the shipped tables) | run to run | why | error against FP64 (relative L2 / worst element) |
|---|---|---|---|
| K1 forward `out` | repeats bit for bit | the deterministic programs: whole tiles, or two K ranges added in a fixed order (the one-wave program of `K1_FORWARD_PROGRAM_RULES_BY_SM`: the same bits) | 1.000 / 1.000 |
| K1 input gradient `out` | repeats bit for bit | the deterministic programs (the rows of `K1_DGRAD_RULES_BY_SM`: another of them, the same bits; the ranges of `K1_DGRAD_PROMO_RANGES_BY_SM`: the deterministic plan with another L2 promotion, the same bits; the input-gradient program of `K1_DGRAD_PROGRAM_RULES_BY_SM` and the nvjet program of `K1_DGRAD_NVJET_RULES_BY_SM`: the same bits) | 1.000 / 1.000 |
| K1 weight gradient `out`, `d_out` or `d_in` at most 1024 | repeats bit for bit | at most two partials per element (two splits), or a fixed order (the cluster program's rank order, chains added in order, the mirror order of four splits, the stream-K program's unit order, the deterministic programs), or whole tiles (one split) | 1.000 / 1.087 |
| K1 weight gradient `out`, both sides above 1024 | repeats bit for bit | the deterministic programs, and the two-split and one-split rows of the tables | 1.000 / 1.000 |
| K2 forward `out` (router logits) | repeats bit for bit | the SIMT program (rank-ordered) or the deterministic program | 1.000 / 1.000 |
| K2 input gradient `out` | repeats bit for bit | the SIMT program or the deterministic program (one split) | 1.000 / 1.000 |
| K2 weight gradient `out` | repeats bit for bit | the SIMT program, the K-order program (partials added in K order), a `"waves"` partition of at most two partials per element, or the deterministic program at the split count of the router tables (partials added in a fixed order) | 1.012 / 1.095 |
| any output with a caller's `splits >= 3` (K1 in the arrival order, K2 weight gradient) | may vary in the last bits | the partials are added in the order they finish; PyTorch's deterministic-algorithms alert fires before such a launch | not measured (no shipped row) |

*Table: whether each output repeats bit for bit across runs, why, and its error against FP64.  The table states the shipped tables with default arguments; a row moves to a program whose bits vary only under the speed rule of [Acceptance rules](#acceptance-rules), and its launch raises the alert above.  The error column is the largest error against FP64 over every benchmark row of the output (and the rows at 16384 tokens), the six input kinds of the accuracy rule and five runs, on a B200, a GB200, a GB300 and an R200 part, as a multiple of the median error of the reference implementation (FlashInfer commit `2e989556`) in the same run: relative L2 error, then the largest element error.  The accuracy rule allows 1.1.*

## Acceptance rules

Two rules decide whether an output of this module may come from a program whose results can change from run to run.
They apply to every program this package adds or keeps, on every supported architecture.

- **Accuracy.** At every benchmarked shape (the rows of [Benchmark](#benchmark)), each output's error against an FP64
  reference of the same operands stays within 1.1 times the median error of the reference implementation (FlashInfer
  commit `2e989556`; flashinfer-ai/flashinfer#8946) at that shape.  This holds for random normal inputs and for
  adversarial inputs: long sums, cancellation, values concentrated in a few elements, and values near the largest
  magnitude of the dtype.  Partial sums stay in FP32 or wider, and each output element is rounded once.  The
  absolute budgets of the table above apply as well.  Outside the benchmarked shapes, two 212-SM `sm_107a` FP32
  weight-gradient ranges keep the program they ran before this change although it missed this rule at some token
  counts: `5120 -> 512` at odd K-block counts from 40962 tokens (up to 1.83 times, at 57345 tokens) and `4096 -> 512`
  at odd counts from 24577 to 32767 tokens (1.12 times at 25601 tokens); see the accuracy and repeatability changes
  under [Planners and rule tables](#planners-and-rule-tables).
- **Speed.** An output may vary from run to run only where that makes the module at least 3 % faster at the shape:
  the package's own time for the shape (its forward, input gradient and weight gradient through the public entry
  points) must drop by at least 3 % against the program whose result repeats bit for bit.  Everywhere else the
  output repeats bit for bit.  No row of the shipped tables takes such a program: with default arguments every output
  repeats bit for bit.  The benchmark checks each operation against a ratio of 0.97 (`ACCEPT_RATIO`) as a proxy for
  this rule; a varying row it accepts is checked again on the package time of its shape.

## Planners and rule tables

A planner decides, per row and device, whether a call runs this package's program and with which
parameters; a row it leaves runs the deterministic program (`cake_dense_gemm_nondet_train_projection_plan` / `cake_dense_gemm_nondet_train_router_plan`
say which).  Only measured rows carry a program: a row runs this package's program where that program was faster
than the deterministic package's shipped plan of the row in every measurement run: a one-sided paired sign test
(p < 0.05) with a median time ratio below 0.99 (the bar of the split, cluster and router verdicts of these tables,
whose outputs repeat bit for bit; the program tables of the forward, the input gradient and the weight gradient state
their own measurement; a verdict whose output varies run to run also needs the 3 % gain of the speed rule, see
[Acceptance rules](#acceptance-rules)), both in eager launches and in CUDA graph replays,
and no more than 3 % slower than that plan next to a kernel holding 8, 16 or 32 SMs on another stream.  A `None` entry is a measured
row that missed; a row without an entry was not measured.  Both run the deterministic program.

The K1 verdicts were measured at 16172 to 16384 tokens.  Those of `K1_SPLIT_RULES` hold at every token count their split
count fits (`num_splits * 64 <= T + 63`), except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS` (the K1 token
tables below: `sm_100a` / `sm_103a` parts of 152 SMs or more, `sm_107a` parts of 208, 212 and 216 SMs), on parts
of exactly 148 SMs those of `K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` (the 148-SM K1 token table below), and, where the
token count spans an odd number of 64-token K blocks, those of `K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` (the odd-block K1
table below); the tables of
`K1_SPLIT_RULES_BY_SM` and the overrides of `K1_SPLIT_OVERRIDES_BY_SM` hold only at 16172 to 16384 tokens
(`K1_SPLIT_RULES_BY_SM_TOKENS`) and, for the rows of `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`, at the token ranges of that
table (the 212-SM K1 token-range table below), except those token counts; the rows of
`K1_SPLIT_RANGE_OVERRIDES_BY_SM` run that table's program at its token ranges (the same table), and those of
`K1_TWO_SPLIT_RANGES_BY_SM` at its token ranges, at either number of 64-token K blocks (the same table on 212 SMs,
`4096 -> 128` FP32 included; on 152-SM `sm_100a` / `sm_103a` parts the 152-SM two-split table below).  Before all of
these, the rows of `K1_CHAIN_RANGES_BY_SM` run the one-split program of several accumulation chains at that table's
token ranges on parts of exactly its SM count, at both block parities (the K1 accumulation-chain table below); next,
the rows of `K1_ONE_SPLIT_RANGES_BY_SM` run the one-split program of one chain at that table's token ranges on parts
of exactly its SM count, at both block parities (the 152-SM one-split table below).  Ahead of every K1 table, the FP32
rows of `K1_DSM_RANGES_BY_SM` run the cluster program at that table's token ranges on parts of exactly its SM count
(`plan_atomic_gemm` asks `k1_dsm_plan` first; the cluster tables below), but at an odd number of 64-token K blocks
inside a range of `K1_DSM_ODD_BLOCK_RANGES_BY_SM`, where they run that table's program (`k1_dsm_odd_block_plan`; the
parity table below).  Next, before every other K1 table, the FP32 `sm_107a` rows of `K1_PARITY_RANGES_BY_SM` run
the program of their range by the parity of their number of 64-token K blocks (`k1_parity_plan`; parts of more SMs
than the table's largest count take that count's ranges, `K1_PARITY_RANGES_ABOVE_SM`), and then the rows of
`K1_ORDERED_RANGES_BY_SM` run that table's
CTA-pair program, its split order included, at its token ranges on parts of exactly its SM count (`k1_ordered_plan`).
A K1 program runs only at token counts where its output met the deterministic package's budgets against FP64 (the accuracy
row above) on every measured seed; the ratio of its FP64 relative Frobenius error to the deterministic program's on
the same operands (q) is reported in the tables below as data, not as a rule.  Where both programs miss the budget,
the row keeps its route.
The router verdicts hold only inside token windows (`ROUTER_RULES`; the router window tables below) and, except the
forward's windows (the rule of the forward SIMT table below), pass a stricter rule: measured in one long-lived process and in a fresh process per token count, and no more than 3 % slower
than the deterministic plan next to such a kernel both in the same launch order and with the router partition launched
before that kernel against the deterministic plan launched after it (the 152-SM head/tail windows of the weight
gradient measured again against the deterministic package's split rules take that slowdown as data, not as a rule:
the head/tail window table below).  A router window also keeps the router's accuracy
budget: for every seed, a relative Frobenius error against FP64 of the exact operands (worst of 20 runs) of at most
`2e-6`, 10 seeds per measured token count (the router FP64 error table below; the error's ratio to that of the plan
the row takes without the window is reported as data, not a gate).  Every other token count
of a router row runs the deterministic program.  The K1 verdicts hold from the floor of `K1_MIN_SM_COUNT` /
`K1_RULE_MIN_SM_COUNT` up (a part with more SMs than the measured ones takes them without a measurement, a part with
fewer runs the deterministic program), except the `sm_107a` token ranges measured on 216-SM parts only, which parts of
more SMs do not take (the entries marked † in the `sm_107a` K1 token table and the odd-block K1 table); the tables of
`K1_SPLIT_RULES_BY_SM` hold for exactly 208 and 212 SMs, the overrides of `K1_SPLIT_OVERRIDES_BY_SM` for exactly 216
SMs.  The router windows hold only on the SM count they were measured on (`ROUTER_SM_COUNT`: 152 on `sm_100a` /
`sm_103a`, 216 on `sm_107a`), and those of `ROUTER_RULES_BY_SM` only on exactly the SM count of their table (148 on
`sm_100a` / `sm_103a`, 208 and 212 on `sm_107a`); every other SM count runs the deterministic plan, except in the
`"waves"` windows, which keep the earlier SM condition of `"waves"` (the deterministic plan's last wave below 80 %
full) and run their partition only where every element takes at most two partials (else the deterministic plan).  A
SIMT window comes first in its row and may overlap the windows after it: on an SM count where it does not
hold, the token count takes the next window that holds it.  The measured parts have 148 and 152 SMs (`sm_100a` /
`sm_103a`) and 208, 212 and 216 SMs (`sm_107a`).

| table | key | value |
|---|---|---|
| `K1_SPLIT_RULES` | `(arch, L, M, N, out_dtype)`: the transposed-store launch `out[l, n, m]` (`M` = the stored inner extent, `d_in` of a weight gradient; `N` = its rows, `d_out`), `out_dtype` `"bf16"` / `"f32"` | `(block_n, num_splits)` (this package's program; `block_n` 192 is the row-major-store program with the operands exchanged) or `None` (the deterministic program) |
| `K1_MIN_SM_COUNT` | `arch` | the fewest SMs for the K1 verdicts of the architecture (`sm_100a` / `sm_103a`: 148, `sm_107a`: 216) |
| `K1_RULE_MIN_SM_COUNT` | a key of `K1_SPLIT_RULES` | the fewest SMs for that row's verdict where it is above `K1_MIN_SM_COUNT` (`sm_103a` `4096 -> 48`: 152) |
| `K1_SPLIT_RULES_BY_SM` | `(arch, sm_count)` | the K1 verdicts of the parts with exactly `sm_count` SMs below a row's floor (keys and values as in `K1_SPLIT_RULES`), at the token counts of `K1_SPLIT_RULES_BY_SM_TOKENS` and of the row's ranges in `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` only; a part without a table, and every other token count, runs every K1 row below the floor on the deterministic program (`sm_107a`: 208 and 212 SMs) |
| `K1_SPLIT_RULE_EXCLUDED_TOKENS` | a key of `K1_SPLIT_RULES` or of a table of `K1_SPLIT_RULES_BY_SM` | `((sm_count, min_tokens, max_tokens), ...)`: the row runs the deterministic program at a token count from `min_tokens` to `max_tokens` (`None`: no upper bound) on a part of at least `sm_count` SMs (`sm_count` at or above the row's SM-count floor) or of exactly `sm_count` SMs (below it); every other token count and part keeps the verdict |
| `K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM` | `(arch, sm_count)`, then a key of `K1_SPLIT_RULES` | `((min_tokens, max_tokens), ...)`: the row runs the deterministic program at a token count from `min_tokens` to `max_tokens` (`None`: no upper bound) on a part of exactly `sm_count` SMs; every other token count and part keeps its routing (`sm_100a` / `sm_103a`: 148 SMs, the 148-SM K1 token table below; `sm_103a`: 152 SMs, the FP32 `5120 -> x` token ranges of the accuracy note) |
| `K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS` | a key of `K1_SPLIT_RULES` or of a table of `K1_SPLIT_RULES_BY_SM` | `((sm_count, min_tokens, max_tokens), ...)` (parts as in `K1_SPLIT_RULE_EXCLUDED_TOKENS`): the row runs the deterministic program at a token count from `min_tokens` to `max_tokens` (`None`: no upper bound) that spans an odd number of 64-token K blocks (`ceil(T / 64)` odd); an even number, and every other token count and part, keeps its routing (FP32 two-split rows of `sm_100a` and `sm_103a`: the odd-block K1 table below) |
| `K1_SPLIT_RULES_BY_SM_TOKENS` | -- | `(min_tokens, max_tokens)`: the token counts (`K` of the weight gradient) the tables of `K1_SPLIT_RULES_BY_SM` and `K1_SPLIT_OVERRIDES_BY_SM` hold for, `(16172, 16384)` (where they were measured) |
| `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` | `(arch, sm_count)`, then a key of that part's table in `K1_SPLIT_RULES_BY_SM` | `((min_tokens, max_tokens), ...)`: further token counts at which the row takes its verdict from the part's table, except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS`; every other row, part and token count keeps `K1_SPLIT_RULES_BY_SM_TOKENS` alone (`sm_107a`: 212 SMs, the 212-SM K1 token-range table below) |
| `K1_SPLIT_RANGE_OVERRIDES_BY_SM` | `(arch, sm_count)`, then a key of that part's table in `K1_SPLIT_RULES_BY_SM` | `((min_tokens, max_tokens, (block_n, num_splits)), ...)`: on a part of exactly `sm_count` SMs below the row's floor, a token count from `min_tokens` to `max_tokens` runs that program, except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS`; a range lies outside `K1_SPLIT_RULES_BY_SM_TOKENS` and the row's `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` (`sm_107a`: 212 SMs, the 212-SM K1 token-range table below) |
| `K1_TWO_SPLIT_RANGES_BY_SM` | `(arch, sm_count)`, then a key as in `K1_SPLIT_RULES` (below the row's floor: a key of that part's table in `K1_SPLIT_RULES_BY_SM`) | `((min_tokens, max_tokens, (block_n, num_splits)), ...)`: on a part of exactly `sm_count` SMs, below the row's floor or at or above it, a token count from `min_tokens` to `max_tokens` runs that program at either number of 64-token K blocks, except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS`; a range lies outside `K1_SPLIT_RULES_BY_SM_TOKENS` and the row's ranges in `K1_SPLIT_RULES_BY_SM_TOKEN_RANGES` and `K1_SPLIT_RANGE_OVERRIDES_BY_SM` (`sm_107a`: 212 SMs, the two-split rows of the 212-SM K1 token-range table below, `4096 -> 128` FP32 without a verdict in the part's table of `K1_SPLIT_RULES_BY_SM`; `sm_100a` / `sm_103a`: 152 SMs, the 152-SM two-split table below) |
| `K1_SPLIT_OVERRIDES_BY_SM` | `(arch, sm_count)` | verdicts (keys and values as in `K1_SPLIT_RULES`) that replace a row's `K1_SPLIT_RULES` verdict on parts of exactly `sm_count` SMs at or above the row's floor, at the token counts of `K1_SPLIT_RULES_BY_SM_TOKENS` only, except the token counts of `K1_SPLIT_RULE_EXCLUDED_TOKENS`; every other token count and SM count keeps the `K1_SPLIT_RULES` verdict (`sm_107a`: 216 SMs, `4096 -> 1024` FP32 (192, 1), the row-major-store program) |
| `K1_CHAIN_RANGES_BY_SM` | `(arch, sm_count)`, then a key of `K1_SPLIT_RULES` | `((min_tokens, max_tokens, (block_n, chains)), ...)`: on a part of exactly `sm_count` SMs, a token count from `min_tokens` to `max_tokens` runs this package's one-split program of that tile width with that many accumulation chains (`k1_chain_plan`; the chain counts this package ships are `K1_CHAINS_CHOICES`), before every table above, `K1_TWO_SPLIT_RANGES_BY_SM` included (`sm_107a`: 212 and 216 SMs, the K1 accumulation-chain table below) |
| `K1_PARITY_RANGES_BY_SM` | `(arch, sm_count)`, then a key as in `K1_SPLIT_RULES` | `((min_tokens, max_tokens, even, odd), ...)`: on a part of exactly `sm_count` SMs (or of more SMs than every count of the architecture there: the ranges of `K1_PARITY_RANGES_ABOVE_SM`), a token count from `min_tokens` to `max_tokens` runs `even` at an even number of 64-token K blocks and `odd` at an odd number, each a `(block_n, splits, chains, order)` of this package's CTA-pair program, `()` for the deterministic program or `None` for the other tables (`k1_parity_plan`), before every table but the cluster ranges (`sm_107a` FP32 `4096 / 5120 -> 512 / 1024`: 208, 212, 216 and more SMs from 961 tokens) |
| `K1_ONE_SPLIT_RANGES_BY_SM` | `(arch, sm_count)`, then a key as in `K1_SPLIT_RULES` | `((min_tokens, max_tokens, block_n), ...)`: on a part of exactly `sm_count` SMs, a token count from `min_tokens` to `max_tokens` runs this package's one-split program of that tile width and one accumulation chain, at an even and an odd number of 64-token K blocks alike, before every table above except `K1_CHAIN_RANGES_BY_SM` (`sm_100a` / `sm_103a`: 152 SMs, the 152-SM one-split table below) |
| `K1_DSM_RANGES_BY_SM` | `(arch, sm_count)`, then a key as in `K1_SPLIT_RULES` (FP32 rows) | `((min_tokens, max_tokens, (block_n, splits)), ...)`: on a part of exactly `sm_count` SMs, a token count from `min_tokens` to `max_tokens` runs this package's cluster program `dense_proj_gemm_dsm_nn_n<block_n>_f32_t_s<splits>` (`k1_dsm_plan`): tiles of 128 `d_in` columns by `block_n` `d_out` rows, one cluster of `splits` CTAs per tile, a K range per CTA, before every other K1 table (`sm_100a`: 148 and 152 SMs, `sm_103a`: 152 SMs; `sm_107a`: 212 SMs, four CTAs of 64-wide tiles at `4096 -> 48` and `4096 -> 64` from 16172 to 16384 tokens; the cluster tables below) |
| `K1_DSM_ODD_BLOCK_RANGES_BY_SM` | `(arch, sm_count)`, then a key of that part's table in `K1_DSM_RANGES_BY_SM` | `((min_tokens, max_tokens, program), ...)`: on a part of exactly `sm_count` SMs, a token count from `min_tokens` to `max_tokens` that spans an odd number of 64-token K blocks and lies in one of the row's cluster ranges runs `program` in place of the cluster program (`k1_dsm_odd_block_plan`): `None` the deterministic program, `(block_n, splits)` the cluster program of that tile as in `K1_DSM_RANGES_BY_SM`, `(block_n, splits, chains, order)` this package's CTA-pair program as in `K1_ORDERED_RANGES_BY_SM`; an even block count keeps the cluster program of the range (`sm_100a`: 148 and 152 SMs, `sm_103a`: 152 SMs, `4096 -> 48` and `4096 -> 64` four CTAs of 64-wide tiles; the parity table below) |
| `K1_ORDERED_RANGES_BY_SM` | `(arch, sm_count)`, then a key as in `K1_SPLIT_RULES` | `((min_tokens, max_tokens, (block_n, splits, chains, order)), ...)`: on a part of exactly `sm_count` SMs, a token count from `min_tokens` to `max_tokens` runs this package's CTA-pair program of that tile width, split count and chain count, its splits added in `order` (`"arrival"` or `"mirror"`, `K1_SPLIT_ORDERS`; `k1_ordered_plan`), before every K1 table but `K1_DSM_RANGES_BY_SM` (`sm_107a`: 212 SMs, FP32 `4096 -> 32` four splits in the mirror order; FP32 `4096 -> 1024` and `5120 -> 1024` four chains at an odd number of K blocks and two at an even number, `5120 -> 512` two chains at an even number; `sm_100a`: 148 SMs, BF16 `4096 -> 1024` one split of two chains of 256-wide tiles from 16172 to 16384 tokens) |
| `K1_FORWARD_RULES_BY_SM` | `(arch, sm_count)`, then `(K, N, T)` of the forward `out[T, N] = X[T, K] @ W[N, K].T` | the deterministic package's planner knobs of the row's program (`K1_FORWARD_SINGLE_CTA_KNOBS`), or a program symbol of that package (planned with the knobs that name it), on a part of exactly `sm_count` SMs at exactly `T` tokens with both operands K-major and a BF16 `out` (`k1_forward_knobs`); every other part, row, token count, layout and output dtype runs the deterministic plan (`sm_100a` / `sm_103a`: 152 SMs, the K1 forward table below; named programs at 16172 to 16384 tokens on 148-SM `sm_100a`, 152-SM `sm_100a` / `sm_103a` and 212-SM `sm_107a` parts; on 212-SM `sm_107a` parts also `kk_n128_hen_pro15` for `5120 -> 32 / 48 / 64` and `8192 -> 128` and `kk_n192_m256_hen_s5_pro15` for `4096 / 5120 -> 512` at 16172, 16231 and 16384 tokens, except `4096 -> 512` at 16384) |
| `K1_FORWARD_PROGRAM_RULES_BY_SM` | `(arch, sm_count)`, then `(K, N, T)` of the forward | this package's one-wave forward program `dense_proj_gemm_wave_lean_kk_n224_ef_e8_pf2_st`, at 512 features and 16172 / 16231 tokens its form with L2 eviction priorities `_lh` (the deterministic plan's bits) on a part of exactly `sm_count` SMs at exactly `T` tokens, both operands K-major, BF16 `out`, before `K1_FORWARD_RULES_BY_SM` (`sm_100a`: 148 and 152 SMs, `sm_103a`: 152 SMs) |
| `K1_DGRAD_RULES_BY_SM` | `(arch, sm_count)`, then `(K, N, T, out_dtype)` of the input gradient `out[T, K] = G[T, N] @ W[N, K]` | a program of the deterministic package (its template without the `dense_proj_det_gemm_` prefix), planned with the knobs that name it (the deterministic plan's bits), on a part of exactly `sm_count` SMs at exactly `T` tokens, unit inner strides (`sm_100a`: 148 and 152 SMs, `sm_103a`: 152 SMs, `sm_107a`: 212 SMs) |
| `K1_DGRAD_PROMO_RANGES_BY_SM` | `(arch, sm_count)`, then `(K, N, out_dtype)` of the input gradient | token ranges `(lo, hi, promo)`: the deterministic plan of the row with the planner knob `promo` (its TMA L2 promotion; the same program and bits) on a part of exactly `sm_count` SMs at `lo <= T <= hi`, unit inner strides, after `K1_DGRAD_RULES_BY_SM` (`k1_dgrad_promo`, `k1_dgrad_knobs`; `sm_100a`: 148 SMs) |
| `K1_DGRAD_PROGRAM_RULES_BY_SM` | as `K1_DGRAD_RULES_BY_SM` | this package's input-gradient program (`dense_proj_gemm_dgrad_kn_n256_f32_ef`, `_ef_tma`, `_tma_lean_free`; `dgrad_program_template`) of FP32 `out` (the deterministic plan's bits), before `K1_DGRAD_NVJET_RULES_BY_SM` and `K1_DGRAD_RULES_BY_SM` (`sm_100a`: 148 and 152 SMs, `sm_103a`: 152 SMs) |
| `K1_DGRAD_NVJET_RULES_BY_SM` | as `K1_DGRAD_RULES_BY_SM` | this package's program `dense_proj_gemm_nvjet_early_kn_n256_lean`, or `dense_proj_gemm_nvjet_early_kn_n256_lean_free` (the epilogue frees the tensor memory under the last store drain), on the deterministic package's plan of `kn_n256_tma1_w8r` (BF16 `out`, the same bits), before `K1_DGRAD_RULES_BY_SM` (`sm_100a`: 148 and 152 SMs, `sm_103a`: 152 SMs, `sm_107a`: 212 SMs) |
| `K1_WGRAD_SK_RULES_BY_SM` | `(arch, sm_count)`, then `(d_in, d_out, T, out_dtype)` of the weight gradient `out[d_out, d_in]` | this package's stream-K program `dense_proj_gemm_sk_nn_n256[_f32]_t` (a split tile summed in unit order: the same bits on every run) on a part of exactly `sm_count` SMs at exactly `T` tokens with the default knobs, before every other weight-gradient table (`sm_100a`: 148 SMs, `sm_103a`: 152 SMs, `sm_107a`: 212 SMs) |
| `K1_WGRAD_DET_RULES_BY_SM` | as `K1_WGRAD_SK_RULES_BY_SM` | a program of the deterministic package (`nn_n192_bz64_hne(_f32_v8_ef)_sks_sb2_pro15`: two token ranges added in a fixed order), planned with the knobs that name it, on a row the planner keeps on the deterministic program, with the default `sk` and `cta_rows` (`sm_100a`: 148 SMs) |
| `ROUTER_RULES` | `(arch, layout, hidden, experts)`: `layout` `"kk"` (forward, `hidden = K`), `"nn_t"` (weight gradient, `hidden` = the launch's `M`) or `"kn"` (input gradient, `hidden` = the launch's `N`), `experts` = the launch's `N` (`K` of the input gradient) | `None` (the deterministic program), token windows `((min_tokens, max_tokens, spec), ...)` (the spec of the window that holds the token count -- `K` of the weight gradient, `M` of the forward and of the input gradient; every other token count: the deterministic program) or a spec alone (every token count: a plan a test or a measurement forces).  A spec is `"waves"`, `"H<f>x<k>"`, `"H<f>x<k>p<h>"`, `"forward"`, `"pairs"`, `"tiles"`, `"S<n>"`, a unit count (this package's stream-K program), `"det_s<n>"` (the deterministic program at `n` splits), `"ksum_S<n>"` (the weight gradient's K-order program at `n` spans per tile, `ROUTER_KSUM_TEMPLATES_BY_SM`) or `"simt_n<tile_n>m<tile_m>s<slices>[<form>]"` (the exact-FP32 SIMT program of the row's layout, `router_simt`: plain without a form, compensated with `g<group>` or, on a weight-gradient row, `dot2`; the only spec of an input-gradient row) or, on a forward row, `"simt_c8_<kind>_l<lanes>_e<experts>t<tokens>"` (the forward's exact-FP32 cluster SIMT program, `router_forward_simt`; the only spec of a forward row) |
| `ROUTER_SM_COUNT` | `arch` | the SM count of the router verdicts of the architecture (`sm_100a` / `sm_103a`: 152, `sm_107a`: 216); a device with any other SM count runs the deterministic program, except in a `"waves"` window (the condition of `router_waves`, at most two partials per element; below a SIMT window too) or with a table of its own in `ROUTER_RULES_BY_SM` |
| `ROUTER_RULES_BY_SM` | `(arch, sm_count)` | the router verdicts of the parts with exactly `sm_count` SMs (keys and values as in `ROUTER_RULES`; `router_rules` picks the table of a device); such a part takes its own table only: a row without an entry, and every token count outside its windows, runs the deterministic program (`sm_100a` / `sm_103a`: 148 SMs; `sm_107a`: 208 and 212 SMs) |
| `ROUTER_KSUM_TEMPLATES_BY_SM` | `(arch, sm_count)` | the K-order program that a `"ksum_S<n>"` window runs on the parts with exactly `sm_count` SMs (`sm_107a`: 212 SMs, `router_fp32_gemm_ts_r6p6_atomic_nn_t_ksum_ht_c2_nreg`); a part without an entry runs the deterministic program in such a window |

*Table: the rule tables.*

| architecture | K1 weight gradient `d_in -> d_out` (`block_n`, `num_splits`), BF16 and FP32 out unless marked | K2 forward | K2 weight gradient |
|---|---|---|---|
| `sm_100a` (>= 148 SMs) | `4096 -> 32, 48, 64`; `5120 -> 32, 48, 64`; `6144 -> 32`: (64, 2).  `4096 -> 128`: deterministic (measured, missed); FP32 on 152 SMs: (64, 2) from 1024 to 4096 tokens (the 152-SM two-split table).  FP32 on 152 SMs, first: `4096 -> 32` (64, 1) from 256 to 2048 tokens, `4096 -> 128` (64, 1) from 256 to 3328 tokens, at every K-block count (the 152-SM one-split table).  FP32 on 152 SMs, before all of these: the cluster program of the 152-SM cluster table (`4096 -> 32` from 1024 to 2048 and from 6144 to 32768 tokens, `4096 -> 48` from 1024 to 2048, from 4096 to 8192 and from 16172 to 32768, `4096 -> 64` and `5120 -> 32` from 1024 to 32768, `4096 -> 128` from 4096 to 8192, `5120 -> 48` from 1024 to 24576, `5120 -> 64` from 1024 to 2048 and from 4096 to 24576, `5120 -> 128` from 1024 to 6144).  FP32 on 148 SMs, before all of these: the cluster program of the 148-SM cluster table (`4096 -> 32`, `5120 -> 32`, `5120 -> 48`, `5120 -> 64` and `5120 -> 128` from 1024 to 32768 tokens, `4096 -> 48` from 1024 to 16384 and at 32768, `4096 -> 64` from 1024 to 6144 and from 12288 to 32768, `4096 -> 128` from 1024 to 12288).  On 148 and 152 SMs `4096 -> 48` and `4096 -> 64` run 32-wide tiles from 16172 to 16384 tokens, and at an odd number of K blocks there four CTAs of 64-wide tiles (the parity table).  BF16 `4096 -> 1024` on 148 SMs from 961 to 16384 tokens: (256, 1) `p = 1`, two chains (`K1_ORDERED_RANGES_BY_SM`) | 256 experts on 152 SMs, in the token windows of the forward SIMT table: the cluster SIMT program (hidden 4096 from 1 to 160 tokens, hidden 5120 and 6144 from 1 to 192: `"simt_c8_kg4_l32_e8t4"` 1 to 8, `"simt_c8_kg4_l32_e8t16"` 9 to 16, `"simt_c8_ffma_l16_e16t32"` 17 to 96, `"simt_c8_ffma_l16_e16t64"` 97 to 128, `"simt_c8_ffma_l16_e16t32"` from 129).  Every other token count and SM count: deterministic (4 splits) | 256 experts, in the token windows of the router window tables: on 152 SMs the SIMT program (hidden 4096 `"simt_n128m64s2"` from 1 to 224 tokens, hidden 5120 and 6144 `"simt_n64m64s2"` from 1 to 256 tokens) and the deterministic program at 6 splits at hidden 6144 (16129 to 16384 tokens); on 148 SMs the SIMT programs of the 148-SM SIMT table below (hidden 4096 from 1 to 224 tokens, hidden 5120 and 6144 from 1 to 256 tokens); hidden 5120 `"waves"` on every other SM count where it applies and gives every element at most two partials (up to 64 tokens).  Every other token count and SM count, hidden 4096 above 224 tokens and hidden 6144 above 256 tokens: deterministic (6 splits; on 148 and 152 SMs 9 splits at hidden 5120 and 3 at hidden 6144 from 16129 to 16384 tokens) |
| `sm_103a` (>= 148 SMs) | `4096 -> 32`; `5120 -> 32, 48, 64`; `6144 -> 32`: (64, 2); `4096 -> 48` (>= 152 SMs): (64, 2); `5120 -> 128`, `6144 -> 128`: (128, 2).  `4096 -> 64`, `4096 -> 128`: deterministic (measured, missed); `4096 -> 128` FP32 on 152 SMs: (64, 2) from 1024 to 6144 tokens (the 152-SM two-split table).  FP32 on 152 SMs, first: `4096 -> 32` (64, 1) from 256 to 2048 tokens, `4096 -> 128` (64, 1) from 256 to 2816 tokens, at every K-block count (the 152-SM one-split table).  FP32 on 152 SMs, before all of these: the cluster program of the 152-SM `sm_103a` cluster table (`4096 -> 32` and `4096 -> 64` from 1024 to 32768 tokens, `4096 -> 48` from 1024 to 6144 and from 12288 to 32768; `4096 -> 48` and `4096 -> 64` 32-wide tiles from 16172 to 16384 tokens, and at an odd number of K blocks there four CTAs of 64-wide tiles (the parity table); `4096 -> 128` from 1024 to 12288, `5120 -> 32` from 1024 to 2048, from 6144 to 8191, from 12288 to 16383 and from 16385 to 32768, `5120 -> 48` from 2048 to 4095, `5120 -> 64` from 1024 to 4095 and from 6144 to 12287, `5120 -> 128` from 1024 to 4095); FP32 on 152 SMs, deterministic: `5120 -> 32` from 8192 to 12287 and at 16384 tokens, `5120 -> 48` from 4096 to 8192, `5120 -> 64` from 4096 to 6143 and from 12288 to 24576, `5120 -> 128` from 4097 to 32768, and BF16 `5120 -> 48` at 16231 tokens (the 152-SM entries of `K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`) | 256 experts on 152 SMs, in the token windows of the forward SIMT table: the cluster SIMT program from 1 to 192 tokens at hidden 4096, 5120 and 6144 (`"simt_c8_kg4_l32_e8t4"` 1 to 16, `"simt_c8_ffma_l16_e16t32"` 17 to 96, `"simt_c8_ffma_l16_e16t64"` 97 to 128, `"simt_c8_ffma_l16_e16t32"` 129 to 192).  Every other token count and SM count: deterministic (4 splits) | 256 experts, in the token windows of the router window tables: on 152 SMs the SIMT program `"simt_n64m64s2"` (hidden 4096 from 1 to 288 tokens, hidden 5120 from 1 to 192 tokens, hidden 6144 from 1 to 256 tokens), the deterministic program at 6 splits at hidden 6144 (16129 to 16384 tokens); on 148 SMs the SIMT programs of the 148-SM SIMT table below (hidden 4096 from 1 to 224 tokens, hidden 5120 and 6144 from 1 to 256 tokens); hidden 5120 `"waves"` on every other SM count where it applies and gives every element at most two partials (up to 64 tokens).  Every other token count and SM count: deterministic (6 splits; on 148 and 152 SMs 9 splits at hidden 5120 and 3 at hidden 6144 from 16129 to 16384 tokens) |
| `sm_107a` (>= 216 SMs) | `4096 -> 32` FP32 (64, 2); `5120 -> 32`, `5120 -> 48` FP32 (64, 2); `4096 -> 32` BF16, `4096 -> 48`, `4096 -> 64`, `5120 -> 64` FP32, `5120 -> 32` BF16, `6144 -> 32`: deterministic (their three- and four-split programs added the partials in the order they finish, see the note below); `8192 -> 128` (128, 2) (FP32 at odd block counts from 1025 to 8192 tokens: on exactly 216 SMs, the odd-block K1 table); `4096 -> 512` FP32 (128, 1) `p = 1`; `5120 -> 512` FP32 (128, 1) `p = 1`; `4096 -> 1024` FP32 (256, 1) `p = 1` (on exactly 216 SMs at 16172 to 16384 tokens: (192, 1) `p = 1`, the row-major-store program, 176 CTAs); `5120 -> 1024` (256, 1) `p = 1`, except the token counts of the `sm_107a` K1 token table (the four FP32 rows with one split: from 40961 tokens up, `5120 -> 1024` from 32769; on more than 216 SMs from 961 tokens up).  `4096 -> 64` BF16, `4096 -> 128`, `4096 -> 512` BF16, `4096 -> 4096` FP32: deterministic (measured, missed) | deterministic (4 splits) | 256 experts, in the token window of the router window tables: none (the head/tail window of hidden 5120 is gone, see the note below).  Every token count and SM count: deterministic (6 splits) |
| `sm_107a` (208 SMs, 16172 to 16384 tokens) | `4096 -> 32`, `5120 -> 32, 48` FP32: (64, 2); `4096 -> 48, 64`, `5120 -> 64` FP32: deterministic; `4096 -> 512`, `5120 -> 512` FP32 (128, 1) `p = 1`; `4096 -> 1024`, `5120 -> 1024` FP32: (256, 1) `p = 1`; except the token counts of the `sm_107a` K1 token table.  `4096 -> 32, 64` BF16, `5120 -> 32` BF16, `6144 -> 32`: deterministic (no verdict: its stream-K program did not complete next to a resident kernel); `4096 -> 512` BF16, `5120 -> 1024` BF16, `4096 -> 4096` FP32, `8192 -> 128`: deterministic (measured, missed) | deterministic (4 splits) | none (the head/tail window of hidden 5120 is gone, see the note below): deterministic (6 splits) |
| `sm_107a` (212 SMs, 16172 to 16384 tokens and the ranges of the 212-SM K1 token-range table) | `4096 -> 32` FP32 (64, 2), from 16172 to 16384 tokens four splits in the mirror order (`K1_ORDERED_RANGES_BY_SM`); `5120 -> 32, 48` FP32 (64, 2); `4096 -> 48, 64` FP32: four CTAs of 64-wide tiles from 16172 to 16384 tokens (`K1_DSM_RANGES_BY_SM`); `5120 -> 64` FP32: deterministic; `8192 -> 128` (128, 2); `4096 -> 512`, `5120 -> 512` FP32 (128, 1) `p = 1`; `4096 -> 1024` FP32: (192, 1) `p = 1` (row-major-store program, 176 CTAs); `5120 -> 1024` FP32: (256, 1) `p = 1`; from 16172 to 16384 tokens both one split of four accumulation chains at an odd number of 64-token K blocks and of two at an even number, and `5120 -> 512` FP32 two chains at an even number (`K1_ORDERED_RANGES_BY_SM`); except the token counts of the `sm_107a` K1 token table.  At the two-split ranges of the 212-SM K1 token-range table, at either number of 64-token K blocks (`K1_TWO_SPLIT_RANGES_BY_SM`): `4096 -> 512`, `5120 -> 512` FP32 (128, 2); `4096 -> 1024` FP32 (192, 2) (row-major-store program); `5120 -> 1024` FP32 (256, 2); `4096 -> 128` FP32 (64, 2) from 1024 to 7680 tokens (no verdict in the 212-SM table: deterministic at every other token count).  `4096 -> 32, 64` BF16, `5120 -> 32` BF16, `6144 -> 32`: deterministic (no verdict, as on 208 SMs); `4096 -> 512` BF16, `4096 -> 4096` FP32: deterministic (measured, missed; `4096 -> 4096` FP32 against the deterministic package's 160-wide tiles of the row); `5120 -> 1024` BF16: (256, 1) `p = 1` at its range of the 212-SM K1 token-range table (`K1_SPLIT_RANGE_OVERRIDES_BY_SM`), deterministic at 16172 to 16384 tokens (1.3 to 2.5 % faster there, missed on 208 SMs); `4096 -> 1024` BF16 at 16172 and 16384 tokens, before all of these: the stream-K program of `K1_WGRAD_SK_RULES_BY_SM` | 256 experts, hidden 4096, 5120 and 6144, in the token windows of the 212-SM table of `ROUTER_RULES_BY_SM` (the forward SIMT table below): the cluster SIMT program (`"simt_c8_kg4_l32_e8t4"` 1 to 8 tokens, `"simt_c8_ffma_l16_e16t32"` 9 to 192).  Every other token count: deterministic (4 splits) | 256 experts, hidden 4096 and 5120 from 1 to 256 tokens and hidden 6144 from 1 to 224 tokens: the SIMT program `"simt_n128m64s2"`; hidden 5120 at 16172 to 16384 tokens: the deterministic program at 12 splits; hidden 6144 at 16172 to 16231 tokens: the K-order program at six spans (`"ksum_S6"`).  Every other token count, and hidden 4096 above 256 tokens: deterministic (6 splits) |

*Table: the rows that run this package's programs.  `4096 -> 128` misses on `sm_100a` / `sm_103a` next to a kernel
holding 32 SMs (its 128 CTAs no longer fit the free SMs) and on `sm_107a` at 16384 tokens; on `sm_107a`, `4096 -> 512`
BF16 (192 CTAs) and `4096 -> 4096` FP32 (216 CTAs) miss next to a kernel holding 32 SMs the same way.  `4096 -> 64`
(`sm_103a`; `sm_107a` BF16) and `4096 -> 48` on 148-SM `sm_103a` parts are not faster than the deterministic plan in
every run.  The `sm_107a` one-split rows store whole tiles: they win where the deterministic package's `sm_107a` program
of the row is slower (the FP32 ones on 216 SMs up to 40960 tokens, `5120 -> 1024` up to 32768, and on more SMs up to 960
tokens: from 45056 tokens up they miss the FP32 budget, the `sm_107a` K1 token table; on 212-SM and 216-SM parts these
FP32 rows run three accumulation chains at one split at the ranges of the K1 accumulation-chain table).  Every K1 choice
also needs `num_splits * 64 <= T + 63` (shorter token counts run the deterministic program); the token counts of the K1
token tables below run the deterministic program (`sm_100a` / `sm_103a`: parts of 152 SMs or more, and parts of exactly
148 SMs; `sm_107a`: 208, 212 and 216 SMs); the 208-SM and 212-SM choices and the 216-SM `4096 -> 1024` FP32 choice hold
only at 16172 to 16384 tokens (`K1_SPLIT_RULES_BY_SM_TOKENS`), and the 212-SM choices of the rows of the 212-SM K1
token-range table also at its ranges.*

**Accuracy and repeatability changes.**  Every shipped row repeats bit for bit.  Every shipped row meets the accuracy
rule above (the six input kinds against the reference implementation, FlashInfer commit `2e989556`; measured at every
benchmark row on a B200, a GB200, a GB300 and an R200 part, the error column of the run-to-run table), except two
212-SM `sm_107a` FP32 weight-gradient ranges, which keep their former program although it missed the rule at some
token counts: `5120 -> 512` at odd K-block counts from 40962 tokens and `4096 -> 512` at odd counts from 24577 to 32767
tokens (the first item below).  The changes:

- K1 weight gradient, `sm_107a` FP32 `4096 -> 512`, `5120 -> 512`, `4096 -> 1024` and `5120 -> 1024` from 961
  tokens (`K1_PARITY_RANGES_BY_SM`): one split of three accumulation chains (equal thirds of the K blocks) missed
  mirrored cancellation (up to 12 times the reference's error) and one split of one chain missed all six kinds (up to
  38 times).  These rows now run one split of two chains at an even number of 64-token K blocks (the bits of two
  splits of one chain) and of four at an odd number (eight at `5120 -> 1024`), or the deterministic program where it
  met the check:
  `4096 -> 512` at its even counts from 12001 tokens and at every count from 32768 to 40961 tokens (faster there), and
  `5120 -> 512` from 40962 to 65536 tokens on 208 SMs (two splits missed the check there at the odd counts).
  On 212 SMs the ranges were rechecked per range and parity where this route ran slower than the former one (the former
  program where it met the check at every measured token count, else the fastest program that did; two ranges keep the
  former program although it missed the check at some counts): `4096 -> 1024` runs the deterministic program at the
  even counts from 24576 tokens and `5120 -> 1024` four chains from 16170 to 16172 tokens; `5120 -> 512` keeps the
  former two splits from 40962 tokens (the deterministic bits at the even counts; at the odd counts they missed the
  check at 40962 and 57345 tokens, up to 1.83 times); `4096 -> 512` runs the deterministic program at its odd counts
  from 24577 tokens, the former program up to 32767 tokens (it missed the check at 25601 tokens, 1.12 times, and met
  it at 24577, 28673, 30721 and 32641).
  208-SM parts take the 212-SM ranges, but keep the deterministic program of the two `4096 ->` rows at 16231 and 16384
  tokens and run two chains at the even counts of `4096 -> 512` from 16172 to 16384 tokens; the deterministic plan of
  208 SMs missed the check next to that window (up to 1.38 times at 16171 and 16385 tokens).  Parts of 216 and more SMs
  also run four chains at the odd counts of `4096 -> 1024` from 40962 to 65536 tokens, where the deterministic plan of
  224 SMs missed the check (1.29 times at 40962 tokens).  At 512 features four chains run on a 128-wide program,
  `dense_proj_gemm_atomic_nn_n128_f32_t_c4` (one wave of 128 CTAs, at the former route's time on a 212-SM part), and
  `5120 -> 1024` runs eight chains on `dense_proj_gemm_atomic_nn_n256_f32_t_c8`: four chains reached 1.24 and 1.27 times
  the reference's worst-element error on the concentrated kind at 16300 and 16385 tokens (255 and 257 K blocks); eight
  chains stay within 1.00 times at every measured token count (0.63 and 0.65 times there) at 1.002 to 1.007 times the
  time of four on a 212-SM part.  Measured on a 212-SM R200 part at the ends of every range, one token on each side and
  the token counts between, the 208-SM, 216-SM and larger plans forced there (the accuracy and the bits are the plan's;
  no time is claimed for those parts): every kind within the check, except in the two 212-SM ranges above that keep
  the former program.
- K1 weight gradient: the `sm_107a` CTA-pair rows of three or four splits (their partials added in the order they
  finish; on 212-SM parts they missed mirrored cancellation, 5.8 to 8.7 times the reference's error) run two splits
  where the two-split program was not slower than the deterministic plan on 212-SM parts (`4096 -> 32`, `5120 -> 32`,
  `5120 -> 48` FP32; it met all six kinds), else the deterministic program; on 212-SM parts they hold in the window of
  16172 to 16384 tokens only (two splits were timed there alone); there FP32 `4096 -> 48` and `4096 -> 64` run four
  CTAs of 64-wide tiles (`K1_DSM_RANGES_BY_SM`; its worst kind 0.55 to 0.58 times the reference's error, 0.84 to 0.87
  times the time of four splits in the arrival order, which these rows ran before), and FP32 `4096 -> 32` runs four
  splits in the mirror order (`K1_ORDERED_RANGES_BY_SM`; its worst kind at most 0.70 times the reference's error, about
  2 % slower than four splits in the arrival order, whose bits vary).  The ranges of the cluster tables that ran three CTAs run two CTAs (rows whose
  reference adds two halves: three missed mirrored cancellation by 5.8 to 8.7 times, two met all six kinds); on 152-SM
  `sm_103a` parts the `5120 -> x` ranges of three CTAs, and `5120 -> 32` at 16384 tokens, run the deterministic
  program (the reference cuts that reduction at 135 / 136 of 256 K blocks and a half split missed by 1.14 to 1.15
  times), and so does `5120 -> 48` BF16 at 16231 tokens (its two-split output missed the permuted mirrored
  cancellation by 1.104 times); the 148-SM `sm_100a` three-CTA ranges of `5120 -> 64` and `5120 -> 128` stay
  (one-chain reference; `5120 -> 128` met all six kinds).  On 148-SM and 152-SM `sm_100a` parts and 152-SM `sm_103a`
  parts `4096 -> 48` and `4096 -> 64` FP32 from 16172 to 16384 tokens run two CTAs of 32-wide tiles at an even number of K blocks (the
  reference's two halves, its error exactly) and, at an odd number, four CTAs of 64-wide tiles (0.60 to 0.64 times the
  reference's relative error, the same bits on every run): there the
  reference sums three shorter partials on most tiles, and every two-way split missed by 1.16 times (the parity table).
  212-SM `sm_107a` FP32 `4096 -> 1024` (the 192-wide row-major program) and `5120 -> 1024` (256-wide) from 16172 to
  16384 tokens run one split of four accumulation chains at an odd number of 64-token K blocks and of two at an even
  number, and `5120 -> 512` (128-wide) two chains at an even number (`K1_ORDERED_RANGES_BY_SM`): at 16384 tokens the
  one-chain program of `4096 -> 1024` missed all six kinds, up to 26.6 times; the three-chain program missed mirrored
  cancellation at 16172 tokens (2.06 times at `4096 -> 1024`, which also missed concentrated values, 1.78 times at
  `5120 -> 1024`) and at 16231 and 16384 tokens at `5120 -> 1024`, and the deterministic program of `5120 -> 512` at
  16231 and 16384 tokens; four chains (worst kind 0.79 and 0.60 times the reference's error, 1.005 to 1.010 times the
  three-chain program's time) and two chains (0.999 to 1.000 times the former route's time) met all six kinds.
  148-SM `sm_100a` BF16 `4096 -> 1024` from 16172 to 16384 tokens runs one split of two chains of 256-wide tiles
  (`K1_ORDERED_RANGES_BY_SM`; one wave of 128 CTAs, the bits of two splits of one chain): its former route, one split
  of one chain, missed mirrored cancellation at every token count of that window (1.87 to 1.95 times the reference's
  error, 1.38 to 1.39 times on the permuted inputs).  On a B200 part at 16172, 16231, 16256, 16300 and 16384 tokens
  two chains met all six kinds (at most 1.08 times the reference's error), repeated bit for bit and had the bits of
  two splits, at 1.003 to 1.005 times one chain's time, 0.897 to 0.906 times two splits' (a second wave of CTAs) and
  0.74 to 0.84 times the deterministic program's.  From 961 to 16171 tokens, where it ran the deterministic program,
  it runs the same two chains (at 10 token counts all six kinds within the check, at most 1.006 times the reference's
  error, and the same bits on every run; 0.42 to 0.84 times the deterministic program's time on B200 parts).
- K2 weight gradient: the head/tail windows (`"H0.5x3"`, `"H1x4p44"`, `"H1x5p44"`, `"H0.5x4"`; 3 to 23 % faster than
  the deterministic program) varied run to run and missed the rule (up to 1.94 times); those rows run the
  deterministic program.  Hidden 6144 at 16129 to 16384 tokens runs six splits instead of the deterministic package's
  three (three missed all six kinds, up to 2.0 times).  Hidden 4096 keeps six splits except at 16384 tokens on 148-SM
  `sm_100a` parts (16 splits, 0.96 to 0.97 times the time of six there); on 152-SM parts 16 splits were not faster
  than six through the package entry.  On 212-SM `sm_107a` parts hidden 6144 at 16172 to 16231 tokens runs the
  K-order program at six spans (its worst kind 1.09 times the reference's error, on mirrored cancellation; 0.93 times
  the time of the deterministic program at six splits; the same bits on every run).  The `"waves"` windows run their
  partition only where every element takes at most two partials (the same bits on every run), else the deterministic
  program.  The router window tables below keep the measurements of the
  head/tail windows as a record; the tables of the code (`ROUTER_RULES`, `ROUTER_RULES_BY_SM`) hold no head/tail
  window.
- The cluster tables below keep their measurements; their three-CTA rows now run as described above.

| K1 row `d_in -> d_out` | `sm_100a` BF16 | `sm_100a` FP32 | `sm_103a` BF16 | `sm_103a` FP32 |
|---|---|---|---|---|
| `4096 -> 32` | 512; 65536 and up† | -- | 512; 65536 and up§ | 512; 32768 and up§ |
| `4096 -> 48` | 512; 49153 and up† | 512 | 512; 32768 and up§ | 512; 32768 and up§ |
| `4096 -> 64` | 512; 65536 and up† | 512 | no verdict | no verdict |
| `5120 -> 32` | 512; 65537 and up† | 512 | 512; 45058 and up‡ | 512 |
| `5120 -> 48` | 512 to 2048; 49153 and up† | 512 to 2048 | 512 to 2048; 40961 and up‡ | 512 to 2048 |
| `5120 -> 64` | 512 to 2048; 8192 | 512 to 2048; 8192 | 512 to 2048; 45058 and up‡ | 512 to 2048 |
| `5120 -> 128` | no verdict | no verdict | 512 to 8192; 32768 and up¶ | 512 to 4096 |
| `6144 -> 32` | 1024 | -- | 512 | 512 |
| `6144 -> 128` | no verdict | no verdict | 512 to 8192 | 512 to 8192 |

*Table: the K1 token counts that run the deterministic program on `sm_100a` / `sm_103a` parts of 152 SMs or more
(`K1_SPLIT_RULE_EXCLUDED_TOKENS`; `--`: none; `no verdict`: the row runs the deterministic program at every token
count).  Every other token count keeps the row's verdict.  Measured at 512, 1024, 2048, 4096, 8192, 16231, 32768 and
65536 tokens on two 152-SM parts per row and token count (GB200: two parts; GB300: four parts, because the strict runs
of 8 `sm_103a` points, 3 of them in this table (`4096 -> 32` BF16, `5120 -> 32` FP32, `6144 -> 32` FP32 at 512 tokens),
ran on two other parts than their screen): a screen of 9 paired groups per row and token count in one long-lived process
per GPU, eager launches and CUDA graph replays, next to a kernel holding 8, 16 or 32 SMs in both launch orders (a launch
order without paired groups: the slowest split-program time against the fastest deterministic time), with the backward
pass and the checkpointed step timed; where a screen run failed, 15 paired groups in a long-lived process and in a fresh
process per row, on two parts per token count (one part for `sm_100a` `6144 -> 32` FP32 at 1024 tokens); and the FP64
error of 5 seeds on each of two parts per architecture (prepared launches, CUDA graph replays, the convenience call)
next to the deterministic program's on the same operands.  A measured token count runs the deterministic program where
the split program failed in at least one run (the K1 rule above; the backward pass or the checkpointed step more than
1.3 % slower than with the deterministic program; or the deterministic package's FP64 error budget missed for some seed)
and the deterministic program leaves the backward pass and the checkpointed step no more than 1.3 % slower (medians and
paired medians, eager launches and graph replays), or where the split program's output missed the deterministic
package's FP64 error budget for some seed (marked ‡, †).  The ratio of the split program's error to the deterministic
program's (q) is data, not part of the rule.  A token count between two consecutive such measured token counts runs the
deterministic program too (not measured).  Marked ‡: the BF16 output of `sm_103a` `5120 -> 32` and `5120 -> 64` met the
BF16 budget on every seed at 32768 tokens (30 seeds on two parts) and at every measured token count up to 45057 tokens
on a 152-SM GB300 part (10 seeds, 20 runs each); at the next measured token count, 47105, both programs missed it for
some seed, so the row keeps the deterministic program from the token count after the last passing one (the split program
misses it on every seed at 65536 tokens, 30 seeds on two parts).  `5120 -> 48` met it on that part on every seed up to
49152 tokens (the deterministic program missed it for one seed at 45057 and at 49152 tokens), but on a 148-SM B300 part,
which runs the same program, it missed it for one of 10 seeds at 40961 and at 40962 tokens while the deterministic
program met it (the 148-SM table below), so its entry starts at 40961 tokens; from 49153 tokens both programs miss it on
the 152-SM part.  Marked ¶: `sm_103a` `5120 -> 128` BF16 is not restored from 32768 tokens.  Its split program met the
BF16 budget up to 45057 tokens as the rows marked ‡ did, but a strict timing re-measurement on that part failed the K1
rule at 5 of 11 token counts from 32768 to 45057 (32768, 33793, 38913, 43009 and 44033: at most 4 of 8 paired groups
faster, up to 1.5 % slower at the operation) and kept it at 34817, 36865, 39937, 40960, 41985 and 45057, so no measured
window holds.  Marked §: the entries of `sm_103a` `4096 -> 32` and `4096 -> 48` reach above 65536 tokens (up to 65536
tokens they come from the rule above).  There neither program meets the BF16 budget on any seed (10 seeds per token
count from 65537 tokens up on a 148-SM part; the deterministic plan of these rows is the same on 148 and 152 SMs), and
the FP32 split program gives the deterministic program's bits at an even number of 64-token K blocks; at an odd number
it met the FP32 budget on every seed at 65537 tokens (q 1.021 to 1.024) and both programs miss it from 98305 tokens up.
It was not measured faster there, so these rows keep the deterministic program from their entry up.  Marked †: `sm_100a`
BF16 entries from 148 SMs up (also in the 148-SM table below; `4096 -> 64` from 152 SMs up, its 148-SM entry is below)
where the split program misses the BF16 budget for some seed (10 seeds per token count on a 148-SM B200 part; the
deterministic plan of these rows is the same on 148 and 152 SMs): at 49153 tokens 8 of 10 seeds met it for `4096 -> 48`
and 9 for `5120 -> 48`; at 65536 tokens 3 for `4096 -> 32`; from 65537 tokens up none.  The deterministic program misses
it there too (`4096 -> 48` 9 of 10 seeds and `5120 -> 48` 8 of 10 at 49153 tokens, `4096 -> 32` 3 of 10 at 65536 tokens
and none from 65537 tokens up), so these rows keep the deterministic program from their entry up.  `5120 -> 32` (10
seeds, 20 runs each, at 32769 to 65025 tokens, finer from 45569 to 49665 tokens, on a 148-SM B200 part and two 152-SM
GB200 parts): both programs met the budget on every seed up to 46593 tokens, and from 47105 tokens each missed it for
some seed at some token counts, neither more accurate at every one, so the row keeps its entry from 65537 tokens.  Token
counts below 512, above 65536 (other rows), between two measured token counts with different outcomes, and on parts with
fewer than 152 SMs were not measured and keep the verdict.  Measured token counts that failed the rule but keep the
verdict, because the deterministic program makes the backward pass or the checkpointed step more than 1.3 % slower
there: BF16 output at 1024 tokens on `sm_103a` `4096 -> 48`, `6144 -> 32`; BF16 output at 2048 tokens on `sm_100a`
`5120 -> 32`, `6144 -> 32`; `sm_103a` `4096 -> 48`; FP32 output at 512 tokens on `sm_100a` `4096 -> 32`, `6144 -> 32`;
FP32 output at 1024 tokens on `sm_100a` `6144 -> 32`; FP32 output at 8192 tokens on `sm_103a` `4096 -> 48`,
`5120 -> 128`.  At 65536 tokens neither program meets the BF16 error budget.*

| K1 row `d_in -> d_out` | `sm_100a` BF16 | `sm_100a` FP32 | `sm_103a` BF16 | `sm_103a` FP32 |
|---|---|---|---|---|
| `4096 -> 32` | 65536 and up† | -- | 65536 and up‡ | 65536 and up‡ |
| `4096 -> 48` | 512; 49153 and up† | 512 | no verdict | no verdict |
| `4096 -> 64` | 512; 65536 and up‡ | 512; 65536 and up‡ | no verdict | no verdict |
| `5120 -> 32` | 65537 and up† | -- | -- | -- |
| `5120 -> 48` | 512 to 2048; 49153 and up† | 512 | 512 to 2048; 40961 to 49151 | 512 to 2048 |
| `5120 -> 64` | 512 to 2048; 8192 | 512 to 1024; 8192 | 512 to 1024; 47105 to 49152 | 512 to 1024 |
| `5120 -> 128` | no verdict | no verdict | 512 to 4096; 32768 to 65536 | 512 to 4096; 32768 to 65536 |
| `6144 -> 32` | -- | -- | -- | -- |
| `6144 -> 128` | no verdict | no verdict | 512 to 8192 | 512 to 8192 |

*Table: the 148-SM K1 token table: the K1 token counts that run the deterministic program on `sm_100a` / `sm_103a` parts
of exactly 148 SMs (`K1_SPLIT_RULE_EXCLUDED_TOKENS_BY_SM`; †: the entries of the 152-SM table above that hold from 148
SMs up; `--`: none; `no verdict`: the row runs the deterministic program at every token count, `sm_103a` `4096 -> 48`
below its 152-SM floor).  Every other token count keeps the row's verdict, and parts of any other SM count keep their
routing.  Measured with the protocol and the rule of the 152-SM table above at the same token counts, on B300
(`sm_103a`) and B200 (`sm_100a`) parts of 148 SMs: a screen of every split-routed row and token count on one part (9
paired groups, one long-lived process per GPU), the points that failed there again on a second part (15 paired groups in
a long-lived process and in a fresh process per point; B300: a part of a second node, which also screened every point;
B200: another part of the same node), and the FP64 error of 5 seeds on two parts at 32768 and 65536 tokens and at the
first and the last token count of every entry and one token either side of it (98304 and 131072 tokens too for the
entries that reach 65536 tokens).  ‡: entries from the FP64 error and the time.  `sm_103a` `5120 -> 48` BF16 at 40961 to
49151 tokens and `5120 -> 64` BF16 at 47105 to 49152 tokens: from the FP64 error alone, measured with 10 seeds (20 runs
each) at 36865 to 61441 tokens: the split program missed the BF16 budget for one of 10 seeds at 40961 and at 40962
tokens (`5120 -> 64`: 47105) while the deterministic program met it on every seed (`5120 -> 48` met it at 41985 to 47105
tokens inside its range); from 49152 tokens (`5120 -> 64`: 49153) both programs miss it for some seed, so the split
program stays there.  Above 65536 tokens the split program of `4096 -> 32` (`sm_103a`) and `4096 -> 64` (`sm_100a`)
gives other bits than the deterministic program where the token count spans an odd number of 64-token blocks (measured
at 65537, 65538, 65600 and 65665 tokens; `4096 -> 64` BF16 on `sm_100a` also at 11 more odd counts from 65728 to 98240
tokens) and the same bits where it spans an even number (65601 to 131072 tokens, 12 counts measured).  There the BF16
output of both programs misses the BF16 budget for some seed, and the FP32 output of both misses the FP32 budget at an
odd count from 98305 tokens up (q 1.021 to 1.023 at the measured odd counts).  At the even counts the split program does
not make the backward pass or the checkpointed step 1 % faster (`sm_100a`: the backward pass up to 3.5 % slower).  So
these rows run the deterministic program from 65536 tokens up.  The BF16 output of `5120 -> 32` (23 token counts from
57336 to 131072 tokens) and `5120 -> 128` (65536 to 131072 tokens) on `sm_103a` keeps the split program on 148 SMs (q
median 1.000 and 1.015, largest 1.26 and 1.15); at 65537 tokens `5120 -> 128` missed the BF16 budget on every seed with
both programs (10 seeds), and `5120 -> 64` missed it for some seed with both at 47104 tokens (split 8 of 10 seeds,
deterministic 9).  Unmeasured token counts keep the verdict.  Measured token counts that failed the rule but keep the
verdict, because the deterministic program makes the backward pass or the checkpointed step more than 1.3 % slower there
on a part: `sm_100a`: BF16 output at 512 tokens on `4096 -> 32`, `5120 -> 32`, `6144 -> 32`, at 1024 on `5120 -> 32`, at
2048 on `6144 -> 32`, at 32768 on `5120 -> 48`, at 65536 on `4096 -> 32`, `4096 -> 48`, `5120 -> 48`, `5120 -> 64`,
`6144 -> 32`; FP32 output at 512 tokens on `4096 -> 32`, at 1024 and 2048 on `5120 -> 48`, at 2048 on `5120 -> 64`, at
65536 on `5120 -> 48`; `sm_103a`: BF16 output at 512 tokens on `4096 -> 32`, `5120 -> 32`, `6144 -> 32`, at 2048 on
`5120 -> 64`, at 8192 on `5120 -> 128`, at 65536 on `5120 -> 48`, `5120 -> 64`, `6144 -> 32`, `6144 -> 128`; FP32 output
at 512 tokens on `6144 -> 32`, at 2048 on `5120 -> 64`, at 8192 on `5120 -> 128`, at 32768 on `4096 -> 32`.*

| K1 row `d_in -> d_out` | parts | token counts with an odd number of 64-token K blocks that run the deterministic program |
|---|---|---|
| `sm_100a` `4096 -> 32` FP32 | 148 SMs and more | 65538 and up |
| `sm_100a` `4096 -> 48` FP32 | 148 SMs and more | 65538 and up |
| `sm_100a` `4096 -> 64` FP32 | 148 SMs and more | 65538 and up |
| `sm_100a` `5120 -> 32` FP32 | 148 SMs and more | 65538 and up |
| `sm_100a` `5120 -> 48` FP32 | 148 SMs and more | 65538 and up |
| `sm_103a` `4096 -> 32` FP32 | 148 SMs and more | 65538 and up |
| `sm_103a` `4096 -> 48` FP32 | 152 SMs and more | 14338 to 32767‡; 65538 and up |
| `sm_107a` `8192 -> 128` FP32 | 217 SMs and more | 1025 to 8192† |

*Table: the odd-block K1 table (`K1_SPLIT_ODD_BLOCK_EXCLUDED_TOKENS`): these rows run the deterministic program at a
token count of the third column that spans an odd number of 64-token K blocks (`ceil(T / 64)` odd); an even number, and
every other row, keeps the routing of the tables above.  In these ranges the deterministic program of the row cuts the
token reduction of its tiles, taken in order, into units of half the tile's K blocks, rounded up (at an odd block count
a unit runs on into the next tile; its automatic two-way split), and adds the partials in a fixed order.  At an even
block count the two-split program's output is the deterministic program's bit for bit.  At an odd count its FP32 output
met the deterministic package's FP32 budget on every seed at every measured odd token count from 1025 to 65537 tokens
(10 seeds each: the `4096` rows at 1025, 1088, 2049, 3000, 4097, 8193, 12289, 16173, 24577, 32769, 49153 and 65537
tokens, `sm_100a` on a 148-SM B200 part and `sm_103a` on a 148-SM B300 part, the deterministic plan of these rows being
the same on 148 and 152 SMs; `sm_100a` `5120 -> 32` and `5120 -> 48` at 12337 to 65537 tokens), so those token counts
run the two-split program; q was 1.03 to 1.15, 1.021 to 1.024 at 65537 tokens and 1.007 to 1.075 for the `5120` rows
(data).  From 98305 tokens up both programs miss the FP32 budget at an odd count; no odd token count from 65538 to 98304
was measured, and the rows run the deterministic program at an odd count from 65538 tokens up.  Marked ‡: a range from
the time, not the budget: on a 152-SM GB300 part, at odd block counts the two-split program of `sm_103a` `4096 -> 48`
left the checkpointed step 2.0 to 2.3 % slower in graph replays at 16173 tokens (two measurements), the backward pass
1.6 and 2.0 % slower in eager launches at 20481 and 28673 tokens, and failed the K1 rule in one of 8 groups at 18433
tokens, while it kept the rule and the 1.3 % bound at 1025, 4097, 6145, 8193, 10241, 12289 and 14337 tokens.  `sm_107a`
`8192 -> 128` FP32 runs its two-split program at odd block counts from 1025 to 8192 tokens on 212 and 216 SMs: there its
output met the FP32 budget on every seed (10 seeds on 212-SM R200 parts, q 1.04 to 1.14 at 1025, 1088, 2049 and 4097
tokens; 10 seeds of 20 runs each on a 216-SM R200 part at 1025, 1088, 2049, 4097, 6145, 8127 and 8128 tokens), and on
that 216-SM part it kept the K1 rule and the 1.3 % bound at those token counts.  Marked †: parts of more than 216 SMs,
which were not measured, keep the deterministic program there, the route they took before the 216-SM measurement.  The
BF16 outputs of these rows had the deterministic program's relative Frobenius error and met the error budget on every
seed where the deterministic program met it.*

| K1 row `d_in -> d_out` (FP32 out) | 152 SMs: token ranges, every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`) | measured token counts outside them |
|---|---|---|
| `sm_100a` `4096 -> 128`: (64, 2) | 1024 to 4096 | 512; 6144; 8192 |
| `sm_103a` `4096 -> 128`: (64, 2) | 1024 to 6144 | 512; 8192 |

*Table: the 152-SM two-split table (`K1_TWO_SPLIT_RANGES_BY_SM` on `sm_100a` / `sm_103a` parts of exactly 152 SMs, above
the SM-count floor): the FP32 row `4096 -> 128`, which has no verdict in `K1_SPLIT_RULES`, runs two splits of 64-wide
tiles, each adding half of an output's K blocks, at every token count of its range, at either number of 64-token K
blocks.  Outside the range, with BF16 output and on parts of any other SM count (148 SMs included) it runs the
deterministic program.  At an even block count its output was the deterministic program's bit for bit (5 seeds at 1024,
2048, 4096, 6144 and 8192 tokens); at 1088 and 2112 tokens (17 and 33 blocks) it was not, with q 1.107 and 1.142 (5
seeds on three parts; data: the deterministic program's FP64 relative Frobenius error there is about 1 % of the FP32
budget).  Measured at 512, 1024, 2048, 4096 and 8192 tokens on two 152-SM parts of each GPU (two GB200 nodes, two GB300
nodes) and at 6144 tokens on one GB200 part and two GB300 parts: 15 paired groups of 20 launches in four runs per part
(two launch orders, a long-lived process and a fresh process per row), eager launches and CUDA graph replays, next to a
kernel holding 8, 16 or 32 SMs in both launch orders, with the backward pass and the checkpointed step timed.  A range
runs between measured token counts at which the two-split program passed the K1 rule in every run on every part, left
the backward pass and the checkpointed step no more than 1.3 % slower than the deterministic program, and met the
deterministic package's FP64 error budget for every seed; the token counts between two consecutive measured ones inside
it, odd block counts included, take the verdict too (not measured).  From 1024 to 4096 tokens its time was 0.70 to 0.79
times the deterministic program's (graph replays, both GPUs; 8.65 against 12.14 us on a GB200 part at 1024 tokens).  At
512 tokens it is slower than the deterministic program, and at 8192 tokens slower next to a kernel holding 32 SMs; at
6144 tokens on the GB200 part its time next to a kernel holding SMs on another stream was 1.099 times the deterministic
program's.*

| K1 row `d_in -> d_out` (FP32 out) | 152 SMs: token ranges, every K-block count (`K1_ONE_SPLIT_RANGES_BY_SM`) | measured token counts: passed | measured token counts: failed |
|---|---|---|---|
| `sm_100a` `4096 -> 32`: (64, 1) | 256 to 2048 | 256; 384; 512; 768; 800; 1024; 1025; 1536; 1600; 2048 | 2049*; 2304; 2560; 2561*; 2816; 3072; 4096 |
| `sm_100a` `4096 -> 128`: (64, 1) | 256 to 3328 | 256; 384; 512; 768; 800; 1024; 1025; 1536; 1600; 2048; 2049; 3072; 3073; 3328 | 3584; 3585*; 3840; 4096 |
| `sm_103a` `4096 -> 32`: (64, 1) | 256 to 2048 | 256; 384; 512; 768; 800; 1024; 1025; 1536; 1600; 2048 | 2049*; 2304; 2560; 2561*; 2816; 3072; 4096 |
| `sm_103a` `4096 -> 128`: (64, 1) | 256 to 2816 | 256; 384; 512; 768; 800; 1024; 1025; 1536; 1600; 2048; 2049; 2304; 2560; 2561; 2816 | 3072; 4096 |

*Table: the 152-SM one-split table (`K1_ONE_SPLIT_RANGES_BY_SM` on `sm_100a` / `sm_103a` parts of exactly 152 SMs):
the FP32 rows `4096 -> 32` and `4096 -> 128` run one split of 64-wide tiles (the transposed-store program
`dense_proj_gemm_atomic_nn_n64_f32_t`; 16 and 32 tiles, one CTA pair each), whose CTA pairs add a tile's whole token
reduction in one accumulator and store it once (no workspace, no ticket), at every token count of the range, at an
even and an odd number of 64-token K blocks alike.  It replaces the routing of the other tables there: two splits of
64-wide tiles, the deterministic program at the excluded token counts (512 on `sm_103a` `4096 -> 32`), and for
`4096 -> 128` the deterministic program outside its two-split range.  Below 256
tokens (not measured), above a range and on parts of any other SM count (148 SMs included) these tables keep the
routing.  Measured against that routing on one GB200 node and one GB300 node, two jobs each (four parts of the node
per job: the four launch layouts and row orders), with the protocol of the K1 rule: in every run 15 paired groups of
20 launches per cell, eager launches and CUDA graph replays, a long-lived process for every token count and a fresh
process per token count, in both row orders, cold L2, the GPU span of the operation from CUPTI; faster in every cell
(a one-sided paired sign test, p < 0.05, with a median time ratio below 0.99), no more than 3 % slower next to a kernel
holding 8, 16 or 32 SMs in the same and in the opposite launch order, and the backward pass and the checkpointed step
(two forwards and the backward pass) no more than 1.3 % slower (ratio of the medians and median of the paired ratios,
eager launches and graph replays).  A range runs from its first measured token count to the last one at which the
program passed, holds no measured token count that failed, and its token counts between two consecutive measured ones
take it too (not measured).  \*: an odd block count at which the one-split program passed against the deterministic
program but failed against the two-split program, which the routing runs there, so the ranges end
below.  Inside the ranges its time (graph replays) was 0.81 to 0.97 times the routing's where that runs two splits
(7.02 against 7.87 us for `4096 -> 32` and 7.17 against 8.63 us for `4096 -> 128` on a GB200 part at 1024 tokens) and
0.81 to 0.87 times where it runs the deterministic program, and the checkpointed step's time 0.96 to 0.99 times the
routing's.  Its output met the deterministic package's FP32 budget against FP64 (relative Frobenius error at most
5e-5, max error at most 1e-3 times the largest reference element) in every one of 20 runs of 10 seeds at every
measured token count of the ranges (worst relative Frobenius error 3.54e-6, worst max error 4.18e-6 times the largest
reference element).  At 256 to 800 tokens that error was the deterministic program's; from 1024 tokens up, where the
deterministic program adds the halves of an output's token reduction in two partials, it was 2.10 to 2.38 times the
deterministic program's (reported, not a rule).*

| K1 row `d_in -> d_out`: (`block_n`, `splits`) | tokens | measured token counts: time (us); routing / cluster; deterministic / cluster |
|---|---|---|
| `4096 -> 32`: (32, 3) | 1024 to 2048 | 1024: 5.66; 1.260; 2.079.  2048: 7.26; 1.263; 1.830 |
| `4096 -> 32`: (32, 2) | 6144 to 32768 | 6144: 12.20; 1.175; 1.567.  8192: 14.59; 1.140; 1.230.  12288: 19.70; 1.097; 1.166.  16384: 24.38; 1.073; 1.132.  24576: 34.46; 1.040; 1.093.  32768: 44.50; 1.032; 1.071 |
| `4096 -> 48`: (64, 3) | 1024 to 2048 | 1024: 6.58; 1.319; 1.804.  2048: 7.76; 1.312; 1.720 |
| `4096 -> 48`: (64, 3) | 4096 to 8192 | 4096: 10.19; 1.237; 1.568.  6144: 12.74; 1.176; 1.505.  8192: 15.22; 1.136; 1.185 |
| `4096 -> 48`: (64, 3) | 16384 to 32768 | 16384: 25.06; 1.074; 1.120.  24576: 34.86; 1.052; 1.084.  32768: 45.02; 1.042; 1.067 |
| `4096 -> 64`: (64, 3) | 1024 to 32768 | 1024: 6.09; 1.399; 1.898.  2048: 7.68; 1.311; 1.710.  3072: 9.20; 1.223; 1.557.  4096: 10.22; 1.234; 1.558.  6144: 12.91; 1.159; 1.473.  8192: 15.02; 1.152; 1.189.  12288: 20.31; 1.097; 1.133.  16384: 25.36; 1.060; 1.087.  24576: 35.01; 1.048; 1.081.  32768: 45.27; 1.038; 1.075 |
| `4096 -> 128`: (64, 2) | 4096 to 6144 | 4096: 10.89; 1.194; 1.528.  6144: 13.55; 1.462; 1.466 |
| `4096 -> 128`: (128, 3) | 6145 to 8192 | 8192: 17.66; 1.044; 1.044 |
| `5120 -> 32`: (32, 2) | 1024 to 32768 | 1024: 6.10; 1.385; 1.446.  2048: 7.80; 1.296; 1.351.  3072: 9.42; 1.239; 1.345.  4096: 10.93; 1.206; 1.387.  6144: 13.88; 1.163; 1.478.  8192: 17.22; 1.128; 1.279.  12288: 23.00; 1.096; 1.281.  16384: 29.13; 1.072; 1.123.  24576: 41.17; 1.047; 1.092.  32768: 52.92; 1.035; 1.076 |
| `5120 -> 48`: (64, 2) | 1024 to 24576 | 1024: 7.14; 1.258; 1.262.  2048: 8.43; 1.261; 1.262.  3072: 9.96; 1.231; 1.288.  4096: 11.42; 1.200; 1.344.  6144: 14.48; 1.153; 1.430.  8192: 17.78; 1.122; 1.257.  12288: 23.98; 1.073; 1.252.  16384: 30.00; 1.064; 1.102.  24576: 42.53; 1.029; 1.068 |
| `5120 -> 64`: (64, 2) | 1024 to 2048 | 1024: 6.54; 1.283; 1.284.  2048: 8.23; 1.275; 1.277 |
| `5120 -> 64`: (64, 2) | 4096 to 24576 | 4096: 11.37; 1.205; 1.334.  6144: 14.38; 1.155; 1.425.  8192: 17.62; 1.131; 1.132.  12288: 23.67; 1.088; 1.130.  16384: 29.79; 1.064; 1.132.  24576: 42.30; 1.035; 1.127 |
| `5120 -> 128`: (128, 2) | 1024 to 6144 | 1024: 7.54; 1.129; 1.135.  2048: 9.54; 1.118; 1.117.  3072: 11.36; 1.132; 1.132.  4096: 13.08; 1.176; 1.176.  6144: 16.74; 1.239; 1.238 |

*Table: the 152-SM cluster table (`K1_DSM_RANGES_BY_SM` on `sm_100a` parts of exactly 152 SMs): the FP32 rows `4096`
/ `5120 -> 32, 48, 64, 128` run the cluster program `dense_proj_gemm_dsm_nn_n<block_n>_f32_t_s<splits>` (tiles of 128
`d_in` columns by `block_n` `d_out` rows, one cluster of `splits` CTAs per tile, CTA `s` taking the `s`-th of
`splits` contiguous runs of the tile's 64-token K blocks; each CTA sends the 4-column quads of its FP32 partial that
another rank owns to that rank in distributed shared memory, adds the `splits` partials of its own quads in
cluster-rank order and stores them once; no CTA leaves before every CTA of the cluster received its quads) at every
token count of a range, before every other K1 table.  It replaces the routing of the other tables there (one or two
splits of 64-wide tiles of the CTA-pair program, or the deterministic program).  Of the tile widths and cluster sizes
screened on two GB200 parts (32, 64 and 128 wide, clusters of 2 to 4 CTAs) a screened token count takes the fastest
that passed the screen, and a token count between two consecutive screened ones takes the lower one's program where
that program passed at both, else the upper one's, which passed at both.  With this table, the rows were measured on
two GB200 parts (152 SMs) at 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576 and 32768 tokens against that
routing (the plan this package gives the part without the table) and against the deterministic program: CUPTI span,
cold L2, one process per GPU pinned to its own cores, both launch orders, two runs per launch order on each part (all
rows in one process, and one process per row); time = the median over the eight runs of each run's median, operation
alone in CUDA graph replays.  A measured token count is in a range when, on both parts, the operation alone was
faster than the routing in every run, eagerly and in CUDA graphs (paired median below 0.99 and a sign test), no more
than 3 % slower than the routing next to a kernel holding 8, 16 or 32 SMs on another stream (and without it, after
the same one-block gate), and the training step (forward, input gradient and weight gradient) and the step with the
forward recomputed, eagerly and in CUDA graphs, were no slower than with the routing (median of the runs at most
1.013 times the routing's, and no run slower than the slowest run of the routing against itself).  A row's ranges run
from a passing measured token count to a passing one and hold no failing one.  Not in a range (the routing of the
other tables holds): below 1024 and above 32768 tokens (not measured); `4096 -> 32` from 2049 to 6143 tokens and
`4096 -> 48` from 2049 to 4095 and from 8193 to 16383 (at 3072 and 4096, and at 3072 and 12288 tokens, an eager step
or step with the forward recomputed had a run slower than the routing's slowest run against itself); `4096 -> 128`
below 4096 tokens and above 8192 (1024, 16384 and 24576: no program passed the screen; 2048, 3072 and 12288: the
operation alone was not faster in every run, or a step had such a run; 32768: 1.51 to 1.55 times the routing next to
a kernel holding SMs); `5120 -> 64` from 2049 to 4095 and above 24576 tokens (3072: a step had such a run; 32768: the
operation alone was not faster in every run); `5120 -> 48` above 24576 and `5120 -> 128` above 6144 tokens (screened:
not faster); and every other part, SM count included.  Its output met the deterministic package's FP32 budget against
FP64 (relative Frobenius error at most 5e-5, max error at most 1e-3 times the largest reference element) in every one
of 20 runs of 10 seeds at every measured token count on both parts (in the ranges: worst relative Frobenius error
1.86e-5, at 32768 tokens; worst max error 2.06e-5 times the largest reference element; the error grows about linearly
with the token count).  The cluster program sums a tile's partials in a fixed order, so it repeats bit for bit (3000
launches per measured configuration at 128 to 1024 tokens, data).*

| K1 row `d_in -> d_out`: (`block_n`, `splits`) | tokens | measured token counts: time (us); routing / cluster; deterministic / cluster |
|---|---|---|
| `4096 -> 32`: (32, 3) | 1024 to 2047 | 1024: 5.61; 1.248; 2.080 |
| `4096 -> 32`: (32, 2) | 2048 to 4095 | 2048: 7.21; 1.262; 1.821 |
| `4096 -> 32`: (32, 3) | 4096 to 8191 | 4096: 9.77; 1.220; 1.611.  6144: 12.29; 1.222; 1.589 |
| `4096 -> 32`: (32, 2) | 8192 to 12287 | 8192: 14.94; 1.125; 1.193 |
| `4096 -> 32`: (32, 3) | 12288 to 16383 | 12288: 19.91; 1.144; 1.185 |
| `4096 -> 32`: (32, 2) | 16384 to 24575 | 16384: 25.26; 1.065; 1.103 |
| `4096 -> 32`: (32, 3) | 24576 to 32768 | 24576: 34.78; 1.078; 1.095.  32768: 45.03; 1.063; 1.064 |
| `4096 -> 48`: (32, 2) | 1024 to 4095 | 1024: 6.28; 1.371; 1.861.  2048: 7.70; 1.312; 1.711 |
| `4096 -> 48`: (64, 3) | 4096 to 6144 | 4096: 10.17; 1.239; 1.563.  6144: 12.68; 1.244; 1.535 |
| `4096 -> 48`: (64, 3) | 12288 to 32768 | 12288: 20.43; 1.157; 1.167.  16384: 25.42; 1.091; 1.110.  24576: 35.30; 1.092; 1.093.  32768: 45.58; 1.067; 1.068 |
| `4096 -> 64`: (32, 2) | 1024 to 2047 | 1024: 5.74; 1.983; 1.994 |
| `4096 -> 64`: (64, 3) | 2048 to 8191 | 2048: 7.62; 1.694; 1.700.  4096: 10.22; 1.543; 1.545.  6144: 12.67; 1.537; 1.539 |
| `4096 -> 64`: (32, 2) | 8192 to 12287 | 8192: 15.12; 1.178; 1.180 |
| `4096 -> 64`: (64, 3) | 12288 to 32768 | 12288: 20.32; 1.176; 1.176.  16384: 25.48; 1.091; 1.089.  24576: 35.26; 1.087; 1.087.  32768: 45.56; 1.059; 1.061 |
| `4096 -> 128`: (64, 2) | 1024 to 12288 | 1024: 6.25; 1.134; 1.915.  2048: 7.93; 1.159; 1.701.  4096: 10.78; 1.189; 1.524.  6144: 14.10; 1.115; 1.430.  8192: 16.47; 1.114; 1.115.  12288: 23.50; 1.032; 1.035 |
| `5120 -> 32`: (32, 2) | 1024 to 2048 | 1024: 6.03; 1.408; 1.465.  2048: 7.78; 1.294; 1.358 |
| `5120 -> 32`: (32, 2) | 6144 to 8191 | 6144: 14.04; 1.154; 1.502 |
| `5120 -> 32`: (32, 3) | 8192 to 12287 | 8192: 17.61; 1.094; 1.266 |
| `5120 -> 32`: (32, 2) | 12288 to 32768 | 12288: 23.26; 1.085; 1.292.  16384: 29.38; 1.064; 1.167.  24576: 42.01; 1.037; 1.118.  32768: 54.38; 1.021; 1.109 |
| `5120 -> 48`: (64, 3) | 1024 to 2047 | 1024: 7.07; 1.262; 1.268 |
| `5120 -> 48`: (64, 2) | 2048 to 4095 | 2048: 8.32; 1.276; 1.276 |
| `5120 -> 48`: (64, 3) | 4096 to 8192 | 4096: 11.62; 1.183; 1.336.  6144: 14.63; 1.147; 1.445.  8192: 17.88; 1.106; 1.257 |
| `5120 -> 64`: (64, 2) | 1024 to 4095 | 1024: 6.50; 1.298; 1.300.  2048: 8.23; 1.288; 1.292 |
| `5120 -> 64`: (64, 3) | 4096 to 6143 | 4096: 11.44; 1.194; 1.357 |
| `5120 -> 64`: (64, 2) | 6144 to 12287 | 6144: 14.65; 1.135; 1.444.  8192: 17.99; 1.096; 1.242 |
| `5120 -> 64`: (64, 3) | 12288 to 24576 | 12288: 24.18; 1.064; 1.248.  16384: 30.56; 1.038; 1.135.  24576: 42.64; 1.037; 1.116 |
| `5120 -> 128`: (128, 2) | 1024 to 4095 | 1024: 7.44; 1.146; 1.146.  2048: 9.47; 1.129; 1.133 |
| `5120 -> 128`: (128, 3) | 4096 to 32768 | 4096: 12.77; 1.221; 1.222.  6144: 16.10; 1.173; 1.322.  8192: 19.39; 1.143; 1.165.  12288: 26.17; 1.090; 1.170.  16384: 32.81; 1.059; 1.094.  24576: 46.09; 1.048; 1.061.  32768: 59.98; 1.027; 1.050 |

*Table: the 152-SM `sm_103a` cluster table (`K1_DSM_RANGES_BY_SM` on `sm_103a` parts of exactly 152 SMs): the rows and
programs of the 152-SM cluster table above, measured on two GB300 parts (152 SMs).  Of the six programs (32, 64 and 128
wide, clusters of 2 and 3 CTAs) a token count of a first screen of both parts (1024, 2048, 4096, 6144, 8192, 12288,
16384, 24576 and 32768 tokens) takes the fastest, where it was faster than the routing on both parts, and where two
consecutive measured token counts take different programs the lower one's runs up to the token count before the upper
one.  With this table the rows were measured at those token counts against the routing and the deterministic program
with the protocol of the 152-SM cluster table (operation alone, training step and step with the forward recomputed,
eagerly and in CUDA graphs; time = the median over the eight runs of each run's median, operation alone in CUDA graph
replays), except that the time next to a kernel holding 8, 16 or 32 SMs was recorded, not required: there the cluster
program lost more than the routing at some token counts (worst 1.55 times the routing's time, `5120 -> 128` at 32768
tokens).  Not in a range (the routing of the other tables holds): below 1024 and above 32768 tokens (not measured);
`4096 -> 128` at 16384, `5120 -> 48` at 12288, 16384 and 24576 and `5120 -> 64` at 32768 tokens (the operation alone was
not faster than the routing in every run); `4096 -> 48` at 8192 and `5120 -> 32` at 4096 tokens (a step or the step with
the forward recomputed had a run slower than the routing's slowest run against itself, or a median of its runs above
1.013 times the routing's); `4096 -> 128` at 24576 and 32768 and `5120 -> 48` at 32768 tokens (not faster than the
routing on both parts in the screen); and every other part, SM count included.  Its output met the deterministic
package's FP32 budget against FP64 in every one of 20 runs of 10 seeds at every measured token count on both parts
(27600 launches, 0 failures; worst relative Frobenius error 1.86e-5, worst max error 2.06e-5 times the largest reference
element), and it repeated bit for bit over 3000 back-to-back launches per configuration (data).*

| K1 row `d_in -> d_out`: (`block_n`, `splits`) | tokens | measured token counts: time (us); routing / cluster; deterministic / cluster |
|---|---|---|
| `4096 -> 32`: (32, 2) | 1024 to 4095 | 1024: 5.78; 1.407; 2.077.  2048: 7.34; 1.309; 1.830 |
| `4096 -> 32`: (32, 3) | 4096 to 6143 | 4096: 10.18; 1.215; 1.613 |
| `4096 -> 32`: (32, 2) | 6144 to 32768 | 6144: 12.70; 1.180; 1.563.  8192: 15.17; 1.134; 1.243.  12288: 20.22; 1.100; 1.181.  16384: 25.32; 1.071; 1.143.  24576: 35.34; 1.042; 1.106.  32768: 45.67; 1.027; 1.075 |
| `4096 -> 48`: (32, 2) | 1024 to 4095 | 1024: 6.62; 1.324; 1.837.  2048: 7.98; 1.287; 1.703 |
| `4096 -> 48`: (64, 3) | 4096 to 16384 | 4096: 10.61; 1.220; 1.555.  6144: 13.30; 1.173; 1.505.  8192: 16.22; 1.100; 1.154.  12288: 20.94; 1.093; 1.156.  16384: 25.99; 1.072; 1.128 |
| `4096 -> 48`: (64, 3) | 32768 | 32768: 46.70; 1.028; 1.067 |
| `4096 -> 64`: (32, 2) | 1024 to 4095 | 1024: 5.95; 1.446; 1.968.  2048: 7.65; 1.328; 1.742 |
| `4096 -> 64`: (64, 3) | 4096 to 6144 | 4096: 10.59; 1.221; 1.554.  6144: 13.45; 1.159; 1.466 |
| `4096 -> 64`: (64, 3) | 12288 to 32768 | 12288: 21.02; 1.091; 1.131.  16384: 26.06; 1.071; 1.098.  24576: 36.54; 1.035; 1.062.  32768: 47.00; 1.021; 1.044 |
| `4096 -> 128`: (64, 2) | 1024 to 12288 | 1024: 6.42; 1.922; 1.932.  2048: 8.18; 1.704; 1.710.  4096: 11.19; 1.530; 1.533.  6144: 14.33; 1.437; 1.438.  8192: 17.14; 1.118; 1.123.  12288: 23.06; 1.060; 1.062 |
| `5120 -> 32`: (32, 2) | 1024 to 4095 | 1024: 6.17; 1.392; 1.468.  2048: 7.99; 1.294; 1.347 |
| `5120 -> 32`: (32, 3) | 4096 to 6143 | 4096: 11.63; 1.190; 1.346 |
| `5120 -> 32`: (32, 2) | 6144 to 32768 | 6144: 14.45; 1.155; 1.460.  8192: 17.75; 1.128; 1.280.  12288: 23.97; 1.086; 1.277.  16384: 30.30; 1.068; 1.124.  24576: 43.13; 1.044; 1.077.  32768: 55.54; 1.034; 1.067 |
| `5120 -> 48`: (64, 3) | 1024 to 2047 | 1024: 7.21; 1.287; 1.260 |
| `5120 -> 48`: (64, 2) | 2048 to 4095 | 2048: 8.62; 1.273; 1.254 |
| `5120 -> 48`: (64, 3) | 4096 to 6143 | 4096: 11.98; 1.202; 1.315 |
| `5120 -> 48`: (64, 2) | 6144 to 32768 | 6144: 15.06; 1.149; 1.411.  8192: 18.19; 1.134; 1.267.  12288: 24.66; 1.080; 1.262.  16384: 31.12; 1.065; 1.104.  24576: 43.95; 1.048; 1.066.  32768: 57.06; 1.027; 1.049 |
| `5120 -> 64`: (64, 3) | 1024 to 2047 | 1024: 6.82; 1.262; 1.263 |
| `5120 -> 64`: (64, 2) | 2048 to 32768 | 2048: 8.51; 1.286; 1.269.  4096: 11.87; 1.210; 1.317.  6144: 14.94; 1.155; 1.411.  8192: 18.20; 1.127; 1.127.  12288: 24.55; 1.090; 1.116.  16384: 30.96; 1.070; 1.116.  24576: 43.67; 1.050; 1.125.  32768: 56.88; 1.029; 1.128 |
| `5120 -> 128`: (128, 2) | 1024 to 4095 | 1024: 7.69; 1.125; 1.128.  2048: 9.80; 1.110; 1.111 |
| `5120 -> 128`: (128, 3) | 4096 to 32768 | 4096: 13.35; 1.185; 1.186.  6144: 16.85; 1.260; 1.260.  8192: 19.88; 1.039; 1.041.  12288: 26.63; 1.043; 1.044.  16384: 33.10; 1.072; 1.073.  24576: 47.38; 1.085; 1.086.  32768: 62.34; 1.094; 1.094 |

*Table: the 148-SM cluster table (`K1_DSM_RANGES_BY_SM` on `sm_100a` parts of exactly 148 SMs): the rows and programs of
the 152-SM cluster table above, measured on two B200 parts (148 SMs).  Of the six programs (32, 64 and 128 wide,
clusters of 2 and 3 CTAs) a token count of a first screen of both parts (1024, 2048, 4096, 6144, 8192, 12288, 16384,
24576 and 32768 tokens) takes the fastest, where it was faster than the routing on both parts, and where two consecutive
measured token counts take different programs the lower one's runs up to the token count before the upper one.  With
this table the rows were measured at those token counts against the routing and the deterministic program with the
protocol of the 152-SM cluster table (operation alone, training step and step with the forward recomputed, eagerly and
in CUDA graphs; time = the median over the eight runs of each run's median, operation alone in CUDA graph replays),
except that the time next to a kernel holding 8, 16 or 32 SMs was recorded, not required: there the cluster program lost
more than the routing at some token counts (worst 1.54 times the routing's time, `5120 -> 128` at 8192 tokens).  Not in
a range (the routing of the other tables holds): below 1024 and above 32768 tokens (not measured); `4096 -> 48` at
24576, `4096 -> 64` at 8192 and `4096 -> 128` at 16384 tokens (a step or the step with the forward recomputed had a run
slower than the routing's slowest run against itself, or a median of its runs above 1.013 times the routing's); `4096 ->
128` at 24576 and 32768 tokens (not faster than the routing on both parts in the screen); and every other part, SM count
included.  Its output met the deterministic package's FP32 budget against FP64 in every one of 20 runs of 10 seeds at
every measured token count on both parts (28000 launches, 0 failures; worst relative Frobenius error 1.87e-5, worst max
error 2.54e-5 times the largest reference element), and it repeated bit for bit over 3000 back-to-back launches per
configuration (data).*

| part | K1 row, FP32 out | 16172 tokens (253 K blocks) | 16231 tokens (254 K blocks) | 16384 tokens (256 K blocks) |
|---|---|---|---|---|
| 148-SM `sm_100a` (B200) | `4096 -> 48` | (64, 4): 27.17 (x0.925) | (32, 2): 26.86 (64-wide: 27.42) | (32, 2): 27.50 (64-wide: 27.73) |
| 148-SM `sm_100a` (B200) | `4096 -> 64` | (64, 4): 26.78 (x0.912) | (32, 2): 26.58 (64-wide: 26.93) | (32, 2): 26.98 (64-wide: 27.30) |
| 152-SM `sm_100a` (GB200) | `4096 -> 48` | (64, 4): 26.34 (x0.909) | (32, 2): 24.91 (64-wide: 26.59) | (32, 2): 25.31 (64-wide: 26.99) |
| 152-SM `sm_100a` (GB200) | `4096 -> 64` | (64, 4): 25.57 (x0.891) | (32, 2): 24.72 (64-wide: 26.43) | (32, 2): 24.82 (64-wide: 26.46) |
| 152-SM `sm_103a` (GB300) | `4096 -> 48` | (64, 4): 26.05 (x0.927) | (32, 2): 25.79 (64-wide: 27.42) | (32, 2): 25.79 (64-wide: 27.52) |
| 152-SM `sm_103a` (GB300) | `4096 -> 64` | (64, 4): 25.09 (x0.907) | (32, 2): 25.49 (64-wide: 27.17) | (32, 2): 25.65 (64-wide: 27.34) |

*Table: the parity table (`K1_DSM_ODD_BLOCK_RANGES_BY_SM`, with the 32-wide two-CTA ranges of `K1_DSM_RANGES_BY_SM` from
16172 to 16384 tokens on 148-SM and 152-SM `sm_100a` parts and 152-SM `sm_103a` parts): the program each row runs and
its time (us); in parentheses the next program that met the accuracy check (64-wide: two CTAs of 64-wide tiles).  At
16172 tokens (an odd block count) both rows run four CTAs of 64-wide tiles, (64, 4); in parentheses its time over that
of four splits of the 64-wide CTA-pair program in the arrival order in the same paired groups (the program
`4096 -> 64` ran there before, whose bits vary run to run; the deterministic program, which `4096 -> 48` ran there
before, took 28.38 to 29.86 us).  Time: the GPU span (CUPTI) with a cold L2 in eager launches, the median over 9
rotated paired groups of 20 steps (the 16172-token column: 7 groups, another run), on one B200, one GB200 and one
GB300 part.  Accuracy: the six input kinds of
the check, 5 repeats each, the program's largest error over the reference's median.  At an even block count both two-CTA
programs sum the reference's two halves (1.00 times its error).  At an odd block count every two-way split (two CTAs of
32-wide or 64-wide tiles, two splits of the CTA-pair program) missed: its worst relative error was 1.155 to 1.159 times
the reference's at 16130, 16172, 16300 and 16320 tokens on the B200 part, and it missed at 16172 tokens on the GB200 and
the GB300 part.  Four CTAs of 64-wide tiles met all six kinds at 16172 and 16300 tokens on the three parts (0.60 to
0.64 times the reference's relative error, 0.60 to 1.00 times its largest element error), and so did four splits of
the CTA-pair program at every token count measured from 16130 to 16384 (0.60 to 0.69 and 0.60 to 0.82 times); the
deterministic program had the reference's error.*

| K1 row `d_in -> d_out` (FP32 out unless marked BF16) | 208 SMs | 212 SMs | 216 SMs |
|---|---|---|---|
| `4096 -> 32` | -- | -- | 16231 |
| `4096 -> 32` BF16 | -- | -- | 512¶ |
| `4096 -> 48` | 16231; 16384 | -- | 512¶; 16231; 16384 |
| `4096 -> 64` | 16231; 16384 | -- | 512¶; 16231; 16384 |
| `4096 -> 512` | 16231; 16384 | 16231; 16384 | 16231; 16384; 40961 and up§; 961 and up on more SMs† |
| `4096 -> 1024` | 16231; 16384‡ | 16231 | 16231; 16384‖; 40961 and up§; 961 and up on more SMs† |
| `5120 -> 32` | -- | -- | 512¶ |
| `5120 -> 32` BF16 | -- | -- | 512¶ |
| `5120 -> 48` | 16231; 16384 | 16231 | 512¶; 16231; 16384 |
| `5120 -> 64` | 16231 | 16231 | 512¶; 16231; 16384 |
| `5120 -> 512` | -- | 16231; 16384 | 40961 and up§; 961 and up on more SMs† |
| `5120 -> 1024` | -- | -- | 32769 and up‖; 961 and up on more SMs† |
| `5120 -> 1024` BF16 | -- | -- | 16231 |
| `6144 -> 32` BF16 | -- | -- | 512¶ |
| `8192 -> 128`, BF16 and FP32 | -- | -- | 512¶; 65536¶ |

*Table: the K1 token counts that run the deterministic program on `sm_107a` parts (`K1_SPLIT_RULE_EXCLUDED_TOKENS`;
`--`: none).  Unmarked ones are token counts of the deterministic package's `sm_107a` rules for single token counts
(16231 and 16384 tokens; `4096 -> 1024` at 16231 only), which give the row a faster deterministic plan there; every
other token count keeps the row's verdict.  On 212 SMs `4096 -> 4096` FP32 runs the deterministic program at every token
count (`K1_SPLIT_RULES_BY_SM`): the deterministic package's rule for that row holds at every token count.  On 208, 212
and 216 SMs the rows were measured at 16172, 16231 and 16384 tokens against those plans: 15 paired groups of 20 steps
per row and token count with the L2 cache flushed before every launch, in a long-lived process per GPU and in a fresh
process per row, in both arm orders, eager launches and CUDA graph replays, next to a kernel holding 8, 16 or 32 SMs in
both launch orders, with the backward pass and the checkpointed step timed (9 paired groups); 212 SMs: eight parts in
two nodes (`4096 -> 1024`: sixteen parts in five nodes), 208 SMs: two parts (`5120 -> 1024`: three), 216 SMs: two parts.
A measured token count runs the deterministic program where the row's program failed the K1 rule above in some run, or
left the backward pass or the checkpointed step more than 1.3 % slower than the deterministic program (ratio of the
medians over the runs, or median of the paired ratios, in eager launches or in graph replays).  `4096 -> 1024` on 212
SMs and `5120 -> 1024` on 208 SMs were measured twice; the rule takes the runs of both measurements together.  On 216
SMs the 192-wide row-major program of `4096 -> 1024` FP32 (the 216-SM override) was measured in the same runs against
the 256-wide tiles and the deterministic plan: faster than both at 16172 and 16384 tokens in every run, not faster than
the deterministic package's 16231-token plan, which that token count keeps (16384 tokens: marked ‖).  The 216-SM entry
of `5120 -> 1024` BF16 (16231 tokens) is not a token count of those rules: the deterministic plan there is the row's
plan at every token count, and the row's program left the backward pass 1.5 % slower in eager launches (ratio of the
medians over the runs).  Marked ‡: an entry of a later strict re-measurement of the restored window on a 208-SM R200
part: at 16384 tokens the deterministic program's plan (160-wide tiles) was faster; the row's program took 61.27 against
53.49 us in graph replays (8 of 8 groups failed the K1 rule) and left the checkpointed step 6.3 % slower in graph
replays.  Marked ‖: entries of a later strict re-measurement on a 216-SM R200 part, each failure reproduced in a second
run: at 16384 tokens the 216-SM override of `4096 -> 1024` failed the K1 rule in 4 and then 5 of its 8 cells (two launch
layouts in both orders, eager launches and graph replays; median time ratios 0.982 to 0.995 there, faster but not below
0.99; 52.90 against the deterministic program's 53.30 us in graph replays), and the one-split program of `5120 -> 1024`
left the checkpointed step 1.6 to 5.3 % slower at 36864 and 40960 tokens (eager launches or graph replays; the operation
took 0.87 to 0.92 times the deterministic program's time, and the output met the FP32 budget), so that row runs the
deterministic program from 32769 tokens up (the same point's A/A control varied the checkpointed step by up to 8.8 %,
but both failures reproduced); at 16385 to 32768 tokens it kept the rule and the bound.  Marked †: on parts of more than
216 SMs, none of which was measured, these four FP32 rows run the deterministic program from 961 tokens up, the route
they took before the 216-SM measurement (on 216 SMs their one-split and three-chain programs were timed at 961 to 40960
tokens).  Marked §: entries of their own, not from the rules above.  On parts of at least 216 SMs these FP32 rows run
the deterministic program from 40961 tokens up (`5120 -> 1024`: from 32769, marked ‖) because their one-split program's
FP64 relative Frobenius error exceeds the deterministic package's FP32 error budget (5e-5) on every seed there.  That
program adds the whole token reduction of an output in one chain, and its error grows about linearly with the token
count. Measured with the 216-SM plan on 212-SM R200 parts (the same program and accumulation order as on 216 SMs; the
error does not depend on the SM count) against an FP64 reference, each row on two parts, 5 seeds, at 32768, 36864,
40960, 45056, 49152, 57344, 65536 and 131072 tokens: within the budget on every seed up to 40960 tokens (1.86e-5 at
16384 tokens, 3.77e-5 at 32768), above it on every seed from 45056 tokens up (7.6e-5 at 65536 tokens and 1.5e-4 at
131072; the deterministic program: 3.8e-5 and 7.6e-5).  From 961 tokens (16 K blocks) up the deterministic program of
these rows cuts the token reduction into units of half an output's K blocks, rounded up (its automatic two-way split),
so below 40961 tokens q is 2.0 to 2.4 (data; up to 960 tokens both programs add one chain and give the same bits).  The
token counts between 40960 and 45056 run the deterministic program too (not measured), and the range has no upper bound:
the error keeps growing with the token count.  The deterministic program is slower there: at 65536 tokens on a 212-SM
part it takes 1.81 (`4096 -> 512`), 1.05 (`4096 -> 1024`), 1.72 (`5120 -> 512`) and 1.02 (`5120 -> 1024`) times the
one-split program's time (CUDA graph replays) and leaves the backward pass or the checkpointed step up to 51.7, 4.8,
41.5 and 3.9 % slower.  Marked ¶: single token counts from a re-measurement of chosen points at 512, 8192, 32768 and
65536 tokens on two 216-SM R200 parts per point (the protocol above: 15 paired groups, eager launches and graph replays,
next to a kernel holding 8, 16 or 32 SMs in both launch orders, the backward pass and the checkpointed step timed).  A
point is an entry where the row's program failed the K1 rule in some run and the deterministic program leaves the
backward pass and the checkpointed step no more than 1.3 % slower on both parts (the worst of the ratio of the medians
and the median of the paired ratios, eager launches and graph replays: the rule of the `sm_100a` / `sm_103a` entries);
the neighbouring token counts were not measured and keep the verdict.  Measured points that failed but keep the verdict,
because the deterministic program makes the backward pass or the checkpointed step more than 1.3 % slower there on a
part: BF16 output at 8192 tokens on `4096 -> 32`, at 32768 tokens on `4096 -> 32` and `5120 -> 1024`, at 65536 tokens on
`5120 -> 1024`; FP32 output at 512 tokens on `4096 -> 32` and `6144 -> 32`.*

| K1 row `d_in -> d_out` (FP32 out unless marked BF16) | 212 SMs: token ranges besides 16172 to 16384 | measured token counts outside them |
|---|---|---|
| `8192 -> 128`, BF16 and FP32 | 1024 to 16172; 16384 to 32768 | 512; 65536 |
| `4096 -> 512`, `4096 -> 1024`, `5120 -> 512`, `5120 -> 1024` | 512 | 1024 to 8192; 32768; 65536 |
| `5120 -> 1024` BF16: (256, 1) `p = 1` (`K1_SPLIT_RANGE_OVERRIDES_BY_SM`) | 512 to 12288 | 32768 |
| `4096 -> 512`, `5120 -> 512`: (128, 2), every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`) | 1024 to 8192; 32768 to 65536 | 512 |
| `4096 -> 1024`: (192, 2), every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`) | 2048 to 8192; 32768 to 65536 | 512; 1024 |
| `5120 -> 1024`: (256, 2), every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`) | 1024 to 8192 | 512; 32768; 65536 |
| `4096 -> 128`: (64, 2), every K-block count (`K1_TWO_SPLIT_RANGES_BY_SM`; no verdict in the 212-SM table) | 1024 to 7680 | 512; 640; 768; 896; 8000; 8192; 12288; 16384 |

*Table: the 212-SM K1 token-range table (`K1_SPLIT_RULES_BY_SM_TOKEN_RANGES`): further token counts at which these rows
of the 212-SM table of `K1_SPLIT_RULES_BY_SM` run their program (the third column runs the deterministic program),
except the token counts of the `sm_107a` K1 token table above.  Measured at 512, 1024, 2048, 4096, 8192, 32768 and 65536
tokens on two 212-SM R200 nodes of four parts each, with the protocol of the window above (one part per launch layout
and row order), and the FP64 error of 5 seeds on two parts.  A measured token count joins a range where the row's
program passed the K1 rule in every run, left the backward pass and the checkpointed step no more than 1.3 % slower than
the deterministic program, and met the deterministic package's FP64 error budget for every seed on both parts.  A range
ends at a measured token count that passed (or at the window) and holds no measured token count that failed; the token
counts between two consecutive measured ones inside it take the verdict too (not measured).  Token counts below 512,
above 65536 and between a measured token count that passed and one that failed run the deterministic program.  The four
one-split FP32 rows take this table at 512 tokens only (their 212-SM program in the window as well); from 960 tokens the
K1 accumulation-chain table below runs three chains first, and the two-split ranges hold more token counts (q of the
one-split program about 2 from 1024 tokens up, data: it adds an output's whole token reduction in one chain).  The FP32
rows `4096 -> 32, 48, 64` and `5120 -> 32, 48, 64` held three splits from 1024 to 65536 tokens; three partials added in
the order they finish vary from run to run and missed the accuracy check ("Accuracy and repeatability changes"), so they
take their two-split or deterministic verdict in the window only (two splits were timed there alone).  The other
rows are not faster than the deterministic program in every run at 512 tokens, nor `8192 -> 128` at 65536.  The
two-split rows (`K1_TWO_SPLIT_RANGES_BY_SM`) run two splits of the row's 212-SM tile width, each adding half of an
output's K blocks, at every token count of their ranges, at either number of 64-token K blocks.  So the token counts in
the third column of the one-split FP32 rows run the two-split program where a two-split range holds them.  At an even
block count from 1024 tokens up their output was the deterministic program's bit for bit (5 seeds on two parts at 1024,
2048, 4096, 8192, 16231, 16384, 32768 and 65536 tokens, on one part at 6000, 12000 and 48000); at an odd count from 17
blocks up it was not, with q 1.01 to 1.17 (data) and within the FP32 budget on every seed (5 seeds, 1025 to 65537
tokens; relative Frobenius error at most 3.8e-5, at 65537 tokens).  They were measured with the protocol above at 512,
1024, 2048, 4096, 8192, 32768 and 65536 tokens (`4096 -> 512` also at 40961, an odd block count) on two 212-SM R200
nodes of four parts each; a range runs between measured token counts at which the two-split program passed the K1 rule
in every run and left the backward pass and the checkpointed step no more than 1.3 % slower than the deterministic
program, and holds none that failed; the token counts between, odd block counts included, take the verdict too (not
measured).  At 512 tokens the two-split program is slower than the deterministic plan; `4096 -> 1024` is not faster in
every run at 1024 tokens; `5120 -> 1024` is not faster in every run at 32768 tokens and slower at 65536.  `4096 -> 128`
FP32 has no program in the 212-SM table either; its two-split range was measured with the same protocol at 512, 640,
768, 896, 1024, 2048, 4096, 6144, 6656, 7168, 7680, 8000, 8192, 12288 and 16384 tokens on two nodes of each of two R200
systems (6144, 8000, 8192, 12288 and 16384 tokens and 640 on one system).  From 1024 to 7680 tokens it passed in every
run (0.70 to 0.76 times the deterministic program's time) and its output was the deterministic program's bit for bit at
an even block count (5 seeds); at an odd count (1088 to 5000 tokens) q was 1.11 to 1.16 (data), so these token counts
run it too.  It failed the rule at 512, 640, 768 and 896 tokens; from 8000 tokens the deterministic plan changes and the
two-split error was 1.54 times its error, and at 12288 and 16384 tokens it was 5 to 7 % slower.  `5120 -> 1024` BF16 has
no program in the 212-SM table (at 16172 to 16384 tokens the deterministic plan is faster); its range runs the 256-wide
one-split program of `K1_SPLIT_RULES` (its 216-SM verdict), measured at 512, 1024, 2048, 4096, 8192, 12288 and 32768
tokens with the same protocol on a node of four parts, twice with the parts' launch layouts and orders rotated: faster
than the deterministic plan in every run up to 12288 tokens, with the FP64 error of the deterministic program on every
seed; at 32768 tokens it failed the K1 rule in every run (0.1 to 0.6 % faster than the deterministic plan alone), and
its checkpointed step measured 2.5 to 4.3 % slower, as did a second deterministic launch against the first there (1.9 to
4.4 %).  On 212 SMs `4096 -> 1024` FP32 keeps the 192-wide program at 512 tokens: the 256-wide program is 15 % faster
alone and leaves the backward pass 4 to 6 % faster, but the checkpointed step 0.2 to 1.2 % slower in graph replays
(median of the paired ratios, ratio of the medians).*

| K1 row `d_in -> d_out`, FP32 out | 212 SMs: `(block_n, chains)`, token ranges | 216 SMs: `(block_n, chains)`, token ranges |
|---|---|---|
| `4096 -> 512` | (128, 3): 960 to 16172; 32768 to 40961 | (128, 3): 961 to 16172; 32768 to 40961 |
| `4096 -> 1024` | (192, 3): 960 to 16172; 32768 to 40961 | (192, 3): 961 to 16172; 32768 to 40961 |
| `5120 -> 512` | (128, 3): 960 to 16172; 32768 to 40961 | (128, 3): 961 to 40961 |
| `5120 -> 1024` | (256, 3): 960 to 40961 | (256, 3): 961 to 16384; 40961 |

*Table: the K1 accumulation-chain table (`K1_CHAIN_RANGES_BY_SM`; from 961 tokens the ranges of
`K1_PARITY_RANGES_BY_SM` come first, so these rows run three chains only at 960 tokens on 212 SMs): token ranges at
which these rows run the one-split program of three accumulation chains (templates `dense_proj_gemm_atomic_nn_n128_f32_t_c3`,
`dense_proj_gemm_atomic_nn_n256_f32_t_c3`, `dense_proj_gemm_atomic_nn_n192_f32_c3`) on parts of exactly that SM count,
in place of the tables above, the 212-SM two-split ranges included: at the even block counts of those ranges both
programs were timed at (1024 to 8192 and 32768 to 40960 tokens) three chains ran the weight gradient 7 to 52 % and the
checkpointed step 2 to 21 % faster than two splits (medians, eager launches and graph replays).  The program cuts each
tile's K blocks into three contiguous chains, each summed from an `init` MMA in an accumulator of its own, and its
epilogue adds them once, in order: no workspace and the same bits on every run.  Its FP64 relative Frobenius error was
0.31 to 0.73 times the deterministic program's on every seed (q, data) (5 seeds on a 212-SM R200 part at 960, 961, 1024,
2048, 4096, 8192, 16172, 16231, 16384, 32768 and 40961 tokens, even and odd block counts), where the one-chain program's
is 2.0 to 2.3 times.  The ranges follow the rule and the construction of the 212-SM K1 token-range table (the K1 rule in
every run, the backward pass and the checkpointed step no more than 1.3 % slower than the deterministic program by the
ratio of the medians over the runs and the median of the paired ratios, eager launches and graph replays).  212 SMs:
measured in two jobs on four parts of one node each (one launch layout and row order per part), at 960, 4096, 16172,
16384, 32768 and 40961 tokens and at 960, 1024, 2048, 4096, 4097, 8192, 16172, 16384, 32768, 32769, 40960 and 40961
tokens; `4096 -> 1024` left the checkpointed step 1.7 % slower at 16384 tokens in one of them.  216 SMs: measured at
960, 961, 1024, 2048, 4096, 8192, 16172, 16231, 16384, 32768 and 40961 tokens on one part (two row orders); up to 960
tokens these rows keep their one-chain program, which gives the deterministic program's bits there.  Token counts
outside the ranges keep the routing of the tables above (216 SMs: the one-chain program up to 40960 tokens,
`5120 -> 1024` up to 32768, within the FP32 budget with q 2.0 to 2.4, and the 216-SM override of `4096 -> 1024` at 16172
to 16383 tokens).*

`"waves"` (`router_waves`) launches whole waves of the device's CTA pairs (`pairs = sm_count // 2`): `c * pairs`
units with `c = max(2, round(tiles * k_iters / (68 * pairs))) + 1`, and only where the deterministic plan's splits
fill their last wave below 80 % (`w / ceil(w)`, `w = splits * tiles / pairs`, `splits` the deterministic package's
count of the row, its `router_splits`: six at every token count of a `"waves"` window); otherwise the row runs the
deterministic program.  At hidden 5120 on 152 SMs that is 80 units at 64 tokens (at most one unit per (tile,
K block) step) and, with six splits, 380 units at 16k tokens; on 148 SMs six splits fill their last wave to 81 % and
the row runs the deterministic program.

`"simt_n<tile_n>m<tile_m>s<slices>"` (`router_simt`, the weight gradient only) runs the exact-FP32 program
`router_fp32_gemm_simt_n<tile_n>m<tile_m>s<slices>` on the CUDA cores: one CTA per `tile_n x tile_m` tile of the
transposed `out` over all tokens, `slices` slices of the CTA each taking every `slices`-th 16-token block of the FP32
operands (read in place through their row strides), one FFMA per product, the slices added in slice order, every
element of `out` stored once (no BF16 parts, no workspace, no zero fill; the bits are fixed by the token count and the
tile, not by the SM count).  It launches after its predecessor, never as a programmatic dependent launch: launched
early, its CTAs took only the SMs the input gradient left and the backward pass was slower.  Its result differs from
the deterministic program's in the last bits (FFMAs of the FP32 operands instead of products of their BF16 parts, in
another summation order).

| SIMT window (weight gradient, 256 experts) | token window | measured at (tokens) | part | weight gradient alone: time / deterministic plan | backward pass / checkpointed step: time / previous plan |
|---|---|---|---|---|---|
| `sm_100a` hidden 4096 `"simt_n128m64s2"` | 1 to 224 | 64, 128, 192, 224 | GB200, 152 SMs (five parts in three nodes) | x0.225-0.665 | x0.381-0.814 / x0.729-0.956 |
| `sm_100a` hidden 5120 `"simt_n64m64s2"` | 1 to 256 | 64, 128, 192, 224, 256 | GB200, 152 SMs (five parts in three nodes) | x0.283-0.882 | x0.470-0.943 / x0.746-0.993 |
| `sm_100a` hidden 6144 `"simt_n64m64s2"` | 1 to 256 | 64, 128, 192, 224, 256 | GB200, 152 SMs (five parts in three nodes) | x0.196-0.848 | x0.331-0.911 / x0.770-0.987 |
| `sm_103a` hidden 4096 `"simt_n64m64s2"` | 1 to 288 | 1, 17, 32, 64, 128, 192, 224, 256, 288 | GB300, 152 SMs (two parts in two nodes) | x0.197-0.808 | x0.398-0.887 / x0.704-0.974 |
| `sm_103a` hidden 5120 `"simt_n64m64s2"` | 1 to 192 | 1, 17, 32, 64, 128, 160, 192 | GB300, 152 SMs (two parts in two nodes) | x0.191-0.710 | x0.650-0.866 / x0.807-0.968 |
| `sm_103a` hidden 6144 `"simt_n64m64s2"` | 1 to 256 | 1, 17, 32, 64, 128, 192, 224, 256 | GB300, 152 SMs (two parts in two nodes) | x0.180-0.798 | x0.352-0.880 / x0.782-0.967 |
| `sm_107a` (`ROUTER_RULES_BY_SM`, 212 SMs) hidden 4096 `"simt_n128m64s2"` | 1 to 256 | 64, 128, 192, 256 | R200, 212 SMs (three parts in two nodes) | x0.308-0.797 | x0.609-0.910 / x0.872-0.989 |
| `sm_107a` (`ROUTER_RULES_BY_SM`, 212 SMs) hidden 5120 `"simt_n128m64s2"` | 1 to 256 | 64, 128, 192, 256 | R200, 212 SMs (three parts in two nodes) | x0.278-0.738 | x0.554-0.832 / x0.853-0.971 |
| `sm_107a` (`ROUTER_RULES_BY_SM`, 212 SMs) hidden 6144 `"simt_n128m64s2"` | 1 to 224 | 64, 128, 192, 224 | R200, 212 SMs (three parts in two nodes; 192, 224 and 256 again on two parts) | x0.270-0.623 | x0.542-0.774 / x0.855-0.957 |

*Table: the SIMT windows (eager launches and CUDA graph replays, one long-lived process and a fresh process per token
count; the backward pass is the input gradient and the weight gradient, the checkpointed step two forward passes and the
backward pass, each measured against the plan of the same token count before the SIMT windows, which at hidden 5120 on
152 SMs is `"waves"`, up to 128 tokens on `sm_100a` and up to 192 on `sm_103a`).  The arms of each row were measured
interleaved, in paired groups.  The 128 x 64 tile on 152 SMs was slower than the previous plan in the backward pass and
the checkpointed step at hidden 5120 and 6144 and 256 tokens, so those rows take the 64 x 64 tile.  At 256 tokens the
128 x 64 tile was 6 to 12 % slower than the deterministic plan in CUDA graphs next to a resident kernel holding 32 SMs
at hidden 4096 on 152 SMs (five parts; 128 CTAs on the 120 SMs left) and up to 12 % slower at hidden 6144 on 212 SMs
(three parts; 192 CTAs on the 180 SMs left; up to 9.4 % on the two parts of a second measurement), so both windows end
at 224 tokens.  224 tokens was measured on two of the GB200 parts and on the two R200 parts of that second measurement.
On `sm_103a` the 64 x 64 tile was up to 3.4 % slower than the deterministic plan in CUDA graphs next to a resident
kernel at 320 tokens at hidden 4096 (one of the two parts) and up to 6.8 % slower at 288 tokens at hidden 6144 (both
parts), so those windows end at 288 and 256 tokens.*

On an input-gradient row (`kn`) `"simt_n<tile_n>m<tile_m>s<slices>"` runs the exact-FP32 program
`router_fp32_dgrad_simt_m<tile_m>n<tile_n>s<slices>`: one CTA per `tile_m x tile_n` tile of `out = G @ W` (`tile_m`
tokens by `tile_n` hidden columns, the last row tile partial) over all experts, `slices` slices of the CTA each taking
every `slices`-th 16-expert block of `G` and `W` (read in place through their row strides), one FFMA per product, the
slices added in slice order, every element of `out` stored once.  Like the weight gradient's program it launches after
its predecessor.  On 152-SM `sm_100a` and `sm_103a` parts the 32 x 64 tile in four slices (`"simt_n64m32s4"`, 128
threads) runs from 1 to 128 tokens at hidden 4096 and 5120 and from 1 to 64 at hidden 6144; on 212-SM `sm_107a` parts
(`ROUTER_RULES_BY_SM`) the 32 x 64 tile in eight slices (`"simt_n64m32s8"`, 256 threads) runs from 1 to 64 tokens and
the 64 x 64 tile in four slices (`"simt_n64m64s4"`, 256 threads) from 65 to 128 at hidden 4096 and 5120, and
`"simt_n64m32s8"` from 1 to 64 at hidden 6144.  At more tokens (from 192; from 128 at hidden 6144) the program was more
than 3 % slower than the deterministic program next to a resident kernel (on `sm_100a`; the `sm_103a` windows were
measured from 1 token to their last token count on two GB300 parts in two nodes).  The other `sm_107a` SM counts run the
deterministic program (not measured).  148-SM `sm_100a` and `sm_103a` parts take the windows of the 148-SM SIMT table
below.

On a forward row (`kk`) `"simt_c8_<kind>_l<lanes>_e<experts>t<tokens>"` (`router_forward_simt`) runs the exact-FP32
program `router_fp32_fwd_simt_c8_<kind>_l<lanes>_e<experts>t<tokens>`: one cluster of eight CTAs (256 threads each) per
`experts x tokens` tile of `out = X @ W.T`, cluster rank `r` taking the `r`-th eighth of `K`, which it stages in
128-column chunks of `X` and `W` (read in place through their row strides) through a two-slot ring of 16-byte copies.
The `lanes` threads of a lane group share a set of outputs and split the columns; they combine their chains by a warp
`xor` butterfly in ascending lane offsets.  Every CTA then publishes its partials in its shared memory, and rank `r`
adds the eight ranks' partials of its eighth of the tile's experts in rank order (distributed shared memory) and stores
each element of `out` once.  With `kind` `ffma` a chain is one FFMA per product; with `kg4` it keeps a sum and its
compensation (a Kahan add per four FFMAs), and the lanes and the ranks merge by TwoSum.  The bits depend on `K`, `kind`
and `lanes` only, not on the token count, the tile, the SM count or the architecture (an exact model of the order
reproduced the output bit for bit on every measured part).  The launch is `(8 N / experts, ceil(T / tokens))` CTAs and
needs `K` a multiple of 1024, `N` a multiple of `experts` and 16-byte rows with strides a multiple of 4 elements; any
other launch, and a caller's split count, runs the deterministic program.  It is a programmatic dependent launch with
the call's PDL flag: it triggers its dependents at entry and waits for its predecessor before its first load.  This
checkout registers four forward programs: `kg4_l32_e8t4` and `ffma_l16_e16t32` on `sm_100a`, `sm_103a` and `sm_107a`,
`ffma_l16_e16t64` on `sm_100a` and `sm_103a`, and `kg4_l32_e8t16` on `sm_100a`.

| row (256 experts; 152 SMs unless marked) | token window: spec | measured at (tokens) | forward alone, worst run: time / plan without the window, graphs; eager | worst next to a resident kernel: same launch order / router launched first | step / checkpointed step: median of the run medians, time / plan without the window |
|---|---|---|---|---|---|
| `sm_100a` (GB200) hidden 4096 | 1 to 8: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8 | x0.250-0.303; x0.238-0.284 | x0.331 / x0.317 | x0.676 / x0.540 |
| `sm_100a` (GB200) hidden 4096 | 9 to 16: `"simt_c8_kg4_l32_e8t16"` | 9, 12, 16 | x0.396-0.403; x0.376-0.380 | x0.494 / x0.476 | x0.686 / x0.545 |
| `sm_100a` (GB200) hidden 4096 | 17 to 96: `"simt_c8_ffma_l16_e16t32"` | 17, 32, 64, 96 | x0.426-0.556; x0.410-0.529 | x0.668 / x0.571 | x0.711 / x0.688 |
| `sm_100a` (GB200) hidden 4096 | 97 to 128: `"simt_c8_ffma_l16_e16t64"` | 97, 112, 128 | x0.612-0.631; x0.584-0.589 | x0.926 / x0.884 | x0.758 / x0.710 |
| `sm_100a` (GB200) hidden 4096 | 129 to 160: `"simt_c8_ffma_l16_e16t32"` | 129, 160 | x0.784-0.788; x0.748-0.749 | x0.908 / x0.878 | x0.878 / x0.846 |
| `sm_100a` (GB200) hidden 5120 | 1 to 8: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8 | x0.232-0.275; x0.219-0.262 | x0.298 / x0.282 | x0.681 / x0.549 |
| `sm_100a` (GB200) hidden 5120 | 9 to 16: `"simt_c8_kg4_l32_e8t16"` | 9, 12, 16 | x0.363-0.367; x0.345-0.350 | x0.458 / x0.381 | x0.681 / x0.551 |
| `sm_100a` (GB200) hidden 5120 | 17 to 96: `"simt_c8_ffma_l16_e16t32"` | 17, 32, 64, 96 | x0.408-0.535; x0.389-0.517 | x0.649 / x0.625 | x0.700 / x0.681 |
| `sm_100a` (GB200) hidden 5120 | 97 to 128: `"simt_c8_ffma_l16_e16t64"` | 97, 112, 128 | x0.588-0.600; x0.563-0.572 | x0.875 / x0.712 | x0.761 / x0.698 |
| `sm_100a` (GB200) hidden 5120 | 129 to 192: `"simt_c8_ffma_l16_e16t32"` | 129, 160, 176, 192 | x0.769-0.894; x0.741-0.855 | x1.019 / x0.991 | x0.948 / x0.926 |
| `sm_100a` (GB200) hidden 6144 | 1 to 8: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8 | x0.211-0.256; x0.204-0.243 | x0.282 / x0.266 | x0.683 / x0.544 |
| `sm_100a` (GB200) hidden 6144 | 9 to 16: `"simt_c8_kg4_l32_e8t16"` | 9, 12, 16 | x0.345-0.348; x0.327-0.333 | x0.438 / x0.421 | x0.702 / x0.555 |
| `sm_100a` (GB200) hidden 6144 | 17 to 96: `"simt_c8_ffma_l16_e16t32"` | 17, 32, 64, 96 | x0.395-0.519; x0.379-0.504 | x0.635 / x0.619 | x0.767 / x0.672 |
| `sm_100a` (GB200) hidden 6144 | 97 to 128: `"simt_c8_ffma_l16_e16t64"` | 97, 112, 128 | x0.557-0.570; x0.536-0.545 | x0.835 / x0.807 | x0.779 / x0.669 |
| `sm_100a` (GB200) hidden 6144 | 129 to 192: `"simt_c8_ffma_l16_e16t32"` | 129, 160, 176, 192 | x0.757-0.878; x0.727-0.850 | x1.006 / x0.934 | x0.931 / x0.915 |
| `sm_103a` (GB300) hidden 4096 | 1 to 16: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8, 9, 12, 16 | x0.251-0.424; x0.246-0.412 | x0.484 / x0.427 | x0.681 / x0.550 |
| `sm_103a` (GB300) hidden 4096 | 17 to 96: `"simt_c8_ffma_l16_e16t32"` | 17, 32, 64, 96 | x0.427-0.540; x0.412-0.520 | x0.643 / x0.629 | x0.698 / x0.676 |
| `sm_103a` (GB300) hidden 4096 | 97 to 128: `"simt_c8_ffma_l16_e16t64"` | 97, 112, 128 | x0.596-0.613; x0.574-0.588 | x0.903 / x0.860 | x0.755 / x0.698 |
| `sm_103a` (GB300) hidden 4096 | 129 to 192: `"simt_c8_ffma_l16_e16t32"` | 129, 160, 176, 192 | x0.764-0.882; x0.736-0.859 | x1.001 / x0.898 | x0.940 / x0.917 |
| `sm_103a` (GB300) hidden 5120 | 1 to 16: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8, 9, 12, 16 | x0.232-0.390; x0.226-0.379 | x0.450 / x0.401 | x0.683 / x0.541 |
| `sm_103a` (GB300) hidden 5120 | 17 to 96: `"simt_c8_ffma_l16_e16t32"` | 17, 32, 64, 96 | x0.403-0.519; x0.392-0.511 | x0.624 / x0.529 | x0.692 / x0.669 |
| `sm_103a` (GB300) hidden 5120 | 97 to 128: `"simt_c8_ffma_l16_e16t64"` | 97, 112, 128 | x0.563-0.580; x0.553-0.563 | x0.845 / x0.706 | x0.751 / x0.687 |
| `sm_103a` (GB300) hidden 5120 | 129 to 192: `"simt_c8_ffma_l16_e16t32"` | 129, 160, 176, 192 | x0.751-0.871; x0.730-0.841 | x0.995 / x0.982 | x0.934 / x0.911 |
| `sm_103a` (GB300) hidden 6144 | 1 to 16: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8, 9, 12, 16 | x0.213-0.370; x0.212-0.364 | x0.435 / x0.373 | x0.681 / x0.547 |
| `sm_103a` (GB300) hidden 6144 | 17 to 96: `"simt_c8_ffma_l16_e16t32"` | 17, 32, 64, 96 | x0.388-0.501; x0.380-0.494 | x0.608 / x0.601 | x0.756 / x0.661 |
| `sm_103a` (GB300) hidden 6144 | 97 to 128: `"simt_c8_ffma_l16_e16t64"` | 97, 112, 128 | x0.540-0.554; x0.526-0.538 | x0.818 / x0.807 | x0.776 / x0.660 |
| `sm_103a` (GB300) hidden 6144 | 129 to 192: `"simt_c8_ffma_l16_e16t32"` | 129, 160, 176, 192 | x0.738-0.862; x0.717-0.836 | x0.987 / x0.920 | x0.923 / x0.903 |
| `sm_107a` (R200, 212 SMs) hidden 4096 | 1 to 8: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8 | x0.262-0.300; x0.273-0.306 | x0.308 / x0.312 | x0.526 / x0.422 |
| `sm_107a` (R200, 212 SMs) hidden 4096 | 9 to 192: `"simt_c8_ffma_l16_e16t32"` | 9, 12, 16, 17, 32, 64, 96, 97, 112, 128, 129, 160, 176, 192 | x0.375-0.795; x0.375-0.771 | x0.806 / x0.811 | x0.895 / x0.852 |
| `sm_107a` (R200, 212 SMs) hidden 5120 | 1 to 8: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8 | x0.247-0.280; x0.256-0.285 | x0.284 / x0.287 | x0.487 / x0.388 |
| `sm_107a` (R200, 212 SMs) hidden 5120 | 9 to 192: `"simt_c8_ffma_l16_e16t32"` | 9, 12, 16, 17, 32, 64, 96, 97, 112, 128, 129, 160, 176, 192 | x0.346-0.780; x0.354-0.760 | x0.787 / x0.791 | x0.863 / x0.846 |
| `sm_107a` (R200, 212 SMs) hidden 6144 | 1 to 8: `"simt_c8_kg4_l32_e8t4"` | 1, 4, 5, 8 | x0.230-0.259; x0.237-0.261 | x0.261 / x0.263 | x0.456 / x0.363 |
| `sm_107a` (R200, 212 SMs) hidden 6144 | 9 to 192: `"simt_c8_ffma_l16_e16t32"` | 9, 12, 16, 17, 32, 64, 96, 97, 112, 128, 129, 160, 176, 192 | x0.332-0.771; x0.333-0.748 | x0.778 / x0.781 | x0.863 / x0.831 |

*Table: the forward SIMT windows of 152-SM GB200 (`sm_100a`) and GB300 (`sm_103a`) parts and of 212-SM R200 (`sm_107a`)
parts (their table in `ROUTER_RULES_BY_SM`; two nodes per part, runs
serial, four runs per node with the arms in alternating orders).  Times are CUPTI spans of the operation's kernels at
cold L2, in eager launches and CUDA graph replays, against the plan of the row without the window (the deterministic
program, four splits).  Forward alone: the worst run per token count, the range over the measured token counts.  Next
to a kernel holding 8, 16 or 32 SMs on another stream: the worst cell.  Step (forward and backward pass) and
checkpointed step (2 x forward + backward): the median over the runs' medians, the larger of eager launches and graph
replays; both arms run the same backward pass.  A window holds where at every measured token count that median of the
operation, the step and the checkpointed step is at most x1.013, no run is slower than the slowest run of an A/A control
(both arms the plan without the window), and no cell next to a resident kernel is above x1.03.  At hidden 4096 on GB200
176 tokens were 3.0 % slower than that plan next to a resident kernel in one run (CUDA graphs; 192 tokens 3.4 % in an
earlier measurement), so that window ends at 160 tokens.  Every window's program kept the router budget at each measured
token count (10 seeds, worst of 20 runs, the same bits in all 20; activations and weights Gaussian and BF16-valued):
a relative Frobenius error against FP64 of at most 5.82e-8 in the compensated windows (x0.36-0.51 of the deterministic
program's) and 1.53e-7 in the FFMA windows (x1.11-1.27; on the 212-SM R200 parts 1.57e-7, x1.108-1.289), all values finite, and no row whose top-8 experts differ from
FP64 outside near ties of the 8th and 9th logits.  cuBLAS FP32 (cuBLAS 13.1.1; per token count the faster of `torch.mm`
without TF32 and `CUBLAS_FP32_EMULATED_BF16X9_MATH`, both within the router budget) took x0.83-2.78 of the program's
time: faster than the program at one token and at some token counts from 160 to 192 at hidden 4096 and 6144.  The deterministic
package's `cake_dense_gemm_det_train_router_forward` took x1.17-4.88 of it in CUDA graphs and x1.56-6.53 in eager launches.  On
the 212-SM R200 parts cuBLAS FP32 (cuBLAS 13.8.0.4, the same two launches; `torch.mm` was the faster at every row) took
x0.790-2.746 of the program's time in CUDA graphs and x0.832-2.677 in eager launches (faster than the program at one token
and at some token counts from 160 to 192), and the deterministic package's forward x1.330-4.604 and x1.365-4.416.  Above 192 tokens the
rows keep the deterministic program (not measured under this rule).  Other SM counts, other rows and the other `sm_107a`
parts (208 and 216 SMs) run the deterministic program.*

`"simt_n<tile_n>m<tile_m>s<slices><form>"` (`form` `g<group>` or `dot2`) runs a compensated exact-FP32 program with
the plain program's tiles, slices, launch and stores: each slice keeps a sum and its compensation, and the slices merge
in slice order by TwoSum with their compensations (every FP32 instruction rounded to nearest, nothing flushed to
zero).  With `g<group>` (`router_fp32_gemm_simt_kahan_n<tile_n>m<tile_m>s<slices>g<group>`,
`router_fp32_dgrad_simt_kahan_m<tile_m>n<tile_n>s<slices>g<group>`; `g1` without the suffix) a slice runs, per output
and group of `group` consecutive K steps, one FMA chain from the negated compensation and one Kahan add (`(group + 3) /
group` FP32 instructions per product).  With `dot2` (the weight gradient only,
`router_fp32_gemm_simt_kahan_dot2_n<tile_n>m<tile_m>s<slices>`) each product is rounded once with its exact error kept
and added by TwoSum, both errors going into the compensation, in blocks of 8 rows without the rows past `K` (10 FP32
instructions per product).  This checkout registers one compensated program, on `sm_100a` and `sm_103a`: the 64 x 64
weight-gradient tile in two slices as `dot2` (128 threads).

On 148-SM B200 (`sm_100a`) and B300 (`sm_103a`) parts the tables of `ROUTER_RULES_BY_SM` name the plain SIMT programs,
which this checkout registers on both architectures (the 64 x 64 and the 128 x 64 weight-gradient tiles in two
slices, the 32 x 64 input-gradient tile in four and in eight slices), and at one token at hidden 4096 the Dot2 program:

| row (256 experts) | token window: spec | time before / new, alone, B200; B300 | deterministic / new, alone, B200; B300 | worst next to a resident kernel, new / before: same launch order / router launched first | backward / checkpointed step, new / before, worst | FP64 relative Frobenius error, worst |
|---|---|---|---|---|---|---|
| weight gradient, hidden 4096 | 2 to 224: `"simt_n128m64s2"` | x1.23-2.48; x1.24-2.44 | x1.46-5.30; x1.48-5.19 | x0.997 / x0.850 | x0.979 / x1.003 | 1.93e-7 |
| weight gradient, hidden 5120 | 1 to 256: `"simt_n64m64s2"` | x1.13-2.94; x1.18-3.10 | x1.13-5.03; x1.18-5.10 | x0.863 / x0.851 | x0.948 / x1.002 | 2.06e-7 |
| weight gradient, hidden 6144 | 1 to 256: `"simt_n64m64s2"` | x1.16-3.07; x1.22-3.15 | x1.16-5.25; x1.22-5.48 | x0.901 / x0.791 | x0.910 / x0.989 | 2.06e-7 |
| input gradient, hidden 4096 | 1 to 64: `"simt_n64m32s8"` | x1.32-1.98; x1.32-2.00 | x1.88-2.06; x1.80-2.05 | x0.822 / x0.776 | x0.869 / x0.971 | 1.22e-7 |
| input gradient, hidden 5120 | 1 to 32: `"simt_n64m32s8"` | x1.26-1.35; x1.28-1.33 | x1.96-2.13; x1.99-2.07 | x0.794 / x0.798 | x0.899 / x0.988 | 1.20e-7 |
| input gradient, hidden 5120 (B200 only) | 64 to 128: `"simt_n64m32s4"` | x1.08-1.39 | x1.08-1.39 | x0.930 / x0.939 | x0.961 / x0.994 | 1.51e-7 |
| input gradient, hidden 6144 | 1 to 32: `"simt_n64m32s8"` | x1.25-1.31; x1.26-1.30 | x1.95-2.11; x2.00-2.06 | x0.803 / x0.807 | x0.905 / x0.986 | 1.23e-7 |

*Table: the SIMT windows of 148-SM B200 (`sm_100a`) and B300 (`sm_103a`) parts (one node of each part), measured
against the plan each token count took before -- the compensated SIMT programs up to 160 tokens of the weight gradient
and up to 32 tokens of the input gradient, the deterministic program above -- and against the deterministic program.
Time alone: the program by itself in CUDA graph replays, cold L2, over the measured token counts of the window, in a
long-lived process and in a fresh process per token count.  Next to a kernel holding 8, 16 or 32 SMs on another stream,
and for the whole backward pass and the checkpointed step (2 x forward + backward): time / the previous plan's, the
worst of both parts, of eager launches and CUDA graph replays and of both processes.  The weight-gradient windows were
measured at 1 (hidden 5120 and 6144), 2, 3, 4, 5, 6, 8, 9, 12, 16, 17, 18, 19, 20, 24, 32, 33, 40, 48, 56, 63, 64, 65,
80, 96, 97, 112, 128, 129, 144, 160, 176, 192, 193, 224 and 256 (hidden 5120 and 6144) tokens, the input-gradient
windows at 1, 2, 16, 32, 33 and 64 tokens (hidden 4096), 1, 2, 16 and 32 (hidden 5120 and 6144) and 64, 65, 96 and 128
(hidden 5120).  A window starts and ends at measured token counts and holds at least one more timed and two more
accuracy-checked token counts inside.  Every window's program kept the router budget at each measured token count:
the FP64 error is the worst seed of 10 and the worst of 20 runs, 0.95 to 2.66 times the deterministic program's
(reported data).  At one token hidden 4096 keeps the Dot2 program: there the 128 x 64 tile left the backward pass up
to 1.9 % (B200) and 3.8 % (B300) slower; at 256 tokens it was up to 11 % (B200) and 13 % (B300) slower next to a
resident kernel.  On B300 the 32 x 64 tile in four slices left the checkpointed step at 128 tokens 0.09 % and 0.18 %
slower than before in eager launches, more than the plan measured against itself in the same runs, so B300 has no
input-gradient window above 32 tokens at hidden 5120.  Past the input-gradient windows (hidden 4096 from 65 tokens,
hidden 5120 from 33 to 63 tokens and on B300 from 33, hidden 6144 from 33) the 32 x 64 tile in four slices and the
64 x 64 tile in four slices met the timing rule at some measured token counts, but no window there holds enough
measured token counts inside, so these token counts run the deterministic program; at 128 tokens at hidden 6144 the
32 x 64 tile in four slices was up to 12 % (B200) and 17 % (B300) slower next to a resident kernel.  152-SM, 212-SM
and 216-SM parts have no compensated window.*

`"H<f>x<k>"` (`router_partition_of`, the weight gradient only) runs the head/tail program
`router_fp32_gemm_atomic_nn_t_ht_c3_nreg` (on `sm_103a` and on 212-SM `sm_107a` parts:
`router_fp32_gemm_ts_r6p6_atomic_nn_t_ht_c2_nreg`, the same products, accumulators and part segments with the BF16 parts
of `A` in tensor memory, `ROUTER_HEAD_TAIL_TEMPLATES` / `ROUTER_HEAD_TAIL_TEMPLATES_BY_SM`):
one head unit per CTA pair, whose spans share about the fraction `f` of
the (row-tile pair, K block) steps, then `k` tail units per CTA pair over the rest (each count a multiple of the
column tile count).  Every pair starts a long head span at once; the short tail spans go to the pairs that finish
first, so a kernel holding SMs on another stream delays a short span rather than a long one.  `"H0.5x3"` is 76 head
and 228 tail units at 152 SMs; `"H0.5x4"` is 104 head and 416 tail units at 208 SMs, 106 head and 424 tail units at
212 SMs and 108 head and 432 tail units at 216 SMs.  A token count too short for one step per span runs the
deterministic program.
`"H<f>x<k>p<h>"` starts only `h` head units (at most one per CTA pair), each about `f` of a pair's even share of the
steps; the other pairs start on the tail spans, so a kernel holding the SMs of up to `pairs - h` CTA pairs delays no
head span.  At 152 SMs `"H1x4p44"` is 44 head and 304 tail units and `"H1x5p44"` 44 head and 380 tail units.

| row of `ROUTER_RULES` (weight gradient, 256 experts) | token window | measured at (tokens) | part | time / deterministic plan, alone | worst next to a resident kernel: same launch order / router launched first |
|---|---|---|---|---|---|
| `sm_100a` hidden 5120 `"waves"` | 1 to 128 | 64, 128 | GB200, 152 SMs | x0.691-0.884 | x0.864 / x0.846 |
| `sm_100a` hidden 5120 `"H0.5x3"` | 10240 to 16384; 16400 to 28672 | 10240, 12288, 16172, 16231, 16384, 32768; again: 12288, 16128, 16384, 16385, 16400, 18432, 20480, 24576, 28672, 32768; at this change: 10240, 11264, 12288, 12289, 16128, 16129, 16384, 16385, 16400, 18432, 20480, 24576, 28672 | GB200, 152 SMs | x0.750-0.883; again x0.857-0.940; at this change x0.856-0.982 | x1.003 / x1.011; again x1.045 / x1.033; at this change x1.072 / x1.044 |
| `sm_103a` hidden 5120 `"waves"` | 1 to 192 | 64, 128, 192 | GB300, 152 SMs | x0.706-0.919 | x1.005 / x0.917 |
| `sm_103a` hidden 5120 `"H0.5x3"` | 16172 to 32768 | 16172, 16231, 16384, 32768; again: 16172, 16200, 16231, 16232, 16384, 16385, 24576, 32768; at this change: 16172, 16231, 16232, 16384, 16385, 20480, 24576, 28672, 32768 | GB300, 152 SMs | x0.820-0.875; again x0.801-0.897; at this change x0.805-0.964 | x0.984 / x1.020; again x1.043 / x0.980; at this change x1.061 / x0.996 |
| `sm_103a` hidden 4096 `"H1x4p44"` | 20480 to 49152 | 20480, 24576, 32768, 49152 | GB300, 152 SMs (four or five parts) | x0.862-0.952 | x1.011 / x1.008 |
| `sm_103a` hidden 6144 `"H1x5p44"` | 16385 to 20480 | 16172, 16231, 16384, 20480; at this change: 16172, 16231, 16384, 16385, 20480 | GB300, 152 SMs (four parts; at this change two) | x0.922-0.956; at this change (16385, 20480) x0.948-0.959 | x1.025 / x0.992; at this change x1.023 / x1.008 |
| `sm_107a` hidden 5120 `"H0.5x4"` | 16172 to 16384 | 16172, 16231, 16384 | R200, 216 SMs (two parts) | x0.793-0.819 | x0.960 / x0.963 |
| `sm_107a` hidden 5120 `"H0.5x4"` (`ROUTER_RULES_BY_SM`, 212 SMs, the head/tail program with the `A` parts in tensor memory) | 16172 to 16384 | 16172, 16231, 16384 | R200, 212 SMs (four parts in two nodes) | x0.735-0.770 | x0.922 / x0.925 |
| `sm_107a` hidden 6144 `"H0.5x4"` (`ROUTER_RULES_BY_SM`, 212 SMs, the head/tail program with the `A` parts in tensor memory) | 16172 to 16384 | 16172, 16231, 16384 | R200, 212 SMs (four parts in two nodes) | x0.831-0.870 | x0.981 / x0.961 |
| `sm_107a` hidden 5120 `"H0.5x4"` (`ROUTER_RULES_BY_SM`, 208 SMs) | 16172 to 16384 | 16172, 16231, 16384 | R200, 208 SMs (two parts, and a third at a lower SM clock) | x0.816-0.830 | x1.002 / x1.015 |

*Table: the token windows of the router weight gradient (eager launches and CUDA graph replays, one long-lived process
and a fresh process per token count; the resident kernel holds 8, 16 or 32 SMs).  The deterministic plan of the
first measurements is six splits.  From 16129 to 16384 tokens on 148- and 152-SM `sm_100a` / `sm_103a` parts the
deterministic package plans nine splits at hidden 5120 and three at hidden 6144 (`ROUTER_SPLIT_RULES`); the 152-SM
head/tail windows at and next to those token counts were measured again ("again", two parts each) against the plan the
row runs without them (the rule's count there, six elsewhere), alone, in the backward pass and in the checkpointed
step, eager and in CUDA graphs.  A token count stays in a window where on both parts the partition was faster alone in
every run (paired sign test, p < 0.05, median ratio below 0.99) and the backward pass and the checkpointed step were no
slower (pooled median ratio at most 1.013, no run slower than the slowest run of that plan against itself); the slowdown
next to a resident kernel ("again" and "at this change" in the last column) is data, not a condition.  The comparison
is what a caller gets: the window through this package's entry against the deterministic plan through it (alone in
eager launches the entry adds about 14 us to the head/tail program and about 6 us to the deterministic plan, a direct
call of the deterministic program's binding under 1 us, so a miss alone in eager launches against that binding alone
does not remove a window).  `sm_100a` hidden 5120 `"H0.5x3"` was kept at the six measured token counts from 16385 to
28672 (against six) and missed at 32768 (the checkpointed step in CUDA graphs on one part); measured once more with
these tables ("at this change"), main's window was faster than the deterministic plan through the entry at 12289,
16128 and 16384 tokens on both parts (at 16129 on one part; on the other one eager checkpointed-step run was 3.3 %
slower), and at 10240, 11264 and 12288 faster than the deterministic program in CUDA graphs, in the backward pass and
in the checkpointed step (alone in eager launches it missed against that program's binding only), and the second
window was kept at 16400 to 28672 and missed at 16385 (one eager checkpointed-step run on one part): its windows are
10240 to 16384 and 16400 to 28672.  `sm_103a` hidden 5120
`"H0.5x3"`, measured once more with these tables ("at this change"), was faster than the deterministic plan at 16172
and 16231 tokens (nine splits) on both parts, alone, in the backward pass and in the checkpointed step, eager and in
CUDA graphs (through the entry against the plan through it on one part: x0.949-0.961 alone in eager launches,
x0.901-0.904 in CUDA graphs; the "again" misses there were one eager run each alone), so its window stays 16172 to
32768.  `sm_103a` hidden 6144 `"H1x5p44"` through the entry was 2.9 to 3.7 % slower alone in eager launches than the
deterministic plan through it at 16172, 16231 and 16384 tokens (three splits; every run on one part; no slower in the
backward pass and the checkpointed step, x0.980-0.986 alone in CUDA graphs) and no slower at 16385 and 20480 (six
splits; alone in eager launches x1.011 and x0.999, x0.996 and x0.999 net of the change of the deterministic package's
own entry between the two runs, x0.957 and x0.964 in CUDA graphs): its window is 16385 to 20480.  The `"waves"` rows
are the FI-main
`"waves"` plan (same kernel text, measured on this package's host path).  On GB200 the rows from 10240 tokens up ran at
the GPU's software power cap: after 21 of their 24 runs `nvidia-smi` read `clocks_throttle_reasons` 0x4, at SM clocks of
1297-1912 MHz (2062 MHz maximum); on the R200 parts it read 0x4 after some rows as well (sampled every 200 ms only
during 6 of the 12 runs at hidden 5120 and 6 of the 10 at hidden 6144, those on two of the four 212-SM parts: 0x4 in
17-47 % of the non-idle samples, median SM clock
2388-2400 MHz, no hardware slowdown or thermal reason), and during the GB300 hidden 4096 / 6144 runs in part of the
500 ms samples (median SM clock 1830-2070 MHz of 2070 MHz).  The arms of each row were measured interleaved, in paired
groups.  Inside the head/tail windows cuBLAS FP32 in
`CUBLAS_FP32_EMULATED_BF16X9_MATH` (the benchmark's `bf16x9` arm) took x1.455-1.565 (GB200 hidden 4096), x1.516-1.594
(GB200 hidden 5120), x1.500-1.640 (GB300), x1.630-1.715 (R200, 216 SMs), x1.809-1.915 (R200, 212 SMs, hidden 5120 and
6144) and x1.639-1.768 (R200, 208 SMs) of the partition's time: at least x1.455.  On 212 SMs the head/tail program with
the BF16 parts of `A` in tensor memory took x0.916-0.954 of the three-chunk-buffer program's time at hidden 5120 and
x0.906-0.950 at hidden 6144 (same partition, alone); with the three-chunk-buffer program hidden 6144 was up to 9.2 %
slower than the deterministic plan next to a resident kernel, so its window holds only with the tensor-memory program.
On the 208-SM and 212-SM parts at full SM clock the three-chunk-buffer program with `"H0.5x3"` was 0.9-2.1 % faster
than with `"H0.5x4"`, but on a 208-SM part at a lower SM clock (about 2000 MHz; there `"H0.5x4"` took x0.931-0.959
alone) it was up to 4.9 % slower than the deterministic plan when launched before a resident kernel, so both SM counts
take `"H0.5x4"`.  The head/tail windows hold
only on the SM count of their part (`ROUTER_SM_COUNT`, or the part's own table in `ROUTER_RULES_BY_SM`), the
`"waves"` windows on every SM count where `"waves"` applies (`router_waves`); every other token count and SM count runs
the deterministic plan.  Measured again on 152-SM `sm_100a` parts against the current deterministic plan, hidden 4096
`"H1x5p44"` (10240 to 16384 tokens; x0.917-0.943 alone in the first GB200 measurement, the cuBLAS ratios above) left
the checkpointed step up to 2.7 % slower in eager launches at 10240 and 12288 tokens, so that row runs the
deterministic plan above its SIMT window; hidden 5120 `"H0.5x3"` was up to 4.5 % slower than that plan next to a
resident kernel at 24576 tokens and up to 6.9 % at 32768, so its window ended at 16384 tokens (16385 to 24575 tokens
were not measured; its windows are 10240 to 16384 and 16400 to 28672 now, above).
Hidden 5120 on `sm_100a` / `sm_103a` ran `"waves"` at every token count before these windows: below 64 tokens the
`"waves"` windows keep it without a measurement; between the windows and above the last one (28672 tokens on
`sm_100a`, 32768 on `sm_103a`) the row now runs the deterministic plan -- there `"waves"` was more than 3 % slower than
that plan next to a resident kernel at every measured token count from 224 to 10240 (x1.046-1.156 on GB200 at 224 to
12288 tokens, x1.059-1.167 on GB300 at 224 to 10240 tokens), and above 32768 tokens nothing was measured.*

The router windows' FP64 error: per seed the relative Frobenius error against FP64 of the exact operands (worst of 20
runs into a NaN-filled `out`) of the window's program, on the benchmark's router inputs (FP32 activations, router weight
and output gradient), 10 seeds per token count.  A window keeps the router budget where this error is at most `2e-6`
for every seed and run; every window below does (at most 3.3e-7).  The last column, the error divided by that of the
plan the row takes without the window on the same operands (the deterministic program; at hidden 5120 on 152-SM parts
up to 128 tokens on `sm_100a` and up to 192 on `sm_103a`, `"waves"`), is reported as data, not as a condition.
The SIMT windows of 148-SM parts are in the 148-SM SIMT table above.

| row (256 experts) | token window | measured at (tokens, 10 seeds each) | part | relative Frobenius error, largest seed | error / error without the window (data) |
|---|---|---|---|---|---|
| `sm_100a` weight gradient hidden 4096 `"simt_n128m64s2"` | 1 to 224 | 1, 113, 224 | GB200, 152 SMs | 1.93e-7 | x2.226-2.423 (x0.932-0.959 at 1 token) |
| `sm_100a` weight gradient hidden 5120 `"simt_n64m64s2"` | 1 to 256 | 1, 128, 129, 256 | GB200, 152 SMs | 2.06e-7 | x2.300-2.536 (x0.928-0.951 at 1 token) |
| `sm_100a` weight gradient hidden 6144 `"simt_n64m64s2"` | 1 to 256 | 1, 129, 256 | GB200, 152 SMs | 2.06e-7 | x2.298-2.531 (x0.927-0.951 at 1 token) |
| `sm_100a` input gradient hidden 4096 `"simt_n64m32s4"` | 1 to 128 | 1, 64, 128 | GB200, 152 SMs | 1.58e-7 | x1.697-2.067 |
| `sm_100a` input gradient hidden 5120 `"simt_n64m32s4"` | 1 to 128 | 1, 64, 128 | GB200, 152 SMs | 1.55e-7 | x1.679-1.967 |
| `sm_100a` input gradient hidden 6144 `"simt_n64m32s4"` | 1 to 64 | 1, 32, 64 | GB200, 152 SMs | 1.59e-7 | x1.661-1.933 |
| `sm_100a` weight gradient hidden 5120 `"waves"` | 1 to 128 | 1, 64, 128 | GB200, 152 SMs | 6.29e-8 | x1.000-1.002 |
| `sm_100a` weight gradient hidden 5120 `"H0.5x3"` | 10240 to 16384; 16400 to 28672 | 10240, 11264, 12288, 12289, 16128, 16129, 16384, 16400, 18432, 20480, 24576, 28672 | GB200, 152 SMs | 1.96e-7 | x1.014-1.152 |
| `sm_103a` weight gradient hidden 4096 `"simt_n64m64s2"` | 1 to 288 | 1, 161, 288 | GB300, 152 SMs | 2.18e-7 | x2.450-2.675 (x0.932-0.959 at 1 token) |
| `sm_103a` weight gradient hidden 5120 `"simt_n64m64s2"` | 1 to 192 | 1, 97, 192 | GB300, 152 SMs | 1.79e-7 | x2.037-2.579 (x0.928-0.951 at 1 token) |
| `sm_103a` weight gradient hidden 6144 `"simt_n64m64s2"` | 1 to 256 | 1, 129, 256 | GB300, 152 SMs | 2.06e-7 | x2.298-2.531 (x0.927-0.951 at 1 token) |
| `sm_103a` input gradient hidden 4096 `"simt_n64m32s4"` | 1 to 128 | 1, 64, 128 | GB300, 152 SMs | 1.58e-7 | x1.697-2.067 |
| `sm_103a` input gradient hidden 5120 `"simt_n64m32s4"` | 1 to 128 | 1, 64, 128 | GB300, 152 SMs | 1.55e-7 | x1.679-1.967 |
| `sm_103a` input gradient hidden 6144 `"simt_n64m32s4"` | 1 to 64 | 1, 32, 64 | GB300, 152 SMs | 1.59e-7 | x1.661-1.933 |
| `sm_100a` forward hidden 4096, 5120, 6144 `"simt_c8_kg4_l32_e8t4"`, `"simt_c8_kg4_l32_e8t16"` | 1 to 16 | 1, 2, 3, 4, 5, 8, 9, 12, 14, 15, 16 | GB200, 152 SMs | 5.82e-8 | x0.36-0.51 |
| `sm_100a` forward hidden 4096, 5120, 6144 `"simt_c8_ffma_l16_e16t32"`, `"simt_c8_ffma_l16_e16t64"` | 17 to 192 (hidden 4096: 160) | 17, 24, 31, 32, 33, 48, 63, 64, 65, 96, 104, 112, 120, 127, 128, 129, 144, 160, 176, 192 | GB200, 152 SMs | 1.53e-7 | x1.11-1.27 |
| `sm_103a` forward hidden 4096, 5120, 6144 `"simt_c8_kg4_l32_e8t4"` | 1 to 16 | 1, 4, 5, 8, 9, 12, 14, 16 | GB300, 152 SMs | 5.82e-8 | x0.359-0.505 |
| `sm_103a` forward hidden 4096, 5120, 6144 `"simt_c8_ffma_l16_e16t32"`, `"simt_c8_ffma_l16_e16t64"` | 17 to 192 | 17, 32, 64, 96, 97, 104, 112, 120, 128, 129, 144, 160, 176, 192 | GB300, 152 SMs | 1.53e-7 | x1.112-1.270 |
| `sm_107a` forward hidden 4096, 5120, 6144 `"simt_c8_kg4_l32_e8t4"` | 1 to 8 | 1, 4, 5, 8 | R200, 212 SMs | 5.82e-8 | x0.359-0.505 |
| `sm_107a` forward hidden 4096, 5120, 6144 `"simt_c8_ffma_l16_e16t32"` | 9 to 192 | 9, 12, 14, 16, 17, 32, 64, 96, 97, 104, 112, 120, 128, 129, 144, 160, 176, 192 | R200, 212 SMs | 1.57e-7 | x1.108-1.289 |
| `sm_103a` weight gradient hidden 4096 `"H1x4p44"` | 20480 to 49152 | 20480, 32768, 49152 | GB300, 152 SMs | 3.23e-7 | x1.198-1.294 |
| `sm_103a` weight gradient hidden 5120 `"waves"` | 1 to 192 | 1, 65, 97, 113, 128, 144, 160, 161, 176, 192 | GB300, 152 SMs | 6.96e-8 | x0.979-1.040 |
| `sm_103a` weight gradient hidden 5120 `"H0.5x3"` | 16172 to 32768 | 16172, 16231, 16232, 16384, 16385, 20480, 24576, 28672, 32768 | GB300, 152 SMs | 2.08e-7 | x1.014-1.152 |
| `sm_103a` weight gradient hidden 6144 `"H1x5p44"` | 16385 to 20480 | 16385, 20480 | GB300, 152 SMs | 2.34e-7 | x1.367-1.399 |
| `sm_107a` weight gradient hidden 4096 `"simt_n128m64s2"` | 1 to 256 | 1, 129, 256 | R200, 212 SMs | 2.06e-7 | x2.300-2.535 (x0.929-0.950 at 1 token) |
| `sm_107a` weight gradient hidden 5120 `"simt_n128m64s2"` | 1 to 256 | 1, 129, 256 | R200, 212 SMs | 2.06e-7 | x2.301-2.536 (x0.928-0.951 at 1 token) |
| `sm_107a` weight gradient hidden 6144 `"simt_n128m64s2"` | 1 to 224 | 1, 97, 192, 193, 208, 224 | R200, 212 SMs | 1.93e-7 | x1.995-2.659 (x0.927-0.951 at 1 token) |
| `sm_107a` input gradient hidden 4096 `"simt_n64m32s8"` | 1 to 64 | 1, 32, 64 | R200, 212 SMs | 1.25e-7 | x1.299-1.530 |
| `sm_107a` input gradient hidden 4096 `"simt_n64m64s4"` | 65 to 128 | 65, 96, 128 | R200, 212 SMs | 1.51e-7 | x1.823-1.860 |
| `sm_107a` input gradient hidden 5120 `"simt_n64m32s8"` | 1 to 64 | 1, 32, 64 | R200, 212 SMs | 1.25e-7 | x1.348-1.535 |
| `sm_107a` input gradient hidden 5120 `"simt_n64m64s4"` | 65 to 128 | 65, 96, 128 | R200, 212 SMs | 1.52e-7 | x1.819-1.859 |
| `sm_107a` input gradient hidden 6144 `"simt_n64m32s8"` | 1 to 64 | 1, 32, 64 | R200, 212 SMs | 1.20e-7 | x1.307-1.585 |
| `sm_107a` weight gradient hidden 5120 `"H0.5x4"` | 16172 to 16384 | 16172, 16231, 16384 | R200, 212 SMs | 1.46e-7 | x0.943-0.952 |
| `sm_107a` weight gradient hidden 6144 `"H0.5x4"` | 16172 to 16384 | 16172, 16231, 16384 | R200, 212 SMs | 1.51e-7 | x0.974-0.983 |

*Table: the router FP64 error.  The SIMT programs repeat bit for bit; an exact model of their FFMA order reproduces
their output bit for bit, so the error follows from the order (each slice's products summed in one FFMA chain along the
reduction), not from the SM count: from about 100 tokens the weight gradient's error is about twice the deterministic
program's, and at one token it is below it (each product rounded once).  The forward's FP64 error rows use Gaussian and BF16-valued
operands (10 seeds each); the bits of the forward's program depend on `K`, the kind and the lanes only, so each row
covers both tiles of its kind.  The errors without the windows are those of the deterministic plan's six splits; the
head/tail partitions split a tile's K range over more spans than those six and add them with the TMA reduce-add.*

The K1 forward rows of `K1_FORWARD_RULES_BY_SM` run the deterministic package's
single-CTA program (`K1_FORWARD_SINGLE_CTA_KNOBS`: one 128-row tile per CTA, 128-wide column tiles, no TMA cache hints;
every tile's whole K in one CTA) in place of that package's plan of the row (CTA pairs of 256-wide tiles with the K
range split two ways, the partials added in a fixed order).  The forward stays deterministic: it repeats bit for bit
and a recomputed forward reproduces its result, with other bits than the deterministic plan; both meet the BF16 budget
against FP64 (10 seeds, 20 runs each, worst run).  A row is in the table only at a token count and SM count where it
was measured and met this rule in every run: the forward alone faster than the deterministic plan (a one-sided paired
sign test, p < 0.05, median ratio below 0.99, eager launches and CUDA graph replays) and no more than 3 % slower than
it next to a kernel holding 8, 16 or 32 SMs on another stream (both launch orders); and the training steps that
contain the forward -- forward + backward, 2 x forward + backward with an FP32 and with a BF16 weight gradient -- not
slower than with the deterministic plan: the median over every run of the paired step-time ratio at most 1.013,
eager launches and graph replays separately, and no run above the largest ratio of the deterministic plan against
itself in the same runs.  The backward pass alone runs the same programs in both and is reported, not judged.
Every other token count, SM count and row was not measured or missed, and runs the deterministic plan, as do the
generic `cake_dense_gemm_nondet_train_projection`, `cake_dense_gemm_nondet_train_projection_prepare` and `cake_dense_gemm_nondet_train_projection_plan` (their caller
passes the planner knobs).

| K1 forward row `d_in -> d_out` (BF16 out, both operands K-major) | 152 SMs: token counts (`K1_FORWARD_RULES_BY_SM`) | measured token counts outside them |
|---|---|---|
| `sm_100a` `4096 -> 512` | 2048 | 3072; 4096 |
| `sm_100a` `5120 -> 512` | 1024, 4096 | -- |
| `sm_103a` `4096 -> 512` | 2048 | 3072; 4096 |
| `sm_103a` `5120 -> 512` | 1024 | 4096 |

*Table: the K1 forward table (`K1_FORWARD_RULES_BY_SM` on `sm_100a` / `sm_103a` parts of exactly 152 SMs, GB200 and
GB300).  `4096 -> 512` at 4096 tokens was up to 4.7 % slower than the deterministic plan next to a kernel holding 32
SMs.  At 3072 tokens (`4096 -> 512`, both parts) and at 4096 tokens on `sm_103a` (`5120 -> 512`) the single op was
faster and the step medians were below 1.000, but one run of a training step with eager launches was further above
the deterministic plan than that plan against itself in the same runs: `sm_100a` `4096 -> 512` 2 x forward + backward
with a BF16 weight gradient 1.007 (the deterministic plan against itself at most 1.002); `sm_103a` `4096 -> 512`
forward + backward 1.015 (at most 1.013); `sm_103a` `5120 -> 512` 2 x forward + backward with an FP32 weight gradient
1.015 (at most 1.012).*

The `*_prepare` and `*_plan` entry points take `splits` (and, for K1, `block_n`) to force a plan, for tests and measurements; the K2
forward takes at most two splits.  `benchmarks/bench_cake_dense_gemm_nondet_train.py` prints, per row, the
`rule_key` / `rule_value` of the plan it measured, in the format of these tables.

## Workspace

A K1 launch of the CTA-pair program with more than one split uses an FP32 slot workspace (`tiles * num_splits * 256 * block_n` elements,
zero at creation: the last split of a partial tile reads the padding columns no split writes, and discards them) and
the ticket counters (`tiles * 512` words, zero at creation, zero again after every complete launch).  A prepared
launch owns both; the convenience path keeps them in an arena per (device, stream), so concurrent launches on
different streams never share them.  The cluster program (`K1_DSM_RANGES_BY_SM`) sums its splits in shared memory:
no workspace, no tickets, no arena.  An arena never frees a buffer before `cake_dense_gemm_nondet_train_convenience_cache_clear()`: a growth (to
at least twice the buffer's size) retires the buffer it replaces, and an entry evicted from the caches
(the deterministic backend's `CONVENIENCE_CACHE_SIZE`) leaves its arena, so a launch captured into a CUDA graph keeps its workspace; the retired
buffers of a kind hold fewer bytes than its live one, which holds less than twice the largest requirement seen or the
16-element minimum, so a kind holds at most four times that requirement or 16 elements, whichever is more.  An arena
stays until `cake_dense_gemm_nondet_train_convenience_cache_clear()` even when no cached entry uses it, one per (device, stream) that made a call:
a process that makes calls on many short-lived
streams (`torch.cuda.ExternalStream` handles) calls it once no captured graph replays.  After an aborted launch, call
`CakeDenseGemmNondetProjectionLaunch.reset_counters()` (or `cake_dense_gemm_nondet_train_convenience_cache_clear()`).  A K2 launch with part tiles zeroes `out` on
the launch stream (one memset) between the split and the program; prepared launches of both programs allocate nothing
and can be captured in a CUDA graph.  The K2 K-order program (`"ksum_S<n>"` windows) does not zero `out`: it uses FP32
slots (`(spans + m_pairs - 1) * n_tiles * 32768` elements, `spans = units / n_tiles`) and ticket counters
(`m_pairs * n_tiles * 512` words, zero at creation and again after every complete launch), owned by a prepared launch
and kept in the arena of the convenience path.
A prepared K2 launch binds everything but the stream when prepared (its FFI arguments are converted once), so a call
issues its launches back to back.  A cached K1 launch of the convenience path orders its arguments once per cache key
(the PDL flag is bound, so it is part of the key) and converts its workspace once per arena growth; a call rebuilds and converts only its operand views.  A K1 call that
runs the deterministic program -- the forward, the input gradient, a weight-gradient row without a verdict -- launches
it through a cached binding of this package: the deterministic package's plan and program (its bits, with its `sk` /
`cta_rows` knobs and its weight-gradient orientation), with its stream-K slabs and slice counters in this package's
arena, keyed by the entry point's own tensors.  A batched forward that the deterministic package runs on its swapped
small-token program makes that package's call (with `enable_pdl` and `weights_static`).
A cached K2 launch of the convenience path orders its arguments once per cache key
and converts the arena's BF16x3 stack once per arena growth, so a call passes only the caller's views (each
converted in its FFI call) and enters the arena's stream context, made once.  A K2 call
that runs the deterministic program -- the input gradient, the forward without a split count, a call without a split
count whose row has no verdict in the router table of the device's SM count (`ROUTER_RULES`, or the part's own table
in `ROUTER_RULES_BY_SM`; not planned here) -- launches it through a cached binding of this package: the
deterministic package's plan, programs and `torch.sum` of its split-K partials (its bits), with the stack and the
partials in this package's arena, bound once per arena growth, and the entry point's own tensors as the launch
views.  It issues the same launches as the deterministic package's convenience path in fewer host operations.  Whether a row has a verdict is kept per device and view geometry, and whether any
router table gives the forward one is kept too, until a router table changes: the router tables count their changes,
so a verdict set or removed at run time is seen at the next call (a table rebound to a plain `dict` is read on every
call).  The router entry points (`cake_dense_gemm_nondet_train_router_forward`, `cake_dense_gemm_nondet_train_router_dgrad`, `cake_dense_gemm_nondet_train_router_wgrad`) cache each call once more by their
own tensors, the operation, the split count and the PDL flag: a call goes from that key to the cached launch its
route takes, without building the transposed operand, routing the row or building the launch's own key; the key is
routed again when a router table or `ROUTES` changes (with a table rebound to a plain `dict`, every call takes the
full path).

## Optimized and prebuilt paths

- **Prepared launches.**  `cake_dense_gemm_nondet_train_projection_prepare`,
  `cake_dense_gemm_nondet_train_projection_wgrad_prepare` and `cake_dense_gemm_nondet_train_router_prepare` plan
  once and return a launch whose call issues only the kernels: no planning, no allocation and no host
  synchronization.  Use them in a training step that runs the same shapes many times, and capture them into a CUDA
  graph.  On a row of a K1 program table (below) the prepared entry points return a
  `cake_backend.CakeDenseGemmNondetProgramLaunch` instead: its call runs the convenience entry point of the row's
  operation with the views and the PDL flag given when prepared, so it runs the same program with the same bits.  Its
  first call plans and binds the row; call it once on the stream before capturing it into a CUDA graph.
- **Which entry points reach the K1 program tables.**  The generic entry points take a table only with 2-D views and no
  knob but `transposed_out` and `enable_pdl`; every other call takes the routing of the planner (`ROUTES`).

| rows of | reached by |
|---|---|
| `K1_FORWARD_PROGRAM_RULES_BY_SM`, `K1_FORWARD_RULES_BY_SM` | `cake_dense_gemm_nondet_train_projection_forward`; `cake_dense_gemm_nondet_train_projection` and `cake_dense_gemm_nondet_train_projection_prepare` of `A = X`, `B = W.T` (both K-major) |
| `K1_DGRAD_PROGRAM_RULES_BY_SM`, `K1_DGRAD_NVJET_RULES_BY_SM`, `K1_DGRAD_RULES_BY_SM` | `cake_dense_gemm_nondet_train_projection_dgrad`; `cake_dense_gemm_nondet_train_projection` and `cake_dense_gemm_nondet_train_projection_prepare` of `A = G`, `B = W` |
| `K1_WGRAD_SK_RULES_BY_SM` | `cake_dense_gemm_nondet_train_projection_wgrad` and `cake_dense_gemm_nondet_train_projection_wgrad_prepare` with the default knobs; `cake_dense_gemm_nondet_train_projection` and `cake_dense_gemm_nondet_train_projection_prepare` of `A = X.T`, `B = G` with `transposed_out=True` |
| `K1_WGRAD_DET_RULES_BY_SM` | `cake_dense_gemm_nondet_train_projection_wgrad` and `cake_dense_gemm_nondet_train_projection_wgrad_prepare` with the default `sk` and `cta_rows` (the generic entry points run the deterministic plan of these rows) |

*Table: the entry points that run the programs of the K1 program tables.*

- **Convenience entry points.**  The entry points of [Names](#names) cache their plans and workspace per (device,
  stream), so the second and later calls with the same geometry skip planning.  Warm a call up once on the stream
  before capturing it into a CUDA graph (see the CUDA graphs row of the table above).
- **Prebuilt libraries.**  Each program loads from `aot/<library>.so` when the package ships it, and otherwise
  FlashInfer's JIT builds it from `build.json` on its first use (this needs `nvcc`).  To build the programs ahead
  of time in a FlashInfer tree, pass `cake_jit.specs_for_arch(arch)` of this package and of the deterministic
  package to `flashinfer.jit.build_jit_specs`.  A vendoring build compiles the `build.json` entries into `aot/`
  and writes the sidecar `aot/build_info.json` (see `cake_build.py`);
  `scripts/check_cake_build_manifests.py --vendored-gate --package cake_dense_gemm_nondet_train --arch <arch> --keep
  --work-dir <dir>` is the reference builder and leaves the vendored copies with their `aot/` libraries in the work directory.
  `cake_build.library_path(name)` says which library a program loads.

## Benchmark

`benchmarks/bench_cake_dense_gemm_nondet_train.py` times every row of the deterministic family's benchmark with the
arms `det` (the deterministic package's prepared launch), `atomic` (this module, prepared; on a row of a K1 program
table the program that table names), `atomic_p1` (the same
program at one split), `atomic_shim` (the operation's public entry point, the package path), `cublas`
(`torch.matmul`), `cublaslt` (a cuBLASLt heuristic sweep) and `bf16x9` (cuBLAS FP32 in
`CUBLAS_FP32_EMULATED_BF16X9_MATH`, the router rows), in paired groups, eager and in CUDA graphs; `--resident`
re-times `det` and `atomic` next to a kernel holding `--resident-sms` SMs on another stream.  Each row has a class:
`large` (in and out features both above 1024), `small-output` (at most 1024 out features), `batched` (the MLA head
batch) or `router`.  Before timing, every arm runs `--bit-runs` times (default 5) and the row records whether its
outputs repeated bit for bit.  The CSV names each row's family, class, operation, K, N, T, output dtype,
architecture and SM count; the markdown report adds the plain cuBLAS and package-path latencies and a per-class
summary; the header records whether CUPTI timed every call (`cupti_used`).  A row is accepted when every paired
group is faster than `det` and the median ratio is below 0.97, in eager and in graph: the bar of the speed rule for an
output whose bits vary run to run ([Acceptance rules](#acceptance-rules)).  A row whose output repeats bit for bit
needs only the bar of its table ([Planners and rule tables](#planners-and-rule-tables)), so a repeating row that is
not accepted can still be a shipped verdict.

## Tested versions

The benchmark of this package (every row, eager and CUDA graph, with the tables of this version) ran on these GPUs:

| GPU | architecture | SMs | driver (CUDA driver API) |
|---|---|---|---|
| NVIDIA B200 | `sm_100a` | 148 | 580.82.07 (13.4) |
| NVIDIA GB200 | `sm_100a` | 152 | 580.126.20 (13.4) |
| NVIDIA GB300 | `sm_103a` | 152 | 580.159.03 (13.4) |
| NVIDIA R200 | `sm_107a` | 212 | 620.43 (13.5) |

*Table: the GPUs of the measured benchmark runs.*

Software: CUDA toolkit 13.4 (`nvcc` V13.4.92, builds the programs), PyTorch 2.14.1+cu130 (CUDA 13.0 runtime), cuBLAS
and cuBLASLt 13.8.0 (the benchmark's library arms), Python 3.12.15, apache-tvm-ffi 0.1.15.post0, cuda-bindings
13.5.0 and cupti-python 13.4.0 (the benchmark's timer).  The minimum toolkits per architecture are in the
architectures row of the table above.

## Helpers of this package

| name (`flashinfer.experimental.cake.cake_dense_gemm_nondet_train`) | reaches |
|---|---|
| `cake_backend.cake_dense_gemm_nondet_train_convenience_cache_clear` | clears the caches and arenas of both packages (invalidates the CUDA graphs that captured a convenience-path call) |
| `cake_backend.bind_launch` | binds a program of this package (a deterministic program binds through that package's `bind_launch`) |
| `cake_jit.{MODULES, KERNELS, ARG_PLANS}`, `cake_jit.select_module`, `cake_jit.load_cake_dense_projection_gemm_module` | this package's programs only (a deterministic template or program name raises; use the deterministic package's `cake_jit`) |
| `cake_backend.use_deterministic_backend`, `cake_backend.deterministic_backend` | bind and return the deterministic package's `cake_backend` (the package whose fixed-order programs this package reuses) |
| `cake_build` | this package's loader (byte-identical to every Cake package's) |

*Table: the helpers of this package and what they reach.*

A module-level assignment in this package (for example a rule table) does not reach the deterministic package; set
the deterministic package's names there.

## Layout of this package

| file | content |
|---|---|
| `__init__.py` | imports nothing; `cake_backend`, `cake_jit`, `cake_build` load on first attribute access |
| `cake_backend.py` | entry points, routing, the alert, the planners, rule tables and the convenience state of this package's programs |
| `cake_jit.py` | the registry of this package's programs (`ARG_PLANS`, `MODULES`, `KERNELS`) and the loader |
| `cake_build.py`, `build.json`, `csrc/cake_dense_projection_gemm/` | the `cake-build/1` loader, manifest and generated sources of this package's programs (written by the Cake exporter) |

## Standalone build (vendoring)

This backend directory builds and runs outside FlashInfer (flashinfer-ai/flashinfer#6553) with one declared
dependency: the deterministic backend directory `cake_dense_gemm_det_train`
(`flashinfer/experimental/cake/cake_dense_gemm_det_train`), whose fixed-order programs this package runs.  `build.json`
declares it (`common.requires.packages`: `["cake_dense_gemm_det_train"]`, sibling directory names).  A vendoring build
ships the two directories together, next to each other.

- This package imports the deterministic package lazily, on the first call that needs it, as its sibling:
  `<parent>.cake_dense_gemm_det_train` when this package sits in a parent package (a FlashInfer tree), the top-level
  package `cake_dense_gemm_det_train` when this package is vendored as a top-level package.  Importing this package
  never imports the deterministic one; without it, the first use raises `ImportError` that names it.
- A tree that holds the deterministic package under another name binds it with
  `cake_backend.use_deterministic_backend(<its cake_backend module>)` before the first call that needs it.
- The package imports nothing else outside its own directory.  It imports `flashinfer` only in `cake_build`'s guarded
  fallback to FlashInfer's JIT when a library is not prebuilt.
- The mode is the package name: no keyword argument and no environment variable selects it.
