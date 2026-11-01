# Cake dense projection GEMM backend with an explicit numerical mode (SM100 / SM103 / SM107)

Experimental backend for the GLM-5.2 training GEMMs -- the BF16 dense projection GEMM (K1: BF16 operands, FP32
accumulation, BF16 or FP32 output, strided views, ragged token counts, batched head projections) and the FP32
router-gate GEMM through split-BF16x3 tensor-core emulation (K2) -- with a selectable numerical mode.  Family issue:
flashinfer-ai/flashinfer#5677.

The deterministic implementation is the package `flashinfer.experimental.dense_projection_det_gemm`
(flashinfer-ai/flashinfer#5677, #7043; see its README for the view contract, the host paths and the per-architecture
programs).  This package keeps its operations, dtypes, accuracy budgets and host paths, and adds programs whose
split-K reductions add the partials in the order they finish.  Only run-to-run bitwise reproducibility is relaxed,
and only where a row adds more than two partials.

Both the API and the backend are experimental: no compatibility guarantee.

## Numerical mode

Every entry point of `cake_backend` takes the keyword-only `deterministic: bool = True`.

| call | runs |
|---|---|
| `deterministic=True` (default) | the deterministic package's canonical entry point with the same arguments: the same programs, the same bits, the same errors |
| `deterministic=False` | this package's programs on the rows that carry a measured verdict for the device (below); the deterministic programs for every other row and operation |
| `deterministic=False` under `torch.use_deterministic_algorithms(True)` | `RuntimeError`; with `warn_only=True` a `UserWarning` and the deterministic path |

*Table: what a call runs, by its `deterministic` argument.  The default does not read PyTorch's setting: it is
deterministic already.*

| operation | `deterministic=False` |
|---|---|
| K1 forward (`projection_forward`, `kk` layouts) | deterministic program (whole output tiles; a recomputed forward reproduces its result) |
| K1 input gradient (`projection_dgrad`, `kn` layouts) | deterministic program (whole output tiles) |
| K1 weight gradient (`projection_wgrad`, `nn` layouts) | the rows of `K1_SPLIT_RULES`: split-K over the ragged token count -- every split writes its FP32 partial tile to a workspace slot; the last split of a tile to arrive (a per-tile ticket counter) adds the other slots in arrival order and rounds once into `out`, then returns the counter to zero.  Every other row: deterministic program |
| K2 forward (`router_forward`, `kk`) | deterministic program (its four splits) on every row; an explicit `splits` of 1 or 2 runs this package's stream-K program (at most two partials per element) |
| K2 input gradient (`router_dgrad`, `kn`) | deterministic program (one split) |
| K2 weight gradient (`router_wgrad`, `nn_t`) | the rows of `ROUTER_RULES` with a partition: stream-K over the K blocks, in spans alike or in long head spans and short tail spans (the head/tail program) -- whole tiles store the one-split result, tiles split between units TMA reduce-add into `out` (zeroed before the program, on the same stream).  Every other row: deterministic program (its six splits) |
| BF16x3 split (`split_fp32_to_bf16x3_`) | the deterministic stage (exact) |

*Table: the routing of `deterministic=False` (`cake_backend.ROUTES`).*

## Numerics, architectures and parallelism

| property | contract |
|---|---|
| operands and accumulation | BF16 operands with FP32 accumulation (K1); FP32 operands split exactly into three BF16 terms with FP32 accumulation (K2).  No silent TF32 or BF16 for the router |
| accuracy | the deterministic package's budgets against FP64 of the exact operands, worst over repeated runs: BF16 out `atol = rtol = 1e-2`; FP32 out relative Frobenius error `<= 5e-5` and max error `<= 1e-3 * max abs(ref)`; router `<= 2e-6` relative Frobenius error and no row whose top-8 experts differ from FP64 (outside near ties of the 8th and 9th logits) |
| rounding | each output element is rounded once, from its complete FP32 sum |
| subnormals | kept: K1 adds the partials with IEEE round-to-nearest adds, K2 reduces with the TMA reduce-add; nothing flushes to zero |
| run-to-run bits, K1 | two splits (every `sm_100a` and `sm_103a` row, `sm_107a` `8192 -> 128`): repeat bit for bit (one add of two values); one split (`sm_107a` rows marked `p = 1` below): repeat bit for bit (whole tiles); three or four splits (the other `sm_107a` rows): vary in the last bits |
| run-to-run bits, K2 | forward: the deterministic program's bits; weight gradient on a `ROUTER_RULES` partition: varies in the last bits (more than two partials per element); with an explicit forward `splits` of 1 or 2: repeats bit for bit (`0 + a + b == 0 + b + a`), so recomputed router logits and top-k match |
| architectures | `sm_100a`, `sm_103a`, `sm_107a`: one `-gencode=arch=compute_<arch>,code=sm_<arch>` build per program, minimum CUDA 12.8 / 12.9 / 13.4 (`build.json`) |
| streams and threads | planning takes a module lock; the convenience path keeps one workspace arena per (device, stream); prepared launches own their workspace |
| CUDA graphs | prepared launches allocate nothing on the device and capture into a CUDA graph (K2: the split, the memset of `out`, the program); no host synchronization |
| other kernels on the device | the programs walk their work persistently over at most `sm_count // 2` CTA pairs and wait on no CTA that is not resident; every routed row was measured next to a kernel holding 8, 16 or 32 SMs on another stream (no more than 3 % slower than the deterministic plan there) |
| context parallelism (KCP) | not applicable: a local operator on the caller's CP-local shards; no collectives; the reduction of parameter gradients across ranks stays with the caller |
| autograd, checkpointing, gradient accumulation | none in either package: callers compose the forward and both gradients; `out` is overwritten, not accumulated into |

*Table: the numerical, architecture and parallelism matrix of `deterministic=False`.*

## Planners and rule tables

A planner decides, per row and device, whether `deterministic=False` runs this package's program and with which
parameters; a row it leaves runs the deterministic program (`plan_dense_projection_gemm` / `plan_router_fp32_gemm`
say which).  Only measured rows carry a program: a row runs this package's program where that program was faster
than the deterministic package's shipped plan of the row in every measurement run: a one-sided paired sign test
(p < 0.05) with a median time ratio below 0.99, both in eager launches and in CUDA graph replays, and no more than
3 % slower than that plan next to a kernel holding 8, 16 or 32 SMs on another stream.  A `None` entry is a measured
row that missed; a row without an entry was not measured.  Both run the deterministic program.

| table | key | value |
|---|---|---|
| `K1_SPLIT_RULES` | `(arch, L, M, N, out_dtype)`: the transposed-store launch `out[l, n, m]` (`M` = the stored inner extent, `d_in` of a weight gradient; `N` = its rows, `d_out`), `out_dtype` `"bf16"` / `"f32"` | `(block_n, num_splits)` (this package's program) or `None` (the deterministic program) |
| `K1_MIN_SM_COUNT` | `arch` | the fewest SMs for the K1 verdicts of the architecture (`sm_100a` / `sm_103a`: 148, `sm_107a`: 216) |
| `K1_RULE_MIN_SM_COUNT` | a key of `K1_SPLIT_RULES` | the fewest SMs for that row's verdict where it is above `K1_MIN_SM_COUNT` (`sm_103a` `4096 -> 48`: 152) |
| `ROUTER_RULES` | `(arch, layout, hidden, experts)`: `layout` `"kk"` (forward, `hidden = K`) or `"nn_t"` (weight gradient, `hidden` = the launch's `M`), `experts` = the launch's `N` | `None` (the deterministic program), token windows `((min_tokens, max_tokens, spec), ...)` (the spec of the window that holds the token count -- `K` of the weight gradient, `M` of the forward; every other token count: the deterministic program) or a spec alone (every token count: a plan a test or a measurement forces).  A spec is `"waves"`, `"H<f>x<k>"`, `"H<f>x<k>p<h>"`, `"forward"`, `"pairs"`, `"tiles"`, `"S<n>"` or a unit count |
| `ROUTER_SM_COUNT` | `arch` | the SM count of the router verdicts of the architecture (`sm_100a` / `sm_103a`: 152, `sm_107a`: 216); a device with any other SM count runs the deterministic program, except in a `"waves"` window (the condition of `router_waves`) |

*Table: the rule tables.*

| architecture | K1 weight gradient `d_in -> d_out` (`block_n`, `num_splits`), BF16 and FP32 out unless marked | K2 forward | K2 weight gradient |
|---|---|---|---|
| `sm_100a` (>= 148 SMs) | `4096 -> 32, 48, 64`; `5120 -> 32, 48, 64`; `6144 -> 32`: (64, 2).  `4096 -> 128`: deterministic (measured, missed) | deterministic (4 splits) | 256 experts, in the token windows of the last table: on 152 SMs hidden 4096 `"H1x5p44"` and hidden 5120 `"H0.5x3"`; hidden 5120 `"waves"` on every SM count where it applies.  Every other token count and SM count, and hidden 6144: deterministic (6 splits) |
| `sm_103a` (>= 148 SMs) | `4096 -> 32`; `5120 -> 32, 48, 64`; `6144 -> 32`: (64, 2); `4096 -> 48` (>= 152 SMs): (64, 2); `5120 -> 128`, `6144 -> 128`: (128, 2).  `4096 -> 64`, `4096 -> 128`: deterministic (measured, missed) | deterministic (4 splits) | 256 experts, in the token windows of the last table: on 152 SMs hidden 5120 `"H0.5x3"`; hidden 5120 `"waves"` on every SM count where it applies.  Every other token count and SM count, hidden 4096 (`"H1x4p44"` up to 7.6 % slower when launched before a resident kernel) and hidden 6144: deterministic (6 splits) |
| `sm_107a` (>= 216 SMs) | `4096 -> 32` BF16 (64, 4), FP32 (64, 3); `4096 -> 48` FP32 (64, 3); `4096 -> 64` FP32 (64, 3); `5120 -> 32` (64, 3); `5120 -> 48`, `5120 -> 64` FP32 (64, 3); `6144 -> 32` (64, 3); `8192 -> 128` (128, 2); `4096 -> 512` FP32 (128, 1) `p = 1`; `5120 -> 512` FP32 (128, 1) `p = 1`; `4096 -> 1024` FP32 (256, 1) `p = 1`; `5120 -> 1024` (256, 1) `p = 1`.  `4096 -> 64` BF16, `4096 -> 128`, `4096 -> 512` BF16, `4096 -> 4096` FP32: deterministic (measured, missed) | deterministic (4 splits) | deterministic (6 splits); hidden 4096 measured, missed (`"H0.5x4"` loses next to a kernel holding 16 SMs that starts during it); hidden 6144: `"H0.5x4"` leaves the backward pass no faster |

*Table: the rows that run this package's programs.  `4096 -> 128` misses on `sm_100a` / `sm_103a` next to a kernel
holding 32 SMs (its 128 CTAs no longer fit the free SMs) and on `sm_107a` at 16384 tokens; on `sm_107a`,
`4096 -> 512` BF16 (192 CTAs) and `4096 -> 4096` FP32 (216 CTAs) miss next to a kernel holding 32 SMs the same way.
`4096 -> 64` (`sm_103a`; `sm_107a` BF16) and `4096 -> 48` on 148-SM `sm_103a` parts are not faster than the
deterministic plan in every run.  The `sm_107a` one-split rows store whole tiles: they win where the deterministic
package's `sm_107a` program of the row is slower.  Every K1 choice also needs `num_splits * 64 <= T + 63` (shorter
token counts run the deterministic program).*

`"waves"` (`router_waves`) launches whole waves of the device's CTA pairs (`pairs = sm_count // 2`): `c * pairs`
units with `c = max(2, round(tiles * k_iters / (68 * pairs))) + 1`, and only where the deterministic plan's six
splits fill their last wave below 80 % (`w / ceil(w)`, `w = 6 * tiles / pairs`); otherwise the row runs the
deterministic program.  At hidden 5120 on 152 SMs that is 80 units at 64 tokens (at most one unit per (tile,
K block) step) and 380 units at 16k tokens; on 148 SMs the deterministic plan's last wave is 81 % full and the row
runs the deterministic program.

`"H<f>x<k>"` (`router_partition_of`, the weight gradient only) runs the head/tail program
`router_fp32_gemm_atomic_nn_t_ht_c3_nreg`: one head unit per CTA pair, whose spans share about the fraction `f` of
the (row-tile pair, K block) steps, then `k` tail units per CTA pair over the rest (each count a multiple of the
column tile count).  Every pair starts a long head span at once; the short tail spans go to the pairs that finish
first, so a kernel holding SMs on another stream delays a short span rather than a long one.  At 152 SMs `"H0.5x3"`
is 76 head and 228 tail units, at 216 SMs `"H0.5x4"` 108 head and 432 tail units; a token count too short for one
step per span runs the deterministic program.
`"H<f>x<k>p<h>"` starts only `h` head units (at most one per CTA pair), each about `f` of a pair's even share of the
steps; the other pairs start on the tail spans, so a kernel holding the SMs of up to `pairs - h` CTA pairs delays no
head span.  At 152 SMs `"H1x5p44"` is 44 head and 380 tail units.

| row of `ROUTER_RULES` (weight gradient, 256 experts) | token window | measured at (tokens) | part | time / deterministic plan, alone | worst next to a resident kernel: same launch order / router launched first |
|---|---|---|---|---|---|
| `sm_100a` hidden 4096 `"H1x5p44"` | 10240 to 16384 | 10240, 12288, 16172, 16231, 16384 | GB200, 152 SMs | x0.917-0.943 | x1.024 / x0.983 |
| `sm_100a` hidden 5120 `"waves"` | 1 to 128 | 64, 128 | GB200, 152 SMs | x0.691-0.884 | x0.864 / x0.846 |
| `sm_100a` hidden 5120 `"H0.5x3"` | 10240 to 32768 | 10240, 12288, 16172, 16231, 16384, 32768 | GB200, 152 SMs | x0.750-0.883 | x1.003 / x1.011 |
| `sm_103a` hidden 5120 `"waves"` | 1 to 192 | 64, 128, 192 | GB300, 152 SMs | x0.706-0.919 | x1.005 / x0.917 |
| `sm_103a` hidden 5120 `"H0.5x3"` | 16172 to 32768 | 16172, 16231, 16384, 32768 | GB300, 152 SMs | x0.820-0.875 | x0.984 / x1.020 |

*Table: the token windows of the router weight gradient (eager launches and CUDA graph replays, one long-lived process
and a fresh process per token count; the resident kernel holds 8, 16 or 32 SMs).  The `"waves"` rows are the FI-main
`"waves"` plan (same kernel text, measured on this package's host path).  On GB200 the rows from 10240 tokens up ran at
the GPU's software power cap: after 21 of their 24 runs `nvidia-smi` read `clocks_throttle_reasons` 0x4, at SM clocks of
1297-1912 MHz (2062 MHz maximum); the arms of each row were measured interleaved, in paired groups.  Inside the
head/tail windows cuBLAS FP32 in `CUBLAS_FP32_EMULATED_BF16X9_MATH` (the benchmark's `bf16x9` arm) took x1.455-1.565
(GB200 hidden 4096), x1.516-1.594 (GB200 hidden 5120) and x1.534-1.610 (GB300) of the partition's time: at least x1.455.
The head/tail windows hold only on the SM count of their part (`ROUTER_SM_COUNT`), the `"waves"` windows on every SM
count where `"waves"` applies (`router_waves`); every other token count and SM count runs the deterministic plan.
Hidden 5120 on `sm_100a` / `sm_103a` ran `"waves"` at every token count before these windows: below 64 tokens the
`"waves"` windows keep it without a measurement; between the two windows and above 32768 tokens the row now runs the
deterministic plan -- there `"waves"` was more than 3 % slower than that plan next to a resident kernel at every
measured token count from 224 to 10240 (x1.046-1.156 on GB200 at 224 to 12288 tokens, x1.059-1.167 on GB300 at 224 to
10240 tokens), and above 32768 tokens nothing was measured.*

`prepare_*` and `plan_*` take `splits` (and, for K1, `block_n`) to force a plan, for tests and measurements; the K2
forward takes at most two splits.  `benchmarks/bench_cake_dense_projection_gemm.py` prints, per row, the
`rule_key` / `rule_value` of the plan it measured, in the format of these tables.

## Workspace

A K1 launch with more than one split uses an FP32 slot workspace (`tiles * num_splits * 256 * block_n` elements,
zero at creation: the last split of a partial tile reads the padding columns no split writes, and discards them) and
the ticket counters (`tiles * 512` words, zero at creation, zero again after every complete launch).  A prepared
launch owns both; the convenience path keeps them in an arena per (device, stream), so concurrent launches on
different streams never share them.  After an aborted launch, call `PreparedAtomicGemm.reset_counters()` (or
`clear_convenience_caches()`).  A K2 launch with part tiles zeroes `out` on the launch stream (one memset) between
the split and the program; prepared launches of both programs allocate nothing and can be captured in a CUDA graph.
A prepared K2 launch binds everything but the stream when prepared (its FFI arguments are converted once), so a call
issues its launches back to back.

## Benchmark

`benchmarks/bench_cake_dense_projection_gemm.py` times every row of the deterministic family's benchmark with the
arms `det` (the deterministic package's prepared launch), `atomic` (`deterministic=False`), `atomic_p1` (the same
program at one split), `cublas` (`torch.matmul`), `cublaslt` (a cuBLASLt heuristic sweep) and `bf16x9` (cuBLAS FP32
in `CUBLAS_FP32_EMULATED_BF16X9_MATH`, the router rows), in paired groups, eager and in CUDA graphs; `--resident`
re-times `det` and `atomic` next to a kernel holding `--resident-sms` SMs on another stream.

## Names

This import path held the dense projection GEMM package until #8046 renamed it to `dense_projection_det_gemm`; #8046
left it as an alias of the renamed modules.  The names keep their spelling and, with the default, their behavior:

| name before #8046 (`dense_projection_gemm.cake_backend`) | deterministic package (`dense_projection_det_gemm.cake_backend`) | this path |
|---|---|---|
| `dense_projection_gemm` | `dense_projection_det_gemm` | `dense_projection_gemm(..., deterministic=True)` |
| `projection_forward`, `projection_dgrad`, `projection_wgrad` | `projection_det_forward`, `projection_det_dgrad`, `projection_det_wgrad` | the same names with `deterministic=` |
| `prepare_dense_projection_gemm`, `prepare_projection_wgrad` | `prepare_dense_projection_det_gemm`, `prepare_projection_det_wgrad` | the same names with `deterministic=` |
| `plan_dense_projection_gemm` | `plan_dense_projection_det_gemm` | the same name with `deterministic=` |
| `router_fp32_gemm`, `router_forward`, `router_dgrad`, `router_wgrad` | `router_fp32_det_gemm`, `router_det_forward`, `router_det_dgrad`, `router_det_wgrad` | the same names with `deterministic=` |
| `prepare_router_fp32_gemm`, `plan_router_fp32_gemm` | `prepare_router_fp32_det_gemm`, `plan_router_fp32_det_gemm` | the same names with `deterministic=` |
| `split_fp32_to_bf16x3_` | `split_fp32_to_bf16x3_det_` | the same name with `deterministic=` (the deterministic stage in both modes) |
| `instance_symbol`, `router_symbol` | `instance_symbol`, `router_symbol` | the same names with `deterministic=` |
| `GemmPlan`, `PreparedGemm` | `DetGemmPlan`, `PreparedDetGemm` | read through; `deterministic=False` plans `AtomicGemmPlan`, `PreparedAtomicGemm` |
| `RouterPlan`, `PreparedRouterGemm` | `RouterDetPlan`, `PreparedRouterDetGemm` | read through; `deterministic=False` plans `AtomicRouterPlan`, `PreparedAtomicRouterGemm` |
| every other module name (`ROW_RULES`, `stream_k_plan`, `wgrad_views`, ...) | the same name | read through to the deterministic package |
| program templates `dense_proj_gemm_*`, `router_fp32_gemm_<layout>`, `split_fp32_bf16x3` | `dense_proj_det_gemm_*`, `router_fp32_det_gemm_<layout>`, `split_fp32_det_bf16x3` | this package's templates are new: `dense_proj_gemm_atomic_nn_n<64\|128\|256>[_f32]_t`, `router_fp32_gemm_atomic_{kk,nn_t,nn_t_ht_c3_nreg}` |

*Table: the names before #8046, their deterministic names, and what this path offers
(`cake_backend.DETERMINISTIC_NAMES`).*

| name (`flashinfer.experimental.dense_projection_gemm`) | default | `deterministic=False` |
|---|---|---|
| the entry points of the table above | the deterministic package's `*_det_*` object | this package's schedule (routing table above) |
| `cake_backend.clear_convenience_caches` | clears the caches and arenas of both packages | -- |
| `cake_jit.{MODULES, KERNELS, ARG_PLANS}` | this package's programs only | -- |
| `cake_jit.select_module`, `cake_jit.load_cake_dense_projection_gemm_module` | a deterministic template or program resolves through the deterministic package | -- |
| `cake_build` | this package's loader (byte-identical to every Cake package's) | -- |

*Table: the names of this path and what they reach.*

A module-level assignment through this path (for example `CONVENIENCE_CACHE_SIZE`) does not reach the deterministic
package; set it there.

## Layout of this package

| file | content |
|---|---|
| `__init__.py` | imports nothing; `cake_backend`, `cake_jit`, `cake_build` load on first attribute access |
| `cake_backend.py` | entry points, mode resolution, routing, the planners, rule tables and the convenience state of this package's programs |
| `cake_jit.py` | the registry of this package's programs (`ARG_PLANS`, `MODULES`, `KERNELS`) and the loader |
| `cake_build.py`, `build.json`, `csrc/cake_dense_projection_gemm/` | the `cake-build/1` loader, manifest and generated sources of this package's programs (written by the Cake exporter) |

The deterministic package is resolved as this package's sibling on first use; the package never imports `flashinfer`
by absolute name.  A consumer that vendors this package vendors the deterministic package too and binds it with
`cake_backend.use_deterministic_backend(<its cake_backend module>)`.
