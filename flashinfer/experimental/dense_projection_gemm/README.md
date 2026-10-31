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
| `deterministic=False` | this package's programs on the rows their planners take (below); the deterministic programs for every other row and operation |
| `deterministic=False` under `torch.use_deterministic_algorithms(True)` | `RuntimeError`; with `warn_only=True` a `UserWarning` and the deterministic path |

*Table: what a call runs, by its `deterministic` argument.*

| operation | `deterministic=False` |
|---|---|
| K1 forward (`projection_forward`, `kk` layouts) | deterministic program (whole output tiles; a recomputed forward reproduces its result) |
| K1 input gradient (`projection_dgrad`, `kn` layouts) | deterministic program (whole output tiles) |
| K1 weight gradient (`projection_wgrad`, `nn` layouts) | split-K over the ragged token count (`k1_plan`): every split writes its FP32 partial tile to a workspace slot; the last split of a tile to arrive (a per-tile ticket counter) adds the other slots in arrival order and rounds once into `out`, then returns the counter to zero |
| K2 forward (`router_forward`, `kk`) | stream-K over the K blocks (`router_units`): whole tiles store the one-split result, tiles split between two units TMA reduce-add into `out` (zeroed before the program, on the same stream); at most two partials per element, so the result repeats bit for bit (`0 + a + b == 0 + b + a`) and recomputed router logits and top-k match |
| K2 input gradient (`router_dgrad`, `kn`) | deterministic program (one split) |
| K2 weight gradient (`router_wgrad`, `nn_t`) | stream-K as the forward, any number of partials per element (`router_units`) |
| BF16x3 split (`split_fp32_to_bf16x3_`) | the deterministic stage (exact) |

*Table: the routing of `deterministic=False` (`cake_backend.ROUTES`).*

Run to run, the K1 weight gradient with two splits and the K2 forward repeat bit for bit (two partials per
element); a K1 plan with three or more splits and the K2 weight gradient vary in the last bits.  Every result stays
within the deterministic package's accuracy budgets.

## Planners and rule tables

A planner decides, per row and device, whether `deterministic=False` runs this package's program and with which
parameters; a row it leaves runs the deterministic program (`plan_dense_projection_gemm` /
`plan_router_fp32_gemm` say which).

| table | key | value |
|---|---|---|
| `K1_SPLIT_RULES` | `(arch, L, M, N, out_dtype)`: the transposed-store launch `out[l, n, m]` (`M` = the stored inner extent, `d_in` of a weight gradient; `N` = its rows, `d_out`), `out_dtype` `"bf16"` / `"f32"` | `(block_n, num_splits)` (this package's program) or `None` (the deterministic program) |
| `K1_MIN_SM_COUNT` | `arch` | the fewest SMs for this package's K1 programs (`sm_107a`: 216) |
| `ROUTER_RULES` | `(arch, layout, hidden, experts)`: `layout` `"kk"` (forward, `hidden = K`) or `"nn_t"` (weight gradient, `hidden` = the launch's `M`), `experts` = the launch's `N` | a partition spec (`"forward"`, `"pairs"`, `"tiles"`, `"S<n>"` or a unit count) or `None` (the deterministic program) |

*Table: the rule tables (empty as shipped; an entry replaces the planner for its row).*

| architecture | K1 weight gradient | K2 forward | K2 weight gradient |
|---|---|---|---|
| `sm_100a` | `d_out <= 128`, `d_in <= 6144`, `L == 1` and `2 * ceil(d_in / 256) * ceil(d_out / 64) <= sm_count // 2`: `(64, 2)` | one unit per CTA pair, at most one per tile | one unit per CTA pair; `hidden >= 6144`: deterministic program |
| `sm_103a` | `d_out <= 128`, `d_in <= 6144`, `L == 1`: two splits of 64-wide tiles (`d_out <= 64` or when they fit one wave of CTA pairs) else of 128-wide tiles, if they fit one wave | as `sm_100a` | one unit per CTA pair |
| `sm_107a` | `K1_SM107A_TABLE` (22 measured `(d_in, d_out, out_dtype)` rows), `sm_count >= 216` only | as `sm_100a` | one unit per CTA pair |

*Table: the planners (`K1_PLANNERS`, `router_default_spec`).  Every K1 choice also needs `num_splits * 64 <=
T + 63` (shorter token counts run the deterministic program).*

`prepare_*` and `plan_*` take `splits` (and, for K1, `block_n`) to force a plan, for tests and measurements; the K2
forward takes at most two splits.  `benchmarks/bench_cake_dense_projection_gemm.py` prints, per row, the
`rule_key` / `rule_value` of the plan it measured, in the format of these tables.

## Workspace

A K1 launch with more than one split uses an FP32 slot workspace (`tiles * num_splits * 256 * block_n` elements, no
initialization) and the ticket counters (`tiles * 512` words, zero at creation, zero again after every complete
launch).  A prepared launch owns both; the convenience path keeps them in an arena per (device, stream), so
concurrent launches on different streams never share them.  After an aborted launch, call
`PreparedAtomicGemm.reset_counters()` (or `clear_convenience_caches()`).  A K2 launch with part tiles zeroes `out` on
the launch stream (one memset) between the split and the program; prepared launches of both programs allocate nothing
and can be captured in a CUDA graph.  A prepared K2 launch binds everything but the stream when prepared (its FFI
arguments are converted once), so a call issues its launches back to back.

## Benchmark

`benchmarks/bench_cake_dense_projection_gemm.py` times every row of the deterministic family's benchmark with the arms
`det` (the deterministic package's prepared launch), `atomic` (`deterministic=False`), `atomic_p1` (the same
program at one split), `cublas` (`torch.matmul`), `cublaslt` (a cuBLASLt heuristic sweep) and `bf16x9` (cuBLAS FP32
in `CUBLAS_FP32_EMULATED_BF16X9_MATH`, the router rows), in paired groups, eager and in CUDA graphs.

## Names

This import path held the dense projection GEMM package until #8046 renamed it to `dense_projection_det_gemm`; #8046
left it as an alias of the renamed modules.  The names keep their spelling and, with the default, their behavior:

| name (`flashinfer.experimental.dense_projection_gemm`) | default | `deterministic=False` |
|---|---|---|
| `cake_backend.{dense_projection_gemm, projection_forward, projection_dgrad, projection_wgrad, prepare_dense_projection_gemm, prepare_projection_wgrad, plan_dense_projection_gemm, router_fp32_gemm, router_forward, router_dgrad, router_wgrad, prepare_router_fp32_gemm, plan_router_fp32_gemm, split_fp32_to_bf16x3_, instance_symbol, router_symbol}` | the deterministic package's `*_det_*` object (`cake_backend.DETERMINISTIC_NAMES`) | this package's schedule (table above) |
| `cake_backend.{GemmPlan, PreparedGemm, RouterPlan, PreparedRouterGemm}` and every other name of the module before #8046 (`ROW_RULES`, `stream_k_plan`, `wgrad_views`, ...) | read through to the deterministic package | -- |
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

The deterministic package is resolved as this package's sibling on first use; the package never imports
`flashinfer` by absolute name.  A consumer that vendors this package vendors the deterministic package too and binds
it with `cake_backend.use_deterministic_backend(<its cake_backend module>)`.
