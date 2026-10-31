# Cake dense projection GEMM backend with an explicit numerical mode (SM100 / SM103 / SM107)

Experimental backend for the GLM-5.2 training GEMMs -- the BF16 dense projection GEMM (K1: BF16 operands, FP32
accumulation, BF16 or FP32 output, strided views, ragged token counts, batched head projections) and the FP32
router-gate GEMM through split-BF16x3 tensor-core emulation (K2) -- with a selectable numerical mode.  Tracking issue:
flashinfer-ai/flashinfer#8069.

The deterministic implementation is the package `flashinfer.experimental.dense_projection_det_gemm`
(flashinfer-ai/flashinfer#5677, #7043; see its README for the view contract, the host paths and the per-architecture
programs).  This package keeps its operations, dtypes, accuracy budgets and host paths, and adds a non-deterministic
schedule whose split-K reductions add the partials with unordered FP32 adds.  Only run-to-run bitwise
reproducibility is relaxed.

Both the API and the backend are experimental: no compatibility guarantee.

## Numerical mode

Every entry point of `cake_backend` takes the keyword-only `deterministic: bool = True`.

| call | runs |
|---|---|
| `deterministic=True` (default) | the deterministic package's canonical entry point with the same arguments: the same programs, the same bits, the same errors |
| `deterministic=False` | the non-deterministic schedule for the operations below; the deterministic programs for every other operation |
| `deterministic=False` under `torch.use_deterministic_algorithms(True)` | `RuntimeError`; with `warn_only=True` a `UserWarning` and the deterministic path |

*Table: what a call runs, by its `deterministic` argument.*

| operation | `deterministic=False` |
|---|---|
| K1 forward (`projection_forward`, `kk` layouts) | deterministic program (whole output tiles; a recomputed forward reproduces its result) |
| K1 input gradient (`projection_dgrad`, `kn` layouts) | deterministic program (whole output tiles) |
| K1 weight gradient (`projection_wgrad`, `nn` layouts) | split-K over the ragged token count: FP32 reduce-add into a zero-invariant workspace, the last part rounds once (`k1_split_parts`; a row kept whole runs the deterministic program) |
| K2 forward (`router_forward`, `kk`) | at most two split-K partials reduce-added into `out` (reproducible: `0 + a + b == 0 + b + a`), so recomputed router logits and top-k match |
| K2 input gradient (`router_dgrad`, `kn`) | deterministic program (one split) |
| K2 weight gradient (`router_wgrad`, `nn_t`) | split-K partials reduce-added into `out` (`router_splits`) |
| BF16x3 split (`split_fp32_to_bf16x3_`) | the deterministic stage (exact) |

*Table: the routing of the non-deterministic mode (`cake_backend.ROUTES`).*

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
| `cake_backend.py` | entry points, mode resolution, routing, the planner hooks and the convenience state of the non-deterministic programs |
| `cake_jit.py` | the registry of this package's programs (`ARG_PLANS`, `MODULES`, `KERNELS`) and the loader |
| `cake_build.py`, `build.json`, `csrc/cake_dense_projection_gemm/` | the `cake-build/1` loader, manifest and generated sources of this package's programs (written by the Cake exporter) |

The deterministic package is resolved as this package's sibling on first use; the package never imports
`flashinfer` by absolute name.  A consumer that vendors this package vendors the deterministic package too and binds
it with `cake_backend.use_deterministic_backend(<its cake_backend module>)`.
