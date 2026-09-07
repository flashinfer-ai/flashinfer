# SM90 (Hopper) CuTe-DSL MoE backend

BF16/FP16 unquantized fused MoE for SM90, written in the CuTe DSL
(`flashinfer/fused_moe/cute_dsl/hopper/` + the `sm90_*` host wrappers).
The contiguous-grouped implementation is exposed through the Python API
(`cute_dsl_fused_moe_bf16` / `CuteDslBf16MoEWrapper`).

## 1. Scope

- Dtypes: BF16 and FP16 activations/weights, f32 accumulation. No
  quantization. Output dtype = input dtype.
- Activation: gated SiLU (SwiGLU), fused into GEMM1's epilogue.
- Routing: pre-routed only (`token_selected_experts` int32 global ids +
  caller-normalized `token_final_scales` fp32); no in-kernel router;
  `top_k` is a compile-time constant.
- EP: `num_local_experts` + `local_expert_offset` select the local expert
  shard (tokens routed entirely outside it contribute zeros). TP is
  shape-only (callers pass per-rank weight shards).
- Shapes: `hidden % 64 == 0` (also GEMM1's reduction dim — no tile-32
  fallback there), `2I % 64 == 0`, `I % 32 == 0` (interleave granularity;
  GEMM2 falls back to tile-k 32 when `I % 64 != 0`); `num_tokens == 0`
  supported.
- Execution: CUDA-graph capturable; PDL on by default (`enable_pdl`);
  fused finalize (default) is atomic and not bitwise-reproducible,
  `use_fused_finalize=False` selects the deterministic two-stage path.

## 2. Execution model: contiguous-grouped MoE in 3 device ops

```
moe_sort (C++ JIT routing)  ->  index maps only, no data movement
GEMM1: gather + grouped GEMM + SwiGLU  ->  intermediate [permuted_m, I]
GEMM2: grouped GEMM + fused finalize   ->  out [num_tokens, hidden]
```

The expert dimension is flattened into M: rows are expert-sorted and each
expert's range is padded up to `tile_size` (= `moe_sort`'s `tile_tokens_dim`).
`moe_sort` produces `tile_idx_to_expert_idx`, `tile_idx_to_mn_limit` (first
invalid row per tile), `permuted_idx_to_expanded_idx` (gather map,
`token*top_k + slot`; garbage on padding rows),
`num_non_exiting_tiles` (device-side valid-tile count), and the inverse
`expanded_idx_to_permuted_idx` (consumed by the deterministic path's
`moe_unpermute`).

Properties:

- **CUDA-graph safe**: the grid is sized for the maximum tile count (a
  host-side function of `num_tokens` only); tiles at
  `m_tile >= num_non_exiting_tiles[0]` exit on device. Nothing in the
  launch depends on routing data.
- **No tensormap updates**: expert weights live in one dense `[E, N, K]`
  allocation and the expert index is the TMA L-coordinate.
- **The permute is never materialized**: GEMM1 gathers A rows through the
  index map; GEMM2's finalize scatters rows back to token space.
- Padding rows inside valid tiles are computed and stored (garbage) and
  masked by `mn_limit` exactly where they would escape (GEMM2's scatter).

Host-side stream choreography (fused mode): the finalize accumulates into
`out`, so the host zeroes it on an aux stream overlapped with GEMM1
(`moe_output_memset_inplace`, a `cudaMemsetAsync` binding — cheaper to
launch than `Tensor.zero_`), with event fork-join only (`main_event` ->
aux memset -> `memset_event` -> main waits before GEMM2). No
`Tensor.record_stream` — it is illegal under CUDA-graph capture and
redundant given the event join.

**PDL** (`enable_pdl`, default on): both GEMMs launch with Programmatic
Dependent Launch (`use_pdl=` launch attribute) so each kernel's prologue
(descriptor prefetch, SMEM/pipeline setup) overlaps its predecessor's
tail. Placement: GEMM1 issues one `griddepcontrol_wait()` on all threads
after the pipeline-init barrier, before its first gmem read (the moe_sort
outputs). GEMM2 hoists the wait into the **load warp only**, immediately
before the A TMA loop — A (GEMM1's output) is the one read that must
wait; the tile maps and `num_non_exiting_tiles` are already ordered
transitively through GEMM1's own dependency on moe_sort, so the meta warp
and consumers start their prologues without waiting. Each kernel ends
with one `griddepcontrol_launch_dependents()`: GEMM1 after the TMA-store
`producer_tail`, GEMM2 after every thread has drained its own
scatter bulk-groups (`cp_async_bulk_wait_group(0)` past the persistent
loop — the per-tile drain is deferred to the next tile's sC reuse). The
device PTX is emitted unconditionally (a no-op without the launch
attribute); only the launch attribute is gated, and `enable_pdl` is part
of every compile cache key. `moe_sort` launches PDL-off and the
deterministic path's `moe_unpermute` PDL-on.
The inter-GEMM `memset_event` wait does not defeat the overlap: the aux
zeroing completes during GEMM1, so GEMM2's early launch is gated only by
GEMM1's trigger.

## 3. GEMM1 — `Sm90ContiguousGatherGroupedGemmActFusionKernel`

`C[r, 0:I] = silu(gate) * up` where `[up | gate] = x[token(r)] @ w1[e].T`.
`w1[e] [2I, hidden]` is the expert's fc1 pack: the model's two separate
projection matrices (gate and up — w1/w3 in Llama-style naming), row-
concatenated and 32-column interleaved (§6), so one GEMM with N = 2I
produces both projections and the epilogue gates them in registers.

Warp specialization (persistent, `StaticPersistentTileScheduler` per
warpgroup — no scheduler warp; the tile maps are read directly from GMEM):

- **Producer warpgroup (128 threads, 40/56 regs)**: all four warps gather A
  with cp.async through the permute map (thread t covers row-group `t//8`,
  16 B chunk `t%8`; rows past `mn_limit` predicated off); warp 0's elected
  lane TMA-loads B with the expert index as L-coordinate.
- **1–2 consumer warpgroups (232 regs)**: WGMMA `MmaF16BF16Op`
  (m64·n·k16, f32 accumulators) + the gated epilogue. Two consumer
  warpgroups iff `tile_m > 64 and tile_n > 128`.

Pipelines: A on `PipelineCpAsync`, B on `PipelineTmaAsync` (consumers wait
on both per K-tile), C on `PipelineTmaStore`. SMEM: K-major bf16 K-tile 64
-> `K_SW128` swizzle; A/B stage count fills SMEM after a fixed 4-stage
epilogue ring (~6 stages at 128x128, 4 at 128x256).

Gated epilogue: w1 is pre-interleaved at **32-column up/gate granularity**
(`[up 0:32 | gate 0:32 | up 32:64 | ...]`), so an even accumulator subtile
is always "up" and the following odd subtile its "gate"; the epilogue emits
`silu(gate)*up` in f32 and TMA-stores `N/2` output columns. Requires
`tile_n % 64 == 0`: the 32-column interleave makes each complete up/gate
pair 64 columns.

## 4. GEMM2 — `Sm90ContiguousGroupedGemmFinalizeFusionKernel`

`out[token(r)] += scale(r) * (intermediate[r] @ w2[e].T)` (fused mode) or
`out[expanded(r)] = intermediate[r] @ w2[e].T` (deterministic mode).

- A (the intermediate) is contiguous -> plain TMA on a single A+B pipeline;
  expert index again as B's L-coordinate. `tile_k` is 64, or **32 when the
  per-rank K is not a multiple of 64** and on prefill tiles at `K >= 384`,
  where the halved stage footprint under the full-tile sC doubles the A/B
  pipeline depth (§5 table).
- A **meta warp** (producer warpgroup, warp 1; 2-stage `PipelineAsync`)
  prefetches per-row `(output_row, scale)` into SMEM one tile ahead:
  fused mode reads `token_final_scales[token, slot]`, deterministic mode
  uses scale 1.0 and the expanded row index. Padding entries are handled
  branchlessly.
- Finalize epilogue: rows are scaled in registers, staged in a row-padded
  linear SMEM tile (16 B-aligned row starts), then scattered **one row per
  thread**: `cp.reduce.async.bulk...add.noftz.{bf16|f16}` / `.f32` (fused —
  the top-k reduction happens in L2/DRAM) or `cp.async.bulk` copies
  (deterministic; a fixed-order `moe_unpermute` then applies scales).

Numerics contract: f32 accumulation in both GEMMs; the fused finalize's
top-k combine accumulates **in the output dtype** (one output-dtype
rounding per route on top of the bf16 intermediate hand-off), and its
bulk-add order is tile-schedule dependent, so results are **not bitwise
reproducible across runs**.
`use_fused_finalize=False` is the alternative: f32 fixed-order combine in
`moe_unpermute` (one final rounding, bitwise-reproducible, ~1 extra
kernel). The deterministic path also avoids the repeated output-dtype
rounding performed by the fused route combine.

## 4.5 Cluster multicast policy

Both kernels support cluster multicast, but only GEMM2 carries it as an
autotuned tactic axis:

- **GEMM1 (2,1) same-expert B-multicast: fixed off.** Expert tiles do not
  necessarily occur in adjacent CTA pairs, and clustering constrains CTA
  scheduling even when a pair cannot share an expert weight tile.
- **GEMM2 (1,2) A-multicast is tuned independently of tile and raster
  order.** It halves intermediate re-reads across paired N-tiles but also
  constrains CTA scheduling. The untuned `-1` fallback uses a shape policy.
  A `(1,2)` candidate is excluded before profiling unless the GEMM2 N-tile
  count is even and every host/kernel can-implement condition passes, so it
  is never timed as a silent `(1,1)` alias.

## 4.6 GEMM2 raster order

The finalize scatter-RMW can dominate L2 traffic at small reduction
dimensions. M-major rasterization pins each concurrent CTA wave to one
output-column slice, confining the RMW working set to an L2-resident band;
the cost is re-reading each A tile once per N tile. Raster order is an
independent GEMM2 tactic axis. The untuned `-1` fallback uses M-major for
large output working sets with inexpensive A re-reads and N-major elsewhere.

## 5. Tile selection, tuning, and compilation

Tactic and fallback selection (`sm90_fused_moe.py`):

| axis | values | selection |
|---|---|---|
| tile_size (= `moe_sort` tile) | 64, 128 | 64 below 64 avg rows/local expert (halves decode padding waste); tiny reductions (I < 192) switch to 128 at 16 rows/local expert to amortize GEMM2's fixed per-tile cost |
| GEMM1 tile_n | (256, 192, 128, 64) at tile_size 128; (128, 64) at 64 | largest divisor of 2I |
| GEMM2 tile_n | (256, 128, 64) / (128, 64) | largest divisor of hidden |
| GEMM2 tile_k | 64; 32 when K % 64 != 0, and on prefill tiles at I >= 384 (doubled pipeline depth: +2..6%) | shape-derived |
| GEMM2 raster | N-major, M-major | independently autotuned; §4.6 defines the fallback |
| GEMM1 cluster | (1,1) | fixed; (2,1) remains a validated low-level option |
| GEMM1 raster | N-major | fixed; M-major remains a low-level option on the GEMM-level wrapper |
| GEMM2 cluster | (1,1), (1,2) | independently autotuned; illegal (1,2) topologies filtered before profiling |

Dispatch goes through the FlashInfer AutoTuner:
`cute_dsl_fused_moe_bf16` routes through
`AutoTuner.choose_one` with `CuteDslFusedMoESm90Runner`, whose tactic space
is `Sm90MoeTactic(tile_size, gemm1_tile_n, gemm2_tile_n, gemm2_tile_k,
gemm2_cluster_shape_mn, gemm2_raster_along_m)` (top-2 legal N tiles per
GEMM, tile_k pinned to the shape heuristic, and the legal cross-product of
the two independent GEMM2 cluster/raster axes). Under the `autotune`
context every tactic
is profiled and the per-bucket winner cached; otherwise the cached winner —
or the heuristic auto-selection as the default tactic — dispatches.
Explicit tile / GEMM2 cluster / raster / buffer keyword overrides bypass the
tuner.

**Compilation and reuse**: each low-level host module owns a process-local
dictionary of compiled callables. On the first real launch of a specialization,
the wrapper calls `cute.compile` with the actual CuTe pointers, problem
dimensions, and current `cuda.CUstream`; subsequent launches reuse that callable.
Pointer dtypes and tactic fields form the key, while dimensions, pointer values,
and the stream remain runtime arguments. There is no separate dispatch lattice
or SM90-specific persistent object cache.

An autotune pass naturally compiles every candidate that it profiles. A process
that loads an existing AutoTuner winner compiles only the selected specialization
on its first launch. Applications must therefore warm the selected shapes and
tactics before CUDA-graph capture or latency-sensitive serving.

**Tactic profiling** uses CUDA-graph replay windows
(`TuningConfig.use_cuda_graph`): without it the per-call host path
(moe_sort launcher, aux-stream events) dominates decode-size measurements
and the argmin ranks noise. The local timing feeds the optional
cross-rank reduction. The graph is captured once per tactic. The heuristic
auto-selection competes as an explicit candidate (tactic `-1`), so a tuned
winner never ranks below the default dispatch in the same measurement
session.

## 6. Weight layout and Python API

The kernel's `w1 [E, 2I, K]` is up/gate-interleaved at 32 columns.
Callers own the repack — a trivial reshape of `[gate; up]`-concatenated
weights, performed once at weight load. The in-tree reference
implementation (used by tests) is the module-level, non-exported
`interleave_up_gate_sm90` in
`sm90_contiguous_gather_grouped_gemm_act_fusion.py`.

Python API: `cute_dsl_fused_moe_bf16` (flat function) and
`CuteDslBf16MoEWrapper` (instance holds the static config; `run()`).
The public API carries an `@flashinfer_api` trace template
(`cute_dsl_fused_moe_bf16`).

## 7. Testing

| tier | what | where |
|---|---|---|
| kernel + e2e unit | tiles/clusters/boundary tiles; bf16+fp16 e2e, auto-select, EP shards (incl. all-routed-outside-shard), autotune (tactic profiling and cached-winner recall), process-local compile reuse, deterministic mode (bitwise), CUDA-graph capture/replay, tiny/empty batch, fail-fast bad inputs | `tests/moe/test_cute_dsl_bf16_gather_grouped_gemm.py`, `test_cute_dsl_bf16_grouped_gemm_finalize.py`, `test_cute_dsl_bf16_moe.py` |

## 8. Performance

Measurements use H200. Each result is the median of three alternating
backend-pair ratios (AB/BA/AB). A result is reported only when the
population CV across rounds is at most 5% for both backends.

Both backends run through `benchmarks/flashinfer_benchmark.py`: CuTe-DSL as
the `cute_dsl_bf16_moe` routine, the baseline as the `cutlass_fused_moe`
routine (`base` variant — unquantized BF16) on identical tensors and routing.
Each backend is autotuned once per suite and the winners are stored to a
config file (`--autotune_cache`); all measured rounds replay the stored
tactics with autotune disabled, and the config file's fingerprint is checked
before every measured leg, so between-round variance is execution-only.
Each per-round value is the median of 50 CUPTI timings after 10 dry runs,
with a cold L2 before every timing. Cells are baseline latency divided by
CuTe-DSL latency, so values greater than 1 mean CuTe-DSL is faster.

Model-derived workloads use routed-expert geometries from the models'
published configs — Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3-Next-80B-A3B,
GLM-4.5-Air, Kimi-K2, DeepSeek-V3, and Mixtral-8x7B — at TP in {1, 4} and
unexpanded input-token counts T in {1, 256, 1024, 4096, 16384}. Cells are
the speedup over the baseline (values greater than 1 mean CuTe-DSL is
faster):

| model (h / I global / E / top_k) | tp -> I/rank | T=1 | 256 | 1024 | 4096 | 16384 |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B (2048/768/128/8) | 1 -> 768 | 1.38x | 1.02x | 1.02x | 1.27x | 1.31x |
| | 4 -> 192 | 1.82x | 1.24x | 1.34x | 1.75x | 1.90x |
| Qwen3-235B-A22B (4096/1536/128/8) | 1 -> 1536 | 1.10x | 1.00x | 0.88x | 1.08x | 1.14x |
| | 4 -> 384 | 1.48x | 1.05x | 1.09x | 1.35x | 1.38x |
| Qwen3-Next-80B-A3B (2048/512/512/10) | 1 -> 512 | 1.70x | 0.98x | 1.03x | 1.17x | 1.48x |
| | 4 -> 128 | 2.30x | 1.17x | 1.26x | 1.73x | 1.82x |
| GLM-4.5-Air (4096/1408/128/8) | 1 -> 1408 | 1.14x | 1.01x | 0.89x | 1.04x | 1.13x |
| | 4 -> 352 | 1.48x | 1.09x | 1.07x | 1.11x | 1.11x |
| Kimi-K2 (7168/2048/384/8) | 1 -> 2048 | 1.08x | 1.00x | 1.03x | 1.02x | 1.12x |
| | 4 -> 512 | 1.35x | 1.01x | 1.04x | 1.12x | 1.18x |
| DeepSeek-V3 (7168/2048/256/8) | 1 -> 2048 | 1.08x | 1.00x | 1.05x | 0.86x | 1.03x |
| | 4 -> 512 | 1.33x | 1.02x | 1.10x | 1.04x | 1.21x |
| Mixtral-8x7B (4096/14336/8/2) | 1 -> 14336 | 1.07x | 0.85x | 0.86x | 0.89x | 0.95x |
| | 4 -> 3584 | 1.18x | 0.90x | 1.07x | 1.07x | 1.05x |

All 70 cells pass the variance gate; their geo-mean speedup is **1.16x**,
and CuTe-DSL is faster in 59. The per-model geo-mean ranges from **0.98x**
for Mixtral-8x7B to **1.41x** for Qwen3-Next-80B-A3B. The advantage grows
with TP (smaller per-rank I) and is largest at T=1 decode (up to 2.30x);
the losses concentrate in Mixtral's very large I/rank at mid batch sizes
and in the T=1024 band of the h=4096 models at TP1. Across cells,
CuTe-DSL's between-round CV has median 0.12%, p95 2.45%, and maximum
4.23%; the baseline's has median 0.14%, p95 1.13%, and maximum 4.09%.

## 9. Limitations

- **No unified `MoELayer` API registration**: the backend is served through
  the direct Python API only.
- The fused finalize is not bitwise-reproducible across runs (bulk-add
  order is tile-schedule dependent); `use_fused_finalize=False` is the
  reproducible mode.
