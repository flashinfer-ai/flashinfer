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
- Shapes: `hidden % 64 == 0` (GEMM1's reduction moves whole 64-element K
  tiles), `2I % 64 == 0`, `I % 32 == 0` (interleave granularity; GEMM2's K
  tail is zero-filled by TMA); `num_tokens == 0` supported.
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

**PDL** (`enable_pdl`, default on) allows dependent kernel setup to overlap
its predecessor's tail. GEMM1 signals dependents after its epilogue and
TMA-store drain. GEMM2 waits before reading the intermediate and signals
its dependents after draining the output scatters. Routing dependencies
and the auxiliary-stream output initialization remain ordered.

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

**Persistent-walk swizzle** (`swizzle_size`, tuned over 1/8/16): the
persistent scheduler walks the (M-tile, N-tile) grid N-fast, so an
expert's weight tile is streamed from HBM once per M-tile row. A swizzle
of `s` groups the walk into blocks of `s` M-tiles, so the CTAs of one
wave share the expert's B tiles through L2 and fetch them once per block.
It is a pure schedule reorder (bitwise-identical output), and the tuner
profiles it only where it can pay: batches with at least eight routed
rows per expert (an expert spans several M-tiles) and expert weights that
fit in L2 (below 48 MiB per expert, or at most 16 experts).

## 4. GEMM2 — `Sm90ContiguousGroupedGemmFinalizeFusionKernel`

`out[token(r)] += scale(r) * (intermediate[r] @ w2[e].T)` (fused mode) or
`out[expanded(r)] = intermediate[r] @ w2[e].T` (deterministic mode).

- A (the intermediate) is contiguous -> plain TMA on a single A+B pipeline;
  expert index again as B's L-coordinate. The K tile is 64 elements (one
  SW128 atom, four WGMMA K-steps per stage); a partial last K tile is
  zero-filled by TMA on both operands.
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
  constrains CTA scheduling. The untuned `-1` fallback keeps `(1,1)`. A
  `(1,2)` candidate is profiled only at per-rank `I >= 192` (the per-problem
  filter, §5; below it the re-read it halves is already small) and is excluded
  unless the GEMM2 N-tile count is even. Unsupported clusters are rejected
  rather than silently downgraded.

## 4.6 GEMM2 raster order

The finalize scatter-RMW can dominate L2 traffic at small reduction
dimensions. M-major rasterization pins each concurrent CTA wave to one
output-column slice, confining the RMW working set to an L2-resident band;
the cost is re-reading each A tile once per N tile. Raster order is a GEMM2
tactic field: the untuned `-1` fallback keeps N-major, and the tuner
profiles M-major on tile 128 with per-rank `I <= 384` (the per-problem
filter, §5), the only region where it won (up to 19.5% on a tiny-I
16k-token prefill).

## 5. Tile selection, tuning, and compilation

Both GEMMs share the routing row tile but tune their other options
independently. Every legal N tile of both GEMMs is a candidate; only
scheduling options with little expected benefit are pruned. Profiling
replaces the routing with a seeded uniform top-k draw over the global
experts, so tactics are ranked under the uneven expert loads of real
routing rather than a balanced assignment.

Autotuning compares candidates, including the fixed default, using CUDA
graph replay to exclude host launch overhead. Winners are cached per token
bucket. Calls use the cached winner or the fixed default unless a tactic
is explicitly selected. Tactic validation belongs to candidate selection,
keeping dispatch lightweight: a persisted winner of another tactic schema
fails at its first dispatch with a `ValueError` naming the expected
structure, so tuning caches are re-tuned after a schema change rather
than validated on every call. Candidate lists depend only on the problem
shapes and the global expert count, so expert-parallel ranks that tune
together profile identical sequences.

Compiled kernels are reused within a process. Loading saved tuning results
still requires compiling the selected kernels, so applications must warm
relevant shapes and tactics before CUDA-graph capture or latency-sensitive
serving.

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

Tests cover numerical correctness across dtypes, tiles, and expert shards;
empty and partial batches; deterministic output; CUDA-graph replay; and
autotuning with persisted results. Host-side tests cover tactic legality,
pruning boundaries, and dispatch behavior.

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
| Qwen3-30B-A3B (2048/768/128/8) | 1 -> 768 | 1.39x | 1.02x | 1.15x | 1.25x | 1.27x |
| | 4 -> 192 | 1.88x | 1.25x | 1.51x | 1.76x | 1.90x |
| Qwen3-235B-A22B (4096/1536/128/8) | 1 -> 1536 | 1.11x | 1.00x | 1.08x | 1.09x | 1.14x |
| | 4 -> 384 | 1.50x | 1.05x | 1.26x | 1.25x | 1.43x |
| Qwen3-Next-80B-A3B (2048/512/512/10) | 1 -> 512 | 1.71x | 0.99x | 1.04x | 1.18x | 1.33x |
| | 4 -> 128 | 2.34x | 1.17x | 1.26x | 1.72x | 1.82x |
| GLM-4.5-Air (4096/1408/128/8) | 1 -> 1408 | 1.14x | 1.00x | 1.06x | 1.08x | 1.16x |
| | 4 -> 352 | 1.54x | 1.09x | 1.15x | 1.10x | 1.07x |
| Kimi-K2 (7168/2048/384/8) | 1 -> 2048 | 1.08x | 1.00x | 1.03x | 1.02x | 1.22x |
| | 4 -> 512 | 1.47x | 1.01x | 1.05x | 1.13x | 1.23x |
| DeepSeek-V3 (7168/2048/256/8) | 1 -> 2048 | 1.08x | 1.00x | 1.04x | 0.90x | 1.10x |
| | 4 -> 512 | 1.33x | 1.02x | 1.10x | 1.02x | 1.16x |
| Mixtral-8x7B (4096/14336/8/2) | 1 -> 14336 | 1.06x | 1.06x | 0.95x | 1.00x | 1.06x |
| | 4 -> 3584 | 1.19x | 1.11x | 1.08x | 1.05x | 1.07x |

All 70 cells pass the variance gate; their geo-mean speedup is **1.19x**.
CuTe-DSL is faster in 62 cells, at 1.00x in 5 (Qwen3-235B-A22B,
GLM-4.5-Air, Kimi-K2 and DeepSeek-V3 at TP1 T=256, Mixtral-8x7B at TP1
T=4096) and slower in 3: DeepSeek-V3 at TP1 T=4096 (0.90x), Mixtral-8x7B
at TP1 T=1024 (0.95x) and Qwen3-Next-80B-A3B at TP1 T=256 (0.99x). The per-model geo-mean ranges
from **1.06x** for Mixtral-8x7B to **1.41x** for Qwen3-30B-A3B and
Qwen3-Next-80B-A3B. The advantage grows with TP (smaller per-rank I) and
is largest at T=1 decode (up to 2.34x) and at T >= 4096. Across cells, CuTe-DSL's between-round CV
has median 0.17%, p95 2.20%, and maximum 2.88%; the baseline's has median
0.15%, p95 2.16%, and maximum 4.65%.

## 9. Limitations

- **No unified `MoELayer` API registration**: the backend is served through
  the direct Python API only.
- The fused finalize is not bitwise-reproducible across runs (bulk-add
  order is tile-schedule dependent); `use_fused_finalize=False` is the
  reproducible mode.
