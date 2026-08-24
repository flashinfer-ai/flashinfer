# SM90 pull-style FP8 MegaMoE tuning + performance notes

This document collects the performance work on the `sm90_fp8_fp8_bf16_pull_cutedsl` mega
backend: the measured microbenchmark results, the benchmark methodology
behind those numbers, the knob surface as it exists today, and the open
perf levers.  It is the companion to `SKILL.md` (drop-update workflow) and
mirrors the structure of the SM100 tree's `TUNING.md`.

Unless noted otherwise, all measurements were taken 2026-08-23 on a single
H200 node (4x NVIDIA H200 141GB, EP=4) at the kernel drop's DSV4-Pro P03
geometry: **384 experts, top-6, hidden 7168, intermediate 3072
(post-SwiGLU; gate+up 6144), gate_up_clamp 10.0**, tokens-per-rank swept
8..32768 in powers of two (13 points) — the same geometry and knobs as the
kernel team's `moe_hopper_fp8/run_token_sweep_benchmark.py`.  The launch
config per point is the drop's token-bucket heuristic table
(`moe_hopper_fp8/heuristic_config.py`, geometry derived from the kernel
team's 2026-08-19 four-rank H200 sweep at the same vendored kernel
sources, plus the locally added per-bucket `token_back_mode` column from
the 2026-08-23 epi-vs-reuse sweep — see the knob list below).
Raw rows: `benchmark_data/20260823/20260823_090730_mega_sm90_heuristic_both.csv`.

## Microbenchmark results (2026-08-23, heuristic launch configs, max-rank µs)

Two timed series per point — the difference is WHAT each call includes:

- **`compute` — the pure compute path.**  Inputs are staged ONCE, then
  `MegaKernelBackend.compute(output=None)` is timed repeatedly: **fused
  mega kernel (dispatch + FC1 + SwiGLU + FC2 + combine) + standalone
  top-k reduce, zero-copy output** (the result stays in the workspace; no
  staging, no output copy).  This is the number to quote for kernel work
  and to compare against the drop's `mega_us + topk_us`.
- **`e2e` — the full production path.**  `MoEEpLayer.forward` is timed:
  **input validation + bf16→fp8 staging quantization + everything in
  `compute` + output copy** into the caller's tensor.  This is the
  serving-relevant number; it has no drop counterpart.

`e2e` time is therefore always ≥ `compute` time (so
`critical_tflops_e2e` ≤ `critical_tflops_compute`); the gap is the
staging + copy overhead quantified in "e2e overhead" below.  TFLOPS use
the drop's per-rank formula (`routed = tok/rank × topk`,
`flops = 2·routed·hidden·(gateup + downproj)`) over the max-rank
(critical-path) time.

The `token back` column is the per-bucket `token_back_mode` the heuristic
table now selects (`epi` = `epi_warps`, `reuse` = `reuse_dispatch_warps`).

**per_tensor** — peak 841 TFLOPS/rank:

| tok/rank | heuristic config                   | token back | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|:----------:|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB ping-pong M128N16 CGA2x1   |    epi     |      830.8 |    7.6 |    954.6 |        6.6 |
|       16 | swap-AB ping-pong M128N16 CGA1x2   |    epi     |     1248.4 |   10.2 |   1402.7 |        9.0 |
|       32 | non-swap M64N256 CGA1x1            |    epi     |     1664.4 |   15.2 |   1784.3 |       14.2 |
|       64 | swap-AB ping-pong M128N64 CGA1x2   |    epi     |     1974.6 |   25.7 |   2136.9 |       23.7 |
|      128 | swap-AB ping-pong M128N32 CGA1x2   |    epi     |     1824.5 |   55.6 |   1986.0 |       51.1 |
|      256 | swap-AB M256N32 CGA2x1             |    epi     |     2029.5 |  100.0 |   2019.1 |      100.5 |
|      512 | swap-AB M256N64 CGA1x1             |    epi     |     2096.0 |  193.6 |   2283.9 |      177.7 |
|     1024 | swap-AB ping-pong M128N64 CGA1x2   |    epi     |     2151.0 |  377.4 |   2323.7 |      349.3 |
|     2048 | non-swap ping-pong M64N128 CGA2x1  |    epi     |     3043.2 |  533.5 |   3191.9 |      508.6 |
|     4096 | non-swap ping-pong M64N128 CGA2x2  |    epi     |     5081.8 |  638.9 |   5279.4 |      615.0 |
|     8192 | swap-AB ping-pong M128N64 CGA1x2   |    epi     |     8569.0 |  757.9 |   8862.6 |      732.7 |
|    16384 | non-swap M64N256 CGA2x1            |   reuse    |    15895.2 |  817.1 |  16718.2 |      776.9 |
|    32768 | non-swap ping-pong M64N128 CGA2x2  |   reuse    |    30902.9 |  840.6 |  32565.7 |      797.6 |

**blockwise** — peak 568 TFLOPS/rank:

| tok/rank | heuristic config                   | token back | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|:----------:|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB M256N16 CGA2x1             |    epi     |      850.6 |    7.5 |   1077.8 |        5.9 |
|       16 | swap-AB M256N16 CGA1x1             |    epi     |     1251.2 |   10.1 |   1526.0 |        8.3 |
|       32 | swap-AB ping-pong M128N16 CGA1x2   |    epi     |     1673.3 |   15.2 |   1924.9 |       13.2 |
|       64 | swap-AB M256N32 CGA2x1             |    epi     |     1804.0 |   28.1 |   2069.7 |       24.5 |
|      128 | swap-AB M256N16 CGA2x1             |    epi     |     1803.6 |   56.3 |   2086.1 |       48.6 |
|      256 | swap-AB ping-pong M128N32 CGA1x2   |    epi     |     2122.8 |   95.6 |   2372.5 |       85.5 |
|      512 | non-swap M64N128 CGA1x1            |    epi     |     2507.8 |  161.8 |   2712.4 |      149.6 |
|     1024 | non-swap M64N128 CGA2x2            |   reuse    |     3055.7 |  265.6 |   3223.7 |      251.8 |
|     2048 | non-swap M64N128 CGA2x2            |   reuse    |     4337.2 |  374.3 |   4488.9 |      361.7 |
|     4096 | non-swap M64N128 CGA1x1            |   reuse    |     6932.5 |  468.4 |   7256.4 |      447.5 |
|     8192 | non-swap M64N128 CGA2x1            |   reuse    |    11824.6 |  549.2 |  12405.7 |      523.5 |
|    16384 | non-swap M64N128 CGA1x2            |   reuse    |    22854.6 |  568.3 |  24392.3 |      532.5 |
|    32768 | non-swap M64N128 CGA2x1            |   reuse    |    45875.4 |  566.2 |  48830.3 |      532.0 |

### e2e overhead (the production path)

`e2e` minus `compute` is ~150-250 µs at small token counts growing to
~1.7-2.8 ms at 32768 — dominated by the torch-composed staging quant plus
the output copy.  The SM100 tree eliminated the analogous cost with a
fused single-launch quant+repack kernel (`FLASHINFER_MEGA_FUSED_STAGE`);
the SM90 tree has no counterpart yet — this is the top e2e lever (see
"Next levers").

## The knob surface

Geometry and behavior knobs are explicit
`Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig` fields, resolved once per
session at workspace allocation.  On top of the explicit fields the tree
now carries the SM100-style tuning stack (`shim/tuner.py`,
`shim/autotune.py`, `shim/knob_cache.py`): the config's `knobs=` field
accepts a knob dict, `"auto"` (collective online autotune on first
compute, winner persisted to the knob cache), or `None` (cache lookup,
then the heuristic table).  The autotune candidate set is the heuristic
winner plus every geometry that wins some bucket of the table (16 today,
derived programmatically) crossed with both validated token-back modes —
32 candidates.

- `fp8_scale_mode` — `"per_tensor"` (per-expert weight scalar + static
  activation calibration scalars, identical on all EP ranks by contract) or
  `"blockwise"` (DeepGEMM-style 128-block fp32 scales; requires
  hidden/intermediate %128).
- `swap_ab` + `pingpong` + `mma_tiler_mnk` + `cluster_shape_mnk` — layout,
  scheduling, tile, and CGA shape.  Leave ALL four `None` to use the drop's
  token-bucket heuristic table (`moe_hopper_fp8/heuristic_config.py`, keyed
  on `fp8_scale_mode` and max tokens/rank, derived from the 2026-08-19
  four-rank H200 DSV4 sweep); setting any one switches to manual mode with
  drop-driver defaults for the rest (non-swap (64, 128, 128), swap-AB
  (256, 32, 128), (128, 32, 128) with ping-pong; cluster (1, 1, 1)).
  Kernel-legal geometry: non-swap M∈{64}, N∈{128,256}; swap-AB M∈{128,256},
  N∈{16,32,64,128}; K=128; CGA (m,n)∈{(1,1),(2,1),(1,2),(2,2)}, k=1.
  Ping-pong needs one physical warpgroup per task tile: N=128 non-swap,
  M=128 swap-AB.
- `load_balance_mode` — `"static"` (default, used by the correctness
  tests) or `"atomic_counter"` (the drop's perf-sweep setting; used by the
  benchmark for reference parity).
- `token_back_mode` — `epi_warps`, `reuse_dispatch_warps`, or
  `standalone_warps` (four dedicated token-back warps).  Left unset it
  follows the per-token-bucket heuristic table (epi_warps small/mid
  buckets, reuse_dispatch_warps at the GEMM-bound tail — per_tensor
  >= 16384, blockwise >= 1024; 2026-08-23 four-rank H200 sweep) and is a
  tuner candidate axis.  All six token_back x reduce combinations are
  kernel-supported; `epi_warps` / `reuse_dispatch_warps` /
  `standalone_warps` are all bit-validated by the `mega_sm90` multirank
  oracles.  `token_back_by_dispatch` remains as a legacy bool alias
  (True -> `reuse_dispatch_warps`).
- `in_kernel_fc2_reduce` — REDG atomic-add combine (bf16 unordered sum,
  nondeterministic; validated in `mega_sm90` with the roundoff-envelope
  band, not measured in the sweep above).
- `fp8_accum_mode`, `kind` (e4m3/e5m2), clamps.

## Sweep methodology + environment (reproduce recipe)

**Hardware / software.**  One H200 node, 4x NVIDIA H200 141GB (sm_90,
cc 9.0) over NVLink.  Python 3.12, torch `2.12.0+cu130`,
`nvshmem4py-cu13`, **`nvidia-cutlass-dsl 4.6.0`** (the drop pins
`4.5.0dev0`; 4.6.0 compiles and runs this SM90 tree).  Whether the SM100
tree's ">=4.6.1 perf floor" finding applies to the SM90 kernels is
UNTESTED — worth one A/B run.

**Harness.**  `benchmarks/bench_moe_ep_sm90_mega.py`, one torchrun process
per GPU:

```bash
torchrun --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py
```

Defaults: heuristic launch configs, tokens 8..32768 (13 points) ×
{per_tensor, blockwise} = 26 rows, drop-recipe fp8 payloads, 5s cooldown
before each timed series, results archived to
`benchmark_data/<date>/<date>_<time>_mega_sm90_<order>_<scale>.csv`
(directories auto-created; the CSV carries the resolved heuristic config
columns).  Axes: `--tokens`, `--scale-mode`,
`--swap-ab`/`--no-swap-ab`/`--both-orders` (fixed layouts instead of the
heuristic), `--mma-tiler M,N`, `--kind`, `--token-back`,
`--load-balance-mode`, `--no-sparse-data`, `--cooldown-s`, `--output-csv`.

**Problem.**  The drop's block-permutation balanced routing over all 384
experts; fp8 payloads per the drop perf recipe (`--no-sparse-data`
switches to dense quantized-randn model data).
per_tensor activation scales are static config scalars identical on every
rank.  Env parity with the drop harness: `NCCL_NVLS_ENABLE=0`,
`NVSHMEM_DISABLE_NVLS=1`.

**Timed regions** (both barrier+sync-fenced per iteration, per-rank CUDA
events, warmup 3 + 20 timed iters, matching the drop's counts):

- `compute` — pre-staged inputs, repeated `compute(output=None)`
  (zero-copy view).  Closest to the drop's `mega_us + topk_us`.
- `e2e` — full `layer.forward` (validation + staging quant + kernel +
  output copy).  The serving-relevant number; no drop counterpart.

Reported per point: min/max/mean/median across ranks of the per-rank
means; `critical_tflops_*` uses the max-rank time.

**Comparison rule** (inherited from the SM100 lessons): match the problem
shape, the data recipe, the routing, AND the timed region before comparing
— note in particular that FI `compute` includes the standalone TopkReduce
tail and the host-launch gap inside its CUDA-event window, while the
drop's `*_mega_us` columns are profiler-extracted kernel time only.

## Next levers

1. **Fused staging kernel** — port the SM100 tree's single-launch
   quant+repack (`shim/quant_stage.py` / `FLASHINFER_MEGA_FUSED_STAGE`)
   to the SM90 fp8 staging path; it is the bulk of the e2e overhead.
2. **Heuristic re-calibration** — the token-bucket table was derived with
   a mega-only, no-post-warmup-alignment metric; under a metric that
   includes the top-k reduce and aligns ranks after warmup, the per-bucket
   winners may shift (especially the small-token and blockwise large-token
   CGA choices).  Re-derive and refresh `heuristic_config.py` when the
   kernel team's next sweep lands.
3. **DSL runtime A/B** — rerun one column on `nvidia-cutlass-dsl>=4.6.1`
   to check whether the SM100 perf-floor finding transfers to SM90.
4. **CUDA-graph capture** — the SM100 mega layer's warmup+capture path is
   kernel-agnostic; validate it on sm90_fp8_fp8_bf16_pull_cutedsl (`test_mega_cuda_graph`
   analog) for decode serving.
