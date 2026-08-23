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
(`moe_hopper_fp8/heuristic_config.py`, derived from the kernel team's
2026-08-19 four-rank H200 sweep at the same vendored kernel sources).
Raw rows: `benchmark_data/20260823/20260823_multirank_heuristic_default_config_t8_32768.csv`.

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

**per_tensor** — peak 845 TFLOPS/rank:

| tok/rank | heuristic config                   | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB ping-pong M128N16 CGA2x1   |      845.4 |    7.5 |   1025.1 |        6.2 |
|       16 | swap-AB ping-pong M128N16 CGA1x2   |     1260.5 |   10.1 |   1420.6 |        8.9 |
|       32 | non-swap M64N256 CGA1x1            |     1768.4 |   14.3 |   1848.8 |       13.7 |
|       64 | swap-AB ping-pong M128N64 CGA1x2   |     1998.2 |   25.4 |   2153.4 |       23.6 |
|      128 | swap-AB ping-pong M128N32 CGA1x2   |     1877.6 |   54.0 |   2038.0 |       49.8 |
|      256 | swap-AB M256N32 CGA2x1             |     1858.6 |  109.2 |   2045.5 |       99.2 |
|      512 | swap-AB M256N64 CGA1x1             |     2149.7 |  188.8 |   2312.6 |      175.5 |
|     1024 | swap-AB ping-pong M128N64 CGA1x2   |     2313.3 |  350.9 |   2471.4 |      328.5 |
|     2048 | non-swap ping-pong M64N128 CGA2x1  |     3101.0 |  523.5 |   3190.1 |      508.9 |
|     4096 | non-swap ping-pong M64N128 CGA2x2  |     5260.1 |  617.3 |   5455.5 |      595.2 |
|     8192 | swap-AB ping-pong M128N64 CGA1x2   |     9124.8 |  711.7 |   9498.4 |      683.7 |
|    16384 | non-swap M64N256 CGA2x1            |    15900.7 |  816.8 |  16694.8 |      778.0 |
|    32768 | non-swap ping-pong M64N128 CGA2x2  |    30727.8 |  845.4 |  32444.7 |      800.6 |

**blockwise** — peak 569 TFLOPS/rank:

| tok/rank | heuristic config                   | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB M256N16 CGA2x1             |      854.4 |    7.4 |   1093.9 |        5.8 |
|       16 | swap-AB M256N16 CGA1x1             |     1255.6 |   10.1 |   1501.4 |        8.4 |
|       32 | swap-AB ping-pong M128N16 CGA1x2   |     1702.4 |   14.9 |   1943.4 |       13.1 |
|       64 | swap-AB M256N32 CGA2x1             |     1861.1 |   27.3 |   2102.7 |       24.1 |
|      128 | swap-AB M256N16 CGA2x1             |     1833.0 |   55.4 |   2074.3 |       48.9 |
|      256 | swap-AB ping-pong M128N32 CGA1x2   |     2187.3 |   92.8 |   2441.0 |       83.1 |
|      512 | non-swap M64N128 CGA1x1            |     2560.3 |  158.5 |   2716.7 |      149.4 |
|     1024 | non-swap M64N128 CGA2x2            |     3048.5 |  266.3 |   3221.1 |      252.0 |
|     2048 | non-swap M64N128 CGA2x2            |     4304.1 |  377.2 |   4494.9 |      361.2 |
|     4096 | non-swap M64N128 CGA1x1            |     6951.8 |  467.1 |   7299.1 |      444.9 |
|     8192 | non-swap M64N128 CGA2x1            |    11684.0 |  555.8 |  12309.7 |      527.5 |
|    16384 | non-swap M64N128 CGA1x2            |    22846.8 |  568.5 |  24192.1 |      536.9 |
|    32768 | non-swap M64N128 CGA2x1            |    45931.5 |  565.5 |  48735.8 |      533.0 |

### e2e overhead (the production path)

`e2e` minus `compute` is ~150-250 µs at small token counts growing to
~1.7-2.8 ms at 32768 — dominated by the torch-composed staging quant plus
the output copy.  The SM100 tree eliminated the analogous cost with a
fused single-launch quant+repack kernel (`FLASHINFER_MEGA_FUSED_STAGE`);
the SM90 tree has no counterpart yet — this is the top e2e lever (see
"Next levers").

## The knob surface (no tuner yet)

The SM90 tree has **no `tuner.py` / `autotune.py` / knob-cache** — geometry
and behavior knobs are explicit `Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig` fields, resolved
once per session at workspace allocation:

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
- `token_back_mode` — `epi_warps` (the correctness-validated default),
  `reuse_dispatch_warps` (the drop's non-ikr perf default, used by the
  benchmark), or `standalone_warps` (four dedicated token-back warps).
  All six token_back x reduce combinations are kernel-supported since the
  combine-surface alignment drop.  `token_back_by_dispatch` remains as a
  legacy bool alias (True -> `reuse_dispatch_warps`).  NOTE: the push modes
  are currently only perf-exercised — add a `mega_sm90` correctness case
  before making one a production default.
- `in_kernel_fc2_reduce` — REDG atomic-add combine (bf16 unordered sum,
  nondeterministic; validated in `mega_sm90` with the roundoff-envelope
  band, not measured in the sweep above).
- `fp8_accum_mode`, `kind` (e4m3/e5m2), clamps.

When an SM90 tuner lands, mirror the SM100 flow (`knobs=` dict / knob
cache / `"auto"` collective online sweep) — the config-field plumbing is
already shaped for it.

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
3. **`reuse_dispatch_warps` correctness case** — add
   `token_back_mode="reuse_dispatch_warps"` to `mega_sm90` so the
   perf-default path is bit-validated like the rest.
4. **DSL runtime A/B** — rerun one column on `nvidia-cutlass-dsl>=4.6.1`
   to check whether the SM100 perf-floor finding transfers to SM90.
5. **Tuner + knob cache** — port the SM100 `tuner.py`/`autotune.py`/knob
   cache stack once the kernel team's tile/knob sweep space for SM90
   stabilizes.
6. **CUDA-graph capture** — the SM100 mega layer's warmup+capture path is
   kernel-agnostic; validate it on sm90_fp8_fp8_bf16_pull_cutedsl (`test_mega_cuda_graph`
   analog) for decode serving.
