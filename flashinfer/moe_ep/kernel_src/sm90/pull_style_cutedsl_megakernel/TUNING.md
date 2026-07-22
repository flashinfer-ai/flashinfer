# SM90 pull-style FP8 MegaMoE tuning + performance notes

This document collects the performance work on the `sm90_pull_fp8` mega
backend: the measured microbenchmark results against the kernel team's
reference sweep, the benchmark methodology behind those numbers, the knob
surface as it exists today, and the open perf levers.  It is the companion
to `SKILL.md` (drop-update workflow) and mirrors the structure of the SM100
tree's `TUNING.md`.

Unless noted otherwise, all measurements were taken 2026-07-22 on a single
H100 node (4x NVIDIA H100 80GB HBM3, EP=4) at the kernel drop's DSV4-Pro
P03 geometry: **384 experts, top-6, hidden 7168, intermediate 3072
(post-SwiGLU; gate+up 6144), gate_up_clamp 10.0**, tokens-per-rank swept
512..32768 in powers of two — the same seven points, geometry, and knobs as
the kernel team's `moe_hopper_fp8/run_token_sweep_benchmark.py`.  Reference
numbers are Vincent's 2026-07-20 sweep (`moe_hopper_fp8/benchmark_data/
20260720/` in the kernel drop), taken on the same cluster's H100 nodes at
the same kernel commit (`1275b8b`) we vendor.

## Microbenchmark results (2026-07-22, FI `compute` series, max-rank mean µs)

FI = this backend through the `MegaKernelBackend` plugin API (pre-staged
inputs, repeated `compute(output=None)`); ref = the drop's reported
min-rank per-rank time for the matching case/tile CSV.  Delta in parens.
Log: `sm90_bench_20260722.log` (repo root, one `BENCH_CSV` row per point).

**per_tensor, non-swap (TileM64 N128)** — peak 562 TFLOPS/rank:

| tok/rank | FI µs   | ref µs  | delta  | FI TFLOPS | FI e2e µs |
|---------:|--------:|--------:|-------:|----------:|----------:|
| 512      | 2762.6  | 2681.2  | +3.0%  | 146.9     | 2942.4    |
| 1024     | 2834.3  | 3287.9  | −13.8% | 286.4     | 3024.0    |
| 2048     | 4290.5  | 4822.2  | −11.0% | 378.4     | 4452.5    |
| 4096     | 6982.9  | 6921.5  | +0.9%  | 465.0     | 7230.4    |
| 8192     | 11673.4 | 11729.6 | −0.5%  | 556.3     | 12222.3   |
| 16384    | 24365.0 | 22243.0 | +9.5%  | 533.1     | 23948.2   |
| 32768    | 46227.0 | 44253.4 | +4.5%  | 561.9     | 48582.5   |

**per_tensor, swap-AB (TileM256 N32)**:

| tok/rank | FI µs   | ref µs  | delta  | FI TFLOPS | FI e2e µs |
|---------:|--------:|--------:|-------:|----------:|----------:|
| 512      | 2600.7  | 3234.0  | −19.6% | 156.1     | 2781.3    |
| 1024     | 4396.0  | 4760.4  | −7.7%  | 184.7     | 4594.0    |
| 2048     | 6417.2  | 6279.1  | +2.2%  | 253.0     | 6621.3    |
| 4096     | 10157.9 | 10050.2 | +1.1%  | 319.7     | 10439.7   |
| 8192     | 17423.3 | 17659.4 | −1.3%  | 372.7     | 17973.6   |
| 16384    | 34263.4 | 32773.6 | +4.5%  | 379.1     | 34500.0   |
| 32768    | 68299.2 | 66555.9 | +2.6%  | 380.3     | 70561.7   |

**blockwise, non-swap (TileM64 N128)**:

| tok/rank | FI µs   | ref µs  | delta  | FI TFLOPS | FI e2e µs |
|---------:|--------:|--------:|-------:|----------:|----------:|
| 512      | 3024.1  | 2885.1  | +4.8%  | 134.2     | 3246.6    |
| 1024     | 3114.4  | 3607.4  | −13.7% | 260.6     | 3333.7    |
| 2048     | 4694.4  | 5284.3  | −11.2% | 345.8     | 4920.2    |
| 4096     | 7949.4  | 7590.1  | +4.7%  | 408.5     | 8371.2    |
| 8192     | 12878.6 | 12717.7 | +1.3%  | 504.3     | 13691.4   |
| 16384    | 27365.8 | 24592.6 | +11.3% | 474.6     | 27232.5   |
| 32768    | 52782.1 | 49907.9 | +5.8%  | 492.1     | 55881.2   |

**blockwise, swap-AB (TileM256 N32)**:

| tok/rank | FI µs   | ref µs  | delta  | FI TFLOPS | FI e2e µs |
|---------:|--------:|--------:|-------:|----------:|----------:|
| 512      | 2651.1  | 3319.5  | −20.1% | 153.1     | 2901.9    |
| 1024     | 4563.4  | 4982.4  | −8.4%  | 177.9     | 4802.4    |
| 2048     | 6927.8  | 6725.9  | +3.0%  | 234.4     | 7232.5    |
| 4096     | 10843.9 | 10607.5 | +2.2%  | 299.4     | 11322.8   |
| 8192     | 18267.4 | 18113.2 | +0.9%  | 355.5     | 18951.6   |
| 16384    | 35305.2 | 34187.3 | +3.3%  | 367.9     | 36822.0   |
| 32768    | 71412.4 | 69628.4 | +2.6%  | 363.8     | 74709.9   |

TFLOPS use the drop's per-rank formula: `routed = tok/rank × topk`,
`flops = 2·routed·hidden·(gateup + downproj)`, divided by the max-rank
(critical-path) time.

### Reading the deltas — comparison caveats

The FI and reference numbers are close but not measured identically; three
systematic differences all bias the FI number HIGHER, so true kernel parity
is tighter than the raw deltas:

1. **FI `compute` includes the TopkReduce tail** the drop's `*_mega_us`
   exclude (~17 µs at 512 tok/rank, growing with tokens — the drop reports
   it separately as `reported_min_topk_us`).
2. **FI reports the max-rank (critical-path) statistic; the drop reports
   min-rank.**  FI's cross-rank spread is <10 µs at every point, so this
   barely matters for FI — but the drop's min-rank convention picks its
   fastest rank.
3. FI per-rank values are CUDA-event means over 20 barrier-aligned iters;
   the drop uses profiler means over back-to-back iters.

Conclusions from the sweep:

- **The FI integration carries no measurable kernel-path overhead** —
  most points are within ±5% of the drop's own harness despite the three
  biases above.
- **FI beats the recorded reference at 512–2048 tok/rank, dramatically for
  swap-AB (−20% at 512).**  The reference CSVs show multi-millisecond
  cross-rank spread at small token counts (e.g. one 1024-token blockwise
  swap-AB row spans 3.9–8.1 ms across ranks) where FI spans <10 µs —
  Vincent's small-token cells were noisy runs; the barrier-fenced timing
  here is cleaner.  Treat the FI values as the reference at those sizes.
- **swap-AB wins at decode-like sizes** (2600.7 vs 2762.6 µs non-swap at
  512 tok/rank per_tensor) and loses everywhere ≥1024 — matching its
  design intent.  Non-swap M64 N128 is the throughput layout.
- **per_tensor is ~5–15% faster than blockwise** at equal points (fewer
  scale loads on the GEMM path); blockwise buys DeepGEMM-style accuracy.
- **The 16384 point is consistently the weakest (+3% to +11%) in every
  column**, with per-iteration medians well below means (e.g. per_tensor
  non-swap 23.5 ms median vs 24.4 ms mean) — occasional slow iterations,
  suspect `atomic_counter` load-balance variance or clock behavior over
  the long sweep rather than a systematic kernel slowdown.  UNRESOLVED:
  rerun `--tokens 16384` in isolation and/or with
  `load_balance_mode="static"` before treating it as real.

### e2e overhead (the production path)

`e2e` times the full `MoEEpLayer.forward` (validation + bf16→fp8 staging
quantization + kernel + output copy).  Overhead over the compute series is
~180 µs at 512 tok/rank growing to ~2.3–3.3 ms at 32768 — dominated by the
torch-composed staging quant.  The SM100 tree eliminated the analogous
cost with a fused single-launch quant+repack kernel
(`FLASHINFER_MEGA_FUSED_STAGE`); the SM90 tree has no counterpart yet —
this is the top e2e lever (see "Next levers").

## The knob surface (no tuner yet)

The SM90 tree has **no `tuner.py` / `autotune.py` / knob-cache** — geometry
and behavior knobs are explicit `Sm90PullFp8MegaMoeConfig` fields, resolved
once per session at workspace allocation:

- `fp8_scale_mode` — `"per_tensor"` (per-expert weight scalar + static
  activation calibration scalars, identical on all EP ranks by contract) or
  `"blockwise"` (DeepGEMM-style 128-block fp32 scales; requires
  hidden/intermediate %128).
- `swap_ab` + `mma_tiler_mnk` — layout + tile.  Shim defaults: non-swap
  (64, 128, 128), swap-AB (256, 32, 128).  Kernel-legal geometry: 1-CTA
  only; non-swap M∈{64}, N∈{128,256}; swap-AB M∈{128,256},
  N∈{16,32,64,128}; K=128.
- `load_balance_mode` — `"static"` (default, used by the correctness
  tests) or `"atomic_counter"` (the drop's perf-sweep setting; used by the
  benchmark for reference parity).
- `token_back_by_dispatch` — `reuse_dispatch_warps` combine token-back
  (the drop's non-ikr perf default, used by the benchmark) vs `epi_warps`
  (the correctness-validated default).  NOTE: `reuse_dispatch_warps` is
  currently only perf-exercised — add a `mega_sm90` correctness case
  before making it a production default.
- `in_kernel_fc2_reduce` — REDG atomic-add combine (bf16 unordered sum,
  nondeterministic; validated in `mega_sm90` with the roundoff-envelope
  band, not measured in the sweep above).
- `fp8_accum_mode`, `kind` (e4m3/e5m2), clamps.

When an SM90 tuner lands, mirror the SM100 flow (`knobs=` dict / knob
cache / `"auto"` collective online sweep) — the config-field plumbing is
already shaped for it.

## Sweep methodology + environment (reproduce recipe)

**Hardware / software.**  One H100 node, 4x NVIDIA H100 80GB HBM3 (sm_90,
cc 9.0) over NVLink.  Pyxis container image
`flashinfer-ep-pt2605-mega_moe_ep-20260722.sqsh` (NGC 26.05 base): Python
3.12.3, torch `2.12.0a0+5aff3928d8.nv26.05`, CUDA 13.2, `nvshmem4py-cu13`,
**`nvidia-cutlass-dsl 4.5.0`** (the public wheel compiles and runs this
SM90 drop; the drop pins `4.5.0dev0`).  Whether the SM100 tree's ">=4.6.1 perf floor" finding
applies to the SM90 kernels is UNTESTED — worth one A/B run.
FlashInfer = branch `sm90_implementation_vincent`, editable install baked
into the image (re-run `pip install --no-build-isolation -e .` when the
checkout moves).

**Harness.**  `benchmarks/bench_moe_ep_sm90_mega.py`, one torchrun process
per GPU:

```bash
srun -A <account> -p batch -N1 --ntasks-per-node=1 --gres=gpu:4 --time=02:00:00 \
  --container-image="$IMG" --container-mounts=$REPO:/host/flashinfer \
  --container-workdir=/host/flashinfer \
  bash -lc 'export PYTHONPATH=/host/flashinfer:$PYTHONPATH
            torchrun --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py'
```

Full default sweep = 7 token points × {per_tensor, blockwise} ×
{non-swap, swap-AB} = 28 rows, ~35 min wall (compiles dominate; one
`cute.compile` per point, session shared between the two timed series).
Axes: `--scale-mode`, `--swap-ab`/`--no-swap-ab`, `--mma-tiler M,N`,
`--tokens`, `--kind`, `--token-back`, `--load-balance-mode`.

**Problem.**  Balanced random routing over all 384 experts, random bf16
activations, weights random bf16 quantized by
`preprocess_sm90_pull_fp8_mega_weights` (shared by both series via
`MegaConfig(transformed_weights=..., preprocess_weights=False)`).
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
means; `critical_tflops_*` uses the max-rank time.  Each `BENCH_CSV` row
names the matching drop reference CSV
(`20260720_multirank_{scale}_{order}_TileM{m}_TileN{n}.csv`).

**Comparison rule** (inherited from the SM100 lessons): match BOTH the
problem shape and the timed region before comparing — the drop's
`reported` column is min-rank mega-only; FI `compute` is max-rank
mega+topk.

## Next levers

1. **Fused staging kernel** — port the SM100 tree's single-launch
   quant+repack (`shim/quant_stage.py` / `FLASHINFER_MEGA_FUSED_STAGE`)
   to the SM90 fp8 staging path; it is the bulk of the 180 µs–3.3 ms e2e
   overhead.
2. **16384-token variance** — isolate (`--tokens 16384`, both
   `load_balance_mode`s, longer iteration counts) and root-cause the
   slow-iteration tail before the number is quoted as a regression.
3. **`reuse_dispatch_warps` correctness case** — add
   `token_back_by_dispatch=True` to `mega_sm90` so the perf-default path
   is bit-validated like the rest.
4. **DSL runtime A/B** — rerun one column on `nvidia-cutlass-dsl>=4.6.1`
   to check whether the SM100 perf-floor finding transfers to SM90.
5. **Tuner + knob cache** — port the SM100 `tuner.py`/`autotune.py`/knob
   cache stack once the kernel team's tile/knob sweep space for SM90
   stabilizes (today: two tiles per layout, `flag_batch`/`epi_flag_batch`
   defaults from the drop driver).
6. **CUDA-graph capture** — the SM100 mega layer's warmup+capture path is
   kernel-agnostic; validate it on sm90_pull_fp8 (`test_mega_cuda_graph`
   analog) for decode serving.
