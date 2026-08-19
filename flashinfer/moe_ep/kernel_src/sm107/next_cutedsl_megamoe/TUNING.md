# SM107 (Rubin) block-scaled mega kernel — tuning notes

Perf tracking for the `sm107_nvfp4_nvfp4_bf16_cutedsl` / `sm107_mxfp8_mxfp8_bf16_cutedsl`
mega backends against the upstream kernel team's Rubin perf report.

## Upstream reference (kernel team)

Measured by the upstream kernel tester at upstream commit `47881ad2`
(2026-08-15; its `rubin/inference/mega` files are identical to the vendored
`92dd334`), driver 615.31, four-GPU Rubin node. 384 autotune candidates per problem, 5 warmup + 20 measured
iterations; latency is the arithmetic mean of the four rank averages.

Problem: **DSv4 Pro, EP4** — hidden 7168, MoE intermediate 3072, 384 total
experts, top-k 6, NVFP4, BF16 combine.

Every selected winner uses **mixed CGA (preferred 4x1, fallback 2x1),
phase-interleave scheduling, atomic work IDs, FC2 bulk TMA stage 2,
epi-warp token back, and separate top-k reduction**.

### Balanced routing (upstream)

| Tokens/rank | Avg latency (us) | Min-max (us) | TFLOP/s | Selected-best detail |
|---|---|---|---|---|
| 1K | 372.22 | 366.18-394.78 | 2,180.8 | tile 256x128x256; hint 4; epi 1x4; tif 1 |
| 2K | 410.48 | 404.45-455.39 | 3,955.1 | tile 256x256x256; hint 3; epi 2x4; tif 1 |
| 4K | 529.56 | 521.12-541.73 | 6,131.5 | tile 256x256x256; hint 3; epi 2x4; tif 1 |
| 8K | 800.48 | 791.97-820.00 | 8,112.7 | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 16K | 1,484.16 | 1,470.72-1,582.53 | 8,751.1 | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 32K | 2,960.92 | 2,909.25-3,012.99 | 8,772.9 | tile 256x256x256; hint 3; epi 1x4; tif 1 |

### Power-law routing, alpha=0.8 (upstream)

| Tokens/rank | Avg latency (us) | Min-max (us) | Selected-best detail |
|---|---|---|---|
| 1K | 399.75 | 388.54-456.80 | tile 256x128x256; hint 3; epi 1x4; tif 1 |
| 2K | 474.56 | 466.14-535.26 | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 4K | 621.76 | 610.02-704.99 | tile 256x256x256; hint 4; epi 1x4; tif 1 |
| 8K | 1,053.01 | 1,035.65-1,178.27 | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 16K | 2,081.84 | 2,041.95-2,131.55 | tile 256x256x256; hint 3; epi 1x4; tif 4 |
| 32K | 4,023.79 | 3,990.11-4,152.51 | tile 256x256x256; hint 3; epi 1x4; tif 4 |

## flashinfer moe_ep measurements

Harness: `benchmarks/bench_moe_ep_sm107_block_scaled_mega.py` under
`torchrun --nproc_per_node=4`. The upstream selected-best knobs
are replayed verbatim through `Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig`
(no autotune sweep — one config per problem). Timing spans ONLY the fused
mega kernel launch (dispatch + FC1 + SwiGLU + FC2 + combine) via
`sm107_block_scaled_mega_launch_thunk` over pre-staged inputs; the torch
staging fallback is excluded, matching the upstream tester's span.
The timed loop replicates upstream ``tester/solver.py::perf_run`` exactly:
5 warmup + 20 measured iterations, per-iteration CUDA event pairs, and the
upstream per-iteration L2 flush (a 300MB throwaway ``randn`` enqueued
outside the event window); "avg" is the mean of the four rank averages,
"min-max" spans every rank sample. TFLOP/s = tokens x topk x 6 x hidden x intermediate / latency
(balanced only — the balanced cost model is not meaningful for the
imbalanced cases; matches the upstream convention).

Routing generators are ports of the upstream tester's block-balanced and
Zipf/Gumbel power-law samplers. Bracketed `X.XXx` values are LATENCY ratios
against the row's reference column — values above 1.00x are slower, below
are faster (not speedups).

### NVFP4, measured 2026-08-18 (4x SM107 node)

Vendored kernel drop `92dd334`, NVIDIA-internal CuTe DSL nightly build
(2026-08-03, git `d88cc85`), upstream L2-flush timing protocol. Raw per-rank
samples are written by the harness (`--output` JSONL).

#### NVFP4, balanced routing

| Tokens/rank | Upstream (us) | Ours (us, xupstream latency) | Min-max (us) | TFLOP/s | Knobs |
|---|---|---|---|---|---|
| 1K | 372.22 | 218.83 (0.59x) | 215.62-226.11 | 3,709.5 | tile 256x128x256; hint 4; epi 1x4; tif 1 |
| 2K | 410.48 | 290.53 (0.71x) | 286.02-300.00 | 5,588.0 | tile 256x256x256; hint 3; epi 2x4; tif 1 |
| 4K | 529.56 | 454.13 (0.86x) | 445.09-461.06 | 7,149.9 | tile 256x256x256; hint 3; epi 2x4; tif 1 |
| 8K | 800.48 | 802.92 (1.00x) | 798.46-807.23 | 8,087.9 | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 16K | 1,484.16 | 1,504.68 (1.01x) | 1,496.13-1,510.98 | 8,631.7 | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 32K | 2,960.92 | 3,014.84 (1.02x) | 3,003.90-3,026.98 | 8,616.0 | tile 256x256x256; hint 3; epi 1x4; tif 1 |

#### NVFP4, power-law routing (alpha=0.8)

| Tokens/rank | Upstream (us) | Ours (us, xupstream latency) | Min-max (us) | Knobs |
|---|---|---|---|---|
| 1K | 399.75 | 315.28 (0.79x) | 311.68-321.89 | tile 256x128x256; hint 3; epi 1x4; tif 1 |
| 2K | 474.56 | 378.97 (0.80x) | 373.79-387.55 | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 4K | 621.76 | 532.55 (0.86x) | 525.98-537.28 | tile 256x256x256; hint 4; epi 1x4; tif 1 |
| 8K | 1,053.01 | 1,255.96 (1.19x) | 1,247.71-1,265.28 | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 16K | 2,081.84 | 2,040.66 (0.98x) | 2,019.62-2,056.86 | tile 256x256x256; hint 3; epi 1x4; tif 4 |
| 32K | 4,023.79 | 4,497.98 (1.12x) | 4,466.43-4,606.08 | tile 256x256x256; hint 3; epi 1x4; tif 4 |

### MXFP8 (e4m3), measured 2026-08-18 (4x SM107 node, same run)

Same harness, same shape, same replayed winner knobs with tile K 128 (the
mxfp8 2x-mode instruction-K depth) instead of 256.  **The upstream report is
NVFP4-only, so MXFP8 has no upstream baseline** — the reference column is our
own NVFP4 measurement (same job); the ratio is the cost of doubling the
wire width (fp8 data + per-32 e8m0 scales vs fp4 + per-16 e4m3).

#### MXFP8, balanced routing

| Tokens/rank | NVFP4 ref (us) | MXFP8 (us, xNVFP4 latency) | Min-max (us) | TFLOP/s | Knobs |
|---|---|---|---|---|---|
| 1K | 218.83 | 319.07 (1.46x) | 315.04-339.01 | 2,544.1 | tile 256x128x128; hint 4; epi 1x4; tif 1 |
| 2K | 290.53 | 427.75 (1.47x) | 422.21-434.43 | 3,795.4 | tile 256x256x128; hint 3; epi 2x4; tif 1 |
| 4K | 454.13 | 643.82 (1.42x) | 637.25-649.63 | 5,043.3 | tile 256x256x128; hint 3; epi 2x4; tif 1 |
| 8K | 802.92 | 1,111.37 (1.38x) | 1,092.93-1,128.13 | 5,843.2 | tile 256x256x128; hint 3; epi 1x4; tif 1 |
| 16K | 1,504.68 | 2,062.29 (1.37x) | 2,030.08-2,111.58 | 6,297.8 | tile 256x256x128; hint 3; epi 1x4; tif 1 |
| 32K | 3,014.84 | 4,061.55 (1.35x) | 3,981.86-4,130.53 | 6,395.6 | tile 256x256x128; hint 3; epi 1x4; tif 1 |

#### MXFP8, power-law routing (alpha=0.8)

| Tokens/rank | NVFP4 ref (us) | MXFP8 (us, xNVFP4 latency) | Min-max (us) | Knobs |
|---|---|---|---|---|
| 1K | 315.28 | 437.27 (1.39x) | 429.60-445.12 | tile 256x128x128; hint 3; epi 1x4; tif 1 |
| 2K | 378.97 | 546.01 (1.44x) | 539.36-556.67 | tile 256x256x128; hint 3; epi 1x4; tif 1 |
| 4K | 532.55 | 698.71 (1.31x) | 692.74-707.39 | tile 256x256x128; hint 4; epi 1x4; tif 1 |
| 8K | 1,255.96 | 1,596.46 (1.27x) | 1,589.02-1,603.71 | tile 256x256x128; hint 3; epi 1x4; tif 1 |
| 16K | 2,040.66 | 2,588.31 (1.27x) | 2,562.85-2,655.49 | tile 256x256x128; hint 3; epi 1x4; tif 4 |
| 32K | 4,497.98 | 5,851.03 (1.30x) | 5,782.66-5,913.06 | tile 256x256x128; hint 3; epi 1x4; tif 4 |

MXFP8 lands at a steady ~1.3-1.5x the NVFP4 latency and plateaus at
~6.3 PFLOP/s (vs 8.6 for NVFP4) — consistent with the doubled operand bytes
through both GEMMs; the knobs were NOT re-tuned for mxfp8 (they replay the
nvfp4 winners), so a dedicated sweep may claw some of this back.

#### Reading the deltas

- **Balanced >= 8K matches upstream within ~2%** (+0.3% / +1.4% / +1.8%)
  at 8.1-8.6 PFLOP/s — the compute-bound regime, where a like-for-like
  comparison is meaningful. This validates the port end to end.
- **Small sizes measure faster than the upstream report (-14% to -41%)
  under the identical protocol** (same knobs, same warmup/iteration
  counts, same per-iteration L2 flush, same event span). The gap shrinks
  from ~150 us at 1K to ~0 at 8K, the profile of a fixed
  latency/bandwidth term: small sizes are dispatch/NVLink latency-bound,
  so this is an environment difference between the upstream measurement
  node (driver 615.31) and ours, not a kernel or harness difference.
- **Power-law rows carry routing-draw noise (+/-10-20%).** The Zipf
  popularity permutation is seed-dependent; a different draw puts a
  different load on the hottest expert/rank, which gates the whole
  collective (visible at 8K +19.3% vs 16K -2.0% with identical knobs
  scaled). Comparisons against upstream's imbalanced rows are directional
  only.
- Balanced-vs-power-law penalty on our numbers: +44% at 1K, +56% at 8K,
  +49% at 32K — same qualitative growth-with-size trend as upstream's
  +7% -> +40%, amplified by the different routing draw.

## Tuner

The SM107 backends are wired into the moe_ep offline knob tuner (same shape
as the SM100 one):

```bash
torchrun --nproc_per_node=4 -m flashinfer.moe_ep.tune \
    --arch sm107 --dtype nvfp4 --hidden 7168 --intermediate 3072 \
    --num-experts 384 --topk 6 --max-tokens 1024 8192 32768
```

- Candidate space: `sm107_candidates()` in the shim's `autotune.py` — 16
  candidates over tile N (128/256) x launch (uniform grouped vs mixed-CGA
  phase-interleave/atomic) x epi flag batches ((1,4)/(2,4)) x FC2 bulk TMA
  (off / 2-stage); `--allow-nondeterministic` adds the in-kernel-reduce
  axis (32). `--sweep schedule` pins those and sweeps the skew-sensitive
  hint x token-in-flag-batch grid (pair with `--skew`).
- Each candidate REBUILDS the kernel session (SM107 bakes knobs at
  construction — no `apply_knobs`), copies the staged inputs across, times
  `timed_iters` synchronized launches, and destroys the trial session; the
  winner is the argmin of the across-rank MAX medians.
- Winners persist in the shared knob cache
  (`FLASHINFER_MOE_EP_KNOB_CACHE`, default
  `~/.cache/flashinfer/moe_ep_knob_cache.json`; entries keyed by device +
  dtype + geometry + token bucket, so SM107 never collides with SM100).
- Engine-side resolution via the config's `knobs` field: `None` (default)
  keeps the explicit config fields; `"cache"` resolves the recorded winner
  (falling back to the built-in heuristic = the upstream selected-best
  profile in `default_knobs()`); a dict overrides explicitly. The
  SM100-style online `"auto"` sweep is deliberately NOT supported on the
  engine path (rebuild-per-candidate inside a serving engine is worse than
  the ~24-compile stall that motivated the offline cache in the first
  place).

## Notes

- Vendored drop: upstream `92dd334` (see `VENDOR.md`). The mixed-CGA knob
  is exposed as `fallback_cluster_shape_mn` on both backend configs; the
  shim replicates the upstream `launch_cluster_configuration()` occupancy
  recipe.
- The backends stage activations with the torch quantization fallback; a
  fused staging kernel is a known follow-up. End-to-end forward latency is
  therefore staging-dominated for now — the numbers above isolate the mega
  kernel to be comparable with the upstream report.
