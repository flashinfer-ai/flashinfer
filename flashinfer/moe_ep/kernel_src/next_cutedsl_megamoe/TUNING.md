# SM107 (Rubin) block-scaled mega kernel — tuning notes

Perf tracking for the `sm107_nvfp4_nvfp4_bf16_cutedsl` / `sm107_mxfp8_mxfp8_bf16_cutedsl`
mega backends against the upstream kernel team's Rubin TS4B report.

## Upstream reference (kernel team, Rubin TS4B)

Measured by the upstream `cutedsl_megamoe` tester at commit `47881ad2`
(`ag_dev/investigate_blackwell`; its `rubin/inference/mega` files are
identical to the vendored `92dd334`), driver 615.31, four-GPU Rubin TS4B
node. 384 autotune candidates per problem, 5 warmup + 20 measured
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
`torchrun --nproc_per_node=4` (sbatch wrapper
`.sqsh_build_logs/run_sm107_bench.sbatch`). The upstream selected-best knobs
are replayed verbatim through `Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig`
(no autotune sweep — one config per problem). Timing spans ONLY the fused
mega kernel launch (dispatch + FC1 + SwiGLU + FC2 + combine) via
`sm107_block_scaled_mega_launch_thunk` over pre-staged inputs; the torch
staging fallback is excluded, matching the upstream tester's span.
Same protocol: 5 warmup + 20 measured iterations, per-iteration CUDA event
pairs; "avg" is the mean of the four rank averages, "min-max" spans every
rank sample. TFLOP/s = tokens x topk x 6 x hidden x intermediate / latency
(balanced only — the balanced cost model is not meaningful for the
imbalanced cases; matches the upstream convention).

Routing generators are ports of the upstream
`tester/generate_inputs.py` block-balanced and Zipf/Gumbel power-law
samplers.

### NVFP4, measured 2026-08-17 — hecate0146 (4x SM107), job 435057

Branch `sm107_support` @ vendored `92dd334` (commits `dd6e5df6`/`ce5b5fd8`),
`nvidia-cutlass-dsl-internal==0.3.0+20260803235612.d88cc85`. Raw samples:
`.sqsh_build_logs/bench_sm107_mega_results_435057.jsonl`, log
`.sqsh_build_logs/sm107_bench_435057.log`.

#### NVFP4, balanced routing

| Tokens/rank | Avg latency (us) | Min-max (us) | TFLOP/s | Upstream (us) | Delta | Knobs |
|---|---|---|---|---|---|---|
| 1K | 221.15 | 216.13-269.76 | 3,670.6 | 372.22 | -40.59% | tile 256x128x256; hint 4; epi 1x4; tif 1 |
| 2K | 293.17 | 286.91-333.06 | 5,537.7 | 410.48 | -28.58% | tile 256x256x256; hint 3; epi 2x4; tif 1 |
| 4K | 458.53 | 452.77-500.32 | 7,081.3 | 529.56 | -13.41% | tile 256x256x256; hint 3; epi 2x4; tif 1 |
| 8K | 809.45 | 801.12-855.10 | 8,022.8 | 800.48 | +1.12% | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 16K | 1,504.62 | 1,496.06-1,551.04 | 8,632.0 | 1,484.16 | +1.38% | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 32K | 3,011.57 | 3,000.77-3,061.28 | 8,625.4 | 2,960.92 | +1.71% | tile 256x256x256; hint 3; epi 1x4; tif 1 |

#### NVFP4, power-law routing (alpha=0.8)

| Tokens/rank | Avg latency (us) | Min-max (us) | Upstream (us) | Delta | Knobs |
|---|---|---|---|---|---|
| 1K | 319.86 | 311.36-369.95 | 399.75 | -19.99% | tile 256x128x256; hint 3; epi 1x4; tif 1 |
| 2K | 379.82 | 370.66-431.42 | 474.56 | -19.96% | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 4K | 535.69 | 526.56-580.67 | 621.76 | -13.84% | tile 256x256x256; hint 4; epi 1x4; tif 1 |
| 8K | 1,269.08 | 1,260.19-1,309.76 | 1,053.01 | +20.52% | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 16K | 2,054.53 | 2,031.97-2,087.46 | 2,081.84 | -1.31% | tile 256x256x256; hint 3; epi 1x4; tif 4 |
| 32K | 4,504.00 | 4,469.70-4,596.80 | 4,023.79 | +11.93% | tile 256x256x256; hint 3; epi 1x4; tif 4 |

### MXFP8 (e4m3), measured 2026-08-17 — hecate (4x SM107), job 435114

Same harness, same shape, same replayed winner knobs with tile K 128 (the
mxfp8 2x-mode instruction-K depth) instead of 256.  **The upstream report is
NVFP4-only, so MXFP8 has no upstream baseline** — the reference column is our
own NVFP4 measurement (job 435057); the ratio is the cost of doubling the
wire width (fp8 data + per-32 e8m0 scales vs fp4 + per-16 e4m3). Raw samples:
`.sqsh_build_logs/bench_sm107_mega_results_435114.jsonl`.

#### MXFP8, balanced routing

| Tokens/rank | Avg latency (us) | Min-max (us) | TFLOP/s | NVFP4 ref (us) | MXFP8/NVFP4 | Knobs |
|---|---|---|---|---|---|---|
| 1K | 319.78 | 313.60-369.79 | 2,538.5 | 221.15 | 1.45x | tile 256x128x128; hint 4; epi 1x4; tif 1 |
| 2K | 432.27 | 424.86-480.58 | 3,755.8 | 293.17 | 1.47x | tile 256x256x128; hint 3; epi 2x4; tif 1 |
| 4K | 645.11 | 636.83-688.29 | 5,033.2 | 458.53 | 1.41x | tile 256x256x128; hint 3; epi 2x4; tif 1 |
| 8K | 1,113.43 | 1,099.94-1,144.19 | 5,832.4 | 809.45 | 1.38x | tile 256x256x128; hint 3; epi 1x4; tif 1 |
| 16K | 2,067.72 | 2,031.58-2,114.78 | 6,281.3 | 1,504.62 | 1.37x | tile 256x256x128; hint 3; epi 1x4; tif 1 |
| 32K | 4,089.07 | 3,968.13-4,202.53 | 6,352.5 | 3,011.57 | 1.36x | tile 256x256x128; hint 3; epi 1x4; tif 1 |

#### MXFP8, power-law routing (alpha=0.8)

| Tokens/rank | Avg latency (us) | Min-max (us) | NVFP4 ref (us) | MXFP8/NVFP4 | Knobs |
|---|---|---|---|---|---|
| 1K | 439.00 | 433.28-480.74 | 319.86 | 1.37x | tile 256x128x128; hint 3; epi 1x4; tif 1 |
| 2K | 545.58 | 535.49-590.02 | 379.82 | 1.44x | tile 256x256x128; hint 3; epi 1x4; tif 1 |
| 4K | 699.82 | 692.22-740.77 | 535.69 | 1.31x | tile 256x256x128; hint 4; epi 1x4; tif 1 |
| 8K | 1,595.73 | 1,584.32-1,637.02 | 1,269.08 | 1.26x | tile 256x256x128; hint 3; epi 1x4; tif 1 |
| 16K | 2,604.58 | 2,569.41-2,667.49 | 2,054.53 | 1.27x | tile 256x256x128; hint 3; epi 1x4; tif 4 |
| 32K | 5,850.91 | 5,792.77-5,894.21 | 4,504.00 | 1.30x | tile 256x256x128; hint 3; epi 1x4; tif 4 |

MXFP8 lands at a steady ~1.3-1.5x the NVFP4 latency and plateaus at
~6.3 PFLOP/s (vs 8.6 for NVFP4) — consistent with the doubled operand bytes
through both GEMMs; the knobs were NOT re-tuned for mxfp8 (they replay the
nvfp4 winners), so a dedicated sweep may claw some of this back.

#### Reading the deltas

- **Balanced >= 8K matches upstream within ~2%** (+1.1% / +1.4% / +1.7%) at
  8.0-8.6 PFLOP/s — the compute-bound regime, where a like-for-like
  comparison is meaningful. This validates the port end to end.
- **Small sizes read FASTER than upstream (-13% to -41%).** Our 20 measured
  launches run back-to-back on-stream with no per-iteration cross-rank
  barrier, so iteration i+1's dispatch overlaps iteration i's combine tail;
  the overlap hides a fixed slice of latency that matters most when the
  kernel is short. Treat the small-token rows as steady-state pipelined
  throughput, not isolated-launch latency — don't celebrate the negative
  deltas.
- **Power-law rows carry routing-draw noise (+/-10-20%).** The Zipf
  popularity permutation is seed-dependent; a different draw puts a
  different load on the hottest expert/rank, which gates the whole
  collective (visible at 8K +20.5% vs 16K -1.3% with identical knobs
  scaled). Comparisons against upstream's imbalanced rows are directional
  only.
- Balanced-vs-power-law penalty on our numbers: +45% at 1K, +57% at 8K,
  +50% at 32K — same qualitative growth-with-size trend as upstream's
  +7% -> +40%, amplified by the different routing draw.

## Tuner

The SM107 backends are wired into the moe_ep offline knob tuner (same shape
as the SM100 one):

```
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
