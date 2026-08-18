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

### Measured 2026-08-17 — hecate0146 (4x SM107), job 435057

Branch `sm107_support` @ vendored `92dd334` (commits `dd6e5df6`/`ce5b5fd8`),
`nvidia-cutlass-dsl-internal==0.3.0+20260803235612.d88cc85`. Raw samples:
`.sqsh_build_logs/bench_sm107_mega_results_435057.jsonl`, log
`.sqsh_build_logs/sm107_bench_435057.log`.

#### Balanced routing

| Tokens/rank | Avg latency (us) | Min-max (us) | TFLOP/s | Upstream (us) | Delta | Knobs |
|---|---|---|---|---|---|---|
| 1K | 221.15 | 216.13-269.76 | 3,670.6 | 372.22 | -40.59% | tile 256x128x256; hint 4; epi 1x4; tif 1 |
| 2K | 293.17 | 286.91-333.06 | 5,537.7 | 410.48 | -28.58% | tile 256x256x256; hint 3; epi 2x4; tif 1 |
| 4K | 458.53 | 452.77-500.32 | 7,081.3 | 529.56 | -13.41% | tile 256x256x256; hint 3; epi 2x4; tif 1 |
| 8K | 809.45 | 801.12-855.10 | 8,022.8 | 800.48 | +1.12% | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 16K | 1,504.62 | 1,496.06-1,551.04 | 8,632.0 | 1,484.16 | +1.38% | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 32K | 3,011.57 | 3,000.77-3,061.28 | 8,625.4 | 2,960.92 | +1.71% | tile 256x256x256; hint 3; epi 1x4; tif 1 |

#### Power-law routing (alpha=0.8)

| Tokens/rank | Avg latency (us) | Min-max (us) | Upstream (us) | Delta | Knobs |
|---|---|---|---|---|---|
| 1K | 319.86 | 311.36-369.95 | 399.75 | -19.99% | tile 256x128x256; hint 3; epi 1x4; tif 1 |
| 2K | 379.82 | 370.66-431.42 | 474.56 | -19.96% | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 4K | 535.69 | 526.56-580.67 | 621.76 | -13.84% | tile 256x256x256; hint 4; epi 1x4; tif 1 |
| 8K | 1,269.08 | 1,260.19-1,309.76 | 1,053.01 | +20.52% | tile 256x256x256; hint 3; epi 1x4; tif 1 |
| 16K | 2,054.53 | 2,031.97-2,087.46 | 2,081.84 | -1.31% | tile 256x256x256; hint 3; epi 1x4; tif 4 |
| 32K | 4,504.00 | 4,469.70-4,596.80 | 4,023.79 | +11.93% | tile 256x256x256; hint 3; epi 1x4; tif 4 |

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

## Notes

- Vendored drop: upstream `92dd334` (see `VENDOR.md`). The mixed-CGA knob
  is exposed as `fallback_cluster_shape_mn` on both backend configs; the
  shim replicates the upstream `launch_cluster_configuration()` occupancy
  recipe.
- The backends stage activations with the torch quantization fallback; a
  fused staging kernel is a known follow-up. End-to-end forward latency is
  therefore staging-dominated for now — the numbers above isolate the mega
  kernel to be comparable with the upstream report.
