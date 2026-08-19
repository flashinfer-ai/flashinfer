# SM107 (Rubin) block-scaled mega kernel — tuning notes

Tuning surface, selected-best knob profiles, and benchmark methodology for
the `sm107_nvfp4_nvfp4_bf16_cutedsl` / `sm107_mxfp8_mxfp8_bf16_cutedsl` mega
backends. Concrete latency numbers are intentionally not recorded here —
they are hardware/driver/DSL-build specific; reproduce them with the
harness below on your own nodes.

## Reference tuning (upstream kernel team)

The upstream kernel tester swept 384 candidates per problem at upstream
commit `47881ad2` (2026-08-15; its `rubin/inference/mega` files are
identical to the vendored `92dd334`) on a four-GPU Rubin node, for the
**DSv4 Pro, EP4** problem — hidden 7168, MoE intermediate 3072, 384 total
experts, top-k 6, NVFP4, BF16 combine.

Every selected winner uses **mixed CGA (preferred 4x1, fallback 2x1),
phase-interleave scheduling, atomic work IDs, FC2 bulk TMA stage 2,
epi-warp token back, and separate top-k reduction**; only the tile,
phase-interleave hint, epi flag batches, and token-in flag batch vary:

| Routing | Tokens/rank | Tile (MxNxK) | Hint | Epi flags (FC1xFC2) | Token-in flag batch |
|---|---|---|---|---|---|
| balanced | 1K | 256x128x256 | 4 | 1x4 | 1 |
| balanced | 2K-4K | 256x256x256 | 3 | 2x4 | 1 |
| balanced | 8K-32K | 256x256x256 | 3 | 1x4 | 1 |
| power-law(0.8) | 1K | 256x128x256 | 3 | 1x4 | 1 |
| power-law(0.8) | 2K | 256x256x256 | 3 | 1x4 | 1 |
| power-law(0.8) | 4K | 256x256x256 | 4 | 1x4 | 1 |
| power-law(0.8) | 8K | 256x256x256 | 3 | 1x4 | 1 |
| power-law(0.8) | 16K-32K | 256x256x256 | 3 | 1x4 | 4 |

(NVFP4 tile K 256 = its 2x-mode instruction depth; for mxfp8 the analogous
tile K is 128.) These profiles are baked into `default_knobs()` in the
shim's `knob_cache.py` (two token buckets: tile N 128 below 2048
tokens/rank, 256 at or above) and into the benchmark harness's per-size
`WINNERS` table.

## Benchmark harness / methodology

`benchmarks/bench_moe_ep_sm107_block_scaled_mega.py` under
`torchrun --nproc_per_node=4`. The selected-best knobs above are replayed
verbatim (no autotune sweep — one config per problem). Timing spans ONLY
the fused mega kernel launch (dispatch + FC1 + SwiGLU + FC2 + combine) via
`sm107_block_scaled_mega_launch_thunk` over pre-staged inputs; the torch
staging fallback is excluded, matching the upstream tester's span.

The timed loop replicates the upstream tester's perf run exactly: 5 warmup
+ 20 measured iterations, per-iteration CUDA event pairs, and a
per-iteration L2 flush (a 300MB throwaway ``randn`` enqueued outside the
event window; ``--no-l2-flush`` disables it). Reported latency is the mean
of the rank averages; min-max spans every rank sample; TFLOP/s =
tokens x topk x 6 x hidden x intermediate / latency (balanced routing
only — the balanced cost model is not meaningful for imbalanced cases).
Raw per-rank samples are written as JSONL (``--output``).

Routing generators are ports of the upstream tester's block-balanced and
Zipf/Gumbel power-law samplers.

## Measured results (qualitative summary)

Measured 2026-08-18 on a 4x SM107 node (vendored drop `92dd334`,
NVIDIA-internal CuTe DSL nightly of 2026-08-03, git `d88cc85`):

- **NVFP4 matches the upstream reference within ~2% in the compute-bound
  regime** (>= 8K tokens/rank), validating the port end to end. At smaller
  token counts our node measured faster than the reference under the
  identical protocol; the gap decays to ~0 by 8K — the profile of a fixed
  latency/bandwidth term (small sizes are dispatch/NVLink latency-bound),
  i.e. a measurement-environment difference, not a kernel difference.
- **MXFP8 runs at roughly 1.3-1.5x the NVFP4 latency** at equal token
  counts — consistent with the doubled operand bytes through both GEMMs —
  and its knobs simply replay the NVFP4 winners (tile K 128); a dedicated
  mxfp8 tuner sweep may claw some of this back.
- **Power-law comparisons carry +/-10-20% routing-draw noise**: the Zipf
  popularity permutation is seed-dependent, and the hottest expert/rank
  gates the whole collective. Treat imbalanced-routing comparisons as
  directional only.

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
  (falling back to the built-in heuristic = the reference selected-best
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
  therefore staging-dominated for now — the benchmark isolates the mega
  kernel to keep results comparable with the upstream tester.
