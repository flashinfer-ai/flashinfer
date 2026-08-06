# Accelerating WAN video generation with FlashInfer

*A walk through `examples/pytorch/wan/`, turning on one feature at a time.*

Text-to-video diffusion is an unusually honest benchmark. A 14B WAN transformer
generating five seconds of 720p runs the same denoiser 100 times over a 75,600-token
sequence, so anything you do to attention or to the GEMMs shows up directly in
wall-clock time — and anything you break shows up directly in the video.

This example ships a WAN 2.1/2.2 transformer wired to FlashInfer, and a
diffusers pipeline that drives it. Four things can be switched on independently:

- **Ulysses context parallelism** — shard the token sequence across GPUs,
  all-to-all inside attention (`flashinfer.comm.UlyssesCommunicator`)
- **VSA** — Video Sparse Attention on FlashInfer's `bsa_attn_blk64_fwd`
- **NVFP4 GEMM, dynamic** — `flashinfer.mm_fp4` with per-forward activation scales
- **NVFP4 GEMM, static** — the same, with the activation scale calibrated once

Below is what each one buys, measured end to end.

## The workload

```
720 x 1280, 81 frames, 50 denoising steps, guidance 5.0, flow-shift 5.0
model: FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers  (VSA-finetuned Wan2.1-14B)
```

The VAE takes that to a `21 x 90 x 160` latent, and WAN's `(1, 2, 2)` patching
takes *that* to a `21 x 45 x 80` grid — **75,600 visual tokens**, 40 heads,
head_dim 128. Self-attention at that length is the dominant cost: an Nsight
Systems trace of the baseline attributes **47% of all GPU time** to the dense
attention kernel and 18% to GEMM.

Every number below is bf16, batch 1, on B200, with 3 warmup denoising steps
before timing. "ms/step" is one denoising step, which is two transformer
forwards because of classifier-free guidance.

## Turning the features on

Cumulative, in the order you would actually adopt them. VSA is at sparsity
**0.9**, the value FastVideo's model card recommends.

### Single B200

| # | Config | denoise | ms/step | vs baseline | incremental |
|---|--------|--------:|--------:|------------:|------------:|
| 1 | baseline (bf16 GEMM + dense SDPA) | 611.0 s | 12220.4 | 1.00x | — |
| 2 | + VSA (sparsity 0.9) | 467.4 s | 9346.9 | **1.31x** | +30.7% |
| 3 | + NVFP4 dynamic | 422.7 s | 8453.4 | **1.45x** | +10.6% |
| 4 | + NVFP4 static | 403.3 s | 8065.0 | **1.52x** | +4.8% |

### 8x B200 (Ulysses)

| # | Config | denoise | ms/step | vs baseline | incremental |
|---|--------|--------:|--------:|------------:|------------:|
| 1 | baseline (bf16 GEMM + dense SDPA) | 83.1 s | 1662.1 | 1.00x | — |
| 2 | + VSA (sparsity 0.9) | 64.7 s | 1293.7 | **1.28x** | +28.4% |
| 3 | + NVFP4 dynamic | 62.6 s | 1251.1 | **1.33x** | +3.4% |
| 4 | + NVFP4 static | 59.2 s | 1182.4 | **1.41x** | +5.8% |

Ulysses itself is the step from the first table to the second: baseline goes 12220 -> 1662 ms/step (**7.35x** on 8 GPUs, 92% efficiency), and the fully-featured row goes 8065 -> 1182 ms/step (**6.82x**, 85%). Sparse attention shards slightly less well than dense, which is expected: the all-to-all payload is unchanged while the compute it overlaps with has shrunk.

## Where the time actually goes

The nsys traces make the two features legible. Summed kernel time divided by
(8 ranks x 3 steps) reproduces the measured ms/step to under half a percent, so
the GPU is essentially never idle and these shares are directly meaningful:

| config | GPU s | breakdown |
|--------|------:|-----------|
| baseline | 39.8 | dense SDPA **47%**, GEMM 18%, other 27%, all-to-all 8% |
| + VSA | 30.8 | attention **14%** + tiling 7%, GEMM 24%, other 48%, all-to-all 7% |
| + NVFP4 dynamic | 29.5 | attention 15%, GEMM **7%**, other 64%, all-to-all 6% |
| + NVFP4 static | 28.0 | attention 15%, GEMM 8%, other 62%, all-to-all 7% |

You can read both features doing exactly their job: VSA collapses attention from
47% to 14%, and NVFP4 collapses GEMM from 24% to 7%. Once both are on, neither
is the bottleneck any more — 62% of the time is elementwise work, norms and
dtype casts, which is where the next round of effort belongs.

## VSA, and how it earns its 28%

VSA replaces dense self-attention with two stages. The `(T, H, W)` token grid is
permuted so each `(4, 4, 4)` spatio-temporal cube becomes 64 contiguous tokens —
one cube, one kernel block. A coarse stage mean-pools each cube to a single
token and runs dense attention over the resulting ~1.4K, which both produces a
global-context term and scores every (query block, KV block) pair. The top
`ceil((1 - sparsity) * num_blocks)` blocks per query block then go through
`bsa_attn_blk64_fwd`, and the two branches combine as
`out = out_coarse * gate_compress + out_fine`.

Two implementation notes that mattered more than expected:

**Call the kernel directly, not through the wrapper.** `BlockSparseAttentionWrapper`
converts the block selection on the host inside `plan()`, which would put a
device sync in every layer of every denoising step, and it cannot forward
per-block valid-token counts — which this workload needs, because `21 x 45 x 80`
does not divide evenly by 4 and 340 of the 1440 blocks are partial.

**Tiling is not free.** The cube permutation started as allocate-zeros +
gather + scatter and cost 28% of VSA's total runtime, roughly 7x more than the
bytes moved justify. Rewriting it as a single gather — padding slots read token 0,
and are masked downstream by `block_valid_mask` in the pooled mean and by
`variable_block_sizes` inside the kernel — plus accumulating the pooled mean in
fp32 without materializing an fp32 copy, and dropping a needless sort of the
selected indices, took VSA from 1.42x to **1.77x** over dense attention at the
per-rank shape.

For reference, at the same sparsity and shape this implementation is **1.27x
faster than FastVideo's own** `video_sparse_attn`, and FlashInfer's
`bsa_attn_blk64_fwd` is **1.75x–1.89x** faster than FastVideo's
ThunderKittens/Triton block-sparse kernel when both are handed an identical
block selection.

### The sparsity caveat

**0.9 is a speed setting, not a quality setting.** At sparsity 0.9 the generated
video is visibly degraded — blown-out colour and lost detail, with global
composition intact. The quality-preserving ceiling on this checkpoint is around
**0.75**, where VSA is roughly break-even against B200's dense attention.

This is not a bug in this port: **FastVideo's own pipeline, DiT and kernel
produce the same failure mode on the same checkpoint** — frame mean/std/Δ of
39.1/77.5/17.72 against our 38.6/77.4/16.07, versus 55.3/72.9/6.25 for dense.
Nor do they keep any layer or any denoising step dense to compensate: the block
class is chosen once for the whole stack, and although their attention metadata
carries `current_timestep`, the top-k only ever reads the constant sparsity (the
knobs that ramp it are training-time). Their own inference default is in fact
`VSA_sparsity = 0.0`.

Beyond that, the tile tables are bit-identical to theirs, the block selection
overlaps 143/144, and the RoPE tables match diffusers exactly. It also is not the
partial blocks, not CFG, not the multistep solver, and not sequence parallelism —
each was ruled out with a dedicated run. See `BENCHMARK.md` for the full
elimination.

So: quote 1.41x when you are talking about kernels, and quote sparsity 0.75 when
you are talking about videos you intend to ship.

## NVFP4, and why "static" needs calibration

The `fp4` backend quantizes weights once and activations per forward. Its
"static" mode exists to skip the per-forward `amax` reduction, and it used to use
a fixed scale of `448 * 6` — which is the scale you would get if the activation
amax were exactly 1.0. Real WAN activations run 4–28, so that placeholder
over-scaled by 4–28x and clipped hard; the resulting videos were washed out
(frame mean 124/255 against 55 for the baseline).

It now observes the real amax over the first few forwards and freezes it:

```
step 0: x_amax= 4.250  sf=632.5     step 2: x_amax=16.750  sf=160.5  <- frozen
step 1: x_amax= 8.750  sf=307.2     step 3+: reuse, no amax reduction
```

Calibration lands in warmup, so the timed steps still pay nothing, and the scale
tensor keeps a stable address so CUDA-graph capture is unaffected. Static now
matches dynamic visually while staying ~5% faster. The FP8 offline paths still
use fixed placeholders and would need the same treatment.

## Ulysses: pick the backend by payload size

`UlyssesCommunicator` offers a fused NVLink-P2P all-to-all alongside the NCCL
path. The fused kernel folds away a full-tensor permute that the NCCL path
performs separately — worth 39% of NCCL's scatter at this shape — but gives it
back on raw transfer:

| payload per collective | NCCL | NVLink fused | winner |
|-----------------------:|-----:|-------------:|--------|
| 1.3 MB | 0.040 ms | **0.020 ms** | fused, 2.07x |
| 21.0 MB | 0.087 ms | **0.056 ms** | fused, 1.57x |
| **96.8 MB (WAN 720p)** | 0.300 ms | 0.302 ms | tie |
| 193.5 MB | 0.556 ms | 0.586 ms | NCCL, 1.05x |

NCCL's raw all-to-all reaches 529 GB/s at 96.8 MB while the fused kernel plateaus
around 320 GB/s, so collapsing two launches into one wins only while launch
overhead dominates. LLM-scale Ulysses (a few MB per collective) sits squarely in
the 1.5–2x region; WAN 720p lands exactly on the crossover. Either backend is
fine here, and they are numerically identical — same seed, same latents, bit for
bit.

## Reproducing

```bash
torchrun --nproc_per_node=8 examples/pytorch/wan/pipeline_wan_flashinfer.py \
  --model-id FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 --flow-shift 5.0 --seed 1024 \
  --gemm-backend fp4 --offline-act-quant --attention-backend torch \
  --use-vsa --vsa-sparsity 0.9 --ulysses-backend nccl \
  --warmup-steps 3 --timing-json timing.json --output wan.mp4
```

That is row 4. Drop `--offline-act-quant` for row 3, `--gemm-backend torch` for
row 2, and `--use-vsa` as well for row 1; drop `torchrun` entirely for the
single-GPU table. `python examples/pytorch/wan/vsa_sanity.py` checks the VSA path
against an eager reference, and `--check-rank-sync` asserts every rank holds
identical latents at every step.

Full methodology, the sparsity elimination, and the Ulysses payload sweep are in
[`BENCHMARK.md`](BENCHMARK.md).
