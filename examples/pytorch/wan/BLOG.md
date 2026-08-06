# Accelerating WAN video generation with FlashInfer

*Turning on one kernel-level feature at a time, and measuring what each one buys.*

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) is a GPU kernel library
for generative inference — attention, GEMM, quantization and communication
primitives, JIT-compiled and callable from PyTorch. It is mostly discussed in the
context of LLM serving, but video diffusion turns out to be an unusually honest
benchmark for it: a 14B [WAN](https://github.com/Wan-Video/Wan2.1) transformer
generating five seconds of 720p runs the same denoiser 100 times over a
75,600-token sequence, so anything you do to attention or to the GEMMs shows up
directly in wall-clock time.

FlashInfer ships a
[WAN example](https://github.com/flashinfer-ai/flashinfer/tree/main/examples/pytorch/wan):
a WAN 2.1/2.2 transformer wired to FlashInfer kernels, plus a
[diffusers](https://github.com/huggingface/diffusers) pipeline that drives it.
Three accelerations can be switched on independently:

- **Ulysses context parallelism**
  ([`UlyssesCommunicator`](https://docs.flashinfer.ai/api/comm.html)) — shards the
  token sequence across GPUs and all-to-alls inside attention, so every rank
  attends over the full sequence with a slice of the heads.
- **VSA — [Video Sparse Attention](https://arxiv.org/abs/2505.13389)**
  ([`bsa_attn_blk64_fwd`](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cute_dsl/sparse/bsa_attn_blk64.py))
  — permutes the token grid so each `(4,4,4)` spatio-temporal cube becomes one
  64-token kernel block, mean-pools every cube to score block pairs, and runs
  token-level attention only inside the top-scoring blocks.
- **NVFP4 GEMM** ([`mm_fp4`](https://docs.flashinfer.ai/api/gemm.html)) — 4-bit
  weights and activations on Blackwell tensor cores, with the activation scale
  either recomputed per forward (*dynamic*) or calibrated during warmup and then
  frozen (*static*).

## The workload

```
720 x 1280, 81 frames, 50 denoising steps, guidance 5.0, flow-shift 5.0
model: FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers  (VSA-finetuned Wan2.1-14B)
```

The [checkpoint](https://huggingface.co/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers)
is Wan2.1-14B finetuned for sparse attention by the
[FastVideo](https://github.com/hao-ai-lab/FastVideo) team. The VAE and WAN's
`(1, 2, 2)` patching turn one clip into a `21 x 45 x 80` grid — **75,600 visual
tokens**, 40 heads, head_dim 128. All numbers below are bf16, batch 1, NVIDIA
B200, 3 warmup steps before timing. One "step" is two transformer forwards
because of classifier-free guidance.

## Results

Cumulative, in the order you would adopt them, at FastVideo's recommended VSA
sparsity of 0.9.

**Single B200**

| Config | ms/step | vs baseline |
|--------|--------:|------------:|
| baseline (bf16 GEMM + dense SDPA) | 12220 | 1.00x |
| + VSA | 9347 | **1.31x** |
| + NVFP4 dynamic | 8453 | **1.45x** |
| + NVFP4 static | 8065 | **1.52x** |

**8x B200 (Ulysses)**

| Config | ms/step | vs baseline |
|--------|--------:|------------:|
| baseline | 1662 | 1.00x |
| + VSA | 1294 | **1.28x** |
| + NVFP4 dynamic | 1251 | **1.33x** |
| + NVFP4 static | 1182 | **1.41x** |

Ulysses is the step between the two tables: 12220 → 1662 ms/step for the
baseline (**7.35x** on 8 GPUs, 92% efficiency) and 8065 → 1182 for the
fully-featured row (**6.82x**, 85%). End to end, five seconds of 720p goes from
**10.2 minutes on one GPU to 59 seconds on eight**.

## Why it works

Nsight Systems traces of the 8-GPU runs, bucketed by kernel:

| Config | dense attn | VSA attn | GEMM | other |
|--------|-----------:|---------:|-----:|------:|
| baseline | **47%** | — | 18% | 35% |
| + VSA | — | 14% (+7% tiling) | 24% | 55% |
| + NVFP4 | — | 15% | **7%** | 78% |

Both features do exactly their job: VSA takes attention from 47% of GPU time to
14%, then NVFP4 takes GEMM from 24% to 7%. Once both are on neither is the
bottleneck — what remains is elementwise work, norms and dtype casts, which is
where the next round of effort belongs.

Two things were worth the engineering. FlashInfer's block-sparse kernel is
**1.75x–1.89x** faster than the reference ThunderKittens/Triton implementation
given an identical block selection, and the cube permutation feeding it — a
single gather rather than zero-fill + gather + scatter — is what took VSA from
1.42x to **1.77x** over dense attention at the per-rank shape. NVFP4's static
mode calibrates the activation scale during warmup instead of assuming one, so
the timed steps skip the amax reduction without paying for it in quality.

## One caveat worth stating plainly

**Sparsity 0.9 is a speed setting, not a quality setting.** At 0.9 the video is
visibly degraded — oversaturated colour and lost detail, composition intact. The
quality-preserving ceiling on this checkpoint is around **0.75**, where VSA is
roughly break-even against B200's very fast dense attention and the NVFP4 rows
carry the win (1.11x on 8 GPUs).

This is not a defect in FlashInfer's implementation: FastVideo's own pipeline,
transformer and kernel produce the same degradation on the same checkpoint at
the same setting, and they keep no layer or denoising step dense to compensate.

So quote 1.41x–1.52x when talking about kernels, and sparsity 0.75 when talking
about videos you intend to ship.

## Reproducing

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer && pip install --no-build-isolation -e .

torchrun --nproc_per_node=8 examples/pytorch/wan/pipeline_wan_flashinfer.py \
  --model-id FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 --flow-shift 5.0 --seed 1024 \
  --gemm-backend fp4 --offline-act-quant --attention-backend torch \
  --use-vsa --vsa-sparsity 0.9 --ulysses-backend nccl \
  --warmup-steps 3 --timing-json timing.json --output wan.mp4
```

That is the last row of the 8-GPU table. Drop `--offline-act-quant`, then
`--gemm-backend torch`, then `--use-vsa` to walk back up it; drop `torchrun` for
the single-GPU table.

Full methodology, the sparsity investigation, the FastVideo comparison and the
Ulysses backend sweep are in
[BENCHMARK.md](https://github.com/flashinfer-ai/flashinfer/blob/main/examples/pytorch/wan/BENCHMARK.md).
