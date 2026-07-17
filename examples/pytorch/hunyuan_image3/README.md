# HunyuanImage-3.0 FlashInfer Example

End-to-end FlashInfer driver for
[`tencent/HunyuanImage-3.0-Instruct`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct).

Inspired by the upstream
[`vllm-omni/examples/offline_inference/hunyuan_image3`](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/hunyuan_image3)
entry-point, but uses FlashInfer kernels for the hot paths of the 32-layer MoE
decoder backbone instead of vLLM.

## What it does

| File | Purpose |
| :--- | :--- |
| `modeling_hunyuan_image3_flashinfer.py` | Drop-in FlashInfer modules built on the **shared** `examples/pytorch/flashinfer_modules.py` (same module set as the wan example): `FlashInferHunyuanImage3Attention` (shared `FlashInferAttentionDispatcher`), `FlashInferHunyuanMoE` (shared `FlashInferFusedMoE`), SwiGLU MLP, shared `FlashInferRMSNorm`, plus `replace_backbone_with_flashinfer(...)` which swaps them into a loaded HF model in place. |
| `pipeline_hunyuan_image3_flashinfer.py` | CLI driver that mirrors vllm-omni's `end2end.py`: loads the model with `trust_remote_code=True`, calls the swap helper, then drives `text2img` / `img2img` / `img2text` / `text2text`. |

## What gets swapped

The HuggingFace model is loaded as-is (so VAE, SigLIP-2 ViT, tokenizer, image
processor, generation orchestration, and `HunyuanImage3Text2ImagePipeline`
stay upstream). For every `HunyuanImage3DecoderLayer` we replace:

| Upstream module | FlashInfer replacement |
| :--- | :--- |
| `HunyuanRMSNorm` (pre-attn, post-attn, final `ln_f`, per-head QK-norm) | shared `FlashInferRMSNorm` (`flashinfer.rmsnorm`) |
| `nn.Linear` (qkv_proj, o_proj, gate_and_up_proj, down_proj) | `FlashInferLinear` (any GEMM backend in `flashinfer_modules.GEMMBackend`) |
| SwiGLU activation (silu+mul) | `flashinfer.silu_and_mul` |
| `HunyuanMoE` (64 routed + 1 shared, top-8) | `FlashInferHunyuanMoE`: the upstream gate (FP32 softmax → top-8 → renormalize) is kept bit-identical; the routed experts run on the shared `FlashInferFusedMoE`, which dispatches to `flashinfer.fused_moe.cutlass_fused_moe` (BF16, per-tensor FP8 W8A8, DeepSeek-style 128×128 block-scale FP8 W8A8 on SM90, or MXFP4 weight-only on SM90) or `flashinfer.fused_moe.trtllm_bf16_routed_moe` (SM100/SM103) over stacked expert weights. The shared expert stays a dense FlashInfer SwiGLU MLP. See the [Fused-MoE APIs section in `examples/pytorch/README.md`](../README.md#fused-moe-apis) for the full backend/API coverage matrix. |
| `HunyuanImage3SDPAAttention` | `FlashInferHunyuanImage3Attention` — GQA-aware wrapper over the shared `FlashInferAttentionDispatcher` (`single_prefill_with_kv_cache` / `cudnn_batch_prefill_with_kv_cache` / `trtllm_batch_context_with_kv_cache` / SDPA) for mask-less prefill (with `causal=True`), plus `flashinfer.decode.single_decode_with_kv_cache` for single-token steps. Falls back to `torch.nn.functional.scaled_dot_product_attention` when a 4D bool mask is provided (the multimodal causal+full mask the model uses for `gen_image` prefill has no FlashInfer equivalent). |

The 2D RoPE, `HunyuanStaticCache` KV cache, VAE / UNet patch in-out, and the
SigLIP-2 ViT are reused unchanged.

## Run

Requires the HuggingFace checkpoint at `tencent/HunyuanImage-3.0-Instruct`
(~169 GB) and GPU(s) with enough memory to host the MoE weights (a single
B200/H200, or several smaller GPUs via `--device-map auto`).

Environment requirements (from the upstream remote code):

- **transformers 4.x** (upstream tests 4.56; the driver fails fast on 5.x —
  the checkpoint's remote code uses the 4.x cache/generation APIs).
- Download to a **dot-free local directory** — with transformers 4.x the repo
  id's `3.0` breaks dynamic-module import:
  ```bash
  hf download tencent/HunyuanImage-3.0-Instruct --local-dir ./HunyuanImage-3-Instruct
  ```

Text to image (single GPU):
```bash
python pipeline_hunyuan_image3_flashinfer.py \
    --model ./HunyuanImage-3-Instruct \
    --modality text2img \
    --prompts "A cute cat sitting on a windowsill watching the sunset" \
    --steps 50 --output ./out
```

Text to image, sharded over 8 GPUs with FlashInfer backends:
```bash
python pipeline_hunyuan_image3_flashinfer.py \
    --model ./HunyuanImage-3-Instruct \
    --modality text2img \
    --prompts "A cute cat sitting on a windowsill watching the sunset" \
    --steps 50 --output ./out \
    --device-map auto --max-memory-per-gpu 90GiB \
    --gemm-backend bf16 --moe-backend cutlass_fp8_blockscale --bench
```

Image editing:
```bash
python pipeline_hunyuan_image3_flashinfer.py \
    --modality img2img \
    --image-path /path/to/image.png \
    --prompts "Make the petals neon pink" \
    --output ./out
```

Image to text (captioning):
```bash
python pipeline_hunyuan_image3_flashinfer.py \
    --modality img2text \
    --image-path /path/to/image.jpg \
    --prompts "Describe the content of the picture."
```

Text to text:
```bash
python pipeline_hunyuan_image3_flashinfer.py \
    --modality text2text \
    --prompts "What is the capital of France?"
```

## FlashInfer config

Backend selection follows the same convention as the wan example:
command-line flags > environment variables > defaults.

| Flag | Env | Values | Default | Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `--gemm-backend` | `FLASHINFER_GEMM_BACKEND` | `torch`, `bf16`, `fp8`, `fp8_sm90`, `bmm_fp8`, `fp8_groupwise`, `fp8_blockscaled`, `batch_deepgemm_fp8`, `fp4`, `bmm_bf16`, `mxfp8`, `bmm_mxfp8` | `torch` | GEMM kernel for swapped Linears. Unsupported backends on this GPU fall back to torch with a warning. |
| `--attention-backend` | `FLASHINFER_ATTENTION_BACKEND` | `auto`, `single`, `cudnn`, `trtllm`, `torch` (alias `sdpa`) | `auto` | Mask-less attention path (shared dispatcher). `auto` uses `single_prefill_with_kv_cache` for batch_size==1 and `cudnn_batch_prefill_with_kv_cache` for batch_size>1; `trtllm` requires SM100/SM103. Decode steps always use `single_decode_with_kv_cache`. With a custom 4D mask we always fall back to SDPA. |
| `--moe-backend` | `FLASHINFER_MOE_BACKEND` | `cutlass`, `cutlass_fp8`, `cutlass_fp8_blockscale`, `cutlass_w4a16`, `cutlass_nvfp4`, `trtllm`, `trtllm_fp8_blockscale`, `trtllm_fp4`, `torch`, `eager` | `cutlass` | Fused-MoE backend for the routed experts. `cutlass` = BF16 `cutlass_fused_moe` (SM89/SM90/SM100+); `cutlass_fp8` = per-tensor FP8 W8A8 on the same kernel; `cutlass_fp8_blockscale` = DeepSeek-style 128x128 block-scale FP8 W8A8 (SM90 only — the only arch with this kernel; needs CUDA >= 12.8); `cutlass_w4a16` = MXFP4 weight-only, BF16 activation on the Hopper mixed-input GEMM (SM90 only); `cutlass_nvfp4` = NVFP4 W4A4 (SM100/SM103/SM110/SM120/SM121); `trtllm` = `trtllm_bf16_routed_moe` (SM100/SM103 only); `trtllm_fp8_blockscale` / `trtllm_fp4` = trtllm-gen DeepSeek-FP8 / NVFP4 routed MoE (SM100/SM103 only); `torch` = eager loop inside `FlashInferFusedMoE`; `eager` keeps the upstream per-expert HunyuanMoE loop. Unsupported backends fall back to `torch` with a warning. See the [per-architecture matrix in `examples/pytorch/README.md`](../README.md#fused-moe-apis). |
| `--moe-impl` | `FLASHINFER_MOE_IMPL` | `flashinfer`, `eager` | — | Deprecated alias: `flashinfer` → `--moe-backend cutlass`, `eager` → `--moe-backend eager`. |
| `--offline-act-quant` | `FLASHINFER_ONLINE_ACT_QUANT=0` | — | online | Use fixed default activation scale instead of computing it from the current tensor (FP8/FP4 backends only). |

## End-to-end validation (8× H20, SM90)

Measured with the pipeline above: `text2img`, 1024×1024, 50 denoising steps,
`--seed 42`, same prompt, `--device-map auto --max-memory-per-gpu 90GiB`,
transformers 4.56. The baseline is `--skip-flashinfer` (upstream eager
per-expert MoE + SDPA). PSNR is computed against the baseline image at
identical seed — for a 32-layer × 50-step diffusion loop, anything above
~20 dB is the same image with minor detail drift (verified visually).
GEMM stays `torch` in the MoE rows because the BF16 FlashInfer GEMMs are
Blackwell-only and fall back to torch on SM90 anyway. The Blackwell-only MoE
backends (`cutlass_nvfp4`, `trtllm`, `trtllm_fp8_blockscale`, `trtllm_fp4`)
fall back to `torch` on this machine and are therefore absent from the
table; on SM100+ they follow the canonical kernel-test recipes (see the
coverage notes in [`examples/pytorch/README.md`](../README.md#fused-moe-apis)).

| Config | ms/step | Total (50 steps) | Speedup | PSNR vs baseline |
| :--- | ---: | ---: | ---: | ---: |
| baseline (`--skip-flashinfer`) | 2404.8 | 120.2 s | 1.00× | — |
| `--moe-backend cutlass` (BF16) | 2226.6 | 111.3 s | 1.08× | 25.0 dB |
| `--moe-backend cutlass_fp8` | 1502.7 | 75.1 s | **1.60×** | 21.5 dB |
| `--moe-backend cutlass_fp8_blockscale` | 1525.9 | 76.3 s | **1.58×** | **32.9 dB** |
| `--moe-backend cutlass_w4a16` | 3145.2 | 157.3 s | 0.76× | 13.8 dB (coherent, composition drift) |
| `--gemm-backend fp8_sm90` + blockscale MoE | 1446.7 | 72.3 s | **1.66×** | 32.7 dB |
| `--gemm-backend bmm_fp8` + blockscale MoE | 1757.4 | 87.9 s | 1.37× | 29.9 dB |

Takeaways:

- The FP8 fused-MoE backends carry the end-to-end win on Hopper: ~1.6× over
  the upstream eager MoE at essentially unchanged image quality.
  `cutlass_fp8_blockscale` (SM90-exclusive DeepSeek-style 128×128 block
  scales) tracks the baseline image closest of all backends.
- `cutlass_w4a16` demonstrates the Hopper mixed-input (MXFP4 weight ×
  BF16 activation) kernel, but is not a win here: the LUT-dequant GEMM is
  slower than BF16 at these shapes, and the coarser weight quantization
  shifts the sampled composition (still a clean, prompt-faithful image).
  Its 4-bit weights would pay off as memory savings only with a real
  4-bit checkpoint; this example quantizes on the fly and keeps the BF16
  masters.
- The fastest overall config quantizes **both** the projections and the MoE:
  `--gemm-backend fp8_sm90 --moe-backend cutlass_fp8_blockscale` (1.66×,
  PSNR 32.7 dB). Per-linear accuracy of the W8A8 kernel is ~cos 0.9993 —
  note the TRT-LLM kernel reads the activation scales in transposed
  `(K // 128, M)` layout with the M stride padded to a multiple of 4;
  passing the natural `(M, K // 128)` layout silently degrades every linear
  to ~cos 0.98, which compounds to pure noise over 32 layers × 50 steps
  (`FlashInferLinear._forward_fp8_sm90` handles the transposition).
- Correctness of every `FlashInferFusedMoE` backend is additionally checked
  against the module's own eager `torch` reference (identical weights and
  routing) at the real model shapes (E=64, top-8, hidden 4096, intermediate
  3072, 512 tokens): `cutlass` cos 0.99998, `cutlass_fp8` 0.9978,
  `cutlass_fp8_blockscale` 0.9980, `cutlass_w4a16` 0.979 (4-bit weights).
- `gen_text` accuracy (greedy decode, baseline vs swapped model with the
  `cutlass_fp8_blockscale` MoE, same loaded weights): first-step logits cos
  0.985–0.998 with identical top-1 tokens on all probes; greedy sequences
  eventually diverge at near-tie tokens (expected for any quantized
  backend under greedy decoding) but stay semantically equivalent and
  factually correct.

## Caveats

- HunyuanImage-3 prefill uses a custom 4D bool attention mask that mixes
  causal (for text) and full-attention (for image) blocks. FlashInfer's
  prefill APIs only accept the `causal` flag and have no equivalent for
  arbitrary masks, so prefill on those paths falls back to SDPA. The
  FlashInfer attention path is used for decode steps and any mask-less
  call. In practice both `gen_image` and `gen_text` always supply that
  mask, so the `single`/`cudnn`/`trtllm` *prefill* selections of
  `--attention-backend` are never reached with this model — only the
  decode kernel (`single_decode_with_kv_cache`) and the SDPA fallback
  run. The knob still matters for the shared dispatcher's other users
  (e.g. the wan example).
- `HunyuanStaticCache` is not a paged KV cache. We use the single-request
  decode kernel after extracting the contiguous KV slice from the static
  cache; this works because the upstream attention has already done
  `repeat_kv` and the cache stores keys/values in `(B, H, S, D)` layout.
- The `trtllm` attention backend is Blackwell-only (SM100/SM103). On
  other GPUs the resolver falls back to `single` / `cudnn` with a
  warning.
- `prepare_weights=True` (default) eagerly quantizes every swapped
  `FlashInferLinear`. Pass `--skip-prepare-weights` to the self-test
  script, or call `replace_backbone_with_flashinfer(..., prepare_weights=False)`
  programmatically, if you intend to `.to(...)` the model afterwards.

## Self-test

`modeling_hunyuan_image3_flashinfer.py` has a `__main__` that loads the
model, applies the swap, and reports how many modules were replaced
(no inference). Useful for sanity-checking the swap on a new machine:

```bash
python modeling_hunyuan_image3_flashinfer.py --gemm-backend bf16
```
