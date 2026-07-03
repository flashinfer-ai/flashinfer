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
| `test_hyimage3_e2e.py` | End-to-end correctness + perf test runnable on **one GPU without the checkpoint weights**: (1) fused-MoE unit test comparing each `FlashInferFusedMoE` backend against an eager reference with identical weights/routing, with per-backend benchmarks; (2) a small random-weight stack of upstream `HunyuanImage3DecoderLayer`s deep-copied and swapped, comparing masked / mask-less-causal prefill outputs and forward latency (wan-style synthetic e2e testing). |
| `bench_kernels_isolated.py` | Isolated FlashInfer kernel benchmark (GEMM / RMSNorm / SwiGLU / MoE / attention) at HunyuanImage-3 shapes, sweeping GEMM backends and online/offline activation-quant modes. |
| [`BENCHMARK.md`](BENCHMARK.md) | H100 PCIe + B200 kernel speedup tables produced by the benchmark above. |

## What gets swapped

The HuggingFace model is loaded as-is (so VAE, SigLIP-2 ViT, tokenizer, image
processor, generation orchestration, and `HunyuanImage3Text2ImagePipeline`
stay upstream). For every `HunyuanImage3DecoderLayer` we replace:

| Upstream module | FlashInfer replacement |
| :--- | :--- |
| `HunyuanRMSNorm` (pre-attn, post-attn, final `ln_f`, per-head QK-norm) | shared `FlashInferRMSNorm` (`flashinfer.rmsnorm`) |
| `nn.Linear` (qkv_proj, o_proj, gate_and_up_proj, down_proj) | `FlashInferLinear` (any GEMM backend in `flashinfer_modules.GEMMBackend`) |
| SwiGLU activation (silu+mul) | `flashinfer.silu_and_mul` |
| `HunyuanMoE` (64 routed + 1 shared, top-8) | `FlashInferHunyuanMoE`: the upstream gate (FP32 softmax → top-8 → renormalize) is kept bit-identical; the routed experts run on the shared `FlashInferFusedMoE`, which dispatches to `flashinfer.fused_moe.cutlass_fused_moe` (BF16 or per-tensor FP8 W8A8) or `flashinfer.fused_moe.trtllm_bf16_routed_moe` (SM100/SM103) over stacked expert weights. The shared expert stays a dense FlashInfer SwiGLU MLP. |
| `HunyuanImage3SDPAAttention` | `FlashInferHunyuanImage3Attention` — GQA-aware wrapper over the shared `FlashInferAttentionDispatcher` (`single_prefill_with_kv_cache` / `cudnn_batch_prefill_with_kv_cache` / `trtllm_batch_context_with_kv_cache` / SDPA) for mask-less prefill (with `causal=True`), plus `flashinfer.decode.single_decode_with_kv_cache` for single-token steps. Falls back to `torch.nn.functional.scaled_dot_product_attention` when a 4D bool mask is provided (the multimodal causal+full mask the model uses for `gen_image` prefill has no FlashInfer equivalent). |

The 2D RoPE, `HunyuanStaticCache` KV cache, VAE / UNet patch in-out, and the
SigLIP-2 ViT are reused unchanged.

## Run

Requires the HuggingFace checkpoint at `tencent/HunyuanImage-3.0-Instruct`
(~169 GB) and a GPU with enough memory to host the MoE weights.

Text to image:
```bash
python pipeline_hunyuan_image3_flashinfer.py \
    --modality text2img \
    --prompts "A cute cat sitting on a windowsill watching the sunset" \
    --steps 50 --output ./out
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
| `--moe-backend` | `FLASHINFER_MOE_BACKEND` | `cutlass`, `cutlass_fp8`, `trtllm`, `torch`, `eager` | `cutlass` | Fused-MoE backend for the routed experts. `cutlass` = BF16 `cutlass_fused_moe` (SM89/SM90/SM100+); `cutlass_fp8` = per-tensor FP8 W8A8 on the same kernel; `trtllm` = `trtllm_bf16_routed_moe` (SM100/SM103 only); `torch` = eager loop inside `FlashInferFusedMoE`; `eager` keeps the upstream per-expert HunyuanMoE loop. Unsupported backends fall back to `torch` with a warning. |
| `--moe-impl` | `FLASHINFER_MOE_IMPL` | `flashinfer`, `eager` | — | Deprecated alias: `flashinfer` → `--moe-backend cutlass`, `eager` → `--moe-backend eager`. |
| `--offline-act-quant` | `FLASHINFER_ONLINE_ACT_QUANT=0` | — | online | Use fixed default activation scale instead of computing it from the current tensor (FP8/FP4 backends only). |

## Caveats

- HunyuanImage-3 prefill uses a custom 4D bool attention mask that mixes
  causal (for text) and full-attention (for image) blocks. FlashInfer's
  prefill APIs only accept the `causal` flag and have no equivalent for
  arbitrary masks, so prefill on those paths falls back to SDPA. The
  FlashInfer attention path is used for decode steps and any mask-less
  call.
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

## End-to-end test (no checkpoint weights needed)

`test_hyimage3_e2e.py` validates correctness and measures performance on a
single GPU using the checkpoint's *remote code only* (config + `.py` files —
never the 165 GB of weights):

```bash
# Fused-MoE unit test + backbone equivalence, sweeping MoE backends:
python test_hyimage3_e2e.py --stage all \
    --moe-backends cutlass cutlass_fp8 trtllm torch

# MoE kernel perf at real HunyuanImage-3 shapes (E=64, top-8, 4096x3072):
python test_hyimage3_e2e.py --stage moe --moe-shapes real \
    --moe-backends cutlass cutlass_fp8 trtllm --moe-num-tokens 1024 4096 8192
```

Stage 1 compares each `FlashInferFusedMoE` backend against an eager
reference with identical weights and routing. Stage 2 builds a small
random-weight stack of upstream `HunyuanImage3DecoderLayer`s, deep-copies
it, applies `replace_backbone_with_flashinfer`, and checks cosine
similarity / max error on masked and mask-less-causal prefill, then reports
baseline vs. FlashInfer forward latency. Exit code is non-zero if any check
falls below `--min-cos`. Point `--model-path` (or
`FLASHINFER_HYIMAGE3_PATH`) at a local checkout of the checkpoint repo to
run offline.
