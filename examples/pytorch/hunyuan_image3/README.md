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
| `modeling_hunyuan_image3_flashinfer.py` | Drop-in FlashInfer modules (`FlashInferHunyuanRMSNorm`, `FlashInferHunyuanImage3Attention`, FlashInfer SwiGLU MLP, fused-cutlass MoE wiring) and `replace_backbone_with_flashinfer(...)` which swaps them into a loaded HF model in place. |
| `pipeline_hunyuan_image3_flashinfer.py` | CLI driver that mirrors vllm-omni's `end2end.py`: loads the model with `trust_remote_code=True`, calls the swap helper, then drives `text2img` / `img2img` / `img2text` / `text2text`. |
| `bench_kernels_isolated.py` | Isolated FlashInfer kernel benchmark (GEMM / RMSNorm / SwiGLU / MoE / attention) at HunyuanImage-3 shapes, sweeping GEMM backends and online/offline activation-quant modes. |
| [`BENCHMARK.md`](BENCHMARK.md) | H100 PCIe + B200 kernel speedup tables produced by the benchmark above. |

## What gets swapped

The HuggingFace model is loaded as-is (so VAE, SigLIP-2 ViT, tokenizer, image
processor, generation orchestration, and `HunyuanImage3Text2ImagePipeline`
stay upstream). For every `HunyuanImage3DecoderLayer` we replace:

| Upstream module | FlashInfer replacement |
| :--- | :--- |
| `HunyuanRMSNorm` (pre-attn, post-attn, final `ln_f`) | `flashinfer.rmsnorm` via `FlashInferHunyuanRMSNorm` |
| `HunyuanRMSNorm` (per-head QK-norm, head_dim=128) | `flashinfer.rmsnorm` on flattened `(B*H*S, head_dim)` |
| `nn.Linear` (qkv_proj, o_proj, gate_and_up_proj, down_proj) | `FlashInferLinear` (any GEMM backend in `flashinfer_modules.GEMMBackend`) |
| SwiGLU activation (silu+mul) | `flashinfer.silu_and_mul` |
| `HunyuanMoE` (64 routed + 1 shared, top-8) | upstream class with `moe_impl='flashinfer'` (calls `flashinfer.fused_moe.cutlass_fused_moe`) |
| `HunyuanImage3SDPAAttention` | `FlashInferHunyuanImage3Attention` — GQA-aware path using `flashinfer.prefill.single_prefill_with_kv_cache` for mask-less prefill, `flashinfer.decode.single_decode_with_kv_cache` for single-token steps, and `flashinfer.cudnn.cudnn_batch_prefill_with_kv_cache` for batched prefill. Falls back to `torch.nn.functional.scaled_dot_product_attention` when a 4D bool mask is provided (the multimodal causal+full mask the model uses for `gen_image` prefill has no FlashInfer equivalent). |

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
| `--attention-backend` | `FLASHINFER_ATTENTION_BACKEND` | `auto`, `single`, `cudnn`, `trtllm`, `sdpa` | `auto` | Mask-less attention path. `auto` uses `single_prefill_with_kv_cache` for batch_size==1 and `cudnn_batch_prefill_with_kv_cache` for batch_size>1. Decode steps always use `single_decode_with_kv_cache`. With a custom 4D mask we always fall back to SDPA. |
| `--moe-impl` | `FLASHINFER_MOE_IMPL` | `flashinfer`, `eager` | `flashinfer` | `flashinfer` calls `flashinfer.fused_moe.cutlass_fused_moe` on stacked expert weights; `eager` keeps the per-expert Python loop. |
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
