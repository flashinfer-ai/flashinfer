# FlashInfer Examples

This directory contains end-to-end examples demonstrating how to use FlashInfer for LLM inference.

## `llm_inference.py` - Complete LLM Inference with FlashInfer

This example shows how to perform complete LLM inference using **only FlashInfer kernels** for all compute-intensive operations. It loads pre-trained models from HuggingFace and replaces all standard PyTorch operations with optimized FlashInfer equivalents.

### FlashInfer Kernels Used

| Operation | FlashInfer Function |
|-----------|---------------------|
| Token Embedding | `flashinfer.embedding` |
| Linear Projections | `flashinfer.linear`, `flashinfer.linear_with_bias` |
| RMS Normalization | `flashinfer.rmsnorm` |
| Rotary Position Embeddings | `flashinfer.apply_rope_pos_ids_inplace` |
| Llama 3.1 RoPE (scaled) | `flashinfer.apply_llama31_rope_pos_ids_inplace` |
| Prefill Attention | `flashinfer.single_prefill_with_kv_cache` |
| Decode Attention | `flashinfer.single_decode_with_kv_cache` |
| SiLU Activation | `flashinfer.silu_and_mul` |
| Top-k/Top-p Sampling | `flashinfer.top_k_top_p_sampling_from_probs` |
| Sparse MoE Routing | `flashinfer.sparse_moe_forward` |

### Supported Models

**Dense Models:**
- `meta-llama/Llama-3.1-8B-Instruct` (BF16)
- `Qwen/Qwen2.5-1.5B-Instruct` (BF16)
- `Qwen/Qwen3-4B-Instruct-2507` (BF16)
- `Qwen/Qwen3-4B-Instruct-2507-FP8` (FP8 quantized)

**Mixture of Experts (MoE) Models:**
- `Qwen/Qwen3-30B-A3B-Instruct-2507` (128 experts, 8 active)
- `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (FP8 quantized)

### Usage

```bash
# Basic usage (default: Qwen2.5-1.5B)
python llm_inference.py

# Specify a different model
python llm_inference.py --model Qwen/Qwen3-4B-Instruct-2507

# FP8 quantized model
python llm_inference.py --model Qwen/Qwen3-4B-Instruct-2507-FP8

# MoE model
python llm_inference.py --model Qwen/Qwen3-30B-A3B-Instruct-2507

# Custom prompt and generation settings
python llm_inference.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt "Explain quantum computing in simple terms" \
    --max-new-tokens 200 \
    --temperature 0.8
```

### Command-line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model ID |
| `--max-new-tokens` | `50` | Maximum number of tokens to generate |
| `--temperature` | `0.7` | Sampling temperature (0 = greedy) |
| `--prompt` | `None` | Custom prompt (uses default if not specified) |

### Requirements

- CUDA-enabled GPU (compute capability 8.0+ recommended)
- FlashInfer installed
- HuggingFace Transformers (`pip install transformers`)
- PyTorch with CUDA support

### How It Works

1. **Model Loading**: Loads the model configuration and tokenizer from HuggingFace
2. **Weight Transfer**: Creates a FlashInfer-based model architecture and copies weights from the HuggingFace model
3. **FP8 Handling**: For FP8 models, performs block-wise dequantization during weight transfer
4. **Inference**: Runs inference using FlashInfer kernels for all operations:
   - Prefill phase: Processes the entire prompt
   - Decode phase: Generates tokens one at a time with KV caching
5. **Sampling**: Uses FlashInfer's optimized top-k/top-p sampling

### Architecture Support

The example supports various model architectures:

- **Llama-style**: RoPE with optional scaling (Llama 3.1)
- **Qwen2-style**: Standard attention with optional QK normalization  
- **Qwen3-style**: QK normalization in attention
- **MoE**: Sparse mixture of experts with configurable top-k routing

### Example Output

```
============================================================
FlashInfer-based LLM Inference
============================================================

Model: Qwen/Qwen2.5-1.5B-Instruct
Model type: qwen
FP8 quantized: False
Device: cuda
Compute dtype: torch.bfloat16

...

Generated text:
The capital of France is Paris.

============================================================
FlashInfer Kernels Used:
============================================================
✓ flashinfer.embedding - Token embedding lookup
✓ flashinfer.linear - Linear projections
✓ flashinfer.rmsnorm - RMS normalization
✓ flashinfer.apply_rope_pos_ids_inplace - Standard RoPE
✓ flashinfer.single_prefill_with_kv_cache - Prefill attention
✓ flashinfer.single_decode_with_kv_cache - Decode attention
✓ flashinfer.silu_and_mul - SiLU activation
✓ flashinfer.top_k_top_p_sampling_from_probs - Sampling
```

