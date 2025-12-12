# FlashInfer Examples

This directory contains end-to-end examples demonstrating how to use FlashInfer for LLM inference.

## New FlashInfer Modules

This example relies on several new modules added to FlashInfer to provide complete coverage of LLM operations:

### `flashinfer/linear.py` - Linear Algebra Operations

Generic linear/GEMM operations using PyTorch as a backend, exposed through FlashInfer's API.

| Function | Description |
|----------|-------------|
| `flashinfer.linear(input, weight)` | Matrix multiplication without bias |
| `flashinfer.linear_with_bias(input, weight, bias)` | Matrix multiplication with bias |
| `flashinfer.embedding(input, weight)` | Token embedding lookup |
| `flashinfer.bmm(input, mat2)` | Batched matrix multiplication |
| `flashinfer.matmul(input, other)` | General matrix multiplication |

**Usage:**
```python
import flashinfer

# Linear projection
output = flashinfer.linear(hidden_states, weight)  # [batch, seq, in] @ [out, in].T -> [batch, seq, out]

# With bias
output = flashinfer.linear_with_bias(hidden_states, weight, bias)

# Embedding lookup
embeddings = flashinfer.embedding(token_ids, embedding_weight)  # [batch, seq] -> [batch, seq, dim]
```

### `flashinfer/sparse_moe.py` - Sparse Mixture of Experts

Sparse MoE routing and computation for mixture-of-experts models.

| Function/Class | Description |
|----------------|-------------|
| `flashinfer.sparse_moe_forward(...)` | Core MoE forward pass with top-k routing |
| `flashinfer.SparseMoeBlock` | Pre-built MoE module class |

**Usage:**
```python
import flashinfer

# Functional API
output, router_logits = flashinfer.sparse_moe_forward(
    hidden_states,      # [batch, seq, hidden_dim]
    gate_weight,        # [num_experts, hidden_dim]
    expert_fn,          # Callable: (Tensor, int) -> Tensor
    num_experts=128,
    top_k=8,
    norm_topk_prob=True,
)

# Module API
moe_block = flashinfer.SparseMoeBlock(
    hidden_size=2048,
    intermediate_size=768,
    num_experts=128,
    top_k=8,
)
output, router_logits = moe_block(hidden_states)
```

### `flashinfer/tensor_parallel.py` - Megatron-style Tensor Parallelism

Tensor-parallel linear layers and communication primitives for distributed inference.

**Communication Primitives (FlashInfer Custom All-Reduce):**

FlashInfer provides optimized all-reduce using the `trtllm_allreduce_fusion` kernel, which is **included directly in FlashInfer** (no TensorRT-LLM or external library required). This kernel is optimized for low-latency intra-node communication and automatically falls back to NCCL if unavailable.

| Function | Description |
|----------|-------------|
| `flashinfer.all_reduce(tensor)` | Optimized all-reduce using `flashinfer.comm.trtllm_allreduce_fusion` (NCCL fallback) |
| `flashinfer.all_gather(tensor, dim)` | All-gather across TP group (NCCL) |
| `flashinfer.reduce_scatter(tensor, dim)` | Reduce-scatter across TP group (NCCL) |
| `flashinfer.is_using_flashinfer_custom_ar()` | Check if FlashInfer custom all-reduce is active |

**Tensor Parallel Layers:**

| Class | Description |
|-------|-------------|
| `flashinfer.ColumnParallelLinear` | Linear with output dim split across GPUs |
| `flashinfer.RowParallelLinear` | Linear with input dim split, all-reduce output |
| `flashinfer.MergedColumnParallelLinear` | Fused QKV or gate+up projections |
| `flashinfer.VocabParallelEmbedding` | Embedding with vocabulary parallelism |
| `flashinfer.TensorParallelMLP` | MLP with TP (gate/up=Column, down=Row) |
| `flashinfer.TensorParallelSparseMoeBlock` | MoE with TP within each expert |

**Parallelism Strategy:**
```
┌──────────────────────────────────────────────────────────────────┐
│                       Megatron-style TP                          │
├──────────────────────────────────────────────────────────────────┤
│  Q, K, V, gate_proj, up_proj  →  ColumnParallel (split output)   │
│  o_proj, down_proj            →  RowParallel (all-reduce output) │
│  Embedding, LM head           →  VocabParallel                   │
│  MoE experts                  →  TP within each expert           │
├──────────────────────────────────────────────────────────────────┤
│  All-Reduce Backend: flashinfer.comm.trtllm_allreduce_fusion     │
│  (No external dependencies - kernel included in FlashInfer)      │
└──────────────────────────────────────────────────────────────────┘
```

**Usage:**
```python
import flashinfer

# Initialize (call in each process) - sets up FlashInfer's optimized all-reduce workspace
flashinfer.init_tensor_parallel(
    tp_size=4,
    tp_rank=rank,
    max_token_num=8192,       # Workspace size for custom all-reduce
    hidden_dim=4096,          # Model hidden dimension
    use_flashinfer_custom_ar=True,  # Use FlashInfer's trtllm_allreduce_fusion
)

# Check if FlashInfer custom all-reduce is active
print(f"Using FlashInfer AR: {flashinfer.is_using_flashinfer_custom_ar()}")

# Column parallel: output is split, no communication
q_proj = flashinfer.ColumnParallelLinear(hidden_size, num_heads * head_dim)

# Row parallel: input is split, output is all-reduced via flashinfer.all_reduce
o_proj = flashinfer.RowParallelLinear(num_heads * head_dim, hidden_size)

# Vocab parallel embedding (uses flashinfer.all_reduce internally)
embed = flashinfer.VocabParallelEmbedding(vocab_size, hidden_size)

# Direct communication primitives (uses trtllm_allreduce_fusion when available)
output = flashinfer.all_reduce(partial_output)
gathered = flashinfer.all_gather(local_tensor, dim=-1)
```

---

## `llm_inference.py` - Complete LLM Inference with FlashInfer

This example shows how to perform complete LLM inference using **only FlashInfer kernels** for all compute-intensive operations. It loads pre-trained models from HuggingFace and replaces all standard PyTorch operations with optimized FlashInfer equivalents.

### FlashInfer Kernels Used

**Existing FlashInfer Kernels:**

| Operation | FlashInfer Function |
|-----------|---------------------|
| RMS Normalization | `flashinfer.rmsnorm` |
| Rotary Position Embeddings | `flashinfer.apply_rope_pos_ids_inplace` |
| Llama 3.1 RoPE (scaled) | `flashinfer.apply_llama31_rope_pos_ids_inplace` |
| Prefill Attention | `flashinfer.single_prefill_with_kv_cache` |
| Decode Attention | `flashinfer.single_decode_with_kv_cache` |
| SiLU Activation | `flashinfer.silu_and_mul` |
| Top-k/Top-p Sampling | `flashinfer.top_k_top_p_sampling_from_probs` |

**New Modules Added for This Example:**

| Operation | FlashInfer Function | Module |
|-----------|---------------------|--------|
| Token Embedding | `flashinfer.embedding` | `linear.py` |
| Linear Projections | `flashinfer.linear`, `flashinfer.linear_with_bias` | `linear.py` |
| Sparse MoE Routing | `flashinfer.sparse_moe_forward` | `sparse_moe.py` |

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

---

## `llm_inference_tp.py` - Tensor-Parallel LLM Inference

This example demonstrates **multi-GPU tensor-parallel inference** using Megatron-style parallelism.

### Tensor Parallelism Strategy

| Layer Type | Parallelism | Communication |
|------------|-------------|---------------|
| Q, K, V projections | Column Parallel | None (split output) |
| O projection | Row Parallel | `flashinfer.all_reduce` (trtllm_allreduce_fusion) |
| gate_proj, up_proj | Column Parallel | None (split output) |
| down_proj | Row Parallel | `flashinfer.all_reduce` (trtllm_allreduce_fusion) |
| Embedding | Vocab Parallel | `flashinfer.all_reduce` (trtllm_allreduce_fusion) |
| LM head | Column Parallel | `flashinfer.all_gather` (NCCL) |
| MoE experts | TP within expert | `flashinfer.all_reduce` per expert |

### Usage

Launch with `torchrun` for multi-GPU:

```bash
# 2-GPU tensor parallel
torchrun --nproc_per_node=2 llm_inference_tp.py --model Qwen/Qwen2.5-1.5B-Instruct

# 4-GPU tensor parallel  
torchrun --nproc_per_node=4 llm_inference_tp.py --model Qwen/Qwen3-4B-Instruct-2507

# 8-GPU tensor parallel for MoE
torchrun --nproc_per_node=8 llm_inference_tp.py --model Qwen/Qwen3-30B-A3B-Instruct-2507
```

### Requirements

- Multiple CUDA GPUs (SM 9.0+/Hopper recommended for best all-reduce performance)
- PyTorch with NCCL support
- `num_attention_heads` and `num_kv_heads` must be divisible by TP size
- For MoE models: works with any number of experts (TP applied within each expert)
- **No external libraries required** - FlashInfer's `trtllm_allreduce_fusion` kernel is self-contained

### Example Output

```
============================================================
FlashInfer Tensor-Parallel LLM Inference
============================================================

Model: Qwen/Qwen2.5-1.5B-Instruct
Tensor Parallel Size: 2
Device: cuda:0
FlashInfer Custom All-Reduce: True

...

Generated text:
The capital of France is Paris.

============================================================
Tensor Parallel Summary:
============================================================
✓ Tensor Parallel Size: 2
✓ ColumnParallelLinear: Q, K, V, gate_proj, up_proj
✓ RowParallelLinear: o_proj, down_proj
✓ VocabParallelEmbedding: Token embeddings
✓ flashinfer.all_reduce: Using FlashInfer trtllm_allreduce_fusion
✓ flashinfer.all_gather: Communication primitive
```
