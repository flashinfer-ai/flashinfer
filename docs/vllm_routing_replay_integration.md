# vLLM Routing Replay Integration Guide

## Overview

FlashInfer supports an optional `routing_replay_out` parameter on its **trtllm-gen backend** MoE
kernel functions (not the Triton MoE path). When provided, the CUDA routing kernel writes all
top-K selected expert IDs per token directly into this tensor during routing — inside the same
fused kernel call that computes the MoE output.

This enables **routing replay** for downstream RL training: vLLM captures which experts were
selected for each token during inference and returns them in the API response.

## API

### `routing_replay_out` Parameter

Available on these vLLM integration path APIs (other MoE entry points also accept this parameter):
- `trtllm_fp8_block_scale_moe()`
- `trtllm_bf16_moe()`
- `trtllm_bf16_routed_moe()`
- `fused_topk_deepseek()`

**Spec:**
```text
routing_replay_out: Optional[torch.Tensor]
  dtype: torch.int16
  shape: (num_tokens_or_larger, top_k)
  Layout: row-major. replay[t, k] = k-th selected expert ID for token t
  Column order matches topk_indices
  When None: zero overhead, the kernel skips the write entirely
  When provided: the kernel writes expert IDs during routing
```

### CUDA Graph Compatibility

The buffer may be **larger** than `num_tokens`. This is intentional: vLLM pre-allocates
the buffer at `max_num_batched_tokens` and reuses it across CUDA graph replays. The kernel
determines write extent from `routing_logits.shape[0]`, not from `routing_replay_out.shape[0]`.

There is no strict `dim0 == num_tokens` validation — only `dim0 >= num_tokens` and
`dim1 == top_k`.

### Memory Layout for vLLM Integration

vLLM uses a device buffer with shape `(num_layers, max_num_batched_tokens, top_k)`:
- `buffer[layer_id]` gives a contiguous `(max_num_batched_tokens, top_k)` view
- This view is passed as `routing_replay_out` to the FlashInfer kernel
- The `(L, N, K)` layout ensures zero-copy per-layer slicing

### Integration Pattern

```python
# Pre-allocate once (during model initialization)
device_buffer = torch.zeros(
    (num_layers, max_num_batched_tokens, top_k),
    dtype=torch.int16,
    device="cuda",
)

# Per-layer forward pass
for layer_id, moe_layer in enumerate(moe_layers):
    replay_slice = device_buffer[layer_id]  # contiguous (N, K) view
    output = trtllm_fp8_block_scale_moe(
        ...,
        routing_replay_out=replay_slice,
    )
```

### Validation

```python
import torch

# Allocate replay buffer
replay = torch.full((num_tokens, top_k), -1, device="cuda", dtype=torch.int16)

# Run MoE
output = trtllm_fp8_block_scale_moe(..., routing_replay_out=replay)

# Verify non-zero (not all -1 sentinel)
assert (replay != -1).any(), "Routing replay data is all sentinel values"
assert (replay >= 0).all() and (replay < num_experts).all(), "Invalid expert IDs"
```
