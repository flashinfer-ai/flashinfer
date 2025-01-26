import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from flashinfer.rope import apply_rope_sgl
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as vLLMRotaryEmbedding
import pytest
from vllm.platforms import current_platform

# TODO
# flashinfer cos_sin_cache + inplace

# 1. test interleave
# 2. test non interleave
# 3. benchmark vllm forward cuda and flashinfer
# 4. benchmark cast and non cast
# benchmark



class FlashInferRotaryEmbedding(RotaryEmbedding):
    
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_out, key_out = apply_rope_sgl(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )
        return query_out, key_out

@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads",
    [
        (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2),
        (64, 32, 2048, 8432, True, torch.bfloat16, "cuda", 2, 199, 4, 1),
        (64, 64, 32, 8000, False, torch.bfloat16, "cuda", 32, 32, 1, 1),
        (64, 64, 32, 8000, False, torch.bfloat16, "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 9231, False, torch.bfloat16, "cuda", 3, 231, 4, 2),
    ]
)
def test_correctness(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool,
    dtype: torch.dtype,
    device: str,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
):
    rope_ref = RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype).to(device)
    rope_flashinfer = FlashInferRotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype).to(device)

    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device)

    query_ref, key_ref = query.clone(), key.clone() 
    query_flashinfer, key_flashinfer = query.clone(), key.clone()

    query_ref_out, key_ref_out = rope_ref.forward_native(pos_ids, query_ref, key_ref)
    query_flashinfer_out, key_flashinfer_out = rope_flashinfer.forward_cuda(pos_ids, query_flashinfer, key_flashinfer)

    import numpy as np
    # Set numpy print options for 256 items per line in scientific notation
    np.set_printoptions(threshold=np.inf, linewidth=4000, precision=4, suppress=False, formatter={'float': lambda x: '{:8.4e}'.format(x)})
    # Set torch print options for 256 items per line in scientific notation
    torch.set_printoptions(threshold=float('inf'), linewidth=4000, sci_mode=True, precision=4)

    with open("query_ref_out.txt", "w") as f:
        f.write(str(query_ref_out))
    with open("query_flashinfer_out.txt", "w") as f:
        f.write(str(query_flashinfer_out))
    # import pdb; pdb.set_trace()
    
    torch.testing.assert_close(query_ref_out, query_flashinfer_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(key_ref_out, key_flashinfer_out, atol=1e-2, rtol=1e-2)

import triton

"""
llama 3 8B
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.40.0.dev0",
  "use_cache": true,
  "vocab_size": 128256
}


"""
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["seq_len"],
        x_vals=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        line_arg="provider",
        line_vals=["flashinfer", "native", "vllm"],
        line_names=["FlashInfer", "Native", "vLLM"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="rope-latency",
        args={"head_size": 4096//32, "rotary_dim": 4096//32, "max_position_embeddings": 65536, "base": 500000, "is_neox_style": True, "dtype": torch.bfloat16, "device": "cuda", "batch_size": 2, "num_q_heads": 32, "num_kv_heads": 8},
    )
)
def benchmark(provider, head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads):
    print(f"provider: {provider}, head_size: {head_size}, rotary_dim: {rotary_dim}, max_position_embeddings: {max_position_embeddings}, base: {base}, is_neox_style: {is_neox_style}, dtype: {dtype}, device: {device}, batch_size: {batch_size}, seq_len: {seq_len}, num_q_heads: {num_q_heads}, num_kv_heads: {num_kv_heads}")
    
    rope_forward = None

    if provider == "vllm":
        rope = vLLMRotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype).to(device)
        rope_forward = rope.forward_cuda
    elif provider == "flashinfer":
        rope = FlashInferRotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype).to(device)
        rope_forward = rope.forward_cuda
    elif provider == "native":
        rope = RotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype).to(device)
        rope_forward = rope.forward_native

    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: rope_forward(pos_ids, query, key), quantiles=quantiles)

    return ms, min_ms, max_ms

def profile_flashinfer():

    head_size = 4096//32
    rotary_dim = 4096//32
    max_position_embeddings = 8192
    base = 500000
    is_neox_style = True
    dtype = torch.bfloat16
    device = "cuda"

    rope = FlashInferRotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype).to(device)

    batch_size = 8
    seq_len = 8192
    num_q_heads = 32
    num_kv_heads = 8

    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device)

    from torch.profiler import profile, record_function, ProfilerActivity
    
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/rope_profile')
    ) as prof:
        with record_function("rope_forward"):
            for i in range(10):
                rope.forward_cuda(pos_ids, query, key)
                prof.step()
    # Print some basic statistics
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10
    ))

def simple_forward():
    head_size = 4096//32
    rotary_dim = 4096//32
    max_position_embeddings = 8192
    base = 500000
    is_neox_style = True
    dtype = torch.bfloat16
    device = "cuda"

    rope = FlashInferRotaryEmbedding(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype).to(device)

    seq_len = 1024
    batch_size = 8
    num_q_heads = 32
    num_kv_heads = 8

    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device)

    rope.forward_cuda(pos_ids, query, key)

if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="rope_sgl_benchmark.png")
    # profile_flashinfer()
    # simple_forward()


