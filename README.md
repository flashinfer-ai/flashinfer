<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-black-background.png?raw=true">
    <img alt="FlashInfer" src="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-white-background.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
Kernel Library for LLM Serving
</h1>

<p align="center">
| <a href="https://flashinfer.ai"><b>Blog</b></a> | <a href="https://docs.flashinfer.ai"><b>Documentation</b></a> | <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>Discussion Forum</b></a> |
</p>

[![Release](https://github.com/flashinfer-ai/flashinfer/actions/workflows/release_wheel.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/release_wheel.yml)
[![Documentation](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml)


FlashInfer is a library for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, SparseAttention, PageAttention, Sampling, and more. FlashInfer focus on LLM serving and inference, and delivers state-of-the-art performance across diverse scenarios.

The unique features of FlashInfer include:
1. **Comprehensive Attention Kernels**: Attention kernels that cover all the common use cases of LLM serving, including *single-request* and *batching* versions of *Prefill*, *Decode*, and *Append* kernels, on different formats of KV-Cache (Padded Tensor, Ragged Tensor, and Page Table).
2. **Optimized Shared-Prefix Batch Decoding**: FlashInfer enhances shared-prefix batch decoding performance through *cascading*, resulting in an impressive **up to 31x speedup** compared to the baseline vLLM PageAttention implementation (for long prompt of 32768 tokens and large batch size of 256).
3. **Accelerate Attention for Compressed/Quantized KV-Cache**: Modern LLMs are often deployed with quantized/compressed KV-Cache to reduce memory traffic. FlashInfer accelerates these scenarios by optimizing performance for *Grouped-Query Attention*, *Fused-RoPE Attention* and *Quantized Attention*.

FlashInfer support PyTorch, TVM and C++ (header-only) APIs, and can be easily integrated into existing projects.

## News
- [Jan 31, 2024] [Blog Post](https://flashinfer.ai/2024/01/08/cascade-inference.html) Cascade Inference: Memory-Efficient Shared Prefix Batch Decoding
- [Jan 31, 2024] [Blog Post](https://flashinfer.ai/2024/01/03/introduce-flashinfer.html) Accelerating Self-Attentions for LLM Serving with FlashInfer

## Getting Started

Using our PyTorch API is the easiest way to get started:

### Installation

We provide prebuilt wheels for Linux and you can try out FlashInfer with the following command:

```bash
# For CUDA 12.4 & torch 2.4
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
# For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html
```

or you can build from source:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer/python
pip install -e .
```

to reduce binary size during build and testing:
```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer/python
# ref https://pytorch.org/docs/stable/generated/torch.cuda.get_device_capability.html#torch.cuda.get_device_capability
export TORCH_CUDA_ARCH_LIST=8.0
pip install -e .
```

### Trying it out

Below is a minimal example of using FlashInfer's single-request decode/append/prefill attention kernels:

```python
import torch
import flashinfer

kv_len = 2048
num_kv_heads = 32
head_dim = 128

k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0) 
v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0) 

# decode attention

num_qo_heads = 32
q = torch.randn(num_qo_heads, head_dim).half().to(0)

o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly

# append attention
append_qo_len = 128
q = torch.randn(append_qo_len, num_qo_heads, head_dim).half().to(0) # append attention, the last 128 tokens in the KV-Cache are the new tokens
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True) # append attention without RoPE on-the-fly, apply causal mask
o_rope_on_the_fly = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True, pos_encoding_mode="ROPE_LLAMA") # append attention with LLaMA style RoPE on-the-fly, apply causal mask

# prefill attention
qo_len = 2048
q = torch.randn(qo_len, num_qo_heads, head_dim).half().to(0) # prefill attention
o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False) # prefill attention without RoPE on-the-fly, do not apply causal mask
```

Check out [documentation](https://docs.flashinfer.ai/) for usage of batch decode/append/prefill kernels and shared-prefix cascading kernels.

## Run Benchmarks

We profile FlashInfer kernel performance with [nvbench](https://github.com/NVIDIA/nvbench) and you can compile and run the benchmarks with the following commands:

```bash
mkdir build
cp cmake/config.cmake build # you can modify the config.cmake to enable/disable benchmarks and change CUDA architectures
cd build
cmake ..
make -j12
```

You can run `./bench_{single/batch}_{prefill/decode}` to benchmark the performance (e.g. `./bench_single_prefill` for single-request prefill attention). `./bench_{single/batch}_{prefill/decode} --help` will show you the available options. 

## C++ API and TVM Bindings

FlashInfer also provides C++ API and TVM bindings, please refer to [documentation](https://docs.flashinfer.ai/) for more details.

## Adoption

Currently FlashInfer is adopted by the following projects:
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- [Punica](https://github.com/punica-ai/punica)
- [SGLang](https://github.com/sgl-project/sglang)
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM)
- [vLLM](https://github.com/vllm-project/vllm)
- [TGI](https://github.com/huggingface/text-generation-inference)

## Acknowledgement

FlashInfer is inspired by [FlashAttention 1&2](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598) and [cutlass](https://github.com/nvidia/cutlass) projects.
