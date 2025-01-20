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
| <a href="https://flashinfer.ai"><b>Blog</b></a> | <a href="https://docs.flashinfer.ai"><b>Documentation</b></a> | <a href="https://join.slack.com/t/flashinfer/shared_invite/zt-2r93kj2aq-wZnC2n_Z2~mf73N5qnVGGA"><b>Slack</b></a>|  <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>Discussion Forum</b></a> |
</p>

[![Release](https://github.com/flashinfer-ai/flashinfer/actions/workflows/release_wheel.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/release_wheel.yml)
[![Documentation](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml)


FlashInfer is a library and kernel generator for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, SparseAttention, PageAttention, Sampling, and more. FlashInfer focuses on LLM serving and inference, and delivers state-of-the-art performance across diverse scenarios.

Check our [v0.2 release blog](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html) for new features!

The core features of FlashInfer include:
1. **Efficient Sparse/Dense Attention Kernels**: Efficient single/batch attention for sparse(paged)/dense KV-storage on CUDA Cores and Tensor Cores (both FA2 & FA3) templates. The vector-sparse attention can achieve 90% of the bandwidth of dense kernels with same problem size.
2. **Load-Balanced Scheduling**: FlashInfer decouples `plan`/`run` stage of attention computation where we schedule the computation of variable-length inputs in `plan` stage to alleviate load-imbalance issue.
3. **Memory Efficiency**: FlashInfer offers [Cascade Attention](https://docs.flashinfer.ai/api/cascade.html#flashinfer.cascade.MultiLevelCascadeAttentionWrapper) for hierical KV-Cache, and implements Head-Query fusion for accelerating Grouped-Query Attention, and efficient kernels for low-precision attention and fused-RoPE attention for compressed KV-Cache.
4. **Customizable Attention**: Bring your own attention variants through JIT-compilation.
5. **CUDAGraph and torch.compile Compatibility**: FlashInfer kernels can be captured by CUDAGraphs and torch.compile for low-latency inference.
6. **Efficient LLM-specific Operators**: High-Performance [fused kernel for Top-P, Top-K/Min-P sampling](https://docs.flashinfer.ai/api/sampling.html) without the need to sorting.

FlashInfer support PyTorch, TVM and C++ (header-only) APIs, and can be easily integrated into existing projects.

## News
- [Dec 16, 2024] [Blog Post](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html) FlashInfer 0.2 - Efficient and Customizable Kernels for LLM Inference Serving
- [Sept 2024] We've launched a [Slack](https://join.slack.com/t/flashinfer/shared_invite/zt-2r93kj2aq-wZnC2n_Z2~mf73N5qnVGGA) workspace for Flashinfer users and developers. Join us for timely support, discussions, updates and knowledge sharing!
- [Jan 31, 2024] [Blog Post](https://flashinfer.ai/2024/01/08/cascade-inference.html) Cascade Inference: Memory-Efficient Shared Prefix Batch Decoding
- [Jan 31, 2024] [Blog Post](https://flashinfer.ai/2024/01/03/introduce-flashinfer.html) Accelerating Self-Attentions for LLM Serving with FlashInfer

## Getting Started

Using our PyTorch API is the easiest way to get started:

### Installation

We provide prebuilt wheels for Linux. You can install FlashInfer with the following command:

```bash
# For CUDA 12.4 & torch 2.4
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
# For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html
```

We also offer nightly-built wheels to try the latest features from the main branch:

```bash
pip install flashinfer -i https://flashinfer.ai/whl/nightly/cu124/torch2.4
```

Alternatively, you can build FlashInfer from source:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install -e . -v
```

By default, FlashInfer uses Just-In-Time (JIT) compilation for its kernels. To pre-compile essential kernels, set the environment variable `FLASHINFER_ENABLE_AOT=1` before running the installation command:

```bash
FLASHINFER_ENABLE_AOT=1 pip install -e . -v
```

For more details, refer to the [Install from Source documentation](https://docs.flashinfer.ai/installation.html#install-from-source).

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

We are thrilled to share that FlashInfer is being adopted by many cutting-edge projects, including but not limited to:
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- [Punica](https://github.com/punica-ai/punica)
- [SGLang](https://github.com/sgl-project/sglang)
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM)
- [vLLM](https://github.com/vllm-project/vllm)
- [TGI](https://github.com/huggingface/text-generation-inference)
- [lorax](https://github.com/predibase/lorax)

## Acknowledgement

FlashInfer is inspired by [FlashAttention 1&2](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598), [cutlass](https://github.com/nvidia/cutlass) and [AITemplate](https://github.com/facebookincubator/AITemplate) projects.

## Citation

If you find FlashInfer helpful in your project or research, please consider citing our [paper](https://arxiv.org/abs/2501.01005):

```bibtex
@article{ye2025flashinfer,
    title = {FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving},
    author = {
      Ye, Zihao and
      Chen, Lequn and
      Lai, Ruihang and
      Lin, Wuwei and
      Zhang, Yineng and
      Wang, Stephanie and
      Chen, Tianqi and
      Kasikci, Baris and
      Grover, Vinod and
      Krishnamurthy, Arvind and
      Ceze, Luis
    },
    journal = {arXiv preprint arXiv:2501.01005},
    year = {2025},
    url = {https://arxiv.org/abs/2501.01005}
}
```
