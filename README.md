# FlashInfer: Kernel Library for LLM Serving

[Blog](https://flashinfer.ai) | [Documentation](https://docs.flashinfer.ai/) | [Discussion Forum](https://github.com/orgs/flashinfer-ai/discussions)

FlashInfer is a library for Language Languages Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention,  PageAttention and LoRA. FlashInfer focus on LLM serving and inference, and delivers state-the-art performance across diverse scenarios.

The unique features of FlashInfer include:
1. **Comprehensive Attention Kernels:**: Attention kernels that cover all the common use cases of LLM serving, including *single-request* and *batching* versions of *Prefill*, *Decode*, and *Append* kernels, on different formats of KV-Cache (Padded Tensor, Ragged Tensor, and Page Table).
2. **Optimized Shared-Prefix Batch Decoding**: FlashInfer enhances shared-prefix batch decoding performance through *cascading*, resulting in an impressive **up to 31x speedup** compared to the baseline vLLM PageAttention implementation (for long prompt of 32768 tokens and large batch size of 256).
3. **Accelerate Attention for Compressed/Quantized KV-Cache**: Modern LLMs are often deployed with quantized/compressed KV-Cache to reduce memory traffic. FlashInfer accelerates these scenarios by optimizing performance for *Grouped-Query Attention*, *Fused-RoPE Attention* and *Quantized Attention*.

FlashInfer support PyTorch, TVM and C++ (header-only) APIs, and can be easily integrated into existing projects.

## News
- [Jan 26, 2024] [Post](https://flashinfer.ai/2024/01/08/cascade-inference.html) Cascade Inference: Memory-Efficient Shared Prefix Batch Decoding
- [Jan 26, 2024] [Post](https://flashinfer.ai/2024/01/03/introduce-flashinfer.html) Accelerating Self-Attentions for LLM Serving with FlashInfer

## Getting Started

Using our PyTorch API is the easiest way to get started:

### Installation

We provide prebuilt wheels for Linux and you can try out FlashInfer with the following command:

```bash
pip install flashinfer -f https://flashinfer.ai/whl/
```

or you can build from source:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer/python
pip install -e .
```

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
- [sglang](https://github.com/sgl-project/sglang)

## Acknowledgement

FlashInfer is inspired by [FlashAttention 1&2](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598) and [cutlass](https://github.com/nvidia/cutlass) projects.
