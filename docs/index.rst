.. FlashInfer documentation master file, created by
   sphinx-quickstart on Sat Jan 20 12:31:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FlashInfer's documentation!
======================================

`Blog <https://flashinfer.ai/>`_ | `Discussion Forum <https://github.com/orgs/flashinfer-ai/discussions>`_ | `GitHub <https://github.com/flashinfer-ai/flashinfer/>`_

FlashInfer is a library and kernel generator for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, PageAttention and LoRA. FlashInfer focus on LLM serving and inference, and delivers state-of-the-art performance across diverse scenarios.

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/recursive_attention
   tutorials/kv_layout

.. toctree::
   :maxdepth: 2
   :caption: PyTorch API Reference

   api/attention
   api/gemm
   api/fused_moe
   api/comm
   api/cascade
   api/sparse
   api/page
   api/sampling
   api/logits_processor
   api/norm
   api/rope
   api/activation
   api/quantization
   api/green_ctx
   api/fp4_quantization
   api/testing
