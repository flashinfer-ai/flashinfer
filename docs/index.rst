.. FlashInfer documentation master file, created by
   sphinx-quickstart on Sat Jan 20 12:31:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FlashInfer's documentation!
======================================

`Blog <https://flashinfer.ai/>`_ | `Discussion Forum <https://github.com/orgs/flashinfer-ai/discussions>`_ | `GitHub <https://github.com/flashinfer-ai/flashinfer/>`_

FlashInfer is a library for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, PageAttention and LoRA. FlashInfer focus on LLM serving and inference, and delivers state-of-the-art performance across diverse scenarios.

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

   api/python/decode
   api/python/prefill
   api/python/cascade
   api/python/sparse
   api/python/page
   api/python/sampling
   api/python/group_gemm
   api/python/norm
   api/python/rope
   api/python/quantization
