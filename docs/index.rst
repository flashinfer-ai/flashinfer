.. FlashInfer documentation master file, created by
   sphinx-quickstart on Sat Jan 20 12:31:26 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FlashInfer's documentation!
======================================

`Blog <https://flashinfer.ai/>`_ | `Discussion Forum <https://github.com/orgs/flashinfer-ai/discussions>`_ | `GitHub <https://github.com/flashinfer-ai/flashinfer/>`_

FlashInfer is a library and kernel generator for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, PageAttention and LoRA. FlashInfer focuses on LLM serving and inference, and delivers state-of-the-art performance across diverse scenarios.

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   installation
   cli
   logging
   fi_trace
   trace_apply
   autotuning
   experimental

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/recursive_attention
   tutorials/kv_layout
   tutorials/generated/jax_tvm_ffi/index
   tutorials/cake_kimi_k3_situ

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: PyTorch API Reference

   Attention <api/reference/attention>
   Linear Attention and State-Space Models <api/reference/linear_attention>
   Matrix Multiplication <api/reference/gemm>
   Mixture of Experts <api/reference/moe>
   Sampling and Logits <api/reference/sampling>
   Tensor Operations and Quantization <api/reference/tensor_ops>
   Backend-Specific APIs <api/reference/backends>
   Communication, Runtime and Testing <api/reference/utilities>
