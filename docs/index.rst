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
   :maxdepth: 1
   :titlesonly:
   :caption: API: Tensor Operations

   Normalization <api/norm>
   Rotary positional encoding <api/rope>
   Activation functions <api/activation>
   Quantization <api/quantization>
   FP4 compatibility imports <api/fp4_quantization>
   Hyper-connections <api/mhc>
   Diffusion operators <api/diffusion_ops>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API: Attention and KV Cache

   Attention entry points <api/attention>
   Combined prefill and decode <api/pod>
   Cascade attention <api/cascade>
   Sparse attention <api/sparse>
   Query-sparse attention <api/qsa>
   Attention scores and indexers <api/attn_scores>
   Paged KV-cache updates <api/page>
   MLA tensor packing <api/concat_ops>
   Cake FMHA <api/cake_fmha>
   Cake fused QK / RoPE / KV append <api/cake_fused_qk_rope_append>
   Cake fused FP8 QK / RoPE / KV append <api/cake_fused_qk_rope_fp8_append>
   cuDNN attention <api/cudnn>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API: Sequence Models

   Kimi Delta Attention <api/kda>
   Gated Delta Rule prefill <api/gdn_prefill>
   Gated Delta Rule decode <api/gdn_decode>
   Fused GDN decode step <api/gdn_fused_decode>
   Paged GDN state <api/gdn_paged>
   Gated Delta Rule 2 <api/gdn2_prefill>
   Gated DeltaProduct <api/gdp_prefill>
   Mamba causal convolution <api/mamba>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API: Matrix Multiplication

   Dense GEMM <api/gemm>
   Grouped GEMM <api/grouped_mm>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API: Mixture of Experts

   Unified fused MoE <api/fused_moe>
   MegaMoE <api/megamoe>
   Kimi K3 expert-parallel MoE <api/moe_ep_kimi_k3>
   Expert-parallel communication <api/moe_ep_communication>
   MXFP4 SiTU MoE <api/cake_mxfp4_situ_moe>
   StepFun MoE (experimental) <api/cake_stepfun_moe>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API: Sampling

   Sampling functions <api/sampling>
   Logits processor pipelines <api/logits_processor>
   Top-K selection <api/topk>
   Cake top-k/top-p sampling <api/cake_sampling>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API: Communication and Utilities

   Collective communication and topology <api/comm>
   Green contexts <api/green_ctx>
   CuTe DSL kernels <api/cute_dsl>
   Testing and benchmarking <api/testing>
