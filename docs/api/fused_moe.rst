.. _apifused_moe:

flashinfer.fused_moe
====================

.. currentmodule:: flashinfer.fused_moe

This module provides fused Mixture-of-Experts (MoE) operations optimized for different backends and data types.

Types and Enums
---------------

.. autosummary::
    :toctree: ../generated

    RoutingMethodType
    WeightLayout

Utility Functions
-----------------

.. autosummary::
    :toctree: ../generated

    convert_to_block_layout
    reorder_rows_for_gated_act_gemm
    interleave_moe_weights_for_sm90_mixed_gemm
    interleave_moe_scales_for_sm90_mixed_gemm
    fused_topk_deepseek

Multi-LoRA MoE (BGMV)
---------------------

Batched Gather-Matrix-Vector kernels for serving multiple LoRA adapters on
top of a Mixture-of-Experts layer (shrink + expand).

.. autosummary::
    :toctree: ../generated

    bgmv_moe
    bgmv_moe_shrink
    bgmv_moe_expand

CUTLASS Fused MoE
-----------------

.. autosummary::
    :toctree: ../generated

    cutlass_fused_moe

TensorRT-LLM Fused MoE
----------------------

.. autosummary::
    :toctree: ../generated

    trtllm_bf16_moe
    trtllm_bf16_routed_moe
    trtllm_fp4_block_scale_moe
    trtllm_fp4_block_scale_routed_moe
    trtllm_fp8_block_scale_moe
    trtllm_fp8_block_scale_routed_moe
    trtllm_fp8_per_tensor_scale_moe
    trtllm_mxint4_block_scale_moe
    trtllm_mxint4_block_scale_routed_moe

CuteDSL Fused MoE
-----------------

The CuteDSL backends are conditionally available when the
``nvidia-cutlass-dsl`` package is installed.

.. autosummary::
    :toctree: ../generated

    cute_dsl_fused_moe_nvfp4
    b12x_fused_moe

.. autoclass:: CuteDslMoEWrapper
    :members:
    :inherited-members:
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: B12xMoEWrapper
    :members:
    :inherited-members:
    :show-inheritance:

    .. automethod:: __init__

MonoMoE (Single-Kernel Block-FP8, SM90a)
-----------------------------------------

Single-kernel top-K Mixture-of-Experts implementation specialized for the
Qwen3.5-35B block-FP8 shape on Hopper (SM90a). The full pipeline — routing,
up-projection, SiLU, down-projection and reduction — runs inside one kernel
launch. Use :func:`has_monomoe` to check availability before calling.

.. autosummary::
    :toctree: ../generated

    has_monomoe
    get_scratchpad_size_bytes
    alloc_scratchpad
    interleave_for_tma_wgmma_up
    mono_moe
