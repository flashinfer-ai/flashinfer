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

CUTLASS Fused MoE
-----------------

.. autosummary::
    :toctree: ../generated

    cutlass_fused_moe
    gen_cutlass_fused_moe_sm100_module

TensorRT-LLM Fused MoE
---------------------

.. autosummary::
    :toctree: ../generated

    trtllm_fp4_block_scale_moe
    trtllm_fp8_block_scale_moe
    trtllm_fp8_per_tensor_scale_moe
