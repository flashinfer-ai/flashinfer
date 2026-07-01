.. _apigemm:

flashinfer.gemm
===============

.. currentmodule:: flashinfer.gemm

This module provides a set of GEMM operations.

BF16 GEMM
---------

.. autosummary::
    :toctree: ../generated

    mm_bf16
    bmm_bf16

FP4 GEMM
--------

.. autosummary::
    :toctree: ../generated

    mm_fp4

BF16 x FP4 GEMM (W4A16)
-----------------------

.. autosummary::
    :toctree: ../generated

    prepare_bf16_fp4_weights
    mm_bf16_fp4

MXFP8 GEMM
----------

.. autosummary::
    :toctree: ../generated

    mm_mxfp8
    bmm_mxfp8

FP8 GEMM
--------

.. autosummary::
    :toctree: ../generated

    mm_fp8
    bmm_fp8
    gemm_fp8_nt_blockscaled
    gemm_fp8_nt_groupwise
    group_gemm_fp8_nt_groupwise
    group_deepgemm_fp8_nt_groupwise
    batch_deepgemm_fp8_nt_groupwise
    fp8_blockscale_gemm_sm90

Mixed Precision GEMM (fp8 x fp4)
--------------------------------

.. autosummary::
    :toctree: ../generated

    group_gemm_mxfp8_mxfp4_nt_groupwise
    group_gemm_nvfp4_nt_groupwise

Router GEMM (DeepSeek-V3 / Mistral / GLM)
-----------------------------------------

.. autosummary::
    :toctree: ../generated

    mm_M1_16_K7168_N128
    mm_M1_16_K7168_N256
    mm_M1_16_K6144_N256
    tinygemm_bf16

Blackwell SM100 GEMM
--------------------

.. autosummary::
    :toctree: ../generated

    tgv_gemm_sm100

Grouped GEMM (CuTe-DSL, Blackwell)
----------------------------------

.. autosummary::
    :toctree: ../generated

    grouped_gemm_nt_masked

Grouped GEMM (Ampere/Hopper)
----------------------------

.. autoclass:: SegmentGEMMWrapper
    :members:
    :exclude-members: forward

    .. automethod:: __init__
