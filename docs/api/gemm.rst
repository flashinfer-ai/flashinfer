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

MXFP8 GEMM
----------

.. autosummary::
    :toctree: ../generated

    mm_mxfp8
    bmm_mxfp8

MXFP8 Groupwise GEMM (cute SM120)
---------------------------------

Cute-DSL MXFP8 groupwise-scaled GEMM family for the NVIDIA RTX PRO 6000
Blackwell Server Edition (SM120). Uses per-row UE8M0 INT-packed
TMA-aligned scale layout; ``scale_granularity_mnk ∈ {(1, 1, 32), (1, 1, 128)}``.
Inputs are quantized via the helpers in
:ref:`MXFP8 Per-Row Layout (cute SM120) <mxfp8-per-row-layout>` under
:mod:`flashinfer.quantization`.

.. autosummary::
    :toctree: ../generated

    gemm_mxfp8_nt_groupwise
    batch_gemm_mxfp8_nt_groupwise
    group_gemm_mxfp8_nt_groupwise
    group_gemm_mxfp8_nt_groupwise_masked
    group_gemm_mxfp8_nt_groupwise_zero_padding
    quantize_mxfp8_for_zero_padding

FP8 GEMM
--------

.. autosummary::
    :toctree: ../generated

    bmm_fp8
    gemm_fp8_nt_groupwise
    group_gemm_fp8_nt_groupwise
    group_deepgemm_fp8_nt_groupwise
    batch_deepgemm_fp8_nt_groupwise

Mixed Precision GEMM (fp8 x fp4)
--------------------------------

.. autosummary::
    :toctree: ../generated

    group_gemm_mxfp4_nt_groupwise

Grouped GEMM (Ampere/Hopper)
----------------------------

.. autoclass:: SegmentGEMMWrapper
    :members:
    :exclude-members: forward

    .. automethod:: __init__
