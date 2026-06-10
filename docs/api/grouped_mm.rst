.. _apigroupedmm:

flashinfer.grouped_mm
=====================

.. currentmodule:: flashinfer.grouped_mm

This module provides grouped matrix multiplication (MoE Grouped GEMM) APIs,
where each expert in a Mixture-of-Experts layer has its own weight matrix
and tokens are routed to experts via ``m_indptr``.

Grouped GEMM
------------

.. autosummary::
    :toctree: ../generated

    grouped_mm_bf16
    grouped_mm_fp4
    grouped_mm_fp8
    grouped_mm_mxfp8

MXFP8 Zero-Padding (SM120 cute backend)
---------------------------------------

.. autosummary::
    :toctree: ../generated

    grouped_mm_mxfp8_nt_groupwise_zero_padding
    quantize_mxfp8_for_zero_padding
