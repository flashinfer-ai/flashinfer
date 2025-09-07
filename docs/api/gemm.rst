.. _apigemm:

flashinfer.gemm
===============

.. currentmodule:: flashinfer.gemm

This module provides a set of GEMM operations.

FP4 GEMM
--------

.. autosummary::
    :toctree: ../generated

    mm_fp4

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
