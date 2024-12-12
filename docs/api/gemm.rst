.. _apigemm:

flashinfer.gemm
===============

.. currentmodule:: flashinfer.gemm

This module provides a set of GEMM operations.

FP8 Batch GEMM
--------------

.. autosummary::
    :toctree: ../generated

    bmm_fp8

Grouped GEMM
------------

.. autoclass:: SegmentGEMMWrapper
    :members:
    :exclude-members: forward

    .. automethod:: __init__
