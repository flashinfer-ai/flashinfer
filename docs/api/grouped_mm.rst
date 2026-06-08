.. _api-grouped-mm:

flashinfer.grouped_mm
=====================

.. currentmodule:: flashinfer.grouped_mm

Grouped matrix multiplication APIs for Mixture-of-Experts (MoE) layers,
where each expert holds its own weight matrix and tokens are routed to
experts via an ``m_indptr`` cumulative-count tensor.

The functions in this module mirror the dense ``flashinfer.gemm.mm_*``
APIs and currently dispatch to the cuDNN MoE backend.

BF16 / FP16
-----------

.. autosummary::
    :toctree: ../generated

    grouped_mm_bf16

FP8
---

.. autosummary::
    :toctree: ../generated

    grouped_mm_fp8

MXFP8
-----

.. autosummary::
    :toctree: ../generated

    grouped_mm_mxfp8

FP4 (NVFP4 / MXFP4)
-------------------

.. autosummary::
    :toctree: ../generated

    grouped_mm_fp4
