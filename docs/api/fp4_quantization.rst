.. _apifp4_quantization:

flashinfer.fp4_quantization
===========================

.. currentmodule:: flashinfer.fp4_quantization

This module provides FP4 quantization operations for LLM inference, supporting various scale factor layouts and quantization formats.

Core Quantization Functions
---------------------------

.. autosummary::
    :toctree: ../generated

    fp4_quantize
    nvfp4_quantize
    nvfp4_batched_quantize
    nvfp4_block_scale_interleave
    e2m1_and_ufp8sf_scale_to_float
    silu_and_mul_fp4_batched_quantize

Matrix Shuffling Utilities
--------------------------

.. autosummary::
    :toctree: ../generated

    shuffle_matrix_a
    shuffle_matrix_sf_a

Types and Enums
---------------

.. autosummary::
    :toctree: ../generated

    SfLayout
