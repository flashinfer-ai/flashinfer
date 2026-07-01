.. _apiquantization:

flashinfer.quantization
=======================

Quantization-related kernels for FP4, FP8, and packbits utilities.

.. currentmodule:: flashinfer.quantization

Types and Enums
---------------

.. autosummary::
    :toctree: ../generated

    SfLayout

Packbits Utilities
------------------

.. autosummary::
    :toctree: ../generated

    packbits
    segment_packbits

FP4 Quantization
----------------

Core kernels for NVFP4 / MXFP4 (de)quantization and the scale-factor
layout helpers used by the FP4 GEMM/MoE pipelines.

.. autosummary::
    :toctree: ../generated

    fp4_quantize
    nvfp4_quantize
    nvfp4_batched_quantize
    mxfp4_quantize
    mxfp4_dequantize
    mxfp4_dequantize_host
    block_scale_interleave
    e2m1_and_ufp8sf_scale_to_float
    scaled_fp4_grouped_quantize
    shuffle_matrix_a
    shuffle_matrix_sf_a

.. note::

    ``flashinfer.quantization.nvfp4_block_scale_interleave`` is an alias
    for :func:`block_scale_interleave` (same Python object). Use either
    name; we document the canonical ``block_scale_interleave`` to avoid
    Sphinx ``duplicate object description`` warnings under ``-W``.

FP4 KV Cache Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~

GPU-accelerated quantization / dequantization for KV-cache data using the
linear (non-swizzled) block-scale layout.

- :func:`nvfp4_kv_dequantize`: SM80+ (Ampere and later)
- :func:`nvfp4_kv_quantize`: SM100+ (Blackwell and later)
- :func:`nvfp4_quantize_paged_kv_cache`

.. autosummary::
    :toctree: ../generated

    nvfp4_kv_quantize
    nvfp4_kv_dequantize
    nvfp4_quantize_paged_kv_cache

FP8 Quantization
----------------

.. autosummary::
    :toctree: ../generated

    mxfp8_quantize
    mxfp8_grouped_quantize
    mxfp8_dequantize_host

.. note::

    ``mxfp8_grouped_quantize`` uses a cuTile backend and requires SM100+ and
    ``cuda.tile`` (a ``requirements.txt`` dependency). ``K`` must be divisible
    by 32 and is padded internally to 128-column tiles.

CuTe-DSL Quantization Kernels (experimental)
--------------------------------------------

The CuTe-DSL backends are conditionally available when the
``nvidia-cutlass-dsl`` package is installed. At runtime they are also
re-exported as ``flashinfer.quantization.{nvfp4,mxfp4,mxfp8}_quantize_cute_dsl``
when available; documenting them here via their canonical submodule
path keeps the docs build from depending on the CuTe-DSL stack being
importable.

.. currentmodule:: flashinfer.quantization.kernels.nvfp4_quantize

.. autosummary::
    :toctree: ../generated

    nvfp4_quantize_cute_dsl
    nvfp4_quantize_per_token_cute_dsl

.. currentmodule:: flashinfer.quantization.kernels.mxfp4_quantize

.. autosummary::
    :toctree: ../generated

    mxfp4_quantize_cute_dsl

.. currentmodule:: flashinfer.quantization.kernels.mxfp8_quantize

.. autosummary::
    :toctree: ../generated

    mxfp8_quantize_cute_dsl
