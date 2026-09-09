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

SVDQuant NVFP4 GEMM (SM100)
---------------------------

.. autosummary::
    :toctree: ../generated

    mm_nvfp4_svdquant
    nvfp4_quantize_smooth
    svdquant_linear

BF16 x FP4 GEMM (W4A16)
-----------------------

.. autosummary::
    :toctree: ../generated

    prepare_bf16_fp4_weights
    mm_bf16_fp4

BF16 x Dual-BF16 Weight GEMM (SM100)
------------------------------------

The dual-BF16 representation stores an FP32 weight as two contiguous BF16
matrices and reconstructs it as ``weight_high + weight_low / 256`` inside the
GEMM. Prepare weights once at model load time. The compute API supports BF16 or
FP32 output and accepts an optional caller-owned workspace for CUDA Graph and
multi-stream use.

.. autosummary::
    :toctree: ../generated

    prepare_dual_bf16_weights
    dual_bf16_weight_gemm_workspace_size
    mm_bf16_dual_weight

Benchmark the kernel against the strict FP32 PyTorch/cuBLAS baseline (weight
preparation and the BF16-to-FP32 activation conversion are excluded from both
timings):

.. code-block:: bash

    python benchmarks/flashinfer_benchmark.py \
        --routine mm_bf16_dual_weight \
        --backends dual-bf16 cublas \
        --m 512 --n 192 --k 4096 \
        --out_dtype float32 --refcheck

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

Low-latency TRT-LLM FP8 GEMM weight prep (also exported from ``flashinfer``):

.. currentmodule:: flashinfer.trtllm_low_latency_gemm

.. autosummary::
    :toctree: ../generated

    prepare_low_latency_gemm_weights

.. currentmodule:: flashinfer.gemm


Mixed Precision GEMM (fp8 x fp4)
--------------------------------

.. autosummary::
    :toctree: ../generated

    group_gemm_mxfp8_mxfp4_nt_groupwise
    group_gemm_nvfp4_nt_groupwise

Router GEMM (DeepSeek-V3 / Mistral / GLM / Kimi-K2 / Kimi-K3)
-------------------------------------------------------------

.. autosummary::
    :toctree: ../generated

    mm_M1_16_K7168_N128
    mm_M1_16_K7168_N256
    mm_M1_16_K6144_N256
    mm_M1_16_K7168_N256_bf16
    mm_M1_16_K7168_N384
    mm_M1_16_K7168_N384_bf16
    mm_M1_16_K7168_N896
    mm_M1_16_K7168_N896_bf16
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
