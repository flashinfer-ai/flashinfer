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

TF32 GEMM
---------

.. autosummary::
    :toctree: ../generated

    mm_tf32

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

Quantized-base GEMM with runtime BF16 LoRA (SM90, SM100, SM103)
---------------------------------------------------------------

``out = alpha_base * (dequant(a) @ dequant(b)^T) + lora_scale * (d @ lora_b^T) [+ bias]`` with the
BF16 LoRA factors consumed as given (Cake generated programs). ``lora_scale=None`` disables the
adapter even if ``d`` / ``lora_b`` are passed; pass a device scalar to apply it.  The NVFP4
operator also takes a prepared adapter (``lora_adapter=prepare_nvfp4_lora_adapter(lora_b,
lora_scale, alpha)``: the adapter is folded once at the switch, with SVDQuant-equivalent numerics).

.. autosummary::
    :toctree: ../generated

    mm_nvfp4_lora
    mm_fp8_lora
    prepare_nvfp4_lora_adapter
    PreparedNvFp4LoraAdapter
    StaleAdapterError

BF16 x FP4 GEMM (W4A16)
-----------------------

.. autosummary::
    :toctree: ../generated

    prepare_bf16_fp4_weights
    mm_bf16_fp4

Gathered-row BF16 x FP4 GEMV (W4A16, SM120/121)
-----------------------------------------------

Per-token GEMV over NVFP4 rows picked by ``row_ids``, each result written to
``out[m, out_col_offset + row_ids[m, j]]``; columns no slot names are left
untouched.  Rows are row-major packed E2M1 with linear (unswizzled) E4M3 block
scales, so any row can be read on its own.  Built for sparse speculative-decoding
LM heads, where each draft token computes exact logits for a few thousand
vocabulary rows chosen by ``flashinfer.top_k``.

.. autosummary::
    :toctree: ../generated

    gemv_bf16_fp4_gather

MXFP8 GEMM
----------

.. autosummary::
    :toctree: ../generated

    mm_mxfp8
    bmm_mxfp8

``mm_mxfp8`` carries a ``"specialized"`` backend: an agent-generated SM107
small-M kernel that ``backend="auto"`` selects only on its measured (M, N, K)
allowlist (``flashinfer/mm_mxfp8_specialized_sm107_workloads.json``). An
explicitly selected backend is never silently swapped and never falls back;
explicit ``backend="specialized"`` that cannot serve a call raises
``NotImplementedError`` with the refusal reason.
``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (read at call time) disables the
kernel: ``"auto"`` keeps the stock path, explicit ``backend="specialized"``
raises.

MXFP8 x MXFP4 GEMM (MXFP4 weight, MXFP8 activation)
----------------------------------------------------

.. autosummary::
    :toctree: ../generated

    mm_mxfp8_mxfp4

FP8 GEMM
--------

.. autosummary::
    :toctree: ../generated

    mm_fp8
    bmm_fp8
    gemm_fp8_nt_blockscaled
    gemm_fp8_nt_groupwise
    group_gemm_fp8_nt_groupwise
    group_gemm_fp8_nt_groupwise_contiguous
    group_deepgemm_fp8_nt_groupwise
    batch_deepgemm_fp8_nt_groupwise
    fp8_blockscale_gemm_sm90

16-bit x FP8 GEMM (FP8 weight, bf16 / fp16 activation)
-------------------------------------------------------

.. autosummary::
    :toctree: ../generated

    mm_bf16_fp8
    mm_bf16_fp8_workspace_bytes
    mm_fp16_fp8
    mm_fp16_fp8_workspace_bytes

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

cuTile GEMM
-----------

.. autosummary::
    :toctree: ../generated

    gemm_alpha_beta
    masked_bmm
    masked_scaled_bmm
    ragged_bmm
    ragged_block_scaled_bmm
    ragged_scaled_bmm

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

PrimsTS dense FP8 and NVFP4 GEMM (SM100, SM103, SM107)
------------------------------------------------------

.. autosummary::
    :toctree: ../generated

    fp8_linear
    fp8_linear_swiglu
    fp8_qkv_qknorm_rope
    fp4_linear
    fp4_linear_swiglu
    fp4_qkv_qknorm_rope
