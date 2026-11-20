.. _apicake_fused_qk_rope_fp8_append:

flashinfer.cake_fused_qk_rope_fp8_append
========================================

``cake_fused_qk_rope_fp8_append`` is the FP8 variant of
:mod:`flashinfer.cake_fused_qk_rope_append`: in one launch it applies the per-head RMSNorm of the
query and key projections and the NeoX-style rotary embedding, quantizes the query to FP8 E4M3
with a per-row scale, and appends the quantized key and value rows into a paged FP8 KV cache.  It
serves prefill and decode (``head_dim`` 128; 8 query heads with 1 KV head, or 64 with 8) on SM90a,
SM100a and SM103a GPUs, writes only into caller-owned buffers, and is CUDA Graph capturable.
:func:`launch_plan` exposes the host-side stage selection.

.. currentmodule:: flashinfer.cake_fused_qk_rope_fp8_append

.. autosummary::
    :toctree: ../generated

    cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache
    launch_plan
