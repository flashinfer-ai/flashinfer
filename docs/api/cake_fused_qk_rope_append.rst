.. _apicake_fused_qk_rope_append:

flashinfer.cake_fused_qk_rope_append
====================================

``cake_fused_qk_rope_append`` fuses, in one launch, the per-head RMSNorm of the query and key
projections, the NeoX-style rotary embedding of both, and the append of the new key and value rows
into a paged BF16 KV cache.  It serves prefill and decode (``head_dim`` 128; 8 query heads with
1 KV head, or 64 with 8) on SM90a, SM100a and SM103a GPUs, writes only into caller-owned buffers,
and is CUDA Graph capturable.  :func:`launch_plan` exposes the host-side stage selection.

.. currentmodule:: flashinfer.cake_fused_qk_rope_append

.. autosummary::
    :toctree: ../generated

    cake_fused_qk_rmsnorm_rope_append_paged_kv_cache
    launch_plan
