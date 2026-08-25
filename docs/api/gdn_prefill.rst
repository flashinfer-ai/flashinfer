.. _apigdn_prefill:

flashinfer.gdn_prefill
======================

Gated Delta-Rule prefill-side kernels. ``chunk_gated_delta_rule`` is the
chunked GDN scan that initializes the recurrent state from a full prompt
before the decode loop takes over.

.. currentmodule:: flashinfer.gdn_prefill

.. autosummary::
    :toctree: ../generated

    chunk_gated_delta_rule

Source-only Blackwell context-parallel backend
----------------------------------------------

The source-only backend covers the complete legal SM100a/SM103a public context-
parallel input domain.  It preserves ``T precompute -> MN precompute -> state
fixup -> CP prefill`` and supports the public dtype, head-mapping, optional
gate/scale, cu-seqlens, output, and packed/indexed/padded/in-place state forms.
The 120 shapes recorded in its manifest are the frozen performance map from
FlashInfer PR #4078, not an input allowlist.

Internally, the public dispatcher caches the shape-specific plan and workspace
but launches the native composite directly, so API-allocated outputs and
rotating input buffers can change address without rebuilding the plan. A
fixed-address internal prepared object can use CUDA Graph replay; indexed
in-place state always stays on the direct composite so preparation cannot
advance aliased recurrent state. Invalid inputs and unsupported architectures
fail closed; legal SM100a/SM103a calls selected for CP do not fall back to an
external CP implementation. The existing ``chunk_gated_delta_rule`` API
remains the only public entry point. Every TMA-backed stage passes its
descriptor through CUDA's ``__grid_constant__`` kernel-argument ABI; the
backend does not retain a process-lifetime descriptor arena.
