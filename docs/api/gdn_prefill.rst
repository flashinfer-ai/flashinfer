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

GDN CP context-parallel backend
-------------------------------

The GDN CP backend covers the complete legal SM100a/SM103a public context-
parallel input domain.  It preserves ``T precompute -> MN precompute -> state
fixup -> CP prefill`` and supports the public dtype, head-mapping, optional
gate/scale, int32/int64 cu-seqlens and state indices, mixed or all-empty
varlen batches, Q/K L2 normalization, checkpoints, output, and arbitrary
positive non-overlapping packed/indexed/in-place state strides.
The 120 shapes recorded in its manifest are the frozen performance map from
FlashInfer PR #4078, not an input allowlist.

On SM100a/SM103a, the ``gdn_cp`` route uses checked-in CUDA sources and is
supported with CUDA 12.8, CUDA 12.9, and CUDA 13 for FP16/BF16 inputs plus
FP32/FP16/BF16 state and FP32 checkpoints. Other SM100 context-parallel DSL
routes, including FP8 state or checkpoints, require CUDA 13 and
``nvidia-cutlass-dsl[cu13]>=4.4.2``.

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
