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

Cake GDN CP context-parallel backend
------------------------------------

Enable Cake CP explicitly with
``chunk_gated_delta_rule(..., backend="cake_gdn", use_cp=True)``.
``backend="auto"`` and ``backend="flashinfer"`` keep the existing FlashInfer
CP routing. With ``backend="cake_gdn"``, ``use_cp=False`` or ``"auto"``
continues to select the non-CP Cake backend.

The Cake GDN CP backend covers the complete legal SM100a/SM103a public context-
parallel input domain.  It preserves ``T precompute -> MN precompute -> state
fixup -> CP prefill`` and supports the public dtype, head-mapping, optional
gate/scale, int32/int64 cu-seqlens and state indices, mixed or all-empty
varlen batches, Q/K L2 normalization, checkpoints, output, and arbitrary
positive non-overlapping packed/indexed/in-place state strides.
The runtime manifest records only the source inventory and loading metadata.
The historical 120-shape performance map and export provenance are preserved
in ``tests/gdn/data/cake_gdn_cp_export_manifest.json``; they do not restrict
the supported inputs.

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
fail closed; explicit Cake CP requests do not fall back to another
CP implementation. The existing ``chunk_gated_delta_rule`` API
remains the only public entry point. Every TMA-backed stage passes its
descriptor through CUDA's ``__grid_constant__`` kernel-argument ABI; the
backend does not retain a process-lifetime descriptor arena.
