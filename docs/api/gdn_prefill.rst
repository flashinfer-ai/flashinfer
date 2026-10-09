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

SM12x FP16-accumulate MMA mode
------------------------------

On GeForce Blackwell (SM12x, for example the RTX 5090) FP32-accumulate tensor-core
``mma.sync`` instructions issue at half the rate of FP16-accumulate ones. Setting
``FLASHINFER_GDN_FP16_ACCUM_MMA=1`` (default off) switches the CuTe-DSL SM120
prefill kernels, that is the non-CP kernel and the CP T precompute, MN precompute
and prefill stages (the state fixup is unchanged), to FP16-accumulate
``m16n8k16`` MMAs: a few K steps accumulate in FP16 and the partial sum is carried
into an FP32 accumulator. The recurrent state, the output accumulation and the
dtypes and layouts of the output, the final state and the checkpoints are
unchanged.

Every matrix-multiply operand, including the intermediates (recurrent state,
``V - S K``, the in-chunk inverse and the attention weights), is rounded to FP16 and
BF16 inputs are converted on the fly, so inputs and intermediates must stay within
the FP16 range (``|x| <= 65504``). The variable is ignored on other architectures
and for ``backend`` other than ``"auto"`` and ``"flashinfer"``.

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
