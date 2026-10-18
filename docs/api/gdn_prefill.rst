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

Cake GDN non-CP backend: host-free launch
-----------------------------------------

``chunk_gated_delta_rule(..., backend="cake_gdn")`` with ``use_cp=False`` or
``"auto"`` runs the Cake non-CP prefill kernels on SM100a/SM103a.  The route
performs no device-to-host synchronization: the sequence count and the token
count are tensor shapes, the kernel variant and the persistent grid follow an
upper bound on the longest sequence, and the kernel resolves ``cu_seqlens``
on the device at every launch (offsets are clamped into
``[0, total_seq_len]``; a non-monotonic pair is an empty sequence, a
documented no-op rather than an out-of-bounds access).

Pass the host copy of the offsets an engine already holds as
``cu_seqlens_cpu`` (keyword-only; a CPU int32/int64 tensor or a sequence of
Python ints of length ``num_seqs + 1`` -- vLLM's ``query_start_loc_cpu``, the
prefix sums of SGLang's ``extend_seq_lens_cpu``).  The route then validates
the offsets on the host before any launch and uses the exact longest sequence
to pick the kernel and grid.  Without it, ``max_seqlen`` (or
``total_seq_len``) is the bound; a loose bound is always correct and only
affects the grid size and whether the stateless single-chunk kernel is
eligible.  Other backends ignore ``cu_seqlens_cpu``.

Int32 ``cu_seqlens`` / ``state_indices`` / ``checkpoint_cu_starts`` reach the
kernel as is; int64 metadata is cast on the stream at every call.  Nothing is
cached by tensor identity, so a metadata buffer rewritten in place is always
read live.  CUDA Graph capture needs no eager warm-up call for the same
metadata: the one-time ``nvcc`` compile and load of a kernel variant is the
only eager step, and replay reads the live buffers.  The consistency of
``checkpoint_cu_starts`` with ``cu_seqlens`` and the checkpoint interval
remains a caller precondition.  See ``csrc/gdn/cake/README.md`` ("Non-CP
prefill: host-free launch") for the full contract.

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
