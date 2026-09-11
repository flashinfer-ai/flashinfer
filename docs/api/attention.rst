.. _apiattention:

FlashInfer Attention Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Experimental Task-Scheduled Attention
=====================================

The experimental Blackwell task-scheduled FMHA context, FMHA decode,
block-sparse FMHA, and MLA decode APIs are imported from
``flashinfer.attention.prims_ts``. Scheduling, tile selection, and split-KV
reduction are automatic implementation details; there are no public tuning
knobs.

See the `PrimTS guide index <https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/attention/prims_ts/README.md>`_
for the public entry points, supported contracts, and examples. Current accuracy
and performance signoff is on SM100a/B200; SM103a/B300 is architecture-gated
but not yet signoff-qualified.

.. currentmodule:: flashinfer.attention.prims_ts

FMHA Context/Prefill
--------------------

.. autosummary::
    :toctree: ../generated

    batch_prefill
    batch_prefill_with_paged_kv_cache

.. autoclass:: BatchPrefillTSWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: BatchPrefillPagedTSWrapper
    :members:

    .. automethod:: __init__

FMHA Decode
-----------

.. autosummary::
    :toctree: ../generated

    batch_decode_with_paged_kv_cache
    get_prims_ts_batch_decode_workspace_size
    prims_ts_batch_decode_with_kv_cache

.. autoclass:: BatchDecodePagedTSWrapper
    :members:

    .. automethod:: __init__

Block-Sparse FMHA
-----------------

.. autosummary::
    :toctree: ../generated

    block_sparse_attention
    block_sparse_attention_with_paged_kv_cache

.. autoclass:: BlockSparseTSWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: BlockSparsePagedTSWrapper
    :members:

    .. automethod:: __init__

MLA Decode
----------

.. autosummary::
    :toctree: ../generated

    batch_decode_mla_with_paged_kv_cache
    get_prims_ts_batch_decode_mla_workspace_size
    prims_ts_batch_decode_with_kv_cache_mla

.. autoclass:: BatchMLADecodePagedTSWrapper
    :members:

    .. automethod:: __init__


flashinfer.decode
=================

.. currentmodule:: flashinfer.decode

Single Request Decoding
-----------------------

.. autosummary::
    :toctree: ../generated

    single_decode_with_kv_cache
    single_decode_with_kv_cache_with_jit_module

Batch Decoding
--------------

.. autosummary::
    :toctree: ../generated

    cudnn_batch_decode_with_kv_cache
    trtllm_batch_decode_with_kv_cache
    xqa_batch_decode_with_kv_cache

DCP Speculative Decode Workspace
--------------------------------

The native Cake FMHA DCP speculative route of
:func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache` uses caller-owned
scratch buffers so a prewarmed invocation can be captured in a CUDA Graph.
The production D256 FP8/page64 ratio-16 profile supports speculative query
lengths 1 through 8 and passes ``head_dim=256`` to the workspace-size helper;
D128 remains the default.

.. currentmodule:: flashinfer

.. autosummary::
    :toctree: ../generated

    get_dcp_spec_workspace_size_bytes
    get_dcp_spec_counter_bytes

.. currentmodule:: flashinfer.decode

.. autoclass:: BatchDecodeWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: BatchDecodeMlaWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    :members:

    .. automethod:: __init__


XQA
---

.. currentmodule:: flashinfer.xqa

.. autosummary::
    :toctree: ../generated

    xqa
    xqa_mla

flashinfer.prefill
==================

Attention kernels for prefill & append attention in both single request and batch serving setting.

.. currentmodule:: flashinfer.prefill

Single Request Prefill/Append Attention
---------------------------------------

.. autosummary::
    :toctree: ../generated

    single_prefill_with_kv_cache
    single_prefill_with_kv_cache_return_lse
    single_prefill_with_kv_cache_with_jit_module

Batch Prefill/Append Attention
------------------------------

.. autosummary::
    :toctree: ../generated

    cudnn_batch_prefill_with_kv_cache
    trtllm_batch_context_with_kv_cache
    trtllm_ragged_attention_deepseek
    fmha_v2_prefill_deepseek
    trtllm_fmha_v2_prefill
    fmha_v2_prefill_sm120

.. autoclass:: BatchPrefillWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: BatchPrefillWithRaggedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__


Unified BatchAttention
----------------------

.. currentmodule:: flashinfer.attention

The ``BatchAttention`` class provides a holistic attention wrapper that automatically dispatches
between paged-prefill and paged-decode based on per-request sequence lengths. It is the
recommended entry point for serving stacks that batch mixed prefill/decode requests in a
single kernel launch.

.. autoclass:: BatchAttention
    :members:

    .. automethod:: __init__

.. autoclass:: BatchAttentionWithAttentionSinkWrapper
    :members:

    .. automethod:: __init__


SM120 NVFP4 Attention
---------------------

.. currentmodule:: flashinfer.nvfp4_attention_sm120

.. autosummary::
    :toctree: ../generated

    nvfp4_attention_sm120_quantize_qkv
    nvfp4_attention_sm120_fwd


flashinfer.mla
==============

MLA (Multi-head Latent Attention) is an attention mechanism proposed in DeepSeek series of models (
`DeepSeek-V2 <https://arxiv.org/abs/2405.04434>`_, `DeepSeek-V3 <https://arxiv.org/abs/2412.19437>`_,
and `DeepSeek-R1 <https://arxiv.org/abs/2501.12948>`_).

See the `Batch MLA backend architecture <https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/batch_mla_backend_architecture.md>`_
for the planned wrapper's ownership, metadata, tensor-representation,
transactionality, and hot-path contracts.

.. currentmodule:: flashinfer.mla

PageAttention for MLA
---------------------

.. autosummary::
    :toctree: ../generated

    trtllm_batch_decode_with_kv_cache_mla
    trtllm_batch_decode_sparse_mla_dsv4
    nvfp4_quantize_pack_sparse_mla_cache
    nvfp4_quantize_append_sparse_mla_cache
    convert_compressed_page_aligned_sparse_indices_to_hca_metadata
    DSV4HCAMetadata
    xqa_batch_decode_with_kv_cache_mla
    supported_sparse_mla_sm120_configs
    SparseMLASm120DecodeConfig
    SparseMLASm120Wrapper

.. note::

    With ``backend="cute-dsl"``, pass ``hca_swa_indices`` as absolute rows into
    the flattened SWA cache and ``hca_compressed_block_tables`` as physical
    compressed-cache page IDs. The SWA table has shape ``[B * Q, 128]`` and may
    express ring rotation or wraparound. Combined tables whose compressed
    segment is a canonical page expansion can opt into compatibility conversion
    with ``hca_sparse_indices_format="compressed-page-aligned"``. SWA entries
    remain arbitrary absolute rows. Precompute that conversion before a CUDA
    Graph or a latency-sensitive loop.

.. autoclass:: BatchMLAPagedAttentionWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: MLAPlanMetadata
    :members:

    ``BatchMLAPagedAttentionWrapper.plan`` accepts the preferred
    ``metadata=MLAPlanMetadata.csr(...)`` or
    ``metadata=MLAPlanMetadata.dense(...)`` keyword form.  The keyword form
    defaults run inputs to packed ``query`` / ``kv_cache`` structural tensors.
    The legacy flat CSR and dense metadata adapters remain available for
    compatibility, emit a ``DeprecationWarning`` once per process, and default
    run inputs to split ``q_nope`` / ``q_pe`` and ``ckv_cache`` / ``kpe_cache``.
    Structural run inputs use an exact tuple grammar: a tensor is packed,
    ``(left, right)`` is split, and ``(packed, (left, right))`` or
    ``((left, right), packed)`` is a trusted redundant form.  The selected form
    must match the planned rank, leading shape, dtype, device, and split widths.

    LSE mode, output dtype/scaling, and KV scaling are plan/run contracts.  A
    run that needs different values must re-plan first, except that deprecated
    flat CSR FA2/FA3 plans temporarily preserve dynamic LSE behavior with a
    ``DeprecationWarning``.  Explicit ``backend="cutlass"`` callers that omit
    ``plan`` remain supported through a deprecated compatibility adapter when
    ``kv_len`` and ``page_table`` are supplied.

CUDA graph plan updates
-----------------------

``BatchMLAPagedAttentionWrapper.update_cuda_graph_plan(metadata=...)`` updates
the dynamic CSR scheduling state of an existing FA2 or FA3 CUDA graph plan
without changing captured buffer addresses.  Construct the wrapper with
``use_cuda_graph=True`` and ``enable_cuda_graph_plan_update=True``, provide its
reserved graph metadata buffers, complete one successful ``plan()``, capture
``run()``, and call the update outside active CUDA graph capture before replay.
The update flag is a temporary compatibility opt-in so legacy graph callers do
not retain update state they never use; it may become the default after the
legacy private replanning bridge is retired.  The first call that passes the
wrapper's lifecycle, capability, and capture checks binds the current CUDA
stream on the wrapper device, even if backend delegation later fails.  Each
corresponding graph replay must execute on that same stream.  Cross-stream
replay and concurrent use are unsupported, and the wrapper cannot observe or
validate the stream used by an external replay.
Successful ``plan()`` starts a new stream-binding lifecycle; a failed plan
preserves the previous binding.  Before replanning on another stream, finish
the old updates and replays, then recapture ``run()`` for the new plan.
``plan()`` rejects a stream switch while the bound stream reports pending work,
using a nonblocking stream query before snapshotting or changing graph buffers.
Staging-slot events alone do not establish completion of publication or replay.
The caller must serialize all use of a wrapper; the in-progress flag only
detects some overlapping calls and is not a thread-safety lock.  Keep externally
owned CUDA streams alive throughout their bound lifecycle.

An opted-in full replan temporarily snapshots the prior schedule and all
reserved CSR buffers for rollback, including the entire ``kv_indices``
reservation.  Its temporary memory and copy cost therefore scale with reserved
capacity, not just the currently used prefix.  This cost belongs to ``plan()``;
steady ``update_cuda_graph_plan()`` calls do not take these snapshots.

The update requires a CUDA device and complete CSR ``MLAPlanMetadata``.  ``qo_indptr``,
``kv_indptr``, and ``kv_len_arr`` are host control tensors and must be
contiguous CPU ``torch.int32`` tensors.  ``kv_indices`` must be contiguous
``torch.int32`` on the wrapper device, must not overlap any capture-reserved
wrapper buffer, and must remain alive until its queued publication completes.
Page indices are not read back to the host.  The addressed page-index prefix
must fit the capture reservation; publication copies only that prefix, so the
unused reserved tail remains unchanged.

The initial plan freezes the backend and generated module, reserved tensor
identities and capacities, batch/output shape, layouts, dtypes, scale and LSE
contracts, launch geometry, ``plan_info``, and staged workspace size.  An
update that would change any frozen value fails.  FA2 and FA3 retain one device
candidate schedule, two pinned-host planner/control staging slots, and their
events during graph planning/warm-up.  When the existing device and pinned
planner workspaces can hold two staged prefixes, those schedules are disjoint
views of their unused tails; otherwise FlashInfer allocates only the missing
region.  The small control tensors remain separate.  No committed schedule or
page-index image is retained.  An update queries the existing slot events
without waiting; if both slots are busy, it fails instead of allocating or
synchronizing.
CUTLASS, cuTile, and any backend that has not explicitly opted in reject this operation.
A slot whose event cannot be recorded is poisoned; if both slots are poisoned,
call ``plan()`` again to create fresh update state.

Validation, planning, staging, and failures before native publication
submission leave the preceding live plan untouched.  Once publication has been
submitted, asynchronous CUDA execution or context failures are outside this
no-sync guarantee by design.

The private ``_cached_module`` and workspace/metadata mirrors used by older
callers remain behavior-compatible in this release, as do legacy flat/CSR
``plan()`` forms and the native planner bridge.  This compatibility bridge is
deprecated in documentation only in this release.  It emits no runtime warning
because untouched older SGLang accesses ``_cached_module`` and can promote
``DeprecationWarning`` to an exception.  The public ``plan()`` /
``update_cuda_graph_plan()`` / ``run()`` lifecycle is the replacement.
The private fast-plan bridge must not be used on a wrapper constructed with
``enable_cuda_graph_plan_update=True``.  Its CPU planner writes the same pinned
workspace used by pending public updates; using the same CUDA stream does not
order those CPU writes.  Keep legacy callers on separate, default-off wrappers.
Removing these private compatibility attributes requires a separately
announced future change and evidence that the applicable support policy no
longer includes callers that depend on them.
