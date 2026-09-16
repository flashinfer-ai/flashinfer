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

    batch_mla_decode_with_paged_kv_cache
    get_prims_ts_batch_mla_decode_workspace_size
    prims_ts_batch_mla_decode_with_kv_cache

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
It is also reachable through
:func:`flashinfer.cake_fmha.cake_batch_decode_with_kv_cache`; the non-null
``causal_seqlens_kv_global`` argument is the explicit add-on selection key.
On SM103, the same Cake entrypoint accepts a device ``request_order`` tensor
for BF16-query, FP8-E4M3 paged decode with head dimension 256.  Precompute an
optional immutable launch plan with
:func:`flashinfer.plan_cake_fmha_request_ordered_paged_decode` before graph
capture. The BF16 Q1/Q6, 8-query-head/1-KV-head route uses a kernel-owned
runtime-length scheduler for batches 1 through 256, with native page-table
rows containing ``ceil(max_seq_len / 64)`` slots. Other request-ordered
geometries retain their route-specific page padding requirements. Follow
:doc:`cake_fmha` for workspace preparation and capture ownership; device
lengths and request order remain mutable within the selected plan's contract.

The planner's optional ``num_kv_splits`` selects an exported split schedule;
omitting it preserves the ordinary schedule selection. For 32 query heads and
2 KV heads, the generated schedules include Q1 with 4 splits and Q6 with 2
splits when ``write_lse=True``, and fused Q6 with 6 splits when
``write_lse=False``. An unavailable combination raises ``ValueError``.
Use the returned plan unchanged: fused and unfused schedules have different
launch layouts, and device lengths and request order remain mutable on replay.

Split plans require a zero-initialized ``multi_ctas_kv_counter_buffer`` with
at least ``batch * q_len * (num_q_heads // 8)`` int32 elements. Their uint8
workspace must contain at least ``plan.workspace_size_bytes`` bytes, which
uses the selected module's scratch layout. In particular, the B1/Q6 route
with 32 query heads, two KV heads and 76 splits requires 43,671,552 bytes;
the generic split formula underestimates this route's storage. When ``plan.write_lse`` is true, also pass a contiguous FP32 ``lse``
tensor with shape ``[batch * q_len, num_q_heads]``. Allocate these buffers
before capture and retain them with the graph. Query tensors produced inside
capture use :class:`flashinfer.cake_fmha.CakeFmhaRequestOrderedCapture` and its
normal finalization step before replay.

.. currentmodule:: flashinfer

.. autosummary::
    :toctree: ../generated

    get_dcp_spec_workspace_size_bytes
    get_dcp_spec_counter_bytes
    plan_cake_fmha_request_ordered_paged_decode
    CakeFmhaRequestOrderedDecodePlan

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
