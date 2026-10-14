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

Calling these APIs is an explicit opt-in and emits an
``ExperimentalWarning`` once per decorated function. They provide no API
compatibility guarantee; generated stable API reference entries are deferred
until graduation. Logging and existing ``fi_trace`` bindings remain available.

QToken-KvBlock-Sparse-Attention
--------------------------------

QToken-KvBlock-Sparse-Attention consumes per-query
``indexer_block_ids[total_q, block_topk]`` and a dense physical
``block_table``. Packed prefill uses ``[total_q, Hq, D]`` with
``qo_indptr``; fixed MTP decode uses ``[B, Nq, G, Hq, D]``.
``kv_block_size`` is the semantic sparse K/V atom and currently supports
only four tokens. The wrapper plans capacity outside CUDA Graph capture and
runs live route metadata on the hot path.


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
optional immutable length-aware schedule with
:func:`flashinfer.plan_cake_fmha_request_ordered_paged_decode` before graph
capture.  Page-table rows must be padded to
``4 * ceil(max_seq_len / 256)`` entries, and the exact tensor/workspace binding
must be invoked once eagerly to initialize its TMA descriptors before capture.
After that prewarm, changing only the order tensor contents does not require
recapture.

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
    :exclude-members: begin_forward, forward, forward_return_lse

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
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: BatchPrefillWithRaggedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__


Causal + Bidirectional Ranges Prefill
-------------------------------------

.. currentmodule:: flashinfer.attention

A batch-prefill wrapper whose fa2 attention variant owns the whole mask:
causal, plus an inclusive per-query key span attended in both directions. The
spans are handed to :meth:`BatchPrefillWithCausalBidirectionalRangesWrapper.run`
as a compact ``int32 [total_q, 2]`` tensor and no mask is materialized, so
nothing scales with ``qo_len * kv_len``. The JIT module is specialized in the
constructor, and the inherited options the variant makes meaningless are
rejected rather than ignored.

.. autoclass:: BatchPrefillWithCausalBidirectionalRangesWrapper
    :members:

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
    trtllm_prefill_with_kv_cache_mla
    trtllm_batch_decode_sparse_mla_dsv4
    nvfp4_quantize_pack_sparse_mla_cache
    nvfp4_quantize_append_sparse_mla_cache
    dsv41_fp4_quantize_pack_sparse_mla_cache
    dsv41_fp4_quantize_append_sparse_mla_cache
    dsv41_fp8_quantize_pack_sparse_mla_cache
    dsv41_fp8_quantize_append_sparse_mla_cache
    convert_compressed_page_aligned_sparse_indices_to_hca_metadata
    DSV4HCAMetadata
    xqa_batch_decode_with_kv_cache_mla
    supported_sparse_mla_sm120_configs
    SparseMLASm120DecodeConfig
    SparseMLASm120Wrapper
    cake_sparse_mla_sm120_dsv4_nvfp4_decode
    cake_sparse_mla_sm120_dsv4_nvfp4_prefill
    cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits
    cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes
    cake_dsv4_nvfp4_rope_quantize_insert
    cake_dsv4_nvfp4_kv_rope_quantize_insert

.. note::

    With ``backend="cake"`` on SM120/SM121 (``kv_cache_format="nvfp4"``), one
    query token runs the Cake split decode kernel and several tokens run the
    decode or the single-launch prefill kernel as chosen by
    ``cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel`` (DeepSeek-V4 sparse MLA
    SM120 tracker: flashinfer#4254).

.. note::

    The NVFP4 cache that route reads is written per layer by the fused
    SM120/SM121 writers ``cake_dsv4_nvfp4_rope_quantize_insert`` (sliding-window
    pool: GPT-J RoPE of the query and latent KV, head-padded ``q_out`` or, with
    ``q_inplace=True`` and no head padding, the query rotated in place, NVFP4
    quantization and paged insert in one launch) and
    ``cake_dsv4_nvfp4_kv_rope_quantize_insert`` (compressed pool with
    ``compress_ratio`` 1 or 2, speculative context with ratio 1); both produce
    the bytes of ``nvfp4_quantize_append_sparse_mla_cache`` applied to the
    BF16-rounded roped rows.

.. note::

    With ``backend="cute-dsl"``, pass ``hca_swa_indices`` as absolute rows into
    the flattened SWA cache and ``hca_compressed_block_tables`` as physical
    compressed-cache page IDs. The SWA table has shape ``[B * Q, 128]`` and may
    express ring rotation or wraparound. Combined tables whose compressed
    segment is a canonical page expansion can opt into compatibility conversion
    with ``hca_sparse_indices_format="compressed-page-aligned"``. SWA entries
    remain arbitrary absolute rows. Precompute that conversion before a CUDA
    Graph or a latency-sensitive loop.

.. note::

    ``kv_cache_format="nvfp4"`` (the 384-byte-per-token DeepSeek-V4 NVFP4 sparse
    cache: 448 NoPE values as E2M1 with one E4M3 scale per 16 values, 64 BF16
    RoPE values, ``page_size * 352`` data bytes followed by ``page_size * 32``
    scale bytes per page) is consumed by ``backend="sparse"`` on SM120 / SM121 and
    by ``backend="cake"`` on SM100 / SM103 (B200 / GB300). The CAKE route takes a
    BF16 query, two independent tables (``sparse_indices`` over ``swa_kv_cache``
    and ``extra_sparse_indices`` over ``compressed_kv_cache`` with their own
    ``*_topk_lens``; ``-1`` entries are masked), ``sinks``, a caller-owned
    ``workspace_buffer`` sized by
    :func:`flashinfer.mla.cake_dsv4.get_cake_dsv4_workspace_bytes` and is
    CUDA-Graph safe. Build the cache with
    :func:`nvfp4_quantize_pack_sparse_mla_cache` /
    :func:`nvfp4_quantize_append_sparse_mla_cache` (one implementation for all
    four architectures). ``backend="auto"`` keeps selecting TRTLLM-GEN on
    SM100 / SM103; pass ``backend="cake"`` explicitly.

.. autoclass:: BatchMLAPagedAttentionWrapper
    :members:

    .. automethod:: __init__
