.. _apiattention:

FlashInfer Attention Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Experimental Task-Scheduled Attention
=====================================

The experimental Blackwell task-scheduled FMHA context, FMHA decode, and MLA
decode APIs are imported from ``flashinfer.attention.prims_ts``. Scheduling,
tile selection, and split-KV reduction are automatic implementation details;
there are no public tuning knobs.

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
for the internal ownership, planning, representation, selection, and hot-path
contracts shared by the wrapper and functional APIs.

.. currentmodule:: flashinfer.mla

PageAttention for MLA
---------------------

.. autosummary::
    :toctree: ../generated

    batch_mla_paged_attention
    trtllm_batch_decode_with_kv_cache_mla
    trtllm_batch_decode_sparse_mla_dsv4
    convert_compressed_page_aligned_sparse_indices_to_hca_metadata
    DSV4HCAMetadata
    xqa_batch_decode_with_kv_cache_mla

.. note::

    With ``backend="cute-dsl"``, pass ``hca_swa_indices`` as absolute rows into
    the flattened SWA cache and ``hca_compressed_block_tables`` as physical
    compressed-cache page IDs. The SWA table has shape ``[B * Q, 128]`` and may
    express ring rotation or wraparound. Combined tables whose compressed
    segment is a canonical page expansion can opt into compatibility conversion
    with ``hca_sparse_indices_format="compressed-page-aligned"``. SWA entries
    remain arbitrary absolute rows. Precompute that conversion before a CUDA
    Graph or a latency-sensitive loop.

``xqa_batch_decode_with_kv_cache_mla`` is deprecated. Use
``batch_mla_paged_attention(..., backend="xqa")`` instead.

.. autoclass:: BatchMLAPagedAttentionWrapper
    :members:

    .. automethod:: __init__
