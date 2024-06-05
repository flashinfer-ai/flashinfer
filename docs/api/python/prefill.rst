.. _apiprefill:

flashinfer.prefill
==================

Attention kernels for prefill & append attention in both single request and batch serving setting.

.. currentmodule:: flashinfer.prefill

Single Request Prefill/Append Attention
---------------------------------------

.. autosummary::
    :toctree: ../../generated

    single_prefill_with_kv_cache
    single_prefill_with_kv_cache_return_lse

Batch Prefill/Append Attention
------------------------------

.. autoclass:: BatchPrefillWithPagedKVCacheWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: BatchPrefillWithRaggedKVCacheWrapper
    :members:
    
    .. automethod:: __init__
