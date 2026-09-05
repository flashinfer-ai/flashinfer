.. _apicudnn:

flashinfer.cudnn
================

cuDNN-backed attention kernels. These wrappers call into NVIDIA's cuDNN runtime
for batch prefill and batch decode, and are typically used as an alternative
backend for ``BatchPrefillWithPagedKVCacheWrapper`` /
``BatchDecodeWithPagedKVCacheWrapper`` when cuDNN is available on the host GPU.

.. currentmodule:: flashinfer.cudnn

.. autosummary::
    :toctree: ../generated

    cudnn_batch_decode_with_kv_cache
    cudnn_batch_prefill_with_kv_cache

Linear attention
----------------

cuDNN's fused SM100 linear-attention engines, reachable either directly or as
``backend="cudnn"`` on :func:`flashinfer.chunk_gated_delta_rule`,
:func:`flashinfer.chunk_gated_delta_rule2`,
:func:`flashinfer.chunk_gated_delta_product` and
:func:`flashinfer.recurrent_kda`. ``"cudnn"`` is never selected implicitly for
GDN or KDA, both of which have FlashInfer kernels of their own; GDN-2 and GDP
have none, so their ``"auto"`` resolves here.

These wrappers gate on one thing only: cudnn-frontend 1.28+ with the
``cutedsl`` extra, the release that first carries the ``graph.gdn`` /
``graph.gdn2`` / ``graph.gdp`` / ``graph.kda`` nodes. There is no cuDNN backend-version floor
-- the FROST engines behind those nodes are CuTeDSL kernels the frontend
compiles itself. Every other requirement, including the SM100 family
(SM100-SM103 and SM107), the head dims, the input dtypes and the head-count
relations, belongs to the engine, which declines a graph it cannot serve (the
per-engine reason is logged by the frontend; the raised
``cudnnGraphNotSupportedError`` itself is generic). Arguments
FlashInfer has that cuDNN's entry points do not -- state checkpointing, indexed
state pools, the context-parallel delta rule, speculative decode -- are
rejected by the routing layer before the call.

The recurrent state crosses this boundary untransposed. FlashInfer holds it
V-major as ``[N, H, V, K]`` and so does cuDNN, so ``initial_state`` and
``output_state`` buffers are passed straight through. cuDNN's ops take the
state in float32 or bfloat16 and return ``final_state`` in whichever was
given, so a bfloat16 state pool crosses with no copy at all. The gate domain
does differ: the GDN, GDN-2 and GDP forget gates are linear-space alpha at
this boundary and log-space in cuDNN, so the wrapper takes a log.

.. autosummary::
    :toctree: ../generated

    cudnn_chunk_gated_delta_product
    cudnn_chunk_gated_delta_rule
    cudnn_chunk_gated_delta_rule2
    cudnn_recurrent_kda
