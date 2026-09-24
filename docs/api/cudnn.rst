.. _apicudnn:

flashinfer.cudnn
================

cuDNN-backed attention kernels. These wrappers call into NVIDIA's cuDNN runtime
for batch prefill and batch decode, and are reachable as ``backend="cudnn"`` on
``BatchPrefillWithPagedKVCacheWrapper`` / ``BatchDecodeWithPagedKVCacheWrapper``
(or directly) when cuDNN is available on the host GPU. For decode the wrapper
backend covers fp16/bf16 GQA with ``return_lse``, CUDA graphs, multi-token
decode (``q_len_per_req > 1``, bottom-right causal), a left sliding window
(``window_left``) and attention ``sinks``; it does not support RoPE, soft-cap
or fp8/NVFP4 KV. A sink at ``q_len_per_req == 1`` is served when the cuDNN
stack's SDPA engines accept it (cudnn-frontend 1.30+ with the FROST engines
enabled); the backend engine raises a not-supported error at the first run.

cuDNN's FROST (CuTe-DSL) SDPA engines are opt-in in cudnn-frontend and are what
make cuDNN decode fast on Blackwell (the d128 / d256 decode tiles, multi-token
rows, sinks at ``q_len_per_req == 1``). Set ``FLASHINFER_CUDNN_FROST_ENGINES=1``
before importing flashinfer to switch them on for the process (FlashInfer forwards
it to the frontend's ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` before its first
``import cudnn``; cudnn-frontend 1.30.0+). With the engines on, the decode
wrapper's ``backend="auto"`` resolves to ``cudnn`` on SM100 for the d128 decode
shapes where the decode tile measures at or ahead of fa2; ``FLASHINFER_DECODE_AUTO_CUDNN``
overrides that choice.

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

These wrappers gate on one thing only: cudnn-frontend 1.29+ with the
``cutedsl`` extra, the release whose ``graph.gdn`` / ``graph.gdn2`` /
``graph.gdp`` / ``graph.kda`` nodes take ``gate_domain``. There is no cuDNN backend-version floor
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
given, so a bfloat16 state pool crosses with no copy at all. The GDN and GDP
forget gates are linear-space alpha at this boundary and cross as
``gate_domain="linear"``; the GDN-2 and KDA gates are log-space on both sides.

.. autosummary::
    :toctree: ../generated

    cudnn_chunk_gated_delta_product
    cudnn_chunk_gated_delta_rule
    cudnn_chunk_gated_delta_rule2
    cudnn_recurrent_kda
