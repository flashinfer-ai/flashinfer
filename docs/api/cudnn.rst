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
stack's SDPA engines accept it (cudnn-frontend 1.30+, whose default SM100 engine
takes the graph the backend engine declines); older frontends raise a
not-supported error at the first run.

cuDNN's CuTe-DSL ("FROST") SDPA engine for SM100 is a default engine of
cudnn-frontend 1.30+: it serves multi-token decode rows with a decode tile, and
an attention sink at ``q_len_per_req == 1`` falls through to it when the backend
engine declines. No environment variable is involved. With that frontend
installed, the decode wrapper's ``backend="auto"`` resolves to ``cudnn`` on
SM100 / SM103 for the multi-token rows (``2 <= q_len_per_req <= 4``) of fp16/bf16
head_dim-128 GQA models when that tile has at least 32 packed rows per CTA and 64
CTAs, where it measures at 0.35-0.95x fa2 (multi-token rows without tensor cores,
which have no fa2 kernel, take cuDNN whenever its decode path can run them), and
for single-token decode of fp16/bf16 head_dim-256 GQA models (groups 4, 8 and 16)
with 96 to 256 (batch x KV heads) CTAs, where the d256 decode tile measures at
0.5-0.95x fa2; ``FLASHINFER_DECODE_AUTO_CUDNN`` overrides the choice. Under CUDA
graphs ``auto`` takes cuDNN only with a caller-owned ``block_tables`` (the
auto-built table cannot grow once captured), and the resolution is frozen after
the first plan. A CUDA-graph wrapper also tells cudnn-frontend 1.31+ that the
graph is replayed (``pygraph(is_cuda_graph_replay_expected=True)``), so the d256
decode tile leads with its split-KV plan, and the single-token d256 band then
starts at 64 CTAs.

Compatible decode runs and replans retain the prepared cuDNN graph. Planning
still stages changing KV lengths and, unless the caller supplies a dense GPU
``block_tables``, constructs that table from CSR metadata. CUDA Graph replay
does not include this host planning work. ``fast_decode_plan`` uses the regular
cuDNN planner for these updates; its FA2/FA3 copy-elision does not apply to
cuDNN. ``workspace_size`` currently raises for both explicit and auto-selected
cuDNN, rather than returning another backend's workspace requirements.

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
