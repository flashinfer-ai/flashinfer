.. _apisparse:

flashinfer.sparse
=================

Kernels for block sparse flashattention.

.. currentmodule:: flashinfer.sparse

.. autoclass:: BlockSparseAttentionWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__


.. autoclass:: VariableBlockSparseAttentionWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/examples/flashinfer-variable-block-sparse.png
        :width: 600
        :alt: variable block sparse attention plan function diagram
        :align: center

    .. automethod:: __init__


flashinfer.msa_ops
==================

Minimax Sparse Attention (MSA) sparse prefill, sparse decode, and top-k
selection dispatch on compute capability 10.0/10.3 (SM100/SM103) and
SM120/SM121 Blackwell GPUs. The proxy-score operations remain SM120/SM121
only. NVFP4 K/V and views split from a packed paged K/V cache are also
SM120/SM121-only; the compute capability 10.0/10.3 attention backend requires
separate contiguous K and V tensors and does not make implicit copies.
The compute capability 10.0/10.3 backend uses TopK16 as its generic contract
and additionally retains four shape-exact routes: paged BF16 decode at
B64/Q8/KV65536/TopK32, 512-thread paged BF16 decode at
B2/Q1/KV257/TopK4, flat
BF16-query/FP8-KV prefill at B3/Q1024/KV8192/TopK8, and paged BF16 prefill at
B3/Q4096/KV8192/TopK4. Neighboring non-TopK16 shapes fail closed instead of
entering a generic kernel. The decode path uses direct persistent M16
ownership for both Q1 and multi-token decode; it does not route BF16 decode
through prefill or split-K.
Frozen BF16-query/FP8-KV Q1 serving shapes use exact or transformed direct
kernels, while paged uniform FP8 Q/K/V supports Q1 through Q32 and returns
BF16 output. Long batch-one BF16 causal prefill uses a selected-block reverse
producer and deterministic reduction once the query reaches 8192 tokens.
Call :func:`flashinfer.msa_ops.supports_packed_kv` with the active device when
integrating a cache manager across these architectures; the legacy aggregate
``SUPPORTS_PACKED_KV`` flag describes the SM120/SM121 backend.
Per-token tensor ``num_valid_pages`` for
:func:`flashinfer.msa_ops.msa_topk_select` is likewise SM120/SM121-only;
compute capability 10.0/10.3 requires a scalar value or ``None`` and rejects
the tensor form before backend dispatch.

CUDA graph capture of sparse prefill or decode on compute capability 10.0/10.3
requires a caller-owned
:class:`flashinfer.msa_ops.MSASparseAttentionWorkspace`. Warm the workspace
eagerly with the exact tensors, options, and capture stream before capture.
The exact decode overrides are eager-only. The exact TopK8 reverse-prefill
route is also eager-only because its reducer uses a host-owned monotonic launch
generation. The exact paged TopK4 route normally captures its producer and
reducer into an internal two-node CUDA graph; while an outer CUDA graph is
being captured it emits the two kernel nodes directly.

Normal callers should leave the Blackwell schedule environment variables
unset. For advanced diagnostics and benchmarking,
``FLASHINFER_MSA_PREFILL_SCHEDULE=m64`` forces the eligible M64 prefill
schedule. ``FLASHINFER_MSA_FP8_Q1_SCHEDULE`` can select
``batch_attention``, ``q1_exact``, ``q1_flat_xform2``,
``q1_paged_xform2``, or ``paged_uniform_fp8`` for an eligible FP8 Q1 route.
Unsupported values, or a schedule incompatible with the input layout and
dtypes, raise :class:`ValueError`.

.. currentmodule:: flashinfer.msa_ops

.. autosummary::
    :toctree: ../generated

    msa_proxy_score
    msa_proxy_score_fp4
    MSASparseAttentionWorkspace
    supports_packed_kv
    msa_sparse_attention
    msa_sparse_decode_attention
    msa_topk_select
