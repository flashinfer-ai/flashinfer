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
Call :func:`flashinfer.msa_ops.supports_packed_kv` with the active device when
integrating a cache manager across these architectures; the legacy aggregate
``SUPPORTS_PACKED_KV`` flag describes the SM120/SM121 backend.

CUDA graph capture of sparse prefill or decode on compute capability 10.0/10.3
requires a caller-owned
:class:`flashinfer.msa_ops.MSASparseAttentionWorkspace`. Warm the workspace
eagerly with the exact tensors, options, and capture stream before capture.

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
