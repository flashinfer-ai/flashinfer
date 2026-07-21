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

Minimax Sparse Attention (MSA) APIs for SM120/SM121 (Blackwell).

.. currentmodule:: flashinfer.msa_ops

.. autosummary::
    :toctree: ../generated

    msa_proxy_score
    msa_proxy_score_fp4
    msa_sparse_attention
    msa_sparse_decode_attention
    msa_topk_select
