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