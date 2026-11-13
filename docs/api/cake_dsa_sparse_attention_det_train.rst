.. _apicake_dsa_sparse_attention_det_train:

flashinfer.cake_dsa_sparse_attention_det_train
==============================================

Cake DeepSeek Sparse Attention training (top-k sparse MLA with absorbed
queries, 64 query heads): forward and backward for SM100a, SM103a and SM107a.
The module is a stable API; its backend package,
``flashinfer.experimental.cake.cake_dsa_sparse_attention_det_train``, is
experimental and is imported on first use.

The mode is part of every name, and no entry takes a ``deterministic``
keyword.  Every output repeats bit for bit for the same inputs, device
architecture and package version: the forward and ``dq`` are fixed-order, and
the backward adds the dK / dV contributions as integers on a power-of-two grid
per destination row, so the order of the additions does not change the
result.  Nothing raises or warns under
``torch.use_deterministic_algorithms(True)``.  The entries take the same
arguments as those of :ref:`flashinfer.cake_dsa_sparse_attention_nondet_train
<apicake_dsa_sparse_attention_nondet_train>`.

Each entry is an autograd call.  The rope entries score the latent and rope
parts; the NoPE entries score the latent part only.  The varlen entries take
packed documents.

.. currentmodule:: flashinfer.cake_dsa_sparse_attention_det_train

.. autosummary::
    :toctree: ../generated

    cake_dsa_sparse_attention_det
    cake_dsa_sparse_attention_det_varlen
    cake_dsa_sparse_attention_det_nope
    cake_dsa_sparse_attention_det_nope_varlen
