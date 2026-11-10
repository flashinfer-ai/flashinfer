.. _apicake_dsa_sparse_attention_nondet_train:

flashinfer.cake_dsa_sparse_attention_nondet_train
=================================================

Cake DeepSeek Sparse Attention training (top-k sparse MLA with absorbed
queries, 64 query heads): forward and backward for SM100a, SM103a and SM107a.
The module is a stable API; its backend package,
``flashinfer.experimental.cake.cake_dsa_sparse_attention_nondet_train``, is
experimental and is imported on first use.

The mode is part of every name, and no entry takes a ``deterministic``
keyword.  The forward and ``dq`` are fixed-order.  The backward adds the dK /
dV contributions into FP32 accumulators with atomics, so those gradients are
not bitwise reproducible run to run.  Under
``torch.use_deterministic_algorithms(True)`` the backward raises PyTorch's
alert before its first backward kernel launch; with ``warn_only=True`` it
warns and runs.  It never falls back to a deterministic implementation.  The
deterministic package ``flashinfer.cake_dsa_sparse_attention_det_train`` has
the same entry points (``det`` in place of ``nondet``); its outputs repeat bit
for bit run to run.

Each entry is an autograd call.  The rope entries score the latent and rope
parts; the NoPE entries score the latent part only.  The varlen entries take
packed documents.

.. currentmodule:: flashinfer.cake_dsa_sparse_attention_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_dsa_sparse_attention_nondet
    cake_dsa_sparse_attention_nondet_varlen
    cake_dsa_sparse_attention_nondet_nope
    cake_dsa_sparse_attention_nondet_nope_varlen
