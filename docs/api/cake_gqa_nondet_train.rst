.. _apicake_gqa_nondet_train:

flashinfer.cake_gqa_nondet_train
================================

Cake non-deterministic GQA attention training for head dimension 256: a varlen
causal forward and backward for SM100a, SM103a and SM107a.  The module is a
stable API; its backend package,
``flashinfer.experimental.cake.cake_gqa_nondet_train``, is experimental and is
imported on first use.

The mode is part of every name, and no entry takes a ``deterministic``
keyword.  Every sum of the forward and the backward has one fixed order: the
backward stores dS^T into a workspace chunk sized by the workspace's
``budget_bytes`` (default 16 GiB), sums dK and dV in tensor memory, computes
dQ as a GEMM over the workspace and folds row-map destinations in window
order.  ``out``, ``lse``, dQ and every dK / dV form therefore repeat bit for
bit run to run for one shape, one ``max_seqlen_k`` and one budget, and
nothing raises PyTorch's determinism alert.

``cake_gqa_nondet`` is the autograd entry (its Function is
``CakeGQANondetFunction``); the forward and backward halves, the workspace and
its size are separate entries.

.. currentmodule:: flashinfer.cake_gqa_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_gqa_nondet
    cake_gqa_nondet_train_forward
    cake_gqa_nondet_train_backward
    cake_gqa_nondet_train_workspace
    cake_gqa_nondet_train_workspace_bytes
