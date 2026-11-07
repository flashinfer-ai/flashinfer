.. _apicake_gqa_nondet_train:

flashinfer.cake_gqa_nondet_train
================================

Cake non-deterministic GQA attention training for head dimension 256: a varlen
causal forward and backward for SM100a, SM103a and SM107a.  The module is a
stable API; its backend package,
``flashinfer.experimental.cake.cake_gqa_nondet_train``, is experimental and is
imported on first use.

The mode is part of every name, and no entry takes a ``deterministic``
keyword.  The forward is fixed-order.  The backward adds the dQ partials (and
the row-map dK / dV adds) with atomic FP32 reductions, so its gradients are not
bitwise reproducible run to run.  Under
``torch.use_deterministic_algorithms(True)`` the backward raises PyTorch's
alert after its checks and before it allocates or launches anything; with
``warn_only=True`` it warns and runs.  It never falls back to a deterministic
implementation.  The deterministic package ``flashinfer.cake_gqa_det_train`` is
planned.

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
