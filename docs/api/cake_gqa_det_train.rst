.. _apicake_gqa_det_train:

flashinfer.cake_gqa_det_train
=============================

Cake deterministic GQA attention training for head dimension 256: a varlen
causal forward and backward for SM100a, SM103a and SM107a.  The module is a
stable API; its backend package,
``flashinfer.experimental.cake.cake_gqa_det_train``, is experimental and is
imported on first use.

The mode is part of every name, and no entry takes a ``deterministic``
keyword.  Every sum of the forward and the backward has one fixed order:
``out``, ``lse``, dQ and every dK / dV form (BF16, FP32 window, FP32 row-map
adds with repeated destination rows) repeat bit for bit run to run.  The bits
depend only on the shapes, ``max_seqlen_k`` and the workspace budget.  Under
``torch.use_deterministic_algorithms(True)``, strict or with
``warn_only=True``, nothing alerts or warns.

The backward's main kernel sums dK and dV of each key block in tensor memory
and stores dS^T into a workspace, one 32 KiB tile per (head pair, key block
of 128 keys, Q tile of 64 rows); dQ is a GEMM over that workspace.  The workspace's ``budget_bytes`` (default 16 GiB) bounds that
storage: a larger call runs in chunks of KV heads, then of key blocks, and
the smallest chunk (one KV head, two key blocks) runs even past it.  A
different budget can move the last bits of dQ, not its accuracy.  A row-map
destination row becomes ``((dst + w_j1) + w_j2) + ...`` over its window rows
in ascending order.  The interface matches the non-deterministic counterpart
:ref:`apicake_gqa_nondet_train`.

``cake_gqa_det`` is the autograd entry (its Function is
``CakeGQADetFunction``); the forward and backward halves, the workspace and
its size are separate entries.

.. currentmodule:: flashinfer.cake_gqa_det_train

.. autosummary::
    :toctree: ../generated

    cake_gqa_det
    cake_gqa_det_train_forward
    cake_gqa_det_train_backward
    cake_gqa_det_train_workspace
    cake_gqa_det_train_workspace_bytes
