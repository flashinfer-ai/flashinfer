.. _apicake_dsa_indexer_nondet_train:

flashinfer.cake_dsa_indexer_nondet_train
========================================

Cake DSA indexer top-k for training (32 heads, head dimension 128, BF16 ``q``
/ ``k``, FP32 ``w``) for SM100a, SM103a and SM107a: for every query, the exact
best ``min(top_k, visible)`` keys by the indexer score, ties to the larger id.
The module is a stable API; its backend package,
``flashinfer.experimental.cake.cake_dsa_indexer_nondet_train``, is
experimental and is imported on first use.

The mode is part of every name, and no entry takes a ``deterministic``
keyword.  Per row, the set of selected ids and score bits repeats bit for bit,
and with it the number of valid slots.  The slot order within a row and the
positions of the ``-1`` / ``-inf`` padding can change between calls.  The
deterministic package ``flashinfer.cake_dsa_indexer_det_train`` selects the
same set and returns it in ascending id order.  Under
``torch.use_deterministic_algorithms(True)`` a call with queries and keys
raises PyTorch's alert before it allocates or launches anything; with
``warn_only=True`` it warns and runs.  It never falls back to a deterministic
implementation.  This package offers the BF16 form only.

.. currentmodule:: flashinfer.cake_dsa_indexer_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_dsa_indexer_nondet_train_topk
    cake_dsa_indexer_nondet_train_topk_workspace_size
