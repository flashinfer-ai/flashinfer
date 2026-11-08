.. _apicake_lm_head_nondet_train:

flashinfer.cake_lm_head_nondet_train
====================================

``cake_lm_head_nondet_train`` holds the non-deterministic Cake training entries of the chunked
LM-head projection with a cross-entropy or policy loss, its per-token log-probabilities, and the
log-probabilities / KL over a ragged candidate support, on SM100 (B200 / GB200), SM103
(B300 / GB300) and SM107 (R200).  The entries take the arguments of the deterministic module
``flashinfer.cake_lm_head_det_train`` and keep its precision boundaries, with two kinds of
difference: each K slice of a sliced dX GEMM is added in an unordered FP32 reduction, so ``dX``
is not bitwise reproducible run to run; and zero-weight policy rows and zero-``dlogp``
log-probability rows skip the gradient GEMMs, so ``loss`` / ``dW`` bits may differ from the
deterministic module's within the accuracy budgets.  No entry takes a ``deterministic`` keyword.
Under ``torch.use_deterministic_algorithms(True)`` an entry raises PyTorch's alert right before a
K-sliced dX reduction and never falls back to the deterministic module.  The backend package is
``flashinfer.experimental.cake.cake_lm_head_nondet_train``; its README lists the output
differences.

.. currentmodule:: flashinfer.cake_lm_head_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_lm_head_nondet_train_loss
    cake_lm_head_nondet_train_logprob
    cake_lm_head_nondet_train_preload
    cake_lm_head_nondet_train_support_logprob
    cake_lm_head_nondet_train_support_kl
