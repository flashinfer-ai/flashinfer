.. _apicake_lm_head_nondet_train:

flashinfer.cake_lm_head_nondet_train
====================================

``cake_lm_head_nondet_train`` holds the non-deterministic Cake training entries of the chunked
LM-head projection with a cross-entropy or policy loss, its per-token log-probabilities, and the
log-probabilities / KL over a ragged candidate support, on SM100 (B200 / GB200), SM103
(B300 / GB300) and SM107 (R200).  The entries take the arguments of the deterministic module
``flashinfer.cake_lm_head_det_train`` and keep its precision boundaries.  Their bits may differ
from the deterministic module's, within the accuracy budgets, where the default path forms an
output in another order: a sliced dX GEMM stores its K slices into an FP32 slab and a slab sum
adds them in slice order (``dX``); zero-weight policy rows and zero-``dlogp`` log-probability
rows skip the gradient GEMMs (``loss`` / ``dW``); a call of fewer than three chunks folds a short
last chunk into the chunk before it (``loss`` / ``dW``); a one-row cross-entropy loss call at
``H = 6144`` runs one-pass programs (``loss`` / ``logp`` / ``lse`` / ``dW``); a log-probability
backward of 2 to 4 chunks runs vocabulary-major by the README's rule (``dX`` / ``dW``); and the
candidate-support dense dX uses a wider vocabulary block (``dX``).  No entry takes a
``deterministic`` keyword.
The module makes no run-to-run bitwise promise for the gradients, and keeps PyTorch's alert as
part of that contract: under ``torch.use_deterministic_algorithms(True)`` an entry raises the
alert right before a K-sliced dX reduction and never falls back to the deterministic module.
The backend package is ``flashinfer.experimental.cake.cake_lm_head_nondet_train``; its README
lists the output differences.

.. currentmodule:: flashinfer.cake_lm_head_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_lm_head_nondet_train_loss
    cake_lm_head_nondet_train_logprob
    cake_lm_head_nondet_train_preload
    cake_lm_head_nondet_train_support_logprob
    cake_lm_head_nondet_train_support_kl
