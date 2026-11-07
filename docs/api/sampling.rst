.. _apisampling:

flashinfer.sampling
===================

Kernels for LLM sampling.

.. important::

   Batch-length ``seed`` and ``offset`` tensors now apply one value per output
   row. Previously, all rows used element zero, even when a batch-length tensor
   was supplied. Calls with distinct values in these tensors can therefore
   produce different samples after this fix. Scalar values, length-one tensors,
   and calls using a shared ``torch.Generator`` retain their existing behavior.

   The output row index still feeds the Philox subsequence. Per-row seeds do not
   make sampling invariant to batch position, even with the same seed and offset.

Top-k and top-p rank tokens by probability (or logit) descending, then by
original token index ascending. Top-k keeps exactly ``min(k, vocab_size)``
positions. Top-p keeps the shortest prefix whose mass reaches its threshold,
including the token that crosses it. Boundary ties keep only the required
prefix, so they can change results previously produced by tie-inclusive filters.
The convention applies to sampling, renormalization, masking, and both
``filter_apply_order`` paths, independently of ``deterministic``. The two filter
orders retain their existing intermediate normalization semantics.

For probability inputs, top-p compares cumulative input probabilities directly
with ``top_p``; it does not multiply the threshold by the input row sum. Supply
nonnegative, normalized probabilities. As before, ``top_p >= 1`` disables top-p
filtering and keeps every token, even if row-sum rounding exceeds 1.
At vocabulary sizes of at least 2048, ``top_p_renorm_probs`` previously scaled
its threshold by the row sum; unnormalized inputs can therefore change their
retained set. Exact-boundary comparisons use the represented input values.

.. seealso::

  For efficient Top-K selection (without sampling), see :ref:`apitopk` which provides
  :func:`~flashinfer.top_k`, :func:`~flashinfer.top_k_page_table_transform`, and
  :func:`~flashinfer.top_k_ragged_transform`.

.. currentmodule:: flashinfer.sampling

.. autosummary::
    :toctree: ../generated

    sampling_from_probs
    sampling_from_logits
    softmax
    top_p_sampling_from_probs
    top_k_sampling_from_probs
    min_p_sampling_from_probs
    top_k_top_p_sampling_from_logits
    top_k_top_p_sampling_from_probs
    top_p_renorm_probs
    top_k_renorm_probs
    top_k_mask_logits
    chain_speculative_sampling
