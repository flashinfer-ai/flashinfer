.. _apisampling:

flashinfer.sampling
===================

Kernels for LLM sampling.

.. note::

   Generator state reservations in ``flashinfer.sampling`` are serialized across
   host threads. This does not synchronize with PyTorch RNG operations or other
   code that mutates the same generator. When sharing a generator across these
   APIs and other RNG operations, callers must serialize both sides with a common
   external lock, or use separate generators. This also applies to the default
   CUDA generator used when ``generator=None``.

.. important::

   Batch-length ``seed`` and ``offset`` tensors now apply one value per output
   row. Previously, all rows used element zero, even when a batch-length tensor
   was supplied. Calls with distinct values in these tensors can therefore
   produce different samples after this fix. Scalar values, length-one tensors,
   and calls using a shared ``torch.Generator`` retain their existing behavior.

   The output row index still feeds the Philox subsequence. Per-row seeds do not
   make sampling invariant to batch position, even with the same seed and offset.

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
    log_softmax
    log_probs
    top_p_sampling_from_probs
    top_k_sampling_from_probs
    min_p_sampling_from_probs
    top_k_top_p_sampling_from_logits
    top_k_top_p_sampling_from_probs
    top_p_renorm_probs
    top_k_renorm_probs
    top_k_mask_logits
    chain_speculative_sampling
