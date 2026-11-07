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

.. _sampling-ties-and-determinism:

Boundary ties and determinism
-----------------------------

Every filter in this module keeps ties at its boundary. Top-k (:func:`~flashinfer.sampling.top_k_renorm_probs`, :func:`~flashinfer.sampling.top_k_mask_logits`, :func:`~flashinfer.sampling.top_k_sampling_from_probs`) keeps every token whose value is ``>=`` the k-th largest, so a row with ties at the k-th value keeps more than ``k`` tokens. Top-p (:func:`~flashinfer.sampling.top_p_renorm_probs`, :func:`~flashinfer.sampling.top_p_sampling_from_probs`) keeps every token whose strictly-greater cumulative mass is below ``top_p``, so tokens equal to the threshold value are all kept. The joint samplers (:func:`~flashinfer.sampling.top_k_top_p_sampling_from_logits`, :func:`~flashinfer.sampling.top_k_top_p_sampling_from_probs`) follow the same rule, with one documented exception: their ``"top_k_first"`` fast path (no ``indices``, ``top_k_max <= 256``, ``num_classes >= 65536``) selects exactly ``k`` tokens with a radix top-k; its choice among tied boundary tokens is fixed by CTA and thread position when ``deterministic=True`` and follows atomic arrival order otherwise.

The ``deterministic`` flag of the samplers selects an order-independent prefix scan in the sampling kernel; it defaults to ``True`` and is a no-op on :func:`~flashinfer.sampling.sampling_from_logits`, which has no scan. The ``is_deterministic`` flag of the renorm ops replaces float atomics with fixed-point integer atomics. Both renorm ops default to ``False``; enable it when replays must be bitwise reproducible. For :func:`~flashinfer.sampling.top_k_renorm_probs` the cost is below measurement noise; for :func:`~flashinfer.sampling.top_p_renorm_probs` the deterministic histogram costs 1.25x to 1.61x at batch sizes of 128 and above, and with the default the kept set itself can change between calls when ``top_p`` lies within rounding of a cumulative-mass boundary.

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
