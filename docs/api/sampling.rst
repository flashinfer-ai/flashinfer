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

   The following RNG mapping applies to the default sampling kernels
   (``batch_invariant=False`` where available). The opt-in batch-invariant ICDF
   samplers retain their separately documented RNG contract.

   Each ``seed`` and ``offset`` tensor must have length one (broadcast) or one
   element per output row. RNG values are indexed by output row, including when
   ``indices`` gathers or repeats input distributions.

   When either RNG tensor has batch length in a multi-row batch, the seed/offset
   pair identifies a request-local random stream. The pair is mixed into a
   Philox seed independently of batch position. Tensor RNG in a single-row batch
   uses the same rule, so per-row requests replay when compacted to one row.
   In this mode, offsets are stream keys, not skip-ahead positions. Change the
   offset for each new sampling call and use distinct keys for independent
   requests. Repeating a per-row pair (for example, with ``torch.full((B,), seed)``
   and a shared offset) replays the same draw for identical distributions.

   **Compatibility:** multi-row batches with only length-one RNG tensors retain
   the released broadcast behavior, including CUDA graph callers. Like scalar
   seeds/offsets and ``torch.Generator``, they use row-dependent subsequences and
   skip-ahead offsets; advance these offsets by the random values consumed.
   The choice depends on tensor lengths and output batch size, not their values.
   Per-row tensors and singleton tensor calls change from the previous RNG
   mapping. Compacting a broadcast-only batch to one row switches to request-key
   semantics. Use per-row RNG tensors from the start when replay across batch
   layouts is required.

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
