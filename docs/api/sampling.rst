.. _apisampling:

flashinfer.sampling
===================

Kernels for LLM sampling.

.. important::

   Tensor ``seed`` and ``offset`` values identify request-local random streams.
   Each tensor must have length one (broadcast) or one element per output row.
   The seed/offset pair is mixed into a Philox seed independently of batch
   position, including for single-row batches. Tensor offsets are stream keys,
   not skip-ahead positions within a shared stream. Change the offset for a
   new sampling call, and use distinct keys for independent requests.

   This changes the previous length-one tensor behavior as well as batch-length
   tensor behavior. Repeating a pair, whether with ``torch.full((B,), seed)`` or
   length-one tensors, replays the same stream in every row; identical input
   distributions therefore produce identical tokens. Scalar seed/offset values
   and calls using ``torch.Generator`` retain their existing row-dependent
   subsequences and skip-ahead offsets.

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
