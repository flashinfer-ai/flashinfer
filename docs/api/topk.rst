.. _apitopk:

flashinfer.topk
===============

Efficient Top-K selection kernels.

.. seealso::

  For Top-K based sampling, see :ref:`apisampling` which provides
  :func:`~flashinfer.sampling.top_k_sampling_from_probs`,
  :func:`~flashinfer.sampling.top_k_top_p_sampling_from_probs`,
  :func:`~flashinfer.sampling.top_k_renorm_probs`, and
  :func:`~flashinfer.sampling.top_k_mask_logits`.

.. currentmodule:: flashinfer

Top-K Selection
---------------

.. autosummary::
  :toctree: ../generated

  top_k
  top_k_page_table_transform
  top_k_ragged_transform

Utility Functions
-----------------

.. autosummary::
  :toctree: ../generated

  topk.can_implement_filtered_topk
