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
  top_k_varlen
  top_k_page_table_transform
  top_k_ragged_transform

.. autofunction:: top_k

Fused sparse-attention output
-----------------------------

``top_k_page_table_transform(..., backend="gvr_2")`` opts into hint-free
GVR selection with page-table translation in the selection kernel. It uses
the existing ``page_size``, ``row_to_batch``, ``page_table_row_starts``,
``out`` and optional ``out_raw_indices`` contract. No intermediate raw
output is allocated when ``out_raw_indices`` is omitted. Short rows emit
their logical positions in order and pad the remaining output with ``-1``.

This backend supports the GVR register, register-image and clustered-register routes,
FP32 full-row scores, and K of 512, 1024 or 2048 on supported GVR GPUs.
Other kernel families raise ``NotImplementedError``; score windows,
deterministic ordering and explicit tie breaking are not supported.
Warm up each configuration before CUDA graph capture. Buffer contents,
including row lengths and page-table entries, can change during replay.
The default ``backend="auto"`` routing is unchanged.

Utility Functions
-----------------

.. autosummary::
  :toctree: ../generated

  topk.can_implement_filtered_topk
