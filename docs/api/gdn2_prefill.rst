.. _apigdn2_prefill:

flashinfer.gdn2_prefill
=======================

Gated Delta-Rule 2 prefill. ``chunk_gated_delta_rule2`` is the chunked GDN-2
scan: GDN's per-head scalar forget and erase gates become per key channel, and
a third per-value-channel write gate scales the incoming value.

FlashInfer carries no GDN-2 kernel of its own, so this runs on cuDNN's fused
SM100 linear-attention engine
(:func:`flashinfer.cudnn.cudnn_chunk_gated_delta_rule2`). It needs an
SM100-family device and cudnn-frontend 1.28+ with the ``cutedsl`` extra; the
engine declines anything else it cannot serve.

.. currentmodule:: flashinfer.gdn2_prefill

.. autosummary::
    :toctree: ../generated

    chunk_gated_delta_rule2
