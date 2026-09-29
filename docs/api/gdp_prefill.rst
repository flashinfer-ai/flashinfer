.. _apigdp_prefill:

flashinfer.gdp_prefill
======================

Gated DeltaProduct prefill. ``chunk_gated_delta_product`` is the chunked GDP
scan: GDN with ``num_householder`` beta-gated Householder updates per token
instead of one, on an expanded k/v/beta sub-token timeline.

FlashInfer carries no GDP kernel of its own, so this runs on cuDNN's fused
SM100 linear-attention engine
(:func:`flashinfer.cudnn.cudnn_chunk_gated_delta_product`). It needs an
SM100-family device and cudnn-frontend 1.28+ with the ``cutedsl`` extra; the
engine declines anything else it cannot serve.

.. currentmodule:: flashinfer.gdp_prefill

.. autosummary::
    :toctree: ../generated

    chunk_gated_delta_product
