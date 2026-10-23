.. _apigdp_prefill:

flashinfer.gdp_prefill
======================

Gated DeltaProduct prefill. ``chunk_gated_delta_product`` is the chunked GDP
scan: GDN with ``num_householder`` beta-gated Householder updates per token
instead of one, on an expanded k/v/beta sub-token timeline.

``backend="auto"`` and ``backend="cudnn"`` run cuDNN's fused SM100
linear-attention engine
(:func:`flashinfer.cudnn.cudnn_chunk_gated_delta_product`); it needs an
SM100-family device and cudnn-frontend 1.28+ with the ``cutedsl`` extra, and
declines anything else it cannot serve.  ``backend="cake"`` runs the Cake GDP
prefill kernel (K=128, V=64, one to four Householder steps per token; SM90)
with the same argument contract.

.. currentmodule:: flashinfer.gdp_prefill

.. autosummary::
    :toctree: ../generated

    chunk_gated_delta_product

Decode and MTP verification (Cake, SM90 / SM100 / SM103)
--------------------------------------------------------

Gated DeltaProduct decode and multi-token (MTP) verification:
K=128, V=64, one to four Householder steps per token, over an FP32
pretranspose state pool with indexed read / write / per-token snapshot slots.
One fused kernel per call; the docstring states the input contracts.

.. currentmodule:: flashinfer.cake_gdp

.. autosummary::
    :toctree: ../generated

    cake_gated_delta_product_mtp
