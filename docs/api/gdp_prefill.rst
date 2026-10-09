.. _apigdp_prefill:

flashinfer.gdp_prefill
======================

Gated DeltaProduct prefill. ``chunk_gated_delta_product`` is the chunked GDP
scan: GDN with ``num_householder`` beta-gated Householder updates per token
instead of one, on an expanded k/v/beta sub-token timeline.

``backend="auto"`` (the default) and ``"cudnn"`` run cuDNN's fused SM100
linear-attention engine
(:func:`flashinfer.cudnn.cudnn_chunk_gated_delta_product`). That needs an
SM100-family device and cudnn-frontend 1.28+ with the ``cutedsl`` extra; the
engine declines anything else it cannot serve.

``backend="flashinfer"`` runs the in-tree GDN prefill kernel over the expanded
sub-token timeline on SM90 and SM100. It requires ``g`` and ``beta`` in
float32, as :func:`flashinfer.gdn_prefill.chunk_gated_delta_rule` does.

.. currentmodule:: flashinfer.gdp_prefill

.. autosummary::
    :toctree: ../generated

    chunk_gated_delta_product
