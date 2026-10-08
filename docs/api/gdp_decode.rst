.. _apigdp_decode:

flashinfer.gdp_decode
=====================

Gated DeltaProduct decode. ``gated_delta_product_mtp`` runs ``num_householder``
beta-gated Householder updates per real token on the GDN MTP decode kernel,
taking ``k``, ``v`` and ``beta`` with a householder axis next to the token axis
while ``q``, the decay logits and the output stay one row per real token.

``num_householder`` comes from the shape; at 1 the call delegates to
:func:`flashinfer.gdn_decode.gated_delta_rule_mtp` and is bit-identical to it.

.. currentmodule:: flashinfer.gdp_decode

.. autosummary::
    :toctree: ../generated

    gated_delta_product_mtp
