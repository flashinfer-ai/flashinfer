.. _apigdn_decode:

flashinfer.gdn_decode
=====================

Gated Delta-Rule decode-side kernels used in Mamba-2 / GDN-style sequence
models. These functions consume a pre-built KV / state cache and run the
recurrent gated delta-rule update for the current decode step.

.. currentmodule:: flashinfer.gdn_decode

.. autosummary::
    :toctree: ../generated

    gated_delta_rule_decode
    gated_delta_rule_decode_pretranspose
    gated_delta_rule_mtp
