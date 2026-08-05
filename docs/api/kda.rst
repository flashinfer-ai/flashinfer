.. _apikda:

flashinfer.kda
==============

Phase-neutral recurrent Kimi Delta Attention (KDA) facade. The public
``recurrent_kda`` entry point keeps decode and speculative decode on
``flashinfer.kda_decode`` while dispatching eligible ordinary multi-token
prefill to the optimized backend described in :ref:`apikda_prefill`.

.. currentmodule:: flashinfer.kda

.. autosummary::
    :toctree: ../generated

    recurrent_kda
