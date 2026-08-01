.. _apikda_decode:

flashinfer.kda_decode
=====================

Recurrent Kimi Delta Attention (KDA) public API. Decode and speculative
decode use the CuTe-DSL backend under ``flashinfer.kda_kernels``. Eligible
ordinary multi-token prefill calls dispatch to the optimized backend described
in :ref:`apikda_prefill`.

.. currentmodule:: flashinfer.kda_decode

.. autosummary::
    :toctree: ../generated

    recurrent_kda
