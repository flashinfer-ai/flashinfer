.. _apikda:

flashinfer.kda
==============

Phase-neutral recurrent Kimi Delta Attention (KDA) facade. The public
``recurrent_kda`` entry point keeps decode and speculative decode on
``flashinfer.kda_decode`` while dispatching eligible ordinary multi-token
prefill to the optimized backend described in :ref:`apikda_prefill`. Which
prefill backend that is depends on the device: SM100a and SM103a use the frozen
FlashKDA-compatible kernels, SM120a uses a CuTe-DSL backend of its own. The two
architecture sets are disjoint, so the public signature and every call outside
the eligible prefill subset are unaffected either way.

.. currentmodule:: flashinfer.kda

.. autosummary::
    :toctree: ../generated

    recurrent_kda
