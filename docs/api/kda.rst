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

``backend`` defaults to ``"auto"``, the only value with fallback semantics on
both phases: it prefers the CuTe DSL kernel for supported plain prefill and the
frozen Cake specialization for its native ``T=1`` decode contract, falling back
rather than raising in either direction. ``"cute-dsl"`` and ``"cake"`` select
those backends strictly. Note that
:func:`flashinfer.kda_decode.recurrent_kda` keeps its own released
``"cute-dsl"`` default, so a defaulted decode call through that facade can
select a different kernel than a defaulted call here.

.. currentmodule:: flashinfer.kda

.. autosummary::
    :toctree: ../generated

    recurrent_kda
