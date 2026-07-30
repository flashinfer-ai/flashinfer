.. _apikda_decode:

flashinfer.kda_decode
=====================

Key-Driven Attention (KDA) decode API. The CuTe-DSL kernel lives under
``flashinfer.kda_kernels``; this module is the public entry point.

On exact SM100a devices, the D128/T5 speculative-decode path with
precomputed gates and in-kernel QK normalization can dispatch to frozen
CUDA coefficient-Gram schedules. The dispatcher selects a value-row split
from the active sequence-head count and the device SM count. Other shapes,
architectures, gate modes, layouts, and optional features continue to use
the existing CuTe-DSL implementation.

.. currentmodule:: flashinfer.kda_decode

.. autosummary::
    :toctree: ../generated

    recurrent_kda
