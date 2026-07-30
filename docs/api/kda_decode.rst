.. _apikda_decode:

flashinfer.kda_decode
=====================

Key-Driven Attention (KDA) decode API. The CuTe-DSL kernel lives under
``flashinfer.kda_kernels``; this module is the public entry point.

The public ``recurrent_kda`` API supports standard decode with one token per
sequence (``T=1``) and packed speculative decode with two or more tokens per
sequence (``T>=2``).

On exact SM100a devices, two D128 packed speculative-decode contracts with
in-kernel QK normalization can dispatch to frozen CUDA schedules:

* ``T=3`` with raw gates, ``use_gate_in_kernel=True``, a negative
  ``lower_bound``, float32 ``A_log`` and ``dt_bias``, ``H=HV=16``, and
  ``N`` in ``{1, 2, 4, 8, 16}``;
* ``T=5`` with precomputed gates, ``use_gate_in_kernel=False``, and no
  ``A_log``, ``dt_bias``, or ``lower_bound``. The dispatcher selects a
  coefficient-Gram value-row split from the active sequence-head count and
  the device SM count.

All other supported token counts, shapes, architectures, gate modes, layouts,
and optional features continue to use the existing CuTe-DSL implementation.

.. currentmodule:: flashinfer.kda_decode

.. autosummary::
    :toctree: ../generated

    recurrent_kda
