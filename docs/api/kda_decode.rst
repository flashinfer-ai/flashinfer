.. _apikda_decode:

flashinfer.kda_decode
=====================

Key-Driven Attention (KDA) decode API. The CuTe-DSL kernel lives under
``flashinfer.kda_kernels``; this module is the public entry point.

The public ``recurrent_kda`` API supports standard decode with one token per
sequence (``T=1``) and packed speculative decode with two or more tokens per
sequence (``T>=2``).

On exact SM100a devices, the D128 ``T=1..6`` family with in-kernel QK
normalization can dispatch to frozen CUDA schedules:

* ``T=3`` with raw gates, ``use_gate_in_kernel=True``, a negative
  ``lower_bound``, float32 ``A_log`` and ``dt_bias``, ``H=HV=16``, and
  ``N`` in ``{1, 2, 4, 8, 16}``;
* ``T`` in ``{1, 2, 4, 5, 6}`` with precomputed gates,
  ``use_gate_in_kernel=False``, and no ``A_log``, ``dt_bias``, or
  ``lower_bound``. ``T=1`` keeps the standard decode API and is normalized to
  the packed frozen ABI with zero-copy views and cached identity metadata.

Each token count exports value-row splits ``{1, 2, 4, 8}`` (except the exact
T3 route, which uses split 4). The T5/T6 schedules use coefficient-Gram
projection. The dispatcher selects a split from the active sequence-head
count and the device SM count.

Other supported shapes, architectures, gate modes, layouts, and optional
features continue to use the existing CuTe-DSL implementation.

.. currentmodule:: flashinfer.kda_decode

.. autosummary::
    :toctree: ../generated

    recurrent_kda
