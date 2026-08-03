.. _apikda_decode:

flashinfer.kda_decode
=====================

Key-Driven Attention (KDA) decode API. The CuTe-DSL kernel lives under
``flashinfer.kda_kernels``; this module is the public entry point.

The public ``recurrent_kda`` API supports standard decode with one token per
sequence (``T=1``) and packed speculative decode with two or more tokens per
sequence (``T>=2``).

Pass ``backend="cake"`` to select the exported Cake backend. On exact SM100a
devices, its D128 ``T=1..6`` family with in-kernel QK normalization exports 23
frozen CUDA modules:

* ``T=3`` with raw gates, ``use_gate_in_kernel=True``, a negative
  ``lower_bound``, float32 ``A_log`` and ``dt_bias``, ``H=HV=16``, and
  ``N`` in ``{1, 2, 4, 8, 16}``;
* four value-row splits for each ``T`` in ``{1, 2, 4, 5, 6}`` with
  precomputed gates, ``use_gate_in_kernel=False``, and no ``A_log``,
  ``dt_bias``, or ``lower_bound``;
* two additional one-warp direct-state ``T=1`` schedules with value-row
  splits 16 and 8. ``T=1`` keeps the standard decode API and is normalized
  to the packed frozen ABI with zero-copy views and cached identity metadata;
  explicit ``T=1`` ``cu_seqlens`` metadata is outside the Cake contract.

Let ``W=N*HV`` be the active sequence/value-head work and ``S`` the device SM
count. The Cake dispatcher selects the direct split-16 schedule for T1 when
``W<=2S`` and the direct split-8 schedule otherwise. It selects split 4 for
T2, split 4 for the exact T3 lower-bound contract, and split 2 for T4. The
T5/T6 coefficient-Gram schedules reproduce Cake's CTA-wave policy: split 8
for ``W<=3S/8``, split 2 for ``3S/8<W<=S/2``, split 4 for
``S/2<W<=3S/4``, split 2 for ``3S/4<W<=3S/2``, and split 1 above that
range.

Once ``backend="cake"`` is selected, every supported call launches exactly one
exported Cake kernel. An unsupported architecture, shape, gate mode, layout,
aliasing pattern, or optional feature raises an error; it never falls back to
CuTe-DSL. The default ``backend="cute-dsl"`` preserves the existing FlashInfer
implementation.

.. currentmodule:: flashinfer.kda_decode

.. autosummary::
    :toctree: ../generated

    recurrent_kda
