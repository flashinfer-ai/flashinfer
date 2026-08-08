.. _apikda_decode:

flashinfer.kda_decode
=====================

Key-Driven Attention (KDA) decode API. The CuTe-DSL kernel lives under
``flashinfer.kda_kernels``; this module is the public entry point.

The public ``recurrent_kda`` API supports standard decode with one token per
sequence (``T=1``) and packed speculative decode with two or more tokens per
sequence (``T>=2``).

Pass ``backend="cake"`` to select the exported Cake backend. On SM100-family
SM100a (B200/GB200) and SM103a (B300/GB300) devices, its D128 ``T=1..6``
family with in-kernel QK normalization exports 23 frozen CUDA bodies:

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
count. SM100a retains the B200-measured policy: direct split 16 for T1 when
``W<=2S`` and direct split 8 otherwise; split 4 for T2; split 2 for T4; and
the T5/T6 CTA-wave policy of split 8 for ``W<=3S/8``, split 2 for
``3S/8<W<=S/2``, split 4 for ``S/2<W<=3S/4``, split 2 for
``3S/4<W<=3S/2``, and split 1 above that range.

SM103a uses its separately measured GB300 policy. T1 selects direct split 16
through a conservative ``W<=32S`` extrapolation guard (measured through
``W/S=26.95``), and direct split 8 beyond it. T2 selects split 8 through
``W<=S/2`` and split 4 above it. T4 selects split 8 through ``W<=S/2``, split
4 through ``W<=S``, split 2 through ``W<=3S/2``, split 1 through ``W<=2S``,
and split 2 above it. T5 keeps the SM100a CTA-wave policy except for a measured
split-1 island at ``3S/4<W<=S``. T6 selects split 8 through ``W<=3S/8``, split
2 through ``W<=S/2``, and split 1 above it. T3 uses its sole exact lower-bound
split-4 specialization on both architectures.

With CUDA 12.9 or newer, JIT and AOT compile all 23 checked-in bodies once for
the ``sm_100f`` family target. The family module URI and cubin artifact can run
on both CC 10.0 and CC 10.3; build workspaces may still materialize separate
cache directories for their local architecture context. Runtime split
selection remains device-specific. A cold-L2 CUPTI A/B against exact-target
cubins measured no aggregate change on B200 (``1.0000x`` exact/family) and
``0.9987x`` on GB300. The GB300 direct-T1 path was the repeatable exception
(``0.9790x``), so its two public direct variants retain exact ``sm_103a``
cubins while every other GB300 route uses ``sm_100f``.

CUDA 12.8 cannot compile ``sm_100f``. On B200 it therefore retains exact
``sm_100a`` modules for all 23 bodies. SM103a requires CUDA 12.9 or newer.
Every binding validates its family or exact-device contract before launch, and
the frozen generated body bytes are identical across all physical targets.

Once ``backend="cake"`` is selected, every supported call launches exactly one
exported Cake kernel. An unsupported architecture, shape, gate mode, layout,
aliasing pattern, or optional feature raises an error; it never falls back to
CuTe-DSL. The default ``backend="cute-dsl"`` preserves the existing FlashInfer
implementation.

Serving-native packed Kimi K3 decode
------------------------------------

``packed_kda_decode`` is a separate serving adapter for the Kimi K3 ``T=1``,
``H=12``, ``K=V=128`` contract. It consumes post-convolution packed QKV plus
raw gate and beta tensors, then fuses Q/K extraction and L2 normalization, the
fixed ``lower_bound=-5`` gate transform, beta sigmoid, and the recurrent update
into one exported Cake kernel. It is distinct from ``recurrent_kda`` (whose
inputs are already split into Q, K, V, gate, and beta tensors) and
``fused_kda_decode`` (which also performs the convolution and gated RMSNorm).

The operator updates a caller-owned bfloat16 state pool in place. A contiguous
int32 ``state_indices`` tensor selects one unique active slot per batch row;
``-1`` is an inactive CUDA-graph padding row that emits zero without touching
state. By default the operator allocates a contiguous bfloat16 output of shape
``[B, 1, 12, 128]``. Supplying a caller-owned output with that exact layout
makes replay allocation-free. All work runs on the caller's current PyTorch
CUDA stream.

Two frozen schedules are selected from host-visible batch size only: the
eight-row value tile for ``B < 32`` and the sixteen-row value tile for
``B >= 32``. Both are built as exact ``sm_100a`` or ``sm_103a`` modules;
compute capability 10.0 requires CUDA 12.8 or newer, and compute capability
10.3 requires CUDA 12.9 or newer. Unsupported devices or contracts raise an
error without falling back to another KDA implementation.

.. currentmodule:: flashinfer.kda_decode

.. autosummary::
    :toctree: ../generated

    fused_kda_decode
    packed_kda_decode
    recurrent_kda
