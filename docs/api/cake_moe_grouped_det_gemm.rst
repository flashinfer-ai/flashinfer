.. _apicake_moe_grouped_det_gemm:

Cake ragged BF16 grouped GEMM (experimental)
============================================

.. currentmodule:: flashinfer.grouped_mm

The experimental ragged BF16 grouped GEMM entry points live in
:mod:`flashinfer.grouped_mm` and hand off to the backend package
:mod:`flashinfer.experimental.cake_moe_grouped_det_gemm`: generated ragged BF16 grouped
GEMM programs for the expert projections of Mixture-of-Experts layers on SM100
(B200 / GB200), SM103 (B300 / GB300) and SM107 (R200).  Both the API and the backend
are experimental (see :ref:`experimental`); calling one of the functions below, or
naming the backend in the stable API, is the explicit opt-in and emits an
``ExperimentalWarning`` once.

Groups are consecutive row ranges of the packed activations, described by an int32
**device** tensor: either ``offs[E]`` of cumulative end offsets or ``m_indptr[E + 1]``
with a leading zero (the ``torch._grouped_mm`` and :func:`flashinfer.grouped_mm.grouped_mm_bf16`
conventions).  The kernels read the offsets on the device; the host never synchronises
on them, no padding is required and group sizes may be imbalanced, empty, one row or
any size that is not a tile multiple.  All operands are bfloat16 with fp32 tensor-core
accumulation; the reductions are bitwise deterministic (no atomics) and empty groups
produce exactly zero weight gradients.

============ ==================================================================== ============================
operation    computation                                                           output
============ ==================================================================== ============================
``fwd``      ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``              ``[sum_m, N]`` bfloat16
``dgrad``    ``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` (W in place)  ``[sum_m, K]`` bfloat16
``wgrad``    ``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]``             ``[E, N, K]`` bfloat16 or float32
============ ==================================================================== ============================

Mode and dispatch rules
-----------------------

This package and its entry points are deterministic only; every public name carries
``det``.

- Every entry below runs the deterministic programs: a fixed reduction order, bitwise
  reproducible run to run.
- No deterministic entry takes a ``deterministic`` argument: passing one (``True`` or
  ``False``, e.g. ``cake_grouped_det_mm(x, w, offs, deterministic=False)``) raises
  ``TypeError``, and so does a fourth argument of ``CakeGroupedDetMm.apply(x, w, offs)``
  other than ``None`` (a stale ``True`` / ``False`` numerical mode included; the fifth
  argument is ``fuse_wgrad_accumulation``).
- ``accumulate=True`` / ``fuse_wgrad_accumulation=True`` add the deterministic weight
  gradient into ``out`` / ``w.main_grad``, bitwise reproducible run to run (below).
- ``torch.use_deterministic_algorithms(True)`` (also with ``warn_only=True``) changes
  nothing: no alert, no warning.
- ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` runs the deterministic
  forward (the forward programs of both Cake packages give the same bits).

The non-deterministic counterpart is a separate package,
``flashinfer.experimental.cake.cake_moe_grouped_gemm``, and its facade
``flashinfer.experimental.cake.grouped_mm`` (see :ref:`apicake_moe_grouped_gemm`),
which take only ``deterministic=False``.  The names of this package before the
deterministic-family rename (``flashinfer.experimental.cake_moe_grouped_gemm``,
``flashinfer.grouped_mm.cake``, ``grouped_gemm_*``, ``cake_grouped_mm`` and the other
names without ``det``) were removed with no alias: importing them raises
``ModuleNotFoundError`` / ``AttributeError``.

Stable API opt-in
-----------------

The forward projection is reachable from the stable grouped GEMM API by naming the
backend::

    out = flashinfer.grouped_mm.grouped_mm_bf16(a, b, m_indptr, backend="cake")

It accepts the ``[cum_m, K]`` / ``[E, N, K]`` / ``[E + 1]`` arguments of the cuDNN
backend, produces bfloat16 outputs only, has no tactic index and requires
``N % 256 == 0`` and ``K % 64 == 0``.

One-shot functions
------------------

.. autosummary::
    :toctree: ../generated

    grouped_det_gemm_fwd
    grouped_det_gemm_dgrad
    grouped_det_gemm_wgrad

Autograd
--------

.. autosummary::
    :toctree: ../generated

    cake_grouped_det_mm

``cake_grouped_det_mm(x, w, offs, *, fuse_wgrad_accumulation=False)`` wraps the
``torch.autograd.Function`` ``CakeGroupedDetMm`` of the backend package: its ``backward``
produces ``x.grad`` with the ``dgrad`` program and ``w.grad`` with the ``wgrad`` program in
the dtype of ``w``.  The Cake programs always reduce in a fixed order; there is no
``deterministic`` argument.  Use :func:`grouped_det_gemm_wgrad` with
``out_dtype=torch.float32`` for an fp32 weight gradient, or
``fuse_wgrad_accumulation=True`` to add it into an fp32 ``w.main_grad`` (below).

Weight-gradient accumulation
----------------------------

``grouped_det_gemm_wgrad(..., out=out, accumulate=True)`` adds the deterministic weight
gradient into the caller's bf16 or fp32 ``out`` (``out[e] += G[e].T @ X[e]``, Megatron's
gradient-accumulation fusion) and ``cake_grouped_det_mm(..., fuse_wgrad_accumulation=True)``
adds it into ``w.main_grad`` in its backward.  Both are keyword-only and off by default;
with the defaults nothing changes.  The launch composes (``plan["accumulate"] ==
"composed"``): the overwrite of the same plan (route, k tile, stage, tail split and its
ordered tail reduce) into a contiguous scratch of ``out``'s dtype, then ``out.add_`` on the
same stream, outside autograd.

- Numerics: ``out`` gets AccumulateGrad's bits of the deterministic weight gradient in its
  dtype (fp32: one fp32 add of the fp32 ``dW``; bf16: the bf16 ``dW`` added in bf16),
  bitwise reproducible run to run; the accuracy budget is the deterministic ``dW``'s plus
  one rounding.  An empty group adds exact zeros (``out.add_`` semantics: values kept,
  ``-0.0`` becomes ``+0.0``); ``sum_m == 0`` leaves ``out`` unchanged.  Torch's
  deterministic mode changes nothing.
- Checks (``ValueError``, before anything is allocated or bound): ``out`` given,
  ``out_dtype`` None or its dtype, bf16 or fp32, no overlap of ``out`` with itself, ``g``,
  ``x`` or ``offs``.  Row and expert padding is never read or written.
- Prepared launches: every ``launch()`` and every CUDA graph replay adds again.  The
  scratch is owned by the prepared launch: keep the launch alive while a graph that
  captured it may replay, and do not run one prepared accumulate launch on two streams at
  once.
- Autograd flag (the Megatron / Transformer Engine convention): ``w.main_grad`` (fp32 or
  bf16, ``w``'s shape and device, unit stride along the last dimension) is checked at a
  forward call that autograd records and read again at the backward; the backward adds
  into it and returns a fresh placeholder for ``w`` when ``w`` has
  ``grad_added_to_main_grad`` (``torch.zeros`` under ``zero_out_wgrad``), else none.  Every
  backward adds, also a ``torch.autograd.grad`` whose ``inputs`` exclude ``w``.
  ``NotImplementedError`` under ``torch.jit.trace``, ``torch.compile`` and a
  ``create_graph`` backward; a reentrant checkpoint passes ``w`` without ``main_grad``
  (``ValueError``).
- Cost (``S = E * N * K``): the scratch takes ``4 * S`` bytes for fp32 (gate/up ``N`` 4096,
  ``K`` 6144: 384 MiB at ``E`` 4 to 3 GiB at ``E`` 32; down ``N`` 6144, ``K`` 2048: 192 MiB
  to 1.5 GiB) and ``8 * S`` bytes of traffic (written, then read back by the add); the
  backward's peak is ``2 * S`` bytes above the default flag's bf16 ``w.grad``.  Under
  ``torch.use_deterministic_algorithms(True)`` with
  ``torch.utils.deterministic.fill_uninitialized_memory`` on, torch also fills the scratch
  and the placeholder with NaN when it allocates them.  The non-deterministic package adds
  in the kernel epilogue instead (:ref:`apicake_moe_grouped_gemm`).

Prepared launches
-----------------

The backend package also holds the prepare-once / launch-many surface:
``flashinfer.experimental.cake_moe_grouped_det_gemm.prepare_grouped_det_gemm_fwd``,
``prepare_grouped_det_gemm_dgrad`` and ``prepare_grouped_det_gemm_wgrad`` validate the
arguments, bind one generated program and return a ``GroupedDetGemmLaunch`` whose
``launch()`` performs no allocation and no host synchronisation and is CUDA-graph
capturable; the same prepared launch stays valid when new offsets or values are
written into the bound tensors.  The package also carries plain functions of the same
names as the entry points above; a build that vendors the package calls them
directly, without the experimental marking and warning.
