.. _apicake_moe_grouped_gemm_det_train:

Cake deterministic ragged BF16 grouped GEMM
===========================================

.. currentmodule:: flashinfer.cake_moe_grouped_gemm_det_train

The module :mod:`flashinfer.cake_moe_grouped_gemm_det_train` holds the deterministic
ragged BF16 grouped GEMM of the expert projections of Mixture-of-Experts training on
SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200).  Its functions are stable
FlashInfer APIs: each one hands off to the function of the same name in the backend
package :mod:`flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train`, which holds
the generated programs.  Importing the module loads none of the backend package.
``flashinfer`` binds the module's names on first access, so for example
``flashinfer.cake_moe_grouped_gemm_det`` needs no import of the module.  To use the
module itself, import it: ``import flashinfer.cake_moe_grouped_gemm_det_train``.

Groups are consecutive row ranges of the packed activations, described by an int32
**device** tensor: either ``offs[E]`` of cumulative end offsets or ``m_indptr[E + 1]``
with a leading zero (the ``torch._grouped_mm`` and :func:`flashinfer.grouped_mm.grouped_mm_bf16`
conventions).  The kernels read the offsets on the device; the host never synchronises
on them, no padding is required and group sizes may be imbalanced, empty, one row or
any size that is not a tile multiple.  All operands are bfloat16 with fp32 tensor-core
accumulation; the reductions are bitwise deterministic (no atomics) and empty groups
produce exactly zero weight gradients.

The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.
Standalone ``wgrad`` calls with ``m_indptr`` must also pass ``num_groups=E``.

============ ==================================================================== ============================
operation    computation                                                           output
============ ==================================================================== ============================
``fwd``      ``Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T``              ``[sum_m, N]`` bfloat16
``dgrad``    ``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]`` (W in place)  ``[sum_m, K]`` bfloat16
``wgrad``    ``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]``             ``[E, N, K]`` bfloat16 or float32
============ ==================================================================== ============================

Mode rules
----------

The module is the mode: every entry below is deterministic, and every public name
carries ``det``.

- Every entry runs the deterministic programs: a fixed reduction order, bitwise
  reproducible run to run.
- No entry takes a ``deterministic`` argument: passing one (``True`` or ``False``, e.g.
  ``cake_moe_grouped_gemm_det(x, w, offs, deterministic=False)``) raises ``TypeError``,
  and so does a fourth argument of ``CakeMoEGroupedGemmDetFunction.apply(x, w, offs)``
  other than ``None`` (the fifth argument is ``fuse_wgrad_accumulation``).
- ``accumulate=True`` / ``fuse_wgrad_accumulation=True`` add the deterministic weight
  gradient into ``out`` / ``w.main_grad``, bitwise reproducible run to run (below).
- ``torch.use_deterministic_algorithms(True)`` (also with ``warn_only=True``) changes
  nothing: no alert, no warning.
- ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` runs the deterministic
  forward (the forward programs of both modes give the same bits).

The non-deterministic mode is the separate module
:mod:`flashinfer.cake_moe_grouped_gemm_nondet_train`
(:ref:`apicake_moe_grouped_gemm_nondet_train`); its entries have the same shapes and
carry ``nondet``.

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

    cake_moe_grouped_gemm_det_train_forward
    cake_moe_grouped_gemm_det_train_dgrad
    cake_moe_grouped_gemm_det_train_wgrad

Autograd
--------

.. autosummary::
    :toctree: ../generated

    cake_moe_grouped_gemm_det

``cake_moe_grouped_gemm_det(x, w, offs, *, fuse_wgrad_accumulation=False)`` wraps the
``torch.autograd.Function`` ``CakeMoEGroupedGemmDetFunction`` of the backend package: its ``backward``
produces ``x.grad`` with the ``dgrad`` program and ``w.grad`` with the ``wgrad`` program in
the dtype of ``w``.  The programs always reduce in a fixed order; there is no
``deterministic`` argument.  Use :func:`cake_moe_grouped_gemm_det_train_wgrad` with
``out_dtype=torch.float32`` for an fp32 weight gradient, or
``fuse_wgrad_accumulation=True`` to add it into an fp32 ``w.main_grad`` (below).

Weight-gradient accumulation
----------------------------

``cake_moe_grouped_gemm_det_train_wgrad(..., out=out, accumulate=True)`` adds the deterministic weight
gradient into the caller's bf16 or fp32 ``out`` (``out[e] += G[e].T @ X[e]``, Megatron's
gradient-accumulation fusion) and ``cake_moe_grouped_gemm_det(..., fuse_wgrad_accumulation=True)``
adds it into ``w.main_grad`` in its backward.  Both are keyword-only and off by default;
with the defaults nothing changes.  The host plans the launch as the overwrite of the
same call (route, k tile, stage, tail split and its ordered tail reduce) and adds on one of
two routes (``plan["accumulate"]``).  ``"fused"``: for an fp32 ``out`` on the 512-wide k
tile's half-release forms, where the accumulate builds of the stage and, for a split tail,
of the tail reduce ship and can load, and ``out`` is 32-byte aligned, the kernels add into
``out`` in place (no scratch).  ``"composed"`` otherwise (``plan["accumulate_reason"]``):
the overwrite into a contiguous scratch of ``out``'s dtype, then ``out.add_`` on the same
stream, outside autograd.  Both routes give the same bits.

- Numerics: ``out`` gets AccumulateGrad's bits of the deterministic weight gradient in its
  dtype (fp32: one fp32 add of the fp32 ``dW``; bf16: the bf16 ``dW`` added in bf16),
  bitwise reproducible run to run; the accuracy budget is the deterministic ``dW``'s plus
  one rounding.  An empty group adds exact zeros (``out.add_`` semantics: values kept,
  ``-0.0`` becomes ``+0.0``); ``sum_m == 0`` leaves ``out`` unchanged.  Torch's
  deterministic mode changes nothing.
- Checks (``ValueError``, before anything is allocated or bound): ``out`` given,
  ``out_dtype`` None or its dtype, bf16 or fp32, no overlap of ``out`` with itself, ``g``,
  ``x`` or ``offs``.  Row and expert padding is never read or written.
- Prepared launches: every ``launch()`` and every CUDA graph replay adds again (a launch
  bumps ``out._version`` at ``launch()`` and at capture, not at a replay).  The composed
  scratch is owned by the prepared launch and allocated on the stream current at the preparation:
  keep the launch alive while a graph that captured it may replay and, after a launch on
  another stream, until that launch completes; do not run one prepared accumulate launch
  on two streams at once.
- Autograd flag (the Megatron / Transformer Engine convention): ``w.main_grad`` (fp32 or
  bf16, ``w``'s shape and device, unit stride along the last dimension, rows and experts
  that do not overlap) is checked at a forward call that autograd records and read again
  at the backward, which also refuses one sharing memory with the operands; the backward adds
  into it and returns a fresh placeholder for ``w`` when ``w`` has
  ``grad_added_to_main_grad`` (``torch.zeros`` under ``zero_out_wgrad``), else none.  Every
  backward adds, also a ``torch.autograd.grad`` whose ``inputs`` exclude ``w``.
  ``NotImplementedError`` under ``torch.jit.trace``, ``torch.compile`` and a
  ``create_graph`` backward; a reentrant checkpoint passes ``w`` without ``main_grad``
  (``ValueError``).
- Cost (``S = E * N * K``): the fused route takes no scratch.  The composed route's
  scratch takes ``4 * S`` bytes for fp32 (gate/up ``N`` 4096,
  ``K`` 6144: 384 MiB at ``E`` 4 to 3 GiB at ``E`` 32; down ``N`` 6144, ``K`` 2048: 192 MiB
  to 1.5 GiB) and ``8 * S`` bytes of traffic (written, then read back by the add); with an
  fp32 ``main_grad`` the backward's peak is ``2 * S`` bytes above the default flag's bf16
  ``w.grad`` (with a bf16 ``main_grad`` the scratch is ``2 * S`` bytes and the peak is
  equal).  Under
  ``torch.use_deterministic_algorithms(True)`` with
  ``torch.utils.deterministic.fill_uninitialized_memory`` on, torch also fills the scratch
  and the placeholder with NaN when it allocates them.

Prepared launches
-----------------

.. autosummary::
    :toctree: ../generated

    cake_moe_grouped_gemm_det_train_prepare_forward
    cake_moe_grouped_gemm_det_train_prepare_dgrad
    cake_moe_grouped_gemm_det_train_prepare_wgrad

Each one validates the arguments, binds one generated program and returns a
``CakeMoEGroupedGemmDetLaunch`` (also importable from the module) whose ``launch()``
performs no host synchronisation and is CUDA-graph capturable; the same prepared launch
stays valid when new offsets or values are written into the bound tensors.  A launch
allocates only the stream workspaces of its programs (the scheduler and wave buffers,
described in the ``cake_backend`` module docstring): the first launch on a CUDA stream
allocates that stream's buffers, later launches on the same stream reuse them and
allocate nothing, and a launch during a CUDA-graph capture, or for a device other than
the current one, allocates a set of its own (from the graph's pool under capture).  The backend package holds plain functions of the same names; a build that
vendors the package calls them directly.

Support query
-------------

.. autosummary::
    :toctree: ../generated

    is_cake_moe_grouped_gemm_det_train_supported

``is_cake_moe_grouped_gemm_det_train_supported(device=None)`` is ``True`` when the device
(default: the current CUDA device) has the programs of every operation: the forward, the
data gradient and the weight gradient.

Removed names
-------------

The former names of this family were removed with no alias: importing one raises
``ModuleNotFoundError``, ``ImportError`` or ``AttributeError``.  Every call in the
table ran the deterministic package; move each one to this module.  The table gives
the versions that had each former name: the v0.7.3 release, the v0.7.4rc1 prerelease
and ``main`` after v0.7.4rc1.  "A former deterministic package" is
``flashinfer.experimental.cake_moe_grouped_gemm`` (v0.7.3; in v0.7.4rc1 an alias of
the next one) or ``flashinfer.experimental.cake_moe_grouped_det_gemm`` (v0.7.4rc1,
``main``).  The former non-deterministic calls (``deterministic=False`` of
``flashinfer.experimental.cake.grouped_mm`` and every entry of the package
``flashinfer.experimental.cake.cake_moe_grouped_gemm``, one level lower) move to
:mod:`flashinfer.cake_moe_grouped_gemm_nondet_train`.

.. list-table::
    :header-rows: 1

    * - Removed
      - Use instead
    * - the former deterministic packages with their submodules; the module
        ``flashinfer.grouped_mm.cake`` (v0.7.3, v0.7.4rc1) or
        ``flashinfer.grouped_mm.cake_det`` (``main``)
      - ``flashinfer.cake_moe_grouped_gemm_det_train`` (backend
        ``flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train``, same submodule
        names)
    * - ``grouped_gemm_fwd`` / ``_dgrad`` / ``_wgrad`` (v0.7.3, v0.7.4rc1) and
        ``grouped_det_gemm_fwd`` / ``_dgrad`` / ``_wgrad`` (v0.7.4rc1, ``main``), from
        ``flashinfer.grouped_mm``, its module above or a former deterministic package
      - ``cake_moe_grouped_gemm_det_train_forward`` / ``_dgrad`` / ``_wgrad``
    * - ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_fwd`` / ``_dgrad`` /
        ``_wgrad`` with the default or ``deterministic=True`` (v0.7.4rc1)
      - ``cake_moe_grouped_gemm_det_train_forward`` / ``_dgrad`` / ``_wgrad``
    * - ``prepare_grouped_gemm_fwd`` / ``_dgrad`` / ``_wgrad`` (v0.7.3, v0.7.4rc1) and
        ``prepare_grouped_det_gemm_fwd`` / ``_dgrad`` / ``_wgrad`` (v0.7.4rc1, ``main``),
        from a former deterministic package
      - ``cake_moe_grouped_gemm_det_train_prepare_forward`` / ``_prepare_dgrad`` /
        ``_prepare_wgrad``
    * - ``GroupedGemmLaunch`` (v0.7.3, v0.7.4rc1) and ``GroupedDetGemmLaunch``
        (v0.7.4rc1, ``main``), from a former deterministic package
      - ``CakeMoEGroupedGemmDetLaunch``
    * - ``cake_grouped_mm`` (v0.7.3, v0.7.4rc1) and ``cake_grouped_det_mm`` (v0.7.4rc1,
        ``main``), from ``flashinfer.grouped_mm``, its module above or a former
        deterministic package
      - ``cake_moe_grouped_gemm_det``
    * - ``flashinfer.experimental.cake.grouped_mm.cake_grouped_mm`` with the default or
        ``deterministic=True`` (v0.7.4rc1)
      - ``cake_moe_grouped_gemm_det``
    * - ``CakeGroupedMm`` (v0.7.3, v0.7.4rc1) and ``CakeGroupedDetMm`` (v0.7.4rc1,
        ``main``), from a former deterministic package
      - ``CakeMoEGroupedGemmDetFunction``
    * - ``cake_backend.grouped_mm_bf16_cake`` of a former deterministic package
        (v0.7.3, v0.7.4rc1)
      - :func:`flashinfer.grouped_mm.grouped_mm_bf16` with ``backend="cake"``
    * - the ``deterministic`` argument of ``cake_grouped_mm`` / ``cake_grouped_det_mm``
        and the fourth argument of ``CakeGroupedMm.apply`` / ``CakeGroupedDetMm.apply``
        (v0.7.3, v0.7.4rc1: every value ran the deterministic programs)
      - none: the module is the mode; passing one raises ``TypeError``

:func:`flashinfer.grouped_mm.grouped_mm_bf16` with ``backend="cake"`` is unchanged: it
runs the forward of this module.
