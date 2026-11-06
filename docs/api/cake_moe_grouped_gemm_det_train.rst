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
- Prepared launches: every ``launch()`` and every CUDA graph replay adds again (the add
  bumps ``out._version`` at ``launch()`` and at capture, not at a replay).  The scratch is
  owned by the prepared launch and allocated on the stream current at the preparation:
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
- Cost (``S = E * N * K``): the scratch takes ``4 * S`` bytes for fp32 (gate/up ``N`` 4096,
  ``K`` 6144: 384 MiB at ``E`` 4 to 3 GiB at ``E`` 32; down ``N`` 6144, ``K`` 2048: 192 MiB
  to 1.5 GiB) and ``8 * S`` bytes of traffic (written, then read back by the add); with an
  fp32 ``main_grad`` the backward's peak is ``2 * S`` bytes above the default flag's bf16
  ``w.grad`` (with a bf16 ``main_grad`` the scratch is ``2 * S`` bytes and the peak is
  equal).  Under
  ``torch.use_deterministic_algorithms(True)`` with
  ``torch.utils.deterministic.fill_uninitialized_memory`` on, torch also fills the scratch
  and the placeholder with NaN when it allocates them.  The non-deterministic module adds
  in the kernel epilogue instead (:ref:`apicake_moe_grouped_gemm_nondet_train`).

Prepared launches
-----------------

.. autosummary::
    :toctree: ../generated

    cake_moe_grouped_gemm_det_train_prepare_forward
    cake_moe_grouped_gemm_det_train_prepare_dgrad
    cake_moe_grouped_gemm_det_train_prepare_wgrad

Each one validates the arguments, binds one generated program and returns a
``CakeMoEGroupedGemmDetLaunch`` (also importable from the module) whose ``launch()``
performs no allocation and no host synchronisation and is CUDA-graph capturable; the
same prepared launch stays valid when new offsets or values are written into the bound
tensors.  The backend package holds plain functions of the same names; a build that
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
``ModuleNotFoundError``, ``ImportError`` or ``AttributeError``.  Every call in the table
ran the deterministic package, including the names without ``det`` (aliases of the
deterministic objects) and the selectable entries of
``flashinfer.experimental.cake.grouped_mm`` called without ``deterministic=False``.
Move each one to this module.  The calls that passed ``deterministic=False`` move to
:mod:`flashinfer.cake_moe_grouped_gemm_nondet_train` instead.

.. list-table::
    :header-rows: 1

    * - Removed
      - Use instead
    * - the packages ``flashinfer.experimental.cake_moe_grouped_det_gemm`` and
        ``flashinfer.experimental.cake_moe_grouped_gemm`` (its alias), with their
        submodules, and the module ``flashinfer.grouped_mm.cake``
      - ``flashinfer.cake_moe_grouped_gemm_det_train`` (backend
        ``flashinfer.experimental.cake.cake_moe_grouped_gemm_det_train``, same submodule
        names)
    * - ``grouped_det_gemm_fwd`` / ``_dgrad`` / ``_wgrad`` and their aliases
        ``grouped_gemm_fwd`` / ``_dgrad`` / ``_wgrad``, from ``flashinfer.grouped_mm``,
        ``flashinfer.grouped_mm.cake`` or the two packages above
      - ``cake_moe_grouped_gemm_det_train_forward`` / ``_dgrad`` / ``_wgrad``
    * - ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_fwd`` / ``_dgrad`` /
        ``_wgrad`` without ``deterministic=False``
      - ``cake_moe_grouped_gemm_det_train_forward`` / ``_dgrad`` / ``_wgrad``
    * - ``prepare_grouped_det_gemm_fwd`` / ``_dgrad`` / ``_wgrad`` and their aliases
        ``prepare_grouped_gemm_fwd`` / ``_dgrad`` / ``_wgrad``, from the two packages above
      - ``cake_moe_grouped_gemm_det_train_prepare_forward`` / ``_prepare_dgrad`` /
        ``_prepare_wgrad``
    * - ``GroupedDetGemmLaunch`` and its alias ``GroupedGemmLaunch``, from the two
        packages above
      - ``CakeMoEGroupedGemmDetLaunch``
    * - ``cake_grouped_det_mm`` and its alias ``cake_grouped_mm``, from
        ``flashinfer.grouped_mm``, ``flashinfer.grouped_mm.cake`` or the two packages
        above
      - ``cake_moe_grouped_gemm_det``
    * - ``flashinfer.experimental.cake.grouped_mm.cake_grouped_mm`` without
        ``deterministic=False``
      - ``cake_moe_grouped_gemm_det``
    * - ``CakeGroupedDetMm`` and its alias ``CakeGroupedMm``, from the two packages above
      - ``CakeMoEGroupedGemmDetFunction``
    * - ``cake_backend.grouped_mm_bf16_cake`` of the two packages above
      - :func:`flashinfer.grouped_mm.grouped_mm_bf16` with ``backend="cake"`` (unchanged)
    * - the ``deterministic`` argument
      - the module of the mode; passing the argument raises ``TypeError``
