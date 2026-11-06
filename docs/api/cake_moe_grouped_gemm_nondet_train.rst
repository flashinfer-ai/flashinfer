.. _apicake_moe_grouped_gemm_nondet_train:

Cake ragged BF16 grouped GEMM, non-deterministic weight gradient
================================================================

.. currentmodule:: flashinfer.cake_moe_grouped_gemm_nondet_train

The module :mod:`flashinfer.cake_moe_grouped_gemm_nondet_train` computes the ragged BF16
grouped GEMM of :ref:`apicake_moe_grouped_gemm_det_train` -- the forward projection, the
data gradient and the weight gradient of the expert projections of Mixture-of-Experts
layers on SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) -- with a weight
gradient whose 512-wide k tile adds its split tail with unordered fp32 reductions.  Its
functions are stable FlashInfer APIs: each one hands off to the function of the same
name in the backend package
:mod:`flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train`.  Importing the
module loads none of the backend package.  ``flashinfer`` binds the module itself on
first access, not its names: they are reached through the module only.

Mode rules
----------

The module is the mode: every entry below is non-deterministic, and every public name
carries ``nondet``.  The deterministic mode is the separate module
:mod:`flashinfer.cake_moe_grouped_gemm_det_train`, whose entries have the same shapes.

- No entry takes a ``deterministic`` argument: passing one (``True`` or ``False``)
  raises ``TypeError`` before anything runs, and so does a fourth argument of
  ``CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)`` other than ``None`` (the fifth
  argument is ``fuse_wgrad_accumulation``).  Nothing here routes to the deterministic
  module, also not as a fallback.
- Under ``torch.use_deterministic_algorithms(True)`` PyTorch's non-determinism alert is
  raised immediately before an order-varying launch only: the weight gradient whose
  512-wide k tile splits its tail (``plan["unordered_tail"]`` and
  ``plan["tail_splits"] > 1``; ``CakeMoEGroupedGemmNondetLaunch.order_varying``), in
  overwrite or accumulate mode -- ``cake_moe_grouped_gemm_nondet_train_wgrad``, a
  prepared launch's ``launch()`` (also when the mode was switched on after the
  preparation; under CUDA graph capture, when the launch is captured) and the backward
  of ``CakeMoEGroupedGemmNondetFunction`` when it computes ``dW``.  The alert names the
  counterpart in :mod:`flashinfer.cake_moe_grouped_gemm_det_train`; it is a
  ``RuntimeError`` (nothing launches), or with ``warn_only=True`` a ``UserWarning``,
  after which the non-deterministic launch runs (it names the caller's line for a direct
  call of ``cake_moe_grouped_gemm_nondet_train_wgrad`` or ``launch()``, a frame of
  PyTorch's autograd for the backward).  The preparation, the forward, the data
  gradient, the 256-wide k tile's weight gradient (ordered tail reduce) and an unsplit
  tail have a fixed order and never alert.
- No tape, saved tensor or workspace crosses between the two modes: each autograd
  function's backward reads only its own saved tensors.

A non-reentrant activation-checkpoint replay of ``cake_moe_grouped_gemm_nondet`` re-runs
the same forward, bitwise.

Numerical contract
------------------

Only the run-to-run bitwise reproducibility of the weight gradient ``dW`` is relaxed.  The
forward ``Y`` and the data gradient ``dX`` are bitwise the deterministic module's (copies
of the same programs; on sm_100a the data gradient's L2 evict-first twin is the package's
own build of the deterministic wide program, whose cache policy changes no arithmetic), and
so is ``dW`` wherever the 256-wide k tile runs.  With the
512-wide k tile, the tiles of the last partial wave are split into k-step units over the
clusters; each unit adds its fp32 accumulator into the tile's fp32 sum with unordered
reductions, and the last unit to arrive rounds the sum once and stores it, in the same
launch (the deterministic module writes fp32 partials and sums them in a fixed order in
a second launch).  With the slack tail placement, a launch whose tail tiles all fit in
the room the full waves leave on the least-loaded clusters runs each tail tile whole on
one cluster instead and touches no sum or ticket; otherwise its tail splits as above.
Every whole tile is reduced by one CTA pair over the group's rows in a fixed order.  The
operands (bf16), the accumulation (fp32), the one round-to-nearest
rounding to bf16 (or the fp32 store), the layouts, the device-resident group offsets (no
host synchronisation, no caller padding), the exact zeros of empty groups and the error
budgets are those of the deterministic module.  The split tail's fp32 sums and tickets
are workspaces kept per device and CUDA stream at zero (every launch leaves them zero);
a CUDA graph capture takes zeroed buffers of its own.

The operator is local: on a context-parallel rank it runs on the rank's local rows, adds
no collective, and leaves the reduction of a parameter gradient across ranks to the
caller.

Weight-gradient accumulation
----------------------------

``cake_moe_grouped_gemm_nondet_train_wgrad(..., out=out, accumulate=True)`` adds the weight gradient into the
caller's bf16 or fp32 ``out`` (``out[e] += G[e].T @ X[e]``, Megatron's gradient-accumulation
fusion) and ``cake_moe_grouped_gemm_nondet(..., fuse_wgrad_accumulation=True)`` adds it into
``w.main_grad`` in its backward.  Both default to off; with the defaults nothing changes.
The fused accumulate builds of this release add into an fp32 ``out`` only; a bf16 ``out``
always takes the composed route (the bf16 overwrite, then the same-dtype ``out.add_``:
autograd's AccumulateGrad bits and cost).  The host plans the launch exactly as the
overwrite of ``out``'s dtype and records the route in ``plan["accumulate"]``:

============== ======================================================================== =========================================================
route          when                                                                     what runs
============== ======================================================================== =========================================================
``"fused"``    512-wide k tile with its unordered split tail, the selected stage has an  one launch whose epilogue adds the tile into ``out``
               accumulate build on the architecture whose library can load, ``out``
               32-byte aligned
``"composed"`` otherwise (``plan["accumulate_reason"]``, the first that fails:          the overwrite into a scratch of ``out``'s dtype, then
               ``"tile_k"``, ``"tail"``, ``"stage"`` -- every bf16 ``out`` --,          ``out.add_``
               ``"min_cuda"``, ``"alignment"``; the last two warn once per process)
============== ======================================================================== =========================================================

An accumulate build is bound by a static rule (nothing is compiled, loaded or test-built to
decide it): its prebuilt library exists, or the version of FlashInfer's JIT toolkit reaches
its ``build.json`` ``min_cuda``.  The fp32 accumulate builds need CUDA 13.4; below it the
call composes instead of failing.  ``out`` must not overlap itself, ``g``, ``x`` or ``offs``
(``ValueError`` on both routes, before anything runs).  Every launch and every CUDA graph
replay adds again.  Numerical contract: the fused route adds the tile's fp32 sum into the
fp32 ``out`` with one round-to-nearest add that keeps subnormals (bitwise
``out0.add_(dW32)`` on every whole tile); a split tail tile adds its unordered sum ``S``
once (``S`` is the overwrite's partial sum, whose unit reductions flush subnormals; the
nondeterminism scope is unchanged); the composed route is ``out.add_`` of the overwrite in
``out``'s dtype; an empty group is not touched by the fused route, while the composed
route has ``out.add_``'s semantics (values kept, ``-0.0`` may become ``+0.0``).  The
composed scratch (the size of the weight gradient) is owned by the prepared launch: keep
the launch alive while a CUDA graph that captured it may replay.  The autograd flag returns
a placeholder gradient for ``w`` when ``w`` has ``grad_added_to_main_grad``
(``torch.zeros`` under ``zero_out_wgrad``), else none; it adds at every backward, also when
``torch.autograd.grad``'s ``inputs`` exclude ``w`` (split backward-weight schedules are not
supported), and raises ``NotImplementedError`` under ``torch.jit.trace``, ``torch.compile``
and a ``create_graph`` backward.

The module hands ``accumulate`` and ``fuse_wgrad_accumulation`` to the package; it has
no deterministic route.  The deterministic module takes the same two switches and always
composes (its deterministic overwrite into a scratch, then ``out.add_``), bitwise
reproducible run to run: see :ref:`apicake_moe_grouped_gemm_det_train`.

One-shot functions
------------------

.. autosummary::
    :toctree: ../generated

    cake_moe_grouped_gemm_nondet_train_forward
    cake_moe_grouped_gemm_nondet_train_dgrad
    cake_moe_grouped_gemm_nondet_train_wgrad

Autograd
--------

.. autosummary::
    :toctree: ../generated

    cake_moe_grouped_gemm_nondet

``cake_moe_grouped_gemm_nondet(x, w, offs, *, fuse_wgrad_accumulation=False)`` returns
``Y``, with ``dX`` and ``dW`` in the backward (``dW`` in the dtype of ``w``, or added
into ``w.main_grad``)::

    from flashinfer.cake_moe_grouped_gemm_nondet_train import (
        cake_moe_grouped_gemm_nondet,
        cake_moe_grouped_gemm_nondet_train_wgrad,
    )

    y = cake_moe_grouped_gemm_nondet(x, w, offs)
    dw = cake_moe_grouped_gemm_nondet_train_wgrad(g, x, offs, out_dtype=torch.float32)

The shapes, the group offsets (``offs[E]`` end offsets or ``m_indptr[E + 1]``) and the
alignment rules are those of :ref:`apicake_moe_grouped_gemm_det_train`.  The stable
``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` stays on the
deterministic module.

Prepared launches
-----------------

.. autosummary::
    :toctree: ../generated

    cake_moe_grouped_gemm_nondet_train_prepare_forward
    cake_moe_grouped_gemm_nondet_train_prepare_dgrad
    cake_moe_grouped_gemm_nondet_train_prepare_wgrad

Each one returns a ``CakeMoEGroupedGemmNondetLaunch`` (also importable from the module)
whose ``plan`` records, for the weight gradient, ``unordered_tail``, ``tail_splits`` and
the sizes of the tail sums and tickets.

Support query
-------------

.. autosummary::
    :toctree: ../generated

    is_cake_moe_grouped_gemm_nondet_train_supported

``is_cake_moe_grouped_gemm_nondet_train_supported(device=None)`` is ``True`` when the
device (default: the current CUDA device) has the programs of every operation: the
forward, the data gradient and the weight gradient.

Names
-----

Both modes have the same entries; the module and every name carry the mode:

.. list-table::
    :header-rows: 1

    * - deterministic (:mod:`flashinfer.cake_moe_grouped_gemm_det_train`)
      - non-deterministic (:mod:`flashinfer.cake_moe_grouped_gemm_nondet_train`)
    * - ``cake_moe_grouped_gemm_det_train_forward`` / ``_dgrad`` / ``_wgrad``
      - ``cake_moe_grouped_gemm_nondet_train_forward`` / ``_dgrad`` / ``_wgrad``
    * - ``cake_moe_grouped_gemm_det_train_prepare_forward`` / ``_prepare_dgrad`` /
        ``_prepare_wgrad``
      - ``cake_moe_grouped_gemm_nondet_train_prepare_forward`` / ``_prepare_dgrad`` /
        ``_prepare_wgrad``
    * - ``CakeMoEGroupedGemmDetLaunch``
      - ``CakeMoEGroupedGemmNondetLaunch``
    * - ``cake_moe_grouped_gemm_det`` / ``CakeMoEGroupedGemmDetFunction``
      - ``cake_moe_grouped_gemm_nondet`` / ``CakeMoEGroupedGemmNondetFunction``
    * - ``is_cake_moe_grouped_gemm_det_train_supported``
      - ``is_cake_moe_grouped_gemm_nondet_train_supported``

The former names of this module were removed with no alias: importing one raises
``ModuleNotFoundError``, ``ImportError`` or ``AttributeError``.  Only the calls that
passed ``deterministic=False`` move here.  The calls of
``flashinfer.experimental.cake.grouped_mm`` without ``deterministic=False`` (the default
``deterministic=True``) ran the deterministic package: they move to
:mod:`flashinfer.cake_moe_grouped_gemm_det_train` (see its table), as do the names
without ``det`` of ``flashinfer.grouped_mm``, ``flashinfer.grouped_mm.cake``,
``flashinfer.experimental.cake_moe_grouped_det_gemm`` and
``flashinfer.experimental.cake_moe_grouped_gemm``, which were aliases of the
deterministic objects.

.. list-table::
    :header-rows: 1

    * - Removed
      - Use instead
    * - the package ``flashinfer.experimental.cake.cake_moe_grouped_gemm`` with its
        submodules (every entry required ``deterministic=False``)
      - ``flashinfer.cake_moe_grouped_gemm_nondet_train`` (backend
        ``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train``, same submodule
        names)
    * - ``grouped_gemm_fwd`` / ``_dgrad`` / ``_wgrad`` with ``deterministic=False``, from
        ``flashinfer.experimental.cake.grouped_mm`` or the package above
      - ``cake_moe_grouped_gemm_nondet_train_forward`` / ``_dgrad`` / ``_wgrad``
    * - ``prepare_grouped_gemm_fwd`` / ``_dgrad`` / ``_wgrad`` with
        ``deterministic=False``, from the package above
      - ``cake_moe_grouped_gemm_nondet_train_prepare_forward`` / ``_prepare_dgrad`` /
        ``_prepare_wgrad``
    * - ``GroupedGemmLaunch``, from the package above
      - ``CakeMoEGroupedGemmNondetLaunch``
    * - ``cake_grouped_mm`` with ``deterministic=False``, from
        ``flashinfer.experimental.cake.grouped_mm`` or the package above
      - ``cake_moe_grouped_gemm_nondet``
    * - ``CakeGroupedMm``, from the package above
      - ``CakeMoEGroupedGemmNondetFunction``
    * - the ``deterministic`` argument of every entry
      - the module of the mode: ``cake_moe_grouped_gemm_det_train`` or
        ``cake_moe_grouped_gemm_nondet_train``; passing the argument raises ``TypeError``

One behavior changes.  Under ``torch.use_deterministic_algorithms(True)``, a call of
``flashinfer.experimental.cake.grouped_mm`` with ``deterministic=False`` ran the
deterministic package and warned once.  This module never falls back: it raises
PyTorch's non-determinism alert (``RuntimeError``) right before an order-varying weight
gradient launch, and with ``warn_only=True`` it warns (``UserWarning``) and runs the
non-deterministic launch.  Use :mod:`flashinfer.cake_moe_grouped_gemm_det_train` where
the deterministic mode must hold.

Features by architecture
------------------------

=============================================== =============== =============== ===============
feature                                         sm_100a         sm_103a         sm_107a
=============================================== =============== =============== ===============
``fwd`` / ``dgrad`` 256- and 512-column tiles   domain schedule domain schedule static schedule
``fwd`` evict-first / 3-3 head-tail twin        -- / --         -- / yes        yes / --
``dgrad`` evict-first / 3-3 head-tail twin      yes / --        -- / yes        yes / --
``dgrad`` probed queue split                    yes             yes             --
``wgrad`` 256-wide k tile, ordered tail reduce  yes             yes             yes
``wgrad`` 512-wide k tile, unordered split tail yes             yes             yes
``wgrad`` 512 wave-synchronised builds          yes             yes             --
``wgrad`` 512 half-release builds               planned, slack  planned, slack  planned, slack
``wgrad`` 512 k256-tail twin                    --              --              slack tail
``wgrad`` output dtypes                         bf16, fp32      bf16, fp32      bf16, fp32
``wgrad`` accumulate, fused, fp32 ``out``       planned, wave   planned, wave   planned, k256t
CUDA toolkit floor                              12.8            12.9            13.4
CUDA toolkit floor, fp32 accumulate builds      13.4            13.4            13.4
=============================================== =============== =============== ===============
