.. _apicake_moe_grouped_gemm_nondet_train:

Cake ragged BF16 grouped GEMM, one-launch weight-gradient tail
==============================================================

.. currentmodule:: flashinfer.cake_moe_grouped_gemm_nondet_train

The module :mod:`flashinfer.cake_moe_grouped_gemm_nondet_train` computes the ragged BF16
grouped GEMM of :ref:`apicake_moe_grouped_gemm_det_train` -- the forward projection, the
data gradient and the weight gradient of the expert projections of Mixture-of-Experts
layers on SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) -- with a weight
gradient whose 512-wide k tile finishes its split tail in the same launch, in a fixed
order (the deterministic module adds a second tail-reduce launch).  Every output repeats
bit for bit for the same inputs, shapes, device SM count and configuration.  Its
functions are stable FlashInfer APIs: each one hands off to the function of the same
name in the backend package
:mod:`flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train`.  Importing the
module loads none of the backend package.  ``flashinfer`` binds the module itself on
first access, not its names: they are reached through the module only.

Mode rules
----------

The module is the mode: every entry below runs the non-deterministic package, and every
public name carries ``nondet``.  The deterministic mode is the separate module
:mod:`flashinfer.cake_moe_grouped_gemm_det_train`, whose entries have the same shapes.

- No entry takes a ``deterministic`` argument: passing one (``True`` or ``False``)
  raises ``TypeError`` before anything runs, and so does a fourth argument of
  ``CakeMoEGroupedGemmNondetFunction.apply(x, w, offs)`` other than ``None`` (the fifth
  argument is ``fuse_wgrad_accumulation``).  Nothing here routes to the deterministic
  module, also not as a fallback.
- Every launch of this release runs in a fixed order, so under
  ``torch.use_deterministic_algorithms(True)`` no launch raises PyTorch's
  non-determinism alert, and the bits are those of a run with the mode off
  (``CakeMoEGroupedGemmNondetLaunch.order_varying`` is ``False`` for every launch).
  The alert stays for an order-varying launch: a weight gradient planned with
  ``plan["unordered_tail"]`` and a split tail, which only an architecture listed in the
  package's ``unordered_tail_archs`` (empty) plans.  Such a launch would raise the
  alert immediately before it runs -- in ``cake_moe_grouped_gemm_nondet_train_wgrad``,
  a prepared launch's ``launch()`` (under CUDA graph capture, when the launch is
  captured) and the backward of ``CakeMoEGroupedGemmNondetFunction`` -- as a
  ``RuntimeError`` (nothing launches), or with ``warn_only=True`` a ``UserWarning``,
  after which the launch runs.  The alert names the counterpart in
  :mod:`flashinfer.cake_moe_grouped_gemm_det_train`.  The preparation never alerts.
- No tape, saved tensor or workspace crosses between the two modes: each autograd
  function's backward reads only its own saved tensors.

A non-reentrant activation-checkpoint replay of ``cake_moe_grouped_gemm_nondet`` re-runs
the same forward, bitwise.

Numerical contract
------------------

Every output (``Y``, ``dX``, ``dW`` in bf16 or fp32, and the accumulated ``out`` or
``w.main_grad``) repeats bit for bit for the same inputs, shapes, device SM count and
configuration.  The forward ``Y`` and the data gradient ``dX`` are bitwise the
deterministic module's (copies of the same programs), and so is ``dW`` wherever the
256-wide k tile runs or the 512-wide k tile does not split its tail.  With the 512-wide
k tile, the tiles of the last partial wave are split into ``tail_splits`` k-step units
over the clusters.
Unit ``s`` of split tail tile ``j`` stores its fp32 partial to slot
``j * tail_splits + s`` of the fp32 slots (``tail_sums``), and the tile's last unit to
arrive sums the tile's slots from 0.0 in slot order, rounds the sum once and stores it,
in the same launch (the deterministic module sums the partials in a second launch).
With the slack tail placement, a launch whose tail tiles all fit in the room the full
waves leave on the least-loaded clusters runs each tail tile whole on one cluster
instead and touches no slot or ticket; otherwise its tail splits as above.  Every whole
tile is reduced by one CTA pair over the group's rows in a fixed order.  The bits of the
512-wide k tile's ``dW`` depend on the SM count and on ``tail_splits``; they can differ
from the deterministic module's bits, whose tail plan and group order differ, within its
error budgets.  The operands (bf16), the accumulation (fp32), the one round-to-nearest
rounding to bf16 (or the fp32 store), the layouts, the device-resident group offsets (no
host synchronisation, no caller padding), the exact zeros of empty groups and the error
budgets are those of the deterministic module.  The slot sum uses fp32 adds without
flush to zero.  The split tail's slots and tickets are workspaces kept per device and
CUDA stream.  Only the tickets rest at zero (every launch leaves them zero); the slots
need no value between launches and are allocated with ``torch.empty``.  A CUDA graph
capture takes buffers of its own.

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
``"fused"``    512-wide k tile with its ordered one-launch split tail, the selected     one launch whose epilogue adds the tile into ``out``
               stage has an accumulate build on the architecture whose library can
               load, ``out`` 32-byte aligned
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
``out0.add_(dW32)`` on every whole tile); a split tail tile adds its slot sum ``S`` once
(``S`` is the value the fp32 overwrite of the same plan stores, so the tile is also
``out0.add_(dW32)`` and repeats bit for bit); the composed route is ``out.add_`` of the
overwrite in ``out``'s dtype; an empty group is not touched by the fused route, while the
composed route has ``out.add_``'s semantics (values kept, ``-0.0`` may become ``+0.0``).  The
composed scratch (the size of the weight gradient) is owned by the prepared launch: keep
the launch alive while a CUDA graph that captured it may replay.  The autograd flag returns
a placeholder gradient for ``w`` when ``w`` has ``grad_added_to_main_grad``
(``torch.zeros`` under ``zero_out_wgrad``), else none; it adds at every backward, also when
``torch.autograd.grad``'s ``inputs`` exclude ``w`` (split backward-weight schedules are not
supported), and raises ``NotImplementedError`` under ``torch.jit.trace``, ``torch.compile``
and a ``create_graph`` backward.

The module hands ``accumulate`` and ``fuse_wgrad_accumulation`` to the package; it has
no deterministic route.  The deterministic module takes the same two switches and always
composes (its deterministic overwrite into a scratch, then ``out.add_``): see
:ref:`apicake_moe_grouped_gemm_det_train`.

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
whose ``plan`` records, for the weight gradient, ``ordered_tail`` (``True`` on the
512-wide k tile), ``unordered_tail`` (``False``), ``tail_splits`` and the sizes of the
tail slots and tickets (``tail_sum_elems``, ``tail_ticket_words``).

The slot buffer holds ``tail_splits`` slots per tail tile, so it is ``tail_splits`` times
larger than one fp32 sum per tile.  For each stream, the package keeps the largest buffer
that a launch needed, for the life of the process.  ``tail_sum_elems`` is at most
``(clusters - 1) * 16 * 256 * 512`` floats (at most 16 units per tail tile): about
600 MiB per stream at 76 clusters (GB300) and about 840 MiB at 106 clusters (R200, 212
SMs).  For example, the gate/up projection with one group of 112,576 rows on GB300 splits
40 tail tiles into 9 units each and takes about 180 MiB (about 20 MiB with one sum per
tile).

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
``ModuleNotFoundError``, ``ImportError`` or ``AttributeError``.  The table lists the
calls that ran the non-deterministic package, in the v0.7.4rc1 prerelease and on
``main`` after it (v0.7.3 had no non-deterministic package).  Every entry of the
package ``flashinfer.experimental.cake.cake_moe_grouped_gemm`` ran only with
``deterministic=False``; any other value raised ``ValueError``.  The module
``flashinfer.experimental.cake.grouped_mm`` selected the mode with its
``deterministic`` argument: in v0.7.4rc1 the default (``True``) ran the
deterministic package, so a call without ``deterministic=False`` moves to
:mod:`flashinfer.cake_moe_grouped_gemm_det_train`, with every former deterministic
name (see its table); on ``main`` only ``deterministic=False`` ran.

.. list-table::
    :header-rows: 1

    * - Removed
      - Use instead
    * - the package ``flashinfer.experimental.cake.cake_moe_grouped_gemm`` with its
        submodules
      - ``flashinfer.cake_moe_grouped_gemm_nondet_train`` (backend
        ``flashinfer.experimental.cake.cake_moe_grouped_gemm_nondet_train``, same submodule
        names)
    * - the module ``flashinfer.experimental.cake.grouped_mm``
      - ``flashinfer.cake_moe_grouped_gemm_nondet_train`` for its calls with
        ``deterministic=False``; :mod:`flashinfer.cake_moe_grouped_gemm_det_train` for
        its other calls (v0.7.4rc1)
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

Under ``torch.use_deterministic_algorithms(True)`` this module runs the same launches as
without the mode and gives the same bits: no launch of this release alerts.  It never
falls back to the deterministic module: use
:mod:`flashinfer.cake_moe_grouped_gemm_det_train` where the deterministic module's bits
are required.  This differs from v0.7.4rc1, where a call of
``flashinfer.experimental.cake.grouped_mm`` with ``deterministic=False`` ran the
deterministic package with a one-time ``UserWarning``, and the package
``flashinfer.experimental.cake.cake_moe_grouped_gemm`` alerted at every weight
gradient call, its preparation included.  On ``main`` after v0.7.4rc1 both alerted
only right before an order-varying weight gradient launch, as this module did before
its split tail became ordered.

Features by architecture
------------------------

================================================== ====================== ====================== ======================
feature                                            sm_100a                sm_103a                sm_107a
================================================== ====================== ====================== ======================
``fwd`` / ``dgrad`` 256- and 512-column tiles      domain schedule        domain schedule        static schedule
``fwd`` evict-first / 3-3 head-tail twin           yes / --               yes / yes              yes / --
``dgrad`` evict-first / 3-3 head-tail twin         yes / --               -- / yes               yes / --
``dgrad`` probed queue split                       yes                    yes                    --
``wgrad`` 256-wide k tile, two-launch tail reduce  yes                    yes                    yes
``wgrad`` 512-wide k tile, ordered one-launch tail yes                    yes                    yes
``wgrad`` 512 wave-synchronised builds             yes                    yes                    --
``wgrad`` 512 half-release builds                  planned, slack         planned, slack         planned, slack
``wgrad`` 512 k256-tail twin                       --                     --                     slack tail
``wgrad`` 512 queue-walk builds                    planned tail           planned tail           --
``wgrad`` output dtypes                            bf16, fp32             bf16, fp32             bf16, fp32
``wgrad`` accumulate, fused, fp32 ``out``          planned, wave, queue   planned, wave, queue   planned, k256t
CUDA toolkit floor                                 12.8                   12.9                   13.4
CUDA toolkit floor, fp32 accumulate builds         13.4                   13.4                   13.4
================================================== ====================== ====================== ======================
