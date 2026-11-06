.. _apicake_moe_grouped_gemm:

Cake ragged BF16 grouped GEMM, non-deterministic weight gradient (experimental)
================================================================================

The functions of ``flashinfer.experimental.cake.grouped_mm`` compute the ragged BF16
grouped GEMM of :ref:`apicake_moe_grouped_det_gemm` -- the forward projection, the data
gradient and the weight gradient of the expert projections of Mixture-of-Experts layers
on SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) -- with the
non-deterministic package ``flashinfer.experimental.cake.cake_moe_grouped_gemm``, whose
512-wide k tile adds the split tail of the weight gradient with unordered fp32
reductions.  Both the API and the backends are experimental (see :ref:`experimental`);
calling one of the functions is the explicit opt-in and emits an ``ExperimentalWarning``
once.  The deterministic implementation is a separate package with ``det`` in every
name (:ref:`apicake_moe_grouped_det_gemm`).

Mode and dispatch rules
-----------------------

The facade functions and the package are non-deterministic only; every ND example passes
``deterministic=False``.

- ``deterministic`` is keyword-only on every entry (``cake_grouped_mm`` included) and
  must be the bool ``False``: the call runs the non-deterministic package.
- Every other value -- the default ``True``, ``None``, ``0``, ``1``, ``"False"`` --
  raises ``ValueError`` naming the deterministic entry (``grouped_det_gemm_fwd`` /
  ``_dgrad`` / ``_wgrad``, ``prepare_grouped_det_gemm_*``, ``cake_grouped_det_mm``) and
  saying "pass deterministic=False for the non-deterministic path"; nothing runs.
  Nothing here routes to the deterministic package, also not as a fallback.
- Under ``torch.use_deterministic_algorithms(True)`` PyTorch's non-determinism alert is
  raised immediately before an order-varying launch only: the weight gradient whose
  512-wide k tile splits its tail (``plan["unordered_tail"]`` and
  ``plan["tail_splits"] > 1``; ``GroupedGemmLaunch.order_varying``), in overwrite or
  accumulate mode -- ``grouped_gemm_wgrad``, a prepared launch's ``launch()`` (also
  when the mode was switched on after the preparation; under CUDA graph capture, when
  the launch is captured) and the backward of ``CakeGroupedMm`` when it computes
  ``dW``.  The alert is a ``RuntimeError`` (nothing launches), or with
  ``warn_only=True`` a ``UserWarning`` naming the caller's line, after which the launch
  runs.  The preparation, the forward, the data gradient, the 256-wide k tile's weight
  gradient (ordered tail reduce) and an unsplit tail have a fixed order and never alert.
- No tape, saved tensor or workspace crosses between the two packages: each autograd
  function's backward reads only its own saved tensors.

The mode of ``cake_grouped_mm`` is fixed at its forward call; a non-reentrant
activation-checkpoint replay re-runs the same forward, bitwise.

Numerical contract of ``deterministic=False``
---------------------------------------------

Only the run-to-run bitwise reproducibility of the weight gradient ``dW`` is relaxed.  The
forward ``Y`` and the data gradient ``dX`` are bitwise the deterministic package's (copies
of the same programs; on sm_100a the data gradient's L2 evict-first twin is the package's
own build of the deterministic wide program, whose cache policy changes no arithmetic), and
so is ``dW`` wherever the 256-wide k tile runs.  With the
512-wide k tile, the tiles of the last partial wave are split into k-step units over the
clusters; each unit adds its fp32 accumulator into the tile's fp32 sum with unordered
reductions, and the last unit to arrive rounds the sum once and stores it, in the same
launch (the deterministic package writes fp32 partials and sums them in a fixed order in
a second launch).  With the slack tail placement, a launch whose tail tiles all fit in
the room the full waves leave on the least-loaded clusters runs each tail tile whole on
one cluster instead and touches no sum or ticket; otherwise its tail splits as above.
Every whole tile is reduced by one CTA pair over the group's rows in a fixed order.  The
operands (bf16), the accumulation (fp32), the one round-to-nearest
rounding to bf16 (or the fp32 store), the layouts, the device-resident group offsets (no
host synchronisation, no caller padding), the exact zeros of empty groups and the error
budgets are those of the deterministic package.  The split tail's fp32 sums and tickets
are workspaces kept per device and CUDA stream at zero (every launch leaves them zero);
a CUDA graph capture takes zeroed buffers of its own.

The operator is local: on a context-parallel rank it runs on the rank's local rows, adds
no collective, and leaves the reduction of a parameter gradient across ranks to the
caller.

Weight-gradient accumulation
----------------------------

``grouped_gemm_wgrad(..., out=out, accumulate=True)`` adds the weight gradient into the
caller's bf16 or fp32 ``out`` (``out[e] += G[e].T @ X[e]``, Megatron's gradient-accumulation
fusion) and ``cake_grouped_mm(..., fuse_wgrad_accumulation=True)`` adds it into
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

The facade hands ``accumulate`` and ``fuse_wgrad_accumulation`` to the package with the
same mode rules; it has no deterministic route.

Functions
---------

The experimental functions are documented by their docstrings (they are not listed for
autodoc before graduation):

- ``grouped_gemm_fwd(x, w, offs, out=None, *, deterministic=True)``: ``Y = X @ W[e].T``
  per group, bf16 ``[sum_m, N]``;
- ``grouped_gemm_dgrad(g, w, offs, out=None, *, deterministic=True)``: ``dX = G @ W[e]``
  per group (``W`` read in place), bf16 ``[sum_m, K]``;
- ``grouped_gemm_wgrad(g, x, offs, out=None, out_dtype=None, num_groups=None, *,
  deterministic=True, accumulate=False)``: ``dW[e] = G[e].T @ X[e]``, bf16 or fp32
  ``[E, N, K]`` (with ``accumulate=True``, ``out[e] += dW[e]``);
- ``cake_grouped_mm(x, w, offs, *, deterministic=True, fuse_wgrad_accumulation=False)``: the
  autograd wrapper (``Y``, with ``dX`` and ``dW`` in the backward, ``dW`` in the dtype of
  ``w``, or added into ``w.main_grad``).

Each takes ``deterministic=False`` only (the default ``True`` raises ``ValueError``,
above)::

    from flashinfer.experimental.cake.grouped_mm import cake_grouped_mm, grouped_gemm_wgrad

    y = cake_grouped_mm(x, w, offs, deterministic=False)
    dw = grouped_gemm_wgrad(g, x, offs, out_dtype=torch.float32, deterministic=False)

The shapes, the group offsets (``offs[E]`` end offsets or ``m_indptr[E + 1]``) and the
alignment rules are those of :ref:`apicake_moe_grouped_det_gemm`.  The prepare-once /
launch-many surface is in the packages:
``flashinfer.experimental.cake.cake_moe_grouped_gemm.prepare_grouped_gemm_fwd``,
``prepare_grouped_gemm_dgrad`` and ``prepare_grouped_gemm_wgrad`` (with
``deterministic=False``) return a ``GroupedGemmLaunch`` whose ``plan`` records, for the
weight gradient, ``unordered_tail``, ``tail_splits`` and the sizes of the tail sums and
tickets.  The stable ``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` stays
on the deterministic package.

Names
-----

The deterministic names carry ``det``; the non-deterministic names omit it:

======================================================== ==============================================================
deterministic (no ``deterministic`` argument)            non-deterministic (``deterministic=False`` only)
======================================================== ==============================================================
``flashinfer.grouped_mm.grouped_det_gemm_fwd``           ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_fwd``
``flashinfer.grouped_mm.grouped_det_gemm_dgrad``         ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_dgrad``
``flashinfer.grouped_mm.grouped_det_gemm_wgrad``         ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_wgrad``
``flashinfer.grouped_mm.cake_grouped_det_mm``            ``flashinfer.experimental.cake.grouped_mm.cake_grouped_mm``
``cake_moe_grouped_det_gemm.prepare_grouped_det_gemm_*`` ``cake.cake_moe_grouped_gemm.prepare_grouped_gemm_*``
``cake_moe_grouped_det_gemm.GroupedDetGemmLaunch``       ``cake.cake_moe_grouped_gemm.GroupedGemmLaunch``
``cake_moe_grouped_det_gemm.CakeGroupedDetMm``           ``cake.cake_moe_grouped_gemm.CakeGroupedMm``
======================================================== ==============================================================

The former det-free aliases of the deterministic objects (the package path
``flashinfer.experimental.cake_moe_grouped_gemm``, the module ``flashinfer.grouped_mm.cake``
and ``flashinfer.grouped_mm.grouped_gemm_*`` / ``cake_grouped_mm``) were removed with no
alias; the det-free spellings name only the non-deterministic objects above.

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
