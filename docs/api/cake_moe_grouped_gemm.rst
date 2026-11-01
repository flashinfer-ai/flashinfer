.. _apicake_moe_grouped_gemm:

Cake ragged BF16 grouped GEMM with a selectable numerical mode (experimental)
=============================================================================

The functions of ``flashinfer.experimental.cake.grouped_mm`` compute the ragged BF16
grouped GEMM of :ref:`apicake_moe_grouped_det_gemm` -- the forward projection, the data
gradient and the weight gradient of the expert projections of Mixture-of-Experts layers
on SM100 (B200 / GB200), SM103 (B300 / GB300) and SM107 (R200) -- and take a
``deterministic: bool = True`` keyword that selects the implementation.  Both the API and
the backends are experimental (see :ref:`experimental`); calling one of the functions is
the explicit opt-in and emits an ``ExperimentalWarning`` once.

Dispatch
--------

================== ================================================ ==================================================================================
``deterministic``  ``torch.are_deterministic_algorithms_enabled()`` runs
================== ================================================ ==================================================================================
``True`` (default) any                                              the deterministic package ``flashinfer.experimental.cake_moe_grouped_det_gemm``
``False``          ``False``                                        ``flashinfer.experimental.cake.cake_moe_grouped_gemm``
``False``          ``True``                                         the deterministic package, with one ``UserWarning`` per entry point
================== ================================================ ==================================================================================

Determinism requested anywhere is never silently dropped.  Called directly, the package
``flashinfer.experimental.cake.cake_moe_grouped_gemm`` accepts only ``deterministic=False``
(``ValueError`` otherwise); under ``torch.use_deterministic_algorithms(True)`` its weight
gradient entry points (``prepare_grouped_gemm_wgrad``, ``grouped_gemm_wgrad`` and the
backward of ``CakeGroupedMm``) raise PyTorch's non-determinism alert (a
``RuntimeError``, or a ``UserWarning`` with ``warn_only=True``) and do not fall back; a
prepared weight gradient checks again at every launch (under CUDA graph capture, when
the launch is captured).  The mode of
``cake_grouped_mm`` is fixed at its forward call; a non-reentrant activation-checkpoint
replay re-runs the same forward, bitwise.

Numerical contract of ``deterministic=False``
---------------------------------------------

Only the run-to-run bitwise reproducibility of the weight gradient ``dW`` is relaxed.  The
forward ``Y`` and the data gradient ``dX`` are bitwise the deterministic package's (copies
of the same programs), and so is ``dW`` wherever the 256-wide k tile runs.  With the
512-wide k tile, the tiles of the last partial wave are split into k-step units over the
clusters; each unit adds its fp32 accumulator into the tile's fp32 sum with unordered
reductions, and the last unit to arrive rounds the sum once and stores it, in the same
launch (the deterministic package writes fp32 partials and sums them in a fixed order in
a second launch).  Every whole tile is reduced by one CTA pair over the group's rows in a
fixed order.  The operands (bf16), the accumulation (fp32), the one round-to-nearest
rounding to bf16 (or the fp32 store), the layouts, the device-resident group offsets (no
host synchronisation, no caller padding), the exact zeros of empty groups and the error
budgets are those of the deterministic package.  The split tail's fp32 sums and tickets
are workspaces kept per device and CUDA stream at zero (every launch leaves them zero);
a CUDA graph capture takes zeroed buffers of its own.

The operator is local: on a context-parallel rank it runs on the rank's local rows, adds
no collective, and leaves the reduction of a parameter gradient across ranks to the
caller.

Functions
---------

The experimental functions are documented by their docstrings (they are not listed for
autodoc before graduation):

- ``grouped_gemm_fwd(x, w, offs, out=None, *, deterministic=True)``: ``Y = X @ W[e].T``
  per group, bf16 ``[sum_m, N]``;
- ``grouped_gemm_dgrad(g, w, offs, out=None, *, deterministic=True)``: ``dX = G @ W[e]``
  per group (``W`` read in place), bf16 ``[sum_m, K]``;
- ``grouped_gemm_wgrad(g, x, offs, out=None, out_dtype=None, num_groups=None, *,
  deterministic=True)``: ``dW[e] = G[e].T @ X[e]``, bf16 or fp32 ``[E, N, K]``;
- ``cake_grouped_mm(x, w, offs, deterministic=True)``: the autograd wrapper (``Y``, with
  ``dX`` and ``dW`` in the backward, ``dW`` in the dtype of ``w``).

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

The deterministic names and their compatibility aliases are unchanged and always
deterministic:

======================================================== ==============================================================
deterministic (unchanged)                                selectable mode
======================================================== ==============================================================
``flashinfer.grouped_mm.grouped_det_gemm_fwd``           ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_fwd``
``flashinfer.grouped_mm.grouped_det_gemm_dgrad``         ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_dgrad``
``flashinfer.grouped_mm.grouped_det_gemm_wgrad``         ``flashinfer.experimental.cake.grouped_mm.grouped_gemm_wgrad``
``flashinfer.grouped_mm.cake_grouped_det_mm``            ``flashinfer.experimental.cake.grouped_mm.cake_grouped_mm``
``cake_moe_grouped_det_gemm.prepare_grouped_det_gemm_*`` ``cake.cake_moe_grouped_gemm.prepare_grouped_gemm_*``
``cake_moe_grouped_det_gemm.GroupedDetGemmLaunch``       ``cake.cake_moe_grouped_gemm.GroupedGemmLaunch``
``cake_moe_grouped_det_gemm.CakeGroupedDetMm``           ``cake.cake_moe_grouped_gemm.CakeGroupedMm``
======================================================== ==============================================================

The legacy names without ``det`` -- ``flashinfer.experimental.cake_moe_grouped_gemm`` and
its ``grouped_gemm_*``, ``prepare_grouped_gemm_*``, ``GroupedGemmLaunch``,
``CakeGroupedMm`` and ``cake_grouped_mm``, and ``flashinfer.grouped_mm.grouped_gemm_fwd`` /
``grouped_gemm_dgrad`` / ``grouped_gemm_wgrad`` / ``cake_grouped_mm`` -- remain aliases of
the deterministic objects.

Features by architecture
------------------------

=============================================== =============== =============== ===============
feature                                         sm_100a         sm_103a         sm_107a
=============================================== =============== =============== ===============
``fwd`` / ``dgrad`` 256- and 512-column tiles   domain schedule domain schedule static schedule
``fwd`` evict-first / 3-3 head-tail twin        -- / --         -- / yes        yes / --
``dgrad`` evict-first / 3-3 head-tail twin      -- / --         -- / yes        yes / --
``dgrad`` probed queue split                    yes             yes             --
``wgrad`` 256-wide k tile, ordered tail reduce  yes             yes             yes
``wgrad`` 512-wide k tile, unordered split tail yes             yes             yes
``wgrad`` 512 wave-synchronised builds          yes             yes             --
``wgrad`` 512 half-release builds               planned walk    planned walk    yes
``wgrad`` 512 k256-tail twin                    --              --              yes
``wgrad`` output dtypes                         bf16, fp32      bf16, fp32      bf16, fp32
CUDA toolkit floor                              12.8            12.9            13.4
=============================================== =============== =============== ===============
