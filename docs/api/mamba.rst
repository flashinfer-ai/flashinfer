.. _apimamba:

flashinfer.mamba
================

Mamba / Mamba-2 state-space-model kernels. These wrap the selective scan and
state-update primitives used in SSM blocks.

.. currentmodule:: flashinfer.mamba

.. autosummary::
    :toctree: ../generated

    selective_state_update
    checkpointing_ssu
    ssd_combined_fwd
    replayssm_materialize

Training causal conv1d (Cake)
-----------------------------

Packed width-4 depthwise causal convolution + SiLU forward / backward for
training on SM100, SM103 and SM107: documents delimited by ``cu_seqlens`` and
convolved on their own, FP32 or BF16 taps and an optional FP32 or BF16
per-channel bias (added after the taps) with FP32 arithmetic and one BF16
rounding, a fused ``dx`` + deterministic ``dweight`` (+ ``dbias``) backward in
two ordinary launches (a main program and a programmatically dependent
fixed-order reduce; no cooperative grid, so kernels resident on other streams
never stall it) whose SiLU' is evaluated at the BF16-rounded
preactivation (default) or, with ``silu_grad_preactivation="fp32"``, at the
unrounded FP32 one for a weight gradient within about 2e-7 rel-L2 of exact, and
an autograd wrapper that saves only the forward's inputs.  The forward's
co-resident CTAs take each channel tile's tokens in chunks from int32 counters
in a forward workspace (``cake_causal_conv1d_det_train_forward_workspace`` /
``cake_causal_conv1d_det_train_forward_workspace_bytes``; zero before its first
launch and left zero by every launch, so one buffer serves every later call of
one stream); when the caller passes none the package keeps one per device and
stream.

.. currentmodule:: flashinfer.cake_causal_conv1d_det_train

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d_det
    cake_causal_conv1d_det_train_forward
    cake_causal_conv1d_det_train_backward
    cake_causal_conv1d_det_train_backward_workspace_bytes
    cake_causal_conv1d_det_train_backward_workspace
    CakeCausalConv1dDetFunction

Context-parallel training causal conv1d (Cake)
----------------------------------------------

The context-parallel (KCP) route of the same convolution: every rank convolves
its fragments of the packed batch, and the first tokens of a fragment that
continues a document read the history rows of the preceding fragment through
the halo collective of the KDA step's context.  ``flashinfer.cake_causal_conv1d_det_train``
keeps the previous names ``cake_causal_conv1d_kcp``, ``cake_causal_conv1d_kcp_multi``
and ``KcpRecomputeHandle`` as aliases of the same objects.

.. currentmodule:: flashinfer.cake_causal_conv1d_det_train_kcp

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d_det_kcp
    cake_causal_conv1d_det_multi_kcp
    RecomputeHandleDetKcp

Training causal conv1d with a selectable numerical mode (Cake, experimental)
----------------------------------------------------------------------------

The same convolution, single-rank and context-parallel, with a
``deterministic: bool = True`` keyword on every entry point.  These entry
points are experimental (``@flashinfer_experimental_api``: the first call warns
once with ``ExperimentalWarning``) and are not listed in the API summaries
above.

.. list-table:: Entry points by numerical mode
   :header-rows: 1

   * -
     - ``deterministic=True`` (default)
     - ``deterministic=False``
   * - single-rank
     - ``flashinfer.cake_causal_conv1d_det_train`` (unchanged)
     - ``flashinfer.experimental.cake.cake_causal_conv1d_train_backend``
   * - context-parallel (KCP)
     - ``flashinfer.cake_causal_conv1d_det_train_kcp`` (unchanged)
     - ``flashinfer.experimental.cake.cake_causal_conv1d_train_backend.kcp``

Dispatching entry points (they select a row of the table by ``deterministic``):

* ``flashinfer.experimental.cake.cake_causal_conv1d_train``:
  ``cake_causal_conv1d``, ``cake_causal_conv1d_train_forward``,
  ``cake_causal_conv1d_train_backward``, the forward / backward workspace
  helpers ``cake_causal_conv1d_train_{forward,backward}_workspace{,_bytes}``
  and ``is_cake_causal_conv1d_train_supported``.
* ``flashinfer.cake_causal_conv1d_train_kcp``: ``cake_causal_conv1d_kcp``,
  ``cake_causal_conv1d_multi_kcp`` and ``RecomputeHandleKcp`` (any object with
  an ``entries`` dict serves either mode).

Each name maps to a deterministic name: ``cake_causal_conv1d_det`` ->
``cake_causal_conv1d``, ``cake_causal_conv1d_det_train_{forward,backward}``
-> ``cake_causal_conv1d_train_{forward,backward}``,
``cake_causal_conv1d_det_kcp`` -> ``cake_causal_conv1d_kcp``,
``cake_causal_conv1d_det_multi_kcp`` -> ``cake_causal_conv1d_multi_kcp``,
``RecomputeHandleDetKcp`` -> ``RecomputeHandleKcp``.  The deterministic names,
the legacy module ``flashinfer.cake_causal_conv1d_train`` and the legacy
top-level names ``flashinfer.cake_causal_conv1d`` /
``flashinfer.cake_causal_conv1d_train_forward`` / ``_backward`` stay bound to
the deterministic objects.

Numerical contract of ``deterministic=False``: the forward runs copies of the
deterministic forward programs, so ``y`` is bitwise the same.  The backward is
one ordinary launch per call (no reduce program, no cooperative grid): ``dx``
is bitwise the deterministic backward's, and the CTAs add their ``dweight`` /
``dbias`` rows into zeroed accumulators with unordered global reductions (FP32,
or FP64 under ``silu_grad_preactivation="fp32"``), rounded once into the
requested dtype.  ``dweight`` / ``dbias`` stay within the deterministic error
budget but are not bitwise reproducible from run to run.  The KCP backward
returns the rank-local FP32 ``dweight`` partial with the same property.

Interaction with ``torch.use_deterministic_algorithms(True)``:

* The dispatching entry points run the deterministic package for
  ``deterministic=False`` and warn once per entry point (``UserWarning``).
  They do not pass a non-deterministic backward workspace to the
  deterministic package.
* The backend entry points never fall back.  The entry points that launch the
  unordered reduction (the single-rank backward, the KCP backward, the KCP
  multi backward, and so the autograd backward) raise PyTorch's alert for an
  operation without a deterministic implementation: a ``RuntimeError``, or a
  ``UserWarning`` with ``warn_only=True``.  The forward and the checkpoint recompute run the
  deterministic forward programs and do not alert.
* Every backend entry point raises ``ValueError`` unless
  ``deterministic=False``: the backend has no deterministic programs.

The non-deterministic backward workspace holds the accumulators and one
ticket counter per channel tile.  Allocate it with
``cake_causal_conv1d_train_backward_workspace(S, C, device, deterministic=False)``
(a zeroed ``uint8`` buffer); every launch leaves it zero, and its size depends
on ``C`` and the route, not on ``S``.  Do not share one workspace between two
backward launches that can run at the same time, for example on two streams.
Without a caller workspace the package keeps one zeroed buffer per device and
stream, and allocates a fresh zeroed buffer inside a CUDA Graph capture.
