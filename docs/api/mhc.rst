.. _apimhc:

flashinfer.mhc
==============

Multi-head residual combination (mHC) kernels used by Agentic-style residual
mixing with a fixed ``HC=4`` sub-head layout.

.. currentmodule:: flashinfer.mhc

.. autosummary::
    :toctree: ../generated

    mhc_post
    mhc_pre_big_fuse
    mhc_pre_big_fuse_with_prenorm

Training mHC (Cake)
-------------------

Four-stream mHC training forward / backward for hidden sizes 4096, 5120 and
7168 on SM100 and SM103 (4096 on SM107); each call takes the hidden size ``H``
from ``X`` ``[T, 4, H]``: the forward mapping, coefficients and collapse
(``forward_prepare``), the residual mixing (``forward_finish``), the branch
gradient and gradient statistics (``backward_prepare``), the single-cast input
gradient with fixed-order FP32 parameter gradients (``backward_finish``),
checkpoint replay (``forward_stats``) and autograd wrappers around the
external branch. The caller holds the workspaces:
``cake_mhc_det_train_forward_workspace_bytes`` / ``cake_mhc_det_train_forward_workspace``
and ``cake_mhc_det_train_backward_workspace_bytes`` /
``cake_mhc_det_train_backward_workspace`` size and allocate a zero-initialized
buffer for one hidden size (``hidden=``, default 4096) that serves the calls of
that hidden size for every token count up to ``T``,
``cake_mhc_det_train_reset_workspace`` returns such a caller-owned buffer to its
first-use state after an interrupted launch or capture, and
``is_cake_mhc_det_train_supported`` reports whether a device (optionally at a given
hidden size) has every exported route.

CUDA graphs: one training iteration -- the direct entry points, or
``cake_mhc_det_train_prepare`` / ``cake_mhc_det_train_finish`` around the external
branch followed by the loss backward, also inside a non-reentrant
``torch.utils.checkpoint`` -- can be captured whole with ``torch.cuda.graph``,
and its replays are bitwise equal to the eager iteration;
``torch.cuda.make_graphed_callables`` of such a layer works as well. Warm up
before the capture: run the iteration a few times eagerly on a side stream, so
that every kernel is built and loaded and the per-device default workspaces
exist before the capture starts. The launches go to the current stream, so they
follow the capture stream. Use the same workspaces (caller-owned, or the
defaults) for the warm-up, the capture and the replays, at the token count the
graph was captured with. Every launch leaves the ticket words of its workspace
at zero, so a replay needs no host-side reset.
``cake_mhc_det_train_reset_workspace`` restores a caller-owned workspace after an
interrupted launch or capture; callers that need this recovery pass their own
workspaces.

Side streams: ``forward_prepare`` and ``backward_finish`` keep running while a
kernel on another stream holds some SMs, or holds one CTA on every SM that was
launched with a shared-memory carveout preference of at least 50 percent
(``cudaFuncAttributePreferredSharedMemoryCarveout``). With the driver's default
carveout such a side CTA leaves room for at most 1 KB of dynamic shared memory
next to it on its SM, so every launch that needs more waits until the side
kernel's first CTA exits; raise the side kernel's carveout preference to avoid
this. Under a side kernel that holds every SM, ``backward_finish`` still waits
at some token counts up to 4096 on SM100 at hidden 4096, at token counts
1-896, 1921-2112, 2689-2816 and 3585-3648 on SM100 at hidden 5120, at token
counts 1-640 and 961-1024 on SM100 at hidden 7168, at token counts up to 1535
(hidden 5120) and 1151 (hidden 7168) on SM103, and on SM107, which keeps the
static schedule.

.. currentmodule:: flashinfer.cake_mhc_det_train

.. autosummary::
    :toctree: ../generated

    cake_mhc_det_train_forward_prepare
    cake_mhc_det_train_forward_stats
    cake_mhc_det_train_forward_finish
    cake_mhc_det_train_backward_prepare
    cake_mhc_det_train_backward_finish
    cake_mhc_det_train_prepare
    cake_mhc_det_train_finish
    CakeMHCDetTrainPrepareFunction
    CakeMHCDetTrainFinishFunction

The model's head, the mean of the last layer's four residual streams, has its
own kernels for hidden sizes 4096, 5120 and 7168 (4096 on SM107):
``cake_mhc_det_train_head_forward`` returns ``Y = X.mean(dim=1)`` bit for bit as
torch computes it for BF16 ``X`` ``[T, 4, H]``,
``cake_mhc_det_train_head_backward`` writes the gradient
``dX[:, i, :] = BF16(dY / 4)`` of each stream or adds it into a caller's
``dX`` (``accumulate=True``), and ``cake_mhc_det_train_head`` is the autograd form.

.. autosummary::
    :toctree: ../generated

    cake_mhc_det_train_head_forward
    cake_mhc_det_train_head_backward
    cake_mhc_det_train_head
    CakeMHCDetTrainHeadFunction
    is_cake_mhc_det_train_head_supported

Training mHC with a selectable numerical mode (Cake, experimental)
------------------------------------------------------------------

``flashinfer.experimental.cake.cake_mhc_train`` is the same training function
with a ``deterministic: bool = True`` keyword on every entry point except
``cake_mhc_train_finish`` (it follows the package of its handle),
``cake_mhc_train_reset_workspace`` and ``route_key``.

* ``deterministic=True`` (default) runs ``flashinfer.cake_mhc_det_train``
  unchanged, for hidden sizes 4096, 5120 and 7168 (4096 on SM107).
* ``deterministic=False`` runs the non-deterministic package
  ``flashinfer.experimental.cake.cake_mhc_train_backend``, for the same hidden
  sizes: 4096, 5120 and 7168 on SM100 / SM103, 4096 on SM107; its entry points
  raise ``ValueError`` for another hidden size. Its backward has no final
  reduction kernel: the column-slab pass adds the ``dfn`` sum of every (column
  slab, token chunk) unit into ``dfn`` with FP32 reductions and folds
  ``dscale`` / ``dbase`` in the same grid.

Numerical contract of ``deterministic=False``: only the summation order of
``dfn`` varies from run to run. Each element lies within ``P`` FP32 roundings
of the fixed-order sum (``P <= 32`` token chunks), and the FP32 reduction
flushes subnormal addends and results to zero. ``dX``, ``dscale``, ``dbase``,
the forward, the statistics replay and ``backward_prepare`` are bitwise the
deterministic values.

Interaction with ``torch.use_deterministic_algorithms(True)``:

* The entry points of ``flashinfer.experimental.cake.cake_mhc_train`` run the
  deterministic package for ``deterministic=False`` and warn once per process
  (``UserWarning``).
* The non-deterministic package never falls back. Its ``backward_finish`` and
  the backward of its prepare node raise PyTorch's alert for an operation
  without a deterministic implementation (``RuntimeError``; with
  ``warn_only=True`` a ``UserWarning``) right before they launch the ``dfn``
  reduction: after their operands pass, and only for ``T > 0``. The prepare
  node alerts also when the mode was enabled between its forward and its
  backward. Its forward,
  statistics, ``backward_prepare`` and workspace entry points do not alert.
* Every compute and workspace entry point of the non-deterministic package
  raises ``ValueError`` unless ``deterministic=False``.

Workspaces: the workspace helpers take the deterministic package's
``hidden=`` keyword (default 4096) in both modes; it accepts the hidden sizes
of the device's architecture. The forward workspace has one layout in both
modes. The ``deterministic=False`` backward workspace is the deterministic one
at the same offsets without the per-chunk FP32 ``dfn`` partials
(``P x 24 x 4H x 4`` bytes at hidden size ``H``), so
``cake_mhc_train_backward_workspace(T, device, hidden=H, deterministic=True)``
serves both modes and ``deterministic=False`` sizes the smaller one. When the
deterministic mode turns a ``deterministic=False`` call with a caller-owned
backward workspace that is too small for the deterministic layout into the
deterministic backward, the call runs on a private zeroed companion buffer of
that workspace, never on the shared default buffer. Inside a CUDA graph
capture a missing companion raises ``ValueError``.

The default workspaces (``workspace=None``) are one buffer per device, hidden
size and power-of-two token-count bucket in each package, shared by every
stream and every graph replay on the device: calls that can run concurrently on
different streams pass caller-owned workspaces. ``deterministic=False`` creates
no default buffer while the current stream of the buffer's device is capturing
a CUDA graph (``ValueError``); warm up with the same ``deterministic`` value,
token count and hidden size before the capture. A companion is allocated
zeroed outside any capture and every completed launch leaves its
ticket words at zero, so it needs no reset; ``cake_mhc_train_reset_workspace``
zeroes the buffer it is given.

The head (``cake_mhc_det_train_head*``) has no cross-token reduction and no
counterpart in this module; use the deterministic names above.

Names: ``flashinfer.cake_mhc_det_train.cake_mhc_det_train_<name>`` ->
``flashinfer.experimental.cake.cake_mhc_train.cake_mhc_train_<name>``,
``is_cake_mhc_det_train_supported`` -> ``is_cake_mhc_train_supported`` and
``CakeMHCDetTrain<Node>Function`` -> ``CakeMHCTrain<Node>Function`` (the
non-deterministic package's nodes). The legacy module
``flashinfer.cake_mhc_train`` keeps the same ``cake_mhc_train_*`` names as
aliases of the deterministic objects, without a ``deterministic`` keyword.
