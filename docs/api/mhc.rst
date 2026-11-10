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

.. note::

   The mode is part of every name: the deterministic module
   ``flashinfer.cake_mhc_det_train`` (``det``) and the non-deterministic module
   ``flashinfer.cake_mhc_nondet_train`` (``nondet``) export the same verbs, the
   model's head is the module pair ``flashinfer.cake_mhc_head_det_train``
   (``det``) and ``flashinfer.cake_mhc_head_nondet_train`` (``nondet``), and no
   entry point takes a ``deterministic`` keyword. Each module imports its
   backend package under ``flashinfer.experimental.cake`` on first use. The
   former names below are removed, with no alias: importing them fails.
   FlashInfer 0.7.3's ``cake_mhc_train`` names were the deterministic package
   at the top level and the non-deterministic one under
   ``flashinfer.experimental.cake``.

   .. list-table::
      :header-rows: 1
      :widths: 50 50

      * - Former name
        - Name now
      * - ``flashinfer.cake_mhc_train.cake_mhc_train_<verb>`` and the top-level
          ``flashinfer.cake_mhc_train_<verb>`` (0.7.3)
        - ``flashinfer.cake_mhc_det_train.cake_mhc_det_train_<verb>`` (the
          entry points and ``*_workspace_bytes`` also at the top level)
      * - ``flashinfer.cake_mhc_train.CakeMHCTrain{Prepare,Finish}Function`` and
          the top-level ``flashinfer.CakeMHCTrain{Prepare,Finish}Function``
          (0.7.3),
          ``flashinfer.cake_mhc_det_train.CakeMHCDetTrain{Prepare,Finish}Function``
        - ``flashinfer.cake_mhc_det_train.CakeMHCDet{Prepare,Finish}Function``
      * - ``flashinfer.cake_mhc_train.is_cake_mhc_train_supported`` (0.7.3)
        - ``flashinfer.cake_mhc_det_train.is_cake_mhc_det_train_supported``
      * - ``flashinfer.cake_mhc_train.route_key`` (0.7.3),
          ``flashinfer.cake_mhc_det_train.route_key``
        - ``flashinfer.cake_mhc_det_train.cake_mhc_det_train_route_key``
      * - ``flashinfer.cake_mhc_train_backend`` (0.7.3),
          ``flashinfer.cake_mhc_det_train_backend``
        - ``flashinfer.experimental.cake.cake_mhc_det_train``
      * - ``flashinfer.cake_mhc_det_train.cake_mhc_det_train_head_<verb>``,
          ``flashinfer.cake_mhc_train.cake_mhc_train_head_<verb>`` and the
          top-level ``flashinfer.cake_mhc_train_head_<verb>`` (0.7.3), for
          ``<verb>`` ``forward`` and ``backward``
        - ``flashinfer.cake_mhc_head_det_train.cake_mhc_head_det_train_<verb>``
          (also at the top level)
      * - ``flashinfer.cake_mhc_det_train.cake_mhc_det_train_head``,
          ``flashinfer.cake_mhc_train.cake_mhc_train_head`` and the top-level
          ``flashinfer.cake_mhc_train_head`` (0.7.3)
        - ``flashinfer.cake_mhc_head_det_train.cake_mhc_head_det`` (also at the
          top level)
      * - ``flashinfer.cake_mhc_det_train.CakeMHCDetTrainHeadFunction``,
          ``flashinfer.cake_mhc_train.CakeMHCTrainHeadFunction`` and the
          top-level ``flashinfer.CakeMHCTrainHeadFunction`` (0.7.3)
        - ``flashinfer.cake_mhc_head_det_train.CakeMHCHeadDetFunction``
      * - ``flashinfer.cake_mhc_det_train.is_cake_mhc_det_train_head_supported``,
          ``flashinfer.cake_mhc_train.is_cake_mhc_train_head_supported`` (0.7.3)
        - ``flashinfer.cake_mhc_head_det_train.is_cake_mhc_head_det_train_supported``
      * - ``flashinfer.cake_mhc_train_head_backend`` (0.7.3),
          ``flashinfer.cake_mhc_det_train_head_backend``
        - ``flashinfer.experimental.cake.cake_mhc_head_det_train``
      * - ``flashinfer.experimental.cake.cake_mhc_train.cake_mhc_train_<verb>``
          with ``deterministic=False``
        - ``flashinfer.cake_mhc_nondet_train.cake_mhc_nondet_train_<verb>``
      * - ``flashinfer.experimental.cake.cake_mhc_train.is_cake_mhc_train_supported``
        - ``flashinfer.cake_mhc_nondet_train.is_cake_mhc_nondet_train_supported``
      * - ``flashinfer.experimental.cake.cake_mhc_train.CakeMHCTrain{Prepare,Finish}Function``
        - ``flashinfer.cake_mhc_nondet_train.CakeMHCNondet{Prepare,Finish}Function``
      * - ``flashinfer.experimental.cake.cake_mhc_train.route_key``
        - ``flashinfer.cake_mhc_nondet_train.cake_mhc_nondet_train_route_key``
      * - ``flashinfer.experimental.cake.cake_mhc_train_backend``
        - ``flashinfer.experimental.cake.cake_mhc_nondet_train``

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
    cake_mhc_det_train_forward_workspace_bytes
    cake_mhc_det_train_backward_workspace_bytes
    CakeMHCDetPrepareFunction
    CakeMHCDetFinishFunction

Training mHC head (Cake)
------------------------

The model's head, the mean of the last layer's four residual streams, is the
deterministic module ``flashinfer.cake_mhc_head_det_train`` with its own
kernels for hidden sizes 4096, 5120 and 7168 (4096 on SM107):
``cake_mhc_head_det_train_forward`` returns ``Y = X.mean(dim=1)`` bit for bit
as torch computes it for BF16 ``X`` ``[T, 4, H]``,
``cake_mhc_head_det_train_backward`` writes the gradient
``dX[:, i, :] = BF16(dY / 4)`` of each stream or adds it into a caller's
``dX`` (``accumulate=True``), and ``cake_mhc_head_det`` is the autograd form.
Every output (``Y``, the written ``dX`` and the accumulated ``dX``) repeats bit
for bit for identical inputs, also in the recompute of an activation
checkpoint. Its non-deterministic counterpart is the module
``flashinfer.cake_mhc_head_nondet_train`` (below).

.. currentmodule:: flashinfer.cake_mhc_head_det_train

.. autosummary::
    :toctree: ../generated

    cake_mhc_head_det_train_forward
    cake_mhc_head_det_train_backward
    cake_mhc_head_det
    CakeMHCHeadDetFunction
    is_cake_mhc_head_det_train_supported

Training mHC, non-deterministic (Cake)
--------------------------------------

``flashinfer.cake_mhc_nondet_train`` is the same training function in the
``nondet`` mode, computed by the package
``flashinfer.experimental.cake.cake_mhc_nondet_train`` (imported on first use)
for the same hidden sizes: 4096, 5120 and 7168 on SM100 / SM103, 4096 on SM107;
its entry points raise ``ValueError`` for another hidden size. Its backward has
no final reduction kernel: the column-slab pass stores the compensated ``dfn``
sums of every (column slab, token chunk) unit in its chunk's slots of the
backward workspace, the grid folds the slots into ``dfn`` once every unit has
stored its sums, and the same grid folds ``dscale`` / ``dbase``.

The mode is part of the names: the entry points are those of
``flashinfer.cake_mhc_det_train`` with ``cake_mhc_nondet_train_`` in place of
``cake_mhc_det_train_`` and the same arguments
(``cake_mhc_nondet_train_forward_prepare`` ... ``cake_mhc_nondet_train_finish``,
the workspace helpers, ``cake_mhc_nondet_train_reset_workspace`` and
``is_cake_mhc_nondet_train_supported``); the autograd nodes are
``CakeMHCNondetPrepareFunction`` and ``CakeMHCNondetFinishFunction``. No entry
point takes a ``deterministic`` keyword, and nothing falls back to the
deterministic package.

Numerical contract of the module: every output repeats bit for bit from run
to run on one device (each sum runs in an order fixed by the token count and
the device). ``dX``, ``dscale``, ``dbase``, the forward, the statistics
replay and ``backward_prepare`` are bitwise the deterministic values; the
column-slab pass folds ``dscale`` and ``dbase`` in the deterministic
reduction's order. Each element of ``dfn`` folds its per-chunk FP32 partials
(``x_p`` of ``P <= 32`` token chunks, bitwise the deterministic package's
partials) in ascending chunk order as a compensated FP32 sum, adds the caller's
``dfn`` last with ``accumulate=True`` and rounds once, so it is within one FP32
rounding of the exact sum plus ``O(P^2 u^2) (|start| + sum |x_p|)``
(``u = 2^-24``). The deterministic package sums the same partials in FP64 and
rounds once, so ``dfn`` equals the deterministic ``dfn`` except where the exact
sum lies within that last term of an FP32 rounding boundary.

Under ``torch.use_deterministic_algorithms(True)`` no entry point raises an
alert, and with ``warn_only=True`` none emits a warning.

Workspaces: the workspace helpers take the deterministic package's
``hidden=`` keyword (default 4096); it accepts the hidden sizes of the device's
architecture. The forward workspace has one layout in both packages. The
backward workspace has this module's own layout: a 256-byte ticket region and
512 bytes of column-slab arrival counts first (zero on first use), the
deterministic regions without the per-chunk FP32 ``dfn`` partials, then the
column slab's per-chunk FP32 ``dfn`` slots
(``min(32, ceil(T / 64)) x 24 x 4H x 4`` bytes at hidden size ``H``). A
deterministic backward workspace does not serve this backward: a buffer that
the deterministic package's backward used must be zeroed with
``cake_mhc_nondet_train_reset_workspace`` before this module uses it.

The default workspaces (``workspace=None``) are one buffer per device, hidden
size and power-of-two token-count bucket in each package, shared by every
stream and every graph replay on the device: calls that can run concurrently on
different streams pass caller-owned workspaces. The module creates no default
buffer while the current stream of the buffer's device is capturing a CUDA
graph (``ValueError``); warm up with the same token count and hidden size
before the capture. Every completed launch leaves the ticket words and the
arrival counts of its workspace at zero, so a replay needs no reset;
``cake_mhc_nondet_train_reset_workspace`` zeroes the buffer it is given.

The model's head is not part of this module: its ``nondet`` module is
``flashinfer.cake_mhc_head_nondet_train`` (below).

.. currentmodule:: flashinfer.cake_mhc_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_mhc_nondet_train_forward_prepare
    cake_mhc_nondet_train_forward_stats
    cake_mhc_nondet_train_forward_finish
    cake_mhc_nondet_train_backward_prepare
    cake_mhc_nondet_train_backward_finish
    cake_mhc_nondet_train_prepare
    cake_mhc_nondet_train_finish
    cake_mhc_nondet_train_forward_workspace_bytes
    cake_mhc_nondet_train_backward_workspace_bytes
    cake_mhc_nondet_train_forward_workspace
    cake_mhc_nondet_train_backward_workspace
    is_cake_mhc_nondet_train_supported
    CakeMHCNondetPrepareFunction
    CakeMHCNondetFinishFunction

Training mHC head, non-deterministic (Cake)
-------------------------------------------

``flashinfer.cake_mhc_head_nondet_train`` is the model's head in the ``nondet``
mode: the same function, tensor contracts, hidden sizes (4096, 5120 and 7168 on
SM100 / SM103, 4096 on SM107) and autograd wiring as
``flashinfer.cake_mhc_head_det_train``, computed by its own backend package
``flashinfer.experimental.cake.cake_mhc_head_nondet_train`` (imported on first
use). ``cake_mhc_head_nondet_train_forward`` returns the stream mean,
``cake_mhc_head_nondet_train_backward`` writes the gradient of each stream or
adds it into a caller's ``dX`` (``accumulate=True``), and
``cake_mhc_head_nondet`` is the autograd form. The top level of ``flashinfer``
re-exports none of these names. No entry point takes a ``deterministic``
keyword, and nothing falls back to the deterministic module.

The head is elementwise per token: it has no cross-token reduction, no atomic
operation and no order-varying step. Its kernels are the deterministic head's,
exported under the library names ``cake_mhc_train_head_*``, so it computes the
same arithmetic in the same order: bit for bit ``X.mean(dim=1)`` and its
gradient. The expected performance is the deterministic head's; this module
offers no speedup. The ``nondet`` contract does not promise run-to-run
identical bits, but this implementation is fixed-order: under
``torch.use_deterministic_algorithms(True)`` its entry points raise no alert,
and with ``warn_only=True`` they emit no warning.

.. currentmodule:: flashinfer.cake_mhc_head_nondet_train

.. autosummary::
    :toctree: ../generated

    cake_mhc_head_nondet_train_forward
    cake_mhc_head_nondet_train_backward
    cake_mhc_head_nondet
    CakeMHCHeadNondetFunction
    is_cake_mhc_head_nondet_train_supported
