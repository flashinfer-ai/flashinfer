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

   Every function and class exported by the deterministic package
   ``flashinfer.cake_mhc_det_train`` and by its backends
   ``flashinfer.cake_mhc_det_train_backend`` and
   ``flashinfer.cake_mhc_det_train_head_backend`` has ``det`` in its name (the
   generic constants and route helpers ``ARCHES``,
   ``SUPPORTED_COMPUTE_CAPABILITIES``, ``TENSOR_ALIGNMENT_BYTES``,
   ``WORKSPACE_ALIGNMENT_BYTES``, ``route_key`` and ``route_modules`` are
   qualified by the module path). Its entry points take no ``deterministic``
   keyword. FlashInfer 0.7.3 named this package ``flashinfer.cake_mhc_train``,
   with ``flashinfer.cake_mhc_train_*`` and ``CakeMHCTrain*`` at the top level.
   Those names are removed, with no alias: replace ``cake_mhc_train`` with
   ``cake_mhc_det_train`` and ``CakeMHCTrain`` with ``CakeMHCDetTrain``. The
   names without ``det`` now belong only to the non-deterministic module
   ``flashinfer.experimental.cake.cake_mhc_train`` described below.

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

Training mHC, non-deterministic (Cake, experimental)
----------------------------------------------------

``flashinfer.experimental.cake.cake_mhc_train`` is the same training function,
computed by the non-deterministic package
``flashinfer.experimental.cake.cake_mhc_train_backend`` for the same hidden
sizes: 4096, 5120 and 7168 on SM100 / SM103, 4096 on SM107; its entry points
raise ``ValueError`` for another hidden size. Its backward has no final
reduction kernel: the column-slab pass adds the ``dfn`` sum of every (column
slab, token chunk) unit into ``dfn`` with FP32 reductions and folds
``dscale`` / ``dbase`` in the same grid.

Every function of the module except ``cake_mhc_train_reset_workspace`` and
``route_key`` takes a keyword-only ``deterministic: bool = True``, and a call
must pass ``deterministic=False`` explicitly. Any other value, the default
``True`` included, raises ``ValueError`` naming the deterministic entry point
to call instead (``flashinfer.cake_mhc_det_train.cake_mhc_det_train_<name>``,
or ``is_cake_mhc_det_train_supported``). Nothing falls back to the
deterministic package: a call without the keyword, which earlier ran the
deterministic package, now raises this error.

Numerical contract of the non-deterministic module: only the summation order
of ``dfn`` varies from run to run. Each element lies within ``P`` FP32
roundings of the fixed-order sum of its per-chunk partials (``P <= 32`` token
chunks), and the FP32 reduction flushes subnormal addends and results to zero;
the chunk sizes follow the host rule, so the low-order bits of ``dfn`` can also
differ from the deterministic ``dfn``. ``dX``, the forward, the statistics
replay and ``backward_prepare`` are bitwise the deterministic values.
``dscale`` and ``dbase`` are bitwise the deterministic package's per-CTA
reduction; in the token-count bands where the deterministic package sums them
over 8-token group rows (hidden 4096 on SM100 / SM103, given in the module
docstring of ``flashinfer.experimental.cake.cake_mhc_train_backend.cake_jit``),
their low-order bits can differ.

Interaction with ``torch.use_deterministic_algorithms(True)``:

* A ``deterministic`` value other than ``False`` raises ``ValueError`` first,
  whether the mode is enabled or not.
* ``cake_mhc_train_backward_finish`` and the backward of a
  ``cake_mhc_train_prepare`` node raise PyTorch's alert for an operation
  without a deterministic implementation right before they launch the ``dfn``
  reduction: after their operands pass, and only for ``T > 0``. The alert is a
  ``RuntimeError``; with ``warn_only=True`` it is a ``UserWarning``, and the
  non-deterministic kernels then run. The prepare node alerts also when the
  mode was enabled between its forward and its backward.
* ``forward_prepare``, ``forward_stats``, ``forward_finish``,
  ``backward_prepare``, the workspace helpers and
  ``is_cake_mhc_train_supported`` give bitwise the same results from run to run
  and raise no such alert or warning.
* Nothing falls back to the deterministic package. For deterministic results
  call ``flashinfer.cake_mhc_det_train``: its entry points give the same
  results under the mode and raise no alert.

Workspaces: the workspace helpers take the deterministic package's
``hidden=`` keyword (default 4096); it accepts the hidden sizes of the device's
architecture. The forward workspace has one layout in both packages. The
non-deterministic backward workspace is the deterministic one at the same
offsets without the per-chunk FP32 ``dfn`` partials (``P x 24 x 4H x 4`` bytes
at hidden size ``H``). The non-deterministic backward accepts any valid
workspace of at least its size, so
``flashinfer.cake_mhc_det_train.cake_mhc_det_train_backward_workspace(T, device, hidden=H)``
also serves it.

The default workspaces (``workspace=None``) are one buffer per device, hidden
size and power-of-two token-count bucket in each package, shared by every
stream and every graph replay on the device: calls that can run concurrently on
different streams pass caller-owned workspaces. The non-deterministic module
creates no default buffer while the current stream of the buffer's device is
capturing a CUDA graph (``ValueError``); warm up with the same token count and
hidden size before the capture. Every completed launch leaves the ticket words
of its workspace at zero, so a replay needs no reset;
``cake_mhc_train_reset_workspace`` (no ``deterministic`` keyword) zeroes the
buffer it is given.

The head (``cake_mhc_det_train_head*``) has no cross-token reduction and no
counterpart in this module; use the deterministic names above.

Names: ``flashinfer.cake_mhc_det_train.cake_mhc_det_train_<name>`` ->
``flashinfer.experimental.cake.cake_mhc_train.cake_mhc_train_<name>``,
``is_cake_mhc_det_train_supported`` -> ``is_cake_mhc_train_supported`` and
``CakeMHCDetTrain<Node>Function`` -> ``CakeMHCTrain<Node>Function`` (the
non-deterministic package's nodes).
