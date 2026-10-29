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

Four-stream mHC training forward / backward for hidden size 4096 on SM100,
SM103 and SM107: the forward mapping, coefficients and collapse
(``forward_prepare``), the residual mixing (``forward_finish``), the branch
gradient and gradient statistics (``backward_prepare``), the single-cast input
gradient with fixed-order FP32 parameter gradients (``backward_finish``),
checkpoint replay (``forward_stats``) and autograd wrappers around the
external branch. The caller holds the workspaces:
``cake_mhc_train_forward_workspace_bytes`` / ``cake_mhc_train_forward_workspace``
and ``cake_mhc_train_backward_workspace_bytes`` /
``cake_mhc_train_backward_workspace`` size and allocate a zero-initialized
buffer that serves every token count up to ``T``,
``cake_mhc_train_reset_workspace`` returns such a caller-owned buffer to its
first-use state after an interrupted launch or capture, and
``is_cake_mhc_train_supported`` reports whether a device has every exported
route.

CUDA graphs: one training iteration -- the direct entry points, or
``cake_mhc_train_prepare`` / ``cake_mhc_train_finish`` around the external
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
``cake_mhc_train_reset_workspace`` restores a caller-owned workspace after an
interrupted launch or capture; callers that need this recovery pass their own
workspaces.

.. currentmodule:: flashinfer.cake_mhc_train

.. autosummary::
    :toctree: ../generated

    cake_mhc_train_forward_prepare
    cake_mhc_train_forward_stats
    cake_mhc_train_forward_finish
    cake_mhc_train_backward_prepare
    cake_mhc_train_backward_finish
    cake_mhc_train_prepare
    cake_mhc_train_finish
    CakeMHCTrainPrepareFunction
    CakeMHCTrainFinishFunction
