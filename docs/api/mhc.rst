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
``cake_mhc_train_reset_workspace`` returns one to its first-use state after an
interrupted launch, and ``is_cake_mhc_train_supported`` reports whether a
device has every exported route.

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
