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
in a forward workspace (``cake_causal_conv1d_train_forward_workspace`` /
``cake_causal_conv1d_train_forward_workspace_bytes``; zero before its first
launch and left zero by every launch, so one buffer serves every later call of
one stream); when the caller passes none the package keeps one per device and
stream.

.. currentmodule:: flashinfer.cake_causal_conv1d_train

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d
    cake_causal_conv1d_kcp
    cake_causal_conv1d_train_forward
    cake_causal_conv1d_train_backward
    cake_causal_conv1d_train_backward_workspace_bytes
    cake_causal_conv1d_train_backward_workspace
    CakeCausalConv1dFunction
