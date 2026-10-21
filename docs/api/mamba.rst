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
convolved on their own, FP32 taps and arithmetic with one BF16 rounding, a
fused ``dx`` + deterministic FP32 ``dweight`` backward in one cooperative
launch, and an autograd wrapper that saves only the forward's inputs.

.. currentmodule:: flashinfer.cake_causal_conv1d_train

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d
    cake_causal_conv1d_train_forward
    cake_causal_conv1d_train_backward
    cake_causal_conv1d_train_backward_workspace_bytes
    cake_causal_conv1d_train_backward_workspace
    CakeCausalConv1dFunction
