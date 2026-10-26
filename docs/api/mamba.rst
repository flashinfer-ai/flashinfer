.. _apimamba:

flashinfer.mamba
================

Mamba / Mamba-2 state-space-model kernels. These wrap the selective scan and
state-update primitives used in SSM blocks.

``SSDCombined`` and ``ssd_combined_fwd`` default to ``backend="auto"``, which
runs the Mamba-2 chunked scan on cuDNN's fused SM100 linear-attention engine
(:func:`flashinfer.cudnn.cudnn_ssd_combined_fwd`) wherever it serves the call
and otherwise keeps the previous defaults (the CuTe kernel for the class,
Cake for the function). cuDNN needs an SM100 / SM103 device and
cudnn-frontend 1.32+ with the ``cutedsl`` extra, serves ``headdim == 64`` and
``dstate == 128`` with bfloat16 or float16 activations and a bfloat16,
float32 or float16 state, takes the dense, ``cu_seqlens`` and ``seq_idx`` forms, and
has no selective ``checkpoint_*`` capture (a call asking for it resolves to
Cake). On GB200 (CUDA-graph replay, 2026-10-13) it measured ahead of Cake by
1.1 to 12.8x on ten of eleven cells across 1k to 128k tokens and 8 to 64 heads
(Cake leads by 10 percent on one 2k-token sequence at 8 heads) and of the
CuTe kernel by 2.6 to 33x on every cell.
``backend="cudnn"``, ``"cake"`` and ``"cute"`` select one explicitly.

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
an autograd wrapper that saves only the forward's inputs.

.. currentmodule:: flashinfer.cake_causal_conv1d_train

.. autosummary::
    :toctree: ../generated

    cake_causal_conv1d
    cake_causal_conv1d_train_forward
    cake_causal_conv1d_train_backward
    cake_causal_conv1d_train_backward_workspace_bytes
    cake_causal_conv1d_train_backward_workspace
    CakeCausalConv1dFunction
