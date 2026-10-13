.. _apinorm:

flashinfer.norm
===============

Kernels for normalization layers.

.. currentmodule:: flashinfer.norm

.. autosummary::
    :toctree: ../generated

    rmsnorm
    rmsnorm_quant
    fused_add_rmsnorm
    fused_add_rmsnorm_quant
    fused_add_rmsnorm_fp8_block_quant
    gemma_rmsnorm
    gemma_fused_add_rmsnorm
    layernorm
    layernorm_quant
    fused_rmsnorm_silu
    fused_qk_rmsnorm_rope
    fused_dit_residual_layernorm_scale_shift
    fused_dit_gate_residual_layernorm_scale_shift
    fused_dit_gate_residual_layernorm_gamma_beta

Training RMSNorm (Cake)
-----------------------

Fused BF16 RMSNorm forward / backward for training on SM100, SM103 and SM107:
FP32 statistics and accumulation, one FP32 reciprocal RMS saved per row,
fused ``dx`` + deterministic FP32 ``dw`` backward, optional residual-add
fusion and an autograd wrapper.

.. currentmodule:: flashinfer.cake_rmsnorm_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm
    cake_rmsnorm_train_forward
    cake_rmsnorm_train_backward
    cake_rmsnorm_train_backward_workspace_bytes
    cake_rmsnorm_train_backward_workspace
    CakeRMSNormFunction
