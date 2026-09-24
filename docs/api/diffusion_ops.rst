.. _apidiffusionops:

flashinfer.diffusion_ops
========================

Fused operators for diffusion-transformer inference.

.. currentmodule:: flashinfer.diffusion_ops

.. autosummary::
    :toctree: ../generated

    minimax_h3_bf16_pre_attention
    minimax_h3_fp8_pre_attention
    minimax_h3_nvfp4_pre_attention
    quantize_minimax_h3_qkv_weight_fp8
    quantize_minimax_h3_qkv_weight_nvfp4
    fused_qk_rmsnorm_rope
    fused_dit_residual_layernorm_scale_shift
    fused_dit_gate_residual_layernorm_scale_shift
    fused_dit_gate_residual_layernorm_gamma_beta
