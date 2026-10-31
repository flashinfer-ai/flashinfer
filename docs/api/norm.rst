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

Fixed-Width Fused Norms
-----------------------

Fused gated-residual and per-head gated RMSNorm kernels for SM100, SM103 and
SM107. They support only the widths in their docstrings (hidden size 8192, head
dim 64); check the matching ``supports_*`` helper and fall back otherwise.

.. autosummary::
    :toctree: ../generated

    gated_residual_rmsnorm
    supports_gated_residual_rmsnorm
    gated_residual_next_rmsnorm
    supports_gated_residual_next_rmsnorm
    rmsnorm_sigmoid_gate
    supports_rmsnorm_sigmoid_gate

Training RMSNorm (Cake)
-----------------------

Fused BF16 RMSNorm forward / backward for training on SM100, SM103 and SM107:
FP32 statistics and accumulation, one FP32 reciprocal RMS saved per row,
fused ``dx`` + deterministic FP32 ``dw`` backward, optional residual-add
fusion and an autograd wrapper.

.. currentmodule:: flashinfer.cake_rmsnorm_det_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_det
    cake_rmsnorm_det_train_forward
    cake_rmsnorm_det_train_backward
    cake_rmsnorm_det_train_backward_workspace_bytes
    cake_rmsnorm_det_train_backward_workspace
    program_windows
    weight_dtypes
    CakeRMSNormDetFunction

Training SiLU-gated RMSNorm (Cake)
----------------------------------

``RMSNorm(x) * weight * SiLU(z)`` with a separate gate input ``z``, forward /
backward for training on SM100, SM103 and SM107: rows of 128 (``[T, H, 128]``
or ``[T * H, 128]``) normalized per head with FP32 statistics and gate, BF16
inputs, outputs and gradients, a weight shared by every token and head whose
gradient is summed in a fixed order (repeated calls return the same bits), and
an autograd wrapper that saves only its inputs.

.. currentmodule:: flashinfer.cake_rmsnorm_silu_gate_det_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_silu_gate_det
    cake_rmsnorm_silu_gate_det_train_forward
    cake_rmsnorm_silu_gate_det_train_backward
    cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
    cake_rmsnorm_silu_gate_det_train_backward_workspace
    CakeRMSNormSiLUGateDetFunction

Training SiLU-gated RMSNorm, opt-in non-deterministic weight gradient (Cake)
---------------------------------------------------------------------------

The names the training SiLU-gated RMSNorm had before the deterministic package
took its ``det`` names. Every entry point takes the keyword ``deterministic``
(default ``True``):

* ``deterministic=True`` runs the deterministic package above and returns its
  bits.
* ``deterministic=False`` runs the same forward kernel and a one-kernel
  backward: ``dx`` and ``dz`` are the deterministic backward's bits, and the
  CTAs add their FP64 weight-gradient column sums into FP64 accumulators of the
  workspace in the order they finish, so the weight gradient can differ by one
  rounding of its dtype between calls. The workspace
  (``cake_rmsnorm_silu_gate_train_backward_workspace(..., deterministic=False)``,
  a few KiB whatever the token count) is zero after every call: zero it once
  and reuse it, one per concurrently running stream.
* Under ``torch.use_deterministic_algorithms(True)``, ``deterministic=False``
  raises ``RuntimeError``; with ``warn_only=True`` it warns and runs the
  deterministic kernels.

.. currentmodule:: flashinfer.cake_rmsnorm_silu_gate_train

.. autosummary::
    :toctree: ../generated

    cake_rmsnorm_silu_gate
    cake_rmsnorm_silu_gate_train_forward
    cake_rmsnorm_silu_gate_train_backward
    cake_rmsnorm_silu_gate_train_backward_workspace_bytes
    cake_rmsnorm_silu_gate_train_backward_workspace
    CakeRMSNormSiLUGateFunction
    is_cake_rmsnorm_silu_gate_train_supported
