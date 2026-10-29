"""Mega-path BF16 weight preprocessing for the SM90 pull-style CuTeDSL kernel.

No quantization: ``preprocess_mega_weights`` only re-lays the canonical bf16
``MoEWeightPack`` out the way the kernel's TMA descriptors expect:

* FC1 gate/up rows interleaved in blocks of ``Fp8GateUpInterleave = 8`` (the
  SM90 kernels' PostSwigluHalf fold -- shared with the FP8 kernel and the
  drop's ``mega_reference_bf16``), then transposed to K-major
  ``(E, hidden, 2I)`` with hidden stride-1;
* FC2 transposed to K-major ``(E, I, hidden)`` with intermediate stride-1
  (a view of the contiguous canonical ``w2`` -- no copy).

The kernel-ready result is one bf16 tensor per leg (no scale planes), i.e.
``TransformedMegaWeights = (fc1_weight, fc2_weight)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ......weights import MoEWeightPack, PrequantizedMoEWeights

if TYPE_CHECKING:
    import torch

# Kernel-ready weights: (fc1_weight, fc2_weight), both K-major bf16.
TransformedMegaWeights = Tuple["torch.Tensor", "torch.Tensor"]

_NAME = "sm90_bf16_bf16_bf16_pull_cutedsl"


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """bf16 canonical weights → kernel-ready SM90 BF16 mega layout.

    Accepts canonical bf16 weights only: this backend never quantizes, so a
    :class:`PrequantizedMoEWeights` pack (or one carrying scale planes) is
    rejected.
    """
    import torch

    from ......core.validation.common import MoEEpConfigError

    # The gate/up 8-row interleave is the same fold the FP8 kernel of this
    # tree uses (both epilogues share Fp8GateUpInterleave); reuse its helper
    # rather than carrying a second copy.
    from ..fp8_fp8_bf16_pull_cutedsl.weights import _interleave_gate_up_8

    if isinstance(weights, PrequantizedMoEWeights):
        raise MoEEpConfigError(
            f"{_NAME} is a native BF16 backend and does not take "
            "PrequantizedMoEWeights; pass canonical bf16 w13/w2, or supply "
            "kernel-ready transformed weights with "
            "MegaConfig.preprocess_weights=False"
        )
    if not isinstance(weights, MoEWeightPack):
        raise MoEEpConfigError(
            f"{_NAME} weights must be MoEWeightPack, got {type(weights).__name__}"
        )
    if (
        getattr(weights, "w13_scale", None) is not None
        or getattr(weights, "w2_scale", None) is not None
    ):
        raise MoEEpConfigError(
            f"{_NAME} is a native BF16 backend and accepts canonical bf16 "
            "weights only (no scale planes)"
        )

    fc1_out = 2 * intermediate_size
    num_experts = weights.w13.shape[0]
    logical_w13_shape = (num_experts, fc1_out, hidden_size)
    logical_w2_shape = (num_experts, hidden_size, intermediate_size)
    for name, tensor, shape in (
        ("w13", weights.w13, logical_w13_shape),
        ("w2", weights.w2, logical_w2_shape),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be torch.bfloat16, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
    if weights.w13.device != weights.w2.device:
        raise ValueError("w13 and w2 must be on the same CUDA device")

    # FC1 N axis: gate/up interleaved in blocks of 8 rows (contiguous
    # (E, 2I, H)), then viewed K-major as (E, H, 2I) with H stride-1.  The
    # transpose keeps K stride-1 WITHOUT a .contiguous() re-pack (which would
    # silently break the K-major invariant the SM90 GEMM's TMA descriptors
    # depend on).
    w13_interleaved = _interleave_gate_up_8(weights.w13, intermediate_size=fc1_out)
    fc1_weight = w13_interleaved.transpose(1, 2)
    # FC2: canonical (E, H, I) row-major -> K-major (E, I, H) with I stride-1.
    fc2_weight = weights.w2.contiguous().transpose(1, 2)
    return fc1_weight, fc2_weight


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    """One-time check for kernel-ready BF16 weights (``preprocess_weights=False``).

    Shape / dtype / K-major stride checks only; the shim frontend re-validates
    per launch.
    """
    import torch

    from ......core.validation.common import MoEEpConfigError

    if world_size <= 0:
        raise MoEEpConfigError(f"world_size must be positive, got {world_size}")
    if num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({num_experts}) must be divisible by world_size ({world_size})"
        )
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise MoEEpConfigError(
            "transformed_weights must be a 2-tuple (fc1_weight, fc2_weight), got "
            f"{type(transformed).__name__}"
        )

    local_experts = num_experts // world_size
    fc1_out = 2 * intermediate_size
    expected = (
        ("fc1", transformed[0], (local_experts, hidden_size, fc1_out)),
        ("fc2", transformed[1], (local_experts, intermediate_size, hidden_size)),
    )
    for label, tensor, shape in expected:
        if not isinstance(tensor, torch.Tensor):
            raise MoEEpConfigError(
                f"transformed_weights {label} must be a torch.Tensor (the BF16 "
                f"path has no scale legs), got {type(tensor).__name__}"
            )
        if tuple(tensor.shape) != shape:
            raise MoEEpConfigError(
                f"transformed_weights {label} must have shape {shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"transformed_weights {label} must be torch.bfloat16, got {tensor.dtype}"
            )
        if not tensor.is_cuda:
            raise MoEEpConfigError(f"transformed_weights {label} must be a CUDA tensor")
        if tensor.stride(1) != 1:
            raise MoEEpConfigError(
                f"transformed_weights {label} must be K-major (stride-1 along "
                f"dim 1), got strides {tuple(tensor.stride())}"
            )


__all__ = [
    "MoEWeightPack",
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
