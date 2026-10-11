"""Mega-path BF16 weight preprocessing for the SM90 pull-style CuTeDSL kernel.

BF16 rides the kernel's per-tensor ABI with unit scales: each leg is the
same ``(weight, weight_sf, activation_dequant_scale, weight_dequant_scale)``
4-tuple as the FP8 per-tensor layout, with BF16 K-major weights, the unit
E8M0 SF placeholder planes, and all-ones dequant scales.  FC1 gate/up is
interleaved in blocks of 8 rows (``Fp8GateUpInterleave``), as for FP8.
"""

from __future__ import annotations

from ......weights import MoEWeightPack, PrequantizedMoEWeights
from ..fp8_fp8_bf16_pull_cutedsl.weights import (
    TransformedMegaWeights,
    _interleave_gate_up_8,
    _swizzle_unit_e8m0_sf,
    _swizzled_flat_e8m0_size,
)

_KERNEL = "sm90_bf16_bf16_bf16_pull_cutedsl"


def preprocess_mega_weights(
    weights: "MoEWeightPack",
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """Canonical BF16 (or FP16/FP32, cast to BF16) weights → kernel layout.

    ``w13`` ``(E, 2I, H)`` gate-first and ``w2`` ``(E, H, I)`` become K-major
    ``(E, H, 2I)`` / ``(E, I, H)`` views (no ``.contiguous()`` after the
    transpose: the TMA descriptors need K stride-1).
    """
    import torch

    from ......core.validation.common import MoEEpConfigError

    if isinstance(weights, PrequantizedMoEWeights):
        raise MoEEpConfigError(
            f"{_KERNEL} computes in BF16 and takes canonical w13/w2 only; got "
            "PrequantizedMoEWeights (use an FP8 SM90 backend for quantized "
            "checkpoints)"
        )
    fc1_out = 2 * intermediate_size
    num_experts = weights.w13.shape[0]
    for name, tensor, shape in (
        ("w13", weights.w13, (num_experts, fc1_out, hidden_size)),
        ("w2", weights.w2, (num_experts, hidden_size, intermediate_size)),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise MoEEpConfigError(
                f"{_KERNEL} {name} must be bf16, fp16 or fp32, got {tensor.dtype}"
            )

    device = weights.w13.device
    w13 = _interleave_gate_up_8(
        weights.w13.to(torch.bfloat16), intermediate_size=fc1_out
    )
    w2 = weights.w2.to(torch.bfloat16).contiguous()
    ones_1 = torch.ones(1, dtype=torch.float32, device=device)
    ones_e = torch.ones(num_experts, dtype=torch.float32, device=device)
    fc1_sf = _swizzle_unit_e8m0_sf(num_experts, fc1_out, hidden_size // 32, device)
    fc2_sf = _swizzle_unit_e8m0_sf(
        num_experts, hidden_size, intermediate_size // 32, device
    )
    return (
        (w13.transpose(1, 2), fc1_sf, ones_1, ones_e),
        (w2.transpose(1, 2), fc2_sf, ones_1, ones_e.clone()),
    )


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    """One-time check for kernel-ready BF16 weights (``preprocess_weights=False``)."""
    import torch

    from ......core.validation.common import MoEEpConfigError
    from ...weight_validation import check_transformed_weight_pair
    from ..fp8_fp8_bf16_pull_cutedsl.weights import _check_leg_structure

    if world_size <= 0 or num_experts % world_size != 0:
        raise MoEEpConfigError(
            f"num_experts ({num_experts}) must be divisible by world_size "
            f"({world_size})"
        )
    _check_leg_structure(transformed)
    local_experts = num_experts // world_size
    fc1_out = 2 * intermediate_size
    for idx, (label, weight_shape, sf_rows, sf_cols) in enumerate(
        (
            ("fc1", (local_experts, hidden_size, fc1_out), fc1_out, hidden_size // 32),
            (
                "fc2",
                (local_experts, intermediate_size, hidden_size),
                hidden_size,
                intermediate_size // 32,
            ),
        )
    ):
        check_transformed_weight_pair(
            transformed[idx][:2],
            label=label,
            num_local_experts=local_experts,
            weight_dtype=torch.bfloat16,
            expected_weight_shape=weight_shape,
            scale_dtype=torch.uint8,
            expected_scale_shape=(
                local_experts,
                _swizzled_flat_e8m0_size(sf_rows, sf_cols),
            ),
        )
        if transformed[idx][0].stride(1) != 1:
            raise MoEEpConfigError(
                f"transformed_weights {label} weight must be K-major (stride-1 "
                f"along dim 1), got strides {tuple(transformed[idx][0].stride())}"
            )
        # The kernel applies these per-tensor slots; BF16 has nothing to
        # dequantize, so anything but ones would silently rescale outputs.
        for name, scale, shape in (
            (f"{label} activation_dequant_scale", transformed[idx][2], (1,)),
            (f"{label} weight_dequant_scale", transformed[idx][3], (local_experts,)),
        ):
            if (
                not isinstance(scale, torch.Tensor)
                or tuple(scale.shape) != shape
                or scale.dtype != torch.float32
                or not bool(torch.all(scale == 1))
            ):
                raise MoEEpConfigError(
                    f"transformed_weights {name} must be a float32 ones tensor "
                    f"of shape {shape}"
                )


__all__ = [
    "TransformedMegaWeights",
    "preprocess_mega_weights",
    "validate_transformed_mega_weights",
]
