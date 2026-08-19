"""Weight preprocessing for the SM120 MXFP4 x MXFP8 split backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ......weights import MoEWeightPack, PrequantizedMoEWeights

if TYPE_CHECKING:
    from ......kernel_src.sm120.split_cutedsl_megakernel import TransformedWeights


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> "TransformedWeights":
    if not isinstance(weights, PrequantizedMoEWeights):
        raise NotImplementedError(
            "SM120 MXFP4 x MXFP8 requires pre-quantized packed E2M1 weights "
            "and both E8M0 scale planes"
        )
    from ......kernel_src.sm120.split_cutedsl_megakernel import (
        transform_prequantized_weights,
    )

    return transform_prequantized_weights(
        weights.w13,
        weights.w2,
        weights.w13_scale,
        weights.w2_scale,
        hidden=hidden_size,
        intermediate=intermediate_size,
    )


def validate_transformed_mega_weights(
    transformed_weights: "TransformedWeights",
    *,
    hidden_size: int,
    intermediate_size: int,
    local_experts: int,
) -> None:
    from ......kernel_src.sm120.split_cutedsl_megakernel import (
        FP4_DTYPE,
        SCALE_DTYPE,
    )

    try:
        (fc1, fc1_scale), (fc2, fc2_scale) = transformed_weights
    except (TypeError, ValueError) as exc:
        raise ValueError("transformed W4A8 weights must be ((fc1, sf1), (fc2, sf2))") from exc
    expected = (
        ("fc1", fc1, (local_experts, 2 * intermediate_size, hidden_size // 2)),
        ("fc2", fc2, (local_experts, hidden_size, intermediate_size // 2)),
    )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape or tensor.dtype != FP4_DTYPE or not tensor.is_cuda:
            raise ValueError(
                f"{name} must be CUDA {FP4_DTYPE} with shape {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
    for name, tensor in (("fc1_scale", fc1_scale), ("fc2_scale", fc2_scale)):
        if (
            tensor.ndim != 2
            or tensor.shape[0] != local_experts
            or tensor.dtype != SCALE_DTYPE
            or not tensor.is_cuda
        ):
            raise ValueError(
                f"{name} must be a CUDA E8M0 2-D tensor with leading extent "
                f"{local_experts}"
            )


__all__ = ["preprocess_mega_weights", "validate_transformed_mega_weights"]
