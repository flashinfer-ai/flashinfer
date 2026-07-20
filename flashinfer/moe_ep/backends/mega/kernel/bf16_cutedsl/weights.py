"""BF16 MegaMoE weight layout conversion and validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from .....weights import MoEWeightPack

if TYPE_CHECKING:
    import torch


TransformedMegaWeights = Tuple[
    Tuple["torch.Tensor", None],
    Tuple["torch.Tensor", None],
]


def _interleave_gate_up_32(
    w13: "torch.Tensor", intermediate_size: int
) -> "torch.Tensor":
    import torch

    if w13.shape[1] != 2 * intermediate_size:
        raise ValueError(
            f"w13 must have {2 * intermediate_size} rows, got {tuple(w13.shape)}."
        )
    if intermediate_size % 32:
        raise ValueError("intermediate_size must be divisible by 32.")
    gate, up = w13[:, :intermediate_size], w13[:, intermediate_size:]
    out = torch.empty_like(w13)
    out.view(w13.shape[0], intermediate_size // 32, 2, 32, w13.shape[2])[:, :, 0].copy_(
        gate.view(w13.shape[0], intermediate_size // 32, 32, w13.shape[2])
    )
    out.view(w13.shape[0], intermediate_size // 32, 2, 32, w13.shape[2])[:, :, 1].copy_(
        up.view(w13.shape[0], intermediate_size // 32, 32, w13.shape[2])
    )
    return out


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """Convert canonical BF16 weights to the kernel's K-major layouts."""
    import torch

    expected_w13 = (weights.w13.shape[0], 2 * intermediate_size, hidden_size)
    expected_w2 = (weights.w2.shape[0], hidden_size, intermediate_size)
    if (
        tuple(weights.w13.shape) != expected_w13
        or tuple(weights.w2.shape) != expected_w2
    ):
        raise ValueError(
            f"expected w13/w2 shapes {expected_w13}/{expected_w2}, got "
            f"{tuple(weights.w13.shape)}/{tuple(weights.w2.shape)}."
        )
    if weights.w13.dtype != torch.bfloat16 or weights.w2.dtype != torch.bfloat16:
        raise ValueError("BF16 MegaMoE requires bfloat16 canonical weights.")
    fc1 = _interleave_gate_up_32(weights.w13, intermediate_size).transpose(1, 2)
    fc2 = weights.w2.transpose(1, 2)
    return (fc1, None), (fc2, None)


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    import torch

    if len(transformed) != 2 or any(len(pair) != 2 for pair in transformed):
        raise ValueError("transformed BF16 weights must be two (weight, None) pairs.")
    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size.")
    local_experts = num_experts // world_size
    expected = (
        (local_experts, hidden_size, 2 * intermediate_size),
        (local_experts, intermediate_size, hidden_size),
    )
    for (weight, scale), shape in zip(transformed, expected, strict=True):
        if scale is not None:
            raise ValueError("BF16 MegaMoE transformed weights do not use scales.")
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != shape:
            raise ValueError(
                f"expected bfloat16 transformed weight with shape {shape}."
            )
