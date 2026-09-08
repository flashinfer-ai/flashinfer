"""Prepare packed W4A16 MegaMoE weights without decoding the full matrix."""

from __future__ import annotations

import torch

from ......weights import MoEWeightPack, PrequantizedMoEWeights
from ..bf16_bf16_bf16_cutedsl.weights import _interleave_gate_up_32

ExpertWeight = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
TransformedMegaWeights = tuple[ExpertWeight, ExpertWeight]


def _global_scale(scale: torch.Tensor | None, weight: torch.Tensor) -> torch.Tensor:
    if scale is None:
        return torch.ones(weight.shape[0], dtype=torch.float32, device=weight.device)
    if scale.dtype != torch.float32 or scale.shape != (weight.shape[0],):
        raise ValueError("W4A16 global weight scales must be FP32 [local_experts]")
    if scale.device != weight.device:
        raise ValueError("W4A16 global weight scales must be on the weight device")
    return scale.contiguous()


def _validate_weight(
    weight: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int, int]
) -> None:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype not in (torch.uint8, fp4_dtype) or tuple(weight.shape) != shape:
        raise ValueError(f"W4A16 packed weights must be uint8/FP4 with shape {shape}")
    expected_scale_shape = (*shape[:2], shape[2] // 8)
    if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"W4A16 block scales must be linear E4M3 with shape {expected_scale_shape}"
        )
    if scale.device != weight.device:
        raise ValueError("W4A16 packed weights and scales must be on the same device")


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
) -> TransformedMegaWeights:
    """Canonical gate/up weights → packed kernel layout and separate alphas.

    FC1 uses alternating 32-row gate/up blocks. FC2 stays in canonical N,K
    order. Global scales remain FP32 epilogue operands: multiplying them into
    decoded BF16 weights would change the existing W4A16 rounding contract.
    """
    local_experts = weights.w13.shape[0]
    if isinstance(weights, PrequantizedMoEWeights):
        w13, w2 = weights.w13, weights.w2
        s13, s2 = weights.w13_scale, weights.w2_scale
    else:
        from ......kernel_src.cutedsl_megamoe import nvfp4_quantize_per_block_16

        if any(
            t.dtype not in (torch.bfloat16, torch.float32)
            for t in (weights.w13, weights.w2)
        ):
            raise ValueError("W4A16 unquantized weights must be BF16 or FP32")
        expected = (
            (local_experts, 2 * intermediate_size, hidden_size),
            (local_experts, hidden_size, intermediate_size),
        )
        if (tuple(weights.w13.shape), tuple(weights.w2.shape)) != expected:
            raise ValueError(f"W4A16 unquantized weights must have shapes {expected}")
        w13, s13 = nvfp4_quantize_per_block_16(weights.w13.float(), 1.0)
        w2, s2 = nvfp4_quantize_per_block_16(weights.w2.float(), 1.0)
    _validate_weight(w13, s13, (local_experts, 2 * intermediate_size, hidden_size // 2))
    _validate_weight(w2, s2, (local_experts, hidden_size, intermediate_size // 2))
    if w13.device != w2.device:
        raise ValueError("W4A16 FC1 and FC2 weights must be on the same device")
    alpha1 = _global_scale(weights.w13_global_scale, w13)
    alpha2 = _global_scale(weights.w2_global_scale, w2)
    return (
        (
            _interleave_gate_up_32(
                w13.view(torch.uint8), intermediate_size
            ).contiguous(),
            _interleave_gate_up_32(s13, intermediate_size).contiguous(),
            alpha1,
        ),
        (w2.view(torch.uint8).contiguous(), s2.contiguous(), alpha2),
    )


def validate_transformed_mega_weights(
    transformed: TransformedMegaWeights,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
) -> None:
    if len(transformed) != 2 or any(len(parts) != 3 for parts in transformed):
        raise ValueError(
            "W4A16 transformed weights require two (weight, scale, alpha) triples"
        )
    local_experts = num_experts // world_size
    shapes = (
        (local_experts, 2 * intermediate_size, hidden_size // 2),
        (local_experts, hidden_size, intermediate_size // 2),
    )
    for (weight, scale, alpha), shape in zip(transformed, shapes, strict=True):
        _validate_weight(weight, scale, shape)
        _global_scale(alpha, weight)
        if not all(t.is_contiguous() for t in (weight, scale, alpha)):
            raise ValueError("W4A16 transformed weight tensors must be contiguous")
    if transformed[0][0].device != transformed[1][0].device:
        raise ValueError("W4A16 FC1 and FC2 weights must be on the same device")
