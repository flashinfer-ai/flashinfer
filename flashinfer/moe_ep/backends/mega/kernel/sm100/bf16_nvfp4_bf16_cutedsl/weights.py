"""Prepare packed W4A16 MegaMoE weights without decoding the full matrix."""

from __future__ import annotations

import torch

from ......weights import MoEWeightPack, PrequantizedMoEWeights
from ..nvfp4_nvfp4_bf16_cutedsl.weights import (
    _as_fp4_weight,
    _interleave_gate_up_16,
)

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
    weight: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    prepared_layout: bool = False,
) -> None:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    weight_shape = (shape[0], shape[2], shape[1]) if prepared_layout else shape
    if (
        weight.dtype not in (torch.uint8, fp4_dtype)
        or tuple(weight.shape) != weight_shape
    ):
        raise ValueError(
            f"W4A16 packed weights must be uint8/FP4 with shape {weight_shape}"
        )
    expected_scale_shape: tuple[int, ...]
    scale_dtypes: tuple[torch.dtype, ...]
    if prepared_layout:
        # block_scale_interleave pads each expert's N rows to 128 and K/16
        # scale columns to 4. Expose one padded native row per expert.
        padded_rows = ((shape[1] + 127) // 128) * 128
        padded_columns = ((shape[2] // 8 + 3) // 4) * 4
        expected_scale_shape = (shape[0], padded_rows * padded_columns)
        scale_dtypes = (torch.float8_e4m3fn, torch.uint8)
        layout = "native per-expert E4M3/uint8"
    else:
        expected_scale_shape = (*shape[:2], shape[2] // 8)
        scale_dtypes = (torch.float8_e4m3fn,)
        layout = "linear E4M3"
    if scale.dtype not in scale_dtypes or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"W4A16 block scales must be {layout} with shape {expected_scale_shape}"
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

    FC1 uses the same alternating 16-row gate/up blocks as W4A4. Packed
    weights expose K-major transpose views [E,K/2,N]; native block scales
    expose [E,padded_scale_bytes]. Both match W4A4's prepared byte layout.
    Runtime launches consume these buffers directly. Global scales
    remain FP32 epilogue operands: multiplying them into
    decoded BF16 weights would change the existing W4A16 rounding contract.
    """
    from flashinfer.quantization import block_scale_interleave

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
        w13, s13 = nvfp4_quantize_per_block_16(
            weights.w13.float().reshape(-1, hidden_size), 1.0
        )
        w2, s2 = nvfp4_quantize_per_block_16(
            weights.w2.float().reshape(-1, intermediate_size), 1.0
        )
        w13 = w13.reshape(local_experts, 2 * intermediate_size, hidden_size // 2)
        s13 = s13.reshape(local_experts, 2 * intermediate_size, hidden_size // 16)
        w2 = w2.reshape(local_experts, hidden_size, intermediate_size // 2)
        s2 = s2.reshape(local_experts, hidden_size, intermediate_size // 16)
    _validate_weight(w13, s13, (local_experts, 2 * intermediate_size, hidden_size // 2))
    _validate_weight(w2, s2, (local_experts, hidden_size, intermediate_size // 2))
    if w13.device != w2.device:
        raise ValueError("W4A16 FC1 and FC2 weights must be on the same device")
    alpha1 = _global_scale(weights.w13_global_scale, w13)
    alpha2 = _global_scale(weights.w2_global_scale, w2)
    w13 = _interleave_gate_up_16(
        w13.view(torch.uint8), intermediate_size=intermediate_size
    )
    s13 = _interleave_gate_up_16(s13, intermediate_size=intermediate_size)
    return (
        (
            _as_fp4_weight(w13).transpose(1, 2),
            block_scale_interleave(s13.contiguous().view(torch.uint8))
            .view(torch.float8_e4m3fn)
            .view(local_experts, -1),
            alpha1,
        ),
        (
            _as_fp4_weight(w2.view(torch.uint8).contiguous()).transpose(1, 2),
            block_scale_interleave(s2.contiguous().view(torch.uint8))
            .view(torch.float8_e4m3fn)
            .view(local_experts, -1),
            alpha2,
        ),
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
        _validate_weight(weight, scale, shape, prepared_layout=True)
        _global_scale(alpha, weight)
        if not weight.transpose(1, 2).is_contiguous():
            raise ValueError(
                "W4A16 transformed weights require a K-major transpose view"
            )
        if not scale.is_contiguous() or not alpha.is_contiguous():
            raise ValueError("W4A16 transformed scales and alphas must be contiguous")
    if transformed[0][0].device != transformed[1][0].device:
        raise ValueError("W4A16 FC1 and FC2 weights must be on the same device")
