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
    weight: torch.Tensor,
    scale: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    native_sf: bool = False,
) -> None:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if weight.dtype not in (torch.uint8, fp4_dtype) or tuple(weight.shape) != shape:
        raise ValueError(f"W4A16 packed weights must be uint8/FP4 with shape {shape}")
    expected_scale_shape: tuple[int, ...]
    if native_sf:
        # block_scale_interleave pads each expert's N rows to128 and K/16
        # scale columns to4, then returns one flat native buffer.
        padded_rows = ((shape[1] + 127) // 128) * 128
        padded_columns = ((shape[2] // 8 + 3) // 4) * 4
        expected_scale_shape = (shape[0] * padded_rows * padded_columns,)
        layout = "native flat"
    else:
        expected_scale_shape = (*shape[:2], shape[2] // 8)
        layout = "linear"
    if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"W4A16 block scales must be {layout} E4M3 with shape {expected_scale_shape}"
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

    FC1 uses alternating32-row gate/up blocks. FC2 stays in canonical N,K
    order. Block scales are converted once to flat native storage here;
    runtime launches consume these prepared buffers directly. Global scales
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
            block_scale_interleave(
                _interleave_gate_up_32(s13, intermediate_size)
                .contiguous()
                .view(torch.uint8)
            ).view(torch.float8_e4m3fn),
            alpha1,
        ),
        (
            w2.view(torch.uint8).contiguous(),
            block_scale_interleave(s2.contiguous().view(torch.uint8)).view(
                torch.float8_e4m3fn
            ),
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
        _validate_weight(weight, scale, shape, native_sf=True)
        _global_scale(alpha, weight)
        if not all(t.is_contiguous() for t in (weight, scale, alpha)):
            raise ValueError("W4A16 transformed weight tensors must be contiguous")
    if transformed[0][0].device != transformed[1][0].device:
        raise ValueError("W4A16 FC1 and FC2 weights must be on the same device")
