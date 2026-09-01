"""Host-side transforms for SM120 packed NVFP4 expert weights."""

from __future__ import annotations

from typing import Tuple

import torch

FP4_DTYPE = torch.float4_e2m1fn_x2
SCALE_DTYPE = torch.float8_e4m3fn
SCALE_BLOCK = 16
SCALE_ROW_PADDING = 128
GATE_UP_INTERLEAVE = 16

TransformedWeights = Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def round_up(value: int, divisor: int) -> int:
    return ceil_div(value, divisor) * divisor


def as_fp4(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == FP4_DTYPE:
        return tensor
    if tensor.dtype == torch.uint8:
        return tensor.view(FP4_DTYPE)
    raise ValueError(
        f"packed NVFP4 data must be uint8 or {FP4_DTYPE}, got {tensor.dtype}"
    )


def as_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == SCALE_DTYPE:
        return tensor
    if tensor.dtype == torch.uint8:
        return tensor.view(SCALE_DTYPE)
    raise ValueError(f"NVFP4 scales must be uint8 or {SCALE_DTYPE}, got {tensor.dtype}")


def interleave_gate_up_16(tensor: torch.Tensor, full_width: int) -> torch.Tensor:
    """Convert canonical gate||up rows to the kernel's 16-row alternation."""

    if tensor.ndim != 3 or tensor.shape[1] != full_width:
        raise ValueError(
            f"expected (experts, {full_width}, columns), got {tuple(tensor.shape)}"
        )
    half = full_width // 2
    if half % GATE_UP_INTERLEAVE:
        raise ValueError(
            f"post-SwiGLU width {half} must be divisible by {GATE_UP_INTERLEAVE}"
        )
    experts, _rows, columns = tensor.shape
    pairs = half // GATE_UP_INTERLEAVE
    gate = (
        tensor[:, :half].contiguous().view(experts, pairs, GATE_UP_INTERLEAVE, columns)
    )
    up = tensor[:, half:].contiguous().view(experts, pairs, GATE_UP_INTERLEAVE, columns)
    output = tensor.new_empty((experts, pairs, 2, GATE_UP_INTERLEAVE, columns))
    output[:, :, 0].copy_(gate)
    output[:, :, 1].copy_(up)
    return output.reshape(experts, full_width, columns).contiguous()


def to_blocked(scale: torch.Tensor) -> torch.Tensor:
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import to_blocked as impl

    return impl(scale)


def stack_byte_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("at least one tensor is required")
    return torch.stack([tensor.view(torch.uint8) for tensor in tensors]).view(
        tensors[0].dtype
    )


def scale_storage_size(rows: int, logical_k: int) -> int:
    return round_up(rows, SCALE_ROW_PADDING) * round_up(
        ceil_div(logical_k, SCALE_BLOCK), 4
    )


def _quantize_weights(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )

    quantized = []
    scales = []
    for expert in range(weight.shape[0]):
        q, sf = nvfp4_quantize_per_block_16(weight[expert].to(torch.float32), 1.0)
        quantized.append(q)
        scales.append(sf)
    return torch.stack(quantized), torch.stack(scales)


def transform_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    *,
    hidden: int,
    intermediate: int,
) -> TransformedWeights:
    """Canonical W13/W2 tensors to the SM120 NVFP4 kernel layout."""

    local_experts = w13.shape[0]
    logical_w13 = (local_experts, 2 * intermediate, hidden)
    logical_w2 = (local_experts, hidden, intermediate)
    packed_w13 = (local_experts, 2 * intermediate, hidden // 2)
    packed_w2 = (local_experts, hidden, intermediate // 2)

    if w13_scale is None or w2_scale is None:
        if w13_scale is not None or w2_scale is not None:
            raise ValueError("both NVFP4 weight-scale planes must be provided together")
        if tuple(w13.shape) != logical_w13 or tuple(w2.shape) != logical_w2:
            raise ValueError(
                f"unquantized weights must have shapes {logical_w13} and {logical_w2}"
            )
        interleaved_w13 = interleave_gate_up_16(w13, 2 * intermediate)
        fc1_weight, fc1_scale = _quantize_weights(interleaved_w13)
        fc2_weight, fc2_scale = _quantize_weights(w2)
    else:
        if tuple(w13.shape) != packed_w13 or tuple(w2.shape) != packed_w2:
            raise ValueError(
                f"packed weights must have shapes {packed_w13} and {packed_w2}"
            )
        expected_w13_scale = (
            local_experts,
            2 * intermediate,
            hidden // SCALE_BLOCK,
        )
        expected_w2_scale = (
            local_experts,
            hidden,
            intermediate // SCALE_BLOCK,
        )
        if tuple(w13_scale.shape) != expected_w13_scale:
            raise ValueError(
                f"w13_scale must have shape {expected_w13_scale}, got {tuple(w13_scale.shape)}"
            )
        if tuple(w2_scale.shape) != expected_w2_scale:
            raise ValueError(
                f"w2_scale must have shape {expected_w2_scale}, got {tuple(w2_scale.shape)}"
            )
        fc1_weight = interleave_gate_up_16(as_fp4(w13), 2 * intermediate)
        fc2_weight = as_fp4(w2).contiguous()
        fc1_scale = interleave_gate_up_16(as_e4m3(w13_scale), 2 * intermediate)
        fc2_scale = as_e4m3(w2_scale).contiguous()

    fc1_swizzled = stack_byte_tensors(
        [to_blocked(fc1_scale[expert]) for expert in range(local_experts)]
    )
    fc2_swizzled = stack_byte_tensors(
        [to_blocked(fc2_scale[expert]) for expert in range(local_experts)]
    )
    return (
        (fc1_weight.contiguous(), fc1_swizzled),
        (fc2_weight.contiguous(), fc2_swizzled),
    )


__all__ = [
    "FP4_DTYPE",
    "SCALE_BLOCK",
    "SCALE_DTYPE",
    "TransformedWeights",
    "ceil_div",
    "round_up",
    "scale_storage_size",
    "to_blocked",
    "transform_weights",
]
