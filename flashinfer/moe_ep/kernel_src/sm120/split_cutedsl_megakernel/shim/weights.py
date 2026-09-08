"""Host-side layout transforms for packed MXFP4 expert weights."""

from __future__ import annotations

from typing import Tuple

import torch


FP4_DTYPE = torch.float4_e2m1fn_x2
SCALE_DTYPE = torch.float8_e8m0fnu
SCALE_BLOCK = 32
SCALE_ROW_PADDING = 128
GATE_UP_INTERLEAVE = 8

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
    raise ValueError(f"packed MXFP4 data must be uint8 or {FP4_DTYPE}, got {tensor.dtype}")


def as_e8m0(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == SCALE_DTYPE:
        return tensor
    if tensor.dtype == torch.uint8:
        return tensor.view(SCALE_DTYPE)
    raise ValueError(f"MX block scales must be uint8 or {SCALE_DTYPE}, got {tensor.dtype}")


def interleave_gate_up_8(tensor: torch.Tensor, full_width: int) -> torch.Tensor:
    """Convert canonical gate||up rows to the kernel's 8-row alternation."""

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
    gate = tensor[:, :half].contiguous().view(
        experts, pairs, GATE_UP_INTERLEAVE, columns
    )
    up = tensor[:, half:].contiguous().view(
        experts, pairs, GATE_UP_INTERLEAVE, columns
    )
    output = tensor.new_empty((experts, pairs, 2, GATE_UP_INTERLEAVE, columns))
    output[:, :, 0].copy_(gate)
    output[:, :, 1].copy_(up)
    return output.reshape(experts, full_width, columns).contiguous()


def to_blocked(scale: torch.Tensor) -> torch.Tensor:
    """Apply the kernel's 32x4x4 E8M0 scale swizzle."""

    if scale.ndim != 2:
        raise ValueError(f"scale must be 2-D, got {scale.ndim}-D")
    rows, columns = scale.shape
    padded_rows = round_up(rows, SCALE_ROW_PADDING)
    padded_columns = round_up(columns, 4)
    if (rows, columns) != (padded_rows, padded_columns):
        padded = torch.zeros(
            (padded_rows, padded_columns), dtype=scale.dtype, device=scale.device
        )
        padded[:rows, :columns] = scale
    else:
        padded = scale
    blocks = padded.view(
        padded_rows // SCALE_ROW_PADDING, SCALE_ROW_PADDING, padded_columns // 4, 4
    ).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def stack_byte_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("at least one tensor is required")
    return torch.stack([tensor.view(torch.uint8) for tensor in tensors]).view(
        tensors[0].dtype
    )


def transform_prequantized_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    hidden: int,
    intermediate: int,
) -> TransformedWeights:
    """Canonical packed W13/W2 and plain E8M0 scales to kernel-ready layout."""

    local_experts = w13.shape[0]
    expected_w13 = (local_experts, 2 * intermediate, hidden // 2)
    expected_w2 = (local_experts, hidden, intermediate // 2)
    expected_w13_scale = (local_experts, 2 * intermediate, hidden // SCALE_BLOCK)
    expected_w2_scale = (local_experts, hidden, intermediate // SCALE_BLOCK)
    for name, tensor, expected in (
        ("w13", w13, expected_w13),
        ("w2", w2, expected_w2),
        ("w13_scale", w13_scale, expected_w13_scale),
        ("w2_scale", w2_scale, expected_w2_scale),
    ):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")

    fc1_weight = interleave_gate_up_8(as_fp4(w13), 2 * intermediate)
    fc2_weight = as_fp4(w2).contiguous()
    fc1_scale = interleave_gate_up_8(as_e8m0(w13_scale), 2 * intermediate)
    fc2_scale = as_e8m0(w2_scale).contiguous()
    fc1_swizzled = stack_byte_tensors(
        [to_blocked(fc1_scale[expert]) for expert in range(local_experts)]
    )
    fc2_swizzled = stack_byte_tensors(
        [to_blocked(fc2_scale[expert]) for expert in range(local_experts)]
    )
    return (fc1_weight, fc1_swizzled), (fc2_weight, fc2_swizzled)


__all__ = [
    "FP4_DTYPE",
    "SCALE_BLOCK",
    "SCALE_DTYPE",
    "TransformedWeights",
    "ceil_div",
    "round_up",
    "transform_prequantized_weights",
]
