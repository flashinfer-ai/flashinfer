"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Independent W4A8 operand and weight simulation used by SM90 tests.
"""

from __future__ import annotations

import math

import torch

from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.nvfp4_checkpoint import (
    NVFP4Checkpoint,
)


E4M3_MAX = 448.0
GROUP_SIZES = (128, 64, 32)
RESIDUAL_SCHEMES = ("generic", "pow2")
POW2_ZERO_SENTINEL = -128
FINITENESS_CHUNK_ELEMENTS = 8 * 1024 * 1024


def _all_finite_in_chunks(
    tensor: torch.Tensor, chunk_elements: int = FINITENESS_CHUNK_ELEMENTS
) -> bool:
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    if tensor.numel() == 0:
        return True
    if tensor.numel() <= chunk_elements:
        return bool(torch.isfinite(tensor).all())
    axis = max(range(tensor.ndim), key=lambda index: tensor.shape[index])
    axis_size = tensor.shape[axis]
    elements_per_slice = tensor.numel() // axis_size
    slices_per_chunk = max(1, chunk_elements // elements_per_slice)
    for start in range(0, axis_size, slices_per_chunk):
        chunk = tensor.narrow(axis, start, min(slices_per_chunk, axis_size - start))
        if not _all_finite_in_chunks(chunk, chunk_elements):
            return False
    return True


def _decode_codes(checkpoint: NVFP4Checkpoint) -> torch.Tensor:
    low = checkpoint.packed_e2m1 & 0x0F
    high = (checkpoint.packed_e2m1 >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(checkpoint.physical_shape)
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=checkpoint.device,
    )
    values = magnitudes[(codes & 0x07).to(torch.int64)]
    return torch.where((codes & 0x08) != 0, -values, values)


def _round_e4m3(values: torch.Tensor) -> torch.Tensor:
    return values.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def simulate_w4a8_operand_bytes(
    packed_e2m1: torch.Tensor,
    promotion_residual: torch.Tensor,
    *,
    residual_scheme: str,
) -> torch.Tensor:
    """Decode v3 physical streams into raw E4M3FN operand bytes.

    Inputs use the device-facing v3 tile coordinates. The returned uint8 tensor
    has shape ``[E,Kt,Nt,64,32]``; its last dimension contains unpacked K32
    bytes in low-even/high-odd order before group-scale or alpha application.
    """

    if residual_scheme not in RESIDUAL_SCHEMES:
        raise ValueError(
            f"residual_scheme must be one of {RESIDUAL_SCHEMES}, "
            f"got {residual_scheme!r}"
        )
    if not isinstance(packed_e2m1, torch.Tensor) or not isinstance(
        promotion_residual, torch.Tensor
    ):
        raise TypeError("packed_e2m1 and promotion_residual must be tensors")
    if packed_e2m1.dtype != torch.uint8 or packed_e2m1.ndim != 5:
        raise ValueError("packed_e2m1 must be uint8 [E,Kt,Nt,64,16]")
    if tuple(packed_e2m1.shape[-2:]) != (64, 16):
        raise ValueError("packed_e2m1 physical tile must be [64,16]")
    expected_residual_shape = (*packed_e2m1.shape[:-1], 2)
    if tuple(promotion_residual.shape) != expected_residual_shape:
        raise ValueError(
            "promotion_residual shape must be "
            f"{expected_residual_shape}, got {tuple(promotion_residual.shape)}"
        )
    expected_dtype = torch.bfloat16 if residual_scheme == "generic" else torch.int8
    if promotion_residual.dtype != expected_dtype:
        raise ValueError(
            f"{residual_scheme} promotion_residual must have dtype {expected_dtype}"
        )
    if not packed_e2m1.is_contiguous() or not promotion_residual.is_contiguous():
        raise ValueError("v3 operand streams must be contiguous")
    if packed_e2m1.device != promotion_residual.device:
        raise ValueError("v3 operand streams must share a device")

    low = packed_e2m1.bitwise_and(0x0F)
    high = packed_e2m1.bitwise_right_shift(4).bitwise_and(0x0F)
    codes = torch.stack((low, high), dim=-1).reshape(
        *packed_e2m1.shape[:-1], packed_e2m1.shape[-1] * 2
    )
    magnitudes = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=packed_e2m1.device,
    )
    decoded = magnitudes[(codes & 0x07).to(torch.int64)]
    decoded = torch.where((codes & 0x08) != 0, -decoded, decoded)

    if residual_scheme == "generic":
        if not _all_finite_in_chunks(promotion_residual):
            raise ValueError("generic promotion_residual must be finite")
        if bool((promotion_residual < 0).any()):
            raise ValueError("generic promotion_residual must be non-negative")
        residual_value = promotion_residual.to(torch.float32)
    else:
        exponent = promotion_residual.to(torch.int32)
        residual_value = torch.where(
            promotion_residual == POW2_ZERO_SENTINEL,
            torch.zeros_like(exponent, dtype=torch.float32),
            torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent),
        )
    normalized = decoded * residual_value.repeat_interleave(16, dim=-1)
    return (
        normalized.clamp(-E4M3_MAX, E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .contiguous()
    )


def _padded_inputs(
    checkpoint: NVFP4Checkpoint, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if group_size not in GROUP_SIZES:
        raise ValueError(f"group_size must be one of {GROUP_SIZES}, got {group_size}")
    experts, rows, logical_k = checkpoint.logical_shape
    padded_k = math.ceil(logical_k / group_size) * group_size
    codes = _decode_codes(checkpoint)[:, :rows, :logical_k]
    scales = checkpoint.scale_e4m3_per16.to(torch.float32).repeat_interleave(
        16, dim=-1
    )[:, :rows, :logical_k]
    if padded_k != logical_k:
        codes = torch.nn.functional.pad(codes, (0, padded_k - logical_k))
        scales = torch.nn.functional.pad(scales, (0, padded_k - logical_k))
    if experts == 0:
        codes = codes.reshape(0, rows, padded_k)
        scales = scales.reshape(0, rows, padded_k)
    return codes, scales, logical_k


def simulate_w4a8(
    checkpoint: NVFP4Checkpoint,
    group_size: int,
    residual_scheme: str,
    *,
    apply_global_alpha: bool = True,
) -> torch.Tensor:
    """Simulate the promoted W4A8 weight represented by one checkpoint."""

    if residual_scheme not in RESIDUAL_SCHEMES:
        raise ValueError(
            f"residual_scheme must be one of {RESIDUAL_SCHEMES}, "
            f"got {residual_scheme!r}"
        )
    codes, scales, logical_k = _padded_inputs(checkpoint, group_size)
    experts, rows, padded_k = codes.shape
    groups = padded_k // group_size
    code_groups = codes.reshape(experts, rows, groups, group_size)
    scale_groups = scales.reshape(experts, rows, groups, group_size)
    unscaled = code_groups * scale_groups
    group_scale = unscaled.abs().amax(dim=-1, keepdim=True) / E4M3_MAX
    group_scale = torch.where(
        group_scale > 0, group_scale, torch.ones_like(group_scale)
    )

    scale_blocks = scale_groups.reshape(experts, rows, groups, group_size // 16, 16)[
        ..., 0
    ]
    if residual_scheme == "generic":
        group_scale = group_scale * (1.0 + 2.0**-7)
        residual_blocks = (
            (scale_blocks / group_scale).to(torch.bfloat16).to(torch.float32)
        )
        residual = residual_blocks.repeat_interleave(16, dim=-1)
    else:
        ratio = scale_blocks / group_scale
        exponent = torch.where(
            ratio > 0,
            torch.round(torch.log2(ratio)),
            torch.zeros_like(ratio),
        ).clamp(POW2_ZERO_SENTINEL + 1, 127)
        residual_blocks = torch.where(
            ratio > 0,
            torch.pow(torch.full_like(exponent, 2.0), exponent),
            torch.zeros_like(exponent),
        )
        residual = residual_blocks.repeat_interleave(16, dim=-1)
        normalized_max = (code_groups * residual).abs().amax(dim=-1, keepdim=True)
        shift = torch.where(
            normalized_max > E4M3_MAX,
            torch.ceil(torch.log2(normalized_max / E4M3_MAX)),
            torch.zeros_like(normalized_max),
        ).clamp_min(0)
        shift_factor = torch.pow(torch.full_like(shift, 2.0), shift)
        group_scale = group_scale * shift_factor
        residual = residual / shift_factor

    rounded = _round_e4m3(code_groups * residual)
    reconstructed = (rounded * group_scale).reshape(experts, rows, padded_k)
    if apply_global_alpha:
        reconstructed = (
            reconstructed * checkpoint.global_alpha_per_expert[:, None, None]
        )
    return reconstructed[:, :, :logical_k].contiguous()


__all__ = ["simulate_w4a8", "simulate_w4a8_operand_bytes"]
