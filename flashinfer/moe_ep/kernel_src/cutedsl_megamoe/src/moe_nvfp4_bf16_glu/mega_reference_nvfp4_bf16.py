# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NVFP4-weight to BF16 transform and MegaMoE reference."""

from __future__ import annotations

from typing import Final, Literal, Optional

import torch

NVFP4_BLOCK_SIZE: Final[int] = 16
NVFP4_GATE_UP_INTERLEAVE: Final[int] = 16
_FP4_DECODE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _unpack_fp4_to_f32(
    packed: torch.Tensor,
    *,
    packed_dim: int,
) -> torch.Tensor:
    if packed.ndim == 0:
        raise ValueError("packed NVFP4 tensor must have at least one dimension.")
    if not -packed.ndim <= packed_dim < packed.ndim:
        raise ValueError(
            f"packed_dim must index a rank-{packed.ndim} tensor, got "
            f"{packed_dim}."
        )
    packed_dim %= packed.ndim
    raw = packed.view(torch.uint8)
    if packed_dim != raw.ndim - 1:
        permutation = list(range(raw.ndim))
        permutation[packed_dim], permutation[-1] = (
            permutation[-1],
            permutation[packed_dim],
        )
        raw = raw.permute(permutation).contiguous()
    else:
        permutation = None
    lut = torch.tensor(_FP4_DECODE, dtype=torch.float32, device=raw.device)
    result = torch.empty(
        (*raw.shape[:-1], raw.shape[-1] * 2),
        dtype=torch.float32,
        device=raw.device,
    )
    result[..., 0::2] = lut[(raw & 0x0F).to(torch.int64)]
    result[..., 1::2] = lut[(raw >> 4).to(torch.int64)]
    if permutation is not None:
        result = result.permute(permutation)
    return result


def _from_blocked(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    row_blocks = (rows + 127) // 128
    col_blocks = (cols + 3) // 4
    padded_rows = row_blocks * 128
    padded_cols = col_blocks * 4
    if flat.numel() != padded_rows * padded_cols:
        raise ValueError("invalid atom-swizzled scale plane size.")
    rearranged = flat.reshape(-1, 32, 16).reshape(-1, 32, 4, 4)
    blocks = rearranged.transpose(1, 2).reshape(-1, 128, 4)
    blocks = blocks.reshape(row_blocks, col_blocks, 128, 4)
    padded = blocks.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return padded[:rows, :cols].contiguous()


def nvfp4_weight_to_bf16(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Decode logical ``(N,K)`` NVFP4 weights at the BF16 MMA boundary.

    Per-expert global scales are deliberately not folded into this tensor.
    The kernel multiplies them into the FP32 FC1/FC2 accumulator, after the
    MMA, which is not numerically equivalent to scaling and re-rounding the
    BF16 operand before the MMA.
    """
    if weight.dtype is not torch.float4_e2m1fn_x2:
        raise TypeError(f"weight must be NVFP4, got {weight.dtype}.")
    if weight_scale.dtype is not torch.float8_e4m3fn:
        raise TypeError(f"weight_scale must be E4M3FN, got {weight_scale.dtype}.")
    decoded = _unpack_fp4_to_f32(weight, packed_dim=-1)
    if decoded.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("weight and weight_scale must be rank 2.")
    n, k = decoded.shape
    expected = (n, k // NVFP4_BLOCK_SIZE)
    if k <= 0 or k % NVFP4_BLOCK_SIZE or tuple(weight_scale.shape) != expected:
        raise ValueError(
            f"expected K divisible by {NVFP4_BLOCK_SIZE} and scale shape "
            f"{expected}, got weight {tuple(decoded.shape)} and "
            f"scale {tuple(weight_scale.shape)}."
        )
    expanded = weight_scale.to(torch.float32).repeat_interleave(
        NVFP4_BLOCK_SIZE, dim=1
    )
    return (decoded * expanded).to(torch.bfloat16)


def nvfp4_weight_from_swizzled_to_bf16(
    weight_kn: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
) -> torch.Tensor:
    """Decode one K-major packed runner weight to logical BF16 ``(N,K)``."""
    if weight_kn.ndim != 2:
        raise ValueError("weight_kn must be rank 2.")
    decoded_kn = _unpack_fp4_to_f32(weight_kn, packed_dim=0)
    reduction_size, output_size = decoded_kn.shape
    raw_scale = _from_blocked(
        weight_scale_swizzled.contiguous().view(-1),
        output_size,
        reduction_size // NVFP4_BLOCK_SIZE,
    )
    expanded = raw_scale.to(torch.float32).repeat_interleave(
        NVFP4_BLOCK_SIZE, dim=1
    )
    return (decoded_kn.transpose(0, 1) * expanded).to(torch.bfloat16)


def compute_megamoe_reference_nvfp4_bf16(
    input_activation: torch.Tensor,
    input_topk_idx: torch.Tensor,
    input_topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    ref_compute_graph: Literal["transformers", "deepgemm"],
    fc2_output_dtype: torch.dtype = torch.bfloat16,
    gate_up_clamp: Optional[float] = None,
    apply_topk_in_fc1: bool = False,
    return_fc1_gateup: bool = False,
    fc1_alpha: Optional[torch.Tensor] = None,
    fc2_alpha: Optional[torch.Tensor] = None,
):
    """Decode weight banks, then delegate routing and dense BF16 GEMMs."""
    ranks, experts = fc1_weight.shape[:2]
    decoded_fc1 = []
    decoded_fc2 = []
    for rank in range(ranks):
        rank_fc1 = []
        rank_fc2 = []
        for expert in range(experts):
            rank_fc1.append(
                nvfp4_weight_from_swizzled_to_bf16(
                    fc1_weight[rank, expert],
                    fc1_weight_sf[rank, expert],
                ).transpose(0, 1)
            )
            rank_fc2.append(
                nvfp4_weight_from_swizzled_to_bf16(
                    fc2_weight[rank, expert],
                    fc2_weight_sf[rank, expert],
                ).transpose(0, 1)
            )
        decoded_fc1.append(torch.stack(rank_fc1))
        decoded_fc2.append(torch.stack(rank_fc2))

    from moe_bf16_glu.mega_reference_bf16 import compute_megamoe_reference

    return compute_megamoe_reference(
        input_activation=input_activation,
        input_topk_idx=input_topk_idx,
        input_topk_weights=input_topk_weights,
        fc1_weight=torch.stack(decoded_fc1),
        fc2_weight=torch.stack(decoded_fc2),
        ref_compute_graph=ref_compute_graph,
        fc2_output_dtype=fc2_output_dtype,
        gate_up_clamp=gate_up_clamp,
        apply_topk_in_fc1=apply_topk_in_fc1,
        return_fc1_gateup=return_fc1_gateup,
        gate_up_interleave=NVFP4_GATE_UP_INTERLEAVE,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
    )


__all__ = [
    "NVFP4_BLOCK_SIZE",
    "NVFP4_GATE_UP_INTERLEAVE",
    "compute_megamoe_reference_nvfp4_bf16",
    "nvfp4_weight_from_swizzled_to_bf16",
    "nvfp4_weight_to_bf16",
]
