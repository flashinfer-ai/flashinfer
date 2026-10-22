# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""RoPE table helper: a serving stack's ``positions`` + ``cos_sin_cache`` -> the per-token ``cos`` / ``sin`` tables the kernels read."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def qwen_3_5_block_rope_tables(
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    out_cos: Optional[torch.Tensor] = None,
    out_sin: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """See :func:`flashinfer.qwen_3_5_block.qwen_3_5_block_rope_tables` (the public, decorated entry)."""
    if not isinstance(positions, torch.Tensor) or positions.dim() != 1:
        raise ValueError("positions must be a 1-D integer tensor [num_tokens]")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"positions must be int32 or int64, got {positions.dtype}")
    if (
        not isinstance(cos_sin_cache, torch.Tensor)
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[1] % 2
    ):
        raise ValueError(
            "cos_sin_cache must be [max_position, rotary_dim] with the cos half then the sin half (rotary_dim even)"
        )
    if cos_sin_cache.device != positions.device:
        raise ValueError(
            f"positions ({positions.device}) and cos_sin_cache ({cos_sin_cache.device}) must share a device"
        )
    half = cos_sin_cache.shape[1] // 2
    rows = cos_sin_cache.index_select(
        0, positions if positions.dtype == torch.int64 else positions.to(torch.int64)
    )
    c, s = rows[:, :half], rows[:, half:]
    for name, out in (("out_cos", out_cos), ("out_sin", out_sin)):
        if out is not None and (
            tuple(out.shape) != (positions.shape[0], 2 * half)
            or out.device != positions.device
        ):
            raise ValueError(
                f"{name} must be [num_tokens={positions.shape[0]}, rope_dim={2 * half}] on {positions.device}"
            )
    cos = (
        torch.cat((c, c), dim=-1, out=out_cos)
        if out_cos is not None
        else torch.cat((c, c), dim=-1)
    )
    sin = (
        torch.cat((s, s), dim=-1, out=out_sin)
        if out_sin is not None
        else torch.cat((s, s), dim=-1)
    )
    return cos, sin
