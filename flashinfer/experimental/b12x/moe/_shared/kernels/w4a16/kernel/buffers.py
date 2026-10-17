"""Reusable W4A16 MoE buffers."""

from __future__ import annotations

import torch


_ROT_SCALES_DUMMY: dict[object, torch.Tensor] = {}


def _rot_scales_dummy(device: torch.device) -> torch.Tensor:
    """Tiny fp16 placeholder for the rot_scales kernel slot when the
    intermediate-rotation epilogue is disabled (never dereferenced by the
    const_expr-gated kernel; keeps one compiled signature across all layouts)."""
    t = _ROT_SCALES_DUMMY.get(device)
    if t is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "W4A16 rotation placeholder is not initialized for CUDA graph "
                "capture; prewarm the planned launch before capturing"
            )
        t = torch.zeros(1, dtype=torch.float16, device=device)
        _ROT_SCALES_DUMMY[device] = t
    return t


def _get_c_tmp(
    elements: int,
    *,
    device: torch.device,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    if scratch is not None:
        if scratch.dtype != torch.float32:
            raise TypeError("W4A16 c_tmp scratch buffers must be torch.float32")
        if scratch.device != device:
            raise ValueError(f"W4A16 c_tmp scratch buffers must be on {device}")
        if not scratch.is_contiguous():
            raise ValueError("W4A16 c_tmp scratch buffers must be contiguous")
        if int(scratch.numel()) >= int(elements):
            return scratch[: int(elements)]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "W4A16 GEMM scratch is not initialized for CUDA graph capture; "
            "provide a preallocated fc*_c_tmp workspace with sufficient capacity"
        )
    return torch.empty((elements,), dtype=torch.float32, device=device)
