# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Validation and data movement shared by the checkpoint-layer weight packers.

A contract maps each required checkpoint tensor name to its full shape and the
dtypes accepted for it. Packers validate every required entry before they
allocate anything large, then move only the rank-local slices to the target
device. FP8 codes always move as ``uint8`` so that no FP8 copy kernel is needed
and every bit pattern is preserved.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

FP8 = torch.float8_e4m3fn
F32 = torch.float32
BF16 = torch.bfloat16
NORM = (BF16, F32)  # 1-D norm weights and biases, widened exactly to FP32

_MISSING_LISTED = 8


def validate_layer(weights, contract) -> torch.device:
    """Check the required entries of a layer mapping; return their common device.

    Entries absent from ``contract`` are ignored. Raises ``ValueError`` naming
    missing entries or a wrong shape, layout or device, and ``TypeError`` for a
    non-tensor entry or a wrong dtype. Nothing is allocated.
    """
    if not isinstance(weights, Mapping):
        raise TypeError(
            "weights must be a mapping from checkpoint tensor names to tensors"
        )
    missing = [name for name in contract if name not in weights]
    if missing:
        listed = ", ".join(missing[:_MISSING_LISTED])
        if len(missing) > _MISSING_LISTED:
            listed += f", ... ({len(missing)} missing in total)"
        raise ValueError(f"missing checkpoint tensors: {listed}")
    device = None
    for name, (shape, dtypes) in contract.items():
        tensor = weights[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must use the torch.strided layout")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype not in dtypes:
            expected = " or ".join(str(dtype) for dtype in dtypes)
            raise TypeError(f"{name} must have dtype {expected}, got {tensor.dtype}")
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name}: weight packing supports CPU or CUDA tensors")
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    return device


def target_device(device) -> torch.device:
    """Normalize the ``device`` argument of a packer (not ``None``)."""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA device")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def check_not_capturing(function: str, *devices: torch.device) -> None:
    for device in devices:
        if device.type == "cuda":
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        f"{function} must run outside CUDA graph capture"
                    )


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return ``tensor`` on ``device``; a no-op view when it is already there."""
    if tensor.dtype == FP8:
        return tensor.view(torch.uint8).to(device).view(FP8)
    return tensor.to(device)


def owned(
    tensor: torch.Tensor, device: torch.device, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Copy ``tensor`` into new contiguous storage on ``device``.

    A ``dtype`` conversion is used only to widen BF16 to FP32, which is exact.
    """
    result = torch.empty(
        tuple(tensor.shape), dtype=dtype or tensor.dtype, device=device
    )
    result.copy_(tensor.detach())
    return result


def narrow_padded(
    tensor: torch.Tensor, dim: int, start: int, length: int, size: int
) -> torch.Tensor:
    """Select ``length`` entries along ``dim``, zero-padded to ``size``.

    Returns a view when no padding is needed and a new tensor otherwise.
    """
    piece = tensor.narrow(dim, start, length)
    if length == size:
        return piece
    shape = list(tensor.shape)
    shape[dim] = size
    if tensor.dtype == FP8:
        result = torch.zeros(shape, dtype=torch.uint8, device=tensor.device)
        result.narrow(dim, 0, length).copy_(piece.view(torch.uint8))
        return result.view(FP8)
    result = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    result.narrow(dim, 0, length).copy_(piece)
    return result
