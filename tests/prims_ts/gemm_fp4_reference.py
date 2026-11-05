# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reference helpers for dense NVFP4 GEMM tests.

The public API consumes 128x4 scale buffers. These helpers build that layout
from logical scales and decode both layouts for numeric checks.
"""

from __future__ import annotations

import torch

FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def unpack_nvfp4(packed: torch.Tensor) -> torch.Tensor:
    """Decode rank-2 row-major packed E2M1 values to FP32."""
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed NVFP4 must be a rank-2 uint8 tensor")
    lut = torch.tensor(FP4_LUT, dtype=torch.float32, device=packed.device)

    def decode(nibbles: torch.Tensor) -> torch.Tensor:
        values = lut[(nibbles & 7).to(torch.long)]
        return torch.where((nibbles & 8) != 0, -values, values)

    return torch.stack((decode(packed & 15), decode((packed >> 4) & 15)), -1).reshape(
        packed.shape[0], packed.shape[1] * 2
    )


def linear_scale_to_128x4(scales: torch.Tensor) -> torch.Tensor:
    """Convert logical row-by-K/16 E4M3 scales to the kernel's 128x4 layout."""
    if scales.ndim != 2:
        raise ValueError("NVFP4 scales must be rank 2")
    rows, cols = scales.shape
    rows_pad, cols_pad = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    result = torch.zeros(
        (rows_pad * cols_pad,), device=scales.device, dtype=torch.uint8
    )
    source = scales.view(torch.uint8) if scales.dtype != torch.uint8 else scales
    row = torch.arange(rows, device=scales.device, dtype=torch.int64)[:, None]
    col = torch.arange(cols, device=scales.device, dtype=torch.int64)[None, :]
    idx = (
        col % 4
        + (col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * (128 * cols_pad)
    )
    result[idx.reshape(-1)] = source.reshape(-1)
    return result


def dequantize_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor | float
) -> torch.Tensor:
    """Reference decode for logical linear E4M3 scales."""
    values = unpack_nvfp4(packed)
    scale = (
        scales.view(torch.float8_e4m3fn).float()
        if scales.dtype == torch.uint8
        else scales.float()
    )
    return (
        values
        * scale.repeat_interleave(16, dim=1)
        * torch.as_tensor(global_scale, device=packed.device, dtype=torch.float32)
    )


def dequantize_nvfp4_128x4(
    packed: torch.Tensor,
    scales_128x4: torch.Tensor,
    output_encode_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Decode the kernel's packed output and 128x4 E4M3 scales."""
    m, packed_n = packed.shape
    sf_cols = packed_n * 2 // 16
    padded_sf_cols = (sf_cols + 3) // 4 * 4
    rows = torch.arange(m, device=packed.device, dtype=torch.int64)[:, None]
    cols = torch.arange(sf_cols, device=packed.device, dtype=torch.int64)[None, :]
    idx = (
        cols % 4
        + (cols // 4) * 512
        + (rows % 32) * 16
        + ((rows % 128) // 32) * 4
        + (rows // 128) * (128 * padded_sf_cols)
    )
    scale = scales_128x4.reshape(-1)[idx].view(torch.float8_e4m3fn).float()
    return (
        unpack_nvfp4(packed)
        * scale.repeat_interleave(16, dim=1)
        / torch.as_tensor(
            output_encode_scale, device=packed.device, dtype=torch.float32
        )
    )
