# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test-only torch quantizer reproducing the ``sageQuant`` recipe.

Q and K are quantized per token block with an absmax scale written in the
trtllm-gen flat scale layout; V is quantized per channel over every token and
batch. FlashInfer ships no quantization kernel, so these helpers exist to
drive the Sage attention kernels from BF16/FP16 inputs in tests only.
"""

from __future__ import annotations

import torch

from flashinfer.attention.prims_ts.sage import (
    flat_scale_numel,
    flat_scale_slot,
    log2_block_size,
)

INT8_TYPE_MAX = 126.9
E4M3_TYPE_MAX = 448.0
ABSMAX_FLOOR = 1e-3


def _type_max(dtype: torch.dtype) -> float:
    if dtype == torch.int8:
        return INT8_TYPE_MAX
    if dtype == torch.float8_e4m3fn:
        return E4M3_TYPE_MAX
    raise ValueError(f"unsupported quantized dtype {dtype}")


def _quantize_with_scale(x: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    """Quantize ``x / scale`` with saturation; ``scale`` broadcasts over ``x``."""

    scaled = x.float() / scale
    type_max = _type_max(dtype)
    if dtype == torch.int8:
        # TypeMax 126.9 leaves headroom so the rounded value stays in int8.
        return torch.round(scaled).clamp(-127, 127).to(torch.int8)
    return scaled.clamp(-type_max, type_max).to(dtype)


def quantize_token_blocks(
    x: torch.Tensor,
    *,
    block_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[B, S, H, D]`` per ``block_size`` tokens of every head.

    Returns the quantized tensor and the fp32 flat-layout scales
    ``[H, flat_scale_numel(B, S, block_size)]``.
    """

    if x.ndim != 4:
        raise ValueError("expected a [B, S, H, D] tensor")
    batch_size, seq_len, num_heads, _ = x.shape
    type_max = _type_max(dtype)
    log2_block = log2_block_size(block_size)
    scales = torch.ones(
        (num_heads, flat_scale_numel(batch_size, seq_len, block_size)),
        dtype=torch.float32,
        device=x.device,
    )
    quantized = torch.empty(x.shape, dtype=dtype, device=x.device)
    for batch_idx in range(batch_size):
        for block_begin in range(0, seq_len, block_size):
            block_end = min(block_begin + block_size, seq_len)
            block = x[batch_idx, block_begin:block_end].float()
            absmax = block.abs().amax(dim=(0, 2)).clamp_min(ABSMAX_FLOOR)
            scale = absmax / type_max
            flat_idx = flat_scale_slot(batch_idx, block_begin, seq_len, log2_block)
            scales[:, flat_idx] = scale
            quantized[batch_idx, block_begin:block_end] = _quantize_with_scale(
                block, scale[None, :, None], dtype
            )
    return quantized, scales


def dequantize_token_blocks(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Invert :func:`quantize_token_blocks` into fp32."""

    batch_size, seq_len, num_heads, _ = quantized.shape
    log2_block = log2_block_size(block_size)
    out = torch.empty(quantized.shape, dtype=torch.float32, device=quantized.device)
    for batch_idx in range(batch_size):
        for block_begin in range(0, seq_len, block_size):
            block_end = min(block_begin + block_size, seq_len)
            flat_idx = flat_scale_slot(batch_idx, block_begin, seq_len, log2_block)
            scale = scales[:, flat_idx]
            out[batch_idx, block_begin:block_end] = (
                quantized[batch_idx, block_begin:block_end].float()
                * scale[None, :, None]
            )
    return out


def quantize_v_channels(
    v: torch.Tensor,
    *,
    smooth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize ``[B, S, Hkv, D]`` V to E4M3 with one fp32 scale per channel.

    With ``smooth`` the per-channel mean over every token and batch is removed
    before quantization and returned as ``v_mean`` ``[Hkv, D]``.
    """

    if v.ndim != 4:
        raise ValueError("expected a [B, S, Hkv, D] tensor")
    values = v.float()
    v_mean = None
    if smooth:
        v_mean = values.mean(dim=(0, 1))
        values = values - v_mean[None, None]
    absmax = values.abs().amax(dim=(0, 1)).clamp_min(ABSMAX_FLOOR)
    v_scale = absmax / E4M3_TYPE_MAX
    quantized = _quantize_with_scale(values, v_scale[None, None], torch.float8_e4m3fn)
    return quantized, v_scale, v_mean


def dequantize_v_channels(
    quantized: torch.Tensor,
    v_scale: torch.Tensor,
    v_mean: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert :func:`quantize_v_channels` into fp32."""

    values = quantized.float() * v_scale[None, None]
    if v_mean is not None:
        values = values + v_mean[None, None]
    return values
