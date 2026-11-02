# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Independent single-GPU fixtures for the MoK kernel tests.

The MXFP8 model follows the MoK recipe without importing the package: 32
consecutive K elements share a UE8M0 byte ``round_up(max(amax * 0.002232142857,
1e-12))`` saturated to 254, data is E4M3 ``x * 2**(127 - byte)`` (round to
nearest even, saturating), and 128 x 128 scale tiles store byte
``(row % 32) * 16 + (row // 32) * 4 + k_block`` as ``[R/128, K/128, 32, 16]``.
"""

import pytest
import torch

SUPPORTED = ((10, 0), (10, 3), (10, 7))


def require_gpu():
    if not torch.cuda.is_available() or (
        torch.cuda.get_device_capability() not in SUPPORTED
    ):
        pytest.skip("Requires an SM100a, SM103a or SM107a CUDA device")


def single_rank_schedule(routes, experts, extra=512):
    """Single-rank padded schedule: per expert its route slots, then -1 padding."""
    slots = [torch.where(routes.flatten() == e)[0] for e in range(experts)]
    count_values = [((s.numel() + 255) // 256) * 256 for s in slots]
    pieces = [
        torch.cat(
            (
                s.int(),
                torch.full(
                    (c - s.numel(),), -1, dtype=torch.int32, device=routes.device
                ),
            )
        )
        for s, c in zip(slots, count_values, strict=True)
    ]
    schedule = torch.cat(pieces)
    actual = schedule.numel()
    schedule = torch.cat(
        (schedule, torch.full((extra,), -1, dtype=torch.int32, device=routes.device))
    )
    peers = torch.where(schedule >= 0, 0, -1).int()
    num_tokens = torch.tensor([actual], device=routes.device, dtype=torch.int32)
    counts = torch.tensor(count_values, device=routes.device, dtype=torch.int32)
    return dict(
        slots=slots,
        count_values=count_values,
        schedule=schedule,
        actual=actual,
        peers=peers,
        num_tokens=num_tokens,
        counts=counts,
    )


def _e8m0_round_up(scale):
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return (exponent + (mantissa != 0).to(torch.int32)).clamp(max=254).to(torch.uint8)


def _quantize_rows(x):
    *lead, rows, cols = x.shape
    blocks = x.float().reshape(*lead, rows, cols // 32, 32)
    amax = blocks.abs().amax(-1)
    scale = torch.clamp(
        amax * torch.tensor(0.002232142857, dtype=torch.float32, device=x.device),
        min=1e-12,
    )
    byte = _e8m0_round_up(scale)
    inverse = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    data = (blocks * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return data.reshape(*lead, rows, cols), byte


def _scale_tiles(byte):
    *lead, rows, kb = byte.shape
    tiles = byte.reshape(-1, rows // 128, 4, 32, kb // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return tiles.reshape(-1, kb // 4, 32, 16).contiguous()


def mxfp8_quantize(x, normal=True, transposed=True):
    """``(fp8, scales, fp8_t, scales_t)`` of a BF16 ``[..., R, K]`` tensor."""
    out = [None] * 4
    if normal:
        data, byte = _quantize_rows(x)
        out[0], out[1] = data, _scale_tiles(byte)
    if transposed:
        data, byte = _quantize_rows(x.transpose(-1, -2).contiguous())
        out[2], out[3] = data, _scale_tiles(byte)
    return tuple(out)


def mxfp8_dequantize(data, scales):
    """FP32 values of E4M3 data and its tile-layout UE8M0 scales."""
    *lead, rows, cols = data.shape
    s = scales.reshape(-1, rows // 128, cols // 128, 32, 4, 4).permute(0, 1, 4, 3, 2, 5)
    byte = s.reshape(*lead, rows, cols // 32).to(torch.int32)
    factor = (byte << 23).view(torch.float32)
    values = data.float().reshape(*lead, rows, cols // 32, 32) * factor[..., None]
    return values.reshape(*lead, rows, cols)


def swiglu(gate, up, limit=None):
    """BF16-rounded SwiGLU on FP32-promoted inputs; optional clamp limit."""
    gate, up = gate.float(), up.float()
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return (gate * torch.sigmoid(gate) * up).bfloat16()
