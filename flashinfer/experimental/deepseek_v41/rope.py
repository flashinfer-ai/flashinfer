# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""BF16 adjacent-pair rotary encoding for query and attention-output heads."""

import torch
import triton as tr
import triton.language as tl

from .quantization import _rotate_pair
from ._utils import _overlaps


@tr.jit
def _rope(
    X,
    FREQ,
    POS,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    X_STRIDE: tl.constexpr,
    SEQUENCE: tl.constexpr,
    INVERSE: tl.constexpr,
):
    row = tl.program_id(0) * 4 + tl.arange(0, 4)
    pair = tl.arange(0, D // 2)
    position = tl.load(POS + row // H, row < T * H, -1)
    valid = (row < T * H) & (position >= 0) & (position < SEQUENCE)
    address = row[:, None].to(tl.int64) * D + pair[None, :] * 2
    source = (
        (row // H)[:, None].to(tl.int64) * X_STRIDE
        + (row % H)[:, None] * D
        + pair[None, :] * 2
    )
    a = tl.load(X + source, valid[:, None], 0).to(tl.float32)
    b = tl.load(X + source + 1, valid[:, None], 0).to(tl.float32)
    a, b = _rotate_pair(a, b, FREQ, position, valid, D, INVERSE)
    tl.store(OUT + address, a, valid[:, None])
    tl.store(OUT + address + 1, b, valid[:, None])


def rope(x, freqs, positions, *, inverse=False, out=None):
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("V4.1 RoPE currently requires SM100/SM103")
    if (
        x.ndim != 3
        or x.shape[1] < 1
        or x.shape[2] not in (128, 512)
        or x.dtype != torch.bfloat16
        or x.stride(2) != 1
        or x.stride(1) != x.shape[2]
        or x.stride(0) < x.shape[1] * x.shape[2]
    ):
        raise ValueError(
            "RoPE requires BF16 [tokens,heads,128|512], dense heads and nonoverlapping tokens"
        )
    tokens, heads, dim = x.shape
    if (
        freqs.ndim != 3
        or freqs.shape[0] < 1
        or freqs.shape[1:] != (32, 2)
        or freqs.dtype != torch.float32
        or freqs.device != x.device
        or not freqs.is_contiguous()
    ):
        raise ValueError(
            "freqs must be contiguous FP32 [sequence,32,2] on the input device"
        )
    if (
        positions.shape != (tokens,)
        or positions.dtype != torch.int32
        or positions.device != x.device
        or not positions.is_contiguous()
    ):
        raise ValueError("positions must be contiguous CUDA int32 [tokens]")
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    elif (
        out.shape != x.shape
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "output must match the input shape/dtype/device and be contiguous"
        )
    if any(t.requires_grad for t in (x, freqs, out)) and torch.is_grad_enabled():
        raise ValueError("RoPE primitive is inference-only")
    if tokens and (
        _overlaps(out, freqs)
        or _overlaps(out, positions)
        or (
            _overlaps(out, x)
            and (out.data_ptr() != x.data_ptr() or out.stride() != x.stride())
        )
    ):
        raise ValueError("output may only alias the identical input view")
    if tokens:
        with torch.cuda.device(x.device):
            _rope[(tr.cdiv(tokens * heads, 4),)](
                x,
                freqs,
                positions,
                out,
                tokens,
                heads,
                dim,
                x.stride(0),
                freqs.shape[0],
                inverse,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out
