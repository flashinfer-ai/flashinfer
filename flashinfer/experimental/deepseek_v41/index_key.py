# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Small-batch BF16 index-key projection, BF16 boundary and FP32 RMSNorm."""

import math

import torch
import triton as tr
import triton.language as tl

from ._utils import _overlaps


@tr.jit
def _key(X, W, NORM, OUT, B: tl.constexpr, EPS: tl.constexpr):
    row = tl.arange(0, 16)
    col = tl.arange(0, 128)
    k = tl.arange(0, 128)
    accumulator = tl.full((16, 128), 0, tl.float32)
    for base in range(4):
        reduction = base * 128 + k
        x = tl.load(X + row[:, None] * 512 + reduction[None, :], row[:, None] < B, 0)
        weight = tl.load(W + col[None, :] * 512 + reduction[:, None])
        accumulator = tl.dot(x, weight, accumulator)
    # The projection's BF16 output is a required numerical boundary.
    projected = accumulator.to(tl.bfloat16).to(tl.float32)
    reciprocal = tl.rsqrt(tl.sum(projected * projected, 1) / 128 + EPS)
    norm = tl.load(NORM + col).to(tl.float32)
    result = (projected * reciprocal[:, None] * norm[None, :]).to(tl.bfloat16)
    tl.store(OUT + row[:, None] * 128 + col[None, :], result, row[:, None] < B)


def index_key(x, weight, norm_weight, *, eps=1e-20, out=None):
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("V4.1 index-key projection currently requires SM100/SM103")
    if x.ndim != 2 or not 1 <= x.shape[0] <= 16 or x.shape[1] != 512:
        raise ValueError("index-key projection requires BF16 [B,512], B1..16")
    batch = x.shape[0]
    for tensor, shape, dtypes in (
        (x, (batch, 512), (torch.bfloat16,)),
        (weight, (128, 512), (torch.bfloat16,)),
        (norm_weight, (128,), (torch.bfloat16, torch.float32)),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype not in dtypes
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("index-key input declaration mismatch")
        if tensor.requires_grad and torch.is_grad_enabled():
            raise ValueError("index-key projection is inference-only")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive finite")
    if out is None:
        out = torch.empty(batch, 128, device=x.device, dtype=torch.bfloat16)
    elif (
        out.shape != (batch, 128)
        or out.dtype != torch.bfloat16
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("index-key output declaration mismatch")
    if any(_overlaps(out, source) for source in (x, weight, norm_weight)):
        raise ValueError("index-key output may not overlap inputs")
    with torch.cuda.device(x.device):
        _key[(1,)](
            x, weight, norm_weight, out, batch, eps, num_warps=4, enable_fp_fusion=False
        )
    return out
