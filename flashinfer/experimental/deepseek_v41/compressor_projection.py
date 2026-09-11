# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Small-batch H5120 -> two D512 FP32 projections using TF32x3."""

import torch
import triton as tr
import triton.language as tl

from ._utils import _overlaps


@tr.jit
def _partials(X, WKV, WGATE, PARTIALS, B: tl.constexpr):
    tile, split = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, 16)
    columns = (tile % 8) * 64 + tl.arange(0, 64)
    reduction = tl.arange(0, 128)
    weight = tl.where(tile < 8, WKV, WGATE)
    accumulator = tl.full((16, 64), 0, tl.float32)
    for step in range(5):
        k = split * 640 + step * 128 + reduction
        x = tl.load(X + rows[:, None] * 5120 + k[None, :], rows[:, None] < B, 0).to(
            tl.float32
        )
        w = tl.load(weight + columns[None, :] * 5120 + k[:, None])
        accumulator = tl.dot(x, w, accumulator, input_precision="tf32x3")
    address = ((split * 2 + tile // 8) * B + rows[:, None]) * 512 + columns[None, :]
    tl.store(PARTIALS + address, accumulator, rows[:, None] < B)


@tr.jit
def _reduce(PARTIALS, KV, SCORE, B: tl.constexpr):
    at = tl.program_id(0) * 256 + tl.arange(0, 256)
    split = tl.arange(0, 8)
    value = tl.load(PARTIALS + split[:, None] * (2 * B * 512) + at[None, :])
    output = tl.sum(value, 0)
    tl.store(KV + at, output, at < B * 512)
    tl.store(SCORE + at - B * 512, output, at >= B * 512)


def compressor_projection(x, wkv, wgate, *, workspace=None, kv=None, score=None):
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("V4.1 compressor projection currently requires SM100/SM103")
    if x.ndim != 2 or not 1 <= x.shape[0] <= 16 or x.shape[1] != 5120:
        raise ValueError("projection requires BF16 [B,5120] with B in1..16")
    batch = x.shape[0]
    for tensor, shape, dtype in (
        (x, (batch, 5120), torch.bfloat16),
        (wkv, (512, 5120), torch.float32),
        (wgate, (512, 5120), torch.float32),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("projection input declaration mismatch")
        if tensor.requires_grad and torch.is_grad_enabled():
            raise ValueError("small compressor projection is inference-only")
    outputs = []
    for tensor, shape in (
        (workspace, (8, 2, batch, 512)),
        (kv, (batch, 512)),
        (score, (batch, 512)),
    ):
        if tensor is None:
            tensor = torch.empty(shape, device=x.device, dtype=torch.float32)
        elif (
            tensor.shape != shape
            or tensor.dtype != torch.float32
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("projection output/workspace declaration mismatch")
        if any(_overlaps(tensor, source) for source in (x, wkv, wgate, *outputs)):
            raise ValueError(
                "projection output/workspace may not overlap other buffers"
            )
        outputs.append(tensor)
    workspace, kv, score = outputs
    with torch.cuda.device(x.device):
        _partials[(16, 8)](x, wkv, wgate, workspace, batch, num_warps=4, num_stages=3)
        _reduce[(2 * batch * 512 // 256,)](workspace, kv, score, batch, num_warps=4)
    return kv, score
