# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Frost WOA decode with FP8 weight storage and source BF16 arithmetic."""

import torch
import triton as tr
import triton.language as tl

from ._utils import _overlaps


@tr.jit
def _frost_woa(X, W, S, Y, ROWS: tl.constexpr, BK: tl.constexpr):
    group = tl.program_id(1)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, BK)
    global_rows = group * 1024 + rows
    acc = tl.zeros((ROWS, BK), tl.float32)
    for start in range(0, 4096, BK):
        columns = start + k
        x = tl.load(X + group * 4096 + columns).to(tl.float32)
        w = tl.load(W + global_rows[:, None] * 4096 + columns[None, :]).to(tl.float32)
        code = tl.load(
            S + (global_rows[:, None] // 32) * 128 + columns[None, :] // 32
        ).to(tl.uint32)
        scale = (code << 23).to(tl.float32, bitcast=True)
        source_weight = (w * scale).to(tl.bfloat16).to(tl.float32)
        acc = tl.fma(x[None, :], source_weight, acc)
    tl.store(Y + group * 1024 + rows, tl.sum(acc, 1))


class _FrostWoaPlan:
    def __init__(self, weight, scales):
        if (
            weight.shape != (8192, 4096)
            or weight.dtype != torch.float8_e4m3fn
            or not weight.is_contiguous()
            or weight.device.type != "cuda"
        ):
            raise ValueError("WOA weight must be contiguous CUDA E4M3 [8192,4096]")
        if (
            scales.shape != (256, 128)
            or scales.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", None))
            or scales.device != weight.device
            or not scales.is_contiguous()
        ):
            raise ValueError(
                "WOA scales must be contiguous E8M0 bytes [256,128] on the weight device"
            )
        if torch.cuda.get_device_capability(weight.device) != (10, 0):
            raise ValueError("Frost WOA currently requires SM100")
        with torch.cuda.device(weight.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare the WOA plan outside CUDA Graph capture")
            codes = scales.view(torch.uint8)
            if not bool(((codes >= 1) & (codes <= 254)).all()):
                raise ValueError("Frost WOA requires E8M0 scale codes 1..254")
        # Retain the original allocations without attaching an autograd path.
        # The caller keeps weight values, scale values, and storage immutable.
        self.weight = weight.detach()
        self.scales = codes.detach()

    def run(self, x, out):
        if (
            x.shape != (1, 8, 4096)
            or x.dtype != torch.bfloat16
            or x.device != self.weight.device
            or not x.is_contiguous()
        ):
            raise ValueError(
                "WOA input must be contiguous BF16 [1,8,4096] on the weight device"
            )
        if out is None:
            out = torch.empty((1, 8, 1024), device=x.device, dtype=x.dtype)
        if (
            out.shape != (1, 8, 1024)
            or out.dtype != x.dtype
            or out.device != x.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "WOA output must be contiguous BF16 [1,8,1024] on the input device"
            )
        if any(_overlaps(out, tensor) for tensor in (x, self.weight, self.scales)):
            raise ValueError(
                "WOA output must not overlap inputs or plan weight storage"
            )
        # Four rows per CTA keep this single-token projection on the bandwidth
        # path. Decode each stored weight to BF16 before the FP32 reduction.
        with torch.cuda.device(x.device):
            _frost_woa[(256, 8)](
                x,
                self.weight,
                self.scales,
                out,
                4,
                4096,
                num_warps=4,
                num_stages=3,
            )
        return out


def make_woa_plan(weight, scales, *, backend):
    if backend != "frost":
        raise ValueError("WOA currently supports only backend='frost'")
    return _FrostWoaPlan(weight, scales)


def woa(x, plan, *, out):
    if not isinstance(plan, _FrostWoaPlan):
        raise TypeError("WOA requires a plan returned by deepseek_v41_woa_plan")
    return plan.run(x, out)
