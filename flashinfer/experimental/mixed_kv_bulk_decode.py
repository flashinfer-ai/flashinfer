# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Paged FP8 MMA decode with compressed V bulk-copied into CTA shared memory."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(
    Q,
    PTR,
    LENS,
    Q4,
    L4,
    OFF,
    M,
    ts,
    qs0,
    qs1,
    H: tl.constexpr,
    D: tl.constexpr,
    RESET_M: tl.constexpr,
):
    r, h = tl.program_id(0), tl.program_id(1)
    t = tl.load(PTR + r)
    use = t < tl.load(PTR + r + 1)
    x = tl.arange(0, D)
    val = tl.load(Q + t.to(tl.int64) * qs0 + h * qs1 + x, use, other=0)
    tl.store(Q4 + (r * H + h) * D + x, val)
    if h == 0:
        tl.store(L4 + r, tl.where(use, tl.load(LENS + r), 0))
        tl.store(OFF + r, r * ts)
    if RESET_M:
        tl.store(M + r * H + h, -float("inf"))


@triton.jit
def _scatter(O4, PTR, LENS, OUT, os0, os1, H: tl.constexpr, D: tl.constexpr):
    r, h = tl.program_id(0), tl.program_id(1)
    t = tl.load(PTR + r)
    if t < tl.load(PTR + r + 1):
        x = tl.arange(0, D)
        y = tl.load(O4 + (r * H + h) * D + x, tl.load(LENS + r) > 0, other=0.0)
        tl.store(OUT + t.to(tl.int64) * os0 + h * os1 + x, y)


def supported(q, k, v, sf, table):
    b, width = table.shape
    return (
        torch.cuda.get_device_capability(q.device) == (10, 3)
        and 8 <= b <= 64
        and 32 < width <= 1024
        and k.shape[1] in (2, 8)
        and k.shape[2:] == (64, 128)
        and q.shape[1:] == (4 * k.shape[1], 128)
        and table.stride(1) == 1
        and width <= table.stride(0) <= (2**31 - 1) // b
        and table.storage_offset() % 4 == 0
        and k.stride(2) == 128
        and v.stride(2) == 64
        and sf.stride(2) == 8
        and all(
            t.stride(-1) == 1
            and t.storage_offset() * t.element_size() % 16 == 0
            and (t.stride(0) * t.element_size()) % 16 == 0
            and (t.stride(1) * t.element_size()) % 16 == 0
            for t in (k, v, sf)
        )
    )


def run(q, k, v, sf, table, lens, ptr, out, scale, vscale):
    from cutlass.cute.typing import Float32
    from flashinfer.cute_dsl.attention.wrappers.batch_decode import (
        _get_compiled_paged_decode_kernel,
        _get_mask_config,
        _slice_workspace,
        _torch_to_cutlass,
    )

    b, w = table.shape
    h, d = q.shape[1:]
    hk = k.shape[1]
    s = 2 if w > 256 else min(4, max(1, 144 // (b * hk)))
    mode = "none" if s == 1 else "kernel"
    kernel = _get_compiled_paged_decode_kernel(
        _torch_to_cutlass(q.dtype),
        _torch_to_cutlass(out.dtype),
        64,
        h,
        hk,
        d,
        1,
        mode,
        None,
        None,
        False,
        False,
        False,
        True,
    )
    q4 = torch.empty((b, 1, h, d), device=q.device, dtype=q.dtype)
    o4 = torch.empty_like(q4, dtype=out.dtype)
    l4, off = torch.empty_like(lens), torch.empty_like(lens)
    if s > 1:
        ws = torch.empty(
            b * h * (s * (d + 2) + 1), device=q.device, dtype=torch.float32
        )
        parts = _slice_workspace(ws, s, b, 1, h, d)
    else:
        parts = (None,) * 4
    # A captured graph may have empty requests anywhere, not just at the end.
    _gather[(b, h)](
        q.view(torch.uint8),
        ptr,
        lens,
        q4.view(torch.uint8),
        l4,
        off,
        parts[0] if s > 1 else off,
        table.stride(0),
        *q.stride()[:2],
        h,
        d,
        s > 1,
    )
    flat = table.as_strided(((b - 1) * table.stride(0) + w,), (1,))
    kb = k.transpose(1, 2)
    # The homogeneous ABI has a V descriptor. It is unused on this schedule;
    # the two final operands describe the actual packed V and its scales.
    kernel(
        s,
        l4,
        off,
        flat,
        kb,
        kb,
        q4,
        o4,
        None,
        *parts,
        None,
        _get_mask_config(None, None),
        Float32(scale),
        Float32(vscale),
        None,
        True,
        v.view(torch.uint16),
        sf.view(torch.uint8),
    )
    _scatter[(b, h)](o4, ptr, lens, out, *out.stride()[:2], h, d)
    return out
