# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Logical sparse token IDs to physical FlashMLA cache slots."""

import torch
import triton as tr
import triton.language as tl

from ._utils import _overlaps


@tr.jit
def _map(
    IDS,
    TABLE,
    OUT,
    K: tl.constexpr,
    SQ: tl.constexpr,
    PAGES: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(IDS + row * K + column, column < K, -1)
    valid = (column < K) & (logical >= 0) & (logical < PAGES * PAGE)
    page = tl.load(TABLE + (row // SQ) * PAGES + logical // PAGE, valid, -1)
    physical = tl.where(valid & (page >= 0), page * PAGE + logical % PAGE, -1)
    tl.store(OUT + row * K + column, physical, column < K)


def paged_indices(indices, block_table, *, page_size=64, out=None):
    if indices.device.type != "cuda" or torch.cuda.get_device_capability(
        indices.device
    ) not in ((10, 0), (10, 3)):
        raise ValueError("V4.1 paged indices currently require SM100/SM103")
    if (
        indices.ndim != 3
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
        or min(indices.shape) <= 0
    ):
        raise ValueError("positive contiguous int32 [B,Sq,K] indices required")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != indices.shape[0]
        or block_table.shape[1] <= 0
        or block_table.dtype != torch.int32
        or block_table.device != indices.device
        or not block_table.is_contiguous()
    ):
        raise ValueError(
            "contiguous int32 [B,pages] block table required on indices device"
        )
    if page_size not in (32, 64, 128):
        raise ValueError("page_size must be 32, 64 or 128")
    if block_table.shape[1] * page_size >= 2**31:
        raise ValueError("logical capacity must fit int32")
    if out is None:
        out = torch.empty_like(indices)
    elif (
        out.shape != indices.shape
        or out.dtype != torch.int32
        or out.device != indices.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "output must match indices shape, device, int32 dtype and contiguous layout"
        )
    if _overlaps(out, indices) or _overlaps(out, block_table):
        raise ValueError("output must not overlap indices or block table")
    b, sq, k = indices.shape
    with torch.cuda.device(indices.device):
        _map[(tr.cdiv(k, 256), b * sq)](
            indices, block_table, out, k, sq, block_table.shape[1], page_size, 256
        )
    return out
