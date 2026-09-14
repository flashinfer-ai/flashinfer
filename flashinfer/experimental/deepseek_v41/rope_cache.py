# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Adjacent-pair RoPE, BF16 rounding and model-specific paged quantization."""

import torch
import triton as tr

from .quantization import _check, _quantize
from ._utils import _overlaps


def rope_quantize_cache(x, freqs, positions, *, format, out, slots, page_size=64):
    _check(x)
    formats = {
        "index_mxfp4": (128, 0, 32, 68),
        "main_kv_fp4": (512, 1, 16, 288),
        "swa_mxfp8": (512, 2, 32, 528),
    }
    if format not in formats:
        raise ValueError("format must be index_mxfp4, main_kv_fp4 or swa_mxfp8")
    dim, code, group, width = formats[format]
    rows = x.shape[0]
    if x.dtype != torch.bfloat16 or x.shape[1] != dim:
        raise ValueError("RoPE cache requires BF16 rows matching the format dimension")
    if (
        freqs.ndim != 3
        or freqs.shape[0] < 1
        or freqs.shape[1:] != (32, 2)
        or freqs.dtype != torch.float32
        or freqs.device != x.device
        or not freqs.is_contiguous()
    ):
        raise ValueError("freqs must be contiguous CUDA FP32 [sequence,32,2]")
    for tensor in (positions, slots):
        if (
            tensor.shape != (rows,)
            or tensor.dtype != torch.int32
            or tensor.device != x.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("positions/slots must be contiguous CUDA int32 [rows]")
    if page_size not in (32, 64, 128):
        raise ValueError("page_size must be 32, 64 or 128")
    if (
        out.ndim != 4
        or out.shape[1:] != (page_size, 1, width)
        or out.dtype != torch.uint8
        or out.device != x.device
        or out.stride()[1:] != (width, width, 1)
    ):
        raise ValueError("paged output declaration mismatch")
    if code == 0:
        if out.stride(0) < page_size * width or out.stride(0) % 512:
            raise ValueError("index cache requires a 512-byte-aligned page stride")
    elif not out.is_contiguous():
        raise ValueError("main/window cache must be contiguous")
    for source in (x, freqs, positions, slots):
        if _overlaps(out, source):
            raise ValueError("cache may not overlap an input")
        if source.requires_grad and torch.is_grad_enabled():
            raise ValueError("RoPE cache has no implicit QAT derivative")
    if rows:
        with torch.cuda.device(x.device):
            _quantize[(tr.cdiv(rows, 4),)](
                x,
                out,
                out,
                rows,
                dim,
                code,
                group,
                4,
                True,
                page_size,
                slots,
                True,
                out.stride(0),
                True,
                freqs,
                positions,
                freqs.shape[0],
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out
