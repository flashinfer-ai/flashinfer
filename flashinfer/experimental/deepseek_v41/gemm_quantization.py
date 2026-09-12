# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""MXFP8 activation quantization directly into DeepGEMM's packed scale layout."""

import torch
import triton as tr
import triton.language as tl

from .quantization import _power2
from ._utils import _overlaps


@tr.jit
def _quantize_gemm(
    X, DATA, SF, M: tl.constexpr, K: tl.constexpr, SF_STRIDE: tl.constexpr
):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    columns = tl.program_id(1) * 512 + tl.arange(0, 512)
    value = tl.load(
        X + rows[:, None].to(tl.int64) * K + columns[None, :],
        (rows[:, None] < M) & (columns[None, :] < K),
        0,
    ).to(tl.float32)
    grouped = value.reshape(4, 16, 32)
    amax = tl.max(tl.abs(grouped), 2)
    scale, encoded = _power2(tl.maximum(amax, 1e-4), 1.0 / 448.0)
    normalized = (grouped / scale[:, :, None]).reshape(4, 512)
    quantized = tl.minimum(tl.maximum(normalized, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(
        DATA + rows[:, None].to(tl.int64) * K + columns[None, :],
        quantized,
        (rows[:, None] < M) & (columns[None, :] < K),
    )
    shifts = tl.arange(0, 4) * 8
    packed = tl.sum(encoded.reshape(4, 4, 4).to(tl.uint32) << shifts[None, None, :], 2)
    sf_columns = tl.program_id(1) * 4 + tl.arange(0, 4)
    tl.store(
        SF + rows[:, None] + sf_columns[None, :].to(tl.int64) * SF_STRIDE,
        packed.to(tl.int32),
        (rows[:, None] < M) & (sf_columns[None, :] < K // 128),
    )


def quantize_gemm(x, *, data=None, scales=None):
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("V4.1 GEMM quantization currently requires SM100/SM103")
    if (
        x.ndim != 2
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float32)
        or not 128 <= x.shape[1] <= 32768
        or x.shape[1] % 128
    ):
        raise ValueError(
            "contiguous BF16/FP32 [M,K], K128..32768 divisible by128 required"
        )
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError("GEMM quantization has no implicit QAT derivative")
    m, k = x.shape
    sf_stride = tr.cdiv(m, 4) * 4
    if data is None:
        data = torch.empty(m, k, device=x.device, dtype=torch.float8_e4m3fn)
    elif (
        data.shape != (m, k)
        or data.dtype != torch.float8_e4m3fn
        or data.device != x.device
        or not data.is_contiguous()
    ):
        raise ValueError("data must be contiguous E4M3 [M,K] on the input device")
    if scales is None:
        scales = torch.empty_strided(
            (m, k // 128), (1, sf_stride), device=x.device, dtype=torch.int32
        )
    elif (
        scales.shape != (m, k // 128)
        or scales.dtype != torch.int32
        or scales.device != x.device
        or scales.stride() != (1, sf_stride)
    ):
        raise ValueError("scales require int32 [M,K/128] with strides(1,round_up(M,4))")
    if m and any(_overlaps(a, b) for a, b in ((data, x), (scales, x), (data, scales))):
        raise ValueError("GEMM quantization buffers may not overlap")
    if m:
        with torch.cuda.device(x.device):
            _quantize_gemm[(tr.cdiv(m, 4), tr.cdiv(k, 512))](
                x, data, scales, m, k, sf_stride, num_warps=4, enable_fp_fusion=False
            )
    return data, scales
