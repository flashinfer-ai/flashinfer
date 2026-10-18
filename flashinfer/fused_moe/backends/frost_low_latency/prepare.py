# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving preparation of canonical MX weights (outside execution)."""

import torch


def _pack_scales(scales):
    e, rows, groups = scales.shape
    return (
        scales.view(torch.uint8)
        .reshape(e, rows // 128, 4, 32, groups // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
    )


def prepare_weights(w1, w1_scale, w2, w2_scale):
    """Pack prequantized weights without changing their values.

    Canonical w1 is [E, 2*I, H/pack] in [up, gate] order; w2 is
    [E, H, I/pack]. MXFP4 uses uint8 with two E2M1 values per byte;
    MXFP8 uses float8_e4m3fn. Scales are contiguous UE8M0 bytes in
    [E, rows, K/32]. I must be divisible by 32; it is padded to 128.
    For already padded weights, set logical I in ``MoEConfig.experts``.
    """
    if w1.dtype not in (torch.uint8, torch.float8_e4m3fn) or w2.dtype != w1.dtype:
        raise TypeError("weights must both be packed uint8 MXFP4 or E4M3 MXFP8")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("weights must be 3-D")
    e, h, _ = w2.shape
    packing = 2 if w1.dtype == torch.uint8 else 1
    i = w2.shape[2] * packing
    if e < 1 or h < 128 or h % 128 or i < 32 or i % 32:
        raise ValueError("require positive E, H divisible by 128 and I divisible by 32")
    expected = (
        (e, 2 * i, h // packing),
        (e, 2 * i, h // 32),
        (e, h, i // packing),
        (e, h, i // 32),
    )
    for tensor, shape in zip((w1, w1_scale, w2, w2_scale), expected, strict=True):
        if (
            tuple(tensor.shape) != shape
            or tensor.device != w1.device
            or not tensor.is_cuda
        ):
            raise ValueError(f"expected CUDA tensor {shape} on {w1.device}")
    for scale in (w1_scale, w2_scale):
        if scale.dtype not in (torch.uint8, torch.float8_e8m0fnu, torch.float8_e4m3fn):
            raise TypeError("scales must carry one-byte UE8M0 codes")
    physical = (i + 127) // 128 * 128
    # Work on byte views: MX scales include codes not representable as E4M3.
    gu = torch.zeros(
        (e, 2, physical, h // packing), dtype=torch.uint8, device=w1.device
    )
    gu[:, :, :i].copy_(w1.view(torch.uint8).reshape(e, 2, i, h // packing).flip(1))
    gs = torch.full((e, 2, physical, h // 32), 127, dtype=torch.uint8, device=w1.device)
    gs[:, :, :i].copy_(w1_scale.view(torch.uint8).reshape(e, 2, i, h // 32).flip(1))
    dw = torch.zeros((e, h, physical // packing), dtype=torch.uint8, device=w1.device)
    dw[:, :, : i // packing].copy_(w2.view(torch.uint8))
    ds = torch.full((e, h, physical // 32), 127, dtype=torch.uint8, device=w1.device)
    ds[:, :, : i // 32].copy_(w2_scale.view(torch.uint8))

    def stripe(x):
        return (
            x.reshape(e, 2, physical // 64, 64, -1)
            .transpose(1, 2)
            .contiguous()
            .reshape(e, 2 * physical, -1)
        )

    return dict(
        w1_weight=stripe(gu).view(w1.dtype),
        w1_weight_sf=_pack_scales(stripe(gs)),
        w2_weight=dw.view(w2.dtype),
        w2_weight_sf=_pack_scales(ds),
    )


def prepare_bf16_weights(w1, w2, activation):
    """Prepare canonical BF16 [up, gate] or non-gated weights without rounding.

    Logical I is padded to 128 for the shared FC2 reduction. Weight preparation
    is outside execution/capture; no activation values or pointers are cached.
    """
    from .fma.activation_config import lower_activation

    lower_activation(activation)
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise TypeError("BF16 preparation requires two bfloat16 weight tensors")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("weights must be 3-D")
    e, h, i = w2.shape
    branches = 2 if activation.is_gated else 1
    if e < 1 or h < 128 or h % 128 or i < 32 or i % 32:
        raise ValueError("require positive E, H divisible by 128 and I divisible by 32")
    if w1.shape != (e, branches * i, h):
        raise ValueError(f"expected w1 shape {(e, branches * i, h)}")
    if not w1.is_cuda or w2.device != w1.device:
        raise ValueError("weights must be CUDA tensors on the same device")
    physical = (i + 127) // 128 * 128
    first = torch.zeros((e, branches, physical, h), dtype=w1.dtype, device=w1.device)
    first[:, :, :i].copy_(w1.reshape(e, branches, i, h))
    second = torch.zeros((e, h, physical), dtype=w2.dtype, device=w2.device)
    second[:, :, :i].copy_(w2)
    return dict(w1_weight=first.view(e, branches * physical, h), w2_weight=second)
