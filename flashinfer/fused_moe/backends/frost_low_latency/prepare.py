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


def prepare_weights(w1, w1_scale, w2, w2_scale, *, activation):
    """Pack prequantized weights without changing their values.

    Canonical w1 is [E, 2*I, H/pack] in [up, gate] order for gated
    activations, or [E, I, H/pack] otherwise; w2 is
    [E, H, I/pack]. MXFP4 uses uint8 with two E2M1 values per byte;
    MXFP8 uses float8_e4m3fn. Scales are contiguous UE8M0 bytes in
    [E, rows, K/32]. I must be divisible by 32; it is padded to 128.
    For already padded weights, set logical I in ``MoEConfig.experts``.
    """
    from .fma.activation_config import lower_activation

    lower_activation(activation)
    branches = 2 if activation.is_gated else 1
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
        (e, branches * i, h // packing),
        (e, branches * i, h // 32),
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
        (e, branches, physical, h // packing), dtype=torch.uint8, device=w1.device
    )
    gu[:, :, :i].copy_(
        w1.view(torch.uint8).reshape(e, branches, i, h // packing).flip(1)
    )
    gs = torch.full(
        (e, branches, physical, h // 32), 127, dtype=torch.uint8, device=w1.device
    )
    gs[:, :, :i].copy_(
        w1_scale.view(torch.uint8).reshape(e, branches, i, h // 32).flip(1)
    )
    dw = torch.zeros((e, h, physical // packing), dtype=torch.uint8, device=w1.device)
    dw[:, :, : i // packing].copy_(w2.view(torch.uint8))
    ds = torch.full((e, h, physical // 32), 127, dtype=torch.uint8, device=w1.device)
    ds[:, :, : i // 32].copy_(w2_scale.view(torch.uint8))

    def stripe(x):
        if not activation.is_gated:
            return x.reshape(e, physical, -1)
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
    return _prepare_float16_weights(w1, w2, activation, torch.bfloat16)


def prepare_fp16_weights(w1, w2, activation):
    return _prepare_float16_weights(w1, w2, activation, torch.float16)


def _prepare_float16_weights(w1, w2, activation, dtype):
    """Prepare canonical 16-bit [up, gate] or non-gated weights without rounding.

    Logical I is padded to 128 for the shared FC2 reduction. Weight preparation
    is outside execution/capture; no activation values or pointers are cached.
    """
    from .fma.activation_config import lower_activation

    lower_activation(activation)
    if w1.dtype != dtype or w2.dtype != dtype:
        raise TypeError(f"Preparation requires two {dtype} weight tensors")
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


def prepare_fp8_per_tensor_weights(w1_bf16, w2_bf16, **kwargs):
    from ...prepare import prepare_cutlass_fp8_per_tensor_weights
    from ...api import SwiGLU
    from .fma.activation_config import lower_activation

    activation = kwargs.get("activation") or SwiGLU()
    lower_activation(activation)
    view = prepare_cutlass_fp8_per_tensor_weights(w1_bf16, w2_bf16, **kwargs)
    e, h, i = view["fc2_expert_weights"].shape
    if h % 128 or i % 32:
        raise ValueError("Frost FP8 requires H divisible by 128 and I by 32")
    p = (i + 127) // 128 * 128
    if p != i:
        branches = 2 if activation.is_gated else 1
        first = torch.zeros(
            (e, branches, p, h),
            device=view["fc1_expert_weights"].device,
            dtype=torch.float8_e4m3fn,
        )
        first[:, :, :i].copy_(view["fc1_expert_weights"].reshape(e, branches, i, h))
        second = torch.zeros(
            (e, h, p),
            device=view["fc2_expert_weights"].device,
            dtype=torch.float8_e4m3fn,
        )
        second[:, :, :i].copy_(view["fc2_expert_weights"])
        view = dict(
            view,
            fc1_expert_weights=first.view(e, branches * p, h),
            fc2_expert_weights=second,
        )
    return view


def prepare_fp8_weight_only_weights(w1, w1_scale, w2, w2_scale, activation):
    """Preserve canonical E4M3 bytes and per-expert dequantization scales.

    Unlike W8A8 preparation, these scales contain weight calibration only.
    BF16 inputs and intermediates do not have quantization multipliers.
    """
    from .fma.activation_config import lower_activation

    lower_activation(activation)
    if w1.dtype is not torch.float8_e4m3fn or w2.dtype is not w1.dtype:
        raise TypeError("W8A16 requires two E4M3 weight tensors")
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
    for scale in (w1_scale, w2_scale):
        if (
            scale.dtype is not torch.float32
            or scale.device != w1.device
            or scale.shape != (e,)
        ):
            raise ValueError(f"weight scales must be FP32 [{e}] on {w1.device}")
    physical = (i + 127) // 128 * 128
    first = torch.zeros((e, branches, physical, h), dtype=torch.uint8, device=w1.device)
    first[:, :, :i].copy_(w1.view(torch.uint8).reshape(e, branches, i, h))
    second = torch.zeros((e, h, physical), dtype=torch.uint8, device=w2.device)
    second[:, :, :i].copy_(w2.view(torch.uint8))
    return dict(
        fc1_expert_weights=first.view(e, branches * physical, h).view(w1.dtype),
        fc2_expert_weights=second.view(w2.dtype),
        fc1_dequant_scale=w1_scale.contiguous(),
        fc2_dequant_scale=w2_scale.contiguous(),
        # Retain the common binding layout; W8A16 does not read this field.
        fc2_act_quant_scale=torch.ones((), device=w1.device, dtype=torch.float32),
    )


def prepare_deepseek_fp8_weights(w1, s1, w2, s2, activation):
    """Preserve canonical E4M3 payloads and FP32 128x128 block scales."""
    from .fma.activation_config import lower_activation

    lower_activation(activation)
    if w1.dtype is not torch.float8_e4m3fn or w2.dtype is not w1.dtype:
        raise TypeError("DeepSeek Frost requires E4M3 weights")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("weights must be 3-D")
    e, h, i = w2.shape
    branches = 2 if activation.is_gated else 1
    if e < 1 or h < 128 or i < 128 or h % 128 or i % 128:
        raise ValueError("DeepSeek Frost requires positive E and H/I divisible by 128")
    if w1.shape != (e, branches * i, h):
        raise ValueError(f"expected w1 shape {(e, branches * i, h)}")
    if not w1.is_cuda or w2.device != w1.device:
        raise ValueError("weights must be CUDA tensors on the same device")
    for scale, shape in (
        (s1, (e, branches * i // 128, h // 128)),
        (s2, (e, h // 128, i // 128)),
    ):
        if (
            scale.dtype is not torch.float32
            or scale.shape != shape
            or scale.device != w1.device
        ):
            raise ValueError(f"block scales must be FP32 {shape} on {w1.device}")
    return dict(
        fc1_expert_weights=w1.contiguous(),
        fc1_block_scales=s1.contiguous(),
        fc2_expert_weights=w2.contiguous(),
        fc2_block_scales=s2.contiguous(),
    )
