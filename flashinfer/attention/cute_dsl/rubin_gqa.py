# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private launch glue for fixed-length Rubin GQA prefill."""

import math
import numbers
import struct
from typing import Optional

import torch


def _positive_scalar(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive host scalar")
    value = float(value)
    try:
        rounded = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        rounded = math.inf
    if not math.isfinite(value) or not math.isfinite(rounded) or rounded <= 0:
        raise ValueError(f"{name} must be finite, positive and representable in FP32")
    return value


def _overlaps(left, right):
    # Every accepted tensor is contiguous, so exact byte intervals suffice and
    # allow disjoint views of the same allocation without accepting output races.
    left_begin, right_begin = left.data_ptr(), right.data_ptr()
    return (
        left_begin < right_begin + right.numel() * right.element_size()
        and right_begin < left_begin + left.numel() * left.element_size()
    )


def _run_rubin_gqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    kernel,
    causal: bool = False,
    lse: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    scale_q: float = 1.0,
    scale_k: float = 1.0,
    scale_v: float = 1.0,
    scale_o: float = 1.0,
) -> torch.Tensor:
    """Validate and launch precompiled GQA on contiguous BSHD tensor views.

    Q/K/V are E4M3 with Q/K dimension 192 and V dimension 128; output is
    BF16 and optional LSE is FP32 in base 2. All sequences use their full
    positive tensor lengths. Causal masking is bottom-right aligned.
    ``kernel`` is the (handle, mask) pair compiled during wrapper planning.
    No tensor contents are copied or read on the host during launch.
    """
    if not isinstance(causal, bool):
        raise ValueError("causal must be bool")
    tensors = [("q", q), ("k", k), ("v", v), ("out", out)]
    if lse is not None:
        tensors.append(("lse", lse))
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous in BSHD order (BSH for lse)")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be on a CUDA SM107 device")
        if tensor.device != q.device:
            raise ValueError("All Rubin GQA tensors must be on the same CUDA device")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must have a 16-byte aligned data pointer")
        if any(dim <= 0 for dim in tensor.shape):
            raise ValueError(f"{name} dimensions and sequence lengths must be positive")
        if tensor.numel() >= 2**31:
            raise ValueError(f"{name} exceeds the supported signed 32-bit index range")
    if any(tensor.ndim != 4 for tensor in (q, k, v, out)):
        raise ValueError("q, k, v and out must have four BSHD dimensions")
    if any(tensor.dtype != torch.float8_e4m3fn for tensor in (q, k, v)):
        raise ValueError("Rubin GQA requires E4M3 Q/K/V")
    if out.dtype != torch.bfloat16:
        raise ValueError("Rubin GQA requires BF16 output")
    batch, q_len, heads_q, head_dim = q.shape
    batch_k, kv_len, heads_k, head_dim_k = k.shape
    if (head_dim, head_dim_k, v.shape[-1], out.shape[-1]) != (192, 192, 128, 128):
        raise ValueError("Rubin GQA requires Q/K dimension 192 and V/O dimension 128")
    if batch_k != batch or tuple(v.shape) != (batch, kv_len, heads_k, 128):
        raise ValueError(
            "Q/K/V batch sizes and K/V sequence/head dimensions must match"
        )
    if tuple(out.shape) != (batch, q_len, heads_q, 128):
        raise ValueError("out shape must be (batch, Q length, Q heads, 128)")
    if heads_q % heads_k or heads_q // heads_k not in (1, 2, 4, 8):
        raise ValueError("Rubin GQA requires a Q/KV head ratio of 1, 2, 4 or 8")
    if causal and q_len > kv_len:
        raise ValueError("Causal Rubin GQA requires Q length <= KV length")
    if lse is not None and (
        lse.dtype != torch.float32 or tuple(lse.shape) != (batch, q_len, heads_q)
    ):
        raise ValueError("lse must be FP32 with shape (batch, Q length, Q heads)")
    outputs = [out] if lse is None else [out, lse]
    for output in outputs:
        if any(_overlaps(output, tensor) for tensor in (q, k, v)):
            raise ValueError("Rubin GQA output tensors must not overlap inputs")
    if lse is not None and _overlaps(out, lse):
        raise ValueError("out and lse must not overlap")

    sm_scale = _positive_scalar(
        "sm_scale", 1.0 / math.sqrt(192) if sm_scale is None else sm_scale
    )
    scale_q = _positive_scalar("scale_q", scale_q)
    scale_k = _positive_scalar("scale_k", scale_k)
    scale_v = _positive_scalar("scale_v", scale_v)
    scale_o = _positive_scalar("scale_o", scale_o)
    score_scale = _positive_scalar(
        "effective score scale", sm_scale * scale_q * scale_k
    )
    # The kernel multiplies its FP32 argument by an FP32 log2(e) constant.
    # Check that conversion as well as the host-side product before launch.
    score_scale_fp32 = struct.unpack("f", struct.pack("f", score_scale))[0]
    log2_e_fp32 = struct.unpack("f", struct.pack("f", math.log2(math.e)))[0]
    _positive_scalar("base-2 score scale", score_scale_fp32 * log2_e_fp32)
    output_scale = _positive_scalar("effective output scale", scale_v / scale_o)

    handle, mask = kernel
    with torch.cuda.device(q.device):
        # TVM-FFI provides the current stream; there is no explicit stream
        # argument in this handle's invocation ABI.
        handle(
            None,
            None,
            None,
            q,
            k,
            v,
            out,
            lse,
            None,
            None,
            None,
            None,
            mask,
            score_scale,
            output_scale,
            False,
            None,
        )
    return out
