# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Vectorised CPU references of the fp8 per-tensor MoE classes (:mod:`.spec_fp8` through :class:`.reference.TorchF32`),
validated against the exact scalar model (:class:`.reference.ExactF32`) by the device-free suite. The GEMM accumulators
come from the device (the opaque ``mma32`` K-step); the chains and the quantizations are evaluated here, bit for bit.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from . import spec_fp8 as S8
from .reference import TorchF32, _f32


def dequant_scale(m: torch.Tensor) -> torch.Tensor:
    """``div(1, m)`` of a per-tensor multiplier (fp32, one correctly rounded division; torch's division is IEEE on both
    devices): the class's dequantization scale, as the weight preparation forms it."""
    m = m.to(torch.float32)
    return torch.div(torch.ones_like(m), m)


def quantize_e4m3_static(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``cvt.rn.satfinite.e4m3(mul(x, m))`` as ``float8_e4m3fn`` of ``x``'s shape (:func:`.spec_fp8.x_quant_e4m3_elem` as torch
    ops: one fp32 multiply, the exact vectorised e4m3 rounding; the rounded values are e4m3 values, so the final dtype
    conversion is exact)."""
    # ``m``: one value (a 0-d or 1-element tensor), or a per-expert ``[E, 1, 1]`` multiplier broadcast against ``[E, N, K]``
    v = S8.x_quant_e4m3_elem(
        TorchF32, x.to(torch.float32), m.to(device=x.device, dtype=torch.float32)
    )
    return v.to(torch.float8_e4m3fn)


def e4m3_bytes(v: torch.Tensor) -> torch.Tensor:
    """The ``float8_e4m3fn`` tensor of e4m3 *values* (as the references return them: exactly representable)."""
    return v.to(torch.float8_e4m3fn)


def _scalar(v) -> torch.Tensor:
    return (
        v.detach().to("cpu", torch.float32).reshape(())
        if isinstance(v, torch.Tensor)
        else TorchF32.const(float(v))
    )


def w8a8_gemm1_cpu(
    g: torch.Tensor,
    u: torch.Tensor,
    s_x,
    s_w_rows: torch.Tensor,
    m_inter,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """The gated W8A8 GEMM1 chain on fp32 (gate, up) accumulators ``[rows, I]`` with the per-row weight scale ``s_w_rows
    [rows]`` (the expert's ``s_w1``): ``alpha = mul(s_x, s_w)``, ``cvt_e4m3(mul(glu_pre(mul(g, alpha), mul(u, alpha)), m_inter))``
    as e4m3 values (fp32)."""
    alpha = S8.w8a8_alpha(TorchF32, _scalar(s_x), _f32(s_w_rows).view(-1, 1))
    return S8.glu_quant_e4m3_elem(
        TorchF32, _f32(g), _f32(u), alpha, _scalar(m_inter), act, params
    )


def w8a8_gemm1_act_cpu(
    acc: torch.Tensor,
    s_x,
    s_w_rows: torch.Tensor,
    m_inter,
    act: str = "relu2",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """The ungated W8A8 GEMM1 chain on fp32 accumulators ``[rows, I]``: ``cvt_e4m3(mul(act(mul(acc, alpha)), m_inter))``."""
    alpha = S8.w8a8_alpha(TorchF32, _scalar(s_x), _f32(s_w_rows).view(-1, 1))
    return S8.act_quant_e4m3_elem(
        TorchF32, _f32(acc), alpha, _scalar(m_inter), act, params
    )


def w8a8_gemm2_cpu(acc: torch.Tensor, s_inter, s_w_rows: torch.Tensor) -> torch.Tensor:
    """The W8A8 GEMM2 chain on fp32 accumulators ``[rows, H]``: ``cvt_bf16(mul(acc, mul(s_inter, s_w2[e])))``."""
    alpha = S8.w8a8_alpha(TorchF32, _scalar(s_inter), _f32(s_w_rows).view(-1, 1))
    return S8.scaled_out_elem(TorchF32, _f32(acc), alpha, "bf16")


def x_quant_e4m3_cpu(x: torch.Tensor, m_x) -> torch.Tensor:
    """``cvt_e4m3(mul(x, m_x))`` as e4m3 values (fp32)."""
    return S8.x_quant_e4m3_elem(TorchF32, _f32(x), _scalar(m_x))
