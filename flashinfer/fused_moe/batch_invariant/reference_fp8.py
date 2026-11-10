# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Vectorised CPU references of the fp8 per-tensor MoE classes (:mod:`.spec_fp8` through :class:`.reference.TorchF32`),
validated against the exact scalar model (:class:`.reference.ExactF32`) by the device-free suite. The GEMM accumulators
come from the device (the opaque ``mma32`` K-step); the chains and the quantizations are evaluated here, bit for bit.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

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


# -- the per-token class ``moe.e4m3.<act>.pertoken``


def pertoken_quant_cpu(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The per-token quantization of the rows of ``x``: ``(q, s)`` with ``q`` the e4m3 *values* (fp32; ``e4m3_bytes`` gives the
    bytes) and ``s [rows]`` the dequantization scales -- ``amax = fmaxc over fabs(x)`` (exact in every order: the max of the row's
    magnitudes), ``m = div(448, nzor(amax, 448))``, ``q = cvt_e4m3(mul(x, m))``, ``s = div(1, m)``; every op through ``TorchF32``."""
    xf = _f32(x)
    amax = xf.abs().amax(dim=1)
    m = S8.pertoken_mult(TorchF32, amax)
    return S8.pertoken_quant_elem(TorchF32, xf, m.view(-1, 1)), S8.pertoken_scale(
        TorchF32, m
    )


def pertoken_gemm1_cpu(
    g: torch.Tensor,
    u: torch.Tensor,
    pts_rows: torch.Tensor,
    sw_g: torch.Tensor,
    sw_u: torch.Tensor,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """GEMM1 of the per-token class on fp32 (gate, up) accumulators ``[rows, I]``: ``cvt_bf16(mul(act(mul(g, s_g)), mul(u, s_u)))`` with
    ``s_g = mul(pts[row], sw_g[col])``, ``s_u = mul(pts[row], sw_u[col])`` (``pts_rows [rows]`` the rows' scales, ``sw_g`` / ``sw_u [I]``
    the expert's channel scales of the gate and the up columns)."""
    p = _f32(pts_rows).view(-1, 1)
    sg = S8.pertoken_elem_scale(TorchF32, p, _f32(sw_g).view(1, -1))
    su = S8.pertoken_elem_scale(TorchF32, p, _f32(sw_u).view(1, -1))
    return S8.glu_scaled2_elem(TorchF32, _f32(g), _f32(u), sg, su, act, "bf16", params)


def pertoken_gemm1_act_cpu(
    acc: torch.Tensor,
    pts_rows: torch.Tensor,
    sw_cols: torch.Tensor,
    act: str = "relu2",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """The ungated GEMM1 chain of the per-token class: ``cvt_bf16(act(mul(acc, mul(pts[row], sw[col]))))``."""
    s = S8.pertoken_elem_scale(
        TorchF32, _f32(pts_rows).view(-1, 1), _f32(sw_cols).view(1, -1)
    )
    return S8.act_scaled_elem(TorchF32, _f32(acc), s, act, "bf16", params)


def pertoken_chain_cpu(
    acc: torch.Tensor,
    pts_rows: torch.Tensor,
    sw_row: torch.Tensor,
    act,
    params: Sequence[Optional[float]] = (),
    gated: bool = True,
) -> torch.Tensor:
    """The per-token class's GEMM1 chain on the grouped kernel's fp32 accumulator rows ``acc [rows, N]`` with the rows' scales
    ``pts_rows`` and the expert's channel scales ``sw_row [N]`` in the stored (interleaved) column order: the gated chains pair the
    interleaved columns (and their channel scales), the ungated ones take every column; bf16 values."""
    if not gated:
        return pertoken_gemm1_act_cpu(acc, pts_rows, sw_row, act, params)
    acc = _f32(acc)
    I = acc.shape[1] // 2
    u = acc.view(-1, I // 8, 2, 8)[:, :, 0, :].reshape(-1, I)
    g = acc.view(-1, I // 8, 2, 8)[:, :, 1, :].reshape(-1, I)
    sw = _f32(sw_row).view(I // 8, 2, 8)
    return pertoken_gemm1_cpu(
        g, u, pts_rows, sw[:, 1, :].reshape(I), sw[:, 0, :].reshape(I), act, params
    )


def pertoken_gemm2_cpu(
    acc: torch.Tensor, pts_rows: torch.Tensor, sw_row: torch.Tensor
) -> torch.Tensor:
    """GEMM2 of the per-token class on fp32 accumulators ``[rows, H]``: ``cvt_bf16(mul(acc, mul(pts[row], sw[col])))``."""
    s = S8.pertoken_elem_scale(
        TorchF32, _f32(pts_rows).view(-1, 1), _f32(sw_row).view(1, -1)
    )
    return S8.scaled_out_elem(TorchF32, _f32(acc), s, "bf16")
