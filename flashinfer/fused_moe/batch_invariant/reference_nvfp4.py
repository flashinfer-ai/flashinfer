# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Vectorised CPU references of the NVFP4 MoE classes' chains (:mod:`.spec_nvfp4` through :class:`.reference.TorchF32`),
validated against the exact scalar model (:class:`.reference.ExactF32`) by the device-free tests. The GEMM accumulators
come from the device (the opaque block-scaled steps); the per-token quantizations are the quantizer class's own reference
(:func:`flashinfer.batch_invariant.quantization.nvfp4_reference.quantize_reference`)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from . import spec_nvfp4 as Q
from .reference import TorchF32, _f32


def _row_scale(alpha_rows: torch.Tensor, pts_rows: torch.Tensor) -> torch.Tensor:
    return Q.w4a4_row_scale(
        TorchF32, _f32(alpha_rows).view(-1, 1), _f32(pts_rows).view(-1, 1)
    )


def w4a4_gemm1_cpu(
    g: torch.Tensor,
    u: torch.Tensor,
    alpha_rows: torch.Tensor,
    pts_rows: torch.Tensor,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """The gated W4A4 GEMM1 chain for whole rows: ``s = mul(alpha_e(row), pts(row))``, ``cvt_bf16(glu_pre(mul(g, s), mul(u,
    s)))``; ``alpha_rows`` / ``pts_rows`` are the per-row scalars (``[rows]`` f32), ``g`` / ``u`` the ``[rows, I]`` fp32
    accumulators."""
    return Q.glu_scaled_elem(
        TorchF32,
        _f32(g),
        _f32(u),
        _row_scale(alpha_rows, pts_rows),
        act,
        "bf16",
        params,
    )


def w4a4_act_cpu(
    acc: torch.Tensor,
    alpha_rows: torch.Tensor,
    pts_rows: torch.Tensor,
    act: str = "relu2",
    params: Sequence[Optional[float]] = (),
) -> torch.Tensor:
    """The ungated W4A4 GEMM1 chain for whole rows: ``cvt_bf16(act(mul(acc, s)))``."""
    return Q.act_scaled_elem(
        TorchF32, _f32(acc), _row_scale(alpha_rows, pts_rows), act, "bf16", params
    )


def w4a4_gemm2_cpu(
    acc: torch.Tensor, alpha_rows: torch.Tensor, pts_rows: torch.Tensor
) -> torch.Tensor:
    """The W4A4 GEMM2 chain for whole rows: ``cvt_bf16(mul(acc, mul(alpha_e(row), pts(row))))``."""
    return Q.scaled_out_elem(
        TorchF32, _f32(acc), _row_scale(alpha_rows, pts_rows), "bf16"
    )


def w4a4_static_gemm1_cpu(
    acc: torch.Tensor,
    alpha_rows: torch.Tensor,
    pts_rows: torch.Tensor,
    gs_rows: torch.Tensor,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The static class's GEMM1 (``moe.nvfp4.<act>.static``) for whole rows of the grouped kernel's fp32 accumulators ``acc [rows,
    N]``: ``s = mul(alpha_e(row), pts(row))``, the fp32 gated value ``glu_pre(mul(g, s), mul(u, s))`` on the interleaved column pairs
    (``u`` at ``16q + r``, ``g`` at ``16q + 8 + r``) or the ungated ``act(mul(acc, s))``, then per 16-column block the per-tensor
    quantizer's chain with the row's expert scale ``gs_rows``. Returns ``(packed [rows, I/2] uint8, sf [rows, I/16] uint8)``: the
    codes and the scale bytes as the kernel stores them."""
    from ...batch_invariant.quantization.nvfp4_reference import TorchF32 as QF32
    from ...batch_invariant.quantization.nvfp4_reference import (
        e2m1_codes,
        e4m3_bytes,
        pack_e2m1,
    )

    acc = _f32(acc)
    s = _row_scale(alpha_rows, pts_rows)
    if not gated:
        v = Q.act_scaled_pre(TorchF32, acc, s, act, params)
    else:
        rows, N = acc.shape
        vv = acc.view(rows, N // 16, 2, 8)
        u, g = (
            vv[:, :, 0, :].reshape(rows, N // 2),
            vv[:, :, 1, :].reshape(rows, N // 2),
        )
        v = Q.glu_scaled_pre(TorchF32, g, u, s, act, params)
    rows, I = v.shape
    nb = I // Q.NVFP4_BLOCK
    vb = v.view(rows, nb, Q.NVFP4_BLOCK)
    amax = QF32.fabs(vb).amax(
        dim=-1
    )  # the canonical tree's value (the max of non-negatives)
    sf_d, inv = Q.nvfp4_block_chain(QF32, amax, _f32(gs_rows).view(-1, 1))
    q = Q.nvfp4_quant_elem(QF32, vb, inv.unsqueeze(-1))  # e2m1 values
    return pack_e2m1(e2m1_codes(q).view(rows, I)), e4m3_bytes(sf_d)


def w4a4_static_gemm2_cpu(
    acc: torch.Tensor, alpha_rows: torch.Tensor, inv2_rows: torch.Tensor
) -> torch.Tensor:
    """The static class's GEMM2: ``cvt_bf16(mul(acc, mul(alpha2_e(row), inv2_e(row))))`` (``inv2_e = div(1, gs2_e)`` as the host
    formed it)."""
    return Q.scaled_out_elem(
        TorchF32, _f32(acc), _row_scale(alpha_rows, inv2_rows), "bf16"
    )
