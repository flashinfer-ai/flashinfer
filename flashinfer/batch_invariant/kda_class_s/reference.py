# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Torch evaluation of the class ``kda.bf16.s16`` with the kernels' operation and rounding points (:mod:`.spec`). The
elementary functions are torch's and the bf16 dot products plain matmuls here, so the kernels agree with it to a
tolerance (a few 1e-3 relative on long sequences), not bit for bit."""

from __future__ import annotations

import math

import torch

from .spec import CHUNK


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(torch.bfloat16).to(x.dtype)


def kda_class_s_reference(
    q, k, v, g, beta, A_log, dt_bias, state, scale=None, lower_bound=-5.0
):
    """``q, k, v, g``: ``[B, T, H, D]`` bf16; ``beta``: ``[B, T, H]`` (logit); ``A_log`` ``[H]``; ``dt_bias``
    ``[H, D]``; ``state`` ``[B, H, D, D]`` bf16 (V first). Returns the fp32 ``[B, T, H, D]`` outputs (bf16 values) and
    the state at the last chunk boundary."""
    B, T, H, D = q.shape
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    qf, kf = q.float(), k.float()
    kn = kf * (1.0 / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6))
    qn = qf * (1.0 / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * scale)
    ell = lower_bound * torch.sigmoid(
        torch.exp(A_log).view(1, 1, H, 1) * (g.float() + dt_bias.view(1, 1, H, D))
    )
    b = torch.sigmoid(beta.float())
    S = state.float()
    out = torch.empty(B, T, H, D, dtype=torch.float32, device=q.device)
    Kc = torch.empty(B, H, 0, D, device=q.device)
    Uc = torch.empty(B, H, 0, D, device=q.device)
    lam = None
    for t in range(T):
        p = Kc.shape[2]
        lam = ell[:, t] if p == 0 else lam + ell[:, t]
        dec = torch.exp(lam)
        kt = _bf16(kn[:, t] * dec)
        ki = _bf16(kn[:, t] * (1.0 / dec))
        qt = _bf16(qn[:, t] * dec)
        Kc = torch.cat([Kc, ki.unsqueeze(2)], 2)
        pred = torch.einsum("bhvk,bhk->bhv", S, kt)
        if p:
            pred = pred + torch.einsum(
                "bhj,bhjv->bhv", torch.einsum("bhjk,bhk->bhj", Kc[:, :, :p], kt), Uc
            )
        u = b[:, t].unsqueeze(-1) * (v[:, t].float() - pred)
        Uc = torch.cat([Uc, u.unsqueeze(2)], 2)
        o = torch.einsum("bhvk,bhk->bhv", S, qt) + torch.einsum(
            "bhj,bhjv->bhv", torch.einsum("bhjk,bhk->bhj", Kc, qt), Uc
        )
        out[:, t] = _bf16(o)
        if Kc.shape[2] == CHUNK:
            Kb = _bf16(Kc * dec.unsqueeze(2))
            S = _bf16(S * dec.unsqueeze(2) + _bf16(Uc).transpose(-1, -2) @ Kb)
            Kc, Uc = Kc[:, :, :0], Uc[:, :, :0]
    return out, S
