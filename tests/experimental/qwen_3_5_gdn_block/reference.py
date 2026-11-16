# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Pure-torch reference of the Qwen 3.5 GDN block (decode and MTP verify), the seeded inputs and the input
conditioning that makes the reference's conv outputs exactly the kernels'.

Rounding points follow the fused chain: each conv tap product rounded to BF16 and the taps summed in FP32 in tap order
(bit-exact in torch), SiLU of the conv in FP64 then BF16; the BA projection rounded to BF16 (exact here: ``hidden`` in
{-1, 0, 1} and ``ba_weight`` in k/256 make every FP32 partial sum exact); beta = sigmoid(b), g = -exp(A_log) *
softplus(a + dt_bias) (threshold 20); q and k L2-normalized with eps 1e-6 on the sum of squares, q scaled by
1/sqrt(Dk); the gated delta rule in FP64; the core output rounded to BF16; the gated RMSNorm (over Dv, of the BF16
core output, times the weight and SiLU(z)) rounded to BF16; then FP8 E4M3 of ``normed * (1 / out_scale)`` with
saturation.

The kernels evaluate the conv SiLU with ``ex2.approx`` (a few FP32 ulps from exact), so a conv output whose exact SiLU
lies within 2^-16 relative of a BF16 rounding midpoint could round either way. :func:`condition_conv_inputs` moves the
newest conv input of every such (token, channel) until none is left, so the reference's BF16 q / k / v are the
kernels' and the SSM states agree to FP32 accumulation-order noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

QWEN38_27B = dict(
    d_model=5120,
    num_v_heads=48,
    num_k_heads=16,
    head_k_dim=128,
    head_v_dim=128,
    conv_kernel=4,
)
_AMBIGUOUS_REL = 2.0**-16
_FP8_MAX = 448.0


def same_bytes(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.view(torch.uint8) == b.view(torch.uint8)).float().mean())


def quant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Static per-tensor FP8: ``satfinite(x * fp32(1 / scale))`` rounded to nearest even."""
    inv = 1.0 / scale.float().reshape(())
    return (x.float() * inv).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)


def conv_silu(window: torch.Tensor, weight: torch.Tensor):
    """``window [..., C, K]`` (oldest tap first) and ``weight [C, K]`` bf16 -> (exact SiLU in FP64, its BF16)."""
    prod = (window.float() * weight.float()).bfloat16().float()
    acc = prod[..., 0]
    for j in range(1, prod.shape[-1]):
        acc = acc + prod[..., j]
    a = acc.double()
    s = a / (1.0 + torch.exp(-a))
    return s, s.float().bfloat16()


def ambiguous(s: torch.Tensor) -> torch.Tensor:
    """Where an FP64 value lies within 2^-16 relative of a BF16 rounding midpoint."""
    bits = s.float().view(torch.int32)
    mid = ((bits & ~0xFFFF) | 0x8000).view(torch.float32).double()
    return (s - mid).abs() <= _AMBIGUOUS_REL * s.abs()


@dataclass
class Case:
    """One seeded problem; ``conv_state`` / ``ssm_state`` are the initial pools (tests run on clones)."""

    qkv: torch.Tensor
    z: torch.Tensor
    hidden: torch.Tensor
    ba_weight: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    ssm_state: torch.Tensor
    state_indices: torch.Tensor
    norm_weight: torch.Tensor
    out_scale: torch.Tensor
    conv_state: torch.Tensor
    conv_weight: torch.Tensor
    cu_seqlens: Optional[torch.Tensor] = None
    num_accepted_tokens: Optional[torch.Tensor] = None
    eps: float = 1e-6


def make_case(
    requests: int,
    tokens: int,
    *,
    seed: int,
    device,
    form: str,
    param_dtypes=(torch.float32, torch.bfloat16, torch.bfloat16),
    conv_dim_first: bool = False,
    g=QWEN38_27B,
) -> Case:
    """Small-magnitude random inputs at geometry ``g``; ``requests * tokens`` rows, distinct live slots, the last
    request (row, for decode) a padding one when there are several. ``qkv`` / ``z`` are strided views of one QKVZ
    projection output, as a serving stack holds them. ``conv_dim_first`` stores the conv pool ``[slots, C, L]``
    instead of channel-innermost ``[slots, L, C]``."""
    torch.manual_seed(seed)
    hv, hk, dk, dv, k = (
        g["num_v_heads"],
        g["num_k_heads"],
        g["head_k_dim"],
        g["head_v_dim"],
        g["conv_kernel"],
    )
    c = 2 * hk * dk + hv * dv
    rows = requests * tokens
    slots = rows + 8
    qkvz = torch.randn(rows, c + hv * dv, device=device).bfloat16()
    state_len = tokens + k - 2
    if conv_dim_first:
        conv = torch.randn(slots, c, state_len, device=device).bfloat16()
    else:
        conv = (
            torch.randn(slots, state_len, c, device=device).bfloat16().transpose(1, 2)
        )
    idx = (torch.randperm(slots - 1, device=device)[:rows] + 1).int()
    if form == "verify":
        idx = idx.view(requests, tokens)
    if requests > 1:
        idx[-1] = 0
    a_dt, dt_dt, w_dt = param_dtypes
    case = Case(
        qkv=qkvz[:, :c],
        z=qkvz[:, c:].view(rows, hv, dv),
        hidden=torch.randint(-1, 2, (rows, g["d_model"]), device=device).bfloat16(),
        ba_weight=(
            torch.randint(-8, 9, (2 * hv, g["d_model"]), device=device).float() / 256
        ).bfloat16(),
        A_log=(torch.randn(hv, device=device) * 0.5).to(a_dt),
        dt_bias=(torch.randn(hv, device=device) * 0.5).to(dt_dt),
        ssm_state=torch.randn(slots, hv, dv, dk, device=device) * 0.1,
        state_indices=idx,
        norm_weight=(1 + 0.2 * torch.randn(dv, device=device)).to(w_dt),
        out_scale=torch.tensor([0.05], device=device),
        conv_state=conv,
        conv_weight=(torch.randn(c, k, device=device) * 0.3).bfloat16(),
    )
    if form == "verify":
        case.cu_seqlens = torch.arange(
            0, rows + 1, tokens, device=device, dtype=torch.int32
        )
        case.num_accepted_tokens = torch.randint(
            1, tokens + 1, (requests,), device=device, dtype=torch.int32
        )
    condition_conv_inputs(case, form)
    return case


def _windows(case: Case, form: str):
    """``[(row slice, conv-state slot, window [T, C, K])]`` of every live request."""
    k = case.conv_weight.shape[1]
    out = []
    if form == "decode":
        for r, slot in enumerate(case.state_indices.tolist()):
            if slot > 0:
                hist = case.conv_state[slot]  # [C, K - 1]
                win = torch.cat([hist, case.qkv[r, :, None]], dim=1)
                out.append((slice(r, r + 1), slot, win[None]))
        return out
    requests, tokens = case.state_indices.shape
    accepted = case.num_accepted_tokens.tolist()
    for n in range(requests):
        a = accepted[n]
        if case.state_indices[n, a - 1] <= 0:
            continue
        slot = int(case.state_indices[n, 0])
        rows = slice(n * tokens, (n + 1) * tokens)
        seq = torch.cat(
            [case.conv_state[slot][:, a - 1 : a + k - 2], case.qkv[rows].T], dim=1
        )
        out.append((rows, slot, seq.unfold(1, k, 1).transpose(0, 1)))
    return out


def condition_conv_inputs(case: Case, form: str, max_rounds: int = 32) -> int:
    """Move the newest input of every live (token, channel) whose conv SiLU is a near-tie until none is left; returns
    the number of inputs moved."""
    moved = 0
    for _ in range(max_rounds):
        hits = 0
        for rows, _, win in _windows(case, form):
            s, _ = conv_silu(win, case.conv_weight)
            bad = ambiguous(s)  # [T, C]
            if bad.any():
                x = case.qkv[rows]
                v = x[bad].float()
                # a step of several BF16 ulps at any magnitude
                x[bad] = (v + 0.0625 * v.abs().clamp_min(1.0)).bfloat16()
                hits += int(bad.sum())
        if hits == 0:
            return moved
        moved += hits
    raise AssertionError("conv input conditioning did not converge")


def _gdn_token(h, q, k, v, b, a, A_log, dt_bias, heads_per_k):
    """One gated-delta-rule step in FP64: ``h [Hv, Dv, Dk]``; q, k ``[Hk, Dk]`` (BF16 values), v ``[Hv, Dv]``."""
    dk = q.shape[-1]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * dk**-0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = q.repeat_interleave(heads_per_k, 0)
    k = k.repeat_interleave(heads_per_k, 0)
    g = -A_log.exp() * torch.nn.functional.softplus(a + dt_bias)
    beta = torch.sigmoid(b)
    h = h * g.exp()[:, None, None]
    innov = (v - (h * k[:, None, :]).sum(-1)) * beta[:, None]
    h = h + innov[:, :, None] * k[:, None, :]
    return h, (h * q[:, None, :]).sum(-1)


def reference(case: Case, form: str):
    """The fused chain on ``case``: ``(out_bf16, out_fp8, ssm_state, conv_state, live_rows)`` after one step."""
    hv, dv, dk = case.ssm_state.shape[1:]
    hk_dk = (case.qkv.shape[1] - hv * dv) // 2
    hk = hk_dk // dk
    rows = case.qkv.shape[0]
    k = case.conv_weight.shape[1]
    dev = case.qkv.device
    out = torch.zeros(rows, hv, dv, dtype=torch.bfloat16, device=dev)
    live = torch.zeros(rows, dtype=torch.bool, device=dev)
    ssm = case.ssm_state.clone()
    conv = case.conv_state.clone()
    ba = (case.hidden.double() @ case.ba_weight.double().T).float().bfloat16().double()
    A_log, dt_bias = case.A_log.double(), case.dt_bias.double()
    w = case.norm_weight.double()
    z = case.z.double()
    for rows_n, slot, win in _windows(case, form):
        _, x = conv_silu(win, case.conv_weight)  # [T, C] bf16
        x = x.double()
        r0 = rows_n.start
        tokens = win.shape[0]
        if form == "decode":
            src, dst = slot, [slot]
            hist = case.conv_state[slot]
            conv[slot] = torch.cat([hist[:, 1:], case.qkv[r0, :, None]], dim=1)
        else:
            n = r0 // tokens
            a = int(case.num_accepted_tokens[n])
            src = int(case.state_indices[n, a - 1])
            dst = case.state_indices[n].tolist()
            old = case.conv_state[slot]
            conv[slot] = torch.cat([old[:, a : a + k - 2], case.qkv[rows_n].T], dim=1)
        h = case.ssm_state[src].double()
        for t in range(tokens):
            r = r0 + t
            q = x[t, :hk_dk].view(hk, dk)
            kk = x[t, hk_dk : 2 * hk_dk].view(hk, dk)
            v = x[t, 2 * hk_dk :].view(hv, dv)
            h, o = _gdn_token(
                h, q, kk, v, ba[r, :hv], ba[r, hv:], A_log, dt_bias, hv // hk
            )
            if dst[t] > 0:
                ssm[dst[t]] = h.float()
            core = o.float().bfloat16().double()  # [Hv, Dv]
            inv = torch.rsqrt(core.square().mean(-1, keepdim=True) + case.eps)
            gate = z[r]
            out[r] = (core * inv * w * gate * torch.sigmoid(gate)).float().bfloat16()
            live[r] = True
    return out, quant_fp8(out, case.out_scale), ssm, conv, live
