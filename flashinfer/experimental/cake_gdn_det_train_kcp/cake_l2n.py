"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Pure-torch FP32 reference of the per-head Q/K L2 norm of the KCP GDN route (``use_qk_l2norm_in_kernel=True``):
the forward ``q_hat = bf16(x * r)``, ``r = 1 / sqrt(sum x^2 + 1e-6)`` per token and 128-channel head, and its
pullback, in the fixed operation orders of the device programs (``fwd_prep_l2n`` / ``qk_apply`` for the forward,
``bwd_dqkg_l2n`` for the pullback).  It imports no kernel; the fake backend, the host's ``group_reduce="host"`` path
and the tests use it, and an FP64 harness can call it for the in-kernel form.

Numeric policy: FTZ off (subnormal inputs and results are kept), round to nearest even everywhere, one BF16
rounding per output.  Every step is a plain IEEE FP32 operation of torch except the ``fma.rn`` of the sum of
squares, which is emulated exactly: ``x^2`` of a BF16 ``x`` is exact in FP64, the FP64 sum is rounded once to FP64
and once to FP32, and that double rounding equals the single FP32 rounding of the exact ``x^2 + s`` (53 >= 2 * 24 + 2;
the tests check it against an exact rational oracle, subnormals and overflow included).  The reference assumes
torch's flush-denormal mode is off (the torch default); it does not change that global mode.

Forward, per token ``t`` and head ``h`` (``x = fp32(q[t, h, :])``)::

    ss: lane c (c = 0..15) holds channels 8c .. 8c+7; s = fma.rn(x0, x0, 0); s = fma.rn(xk, xk, s), k = 1..7
        ascending; then s += shfl_xor(s, m) for m = 8, 4, 2, 1   (torch: view(16, 8) FMA chains, then the halving
        tree a[:h] + a[h:2h], h = 8, 4, 2, 1)
    r     = div.rn(1, sqrt.rn(ss + 1e-6))                     FP32 (eps = fp32(1e-6) = 0x358637BD); saved as rstd
    q_hat = cvt.rn.bf16(mul.rn(x, r))

Pullback (``D`` = the FP32 gradient of ``q_hat``: for GDN the complete FP32 group sum of the value heads)::

    y   = mul.rn(x, r)                                        FP32 from the raw q and the saved r (never bf16(y))
    p   = mul.rn(D, y)
    dot = within each 32-channel quadrant the channel tree, pairs c <-> c^m, m = 16, 8, 4, 2, 1 (torch: view(4, 32),
          halving tree h = 16 .. 1), then the quadrants ((S0 + S1) + S2) + S3
    a   = mul.rn(y, dot);  b = sub.rn(D, a);  dq = mul.rn(r, b)   then cvt.rn.bf16 once (by the caller)
"""

from __future__ import annotations

import torch

HEAD_DIM = 128
EPS = 1e-6  # fp32(1e-6) = 0x358637BD
SS_LANES = 16  # forward: lanes per (token, head) row, 8 channels each
SS_PER_LANE = HEAD_DIM // SS_LANES
DOT_QUADRANTS = 4  # pullback: 32-channel quadrants
DOT_WIDTH = HEAD_DIM // DOT_QUADRANTS
F32, F64, BF16 = torch.float32, torch.float64, torch.bfloat16


def _halving(a: torch.Tensor) -> torch.Tensor:
    """The fixed tree over the last axis (a power of two): ``a[:h] + a[h:2h]`` for ``h = n/2, ..., 1``."""
    h = a.shape[-1] // 2
    while h >= 1:
        a = a[..., :h] + a[..., h : 2 * h]
        h //= 2
    return a[..., 0]


def sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """FP32 ``ss`` of FP32 rows ``x`` ``[..., 128]`` (BF16 values) in the forward order above."""
    xd = x.to(F32).reshape(*x.shape[:-1], SS_LANES, SS_PER_LANE).to(F64)
    s = (xd[..., 0] * xd[..., 0]).to(
        F32
    )  # fma.rn(x0, x0, 0): x0^2 exact in FP64, one rounding
    for k in range(1, SS_PER_LANE):
        s = (xd[..., k] * xd[..., k] + s.to(F64)).to(
            F32
        )  # fma.rn(xk, xk, s), emulated exactly
    return _halving(s)


def rstd_of(ss: torch.Tensor) -> torch.Tensor:
    """``r = div.rn(1, sqrt.rn(ss + 1e-6))`` (FP32)."""
    # each op in FP64, rounded once to FP32, is the correctly rounded FP32 op (FP64 carries > 2 x 24 + 2 bits);
    # torch's FP32 sqrt on CPU is not sqrt.rn bit for bit
    t = (ss.to(F64) + torch.tensor(EPS, dtype=F32).to(F64)).to(F32)
    s = torch.sqrt(t.to(F64)).to(F32)
    return (1.0 / s.to(F64)).to(F32)


def qk_apply_ref(x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``q_hat = cvt.rn.bf16(mul.rn(x, r))`` of BF16 rows ``x`` ``[..., 128]`` and FP32 ``r`` ``[...]``."""
    return (x.to(F32) * r[..., None]).to(BF16)


def qk_l2norm_ref(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_hat, rstd)`` of BF16 ``x`` ``[..., 128]`` (any strides): ``q_hat`` BF16 contiguous like ``x``, ``rstd``
    FP32 ``[...]``.  Bitwise the device forward (``fwd_prep_l2n``; ``qk_apply`` on the same ``rstd``)."""
    if x.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"qk_l2norm_ref: the last axis must be {HEAD_DIM}, got {tuple(x.shape)}"
        )
    if x.dtype != BF16:
        raise ValueError(f"qk_l2norm_ref: expected bfloat16 rows, got {x.dtype}")
    r = rstd_of(sum_of_squares(x.to(F32)))
    return qk_apply_ref(x, r).contiguous(), r.contiguous()


def l2n_dot(p: torch.Tensor) -> torch.Tensor:
    """The pullback's FP32 ``dot`` of products ``p`` ``[..., 128]``: the channel tree in each 32-channel quadrant,
    then ``((S0 + S1) + S2) + S3``."""
    S = _halving(p.reshape(*p.shape[:-1], DOT_QUADRANTS, DOT_WIDTH))
    return ((S[..., 0] + S[..., 1]) + S[..., 2]) + S[..., 3]


def l2n_pullback(x_raw: torch.Tensor, r: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """FP32 gradient of the raw rows: ``r * (D - y * dot(D * y))`` with ``y = x_raw * r`` (FP32, from the raw BF16
    rows and the saved ``r``), in the order above; the caller rounds once to BF16.  ``x_raw`` BF16 ``[..., 128]``,
    ``r`` FP32 ``[...]``, ``D`` FP32 ``[..., 128]``."""
    rr = r.to(F32)[..., None]
    y = x_raw.to(F32) * rr
    Df = D.to(F32)
    dot = l2n_dot(Df * y)
    return rr * (Df - y * dot[..., None])


def flush_denormal_active(device: torch.device | str = "cpu") -> bool:
    """Whether FP32 subnormal results are flushed to zero on ``device`` (the reference needs this ``False``)."""
    tiny = torch.tensor(2.0**-126, dtype=F32, device=device)
    return bool((tiny * torch.tensor(0.5, dtype=F32, device=device)).item() == 0.0)


__all__ = [
    "EPS",
    "flush_denormal_active",
    "l2n_dot",
    "l2n_pullback",
    "qk_apply_ref",
    "qk_l2norm_ref",
    "rstd_of",
    "sum_of_squares",
]
