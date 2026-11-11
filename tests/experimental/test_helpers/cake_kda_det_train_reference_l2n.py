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

References of the in-kernel Q/K L2 norm of the Cake packed KDA chunked training package
(``use_qk_l2norm_in_kernel=True``).

Bitwise device-order references (pure torch, FP32, round to nearest even, FTZ off; the order of the device
programs and of the GDN package's ``cake_l2n`` reference, which the KDA rows must equal bit for bit):

* :func:`qk_l2norm` -- ``(q_hat, rstd)`` of BF16 raw rows ``x [..., 128]``: ``ss`` as 16 lanes of 8 channels, each
  lane ``fma.rn(x_c, x_c, s)`` in ascending channel order from zero, then the halving tree over the lanes (``h = 8,
  4, 2, 1``); ``r = div.rn(1, sqrt.rn(ss + 1e-6))``; ``q_hat = cvt.rn.bf16(mul.rn(x, r))``.
* :func:`l2n_pullback` -- the FP32 gradient of the raw rows from the FP32 gradient ``D`` of ``q_hat``:
  ``y = mul.rn(x, r)`` (never ``bf16(y)``), ``p = mul.rn(D, y)``, ``dot`` the halving tree within each 32-channel
  quadrant (``h = 16 .. 1``) then ``((S0 + S1) + S2) + S3``, ``dq = mul.rn(r, sub.rn(D, mul.rn(y, dot)))``; the
  caller rounds once to BF16.

The ``fma.rn`` and the correctly rounded ``sqrt`` / ``div`` are emulated exactly through FP64 (one FP64 rounding
then one FP32 rounding equals the single FP32 rounding: 53 >= 2 * 24 + 2).

The exact reference of the whole layer (:func:`kda_exact_l2n_with_grads`) normalizes the raw rows in FP64 (``x /
sqrt(sum x^2 + 1e-6)``) inside the autograd graph of the exact per-document recurrence, so its ``dq`` / ``dk`` are
the exact raw-row gradients the accuracy rows compare against.
"""

from __future__ import annotations

import torch

from .cake_kda_det_train_reference import (
    GRAD_NAMES,
    RAW_GRAD_NAMES,
    LOWER_BOUND,
    kda_exact_forward,
    raw_gate,
)

HEAD_DIM = 128
EPS = 1e-6  # fp32(1e-6) = 0x358637BD
F32, F64, BF16 = torch.float32, torch.float64, torch.bfloat16


def _tree(a: torch.Tensor) -> torch.Tensor:
    """The halving tree over the last axis (a power of two): ``a[:h] + a[h:2h]`` for ``h = n/2, ..., 1``."""
    n = a.shape[-1]
    while n > 1:
        n //= 2
        a = a[..., :n] + a[..., n : 2 * n]
    return a[..., 0]


def sum_of_squares(x: torch.Tensor) -> torch.Tensor:
    """FP32 ``ss`` of BF16 / FP32 rows ``x [..., 128]`` in the device order (16 lanes of 8 FMA steps, then the lane
    tree)."""
    lanes = x.to(F32).reshape(*x.shape[:-1], 16, 8).to(F64)
    s = torch.zeros(lanes.shape[:-1], dtype=F32, device=x.device)
    for c in range(8):
        s = (lanes[..., c] * lanes[..., c] + s.to(F64)).to(F32)  # fma.rn(x_c, x_c, s)
    return _tree(s)


def rstd(ss: torch.Tensor) -> torch.Tensor:
    """``div.rn(1, sqrt.rn(add.rn(ss, 1e-6)))`` in FP32, each step correctly rounded."""
    t = (ss.to(F64) + torch.tensor(EPS, dtype=F32).to(F64)).to(F32)
    s = torch.sqrt(t.to(F64)).to(F32)
    return (1.0 / s.to(F64)).to(F32)


def qk_l2norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q_hat BF16 [..., 128] contiguous, rstd FP32 [...])`` of BF16 rows ``x`` (any strides)."""
    if x.shape[-1] != HEAD_DIM or x.dtype != BF16:
        raise ValueError(
            f"qk_l2norm: BF16 rows [..., {HEAD_DIM}] expected, got {x.dtype} {tuple(x.shape)}"
        )
    r = rstd(sum_of_squares(x))
    return (x.to(F32) * r[..., None]).to(BF16).contiguous(), r.contiguous()


def qk_rstd(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """The saved set's ``qk_rstd`` FP32 ``[2, S, H]`` (plane 0 of ``q``, plane 1 of ``k``) of raw ``[S, H, 128]``
    rows."""
    return torch.stack((qk_l2norm(q)[1], qk_l2norm(k)[1]), 0).contiguous()


def l2n_dot(p: torch.Tensor) -> torch.Tensor:
    """The pullback's FP32 ``dot`` of products ``p [..., 128]``: per 32-channel quadrant the halving tree, then
    ``((S0 + S1) + S2) + S3``."""
    S = _tree(p.reshape(*p.shape[:-1], 4, 32))
    return ((S[..., 0] + S[..., 1]) + S[..., 2]) + S[..., 3]


def l2n_pullback(x: torch.Tensor, r: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """FP32 gradient of the raw rows ``x`` (BF16 ``[..., 128]``) from ``r`` (FP32 ``[...]``) and the FP32 gradient
    ``D`` of ``q_hat``, in the device order; the caller rounds once to BF16."""
    rr = r.to(F32)[..., None]
    y = x.to(F32) * rr
    Df = D.to(F32)
    dot = l2n_dot(Df * y)
    return rr * (Df - y * dot[..., None])


def _normalize64(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + EPS)


def kda_exact_l2n_with_grads(
    q, k, v, g, beta, dout, cu_seqlens, scale, dtype=torch.float64
):
    """The exact forward output and the five input gradients (:data:`GRAD_NAMES` order) with the in-kernel Q/K L2
    norm: the raw BF16 ``q`` / ``k`` are normalized in ``dtype`` inside the graph, so ``dq`` / ``dk`` are the raw
    rows' exact gradients."""
    leaves = [x.detach().to(dtype).requires_grad_(True) for x in (q, k, v, g, beta)]
    q64, k64, v64, g64, b64 = leaves
    o = kda_exact_forward(
        _normalize64(q64),
        _normalize64(k64),
        v64,
        g64,
        b64,
        cu_seqlens,
        scale,
        dtype=dtype,
    )
    grads = torch.autograd.grad(o, leaves, dout.to(dtype))
    return o.detach(), dict(zip(GRAD_NAMES, (x.detach() for x in grads), strict=True))


def kda_exact_raw_l2n_with_grads(
    q,
    k,
    v,
    raw,
    beta,
    A_log,
    dt_bias,
    dout,
    cu_seqlens,
    scale,
    lower_bound: float = LOWER_BOUND,
    dtype=torch.float64,
):
    """:func:`kda_exact_l2n_with_grads` in the raw gate mode (:data:`RAW_GRAD_NAMES` order)."""
    leaves = [
        x.detach().to(dtype).requires_grad_(True)
        for x in (q, k, v, raw, beta, A_log, dt_bias)
    ]
    q64, k64, v64, raw64, beta64, a64, b64 = leaves
    g64 = raw_gate(raw64, a64, b64, lower_bound, dtype)
    o = kda_exact_forward(
        _normalize64(q64),
        _normalize64(k64),
        v64,
        g64,
        beta64,
        cu_seqlens,
        scale,
        dtype=dtype,
    )
    grads = torch.autograd.grad(o, leaves, dout.to(dtype))
    return o.detach(), dict(
        zip(RAW_GRAD_NAMES, (x.detach() for x in grads), strict=True)
    )


__all__ = [
    "EPS",
    "kda_exact_l2n_with_grads",
    "kda_exact_raw_l2n_with_grads",
    "l2n_dot",
    "l2n_pullback",
    "qk_l2norm",
    "qk_rstd",
    "rstd",
    "sum_of_squares",
]
