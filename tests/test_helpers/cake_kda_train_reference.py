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

Seeded inputs and the exact per-document reference of the packed KDA chunked
training problem (GLM-5.3-Flash geometry: head dimension 128, natural-log
decay gate, sigmoid beta), shared by ``tests/experimental/test_cake_kda_train.py``,
``tests/experimental/test_cake_kda_train_api.py`` and
``benchmarks/bench_cake_kda_train.py``.

The reference is the token recurrence of one document, in FP64 with autograd,
state ``S`` in ``R^{K x V}`` starting from zero::

    S   = S * exp(g_t)[:, None]                   # per-key-dimension decay
    S   = S + (beta_t * k_t) (x) (v_t - k_t^T S)  # delta rule
    o_t = scale * q_t^T S

with the BF16 ``q``, ``k``, ``v``, ``beta`` and the FP32 ``g`` taken as exact
FP64 values.  The recurrence is checkpointed in 64-token blocks so the
backward of a few thousand tokens fits in memory.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import torch
from torch.utils.checkpoint import checkpoint

D = 128
LOWER_BOUND = -5.0  # the gate's lower bound: g = -5 * sigmoid(...)
GRAD_NAMES = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd.grad order of the leaves (q, k, v, g, beta)


def cu_seqlens_of(lengths: Sequence[int]) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    return cu


def make_inputs(
    lengths: Sequence[int],
    H: int,
    device: str | torch.device = "cuda",
    *,
    seed: int | None = None,
    with_dout: bool = True,
    batched: bool = False,
) -> dict[str, torch.Tensor]:
    """The packed GLM operands of documents of ``lengths`` tokens with ``H`` heads: ``q, k, v [S, H, 128]`` BF16
    (``q`` / ``k`` L2-normalized in FP32 with eps 1e-6 before the BF16 store), ``g [S, H, 128]`` FP32
    ``= -5 * sigmoid(exp(A_log) * (raw + dt_bias))``, ``beta [S, H]`` BF16 ``= sigmoid(logits)``, ``cu_seqlens``
    int32 on the device and (with ``with_dout``) ``dout [S, H, 128]`` BF16.  ``batched`` adds the leading batch
    axis of 1.  Device-independent: a CPU generator seeded from ``seed`` (default: a CRC of the shape)."""
    S = int(sum(lengths))
    if seed is None:
        seed = zlib.crc32(f"{list(lengths)}:{H}".encode())
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def l2norm_bf16(x: torch.Tensor) -> torch.Tensor:
        return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    q = l2norm_bf16(randn(S, H, D))
    k = l2norm_bf16(randn(S, H, D))
    v = (randn(S, H, D) * 0.5).to(torch.bfloat16)
    a_log = torch.log(
        torch.rand(H, generator=gen, dtype=torch.float32) + 1.0
    )  # decay rates in [1, 2)
    dt_bias = randn(H, D) * 0.5
    raw = randn(S, H, D)
    g = LOWER_BOUND * torch.sigmoid(
        torch.exp(a_log)[None, :, None] * (raw + dt_bias[None])
    )
    beta = torch.sigmoid(randn(S, H)).to(torch.bfloat16)
    cu = torch.tensor(cu_seqlens_of(lengths), dtype=torch.int32)
    out = {"q": q, "k": k, "v": v, "g": g.float(), "beta": beta, "cu_seqlens": cu}
    if with_dout:
        out["dout"] = (randn(S, H, D) * 0.5).to(torch.bfloat16)
    tensors = {name: t.to(device) for name, t in out.items()}
    if batched:
        for name in ("q", "k", "v", "g", "beta", "dout"):
            if name in tensors:
                tensors[name] = tensors[name].unsqueeze(0)
    return tensors


def _block(S0, q, k, v, g, beta, scale):
    # S0 [H, K, V]; q, k, v, g [T, H, D]; beta [T, H]  (all FP64)
    St = S0
    outs = []
    for t in range(q.shape[0]):
        St = St * torch.exp(g[t])[:, :, None]
        kt = k[t]
        pred = torch.einsum("hk,hkv->hv", kt, St)
        St = St + torch.einsum("hk,hv->hkv", beta[t][:, None] * kt, v[t] - pred)
        outs.append(scale * torch.einsum("hk,hkv->hv", q[t], St))
    return torch.stack(outs, 0), St


def kda_exact_forward(
    q, k, v, g, beta, cu_seqlens, scale, block=64, dtype=torch.float64
):
    """``q, k, v, g [S, H, D]``, ``beta [S, H]``, ``cu_seqlens`` 1-D int (device or host): ``out [S, H, D]`` in
    ``dtype`` from the exact per-document recurrence."""
    S, H, K = q.shape
    V = v.shape[-1]
    o = torch.zeros(S, H, V, dtype=dtype, device=q.device)
    cu = [int(x) for x in cu_seqlens.tolist()]
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            continue
        St = torch.zeros(H, K, V, dtype=dtype, device=q.device)
        for b0 in range(s0, s1, block):
            b1 = min(b0 + block, s1)
            ob, St = checkpoint(
                _block,
                St,
                q[b0:b1],
                k[b0:b1],
                v[b0:b1],
                g[b0:b1],
                beta[b0:b1],
                scale,
                use_reentrant=False,
            )
            o[b0:b1] = ob
    return o


def kda_exact_with_grads(
    q, k, v, g, beta, dout, cu_seqlens, scale, dtype=torch.float64
):
    """The exact forward output and the five input gradients (``GRAD_NAMES`` order), all in ``dtype``, from the
    GLM operands (BF16 ``q, k, v, beta, dout``, FP32 ``g``) of shape ``[S, H, D]`` / ``[S, H]``."""
    leaves = [x.detach().to(dtype).requires_grad_(True) for x in (q, k, v, g, beta)]
    o = kda_exact_forward(*leaves, cu_seqlens, scale, dtype=dtype)
    grads = torch.autograd.grad(o, leaves, dout.to(dtype))
    return o.detach(), dict(zip(GRAD_NAMES, (x.detach() for x in grads), strict=True))


def error_metrics(cand: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """max |err|, RMS err, relative RMS (err RMS / ref RMS) and the reference RMS, computed in FP64."""
    c = cand.detach().to(torch.float64)
    r = ref.detach().to(torch.float64)
    err = c - r
    rms_ref = torch.sqrt(torch.mean(r * r)).item() if r.numel() else 0.0
    rms_err = torch.sqrt(torch.mean(err * err)).item() if err.numel() else 0.0
    return {
        "max_abs": err.abs().max().item() if err.numel() else 0.0,
        "rms": rms_err,
        "rel_rms": (rms_err / rms_ref)
        if rms_ref > 0
        else float("inf")
        if rms_err > 0
        else 0.0,
        "ref_rms": rms_ref,
    }


def elementwise_gate(
    cand: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float
) -> dict:
    """``|cand - ref| <= atol + rtol * |ref|`` elementwise in FP32; the mismatch and non-finite counts."""
    c = cand.detach().to(torch.float32)
    r = ref.detach().to(torch.float32)
    bad = (c - r).abs() > (atol + rtol * r.abs())
    nonfinite = int((~torch.isfinite(c)).sum().item())
    mismatch = int(bad.sum().item())
    return {
        "mismatch": mismatch,
        "numel": int(c.numel()),
        "nonfinite": nonfinite,
        "pass": mismatch == 0 and nonfinite == 0,
    }
