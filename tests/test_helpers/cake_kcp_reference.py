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

FP64 reference of the packed KDA training problem for the context-parallel (KCP) tests, generalized from
``tests/test_helpers/cake_kda_train_reference.py`` (per-document token recurrence, 64-token checkpointed blocks) to
a token interval started from a carried state, in broadcast-multiply form (no size-1 bmm, so the module also runs
on sm_107a where Triton cannot build the bmm lowering).

Semantics (FLA ``naive_recurrent_kda``): per token t of one document, state S in R^{K x V} (per head),

    S   = S * exp(g_t)[:, None]                   # per-key-dimension natural-log decay
    S   = S + (beta_t * k_t) (x) (v_t - k_t^T S)  # delta rule
    o_t = scale * q_t^T S

with the BF16 q / k / v / beta and the FP32 g taken as exact FP64 values.

API
---
recur_interval(S0, q, k, v, g, beta, scale=SCALE, block=64) -> (o, S_T)
    FP64 recurrence over ONE token interval of ONE document from the initial state ``S0`` [H, 128, 128] (FP64; may
    require grad).  ``q, k, v, g`` [T, H, 128], ``beta`` [T, H] (any float dtype, cast to FP64 inside; T may be 0).
    Checkpointed in ``block``-token blocks (``use_reentrant=False``).  Differentiable w.r.t. ``S0`` and all operands:
    this is the oracle of a fragment's forward from a carried state (``S_in`` -> local forward -> ``S_out``).
recur_interval_adjoint(S0, q, k, v, g, beta, dout, dS_T=None, scale=SCALE) -> dict
    The adjoint of one interval: ``{"o", "S_T", "dq", "dk", "dv", "dg", "dbeta", "dS0"}`` = gradients of
    <dout, o> + <dS_T, S_T> (``dS_T=None``: no final-state cotangent).  ``dS0`` is the FP64 oracle of a fragment's
    ``dS_in(f) = M_f^T dS_out(f) + dS_in^local(f)``, i.e. the cotangent a predecessor rank must receive.
transfer(S0, q, k, v, g, beta) -> (M, B)   [documentation helper for unit tests]
    The affine map of an interval, S_T = M . S_0 + B in the sense S_T[h] = M[h] @ S_0[h] + B[h] (the per-head
    K x K matrix M acts on the key axis of the state; the V axis is untouched by the recurrence), obtained in FP64
    from ``recur_interval`` by propagating the identity basis.  Cost O(K) recurrences: for unit tests on short
    intervals only.
reference_pack(row, device, progress=None) -> (out, grads)
    The whole GLOBAL pack of ``row`` (``cake_kcp_inputs.global_operands``), each document from the zero state, with
    autograd per document; ``grads = {"dq", "dk", "dv", "dg", "dbeta"}``; all FP64 ``[T, H, 128]`` / ``[T, H]``.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers.cake_kcp_inputs import global_operands
from tests.test_helpers.cake_kcp_shapes import D, Row

SCALE = D**-0.5
BLOCK = 64
# autograd.grad order of the leaves (q, k, v, g, beta)
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")


def _block(S0, q, k, v, g, beta, scale):
    # S0 [H, K, V]; q, k, v, g [T, H, D]; beta [T, H]  (all FP64).  Broadcast multiplies + FP64 sums: no bmm/einsum.
    St = S0
    outs = []
    for t in range(q.shape[0]):
        St = St * torch.exp(g[t])[:, :, None]
        kt = k[t]
        pred = (kt[:, :, None] * St).sum(1)  # k_t^T S -> [H, V]
        # (beta_t k_t) (x) (v_t - pred)
        St = St + (beta[t][:, None] * kt)[:, :, None] * (v[t] - pred)[:, None, :]
        outs.append(scale * (q[t][:, :, None] * St).sum(1))  # q_t^T S -> [H, V]
    return torch.stack(outs, 0), St


def recur_interval(S0, q, k, v, g, beta, scale: float = SCALE, block: int = BLOCK):
    f64 = torch.float64
    q, k, v, g, beta = (x.to(f64) for x in (q, k, v, g, beta))
    St = S0.to(f64)
    T, H = q.shape[0], q.shape[1]
    outs = []
    for b0 in range(0, T, block):
        b1 = min(b0 + block, T)
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
        outs.append(ob)
    if outs:
        o = torch.cat(outs, 0)
    else:
        o = q.new_zeros(0, H, v.shape[-1])
    return o, St


def recur_interval_adjoint(
    S0, q, k, v, g, beta, dout, dS_T=None, scale: float = SCALE
) -> dict:
    f64 = torch.float64
    leaves = [x.detach().to(f64).requires_grad_(True) for x in (S0, q, k, v, g, beta)]
    o, S_T = recur_interval(*leaves, scale=scale)
    outputs, cot = [o], [dout.to(f64)]
    if dS_T is not None:
        outputs.append(S_T)
        cot.append(dS_T.to(f64))
    grads = torch.autograd.grad(outputs, leaves, cot, allow_unused=True)
    grads = [
        torch.zeros_like(l) if gr is None else gr
        for l, gr in zip(leaves, grads, strict=True)
    ]
    return {
        "o": o.detach(),
        "S_T": S_T.detach(),
        "dS0": grads[0],
        "dq": grads[1],
        "dk": grads[2],
        "dv": grads[3],
        "dg": grads[4],
        "dbeta": grads[5],
    }


def transfer(
    S0, q, k, v, g, beta, scale: float = SCALE
) -> tuple[torch.Tensor, torch.Tensor]:
    """(M [H, K, K], B [H, K, V]) with S_T[h] = M[h] @ S_0[h] + B[h]; FP64; O(K) recurrences (unit tests only)."""
    f64 = torch.float64
    H, K, V = S0.shape
    with torch.no_grad():
        _, B = recur_interval(
            torch.zeros(H, K, V, dtype=f64, device=q.device),
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
        )
        M = torch.zeros(H, K, K, dtype=f64, device=q.device)
        for i in range(K):
            E = torch.zeros(H, K, V, dtype=f64, device=q.device)
            # basis along the key axis, broadcast over V: S_T = M[:, :, i] (x) 1_V + B
            E[:, i, :] = 1.0
            _, ST = recur_interval(E, q, k, v, g, beta, scale=scale)
            M[:, :, i] = (ST - B)[:, :, 0]
    return M, B


def reference_pack(
    row: Row, device, progress=None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``(out, grads)`` of the whole pack of ``row`` on ``device`` (see the module docstring); ``progress(msg)``, when
    given, is called once per finished document."""
    f64 = torch.float64
    H = row.h
    ops = global_operands(row, H, device)
    T = row.total
    out = torch.zeros(T, H, D, dtype=f64, device=device)
    grads = {
        n: torch.zeros(ops[src].shape, dtype=f64, device=device)
        for n, src in zip(GRAD_NAMES, ("q", "k", "v", "g", "beta"), strict=True)
    }
    cu = row.cu_seqlens
    for d in range(len(row.lengths)):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            continue
        leaves = [
            ops[n][s0:s1].to(f64).requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        S0 = torch.zeros(H, D, D, dtype=f64, device=device)
        o, _ = recur_interval(S0, *leaves, scale=SCALE)
        gl = torch.autograd.grad(o, leaves, ops["dout"][s0:s1].to(f64))
        out[s0:s1] = o.detach()
        for n, gr in zip(GRAD_NAMES, gl, strict=True):
            grads[n][s0:s1] = gr
        del leaves, o, gl
        if progress is not None:
            progress(f"{row.name} doc {d}/{len(row.lengths)} done, {s1 - s0} tokens")
    return out, grads
