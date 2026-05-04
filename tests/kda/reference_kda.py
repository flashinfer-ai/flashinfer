"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

KDA — slow PyTorch reference
============================

Pure-PyTorch reference implementation of the same prefill chain that
``flashinfer.kda.chunk_kda_fwd`` computes. Only used by the test suite to
sanity-check the optimized kernels — never imported from the production
fast path.

The reference factors as two stages, mirroring the optimized kernel's
contract:

1. **Gate preprocessing** — applies the same activation that the K1 kernel
   does inline:

   * ``softplus`` mode :  ``g_act = -exp(A_log) * softplus(g + dt_bias)``
   * ``safe_gate`` mode:  ``g_act = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``

2. **Chunked KDA recurrence** — the per-chunk math from
   ``fla.ops.kda.naive.naive_chunk_kda``: in-chunk causal attention with a
   block-Akk (``I + L``) factor and a recurrent state update across chunks.
"""

from __future__ import annotations

import math
import torch
from einops import rearrange


def _kda_gate_preprocess(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    safe_gate: bool,
    lower_bound: float | None,
) -> torch.Tensor:
    """Apply the same per-token gate activation that the K1 kernel does.

    g       : ``[B, T, H, K]`` bf16
    A_log   : ``[H]``        fp32  -> per-head log-decay-rate
    dt_bias : ``[H, K]`` (or ``[H*K]``) fp32, optional
    """
    H, K = g.shape[-2], g.shape[-1]
    g_f32 = g.float()
    if dt_bias is not None:
        bias = dt_bias.float()
        if bias.dim() == 1:
            bias = bias.view(H, K)
        # broadcast bias over (B, T)
        g_f32 = g_f32 + bias[None, None, :, :]
    exp_A = A_log.float().exp()  # [H]
    if safe_gate:
        if lower_bound is None:
            lower_bound = -5.0
        # g_act = lower_bound * sigmoid(exp(A_log) * (g + bias))
        g_act = lower_bound * torch.sigmoid(exp_A[None, None, :, None] * g_f32)
    else:
        # g_act = -exp(A_log) * softplus(g + bias)
        g_act = -exp_A[None, None, :, None] * torch.nn.functional.softplus(g_f32)
    return g_act


def _naive_chunk_kda_eqlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_act: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int = 64,
):
    """Chunk-wise KDA, equal-length only. ``g_act`` here is already the
    activated, *unaccumulated* gate (preprocessing applied)."""
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = T // BT
    assert T % BT == 0, "naive eqlen reference requires T % BT == 0"

    q, k, v, g_act, beta = (
        rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float)
        for x in (q, k, v, g_act, beta)
    )
    q = q * scale
    g = g_act.cumsum(-2)

    mask_diag = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = torch.zeros(*q.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]
    A = -A.masked_fill(mask_diag, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S = S + initial_state.to(q)
    o = torch.zeros_like(v)
    mask_strict_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        Aii = torch.zeros(B, H, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            Aii[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        Aii = Aii.masked_fill(mask_strict_upper, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aii @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S = S + rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i

    o = rearrange(o, 'b h n c d -> b (n c) h d').to(dtype)
    if not output_final_state:
        return o, None
    return o, S


def reference_chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
):
    """Reference KDA chunk forward — accepts the same inputs as
    ``flashinfer.kda.chunk_kda_fwd`` and returns ``(o, final_state)``.

    For varlen, processes each sequence independently and concatenates.
    """
    g_act = _kda_gate_preprocess(g, A_log, dt_bias, safe_gate, lower_bound)
    BT = chunk_size

    if cu_seqlens is None:
        return _naive_chunk_kda_eqlen(
            q, k, v, g_act, beta, scale,
            initial_state, output_final_state, chunk_size=BT,
        )

    # Varlen: split, run per-sequence, concatenate.
    cu_cpu = cu_seqlens.detach().to('cpu').tolist()
    N = len(cu_cpu) - 1
    o_chunks = []
    finals = []
    for i in range(N):
        bos, eos = cu_cpu[i], cu_cpu[i + 1]
        Tlen = eos - bos
        Tpad = ((Tlen + BT - 1) // BT) * BT
        q_i = torch.zeros(1, Tpad, q.shape[2], q.shape[3], dtype=q.dtype, device=q.device)
        k_i = torch.zeros_like(q_i)
        v_i = torch.zeros(1, Tpad, v.shape[2], v.shape[3], dtype=v.dtype, device=v.device)
        g_i_act = torch.zeros(1, Tpad, g_act.shape[2], g_act.shape[3], dtype=g_act.dtype, device=g_act.device)
        beta_i = torch.zeros(1, Tpad, beta.shape[2], dtype=beta.dtype, device=beta.device)
        q_i[:, :Tlen] = q[:, bos:eos]
        k_i[:, :Tlen] = k[:, bos:eos]
        v_i[:, :Tlen] = v[:, bos:eos]
        g_i_act[:, :Tlen] = g_act[:, bos:eos]
        beta_i[:, :Tlen] = beta[:, bos:eos]
        s0 = None if initial_state is None else initial_state[i:i+1]
        o_full, s_final = _naive_chunk_kda_eqlen(
            q_i, k_i, v_i, g_i_act, beta_i, scale,
            s0, True, chunk_size=BT,
        )
        o_chunks.append(o_full[:, :Tlen])
        finals.append(s_final)
    o_cat = torch.cat(o_chunks, dim=1)
    final_state = torch.cat(finals, dim=0) if (output_final_state and finals) else None
    return o_cat, final_state
