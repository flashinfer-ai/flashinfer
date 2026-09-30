"""Sequential GDN oracle and a separately rounded chunk-algebra mirror.

The sequential recurrence is the correctness oracle. The chunk mirror exposes
preparation operands for diagnosing the CuTe implementation; it is not a
production fallback. States use [head, value_dim, key_dim] throughout.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .._shared.kda_math import l2_normalize
from .._shared.delta_prefill.reference import (
    MirrorPolicy, _bf16, _neumann_inverse, _recur_tile, _scalar, _validate_packed,
)

HEAD_DIM = 128
CHUNK = 16


def gdn_log_decay(a, dt_bias, A_log):
    """Natural-log state decay for each token and value head, in FP32."""
    return -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float(), threshold=20)


def gdn_beta(b):
    """GDN's update gate, rounded through BF16 as in recurrent decode."""
    return torch.sigmoid(b.float()).to(torch.bfloat16).float()


def recurrent_gdn(
    q, k, v, a, b, A_log, dt_bias, *, initial_state,
    checkpoint_offset=-1, scale=None, eps=1e-6, qk_l2norm=True,
):
    """Return BF16 token outputs, FP32 final state, and an optional checkpoint."""
    scale = HEAD_DIM**-0.5 if scale is None else float(scale)
    heads = v.shape[1]
    if heads != 3 * q.shape[1] or q.shape != k.shape:
        raise ValueError("GDN requires matching Q/K and three value heads per key head")
    qf = l2_normalize(q, eps) if qk_l2norm else q.float()
    kf = l2_normalize(k, eps) if qk_l2norm else k.float()
    head_map = torch.arange(heads, device=q.device) // 3
    qf = qf[:, head_map] * scale
    kf = kf[:, head_map]
    decay = torch.exp(gdn_log_decay(a, dt_bias, A_log))
    beta = gdn_beta(b)
    state = initial_state.float().clone()
    output = torch.empty_like(v)
    checkpoint = state.clone() if checkpoint_offset == 0 else None
    for t in range(q.shape[0]):
        state = state * decay[t, :, None, None]
        delta = v[t].float() - torch.einsum("hvk,hk->hv", state, kf[t])
        state = state + (delta * beta[t, :, None])[:, :, None] * kf[t, :, None, :]
        output[t] = torch.einsum("hvk,hk->hv", state, qf[t]).to(torch.bfloat16)
        if t + 1 == checkpoint_offset:
            checkpoint = state.clone()
    return output, state, checkpoint


def prepare_chunk(q, k, a, b, A_log, dt_bias, *, scale=None, eps=1e-6, qk_l2norm=True):
    """Prepare one padded 16-token tile with the production rounding points."""
    rows = q.shape[0]
    if not 0 < rows <= CHUNK:
        raise ValueError("a chunk must contain 1..16 tokens")
    scale = HEAD_DIM**-0.5 if scale is None else float(scale)
    heads = a.shape[1]
    head_map = torch.arange(heads, device=q.device) // 3
    qf = l2_normalize(q, eps) if qk_l2norm else q.float()
    kf = l2_normalize(k, eps) if qk_l2norm else k.float()
    qn = torch.zeros((heads, CHUNK, HEAD_DIM), device=q.device)
    kn = torch.zeros_like(qn)
    qn[:, :rows] = qf[:, head_map].transpose(0, 1)
    kn[:, :rows] = kf[:, head_map].transpose(0, 1)
    log = torch.zeros((heads, CHUNK), device=q.device)
    log[:, :rows] = gdn_log_decay(a, dt_bias, A_log).transpose(0, 1)
    beta = torch.zeros_like(log)
    beta[:, :rows] = gdn_beta(b).transpose(0, 1)
    cumulative = torch.zeros_like(log)
    suffix = torch.zeros_like(log)
    running = torch.zeros(heads, device=q.device)
    for t in range(CHUNK):
        running = running + log[:, t]
        cumulative[:, t] = running
    running = torch.zeros_like(running)
    for t in range(CHUNK - 1, -1, -1):
        suffix[:, t] = running
        running = running + log[:, t]
    pair = torch.zeros((heads, CHUNK, CHUNK), device=q.device)
    for j in range(CHUNK):
        running = torch.zeros_like(running)
        pair[:, j, j] = 1
        for i in range(j + 1, CHUNK):
            running = running + log[:, i]
            pair[:, i, j] = torch.exp(running)
    lower = beta[:, :, None] * (_bf16(kn) @ _bf16(kn).transpose(-1, -2)) * pair
    lower = torch.tril(lower, diagonal=-1)
    inverse = _neumann_inverse(lower, CHUNK)
    mqk = (_bf16(qn) @ _bf16(kn).transpose(-1, -2)) * pair * scale
    lam = torch.exp(cumulative)[:, :, None]
    return {
        "q_tilde": _bf16(qn * lam * scale),
        "k_tilde": _bf16(kn * lam),
        "k_r": _bf16(kn * torch.exp(suffix)[:, :, None]),
        "lambda_c": torch.exp(cumulative[:, -1, None]).expand(heads, HEAD_DIM),
        "beta": beta, "L": lower, "inv": inverse,
        "inv_op": _bf16(inverse), "mqk": _bf16(mqk),
    }


def chunk_mirror(q, k, v, a, b, A_log, dt_bias, *, initial_state,
                 checkpoint_offset=-1, scale=None, eps=1e-6, qk_l2norm=True):
    """Run the chunk algebra; use recurrent_gdn as the independent oracle."""
    state = initial_state.float().clone()
    output = torch.empty_like(v)
    checkpoint = state.clone() if checkpoint_offset == 0 else None
    for start in range(0, q.shape[0], CHUNK):
        end = min(start + CHUNK, q.shape[0])
        prep = prepare_chunk(q[start:end], k[start:end], a[start:end], b[start:end], A_log, dt_bias,
                             scale=scale, eps=eps, qk_l2norm=qk_l2norm)
        values = torch.zeros((CHUNK, *v.shape[1:]), dtype=v.dtype, device=v.device)
        values[:end-start] = v[start:end]
        out, state, _ = _recur_tile(state, values, prep, rows=end-start, chunk=CHUNK, policy=MirrorPolicy())
        output[start:end] = out[:end-start]
        if end == checkpoint_offset:
            checkpoint = state.clone()
    return output, state, checkpoint


def prefill_gdn(q, k, v, a, b, A_log, dt_bias, recurrent_state, cu_seqlens,
                initial_state_indices, final_state_indices, checkpoint_state_indices,
                checkpoint_offsets, num_seqs, num_tokens, *, scale=None, eps=1e-6,
                qk_l2norm=True, null_state_index=None, output=None):
    """Apply the sequential oracle to packed requests and caller-owned state slots."""
    scale = HEAD_DIM**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("scale and eps must be finite and positive")
    spans = _validate_packed(
        cu_seqlens=cu_seqlens, initial_state_indices=initial_state_indices,
        final_state_indices=final_state_indices, checkpoint_state_indices=checkpoint_state_indices,
        checkpoint_offsets=checkpoint_offsets, num_seqs=_scalar(num_seqs), num_tokens=_scalar(num_tokens),
        token_capacity=q.shape[0], seq_capacity=cu_seqlens.numel()-1,
        state_slots=recurrent_state.shape[0], chunk=CHUNK, null_state_index=null_state_index,
    )
    if output is None:
        output = torch.zeros_like(v)
    for seq, (start, end) in enumerate(spans):
        initial, final, checkpoint = (int(t[seq]) for t in (
            initial_state_indices, final_state_indices, checkpoint_state_indices))
        offset = int(checkpoint_offsets[seq])
        state = (torch.zeros_like(recurrent_state[0]) if initial == null_state_index
                 else recurrent_state[initial])
        out, state, saved = recurrent_gdn(
            q[start:end], k[start:end], v[start:end], a[start:end], b[start:end], A_log, dt_bias,
            initial_state=state, checkpoint_offset=offset if offset > 0 else -1,
            scale=scale, eps=eps, qk_l2norm=qk_l2norm,
        )
        output[start:end] = out
        if saved is not None and checkpoint != null_state_index:
            recurrent_state[checkpoint].copy_(saved)
        if final != null_state_index:
            recurrent_state[final].copy_(state)
    return output


__all__ = ["chunk_mirror", "gdn_beta", "gdn_log_decay", "prefill_gdn", "prepare_chunk", "recurrent_gdn"]
