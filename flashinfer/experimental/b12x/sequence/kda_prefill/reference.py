"""PyTorch oracles for chunked lower-bounded KDA prefill.

Three oracles share one contract:

``recurrent_kda`` runs the fp32 token recurrence for one packed sequence and is
the ground truth. ``prefill_kda`` applies it to a packed batch over a
recurrent-state pool, honouring the same packed-metadata contract as the kernel.
``prefill_kda_chunk_mirror`` implements the kernel's chunked algorithm with its
exact rounding points so kernel stages can be compared tensor by tensor; a
``MirrorPolicy`` selects alternative precisions for offline studies and a
``MirrorTrace`` records every per-tile intermediate.

State layout: the pool is ``[slot, head, value_dim, key_dim]`` (the transpose of
the mathematical ``[key_dim, value_dim]`` state); oracles keep that orientation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .._shared.kda_math import KDA_HEAD_DIM, kda_beta, kda_log_decay, l2_normalize

from .._shared.delta_prefill.reference import (
    MirrorPolicy, _bf16, _neumann_inverse, _recur_tile, _scalar, _validate_packed,
)

LOG2E = 1.4426950408889634


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float,
    initial_state: torch.Tensor,
    checkpoint_offset: int = -1,
    scale: float | None = None,
    eps: float = 1e-6,
    qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the fp32 token recurrence for one sequence.

    ``q, k, v, raw_g`` are ``[T, heads, 128]``, ``raw_beta`` is ``[T, heads]``,
    ``initial_state`` is ``[heads, 128, 128]`` in ``[value_dim, key_dim]``
    order. Returns the bf16 output ``[T, heads, 128]``, the fp32 final state in
    the same orientation, and the state after ``checkpoint_offset`` tokens
    (``None`` unless ``0 <= checkpoint_offset <= T``).
    """
    tokens = int(q.shape[0])
    heads = int(q.shape[1])
    scale_value = KDA_HEAD_DIM**-0.5 if scale is None else float(scale)
    qf = l2_normalize(q, eps) if qk_l2norm else q.float()
    kf = l2_normalize(k, eps) if qk_l2norm else k.float()
    qf = qf * scale_value
    vf = v.float()
    log_decay = kda_log_decay(raw_g, dt_bias, A_log, lower_bound)
    beta = kda_beta(raw_beta)
    state = initial_state.float().transpose(-1, -2).clone()  # [heads, K, V]
    output = torch.empty(
        (tokens, heads, KDA_HEAD_DIM), dtype=torch.bfloat16, device=q.device
    )
    checkpoint = None
    if checkpoint_offset == 0:
        checkpoint = state.transpose(-1, -2).contiguous()
    for t in range(tokens):
        state = state * torch.exp(log_decay[t])[:, :, None]
        k_t = kf[t]
        delta = vf[t] - torch.einsum("hk,hkv->hv", k_t, state)
        state = state + (beta[t][:, None] * k_t)[:, :, None] * delta[:, None, :]
        output[t] = torch.einsum("hk,hkv->hv", qf[t], state).to(torch.bfloat16)
        if t + 1 == checkpoint_offset:
            checkpoint = state.transpose(-1, -2).contiguous()
    return output, state.transpose(-1, -2).contiguous(), checkpoint


def prefill_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor | int,
    num_tokens: torch.Tensor | int,
    *,
    lower_bound: float = -5.0,
    scale: float | None = None,
    eps: float = 1e-6,
    qk_l2norm: bool = True,
    null_state_index: int | None = None,
    chunk: int = 16,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fp32 recurrence for a packed batch over a state pool.

    Sequence ``i`` consumes tokens ``cu_seqlens[i]:cu_seqlens[i+1]``, reads its
    initial state from ``initial_state_indices[i]`` (zero when null), writes its
    final state to ``final_state_indices[i]`` and, when ``checkpoint_offsets[i]``
    is a positive multiple of ``chunk``, the state after that many tokens to
    ``checkpoint_state_indices[i]``. Output rows at or beyond ``num_tokens`` are
    left untouched when ``output`` is supplied.
    """
    lower_bound_value = float(lower_bound)
    if not math.isfinite(lower_bound_value) or not -5.0 <= lower_bound_value < 0.0:
        raise ValueError("lower_bound must be in [-5, 0)")
    heads = int(q.shape[1])
    token_capacity = int(q.shape[0])
    seq_capacity = int(cu_seqlens.numel()) - 1
    live_seqs = _scalar(num_seqs)
    live_tokens = _scalar(num_tokens)
    spans = _validate_packed(
        cu_seqlens=cu_seqlens,
        initial_state_indices=initial_state_indices,
        final_state_indices=final_state_indices,
        checkpoint_state_indices=checkpoint_state_indices,
        checkpoint_offsets=checkpoint_offsets,
        num_seqs=live_seqs,
        num_tokens=live_tokens,
        token_capacity=token_capacity,
        seq_capacity=seq_capacity,
        state_slots=int(recurrent_state.shape[0]),
        chunk=chunk,
        null_state_index=null_state_index,
    )
    if output is None:
        output = torch.zeros(
            (token_capacity, heads, KDA_HEAD_DIM), dtype=torch.bfloat16, device=q.device
        )

    def is_null(slot: int) -> bool:
        return null_state_index is not None and slot == null_state_index

    for request, (start, end) in enumerate(spans):
        initial = int(initial_state_indices[request])
        final = int(final_state_indices[request])
        checkpoint_slot = int(checkpoint_state_indices[request])
        offset = int(checkpoint_offsets[request])
        if is_null(initial):
            state = torch.zeros(
                (heads, KDA_HEAD_DIM, KDA_HEAD_DIM), dtype=torch.float32, device=q.device
            )
        else:
            state = recurrent_state[initial].float()
        out, final_state, checkpoint = recurrent_kda(
            q[start:end],
            k[start:end],
            v[start:end],
            raw_g[start:end],
            raw_beta[start:end],
            A_log,
            dt_bias,
            lower_bound=lower_bound_value,
            initial_state=state,
            checkpoint_offset=offset if offset > 0 else -1,
            scale=scale,
            eps=eps,
            qk_l2norm=qk_l2norm,
        )
        output[start:end] = out
        if checkpoint is not None and not is_null(checkpoint_slot):
            recurrent_state[checkpoint_slot].copy_(checkpoint.to(recurrent_state.dtype))
        if not is_null(final):
            recurrent_state[final].copy_(final_state.to(recurrent_state.dtype))
    return output


@dataclass
class MirrorTrace:
    """Per-(sequence, tile) intermediates recorded by the chunk mirror."""

    k1: dict[tuple[int, int], dict[str, torch.Tensor]] = field(default_factory=dict)
    k2: dict[tuple[int, int], dict[str, torch.Tensor]] = field(default_factory=dict)
    checkpoints: dict[int, torch.Tensor] = field(default_factory=dict)


def _prepare_tile(
    q: torch.Tensor,
    k: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    rows: int,
    chunk: int,
    lower_bound: float,
    scale: float,
    eps: float,
    qk_l2norm: bool,
    policy: MirrorPolicy,
) -> dict[str, torch.Tensor]:
    """K1 mirror for one tile of all heads; inputs are ``[chunk, heads, ...]``."""
    heads = int(q.shape[1])
    mask = torch.zeros((chunk, 1, 1), dtype=torch.bool, device=q.device)
    mask[:rows] = True
    g2 = kda_log_decay(raw_g, dt_bias, A_log, lower_bound) * LOG2E
    g2 = torch.where(mask, g2, torch.zeros_like(g2))
    cum = torch.empty_like(g2)
    running = torch.zeros_like(g2[0])
    for t in range(chunk):
        running = running + g2[t]
        cum[t] = running
    lam = torch.exp2(cum)
    lam_inv = torch.exp2(-cum)
    lam_c = torch.exp2(cum[chunk - 1])
    lam_r = torch.exp2(cum[chunk - 1][None] - cum)
    qn = l2_normalize(q, eps) if qk_l2norm else q.float()
    kn = l2_normalize(k, eps) if qk_l2norm else k.float()
    qn = torch.where(mask, qn, torch.zeros_like(qn))
    kn = torch.where(mask, kn, torch.zeros_like(kn))
    scale_value = _bf16(torch.tensor(scale)).item() if policy.scale_dtype == "bf16" else scale
    if policy.operands == "fp32":
        q_tilde = qn * lam * scale_value
        k_tilde = kn * lam
        k_inv = kn * lam_inv
        k_r = kn * lam_r
    elif policy.single_rounding:
        q_tilde = _bf16(qn * lam * scale_value)
        k_tilde = _bf16(kn * lam)
        k_inv = _bf16(kn * lam_inv)
        k_r = _bf16(kn * lam_r)
    else:
        qb, kb = _bf16(qn), _bf16(kn)
        q_tilde = _bf16(_bf16(qb * _bf16(lam)) * scale_value)
        k_tilde = _bf16(kb * _bf16(lam))
        k_inv = _bf16(kb * _bf16(lam_inv))
        k_r = _bf16(k_inv * _bf16(lam_c)[None])
    beta = kda_beta(raw_beta)
    beta = torch.where(mask[:, :, 0], beta, torch.zeros_like(beta))
    # [heads, chunk, 128] operands for the per-head GEMMs.
    q_h, k_h, kinv_h = (x.transpose(0, 1) for x in (q_tilde, k_tilde, k_inv))
    beta_h = beta.transpose(0, 1)
    causal = torch.tril(torch.ones((chunk, chunk), dtype=torch.bool, device=q.device), -1)
    lower = beta_h[:, :, None] * (k_h @ kinv_h.transpose(-1, -2))
    lower = torch.where(causal, lower, torch.zeros_like(lower))
    inverse = _neumann_inverse(lower, chunk)
    inclusive = torch.tril(torch.ones((chunk, chunk), dtype=torch.bool, device=q.device))
    mqk = q_h @ kinv_h.transpose(-1, -2)
    mqk = torch.where(inclusive, mqk, torch.zeros_like(mqk))
    return {
        "g_cum": cum.transpose(0, 1),
        "lambda_c": lam_c,
        "q_tilde": q_h,
        "k_tilde": k_h,
        "k_inv": kinv_h,
        "k_r": k_r.transpose(0, 1),
        "beta": beta_h,
        "L": lower,
        "inv": inverse,
        "inv_op": _bf16(inverse) if policy.inv_operand == "bf16" else inverse,
        "mqk": _bf16(mqk) if policy.operands == "bf16" else mqk,
        "heads": torch.tensor(heads),
    }


def prefill_kda_chunk_mirror(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    checkpoint_state_indices: torch.Tensor,
    checkpoint_offsets: torch.Tensor,
    num_seqs: torch.Tensor | int,
    num_tokens: torch.Tensor | int,
    *,
    lower_bound: float = -5.0,
    scale: float | None = None,
    eps: float = 1e-6,
    qk_l2norm: bool = True,
    null_state_index: int | None = None,
    chunk: int = 16,
    policy: MirrorPolicy | None = None,
    trace: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, MirrorTrace]:
    """Run the kernel's chunked algorithm with its rounding points.

    Same contract as :func:`prefill_kda`. With ``trace=True`` the per-tile
    K1 and K2 intermediates are returned alongside the output.
    """
    lower_bound_value = float(lower_bound)
    if not math.isfinite(lower_bound_value) or not -5.0 <= lower_bound_value < 0.0:
        raise ValueError("lower_bound must be in [-5, 0)")
    if chunk & (chunk - 1) or chunk < 2:
        raise ValueError("chunk must be a power of two")
    policy = MirrorPolicy() if policy is None else policy
    heads = int(q.shape[1])
    scale_value = KDA_HEAD_DIM**-0.5 if scale is None else float(scale)
    token_capacity = int(q.shape[0])
    seq_capacity = int(cu_seqlens.numel()) - 1
    live_seqs = _scalar(num_seqs)
    live_tokens = _scalar(num_tokens)
    spans = _validate_packed(
        cu_seqlens=cu_seqlens,
        initial_state_indices=initial_state_indices,
        final_state_indices=final_state_indices,
        checkpoint_state_indices=checkpoint_state_indices,
        checkpoint_offsets=checkpoint_offsets,
        num_seqs=live_seqs,
        num_tokens=live_tokens,
        token_capacity=token_capacity,
        seq_capacity=seq_capacity,
        state_slots=int(recurrent_state.shape[0]),
        chunk=chunk,
        null_state_index=null_state_index,
    )
    if output is None:
        output = torch.zeros(
            (token_capacity, heads, KDA_HEAD_DIM), dtype=torch.bfloat16, device=q.device
        )
    record = MirrorTrace()

    def is_null(slot: int) -> bool:
        return null_state_index is not None and slot == null_state_index

    def padded(x: torch.Tensor, start: int, rows: int) -> torch.Tensor:
        tile = torch.zeros((chunk, *x.shape[1:]), dtype=x.dtype, device=x.device)
        tile[:rows] = x[start : start + rows]
        return tile

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for request, (start, end) in enumerate(spans):
            initial = int(initial_state_indices[request])
            final = int(final_state_indices[request])
            checkpoint_slot = int(checkpoint_state_indices[request])
            offset = int(checkpoint_offsets[request])
            if is_null(initial):
                state = torch.zeros(
                    (heads, KDA_HEAD_DIM, KDA_HEAD_DIM),
                    dtype=torch.float32,
                    device=q.device,
                )
            else:
                state = recurrent_state[initial].float()
            length = end - start
            for local in range((length + chunk - 1) // chunk):
                tile_start = start + local * chunk
                rows = min(chunk, end - tile_start)
                prep = _prepare_tile(
                    padded(q, tile_start, rows),
                    padded(k, tile_start, rows),
                    padded(raw_g, tile_start, rows),
                    padded(raw_beta, tile_start, rows),
                    A_log,
                    dt_bias,
                    rows=rows,
                    chunk=chunk,
                    lower_bound=lower_bound_value,
                    scale=scale_value,
                    eps=eps,
                    qk_l2norm=qk_l2norm,
                    policy=policy,
                )
                out_tile, state, step = _recur_tile(
                    state,
                    padded(v, tile_start, rows),
                    prep,
                    rows=rows,
                    chunk=chunk,
                    policy=policy,
                )
                output[tile_start : tile_start + rows] = out_tile[:rows]
                if trace:
                    record.k1[(request, local)] = prep
                    record.k2[(request, local)] = step
                if offset > 0 and (local + 1) * chunk == offset:
                    if trace:
                        record.checkpoints[request] = state.clone()
                    if not is_null(checkpoint_slot):
                        recurrent_state[checkpoint_slot].copy_(state.to(recurrent_state.dtype))
            if not is_null(final):
                recurrent_state[final].copy_(state.to(recurrent_state.dtype))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return (output, record) if trace else output


__all__ = [
    "LOG2E",
    "MirrorPolicy",
    "MirrorTrace",
    "prefill_kda",
    "prefill_kda_chunk_mirror",
    "recurrent_kda",
]
