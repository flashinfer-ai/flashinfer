"""Explicit PyTorch oracle for packed sequential GDN decode."""

from __future__ import annotations

import math

import torch

from .._shared.kda_math import kda_beta, kda_log_decay, l2_normalize


def _scalar(value: torch.Tensor | int) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("device count tensors must contain one element")
        return int(value.item())
    return int(value)


def decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor | int,
    num_tokens: torch.Tensor | int,
    *,
    key_heads: int,
    value_heads: int,
    gate_activation: str,
    eps: float = 1e-6,
    scale: float | None = None,
    qk_l2norm: bool = True,
    null_state_index: int | None = None,
) -> torch.Tensor:
    """Return gated output capacity while updating state checkpoints in place."""
    key_head_dim = 128
    value_head_dim = 128
    ratio = value_heads // key_heads
    scale_value = key_head_dim**-0.5 if scale is None else float(scale)
    live_seqs = _scalar(num_seqs)
    live_tokens = _scalar(num_tokens)
    max_tokens = int(mixed_qkv.shape[0])
    max_seqs, columns = map(int, state_indices.shape)
    if live_seqs < 0 or live_seqs > max_seqs:
        raise ValueError(f"num_seqs={live_seqs} exceeds capacity {max_seqs}")
    if live_tokens < 0 or live_tokens > max_tokens:
        raise ValueError(f"num_tokens={live_tokens} exceeds capacity {max_tokens}")
    if int(query_start_loc[0]) != 0:
        raise ValueError("query_start_loc[0] must be zero")
    if int(query_start_loc[live_seqs]) != live_tokens:
        raise ValueError("query_start_loc[num_seqs] must equal num_tokens")

    request_spans: list[tuple[int, int, int]] = []
    active_cells: set[int] = set()
    for request in range(live_seqs):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        accepted = int(num_accepted_tokens[request])
        if start < 0 or end < start or end > live_tokens:
            raise ValueError(f"invalid query interval [{start}, {end})")
        if end - start > columns:
            raise ValueError("request token count exceeds state-index columns")
        if accepted < 1 or accepted > columns:
            raise ValueError("num_accepted_tokens must select a state-index column")
        request_spans.append((start, end, accepted))
        if end == start:
            continue
        source_slot = int(state_indices[request, accepted - 1])
        if null_state_index is not None and source_slot == null_state_index:
            continue
        for column in set(range(end - start)) | {accepted - 1}:
            slot = int(state_indices[request, column])
            if null_state_index is not None and slot == null_state_index:
                continue
            if slot < 0 or slot >= recurrent_state.shape[0]:
                raise IndexError(f"state index {slot} is out of range")
            if slot in active_cells:
                raise ValueError(f"duplicate active state index {slot}")
            active_cells.add(slot)

    output = torch.zeros(
        (max_tokens, value_heads, value_head_dim),
        dtype=torch.bfloat16,
        device=mixed_qkv.device,
    )
    q_width = key_heads * key_head_dim
    q = mixed_qkv[:, :q_width].view(-1, key_heads, key_head_dim).float()
    k = mixed_qkv[:, q_width : 2 * q_width].view(-1, key_heads, key_head_dim).float()
    v = mixed_qkv[:, 2 * q_width :].view(-1, value_heads, value_head_dim).float()
    if qk_l2norm:
        q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    q *= scale_value

    for request, (start, end, accepted) in enumerate(request_spans):
        if end == start:
            continue
        source_slot = int(state_indices[request, accepted - 1])
        if null_state_index is not None and source_slot == null_state_index:
            continue
        for value_head in range(value_heads):
            key_head = value_head // ratio
            state = recurrent_state[source_slot, value_head].float()
            for relative_token, token in enumerate(range(start, end)):
                softplus_input = (
                    a[token, value_head].float() + dt_bias[value_head].float()
                )
                softplus = torch.where(
                    softplus_input <= 20.0,
                    torch.log1p(torch.exp(softplus_input)),
                    softplus_input,
                )
                decay = torch.exp(-torch.exp(A_log[value_head].float()) * softplus)
                beta = (
                    torch.sigmoid(b[token, value_head].float())
                    .to(torch.bfloat16)
                    .float()
                )
                state = state * decay
                delta = v[token, value_head] - state.mv(k[token, key_head])
                state = state + (delta * beta).unsqueeze(1) * k[
                    token, key_head
                ].unsqueeze(0)
                output[token, value_head] = state.mv(q[token, key_head]).to(
                    torch.bfloat16
                )
                destination_slot = int(state_indices[request, relative_token])
                if (
                    null_state_index is not None
                    and destination_slot == null_state_index
                ):
                    continue
                recurrent_state[destination_slot, value_head].copy_(
                    state.to(recurrent_state.dtype)
                )

    values = output[:live_tokens].float()
    values *= torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + float(eps))
    normalized = values.to(torch.bfloat16)
    if norm_weight.dtype == torch.bfloat16:
        weighted = (normalized * norm_weight).to(torch.bfloat16)
    elif norm_weight.dtype == torch.float32:
        weighted = normalized.float() * norm_weight
    else:
        raise TypeError("norm_weight must be bfloat16 or float32")
    gate = torch.sigmoid(z[:live_tokens].float())
    if gate_activation == "silu":
        gate *= z[:live_tokens].float()
    elif gate_activation != "sigmoid":
        raise ValueError(f"unsupported gate_activation={gate_activation!r}")
    output[:live_tokens].copy_((weighted * gate).to(torch.bfloat16))
    return output


def decode_kda(
    mixed_qkv: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    z: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    state_indices: torch.Tensor,
    num_seqs: torch.Tensor | int,
    num_tokens: torch.Tensor | int,
    *,
    heads: int,
    lower_bound: float = -5.0,
    eps: float = 1e-6,
    scale: float | None = None,
    qk_l2norm: bool = True,
    null_state_index: int | None = None,
) -> torch.Tensor:
    """Return lower-bounded KDA output while updating state checkpoints."""
    key_head_dim = value_head_dim = 128
    scale_value = key_head_dim**-0.5 if scale is None else float(scale)
    lower_bound_value = float(lower_bound)
    if not math.isfinite(lower_bound_value) or lower_bound_value >= 0:
        raise ValueError("lower_bound must be finite and negative")
    live_seqs = _scalar(num_seqs)
    live_tokens = _scalar(num_tokens)
    max_tokens = int(mixed_qkv.shape[0])
    max_seqs, columns = map(int, state_indices.shape)
    if live_seqs < 0 or live_seqs > max_seqs:
        raise ValueError(f"num_seqs={live_seqs} exceeds capacity {max_seqs}")
    if live_tokens < 0 or live_tokens > max_tokens:
        raise ValueError(f"num_tokens={live_tokens} exceeds capacity {max_tokens}")
    if int(query_start_loc[0]) != 0:
        raise ValueError("query_start_loc[0] must be zero")
    if int(query_start_loc[live_seqs]) != live_tokens:
        raise ValueError("query_start_loc[num_seqs] must equal num_tokens")

    request_spans: list[tuple[int, int, int]] = []
    active_cells: set[int] = set()
    for request in range(live_seqs):
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        accepted = int(num_accepted_tokens[request])
        if start < 0 or end < start or end > live_tokens:
            raise ValueError(f"invalid query interval [{start}, {end})")
        if end - start > columns:
            raise ValueError("request token count exceeds state-index columns")
        if accepted < 1 or accepted > columns:
            raise ValueError("num_accepted_tokens must select a state-index column")
        request_spans.append((start, end, accepted))
        if end == start:
            continue
        source_slot = int(state_indices[request, accepted - 1])
        if null_state_index is not None and source_slot == null_state_index:
            continue
        for column in set(range(end - start)) | {accepted - 1}:
            slot = int(state_indices[request, column])
            if null_state_index is not None and slot == null_state_index:
                continue
            if slot < 0 or slot >= recurrent_state.shape[0]:
                raise IndexError(f"state index {slot} is out of range")
            if slot in active_cells:
                raise ValueError(f"duplicate active state index {slot}")
            active_cells.add(slot)

    expected_mixed_width = heads * (2 * key_head_dim + value_head_dim)
    if tuple(mixed_qkv.shape) != (max_tokens, expected_mixed_width):
        raise ValueError(
            "mixed_qkv must have shape "
            f"{(max_tokens, expected_mixed_width)}, got {tuple(mixed_qkv.shape)}"
        )
    if tuple(raw_g.shape) != (max_tokens, heads, key_head_dim):
        raise ValueError("raw_g must have shape [max_tokens, heads, 128]")
    if tuple(raw_beta.shape) != (max_tokens, heads):
        raise ValueError("raw_beta must have shape [max_tokens, heads]")
    if tuple(dt_bias.shape) != (heads, key_head_dim):
        raise ValueError("dt_bias must have shape [heads, 128]")
    if tuple(A_log.shape) != (heads,):
        raise ValueError("A_log must have shape [heads]")

    output = torch.zeros(
        (max_tokens, heads, value_head_dim),
        dtype=torch.bfloat16,
        device=mixed_qkv.device,
    )
    q_width = heads * key_head_dim
    q = mixed_qkv[:, :q_width].view(-1, heads, key_head_dim).float()
    k = mixed_qkv[:, q_width : 2 * q_width].view(
        -1, heads, key_head_dim
    ).float()
    v = mixed_qkv[:, 2 * q_width :].view(-1, heads, value_head_dim).float()
    if qk_l2norm:
        q = l2_normalize(q)
        k = l2_normalize(k)
    q *= scale_value
    log_decay_all = kda_log_decay(raw_g, dt_bias, A_log, lower_bound_value)
    beta_all = kda_beta(raw_beta)

    for request, (start, end, accepted) in enumerate(request_spans):
        if end == start:
            continue
        source_slot = int(state_indices[request, accepted - 1])
        if null_state_index is not None and source_slot == null_state_index:
            continue
        for head in range(heads):
            state = recurrent_state[source_slot, head].float()
            for relative_token, token in enumerate(range(start, end)):
                log_decay = log_decay_all[token, head]
                state = state * torch.exp(log_decay).unsqueeze(0)
                beta = beta_all[token, head]
                delta = v[token, head] - state.mv(k[token, head])
                state = state + (delta * beta).unsqueeze(1) * k[
                    token, head
                ].unsqueeze(0)
                output[token, head] = state.mv(q[token, head]).to(torch.bfloat16)
                destination_slot = int(state_indices[request, relative_token])
                if (
                    null_state_index is not None
                    and destination_slot == null_state_index
                ):
                    continue
                recurrent_state[destination_slot, head].copy_(
                    state.to(recurrent_state.dtype)
                )

    values = output[:live_tokens].float()
    normalized = values * torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True) + float(eps)
    )
    if norm_weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("norm_weight must be bfloat16 or float32")
    weighted = normalized * norm_weight.float()
    output[:live_tokens].copy_(
        (weighted * torch.sigmoid(z[:live_tokens].float())).to(torch.bfloat16)
    )
    return output


__all__ = ["decode", "decode_kda"]
