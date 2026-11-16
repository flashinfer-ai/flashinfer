"""PyTorch correctness oracles for learned low-rank HyperConnection math."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _logical_state(state: torch.Tensor, streams: int) -> torch.Tensor:
    if state.ndim != 2:
        raise ValueError(
            "state must have shape [tokens, streams * hidden_size], "
            f"got {tuple(state.shape)}"
        )
    width = int(state.shape[1])
    if streams <= 0 or width % streams != 0:
        raise ValueError(f"state width {width} must be divisible by streams={streams}")
    return state.unflatten(-1, (streams, width // streams))


def grouped_rmsnorm(
    state: torch.Tensor,
    weight: torch.Tensor,
    *,
    streams: int,
    eps: float,
    zero_centered: bool = True,
) -> torch.Tensor:
    """RMSNorm each stream with optional Gemma zero-centered affine weights."""
    logical = _logical_state(state, streams)
    if tuple(weight.shape) != (int(state.shape[1]),):
        raise ValueError(
            f"weight must have shape {(int(state.shape[1]),)}, "
            f"got {tuple(weight.shape)}"
        )
    normalized = logical.float() * torch.rsqrt(
        logical.float().square().mean(dim=-1, keepdim=True) + float(eps)
    )
    offset = 1.0 if zero_centered else 0.0
    return (normalized.flatten(-2) * (offset + weight.float())).to(state.dtype)


def scaled_silu(projected_down: torch.Tensor, *, streams: int) -> torch.Tensor:
    """Apply the bottleneck activation ``SiLU(projected_down / streams)``."""
    return F.silu(projected_down / int(streams))


def gate_mean(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    streams: int,
) -> torch.Tensor:
    """Sigmoid-gate normalized streams and return their hidden-wise mean."""
    logical = _logical_state(normalized, streams)
    if gate_logits.shape != normalized.shape:
        raise ValueError(
            "gate_logits must match normalized shape, got "
            f"{tuple(gate_logits.shape)} and {tuple(normalized.shape)}"
        )
    gate = torch.sigmoid(gate_logits).view_as(logical)
    return (gate * logical).mean(dim=1).to(normalized.dtype)


def combine(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    *,
    streams: int,
) -> torch.Tensor:
    """Inject one block output into every stream and round to state dtype."""
    logical = _logical_state(state, streams)
    tokens, _, hidden_size = map(int, logical.shape)
    if tuple(block_output.shape) != (tokens, hidden_size):
        raise ValueError(
            f"block_output must have shape {(tokens, hidden_size)}, "
            f"got {tuple(block_output.shape)}"
        )
    if tuple(injection_logits.shape) != (tokens, streams):
        raise ValueError(
            f"injection_logits must have shape {(tokens, streams)}, "
            f"got {tuple(injection_logits.shape)}"
        )
    scale = 2.0 * torch.sigmoid(injection_logits.float() / streams)
    combined = logical.float() + block_output.float().unsqueeze(1) * scale.unsqueeze(-1)
    return combined.flatten(-2).to(state.dtype)


def combine_norm(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    *,
    streams: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine, round, then normalize using the rounded multi-stream state."""
    combined = combine(
        state,
        block_output,
        injection_logits,
        streams=streams,
    ).to(state.dtype)
    normalized = grouped_rmsnorm(
        combined,
        next_norm_weight,
        streams=streams,
        eps=eps,
    )
    return combined, normalized


__all__ = [
    "grouped_rmsnorm",
    "scaled_silu",
    "gate_mean",
    "combine",
    "combine_norm",
]
