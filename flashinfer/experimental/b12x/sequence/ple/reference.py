"""Exact, intentionally unoptimized PyTorch oracles for PLE."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gemma_rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    group_size: int | None = None,
) -> torch.Tensor:
    """Gemma RMSNorm with a zero-centered affine weight."""
    input_dtype = x.dtype
    xf = x.float()
    if group_size is None:
        variance = xf.square().mean(dim=-1, keepdim=True)
        normalized = xf * torch.rsqrt(variance + eps)
    else:
        if xf.shape[-1] % int(group_size) != 0:
            raise ValueError(
                f"last dimension {xf.shape[-1]} is not divisible by {group_size}"
            )
        grouped = xf.unflatten(-1, (xf.shape[-1] // int(group_size), int(group_size)))
        variance = grouped.square().mean(dim=-1, keepdim=True)
        normalized = (grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return (normalized * (1.0 + weight.float())).to(input_dtype)


def ple_projected_sequence_reference(
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    eps: float,
    dilation: int,
    prior_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the PLE contribution and newest normalized-convolution state.

    The projections and embedding lookup intentionally remain outside this
    oracle. ``prior_state`` is ``[S*H, dilation*(K-1)]`` ordered oldest first.
    """
    if residual.ndim != 3:
        raise ValueError("residual must have shape [tokens, streams, hidden]")
    tokens, streams, hidden = residual.shape
    expected_key = (tokens, streams, hidden)
    if tuple(key.shape) != expected_key:
        raise ValueError(f"key must have shape {expected_key}, got {tuple(key.shape)}")
    if tuple(value.shape) != (tokens, hidden):
        raise ValueError(
            f"value must have shape {(tokens, hidden)}, got {tuple(value.shape)}"
        )
    channels = streams * hidden
    u, u_n = ple_projected_u_reference(
        residual,
        key,
        value,
        k_norm_weight=k_norm_weight,
        q_norm_weight=q_norm_weight,
        u_norm_weight=u_norm_weight,
        eps=eps,
    )
    channels = streams * hidden
    if conv_weight.ndim == 2:
        if conv_weight.shape[0] != channels:
            raise ValueError(f"conv_weight must have {channels} channels")
        conv = conv_weight.unsqueeze(1)
    elif conv_weight.ndim == 3 and conv_weight.shape[1] == 1:
        if conv_weight.shape[0] != channels:
            raise ValueError(f"conv_weight must have {channels} channels")
        conv = conv_weight
    else:
        raise ValueError("conv_weight must have shape [S*H,K] or [S*H,1,K]")
    kernel_size = int(conv.shape[-1])
    dilation = int(dilation)
    if kernel_size < 2 or dilation <= 0:
        raise ValueError("PLE requires kernel_size >= 2 and positive dilation")
    state_length = dilation * (kernel_size - 1)

    if prior_state is None:
        prior_state = torch.zeros(
            (channels, state_length),
            dtype=u_n.dtype,
            device=u_n.device,
        )
    if tuple(prior_state.shape) != (channels, state_length):
        raise ValueError(
            f"prior_state must have shape {(channels, state_length)}, "
            f"got {tuple(prior_state.shape)}"
        )
    if prior_state.dtype != u_n.dtype or prior_state.device != u_n.device:
        raise ValueError("prior_state must match the normalized input dtype and device")

    conv_input = torch.cat((prior_state.transpose(0, 1), u_n), dim=0)
    convolved = F.conv1d(
        conv_input.transpose(0, 1).unsqueeze(0),
        conv,
        bias=None,
        dilation=dilation,
        groups=channels,
    )
    convolved = convolved.squeeze(0).transpose(0, 1).view_as(u)
    contribution = u + F.silu(convolved).to(u.dtype)
    newest_state = conv_input[-state_length:].transpose(0, 1).contiguous()
    return contribution, newest_state


def ple_projected_u_reference(
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    u_norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw gated ``U`` and its grouped-normalized convolution input."""
    if residual.ndim != 3:
        raise ValueError("residual must have shape [tokens, streams, hidden]")
    tokens, streams, hidden = residual.shape
    channels = streams * hidden
    if tuple(key.shape) != (tokens, streams, hidden):
        raise ValueError(
            f"key must have shape {(tokens, streams, hidden)}, got {tuple(key.shape)}"
        )
    if tuple(value.shape) != (tokens, hidden):
        raise ValueError(
            f"value must have shape {(tokens, hidden)}, got {tuple(value.shape)}"
        )
    normalized_weights: list[torch.Tensor] = []
    for name, tensor in (
        ("k_norm_weight", k_norm_weight),
        ("q_norm_weight", q_norm_weight),
        ("u_norm_weight", u_norm_weight),
    ):
        if tuple(tensor.shape) not in ((streams, hidden), (channels,)):
            raise ValueError(
                f"{name} must have shape {(channels,)} or {(streams, hidden)}, "
                f"got {tuple(tensor.shape)}"
            )
        normalized_weights.append(tensor.flatten())

    key_n = gemma_rmsnorm_reference(
        key.flatten(-2),
        normalized_weights[0],
        eps=eps,
        group_size=hidden,
    ).view_as(key)
    query_n = gemma_rmsnorm_reference(
        residual.flatten(-2),
        normalized_weights[1],
        eps=eps,
        group_size=hidden,
    ).view_as(residual)
    similarity = (key_n.float() * query_n.float()).sum(-1) / math.sqrt(hidden)
    warped = torch.sign(similarity) * torch.sqrt(similarity.abs().clamp_min(1e-6))
    gate = torch.sigmoid(warped).to(value.dtype)
    u = gate.unsqueeze(-1) * value.unsqueeze(1)
    u_n = gemma_rmsnorm_reference(
        u.flatten(-2),
        normalized_weights[2],
        eps=eps,
        group_size=hidden,
    )
    return u, u_n


def ple_projected_packed_reference(
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_start_loc: torch.Tensor,
    *,
    k_norm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    eps: float,
    dilation: int,
    prior_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the sequence oracle independently to packed requests."""
    if query_start_loc.ndim != 1:
        raise ValueError("query_start_loc must be rank 1")
    starts = [int(value) for value in query_start_loc.detach().cpu().tolist()]
    if not starts or starts[0] != 0 or starts[-1] != int(residual.shape[0]):
        raise ValueError("query_start_loc must span exactly all packed tokens")
    if any(left > right for left, right in zip(starts, starts[1:], strict=False)):
        raise ValueError("query_start_loc must be nondecreasing")
    num_seqs = len(starts) - 1
    channels = int(residual.shape[1] * residual.shape[2])
    kernel_size = int(conv_weight.shape[-1])
    state_length = int(dilation) * (kernel_size - 1)
    if prior_states is None:
        prior_states = torch.zeros(
            (num_seqs, channels, state_length),
            dtype=residual.dtype,
            device=residual.device,
        )
    if tuple(prior_states.shape) != (num_seqs, channels, state_length):
        raise ValueError(
            f"prior_states must have shape {(num_seqs, channels, state_length)}"
        )
    output = torch.empty_like(residual)
    newest_states = torch.empty_like(prior_states)
    for request, (start, end) in enumerate(zip(starts, starts[1:], strict=False)):
        contribution, newest = ple_projected_sequence_reference(
            residual[start:end],
            key[start:end],
            value[start:end],
            k_norm_weight=k_norm_weight,
            q_norm_weight=q_norm_weight,
            u_norm_weight=u_norm_weight,
            conv_weight=conv_weight,
            eps=eps,
            dilation=dilation,
            prior_state=prior_states[request],
        )
        output[start:end].copy_(contribution)
        newest_states[request].copy_(newest)
    return output, newest_states


__all__ = [
    "gemma_rmsnorm_reference",
    "ple_projected_u_reference",
    "ple_projected_sequence_reference",
    "ple_projected_packed_reference",
]
