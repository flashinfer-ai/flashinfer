"""Explicit PyTorch oracle for MTP feedback fusion."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _require_bf16(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply zero-centered Gemma RMSNorm over the final dimension."""
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if tuple(weight.shape) != (int(x.shape[-1]),):
        raise ValueError(
            f"weight must have shape ({int(x.shape[-1])},), got {tuple(weight.shape)}"
        )
    if weight.device != x.device:
        raise ValueError(f"weight must be on {x.device}, got {weight.device}")
    _require_bf16("x", x)
    _require_bf16("weight", weight)
    values = x.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + eps_value)
    return (normalized * (1.0 + weight.float())).to(x.dtype)


def feedback(
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor,
    hidden_fc_weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return the BF16 ``[T,S,H]`` input to the MTP draft layer."""
    if token_embedding.ndim != 2:
        raise ValueError(
            "token_embedding must have shape [tokens, hidden], got "
            f"{tuple(token_embedding.shape)}"
        )
    if multi_state.ndim != 3:
        raise ValueError(
            "multi_state must have shape [tokens, streams, hidden], got "
            f"{tuple(multi_state.shape)}"
        )
    tokens, hidden = map(int, token_embedding.shape)
    state_tokens, streams, state_hidden = map(int, multi_state.shape)
    if (state_tokens, state_hidden) != (tokens, hidden):
        raise ValueError(
            "multi_state token/hidden dimensions must match token_embedding; "
            f"got {tuple(multi_state.shape)} and {tuple(token_embedding.shape)}"
        )
    expected_shapes = {
        "token_norm_weight": (hidden,),
        "state_norm_weight": (streams * hidden,),
        "embedding_fc_weight": (hidden, hidden),
        "hidden_fc_weight": (hidden, hidden),
    }
    tensors = {
        "token_embedding": token_embedding,
        "multi_state": multi_state,
        "token_norm_weight": token_norm_weight,
        "state_norm_weight": state_norm_weight,
        "embedding_fc_weight": embedding_fc_weight,
        "hidden_fc_weight": hidden_fc_weight,
    }
    for name, shape in expected_shapes.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
    for name, tensor in tensors.items():
        if tensor.device != token_embedding.device:
            raise ValueError(
                f"{name} must be on {token_embedding.device}, got {tensor.device}"
            )
        _require_bf16(name, tensor)

    token_normalized = gemma_rmsnorm(
        token_embedding,
        token_norm_weight,
        eps=eps,
    )
    token_path = F.linear(token_normalized, embedding_fc_weight).to(torch.bfloat16)
    state_normalized = gemma_rmsnorm(
        multi_state.flatten(-2),
        state_norm_weight,
        eps=eps,
    ).view(tokens, streams, hidden)
    state_path = F.linear(state_normalized, hidden_fc_weight).to(torch.bfloat16)
    return (state_path + token_path.unsqueeze(1)).to(torch.bfloat16)


__all__ = ["gemma_rmsnorm", "feedback"]
