"""KDA (lower-bounded gated delta rule) scalar recipes shared by references.

These helpers define the fp32 math that both the decode and prefill oracles
use for the per-key-coordinate decay gate, the update gate, and the query/key
L2 normalization. They operate on whole tensors so the same expression serves
a single token (decode) and a packed sequence (prefill).
"""

from __future__ import annotations

import torch

KDA_HEAD_DIM = 128


def kda_log_decay(
    raw_g: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    """Return the natural-log decay per key coordinate, in ``[lower_bound, 0)``.

    ``raw_g`` is ``[..., heads, 128]``, ``dt_bias`` is ``[heads, 128]`` and
    ``A_log`` is ``[heads]``; the result is fp32 with the shape of ``raw_g``.
    """
    rate = torch.exp(A_log.float())
    lead = raw_g.dim() - 2
    rate = rate.reshape((1,) * lead + (-1, 1))
    bias = dt_bias.float().reshape((1,) * lead + tuple(dt_bias.shape))
    return float(lower_bound) * torch.sigmoid(rate * (raw_g.float() + bias))


def kda_beta(raw_beta: torch.Tensor) -> torch.Tensor:
    """Return the fp32 update gate ``sigmoid(raw_beta)``."""
    return torch.sigmoid(raw_beta.float())


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return ``x / sqrt(sum(x^2) + eps)`` over the last axis in fp32."""
    x = x.float()
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + float(eps))


__all__ = ["KDA_HEAD_DIM", "kda_beta", "kda_log_decay", "l2_normalize"]
