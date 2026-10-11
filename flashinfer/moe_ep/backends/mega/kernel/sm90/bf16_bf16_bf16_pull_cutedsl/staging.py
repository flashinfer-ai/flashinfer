"""Stage BF16 activations + routing into SM90 pull-style mega-MoE symmetric buffers."""

from __future__ import annotations

import torch

from ..fp8_fp8_bf16_pull_cutedsl.staging import _note_staged_tokens


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    """Copy BF16 tokens and routing; pad rows past the batch get ``-1`` routes.

    The per-tensor E8M0 SF wire (``x_sf``) is dispatched but never read by
    the BF16 GEMMs, so it is not staged.
    """
    num_tokens = hidden_states.shape[0]
    x[:num_tokens].copy_(hidden_states)
    topk_idx_out[:num_tokens].copy_(topk_ids)
    topk_weights_out[:num_tokens].copy_(topk_weights)
    if num_tokens < x.shape[0]:
        topk_idx_out[num_tokens:].fill_(-1)
    _note_staged_tokens(topk_idx_out, num_tokens)
