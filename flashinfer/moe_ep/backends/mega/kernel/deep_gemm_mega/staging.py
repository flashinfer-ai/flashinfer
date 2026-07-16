"""Stage bf16 activations + routing into mega-MoE symmetric buffers."""

from __future__ import annotations

import torch


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError("hidden_size must be a multiple of 128.")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")
    from deep_gemm.utils import per_token_cast_to_fp8

    x_q, x_sf_q = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    x_fp8.copy_(x_q)
    x_sf.copy_(x_sf_q)
    topk_idx_out.copy_(topk_ids.to(torch.int64))
    topk_weights_out.copy_(topk_weights)


__all__ = ["stage_mega_moe_inputs"]
