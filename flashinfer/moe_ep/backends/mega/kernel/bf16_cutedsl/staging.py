"""Stage BF16 activations and routing for BF16 MegaMoE."""

from __future__ import annotations

import torch

from .....core.validation.common import MoEEpConfigError, validate_mega_forward_inputs


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    num_tokens = hidden_states.shape[0]
    x[:num_tokens].copy_(hidden_states)
    topk_idx_out[:num_tokens].copy_(topk_ids)
    topk_weights_out[:num_tokens].copy_(topk_weights)
    if num_tokens < x.shape[0]:
        topk_idx_out[num_tokens:].fill_(-1)


def validate_bf16_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    scales: torch.Tensor | None = None,
) -> None:
    if not quantize_input:
        raise MoEEpConfigError(
            "BF16 MegaMoE has no pre-quantized activation path; "
            "set MegaConfig.quantize_input=True."
        )
    if scales is not None:
        raise MoEEpConfigError("BF16 MegaMoE does not accept MoEEpTensors.scales.")
    validate_mega_forward_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        fleet_params,
        top_k=top_k,
        quantize_input=True,
    )
