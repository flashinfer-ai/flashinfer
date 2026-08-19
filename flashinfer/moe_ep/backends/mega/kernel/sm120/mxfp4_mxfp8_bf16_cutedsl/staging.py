"""Forward-input validation and staging for SM120 W4A8 MegaMoE."""

from __future__ import annotations

import torch

from ......core.validation.common import MoEEpConfigError, validate_mega_forward_inputs


def validate_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    scales: torch.Tensor | None,
) -> None:
    if quantize_input:
        validate_mega_forward_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            fleet_params,
            top_k=top_k,
            quantize_input=True,
        )
        return

    from ......kernel_src.sm120.split_cutedsl_megakernel import (
        ACTIVATION_DTYPE,
        SCALE_DTYPE,
    )

    tokens = hidden_states.shape[0]
    hidden = fleet_params.token_hidden_size
    if hidden_states.shape != (tokens, hidden) or hidden_states.dtype != ACTIVATION_DTYPE:
        raise MoEEpConfigError(
            f"pre-quantized activation must be {ACTIVATION_DTYPE} with shape "
            f"({tokens}, {hidden})"
        )
    if scales is None or scales.dtype != SCALE_DTYPE or scales.shape[0] != tokens:
        raise MoEEpConfigError("pre-quantized activation requires matching E8M0 scales")
    if topk_ids.shape != (tokens, top_k) or topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("routing tensors do not match token count/top-k")
    if tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError("token count exceeds max_tokens_per_rank")


__all__ = ["validate_forward_inputs"]
