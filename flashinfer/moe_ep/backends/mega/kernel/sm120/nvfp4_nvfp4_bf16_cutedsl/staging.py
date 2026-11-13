"""Forward-input validation for SM120 NVFP4 x NVFP4 MegaMoE."""

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

    from ......kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        ACTIVATION_DTYPE,
        SCALE_DTYPE,
        ceil_div,
    )

    tokens = hidden_states.shape[0]
    hidden = fleet_params.token_hidden_size
    expected_data = (tokens, hidden // 2)
    if hidden_states.shape != expected_data or hidden_states.dtype != ACTIVATION_DTYPE:
        raise MoEEpConfigError(
            f"pre-quantized activation must be {ACTIVATION_DTYPE} with shape "
            f"{expected_data}"
        )
    scale_columns = ceil_div(hidden, 16)
    if (
        scales is None
        or scales.dtype != SCALE_DTYPE
        or scales.ndim != 2
        or scales.shape[0] != tokens
        or scales.shape[1] < scale_columns
    ):
        raise MoEEpConfigError(
            "pre-quantized activation requires matching E4M3 block-16 scales"
        )
    if topk_ids.shape != (tokens, top_k) or topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("routing tensors do not match token count/top-k")
    if tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError("token count exceeds max_tokens_per_rank")


__all__ = ["validate_forward_inputs"]
