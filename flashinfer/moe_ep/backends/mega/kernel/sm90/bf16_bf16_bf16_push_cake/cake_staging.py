"""Input validation for the SM90 native BF16 push mega-MoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ......core.validation.common import MoEEpConfigError, validate_mega_forward_inputs

if TYPE_CHECKING:
    import torch

    from ......config import FleetParams


def validate_sm90_cake_bf16_forward_inputs(
    hidden_states: "torch.Tensor",
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    fleet_params: "FleetParams",
    *,
    top_k: int,
    quantize_input: bool,
    scales: "torch.Tensor | None",
) -> None:
    import torch

    if not quantize_input:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake consumes native bf16 activations and has no "
            "pre-quantized activation path; MegaConfig.quantize_input must be True"
        )
    if scales is not None:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake does not take activation scales (native BF16)"
        )
    if topk_ids.ndim != 2:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake topk_ids must be 2D [num_tokens, top_k], got "
            f"shape {tuple(topk_ids.shape)}"
        )
    if topk_weights.ndim != 2:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake topk_weights must be 2D [num_tokens, top_k], got "
            f"shape {tuple(topk_weights.shape)}"
        )
    validate_mega_forward_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        fleet_params,
        top_k=top_k,
        quantize_input=quantize_input,
        scales=scales,
    )
    if hidden_states.dtype != torch.bfloat16:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake hidden_states must be torch.bfloat16, got "
            f"{hidden_states.dtype}"
        )
    if topk_ids.dtype != torch.int32:
        raise MoEEpConfigError(
            f"sm90_bf16_bf16_bf16_push_cake topk_ids must be torch.int32, got {topk_ids.dtype}"
        )
    if topk_weights.dtype != torch.float32:
        raise MoEEpConfigError(
            "sm90_bf16_bf16_bf16_push_cake topk_weights must be torch.float32, got "
            f"{topk_weights.dtype}"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("topk_ids", topk_ids),
        ("topk_weights", topk_weights),
    ):
        if not tensor.is_cuda:
            raise MoEEpConfigError(
                f"sm90_bf16_bf16_bf16_push_cake {name} must be a CUDA tensor"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(
                f"sm90_bf16_bf16_bf16_push_cake {name} must be contiguous"
            )
