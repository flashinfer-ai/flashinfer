"""Input validation for the SM90 push FP8 mega-MoE backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .....core.validation.common import MoEEpConfigError, validate_mega_forward_inputs

if TYPE_CHECKING:
    import torch

    from .....config import FleetParams


def validate_sm90_push_fp8_forward_inputs(
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
            "sm90_push_fp8 accepts native bf16 activations and performs the "
            "1x128 FP8 conversion in dispatch; MegaConfig.quantize_input must be True"
        )
    if topk_ids.ndim != 2:
        raise MoEEpConfigError(
            "sm90_push_fp8 topk_ids must be 2D [num_tokens, top_k], got "
            f"shape {tuple(topk_ids.shape)}"
        )
    if topk_weights.ndim != 2:
        raise MoEEpConfigError(
            "sm90_push_fp8 topk_weights must be 2D [num_tokens, top_k], got "
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
            f"sm90_push_fp8 hidden_states must be torch.bfloat16, got "
            f"{hidden_states.dtype}"
        )
    if topk_ids.dtype != torch.int32:
        raise MoEEpConfigError(
            f"sm90_push_fp8 topk_ids must be torch.int32, got {topk_ids.dtype}"
        )
    if topk_weights.dtype != torch.float32:
        raise MoEEpConfigError(
            "sm90_push_fp8 topk_weights must be torch.float32, got "
            f"{topk_weights.dtype}"
        )
    if not hidden_states.is_cuda:
        raise MoEEpConfigError("sm90_push_fp8 inputs must be CUDA tensors")
    device = hidden_states.device
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("topk_ids", topk_ids),
        ("topk_weights", topk_weights),
    ):
        if tensor.device != device:
            raise MoEEpConfigError(
                f"sm90_push_fp8 {name} must be on {device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"sm90_push_fp8 {name} must be contiguous")
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("", "0"):
        if not bool(torch.isfinite(hidden_states.float()).all()):
            raise MoEEpConfigError("sm90_push_fp8 hidden_states must be finite")
        if not bool(torch.isfinite(topk_weights).all()):
            raise MoEEpConfigError("sm90_push_fp8 topk_weights must be finite")


__all__ = ["validate_sm90_push_fp8_forward_inputs"]
