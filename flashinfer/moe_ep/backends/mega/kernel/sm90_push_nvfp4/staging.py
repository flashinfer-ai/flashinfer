"""Input validation for the SM90 push NVFP4 mega-MoE backend."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .....core.validation.common import MoEEpConfigError, validate_mega_forward_inputs

if TYPE_CHECKING:
    import torch

    from .....config import FleetParams


def validate_sm90_push_nvfp4_forward_inputs(
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
            "sm90_push_nvfp4 accepts native bf16 activations and performs its "
            "wire conversion internally; MegaConfig.quantize_input must be True"
        )
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise MoEEpConfigError(
            "sm90_push_nvfp4 routing tensors must be 2D [num_tokens, top_k]"
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
    expected = (
        ("hidden_states", hidden_states, torch.bfloat16),
        ("topk_ids", topk_ids, torch.int32),
        ("topk_weights", topk_weights, torch.float32),
    )
    if not hidden_states.is_cuda:
        raise MoEEpConfigError("sm90_push_nvfp4 inputs must be CUDA tensors")
    device = hidden_states.device
    for name, tensor, dtype in expected:
        if tensor.dtype != dtype:
            raise MoEEpConfigError(
                f"sm90_push_nvfp4 {name} must be {dtype}, got {tensor.dtype}"
            )
        if tensor.device != device:
            raise MoEEpConfigError(
                f"sm90_push_nvfp4 {name} must be on {device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"sm90_push_nvfp4 {name} must be contiguous")
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("", "0"):
        if torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "FLASHINFER_VALIDATE_INPUTS is not supported during CUDA graph capture"
            )
        if not bool(torch.isfinite(hidden_states.float()).all()):
            raise MoEEpConfigError("sm90_push_nvfp4 hidden_states must be finite")
        if not bool(torch.isfinite(topk_weights).all()):
            raise MoEEpConfigError("sm90_push_nvfp4 topk_weights must be finite")


__all__ = ["validate_sm90_push_nvfp4_forward_inputs"]
