"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Input validation for the SM90 push BF16 mega-MoE backend.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ......core.validation.common import MoEEpConfigError, validate_mega_forward_inputs

if TYPE_CHECKING:
    import torch

    from ......config import FleetParams


def validate_sm90_push_bf16_forward_inputs(
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
            "sm90_push_bf16 accepts native BF16 activations; "
            "MegaConfig.quantize_input must be True"
        )
    if topk_ids.ndim != 2 or topk_weights.ndim != 2:
        raise MoEEpConfigError(
            "sm90_push_bf16 routing tensors must be 2D [num_tokens, top_k]"
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
        raise MoEEpConfigError("sm90_push_bf16 inputs must be CUDA tensors")
    device = hidden_states.device
    for name, tensor, dtype in expected:
        if tensor.dtype != dtype:
            raise MoEEpConfigError(
                f"sm90_push_bf16 {name} must be {dtype}, got {tensor.dtype}"
            )
        if tensor.device != device:
            raise MoEEpConfigError(
                f"sm90_push_bf16 {name} must be on {device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"sm90_push_bf16 {name} must be contiguous")
    if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("", "0"):
        if torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "FLASHINFER_VALIDATE_INPUTS is not supported during CUDA graph capture"
            )
        if not bool(torch.isfinite(hidden_states.float()).all()):
            raise MoEEpConfigError("sm90_push_bf16 hidden_states must be finite")
        if not bool(torch.isfinite(topk_weights).all()):
            raise MoEEpConfigError("sm90_push_bf16 topk_weights must be finite")


__all__ = ["validate_sm90_push_bf16_forward_inputs"]
