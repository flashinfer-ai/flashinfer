"""Input validation for the exact BF16 rank-major MegaMoE backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ......core.validation.common import MoEEpConfigError, validate_mega_forward_inputs

if TYPE_CHECKING:
    import torch

    from ......config import FleetParams


def validate_rank_major_forward_inputs(
    hidden_states: "torch.Tensor",
    topk_ids: "torch.Tensor",
    topk_weights: "torch.Tensor",
    fleet_params: "FleetParams",
    *,
    top_k: int,
    quantize_input: bool,
    scales: "torch.Tensor | None",
) -> None:
    """Validate the fixed per-rank input ABI without inspecting device values."""
    import torch

    if not quantize_input:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda requires native BF16 "
            "activations and MegaConfig.quantize_input=True"
        )
    if scales is not None:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda does not accept "
            "MoEEpTensors.scales"
        )
    validate_mega_forward_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        fleet_params,
        top_k=top_k,
        quantize_input=True,
    )
    if tuple(hidden_states.shape) != (128, 7168):
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda hidden_states must have "
            f"shape (128, 7168), got {tuple(hidden_states.shape)}"
        )
    if tuple(topk_ids.shape) != (128, 8):
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda topk_ids must have shape "
            f"(128, 8), got {tuple(topk_ids.shape)}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda hidden_states must be "
            f"torch.bfloat16, got {hidden_states.dtype}"
        )
    if topk_ids.dtype != torch.int64:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda topk_ids must be "
            f"torch.int64, got {topk_ids.dtype}"
        )
    if topk_weights.dtype != torch.float32:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda topk_weights must be "
            f"torch.float32, got {topk_weights.dtype}"
        )
    if not hidden_states.is_cuda:
        raise MoEEpConfigError(
            "sm100_bf16_bf16_bf16_rank_major_cuda inputs must be CUDA tensors"
        )
    device = hidden_states.device
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("topk_ids", topk_ids),
        ("topk_weights", topk_weights),
    ):
        if tensor.device != device:
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must be on "
                f"{device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise MoEEpConfigError(
                f"sm100_bf16_bf16_bf16_rank_major_cuda {name} must be contiguous"
            )


__all__ = ["validate_rank_major_forward_inputs"]
