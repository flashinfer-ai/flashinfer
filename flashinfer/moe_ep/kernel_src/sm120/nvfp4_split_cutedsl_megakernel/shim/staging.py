"""BF16/NVFP4 input staging for the SM120 NVFP4 split backend."""

from __future__ import annotations

import torch

from .weights import FP4_DTYPE, SCALE_BLOCK, SCALE_DTYPE, ceil_div, round_up

ACTIVATION_DTYPE = FP4_DTYPE


def _stage_bf16_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    staged_topk_ids: torch.Tensor,
    staged_topk_weights: torch.Tensor,
    *,
    norm_const: float,
) -> None:
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        fused_quant_stage,
        fused_quant_stage_supported,
        nvfp4_quantize_per_block_16,
    )

    tokens = hidden_states.shape[0]
    if fused_quant_stage_supported(hidden_states, quant_type="nvfp4"):
        fused_quant_stage(
            hidden_states,
            topk_ids,
            topk_weights,
            x,
            x_scale,
            staged_topk_ids,
            staged_topk_weights,
            quant_type="nvfp4",
            norm_const=norm_const,
        )
        return

    quantized, raw_scale = nvfp4_quantize_per_block_16(
        hidden_states.to(torch.float32), norm_const
    )
    x[:tokens].copy_(quantized)
    x_scale[:tokens].zero_()
    x_scale[:tokens, : raw_scale.shape[1]].copy_(raw_scale)
    staged_topk_ids[:tokens].copy_(topk_ids)
    staged_topk_weights[:tokens].copy_(topk_weights)
    if tokens < x.shape[0]:
        staged_topk_ids[tokens:].fill_(-1)


def stage_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_scale: torch.Tensor,
    staged_topk_ids: torch.Tensor,
    staged_topk_weights: torch.Tensor,
    *,
    quantize_input: bool,
    scales: torch.Tensor | None,
    norm_const: float = 1.0,
) -> None:
    tokens, hidden_or_packed = hidden_states.shape
    if tokens > x.shape[0]:
        raise ValueError(f"{tokens} tokens exceed workspace capacity {x.shape[0]}")
    logical_hidden = x.shape[1] * 2
    if quantize_input:
        if hidden_or_packed != logical_hidden:
            raise ValueError(
                f"BF16 input width must be {logical_hidden}, got {hidden_or_packed}"
            )
        _stage_bf16_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            x,
            x_scale,
            staged_topk_ids,
            staged_topk_weights,
            norm_const=norm_const,
        )
        return

    if hidden_states.dtype != ACTIVATION_DTYPE or scales is None:
        raise ValueError(
            f"pre-quantized input requires {ACTIVATION_DTYPE} data and E4M3 scales"
        )
    valid_scale_columns = ceil_div(logical_hidden, SCALE_BLOCK)
    padded_scale_columns = round_up(valid_scale_columns, 4)
    raw_scale = scales
    if raw_scale.dtype != SCALE_DTYPE:
        if raw_scale.dtype != torch.uint8:
            raise ValueError(f"activation scales must be {SCALE_DTYPE} or uint8")
        raw_scale = raw_scale.view(SCALE_DTYPE)
    if raw_scale.shape[1] < valid_scale_columns:
        raise ValueError(
            f"activation scale width {raw_scale.shape[1]} is smaller than "
            f"{valid_scale_columns}"
        )
    if x_scale.shape[1] < padded_scale_columns:
        raise ValueError("workspace activation-scale row is not padded to four bytes")

    x[:tokens].view(torch.uint8).copy_(hidden_states.view(torch.uint8))
    x_scale[:tokens].zero_()
    x_scale[:tokens, :valid_scale_columns].view(torch.uint8).copy_(
        raw_scale[:, :valid_scale_columns].view(torch.uint8)
    )
    staged_topk_ids[:tokens].copy_(topk_ids)
    staged_topk_weights[:tokens].copy_(topk_weights)
    if tokens < x.shape[0]:
        staged_topk_ids[tokens:].fill_(-1)


__all__ = ["ACTIVATION_DTYPE", "stage_inputs"]
