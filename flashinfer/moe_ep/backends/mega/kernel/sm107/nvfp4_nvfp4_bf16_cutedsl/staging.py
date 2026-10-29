"""Stage bf16 activations + routing into SM107 nvfp4 mega symmetric buffers."""

from __future__ import annotations

import os

import torch


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_nvfp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    input_norm_const: float = 1.0,
) -> int:
    """bf16 ``hidden_states`` -> packed NVFP4 activation + FP8-E4M3 block scales.

    The packed data times its block scales approximates
    ``input_norm_const * hidden_states``; fc1_alpha must compensate for this
    normalization and any weight normalization. Returns the staged token count.
    """
    from ......kernel_src.sm107.next_cutedsl_megamoe import (
        Nvfp4BlockSize,
        ceil_div,
        quantize_nvfp4_block16,
    )
    from ..validation import validate_input_norm_const

    validate_input_norm_const(input_norm_const)

    num_tokens, hidden = hidden_states.shape
    capacity = x_nvfp4.shape[0]
    if num_tokens == 0:
        # Clear old routes even when this batch has no live tokens.
        topk_idx_out.fill_(-1)
        return 0
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")

    if (
        os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") != "0"
        and hidden_states.is_contiguous()
        and hidden_states.data_ptr() % 16 == 0
        and hidden % 64 == 0
        and x_sf.shape[1] == hidden // 16
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and topk_ids.data_ptr() % 16 == 0
        and topk_weights.data_ptr() % 16 == 0
        and topk_weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        from ......kernel_src.sm100.cutedsl_megamoe import (
            fused_quant_stage,
            note_staged_tokens,
        )

        # Other layers can stage prequantized inputs into this pooled workspace.
        # Their live counts do not update the shared stager's tail memo.
        note_staged_tokens(topk_idx_out, capacity)
        fused_quant_stage(
            hidden_states,
            topk_ids,
            topk_weights,
            x_nvfp4,
            x_sf,
            topk_idx_out,
            topk_weights_out,
            quant_type="nvfp4",
            norm_const=input_norm_const,
            scale_multiplier=input_norm_const / 6.0,
        )
        return num_tokens

    q, sf = quantize_nvfp4_block16(
        hidden_states.to(torch.float32), norm_const=input_norm_const
    )

    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    if x_sf.shape[1] < hidden_sf_cols:
        raise ValueError(
            f"x_sf trailing dim ({x_sf.shape[1]}) is smaller than required "
            f"{hidden_sf_cols}."
        )

    x_nvfp4[:num_tokens].view(torch.uint8).copy_(q.view(torch.uint8))
    x_sf[:num_tokens].view(torch.uint8).zero_()
    x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(sf.view(torch.uint8))
    topk_idx_out[:num_tokens].copy_(topk_ids.to(torch.int32))
    # Masked scores may be non-finite; zero them before late-weight combine.
    topk_weights_out[:num_tokens].copy_(
        topk_weights.to(torch.float32).masked_fill(topk_ids == -1, 0)
    )
    if num_tokens < capacity:
        topk_idx_out[num_tokens:capacity].fill_(-1)
    return num_tokens


def validate_sm107_nvfp4_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    scales: torch.Tensor | None = None,
) -> None:
    """SM107 nvfp4 mega-path validation (bf16 staging or pre-staged NVFP4)."""
    from ..validation import validate_forward_metadata

    validate_forward_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        fleet_params,
        top_k=top_k,
        quantize_input=quantize_input,
        quant_kind="nvfp4",
        scales=scales,
    )
