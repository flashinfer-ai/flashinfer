"""Stage bf16 activations + routing into SM107 GLU mega symmetric buffers."""

from __future__ import annotations

import os

import torch


def _data_dtype(kind: str) -> torch.dtype:
    if kind == "mxfp4":
        return getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    if kind in ("mxfp8_e4m3", "mxfp4_mxfp8"):
        return torch.float8_e4m3fn
    if kind == "mxfp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unsupported MX activation kind {kind!r}")


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    kind: str = "mxfp8_e4m3",
    num_experts: int | None = None,
    workspace=None,
) -> int:
    """bf16 ``hidden_states`` -> MXFP4 or MXFP8 with per-32 E8M0 scales.

    Returns the staged token count. Unsupported layouts use Torch staging.
    """
    from ......kernel_src.sm107.next_cutedsl_megamoe import (
        Mxfp8BlockSize,
        ceil_div,
        quantize_mxfp4_block32,
        quantize_mxfp8_block32,
    )

    num_tokens, hidden = hidden_states.shape
    capacity = x_fp8.shape[0]
    if num_tokens == 0:
        # Clear old routes even when this batch has no live tokens.
        topk_idx_out.fill_(-1)
        return 0
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")

    fused_stage = (
        os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") != "0"
        and hidden_states.is_cuda
        and hidden_states.is_contiguous()
        and hidden_states.data_ptr() % 16 == 0
        and hidden % 128 == 0
        and x_sf.shape[1] == hidden // 32
        and topk_ids.shape[1] <= 128
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and topk_ids.data_ptr() % 16 == 0
        and topk_weights.data_ptr() % 16 == 0
        and topk_weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    check_routes = (
        num_experts is not None
        and os.environ.get("FLASHINFER_SM107_MEGA_CHECK_ROUTES", "0") != "0"
    )
    fused_check = (
        check_routes
        and fused_stage
        and topk_ids.shape[1] <= 32
        and os.environ.get("FLASHINFER_SM107_FUSED_ROUTE_CHECK", "1") != "0"
    )
    if check_routes and not fused_check:
        from ..validation import validate_routing_values

        validate_routing_values(
            topk_ids, topk_weights, num_experts, workspace=workspace
        )

    if fused_stage:
        from ......kernel_src.sm100.cutedsl_megamoe import (
            fused_quant_stage,
            note_staged_tokens,
        )

        # Pooled workspaces may have been written by another staging path.
        note_staged_tokens(topk_idx_out, capacity)
        fused_quant_stage(
            hidden_states,
            topk_ids,
            topk_weights,
            x_fp8,
            x_sf,
            topk_idx_out,
            topk_weights_out,
            quant_type="mxfp8_e4m3" if kind == "mxfp4_mxfp8" else kind,
            mxfp8_normal_scales=True,
            num_experts=num_experts if fused_check else None,
            fuse_tail_mask=os.environ.get("FLASHINFER_SM107_FUSED_TAIL_MASK", "1")
            != "0",
            zero_masked_scores=False,
        )
        return num_tokens

    q, sf = (
        quantize_mxfp4_block32(hidden_states.to(torch.float32))
        if kind == "mxfp4"
        else quantize_mxfp8_block32(hidden_states.to(torch.float32), _data_dtype(kind))
    )

    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    if x_sf.shape[1] < hidden_sf_cols:
        raise ValueError(
            f"x_sf trailing dim ({x_sf.shape[1]}) is smaller than required "
            f"{hidden_sf_cols}."
        )

    x_fp8[:num_tokens].view(torch.uint8).copy_(q.view(torch.uint8))
    x_sf[:num_tokens].view(torch.uint8).zero_()
    x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(sf.view(torch.uint8))
    topk_idx_out[:num_tokens].copy_(topk_ids.to(torch.int32))
    topk_weights_out[:num_tokens].copy_(topk_weights)
    if num_tokens < capacity:
        topk_idx_out[num_tokens:capacity].fill_(-1)
    return num_tokens


def validate_sm107_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    kind: str = "mxfp8_e4m3",
    scales: torch.Tensor | None = None,
) -> None:
    """SM107 mega-path validation (bf16 staging or pre-staged mxfp8)."""
    from ..validation import validate_forward_metadata

    validate_forward_metadata(
        hidden_states,
        topk_ids,
        topk_weights,
        fleet_params,
        top_k=top_k,
        quantize_input=quantize_input,
        quant_kind=kind,
        scales=scales,
    )
