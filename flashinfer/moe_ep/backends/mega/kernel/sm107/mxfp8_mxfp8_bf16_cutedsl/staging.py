"""Stage bf16 activations + routing into SM107 GLU mega symmetric buffers."""

from __future__ import annotations

import os

import torch


def _data_dtype(kind: str) -> torch.dtype:
    if kind in ("mxfp8_e4m3", "mxfp4_mxfp8"):
        return torch.float8_e4m3fn
    if kind == "mxfp8_e5m2":
        return torch.float8_e5m2
    raise ValueError(f"unsupported MXFP8 activation kind {kind!r}")


def _fused_quant_type(kind: str) -> str:
    # MXFP4-weight kinds still take MXFP8 E4M3 activations.
    return "mxfp8_e5m2" if kind == "mxfp8_e5m2" else "mxfp8_e4m3"


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
) -> int:
    """bf16 ``hidden_states`` -> MXFP8 activation + E8M0 block scales.

    Uses the fused SM100 CuTe-DSL stager (one launch: quantize + routing
    repack) when the layout allows, else the torch-composed path. Both round
    the E8M0 scale up from ``amax / fp8_max`` and the data to nearest even, so
    they stage identical data; only an all-zero block's (unused) scale byte
    may differ. Returns the staged token count.
    """
    from ......kernel_src.sm107.next_cutedsl_megamoe import (
        Mxfp8BlockSize,
        ceil_div,
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

    if (
        os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") != "0"
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and hidden_states.data_ptr() % 16 == 0
        and hidden % (4 * Mxfp8BlockSize) == 0
        and x_fp8.dtype == _data_dtype(kind)
        and x_sf.shape[1] == hidden // Mxfp8BlockSize
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
            x_fp8,
            x_sf,
            topk_idx_out,
            topk_weights_out,
            quant_type=_fused_quant_type(kind),
        )
        return num_tokens

    q, sf = quantize_mxfp8_block32(hidden_states.to(torch.float32), _data_dtype(kind))

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
    topk_weights_out[:num_tokens].copy_(topk_weights.to(torch.float32))
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
