"""Stage bf16 activations + routing into mega-MoE symmetric buffers."""

from __future__ import annotations

import os

import torch


def _use_fused_stage() -> bool:
    # Bisection escape hatch back to the multi-kernel torch staging path.
    return os.environ.get("FLASHINFER_MEGA_FUSED_STAGE", "1") != "0"


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    """bf16 ``hidden_states`` → per-32 ue8m0 fp8 for the deep_gemm mega kernel.

    Default path is the fused single-launch ``DataPreprocess`` staging kernel:
    deep_gemm's ``per_token_cast_to_fp8(use_ue8m0=True, gran_k=32,
    use_packed_ue8m0=True)`` is byte-identical to the cutedsl ``mxfp8_e4m3``
    recipe (verified data + scales; the "packed" int32 scales are the same
    e8m0 bytes viewed 4-per-word), so the deep_gemm buffers are staged through
    byte views. ``FLASHINFER_MEGA_FUSED_STAGE=0`` restores the torch path.
    """
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError("hidden_size must be a multiple of 128.")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.sm100.cutedsl_megamoe import (
        fused_quant_stage,
        fused_quant_stage_supported,
    )

    if _use_fused_stage() and fused_quant_stage_supported(
        hidden_states, quant_type="mxfp8_e4m3"
    ):
        x_out = (
            x_fp8
            if x_fp8.dtype == torch.float8_e4m3fn
            else x_fp8.view(torch.float8_e4m3fn)
        )
        # deep_gemm packs 4 e8m0 scales per int32 word; same bytes, new view.
        sf_out = (
            x_sf
            if x_sf.dtype == torch.float8_e8m0fnu
            else x_sf.view(torch.float8_e8m0fnu)
        )
        fused_quant_stage(
            hidden_states,
            topk_ids,
            topk_weights,
            x_out,
            sf_out,
            topk_idx_out,
            topk_weights_out,
            quant_type="mxfp8_e4m3",
        )
        return

    from deep_gemm.utils import per_token_cast_to_fp8

    x_q, x_sf_q = per_token_cast_to_fp8(
        hidden_states,
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    x_fp8.copy_(x_q)
    x_sf.copy_(x_sf_q)
    topk_idx_out.copy_(topk_ids.to(torch.int64))
    topk_weights_out.copy_(topk_weights)


__all__ = ["stage_mega_moe_inputs"]
