"""Stage bf16 activations + routing into SM107 nvfp4 mega symmetric buffers."""

from __future__ import annotations

import torch


def _fast_nvfp4_quantize(hidden_states: torch.Tensor, norm_const: float, out=None):
    """One-launch NVFP4 quantization (flashinfer CuTe-DSL kernel, linear E4M3 scales).

    The torch-composed ``quantize_nvfp4_block16`` materialises an 8-way fp32 distance
    tensor per element (~32x the bf16 input bytes, >10 launches); this kernel reads the
    bf16 rows once. Same recipe: ``sf = e4m3(amax / 6 * norm_const)``, data RN-even E2M1
    of ``x * norm_const / sf`` (the TRT-LLM / vLLM ``cvt`` semantics; reciprocal fast-math
    unless ``FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH=1``). Returns ``None`` when the input
    is not supported, so the caller falls back to the torch path.
    """
    if hidden_states.dtype not in (torch.bfloat16, torch.float16):
        return None
    if hidden_states.shape[1] % 64 or not hidden_states.is_contiguous():
        return None
    from .......quantization.kernels.nvfp4_quantize import (
        SF_LAYOUT_LINEAR,
        nvfp4_quantize_cute_dsl,
    )

    q, sf = nvfp4_quantize_cute_dsl(
        hidden_states,
        float(norm_const),
        sf_layout=SF_LAYOUT_LINEAR,
        nvfp4_4over6=None,
        out=out,
    )
    return q, sf.reshape(hidden_states.shape[0], -1)


def _direct_outputs(hidden_states, x_nvfp4, x_sf, hidden_sf_cols):
    """Workspace destinations for a quantize-in-place staging, or None.

    The packed rows ``x_nvfp4[:n]`` and the linear scale rows ``x_sf[:n]`` are written
    by the quantizer itself (no DtoD copies) when the workspace scale plane has
    exactly ``hidden / 16`` columns (then its flat prefix is the linear layout) and
    holds the padded scale rows the TMA path writes (``round_up(n, 16)``)."""
    n = hidden_states.shape[0]
    if (
        x_sf.shape[1] != hidden_sf_cols
        or not x_sf.is_contiguous()
        or not x_nvfp4.is_contiguous()
        or x_sf.shape[0] < ((n + 15) // 16) * 16
    ):
        return None
    return (
        x_nvfp4[:n].view(torch.uint8),
        x_sf.view(torch.uint8).view(-1),
    )


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
    routes_staged: bool = False,
) -> int:
    """bf16 ``hidden_states`` -> packed NVFP4 activation + FP8-E4M3 block scales.

    The packed data times its block scales approximates
    ``input_norm_const * hidden_states``; fc1_alpha must compensate for this
    normalization and any weight normalization. Returns the staged token count.
    ``routes_staged=True``: the routing plane (``topk_idx_out`` /
    ``topk_weights_out``, capacity tail included) was already written by the
    one-launch route staging, so it is not copied again.
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

    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    if x_sf.shape[1] < hidden_sf_cols:
        raise ValueError(
            f"x_sf trailing dim ({x_sf.shape[1]}) is smaller than required "
            f"{hidden_sf_cols}."
        )

    direct = _direct_outputs(hidden_states, x_nvfp4, x_sf, hidden_sf_cols)
    fast = (
        _fast_nvfp4_quantize(hidden_states, input_norm_const, out=direct)
        if direct is not None
        else None
    )
    if fast is None:
        fast = _fast_nvfp4_quantize(hidden_states, input_norm_const)
        if fast is not None:
            q, sf = fast
        else:
            q, sf = quantize_nvfp4_block16(
                hidden_states.to(torch.float32), norm_const=input_norm_const
            )
        x_nvfp4[:num_tokens].view(torch.uint8).copy_(q.view(torch.uint8))
        if x_sf.shape[1] != hidden_sf_cols:
            x_sf[:num_tokens].view(torch.uint8).zero_()
        x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(sf.view(torch.uint8))
    if not routes_staged:
        topk_idx_out[:num_tokens].copy_(topk_ids.to(torch.int32))
        topk_weights_out[:num_tokens].copy_(topk_weights.to(torch.float32))
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
