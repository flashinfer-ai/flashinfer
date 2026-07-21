"""Stage bf16 activations + routing into MXFP8 mega-MoE symmetric buffers."""

from __future__ import annotations

import torch

from .....core.validation.common import MoEEpConfigError


def _mxfp8_data_dtype(kind: str) -> torch.dtype:
    # Backend talks only to the cutedsl_megamoe shim (never src/ directly); the
    # package import also bootstraps sys.path for the kernel packages.
    from .....kernel_src.cutedsl_megamoe import kind_data_dtype

    return kind_data_dtype(kind)


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
) -> None:
    """bf16 ``hidden_states`` → MXFP8 activation + E8M0 block scales."""
    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.cutedsl_megamoe import (
        Mxfp8BlockSize,
        ceil_div,
        mxfp8_quantize_per_block_32,
        round_up,
    )

    num_tokens, hidden = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden % 128 != 0:
        raise ValueError("hidden_size must be a multiple of 128.")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")

    data_dtype = _mxfp8_data_dtype(kind)
    activation_fp32 = hidden_states.to(torch.float32)
    q, sf = mxfp8_quantize_per_block_32(activation_fp32, data_dtype)

    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)
    if x_sf.shape[1] < hidden_sf_cols_padded:
        raise ValueError(
            f"x_sf trailing dim ({x_sf.shape[1]}) is smaller than required "
            f"{hidden_sf_cols_padded}."
        )

    x_fp8[:num_tokens].view(torch.uint8).copy_(q.view(torch.uint8))
    x_sf[:num_tokens].zero_()
    x_sf[:num_tokens, :hidden_sf_cols].view(torch.uint8).copy_(sf.view(torch.uint8))
    topk_idx_out[:num_tokens].copy_(topk_ids)
    topk_weights_out[:num_tokens].copy_(topk_weights)

    capacity = x_fp8.shape[0]
    if num_tokens < capacity:
        topk_idx_out[num_tokens:capacity].fill_(-1)


def validate_mxfp8_forward_inputs(
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
    """MXFP8 mega-path validation (bf16 staging or pre-staged fp8)."""
    from .....core.validation.common import validate_mega_forward_inputs

    if quantize_input:
        validate_mega_forward_inputs(
            hidden_states,
            topk_ids,
            topk_weights,
            fleet_params,
            top_k=top_k,
            quantize_input=True,
        )
        return

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.cutedsl_megamoe import (
        Mxfp8BlockSize,
        Mxfp8ScaleDtype,
        ceil_div,
    )

    num_tokens = hidden_states.shape[0]
    hidden = fleet_params.token_hidden_size
    data_dtype = _mxfp8_data_dtype(kind)
    if scales is None:
        raise MoEEpConfigError(
            "MoEEpTensors.scales is required when MegaConfig.quantize_input=False"
        )
    if num_tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError(
            f"token count {num_tokens} exceeds "
            f"max_tokens_per_rank={fleet_params.max_tokens_per_rank}"
        )
    if hidden_states.ndim != 2 or hidden_states.shape[1] != hidden:
        raise MoEEpConfigError(
            f"pre-staged MXFP8 hidden_states must be 2D with shape "
            f"[num_tokens, {hidden}], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != data_dtype:
        raise MoEEpConfigError(
            f"pre-staged MXFP8 hidden_states must have dtype {data_dtype}, "
            f"got {hidden_states.dtype}"
        )
    if topk_ids.shape != (num_tokens, top_k):
        raise MoEEpConfigError(
            f"topk_ids must have shape ({num_tokens}, {top_k}), "
            f"got {tuple(topk_ids.shape)}"
        )
    if topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("topk_weights and topk_ids must have the same shape")

    hidden_sf_cols = ceil_div(hidden, Mxfp8BlockSize)
    if scales.ndim != 2 or scales.shape[0] != num_tokens:
        raise MoEEpConfigError(
            f"scales must be 2D with leading dim {num_tokens}, got {tuple(scales.shape)}"
        )
    if scales.dtype != Mxfp8ScaleDtype:
        raise MoEEpConfigError(
            f"scales must have dtype {Mxfp8ScaleDtype}, got {scales.dtype}"
        )
    if scales.shape[1] < hidden_sf_cols:
        raise MoEEpConfigError(
            f"scales.shape[1] ({scales.shape[1]}) must be >= {hidden_sf_cols} "
            f"for hidden={hidden}"
        )
