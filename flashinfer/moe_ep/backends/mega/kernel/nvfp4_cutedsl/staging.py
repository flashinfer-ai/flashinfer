"""Stage bf16 activations + routing into NVFP4 mega-MoE symmetric buffers."""

from __future__ import annotations

import torch

from .....core.validation.common import MoEEpConfigError


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_nvfp4: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    norm_const: float = 1.0,
) -> None:
    """bf16 ``hidden_states`` → NVFP4 activation + fp8 block scales."""
    # Backend talks only to the cutedsl_megamoe shim (never src/ directly); the
    # package import also bootstraps sys.path for the kernel packages.
    from .....kernel_src.cutedsl_megamoe import (
        Nvfp4BlockSize,
        ceil_div,
        nvfp4_quantize_per_block_16,
        round_up,
    )

    num_tokens, hidden = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden % 128 != 0:
        raise ValueError("hidden_size must be a multiple of 128.")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")

    activation_fp32 = hidden_states.to(torch.float32)
    q, sf = nvfp4_quantize_per_block_16(activation_fp32, norm_const)

    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)
    if x_sf.shape[1] < hidden_sf_cols_padded:
        raise ValueError(
            f"x_sf trailing dim ({x_sf.shape[1]}) is smaller than required "
            f"{hidden_sf_cols_padded}."
        )

    x_nvfp4[:num_tokens].copy_(q)
    x_sf[:num_tokens].zero_()
    x_sf[:num_tokens, :hidden_sf_cols].copy_(sf)
    topk_idx_out[:num_tokens].copy_(topk_ids)
    topk_weights_out[:num_tokens].copy_(topk_weights)

    capacity = x_nvfp4.shape[0]
    if num_tokens < capacity:
        topk_idx_out[num_tokens:capacity].fill_(-1)


def validate_nvfp4_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    scales: torch.Tensor | None = None,
) -> None:
    """NVFP4 mega-path validation (bf16 staging or pre-staged NVFP4)."""
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

    num_tokens = hidden_states.shape[0]
    hidden = fleet_params.token_hidden_size
    if scales is None:
        raise MoEEpConfigError(
            "MoEEpTensors.scales is required when MegaConfig.quantize_input=False"
        )
    if num_tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError(
            f"token count {num_tokens} exceeds "
            f"max_tokens_per_rank={fleet_params.max_tokens_per_rank}"
        )
    if hidden % 2 != 0:
        raise MoEEpConfigError(
            f"token_hidden_size ({hidden}) must be even for NVFP4 packing"
        )
    packed_hidden = hidden // 2
    if hidden_states.ndim != 2 or hidden_states.shape[1] != packed_hidden:
        raise MoEEpConfigError(
            f"pre-staged NVFP4 hidden_states must be 2D with shape "
            f"[num_tokens, {packed_hidden}], got {tuple(hidden_states.shape)}"
        )
    if topk_ids.shape != (num_tokens, top_k):
        raise MoEEpConfigError(
            f"topk_ids must have shape ({num_tokens}, {top_k}), "
            f"got {tuple(topk_ids.shape)}"
        )
    if topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("topk_weights and topk_ids must have the same shape")

    # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
    from .....kernel_src.cutedsl_megamoe import Nvfp4BlockSize, ceil_div

    hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
    if scales.ndim != 2 or scales.shape[0] != num_tokens:
        raise MoEEpConfigError(
            f"scales must be 2D with leading dim {num_tokens}, got {tuple(scales.shape)}"
        )
    if scales.shape[1] < hidden_sf_cols:
        raise MoEEpConfigError(
            f"scales.shape[1] ({scales.shape[1]}) must be >= {hidden_sf_cols} "
            f"for hidden={hidden}"
        )
