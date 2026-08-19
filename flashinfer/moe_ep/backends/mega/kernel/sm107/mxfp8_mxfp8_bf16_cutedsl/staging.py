"""Stage bf16 activations + routing into SM107 GLU mega symmetric buffers."""

from __future__ import annotations

import torch

from ......core.validation.common import MoEEpConfigError


def _data_dtype(kind: str) -> torch.dtype:
    return torch.float8_e4m3fn if kind == "mxfp8_e4m3" else torch.float8_e5m2


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

    Torch-composed staging (the ``next/`` drop has no fused staging kernel
    yet). Returns the staged token count.
    """
    # Backend talks only to the next_cutedsl_megamoe shim (never src/ directly).
    from ......kernel_src.sm107.next_cutedsl_megamoe import (
        Mxfp8BlockSize,
        ceil_div,
        quantize_mxfp8_block32,
    )

    num_tokens, hidden = hidden_states.shape
    capacity = x_fp8.shape[0]
    if num_tokens == 0:
        # A zero-token step still owns the buffer: rows a previous batch left
        # routed must be re-masked or they would dispatch as stale live tokens.
        topk_idx_out.fill_(-1)
        return 0
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")

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
    from ......core.validation.common import validate_mega_forward_inputs

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

    from ......kernel_src.sm107.next_cutedsl_megamoe import Mxfp8BlockSize, ceil_div

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
    if hidden_states.ndim != 2 or hidden_states.shape[1] != hidden:
        raise MoEEpConfigError(
            f"pre-staged MXFP8 hidden_states must be 2D with shape "
            f"[num_tokens, {hidden}], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != _data_dtype(kind):
        raise MoEEpConfigError(
            f"pre-staged MXFP8 hidden_states must have dtype {_data_dtype(kind)}, "
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
    if scales.shape[1] < hidden_sf_cols:
        raise MoEEpConfigError(
            f"scales.shape[1] ({scales.shape[1]}) must be >= {hidden_sf_cols} "
            f"for hidden={hidden}"
        )
