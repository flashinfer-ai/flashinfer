"""Activation and routing staging for SM90 Humming MXFP4 x FP8 MegaMoE."""

from __future__ import annotations

import torch

from ......core.validation.common import MoEEpConfigError

_E4M3_DTYPE = torch.float8_e4m3fn
_E4M3_MAX = float(torch.finfo(_E4M3_DTYPE).max)
_PER_TOKEN_SCALE_EPS = 1.0e-30
_MXFP4_SCALE_WIRE_COLS = 4


def _note_staged_tokens(workspace_topk_idx: torch.Tensor, num_tokens: int) -> None:
    """Remember the live token count for compute(output=None)."""
    workspace_topk_idx._sm90_mxfp4_staged_tokens = num_tokens  # type: ignore[attr-defined]


def staged_tokens(workspace_topk_idx: torch.Tensor) -> int | None:
    """Return the live token count from the most recent stage."""
    return getattr(workspace_topk_idx, "_sm90_mxfp4_staged_tokens", None)


def _quantize_e4m3_per_token_full_hidden(
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each full hidden row with one FP32 dequant scale."""
    if hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must be 2D [tokens, hidden], got "
            f"{tuple(hidden_states.shape)}"
        )
    fp32 = hidden_states.to(torch.float32)
    absmax = fp32.abs().amax(dim=1, keepdim=True)
    scale = (absmax / _E4M3_MAX).clamp_min(_PER_TOKEN_SCALE_EPS)
    quantized = (fp32 / scale).to(_E4M3_DTYPE)
    return quantized, scale.to(torch.float32).contiguous()


def _pack_per_token_scale_for_dispatch(scale: torch.Tensor) -> torch.Tensor:
    """Replicate logical [T, 1] FP32 scales into the 16-byte [T, 4] wire."""
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    if scale.ndim != 2 or scale.shape[1] != 1:
        raise ValueError(
            "MXFP4 activation scales must have logical shape [tokens, 1], got "
            f"{tuple(scale.shape)}"
        )
    if scale.dtype != torch.float32:
        raise ValueError(
            f"MXFP4 activation scales must be torch.float32, got {scale.dtype}"
        )
    return scale.expand(scale.shape[0], _MXFP4_SCALE_WIRE_COLS).contiguous()


def _validate_stage_targets(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    if hidden_states.ndim != 2:
        raise ValueError(
            "hidden_states must be 2D [tokens, hidden], got "
            f"{tuple(hidden_states.shape)}"
        )
    num_tokens, hidden = hidden_states.shape
    capacity = x_fp8.shape[0]
    if num_tokens > capacity:
        raise ValueError(
            f"token count {num_tokens} exceeds workspace capacity {capacity}"
        )
    if x_fp8.ndim != 2 or x_fp8.shape[1] != hidden:
        raise ValueError(
            f"x_fp8 must have shape [capacity, {hidden}], got {tuple(x_fp8.shape)}"
        )
    if x_fp8.dtype != _E4M3_DTYPE:
        raise ValueError(f"x_fp8 must have dtype {_E4M3_DTYPE}, got {x_fp8.dtype}")
    if x_sf.shape != (capacity, _MXFP4_SCALE_WIRE_COLS):
        raise ValueError(
            "x_sf must have the physical FP32 [capacity, 4] Humming wire shape; "
            f"got {tuple(x_sf.shape)}"
        )
    if x_sf.dtype != torch.float32:
        raise ValueError(f"x_sf must have dtype torch.float32, got {x_sf.dtype}")
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_ids and topk_weights must be matching 2D tensors")
    top_k = topk_ids.shape[1]
    if topk_idx_out.shape != (capacity, top_k):
        raise ValueError(
            f"topk_idx_out must have shape ({capacity}, {top_k}), got "
            f"{tuple(topk_idx_out.shape)}"
        )
    if topk_weights_out.shape != (capacity, top_k):
        raise ValueError(
            f"topk_weights_out must have shape ({capacity}, {top_k}), got "
            f"{tuple(topk_weights_out.shape)}"
        )
    if topk_idx_out.dtype != torch.int64:
        raise ValueError(
            f"topk_idx_out must have dtype torch.int64, got {topk_idx_out.dtype}"
        )
    if topk_weights_out.dtype != torch.float32:
        raise ValueError(
            "topk_weights_out must have dtype torch.float32, got "
            f"{topk_weights_out.dtype}"
        )
    tensors = (
        hidden_states,
        topk_weights,
        topk_ids,
        x_fp8,
        x_sf,
        topk_idx_out,
        topk_weights_out,
    )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError(
            "all staging inputs and workspace tensors must share one device"
        )


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    quantize_input: bool,
    scales: torch.Tensor | None = None,
) -> None:
    """Stage E4M3 tokens and one-per-token scales into symmetric buffers.

    Quantized input uses one absmax-derived E4M3 scale over the complete hidden
    row. Pre-staged input supplies E4M3 payload plus logical FP32 [T, 1]
    dequant scales. In both cases the physical symmetric scale wire is [T, 4]
    with all four columns equal.
    """
    _validate_stage_targets(
        hidden_states,
        topk_weights,
        topk_ids,
        x_fp8,
        x_sf,
        topk_idx_out,
        topk_weights_out,
    )
    num_tokens = hidden_states.shape[0]
    capacity = x_fp8.shape[0]
    if num_tokens == 0:
        topk_idx_out.fill_(-1)
        _note_staged_tokens(topk_idx_out, 0)
        return

    if quantize_input:
        quantized, logical_scale = _quantize_e4m3_per_token_full_hidden(hidden_states)
    else:
        if scales is None:
            raise ValueError("pre-staged MXFP4 input requires FP32 [T, 1] scales")
        if scales.device != hidden_states.device:
            raise ValueError(
                "pre-staged activation payload and scales must share a device"
            )
        if scales.shape != (num_tokens, 1) or scales.dtype != torch.float32:
            raise ValueError(
                "pre-staged MXFP4 scales must be torch.float32 with shape "
                f"({num_tokens}, 1); got {scales.dtype} {tuple(scales.shape)}"
            )
        quantized = hidden_states
        logical_scale = scales.contiguous()

    x_fp8[:num_tokens].view(torch.uint8).copy_(quantized.contiguous().view(torch.uint8))
    x_sf[:num_tokens].copy_(_pack_per_token_scale_for_dispatch(logical_scale))
    topk_idx_out[:num_tokens].copy_(topk_ids)
    topk_weights_out[:num_tokens].copy_(topk_weights)
    if num_tokens < capacity:
        topk_idx_out[num_tokens:capacity].fill_(-1)
    _note_staged_tokens(topk_idx_out, num_tokens)


def validate_sm90_mxfp4_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    scales: torch.Tensor | None = None,
) -> None:
    """Validate BF16-to-E4M3 or pre-staged E4M3 Humming inputs."""
    from ......core.validation.common import validate_mega_forward_inputs

    validate_mega_forward_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        fleet_params,
        top_k=top_k,
        quantize_input=quantize_input,
        scales=scales,
    )
    num_tokens = hidden_states.shape[0]
    if quantize_input:
        if hidden_states.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                "MXFP4 Humming quantized input requires BF16 hidden_states; "
                f"got {hidden_states.dtype}"
            )
        if scales is not None:
            raise MoEEpConfigError(
                "MoEEpTensors.scales must be None when quantize_input=True; "
                "the backend derives one full-hidden scale per token"
            )
    else:
        if hidden_states.dtype != _E4M3_DTYPE:
            raise MoEEpConfigError(
                f"pre-staged hidden_states must have dtype {_E4M3_DTYPE}; "
                f"got {hidden_states.dtype}"
            )
        if scales is None or scales.shape != (num_tokens, 1):
            got = None if scales is None else tuple(scales.shape)
            raise MoEEpConfigError(
                f"pre-staged MXFP4 scales must have shape ({num_tokens}, 1); got {got}"
            )
        if scales.dtype != torch.float32:
            raise MoEEpConfigError(
                f"pre-staged MXFP4 scales must be torch.float32; got {scales.dtype}"
            )

    if topk_ids.dtype != torch.int64:
        raise MoEEpConfigError(f"topk_ids must be torch.int64; got {topk_ids.dtype}")
    if topk_weights.dtype != torch.float32:
        raise MoEEpConfigError(
            f"topk_weights must be torch.float32; got {topk_weights.dtype}"
        )
    named_tensors = [
        ("hidden_states", hidden_states),
        ("topk_ids", topk_ids),
        ("topk_weights", topk_weights),
    ]
    if scales is not None:
        named_tensors.append(("scales", scales))
    for name, tensor in named_tensors:
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"{name} must be contiguous")
        if not tensor.is_cuda:
            raise MoEEpConfigError(f"{name} must be a CUDA tensor")
    if len({tensor.device for _, tensor in named_tensors}) != 1:
        raise MoEEpConfigError("all forward tensors must share one CUDA device")


__all__ = [
    "stage_mega_moe_inputs",
    "staged_tokens",
    "validate_sm90_mxfp4_forward_inputs",
]
