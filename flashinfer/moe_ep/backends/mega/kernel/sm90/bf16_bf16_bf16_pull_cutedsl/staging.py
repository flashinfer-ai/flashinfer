"""Stage bf16 activations + routing into SM90 BF16 mega-MoE symmetric buffers."""

from __future__ import annotations

import torch

from ......core.validation.common import MoEEpConfigError

_NAME = "sm90_bf16_bf16_bf16_pull_cutedsl"


def _note_staged_tokens(workspace_topk_idx: torch.Tensor, num_tokens: int) -> None:
    """Remember the live token count for ``compute(output=None)``.

    Same mechanism as the FP8 twin: the count rides on the staging tensor
    object (same lifetime as the workspace, overwritten every stage).
    """
    workspace_topk_idx._sm90_staged_tokens = num_tokens  # type: ignore[attr-defined]


def staged_tokens(workspace_topk_idx: torch.Tensor) -> int | None:
    """Token count from the last stage into this workspace, or None."""
    return getattr(workspace_topk_idx, "_sm90_staged_tokens", None)


def stage_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_bf16: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    """Copy bf16 ``hidden_states`` + routing into the symmetric buffers.

    No quantization: the kernel pulls the bf16 rows as-is.  Pad rows beyond
    ``num_tokens`` get ``topk_idx == -1`` (the kernel's pad mask).
    """
    num_tokens, hidden = hidden_states.shape
    if num_tokens == 0:
        # An empty batch must not inherit the previous stage's routing or
        # token count: clear the sentinel plane and record 0 so a later
        # compute(output=None) sees the empty batch.
        topk_idx_out.fill_(-1)
        _note_staged_tokens(topk_idx_out, 0)
        return
    if hidden != x_bf16.shape[1]:
        raise ValueError(
            f"hidden_states hidden ({hidden}) must match the workspace "
            f"({x_bf16.shape[1]})."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights and topk_ids must have the same shape.")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(
            f"{_NAME} hidden_states must be torch.bfloat16, got {hidden_states.dtype}."
        )

    x_bf16[:num_tokens].copy_(hidden_states)
    topk_idx_out[:num_tokens].copy_(topk_ids)
    topk_weights_out[:num_tokens].copy_(topk_weights)

    capacity = x_bf16.shape[0]
    if num_tokens < capacity:
        topk_idx_out[num_tokens:capacity].fill_(-1)
    _note_staged_tokens(topk_idx_out, num_tokens)


def validate_sm90_bf16_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    top_k: int,
    quantize_input: bool,
    scales: torch.Tensor | None = None,
) -> None:
    """SM90 BF16 mega-path validation.

    The activation is bf16 either way: ``quantize_input=True`` is the normal
    layer path, ``quantize_input=False`` ("pre-staged") accepts the same bf16
    rows (there is nothing to quantize) but must not carry scales.
    """
    from ......core.validation.common import validate_mega_forward_inputs

    if scales is not None:
        raise MoEEpConfigError(f"{_NAME} does not take activation scales (native BF16)")
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
    if num_tokens > fleet_params.max_tokens_per_rank:
        raise MoEEpConfigError(
            f"token count {num_tokens} exceeds "
            f"max_tokens_per_rank={fleet_params.max_tokens_per_rank}"
        )
    if hidden_states.ndim != 2 or hidden_states.shape[1] != hidden:
        raise MoEEpConfigError(
            f"pre-staged hidden_states must be 2D with shape "
            f"[num_tokens, {hidden}], got {tuple(hidden_states.shape)}"
        )
    if hidden_states.dtype != torch.bfloat16:
        raise MoEEpConfigError(
            f"{_NAME} pre-staged hidden_states must be torch.bfloat16, "
            f"got {hidden_states.dtype}"
        )
    if topk_ids.shape != (num_tokens, top_k):
        raise MoEEpConfigError(
            f"topk_ids must have shape ({num_tokens}, {top_k}), "
            f"got {tuple(topk_ids.shape)}"
        )
    if topk_weights.shape != topk_ids.shape:
        raise MoEEpConfigError("topk_weights and topk_ids must have the same shape")
