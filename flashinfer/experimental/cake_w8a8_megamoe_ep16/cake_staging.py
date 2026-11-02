"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Forward-input validation and staging for the Cake W8A8 MXFP8 MegaMoE EP16 route.
#
# Input contract (per rank): ``hidden_states`` bf16 ``[n, 3072]``, ``topk_ids``
# int64 ``[n, 8]`` with global expert ids in ``[0, 512)`` or ``-1`` (a masked
# slot, which contributes zero), ``topk_weights`` float32 ``[n, 8]``;
# ``0 <= n <= bound <= 64``. Every rank's ``local_pack`` waits for one pushed
# slot per (source rank, token) up to ``tokens_per_rank``, so every rank
# publishes the same rank-uniform ``bound`` and pads rows ``[n, bound)`` with
# zero activations, ``-1`` routes (owned by no rank: ``publish_push`` still
# pushes the slot's 64-byte ids+weights tail and flag, and ``local_pack`` packs
# nothing for it) and zero weights. The padded output rows are never returned.
# The publish-push program quantizes the staged bf16 rows in-kernel;
# pre-quantized activations are not accepted.
#
# FC2 writes a term slot only for a route ``local_pack`` packed, and the
# combine program sums all eight slots of a token unconditionally; the
# publish-push program zeroes this rank's combine read window of the term
# buffer in-kernel before it releases its inbox flags, so a masked or
# overflowed slot adds zero instead of a previous forward's term. Staging
# itself launches nothing and performs no memset.

from __future__ import annotations

import torch

from ...moe_ep.core.validation.common import (
    MoEEpConfigError,
    validate_mega_forward_inputs,
)
from .cake_config import KERNEL_NAME
from .cake_plan import HIDDEN, MAX_TOKENS_PER_RANK, TOP_K

MASKED_ROUTE = -1


def resolve_bound(
    num_tokens: int, compile_tokens_per_rank: int | None, capacity: int
) -> int:
    """Rank-uniform token bound of one invocation.

    ``compile_tokens_per_rank`` (set collectively through
    ``MoEEpMegaLayer.stage_inputs``) wins when given; otherwise the workspace
    capacity (``FleetParams.max_tokens_per_rank``) is the bound.
    """

    if not 1 <= capacity <= MAX_TOKENS_PER_RANK:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: workspace capacity must be in [1, {MAX_TOKENS_PER_RANK}], got {capacity}"
        )
    bound = capacity if compile_tokens_per_rank is None else compile_tokens_per_rank
    if isinstance(bound, bool) or not isinstance(bound, int):
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: compile_tokens_per_rank must be an int or None"
        )
    if not 1 <= bound <= capacity:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: compile_tokens_per_rank={bound} must be in [1, {capacity}] "
            "(the workspace capacity)"
        )
    if num_tokens > bound:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: token bound {bound} is below this rank's {num_tokens} tokens"
        )
    return bound


def validate_forward_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    fleet_params,
    *,
    quantize_input: bool,
    scales: torch.Tensor | None,
) -> None:
    """Per-workspace (once) validation of the forward tensors."""

    if not quantize_input or scales is not None:
        raise MoEEpConfigError(
            f"{KERNEL_NAME} quantizes activations in-kernel; prequantized inputs "
            "(MegaConfig.quantize_input=False / MoEEpTensors.scales) are not supported"
        )
    validate_mega_forward_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        fleet_params,
        top_k=TOP_K,
        quantize_input=True,
    )
    if hidden_states.dtype != torch.bfloat16:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: hidden_states must be bf16, got {hidden_states.dtype}"
        )
    if hidden_states.shape[1] != HIDDEN:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: hidden_states must have {HIDDEN} columns, got {hidden_states.shape[1]}"
        )
    if topk_ids.dtype != torch.int64:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: topk_ids must be int64, got {topk_ids.dtype}"
        )
    if topk_weights.dtype != torch.float32:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: topk_weights must be float32, got {topk_weights.dtype}"
        )
    for name, tensor in (
        ("hidden_states", hidden_states),
        ("topk_ids", topk_ids),
        ("topk_weights", topk_weights),
    ):
        if not tensor.is_cuda:
            raise MoEEpConfigError(f"{KERNEL_NAME}: {name} must be a CUDA tensor")
        if not tensor.is_contiguous():
            raise MoEEpConfigError(f"{KERNEL_NAME}: {name} must be contiguous")
        if tensor.device != hidden_states.device:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: {name} must be on {hidden_states.device}, got {tensor.device}"
            )


def stage_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    hidden_out: torch.Tensor,
    ids_out: torch.Tensor,
    weights_out: torch.Tensor,
    bound: int,
) -> int:
    """Copy one invocation into the fixed-capacity staging buffers.

    Rows ``[n, bound)`` are padded (zero hidden, ``-1`` routes, zero weights);
    rows at and above ``bound`` are left untouched (the programs read only
    ``bound`` rows). Runs on the current stream, allocation-free. Returns ``n``.
    """

    n = int(hidden_states.shape[0])
    if n > bound or bound > hidden_out.shape[0]:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: cannot stage {n} tokens under bound {bound} into a "
            f"{hidden_out.shape[0]}-row workspace"
        )
    if n:
        hidden_out[:n].copy_(hidden_states)
        ids_out[:n].copy_(topk_ids)
        weights_out[:n].copy_(topk_weights)
    if n < bound:
        hidden_out[n:bound].zero_()
        ids_out[n:bound].fill_(MASKED_ROUTE)
        weights_out[n:bound].zero_()
    return n


__all__ = [
    "MASKED_ROUTE",
    "resolve_bound",
    "stage_inputs",
    "validate_forward_inputs",
]
