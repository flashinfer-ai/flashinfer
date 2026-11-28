# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Load-time packing of one decoder layer's MoE weights for one TP rank.

The input is the layer's full (unsharded) FP8 checkpoint, keyed by the Hugging
Face GLM-5 tensor names without the ``model.layers.{i}.`` prefix. Checkpoint
FP8 codes and FP32 scales are preserved; bytes are only rearranged.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from ._checkpoint import (
    BF16,
    F32,
    FP8,
    NORM,
    check_not_capturing,
    owned,
    target_device,
    validate_layer,
)
from .geometry import (
    EXPERT_BANKS,
    EXPERTS,
    HIDDEN_SIZE,
    LOCAL_INTERMEDIATE,
    MOE_INTERMEDIATE_SIZE,
    check_rank,
)


@dataclass(frozen=True)
class MoEWeights:
    """One rank's packed weights, shared by q1 and q4 plans.

    Bank zero is the shared expert; bank e+1 is routed expert e. Intermediate
    dimensions are tensor-parallel shards. Router, bias and gamma are replicated.
    ``gate_up`` and ``down`` hold FP8 codes (with ``gate_up`` also carrying its
    FP32 scales) in the kernel layouts documented in ``_moe_packing``;
    ``down_scale`` holds the FP32 scales of ``down``.
    """

    gamma: torch.Tensor
    router: torch.Tensor
    bias: torch.Tensor
    gate_up: torch.Tensor
    down: torch.Tensor
    down_scale: torch.Tensor


_SCALE_BLOCK = 128
# Bank b reads its projections under this prefix: the shared expert, then the
# routed experts in order.
_BANKS = ("mlp.shared_experts",) + tuple(f"mlp.experts.{e}" for e in range(EXPERTS))
_GATE_UP_SHAPES = (
    (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE),
    (MOE_INTERMEDIATE_SIZE // _SCALE_BLOCK, HIDDEN_SIZE // _SCALE_BLOCK),
)
_DOWN_SHAPES = tuple(shape[::-1] for shape in _GATE_UP_SHAPES)

# Checkpoint name -> (full shape, accepted dtypes).
_REPLICATED_INPUTS = {
    "post_attention_layernorm.weight": ((HIDDEN_SIZE,), NORM),
    "mlp.gate.weight": ((EXPERTS, HIDDEN_SIZE), (BF16,)),
    "mlp.gate.e_score_correction_bias": ((EXPERTS,), (F32, BF16)),
}


def _expert_inputs():
    contract = {}
    for prefix in _BANKS:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            weight, scale = (
                _DOWN_SHAPES if projection == "down_proj" else _GATE_UP_SHAPES
            )
            contract[f"{prefix}.{projection}.weight"] = (weight, (FP8,))
            contract[f"{prefix}.{projection}.weight_scale_inv"] = (scale, (F32,))
    return contract


_CONTRACT = _REPLICATED_INPUTS | _expert_inputs()


def shard_moe_weights(
    weights: Mapping[str, torch.Tensor], *, rank: int
) -> dict[str, torch.Tensor]:
    """Return the 1545 MoE checkpoint tensors of ``rank`` as rank-local views.

    Validates ``rank`` and ``weights`` exactly as :func:`pack_moe_weights`
    does and keeps the checkpoint names and dtypes. Rank ``r`` keeps
    intermediate channels ``[256 * r, 256 * r + 256)`` of the shared and every
    routed expert: ``gate_proj`` and ``up_proj`` rows (FP8 ``[256, 6144]``,
    scale rows ``[2 * r, 2 * r + 2)`` as FP32 ``[2, 48]``) and ``down_proj``
    columns (FP8 ``[6144, 256]``, scale columns as FP32 ``[48, 2]``). The norm
    weight, router and correction bias are returned unchanged.
    """
    rank = check_rank(rank)
    validate_layer(weights, _CONTRACT)
    rows = slice(LOCAL_INTERMEDIATE * rank, LOCAL_INTERMEDIATE * (rank + 1))
    blocks = LOCAL_INTERMEDIATE // _SCALE_BLOCK
    scale_rows = slice(blocks * rank, blocks * (rank + 1))
    local = {name: weights[name] for name in _REPLICATED_INPUTS}
    for prefix in _BANKS:
        for projection in ("gate_proj", "up_proj"):
            name = f"{prefix}.{projection}"
            local[f"{name}.weight"] = weights[f"{name}.weight"][rows]
            local[f"{name}.weight_scale_inv"] = weights[f"{name}.weight_scale_inv"][
                scale_rows
            ]
        name = f"{prefix}.down_proj"
        local[f"{name}.weight"] = weights[f"{name}.weight"][:, rows]
        local[f"{name}.weight_scale_inv"] = weights[f"{name}.weight_scale_inv"][
            :, scale_rows
        ]
    return local


def _stack_banks(local, projection, shapes, device):
    """Stack one projection of every bank into new ``[257, ...]`` storage."""
    codes = torch.empty((EXPERT_BANKS, *shapes[0]), dtype=torch.uint8, device=device)
    scales = torch.empty((EXPERT_BANKS, *shapes[1]), dtype=F32, device=device)
    for bank, prefix in enumerate(_BANKS):
        name = f"{prefix}.{projection}"
        codes[bank].copy_(local[f"{name}.weight"].view(torch.uint8))
        scales[bank].copy_(local[f"{name}.weight_scale_inv"])
    return codes.view(FP8), scales


def pack_moe_weights(
    weights: Mapping[str, torch.Tensor], *, rank: int, device=None
) -> MoEWeights:
    """Pack the MoE weights of one rank from a full checkpoint layer.

    ``weights`` maps the checkpoint names of ``_CONTRACT`` to the layer's full
    tensors; other entries are ignored. Slicing follows
    :func:`shard_moe_weights`. All required tensors share one CPU or CUDA
    device; noncontiguous views and aliasing are accepted. Only the rank-local
    slices are copied to ``device`` (default: the input device), stacked as
    gate and up FP8 ``[257, 256, 6144]`` with FP32 ``[257, 2, 48]`` scales and
    down FP8 ``[257, 6144, 256]`` with FP32 ``[257, 48, 2]`` scales, and then
    packed. No value is quantized or converted, except the exact widening of
    a BF16 norm weight or bias to FP32. The result lives on ``device``, owns
    its storage and aliases no input. Call outside CUDA graph capture.
    """
    from ._moe_packing import pack_down_slices, pack_gate_up_pages

    rank = check_rank(rank)
    target = None if device is None else target_device(device)
    local = shard_moe_weights(weights, rank=rank)
    source = local["mlp.gate.weight"].device
    target = source if target is None else target
    check_not_capturing("pack_moe_weights", source, target)
    local_shapes = (
        (LOCAL_INTERMEDIATE, HIDDEN_SIZE),
        (LOCAL_INTERMEDIATE // _SCALE_BLOCK, HIDDEN_SIZE // _SCALE_BLOCK),
    )
    device_guard = torch.cuda.device(target) if target.type == "cuda" else nullcontext()
    with device_guard, torch.no_grad():
        gate, gate_scale = _stack_banks(local, "gate_proj", local_shapes, target)
        up, up_scale = _stack_banks(local, "up_proj", local_shapes, target)
        gate_up = pack_gate_up_pages(gate, gate_scale, up, up_scale)
        del gate, gate_scale, up, up_scale
        down, down_scale = _stack_banks(
            local, "down_proj", tuple(shape[::-1] for shape in local_shapes), target
        )
        packed_down, packed_down_scale = pack_down_slices(down, down_scale)
        del down, down_scale
        return MoEWeights(
            gamma=owned(local["post_attention_layernorm.weight"], target, F32),
            router=owned(local["mlp.gate.weight"], target),
            bias=owned(local["mlp.gate.e_score_correction_bias"], target, F32),
            gate_up=gate_up.contiguous(),
            down=packed_down.contiguous(),
            down_scale=packed_down_scale.contiguous(),
        )
