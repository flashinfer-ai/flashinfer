# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Load-time packing of one decoder layer's attention and indexer weights.

The input is the layer's full (unsharded) FP8 checkpoint, keyed by the Hugging
Face GLM-5 tensor names without the ``model.layers.{i}.`` prefix. Logical rank
zero packs the DSA indexer; ranks one through seven each pack ten MLA head
slots. The packer takes the rank's slices itself, so every rank can pass the
same layer mapping.
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
    narrow_padded,
    owned,
    target_device,
    to_device,
    validate_layer,
)
from .geometry import (
    HEADS_PER_MLA_RANK,
    NUM_HEADS,
    QK_NOPE_HEAD_DIM,
    ROPE_SIZE,
    V_HEAD_DIM,
    check_rank,
)


@dataclass(frozen=True)
class AttentionWeights:
    """Owned weights for one logical TP rank, shared by q1 and q4 plans.

    ``tensors`` contains contiguous, independently owned output storage. It does
    not retain the checkpoint mapping or alias its tensors. Treat this mapping
    and its tensors as read-only after constructing a plan. Projection pages
    contain FP8 codes and FP32 scale bytes in uint8 storage, laid out as
    documented in ``_attention_packing``; ``indexer_weights_proj`` is BF16 and
    the norm vectors are FP32.
    """

    rank: int
    tensors: dict[str, torch.Tensor]


_SCALE_BLOCK = 128
_Q_HEAD_ROWS = QK_NOPE_HEAD_DIM + ROPE_SIZE  # q_b_proj rows per head (256)
_KV_B_HEAD_ROWS = QK_NOPE_HEAD_DIM + V_HEAD_DIM  # kv_b_proj rows per head (448)
_V_COLUMNS = V_HEAD_DIM  # o_proj columns per head (256)

# Checkpoint name -> (full shape, accepted dtypes).
_COMMON_INPUTS = {
    "input_layernorm.weight": ((6144,), NORM),
    "self_attn.q_a_proj.weight": ((2048, 6144), (FP8,)),
    "self_attn.q_a_proj.weight_scale_inv": ((16, 48), (F32,)),
    "self_attn.q_a_layernorm.weight": ((2048,), NORM),
}
_INDEXER_INPUTS = {
    "self_attn.indexer.wk.weight": ((128, 6144), (FP8,)),
    "self_attn.indexer.wk.weight_scale_inv": ((1, 48), (F32,)),
    "self_attn.indexer.wq_b.weight": ((4096, 2048), (FP8,)),
    "self_attn.indexer.wq_b.weight_scale_inv": ((32, 16), (F32,)),
    "self_attn.indexer.weights_proj.weight": ((32, 6144), (BF16,)),
    "self_attn.indexer.k_norm.weight": ((128,), NORM),
    "self_attn.indexer.k_norm.bias": ((128,), NORM),
}
_MLA_INPUTS = {
    "self_attn.kv_a_proj_with_mqa.weight": ((576, 6144), (FP8,)),
    "self_attn.kv_a_proj_with_mqa.weight_scale_inv": ((5, 48), (F32,)),
    "self_attn.kv_a_layernorm.weight": ((512,), NORM),
    "self_attn.q_b_proj.weight": ((16384, 2048), (FP8,)),
    "self_attn.q_b_proj.weight_scale_inv": ((128, 16), (F32,)),
    "self_attn.kv_b_proj.weight": ((28672, 512), (FP8,)),
    "self_attn.kv_b_proj.weight_scale_inv": ((224, 4), (F32,)),
    "self_attn.o_proj.weight": ((6144, 16384), (FP8,)),
    "self_attn.o_proj.weight_scale_inv": ((48, 128), (F32,)),
}
_INDEXER_CONTRACT = _COMMON_INPUTS | _INDEXER_INPUTS
_MLA_CONTRACT = _COMMON_INPUTS | _MLA_INPUTS


def _head_slots(rank: int) -> tuple[int, int]:
    """Return the first global head and the number of real heads of MLA ``rank``."""
    first = HEADS_PER_MLA_RANK * (rank - 1)
    return first, min(HEADS_PER_MLA_RANK, NUM_HEADS - first)


def shard_attention_weights(
    weights: Mapping[str, torch.Tensor], *, rank: int
) -> dict[str, torch.Tensor]:
    """Return the checkpoint tensors that ``rank`` packs, with rank-local shapes.

    Validates ``rank`` and ``weights`` exactly as
    :func:`pack_attention_weights` does and keeps the checkpoint names and
    dtypes. Rank 0 receives its eleven indexer-rank tensors unchanged. MLA rank
    ``r`` owns global heads ``h0 = 10 * (r - 1)`` to ``h0 + hv - 1`` with
    ``hv = min(10, 64 - h0)`` and receives thirteen tensors: the replicated ones
    unchanged, and

    * ``self_attn.q_b_proj.weight`` rows ``[256 * h0, 256 * (h0 + hv))`` as
      FP8 ``[2560, 2048]``, scale rows ``[2 * h0, 2 * (h0 + hv))`` as FP32
      ``[20, 16]``;
    * ``self_attn.kv_b_proj.weight`` rows ``[448 * h0, 448 * (h0 + hv))`` as
      FP8 ``[4480, 512]``, scale rows ``[3.5 * h0, 3.5 * (h0 + hv))`` as FP32
      ``[35, 4]`` (exact 128-row block boundaries, since 448 * 10 = 35 * 128
      and 448 * 4 = 14 * 128);
    * ``self_attn.o_proj.weight`` columns ``[256 * h0, 256 * (h0 + hv))`` as
      FP8 ``[6144, 2560]``, scale columns ``[2 * h0, 2 * (h0 + hv))`` as FP32
      ``[48, 20]``.

    Ranks 0 through 6 receive views of the inputs. Rank 7 owns four real heads;
    its three per-head projections and their scales are new tensors whose six
    absent head slots hold zero codes and zero scales.
    """
    rank = check_rank(rank)
    contract = _INDEXER_CONTRACT if rank == 0 else _MLA_CONTRACT
    validate_layer(weights, contract)
    local = {name: weights[name] for name in contract}
    if rank == 0:
        return local
    first, count = _head_slots(rank)
    slots = HEADS_PER_MLA_RANK
    for name, dim, unit in (
        ("self_attn.q_b_proj", 0, _Q_HEAD_ROWS),
        ("self_attn.kv_b_proj", 0, _KV_B_HEAD_ROWS),
        ("self_attn.o_proj", 1, _V_COLUMNS),
    ):
        weight, scale = f"{name}.weight", f"{name}.weight_scale_inv"
        start, length, size = unit * first, unit * count, unit * slots
        local[weight] = narrow_padded(weights[weight], dim, start, length, size)
        # Every rank's head range starts and ends on a 128-row scale block.
        local[scale] = narrow_padded(
            weights[scale],
            dim,
            start // _SCALE_BLOCK,
            length // _SCALE_BLOCK,
            size // _SCALE_BLOCK,
        )
    return local


def pack_attention_weights(
    weights: Mapping[str, torch.Tensor], *, rank: int, device=None
) -> AttentionWeights:
    """Pack the attention weights of one rank from a full checkpoint layer.

    ``weights`` maps checkpoint names to the layer's full tensors; the required
    names, shapes and dtypes are those of ``_COMMON_INPUTS`` plus
    ``_INDEXER_INPUTS`` on rank 0 or ``_MLA_INPUTS`` on ranks 1 through 7, and
    other entries are ignored. Slicing follows :func:`shard_attention_weights`.
    All required tensors share one CPU or CUDA device; noncontiguous views are
    accepted. Rank-local slices are taken on that device and copied to
    ``device`` (default: the input device), where the kernel layouts are built.
    No value is quantized or converted, except the exact widening of BF16 norm
    weights to FP32.

    Rank 0 returns ``x_gamma``, ``q_gamma``, ``q_a_indexer_wk``,
    ``indexer_weights_proj``, ``indexer_wq_b``, ``ln_gamma`` and ``ln_beta``.
    Ranks 1 through 7 return ``x_gamma``, ``q_gamma``, ``kv_gamma``,
    ``qkv_a_proj``, ``q_b_proj``, ``w_uk``, ``w_uv`` and ``o_proj``. The
    returned tensors live on ``device``, own their storage and alias no input.
    Call outside CUDA graph capture.
    """
    from . import _attention_packing as packing

    rank = check_rank(rank)
    target = None if device is None else target_device(device)
    local = shard_attention_weights(weights, rank=rank)
    source = local["self_attn.q_a_proj.weight"].device
    target = source if target is None else target
    check_not_capturing("pack_attention_weights", source, target)
    device_guard = torch.cuda.device(target) if target.type == "cuda" else nullcontext()
    with device_guard, torch.no_grad():
        w = {name: to_device(tensor, target) for name, tensor in local.items()}
        del local
        tensors = {
            "x_gamma": owned(w["input_layernorm.weight"], target, F32),
            "q_gamma": owned(w["self_attn.q_a_layernorm.weight"], target, F32),
        }
        q_a = (w["self_attn.q_a_proj.weight"], w["self_attn.q_a_proj.weight_scale_inv"])
        if rank == 0:
            tensors.update(
                q_a_indexer_wk=packing.pack_q_a_indexer_wk(
                    *q_a,
                    w["self_attn.indexer.wk.weight"],
                    w["self_attn.indexer.wk.weight_scale_inv"],
                ),
                indexer_weights_proj=packing.pack_indexer_weights_proj(
                    w["self_attn.indexer.weights_proj.weight"]
                ),
                indexer_wq_b=packing.pack_indexer_wq_b(
                    w["self_attn.indexer.wq_b.weight"],
                    w["self_attn.indexer.wq_b.weight_scale_inv"],
                ),
                ln_gamma=owned(w["self_attn.indexer.k_norm.weight"], target, F32),
                ln_beta=owned(w["self_attn.indexer.k_norm.bias"], target, F32),
            )
        else:
            w_uk, w_uk_scale, w_uv, w_uv_scale = packing.split_kv_b_proj(
                w["self_attn.kv_b_proj.weight"],
                w["self_attn.kv_b_proj.weight_scale_inv"],
            )
            tensors.update(
                kv_gamma=owned(w["self_attn.kv_a_layernorm.weight"], target, F32),
                qkv_a_proj=packing.pack_qkv_a_proj(
                    *q_a,
                    w["self_attn.kv_a_proj_with_mqa.weight"],
                    w["self_attn.kv_a_proj_with_mqa.weight_scale_inv"],
                ),
                q_b_proj=packing.pack_q_b_proj(
                    w["self_attn.q_b_proj.weight"],
                    w["self_attn.q_b_proj.weight_scale_inv"],
                ),
                w_uk=packing.pack_w_uk(w_uk, w_uk_scale),
                w_uv=packing.pack_w_uv(w_uv, w_uv_scale),
                o_proj=packing.pack_o_proj(
                    w["self_attn.o_proj.weight"],
                    w["self_attn.o_proj.weight_scale_inv"],
                ),
            )
    return AttentionWeights(rank=rank, tensors=tensors)


_PACKED_INDEXER = {
    "x_gamma": ((6144,), F32),
    "q_gamma": ((2048,), F32),
    "q_a_indexer_wk": ((68 * 6 * 32896,), torch.uint8),
    "indexer_weights_proj": ((2, 98304), BF16),
    "indexer_wq_b": ((128, 2, 32896), torch.uint8),
    "ln_gamma": ((128,), F32),
    "ln_beta": ((128,), F32),
}
_PACKED_MLA = {
    "x_gamma": ((6144,), F32),
    "q_gamma": ((2048,), F32),
    "kv_gamma": ((512,), F32),
    "qkv_a_proj": ((82 * 6 * 32896,), torch.uint8),
    "q_b_proj": ((80, 2, 32896), torch.uint8),
    "w_uk": ((80, 12416), torch.uint8),
    "w_uv": ((80, 16512), torch.uint8),
    "o_proj": ((128, 5, 24704), torch.uint8),
}


def validate_packed(
    weights: AttentionWeights, rank: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Validate the packed ABI before uploading any pointer-only descriptors.

    Return the checked tensor mapping. No device data is read or synchronized;
    preparation remains responsible for collectively propagating any failure.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < 8:
        raise ValueError("rank must be an integer in [0, 7]")
    if (
        not isinstance(weights, AttentionWeights)
        or isinstance(weights.rank, bool)
        or not isinstance(weights.rank, int)
        or weights.rank != rank
    ):
        raise ValueError("weights must be packed for this logical rank")
    if not isinstance(weights.tensors, dict):
        raise TypeError("packed weights.tensors must be a dictionary")
    tensors = weights.tensors
    contract = _PACKED_INDEXER if rank == 0 else _PACKED_MLA
    missing = contract.keys() - tensors.keys()
    extra = tensors.keys() - contract.keys()
    if missing or extra:
        raise ValueError(
            f"rank {rank} packed weight keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(map(str, extra))}"
        )
    device = torch.device(device)
    for name, (shape, dtype) in contract.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"packed {name} must be a torch.Tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"packed {name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise TypeError(
                f"packed {name} must have dtype {dtype}, got {tensor.dtype}"
            )
        if tensor.device != device:
            raise ValueError(f"packed {name} must be on {device}, got {tensor.device}")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"packed {name} must be contiguous with strided layout")
        if tensor.data_ptr() % 16:
            raise ValueError(f"packed {name} must be 16-byte aligned")
        if tensor.requires_grad:
            raise ValueError(f"packed {name} does not support autograd")
    return tensors
