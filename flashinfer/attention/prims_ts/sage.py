# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sage attention inputs for the PrimTS decode kernels.

Sage attention runs ``QK^T`` on 8-bit Q/K with one dequantization scale per
token block and ``PV`` on E4M3 P/V with one scale per V channel. The scale
tensors follow the trtllm-gen flat layout produced by TensorRT-LLM's
``sageQuant``: per head, sequence ``b`` starts at ``b * S // blk + b`` and token
``t`` uses slot ``t // blk`` inside it, so ``ceil(B * S / blk) + B - 1`` slots
cover a fixed ``[B, S, H, D]`` tensor.

:class:`SageAttentionConfig` is the compile-time recipe a plan is built for;
:class:`SageAttentionParams` carries the scale tensors of one run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

import torch

from flashinfer.utils import ceil_div

SAGE_K_BLOCK_SIZES = (16, 32, 64, 128, 256)
# The Sage scale slots of the contiguous attention adapter, in ABI order.
SAGE_ADAPTER_SLOTS = ("q_scale", "k_scale", "v_scale", "v_mean")

_T = TypeVar("_T")


@dataclass(frozen=True)
class SageAttentionConfig:
    """Compile-time Sage attention recipe of one plan.

    ``q_block_size`` tokens share one Q scale and must be a power of two no
    larger than the planned Q tile; ``k_block_size`` tokens share one K scale
    and must be one of ``SAGE_K_BLOCK_SIZES``. ``v_mean`` says whether every
    run supplies a per-channel V mean that is added back to the normalized
    output. The defaults are TensorRT-LLM's production recipe ``(1, 16, 1)``.
    """

    q_block_size: int = 1
    k_block_size: int = 16
    v_mean: bool = False


@dataclass(frozen=True)
class SageAttentionParams:
    """Per-block Q/K scales and per-channel V scales of one run.

    ``q_scale`` is fp32 ``[Hq, flat_scale_numel(B, Sq, q_block_size)]`` and
    ``k_scale`` is fp32 ``[Hkv, flat_scale_numel(B, Skv, k_block_size)]`` in
    the trtllm-gen flat layout, with the block sizes of the plan's
    :class:`SageAttentionConfig`. ``v_scale`` and ``v_mean`` are fp32
    ``[Hkv, D]``; ``v_mean`` is present exactly when the plan's
    :class:`SageAttentionConfig` has ``v_mean=True``. ``k_summary_scale`` carries the flat-layout scales of
    block-sparse proxy summaries and is not consumed by dense attention. Every
    scale must be positive and finite; the kernel does not check the values.
    """

    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    k_summary_scale: torch.Tensor | None = None
    v_mean: torch.Tensor | None = None


def flat_scale_numel(batch_size: int, seq_len: int, block_size: int) -> int:
    """Return the per-head slot count of the flat scale layout."""

    return ceil_div(batch_size * seq_len, block_size) + batch_size - 1


def is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""

    return value > 0 and value & (value - 1) == 0


def log2_block_size(block_size: int) -> int:
    """Return ``log2`` of a power-of-two scale block size.

    Raises ``ValueError`` for any other value, so it doubles as the
    power-of-two check of a block size.
    """

    if not is_power_of_two(block_size):
        raise ValueError(f"block size must be a power of two, got {block_size}")
    return block_size.bit_length() - 1


def sage_adapter_slots(values: Mapping[str, _T | None]) -> tuple[_T | None, ...]:
    """Arrange Sage values named by scale into the adapter slots.

    A slot whose name is absent or ``None`` stays ``None``: a recipe without
    a V mean leaves the ``v_mean`` slot empty and the adapter binds a null
    pointer for it.
    """

    return tuple(values.get(name) for name in SAGE_ADAPTER_SLOTS)


def flat_scale_slot(batch_idx, token_idx, seq_len, log2_block: int):
    """Return the per-head slot of one token in the flat scale layout.

    Sequence ``b`` starts at ``b * S // blk + b``; the extra ``b`` keeps the
    last block of one sequence and the first block of the next in distinct
    slots when the block size does not divide the sequence length. Block
    sizes are powers of two, so the divisions are shifts; the plain
    arithmetic serves host integers and device ``Int32`` values alike.
    """

    return ((batch_idx * seq_len) >> log2_block) + batch_idx + (token_idx >> log2_block)


def _validate_scale_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_shape: tuple[int, int],
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype float32, got {tensor.dtype}")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} data pointer must be 16-byte aligned")
    if tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


def sage_scale_shapes(
    config: SageAttentionConfig,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, tuple[int, int]]:
    """Return the shape of every scale tensor a plan consumes, by field name.

    Q/K scales are ``[heads, flat slots]`` in the flat layout of the recipe's
    block sizes, V scales and means are ``[kv heads, head dim]``; ``v_mean``
    appears only when the recipe has one.
    """

    shapes = {
        "q_scale": (
            num_qo_heads,
            flat_scale_numel(batch_size, seq_len_q, config.q_block_size),
        ),
        "k_scale": (
            num_kv_heads,
            flat_scale_numel(batch_size, seq_len_kv, config.k_block_size),
        ),
        "v_scale": (num_kv_heads, head_dim),
    }
    if config.v_mean:
        shapes["v_mean"] = (num_kv_heads, head_dim)
    return shapes


def validate_sage_params(
    params: SageAttentionParams,
    config: SageAttentionConfig,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> None:
    """Validate the scale tensors of one run against the planned recipe.

    Plan compilation validates the recipe; this validates what a run supplies:
    the tensors and their agreement with ``config.v_mean``.
    """

    if not isinstance(params, SageAttentionParams):
        raise TypeError("sage must be a SageAttentionParams instance")
    if (params.v_mean is None) == config.v_mean:
        raise ValueError(
            "v_mean is required by a plan configured with v_mean=True and "
            "rejected otherwise"
        )
    if params.k_summary_scale is not None:
        raise ValueError(
            "k_summary_scale is consumed only by block-sparse proxy routes"
        )
    shapes = sage_scale_shapes(
        config,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    for name, expected_shape in shapes.items():
        _validate_scale_tensor(
            getattr(params, name), name, expected_shape=expected_shape, device=device
        )


__all__ = [
    "SAGE_ADAPTER_SLOTS",
    "SAGE_K_BLOCK_SIZES",
    "SageAttentionConfig",
    "SageAttentionParams",
    "flat_scale_numel",
    "flat_scale_slot",
    "is_power_of_two",
    "log2_block_size",
    "sage_adapter_slots",
    "sage_scale_shapes",
    "validate_sage_params",
]
