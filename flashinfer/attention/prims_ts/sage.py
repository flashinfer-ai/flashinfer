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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flashinfer.utils import ceil_div

SAGE_K_BLOCK_SIZES = (16, 32, 64, 128, 256)
SAGE_QK_DTYPES = (torch.float8_e4m3fn,)
SAGE_V_DTYPE = torch.float8_e4m3fn
SAGE_OUTPUT_DTYPES = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class SageAttentionParams:
    """Per-block Q/K scales and per-channel V scales for one attention call.

    ``q_scale`` is fp32 ``[Hq, flat_scale_numel(B, Sq, q_block_size)]`` and
    ``k_scale`` is fp32 ``[Hkv, flat_scale_numel(B, Skv, k_block_size)]`` in
    the trtllm-gen flat layout. ``v_scale`` and the optional ``v_mean`` are
    fp32 ``[Hkv, D]``; ``v_mean`` is added back to the normalized output when
    V was quantized around its per-channel mean. ``k_summary_scale`` is fp32
    ``[Hkv, flat_scale_numel(B, num_kv_blocks, k_block_size)]``: the flat-layout
    scales of block-sparse proxy K summaries, quantized as one more K sequence
    of ``num_kv_blocks`` tokens. It is required by proxy routes and rejected
    otherwise. Every scale must be positive and finite; the kernel does not
    check the values.
    """

    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    k_summary_scale: torch.Tensor | None = None
    v_mean: torch.Tensor | None = None
    q_block_size: int = 1
    k_block_size: int = 16


def sage_v_mean_launch_tensor(params: SageAttentionParams) -> torch.Tensor:
    """Return the tensor bound to the V mean slot of a Sage launch.

    The adapter signature always carries the slot; without ``v_mean`` the
    kernel never dereferences it, so ``v_scale`` stands in.
    """

    return params.v_scale if params.v_mean is None else params.v_mean


def flat_scale_numel(batch_size: int, seq_len: int, block_size: int) -> int:
    """Return the per-head slot count of the flat scale layout."""

    return ceil_div(batch_size * seq_len, block_size) + batch_size - 1


def log2_block_size(block_size: int) -> int:
    """Return ``log2`` of a power-of-two scale block size.

    Raises ``ValueError`` for any other value, so it doubles as the
    power-of-two check of a block size.
    """

    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block size must be a power of two, got {block_size}")
    return block_size.bit_length() - 1


def flat_scale_slot(batch_idx, token_idx, seq_len, log2_block: int):
    """Return the per-head slot of one token in the flat scale layout.

    Sequence ``b`` starts at ``b * S // blk + b``; the extra ``b`` keeps the
    last block of one sequence and the first block of the next in distinct
    slots when the block size does not divide the sequence length. Block
    sizes are powers of two, so the divisions are shifts; the plain
    arithmetic serves host integers and device ``Int32`` values alike.
    """

    return ((batch_idx * seq_len) >> log2_block) + batch_idx + (token_idx >> log2_block)


def _validate_block_sizes(params: SageAttentionParams, tile_size_q: int) -> None:
    for name, value in (
        ("q_block_size", params.q_block_size),
        ("k_block_size", params.k_block_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a Python integer")
    if params.k_block_size not in SAGE_K_BLOCK_SIZES:
        raise ValueError(
            f"k_block_size must be one of {SAGE_K_BLOCK_SIZES}, "
            f"got {params.k_block_size}"
        )
    message = (
        "q_block_size must be a power of two no larger than the Q tile "
        f"({tile_size_q}), got {params.q_block_size}"
    )
    try:
        log2_block_size(params.q_block_size)
    except ValueError:
        raise ValueError(message) from None
    if params.q_block_size > tile_size_q:
        raise ValueError(message)


def _validate_dtypes(
    q_dtype: torch.dtype, kv_dtype: torch.dtype, out_dtype: torch.dtype
) -> None:
    if q_dtype == torch.int8:
        raise NotImplementedError("INT8 Q/K Sage attention is not supported yet")
    if q_dtype != kv_dtype or q_dtype not in SAGE_QK_DTYPES:
        raise ValueError(
            "Sage attention requires Q and K in one dtype from "
            f"{SAGE_QK_DTYPES}, got Q {q_dtype} and K {kv_dtype}"
        )
    if kv_dtype != SAGE_V_DTYPE:
        raise ValueError(f"Sage attention requires V in {SAGE_V_DTYPE}, got {kv_dtype}")
    if out_dtype not in SAGE_OUTPUT_DTYPES:
        raise ValueError(
            f"Sage attention requires an output dtype from {SAGE_OUTPUT_DTYPES}, "
            f"got {out_dtype}"
        )


def _validate_scale_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_shape: tuple[int, int],
    device: torch.device | None,
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
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


def validate_sage_params(
    params: SageAttentionParams,
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tile_size_q: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    device: torch.device | None = None,
    summary_seq_len: int | None = None,
) -> None:
    """Validate the block sizes, dtypes and scale tensors of one call.

    ``summary_seq_len`` is the number of KV blocks when block-sparse proxy
    routes are enabled; ``k_summary_scale`` must then cover that summary
    sequence in the flat layout and must be absent otherwise.
    """

    if not isinstance(params, SageAttentionParams):
        raise TypeError("sage must be a SageAttentionParams instance")
    _validate_block_sizes(params, tile_size_q)
    _validate_dtypes(q_dtype, kv_dtype, out_dtype)
    if summary_seq_len is None:
        if params.k_summary_scale is not None:
            raise ValueError(
                "k_summary_scale is consumed only by block-sparse proxy routes"
            )
    elif params.k_summary_scale is None:
        raise ValueError("k_summary_scale is required by block-sparse proxy routes")
    else:
        _validate_scale_tensor(
            params.k_summary_scale,
            "k_summary_scale",
            expected_shape=(
                num_kv_heads,
                flat_scale_numel(batch_size, summary_seq_len, params.k_block_size),
            ),
            device=device,
        )
    _validate_scale_tensor(
        params.q_scale,
        "q_scale",
        expected_shape=(
            num_qo_heads,
            flat_scale_numel(batch_size, seq_len_q, params.q_block_size),
        ),
        device=device,
    )
    _validate_scale_tensor(
        params.k_scale,
        "k_scale",
        expected_shape=(
            num_kv_heads,
            flat_scale_numel(batch_size, seq_len_kv, params.k_block_size),
        ),
        device=device,
    )
    _validate_scale_tensor(
        params.v_scale,
        "v_scale",
        expected_shape=(num_kv_heads, head_dim),
        device=device,
    )
    if params.v_mean is not None:
        _validate_scale_tensor(
            params.v_mean,
            "v_mean",
            expected_shape=(num_kv_heads, head_dim),
            device=device,
        )


__all__ = [
    "SAGE_K_BLOCK_SIZES",
    "SAGE_OUTPUT_DTYPES",
    "SAGE_QK_DTYPES",
    "SAGE_V_DTYPE",
    "SageAttentionParams",
    "flat_scale_numel",
    "flat_scale_slot",
    "log2_block_size",
    "sage_v_mean_launch_tensor",
    "validate_sage_params",
]
