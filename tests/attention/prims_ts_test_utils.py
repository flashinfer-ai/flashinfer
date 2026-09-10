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

"""Input builders shared by the PrimTS decode, block-sparse and Sage tests.

Block-sparse patterns are nested tuples ``patterns[batch][kv_head][q_row]``
of the exact KV block ids selected by one BSR row.
"""

from __future__ import annotations

import math

import pytest
import torch

from flashinfer.attention.prims_ts.sage import SageAttentionParams, flat_scale_numel

HEAD_DIM = 128
FP8 = torch.float8_e4m3fn
Patterns = tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]

# The block-sparse and Sage kernels run on SM100 and SM103; the dense decode
# tests keep their stricter SM100-only signoff marker.
REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS block-sparse and Sage attention require SM100 or SM103",
)


def dense_stream_columns(kv_tile_size: int, device: torch.device) -> list[torch.Tensor]:
    """Return the tile columns of each online-softmax stream of a dense KV tile.

    KV256 splits a tile into two spatial halves of two K64 atoms each; KV128
    runs one stream over the whole tile.
    """

    tile_columns = torch.arange(kv_tile_size, device=device)
    if kv_tile_size == 256:
        return [
            torch.cat((tile_columns[0:64], tile_columns[128:192])),
            torch.cat((tile_columns[64:128], tile_columns[192:256])),
        ]
    return [tile_columns]


def make_sage_decode_config(
    *,
    tile_size_q: int,
    tile_size_kv: int,
    qkv_dtype=None,
    v_dtype=None,
    o_dtype=None,
    sage_args: dict[str, object] | None = None,
    mask_type: str = "dense",
    qkv_layout: str = "contiguousKv",
    num_tokens_per_page: int = 32,
    split_kv_mode: str = "disabled",
    splits_kv: int = 1,
):
    """Build a dense contiguous Sage profile the way the dense wrapper does.

    ``qkv_dtype`` defaults to E4M3 and ``o_dtype`` to BF16; ``sage_args``
    override or extend the profile arguments.
    """

    from cutlass import BFloat16, Float8E4M3FN

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        make_decode_config,
    )

    args: dict[str, object] = {
        "use_keeps_mma_ab": True,
        "tile_size_q": tile_size_q,
        "tile_size_kv": tile_size_kv,
        "groups_tokens_heads_q": True,
        "sage_k_block_size": 16,
        "sage_q_block_size": 1,
    }
    if sage_args is not None:
        args.update(sage_args)
    heads_q_per_kv = 1 if tile_size_q == 64 else 8
    return make_decode_config(
        headdim=HEAD_DIM,
        args=args,
        seq_len_q=64 if tile_size_q == 64 else 16,
        seq_len_kv=1000,
        batch_size=2,
        num_heads_q=8,
        num_heads_kv=8 // heads_q_per_kv,
        qkv_dtype=Float8E4M3FN if qkv_dtype is None else qkv_dtype,
        v_dtype=v_dtype,
        o_dtype=BFloat16 if o_dtype is None else o_dtype,
        qkv_layout=qkv_layout,
        num_tokens_per_page=num_tokens_per_page,
        split_kv_mode=split_kv_mode,
        splits_kv=splits_kv,
        mask_type=mask_type,
        auto_tuner=False,
    )


def make_sage_params(
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int = HEAD_DIM,
    q_block_size: int = 1,
    k_block_size: int = 16,
    with_mean: bool = False,
    with_summary_scale: bool = False,
    kv_block_size: int = 64,
    device: torch.device | str = "cpu",
) -> SageAttentionParams:
    """Return random positive scales of one geometry in the flat layout.

    ``with_summary_scale`` adds the K summary scales that proxy routes
    require, covering ``ceil(seq_len_kv / kv_block_size)`` summaries.
    """

    def flat_scales(num_heads: int, seq_len: int, block_size: int) -> torch.Tensor:
        return torch.rand(
            (num_heads, flat_scale_numel(batch_size, seq_len, block_size)),
            device=device,
        )

    num_kv_blocks = math.ceil(seq_len_kv / kv_block_size)
    return SageAttentionParams(
        q_scale=flat_scales(num_qo_heads, seq_len_q, q_block_size),
        k_scale=flat_scales(num_kv_heads, seq_len_kv, k_block_size),
        v_scale=torch.rand((num_kv_heads, head_dim), device=device),
        k_summary_scale=(
            flat_scales(num_kv_heads, num_kv_blocks, k_block_size)
            if with_summary_scale
            else None
        ),
        v_mean=(
            torch.randn((num_kv_heads, head_dim), device=device) if with_mean else None
        ),
        q_block_size=q_block_size,
        k_block_size=k_block_size,
    )


def widest_bsr_row(patterns: Patterns) -> int:
    """Return the widest BSR row of a pattern set, the plan's capacity bound."""

    return max(len(row) for batch in patterns for head in batch for row in head)


def make_bsr(
    patterns: Patterns, device: torch.device | str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact Int32 ``(block_indptr, block_indices)`` of a pattern set."""

    flat_indices: list[int] = []
    pointer_batches: list[list[list[int]]] = []
    for batch in patterns:
        pointer_heads: list[list[int]] = []
        for head in batch:
            pointers = [len(flat_indices)]
            for row in head:
                flat_indices.extend(row)
                pointers.append(len(flat_indices))
            pointer_heads.append(pointers)
        pointer_batches.append(pointer_heads)
    return (
        torch.tensor(pointer_batches, device=device, dtype=torch.int32),
        torch.tensor(flat_indices, device=device, dtype=torch.int32),
    )


def make_exact_block_bits(
    patterns: Patterns,
    num_kv_blocks: int,
    device: torch.device | str = "cuda",
    *,
    set_padding_bits: bool = False,
) -> torch.Tensor:
    """Return the packed UInt32 exact-block bitmap of a pattern set.

    ``set_padding_bits`` raises the out-of-range bits of the final word, which
    the route preparation must ignore.
    """

    words_per_row = math.ceil(num_kv_blocks / 32)
    packed_batches: list[list[list[list[int]]]] = []
    for batch in patterns:
        packed_heads: list[list[list[int]]] = []
        for head in batch:
            packed_rows: list[list[int]] = []
            for exact_blocks in head:
                words = [0] * words_per_row
                for block_idx in exact_blocks:
                    words[block_idx // 32] |= 1 << (block_idx % 32)
                if set_padding_bits and num_kv_blocks % 32:
                    words[-1] |= (-1 << (num_kv_blocks % 32)) & 0xFFFFFFFF
                packed_rows.append(words)
            packed_heads.append(packed_rows)
        packed_batches.append(packed_heads)
    return torch.tensor(packed_batches, device=device, dtype=torch.uint32)


def token_mask_valid_sets(
    batch_size: int, seq_len_kv: int
) -> tuple[frozenset[int], ...]:
    """Return per-batch valid token sets that drop two of every seven tokens."""

    return tuple(
        frozenset(
            token_idx
            for token_idx in range(seq_len_kv)
            if (token_idx + batch_idx) % 7 not in (0, 3)
        )
        for batch_idx in range(batch_size)
    )


def pack_token_mask(
    seq_len_kv: int,
    valid_by_batch: tuple[frozenset[int], ...],
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Return the UInt32 ``[B, ceil(seq_len_kv / 32)]`` token validity bitmap."""

    packed_by_batch: list[list[int]] = []
    for valid_tokens in valid_by_batch:
        words = [0] * math.ceil(seq_len_kv / 32)
        for token_idx in valid_tokens:
            words[token_idx // 32] |= 1 << (token_idx % 32)
        packed_by_batch.append(words)
    return torch.tensor(packed_by_batch, device=device, dtype=torch.uint32)


def block_mean(
    x: torch.Tensor, kv_block_size: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Return the per-block means of the structural tokens of ``[B, S, H, D]``.

    The result is fp32 unless ``dtype`` is given.
    """

    blocks = [
        x[:, begin : begin + kv_block_size].float().mean(dim=1)
        for begin in range(0, x.shape[1], kv_block_size)
    ]
    means = torch.stack(blocks, dim=1)
    return means if dtype is None else means.to(dtype)


def heavy_tailed(
    shape: tuple[int, ...], *, device: torch.device, magnitude: float = 1.0
) -> torch.Tensor:
    """Return BF16 normal values with log-uniform per-element spread."""

    x = torch.randn(shape, device=device) * magnitude
    return (x * torch.exp((torch.rand(shape, device=device) - 0.5) * 3.2)).to(
        torch.bfloat16
    )
