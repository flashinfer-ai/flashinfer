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

HEAD_DIM = 128
FP8 = torch.float8_e4m3fn
Patterns = tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]

REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="PrimTS block-sparse attention requires SM100 or SM103",
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
