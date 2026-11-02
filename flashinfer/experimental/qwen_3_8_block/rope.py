# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""RoPE table helpers: the POSITION-indexed ``cos_pos`` / ``sin_pos`` tables the decode step reads (formed once at load), and the
Qwen 3.5 package's per-token helper re-exported for the unfused arm."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..qwen_3_5_block.rope import qwen_3_5_block_rope_tables

#: the per-token ``cos`` / ``sin`` rows from ``positions`` + a ``cos_sin_cache`` (the unfused arm, ``rope_position_tables_only=False``)
qwen_3_8_block_rope_tables = qwen_3_5_block_rope_tables


def qwen_3_8_block_rope_position_tables(
    cos_sin_cache: torch.Tensor,
    *,
    num_positions: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """See :func:`flashinfer.qwen_3_8_block.qwen_3_8_block_rope_position_tables` (the public, decorated entry)."""
    if (
        not isinstance(cos_sin_cache, torch.Tensor)
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[1] % 2
        or cos_sin_cache.shape[0] < 1
    ):
        raise ValueError(
            "cos_sin_cache must be [max_position >= 1, rotary_dim] with the cos half then the sin half (rotary_dim even)"
        )
    rows = int(cos_sin_cache.shape[0])
    n = rows if num_positions is None else int(num_positions)
    if isinstance(num_positions, bool) or n < 1:
        raise ValueError(
            f"num_positions must be a positive int (the table rows: at least max_pages x page_size), got {num_positions!r}"
        )
    half = int(cos_sin_cache.shape[1]) // 2
    cache = cos_sin_cache if dtype is None else cos_sin_cache.to(dtype)
    if n <= rows:
        src = cache[:n]
    else:
        # positions past the cache's rows are never reached by a token (the block table's reach may exceed max_position_embeddings by
        # up to page_size - 1 rows): the last row is repeated, so the FORM rule ``rows >= max_pages x page_size`` holds without inventing angles
        src = torch.cat((cache, cache[-1:].expand(n - rows, cache.shape[1])), dim=0)
    c, s = src[:, :half], src[:, half:]
    cos = torch.cat((c, c), dim=-1)[None].contiguous()
    sin = torch.cat((s, s), dim=-1)[None].contiguous()
    return cos, sin
