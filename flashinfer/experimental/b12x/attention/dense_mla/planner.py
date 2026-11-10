"""Capture-static launch planning for dense Kimi-K3 MLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch

Mode = Literal["decode", "extend", "verify"]

_CANDIDATES_PER_CHUNK = 64
_MAX_CEIL_WAVES = 3


@dataclass(frozen=True, kw_only=True)
class Budget:
    """Optional caller capacity clamps; B12X still chooses launch policy."""

    max_splits: int | None = None
    max_partial_rows: int | None = None

    def __post_init__(self) -> None:
        if self.max_splits is not None and int(self.max_splits) <= 0:
            raise ValueError("max_splits must be positive")
        if self.max_partial_rows is not None and int(self.max_partial_rows) < 0:
            raise ValueError("max_partial_rows must be non-negative")


def _host_i32_values(values: torch.Tensor | Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError("sequence metadata must be rank-1")
        if values.dtype != torch.int32:
            raise TypeError("sequence metadata must be torch.int32")
        values = values.detach().cpu().tolist()
    return tuple(int(value) for value in values)


def infer_dense_mla_mode(
    cu_seqlens_q: torch.Tensor | Sequence[int],
) -> Literal["decode", "extend"]:
    """Infer decode versus extend from host-visible cumulative Q lengths."""
    cu = _host_i32_values(cu_seqlens_q)
    if len(cu) < 2 or cu[0] != 0:
        raise ValueError("cu_seqlens_q must start at zero and contain a request")
    q_lens = tuple(right - left for left, right in zip(cu, cu[1:], strict=True))
    if any(length <= 0 for length in q_lens):
        raise ValueError("every request must contain at least one query row")
    return "decode" if all(length == 1 for length in q_lens) else "extend"


def _wave_balanced_splits(
    *,
    num_chunks: int,
    independent_tiles: int,
    sm_count: int,
) -> int:
    """Choose contiguous chunk ranges with at most three active CTA waves.

    This is a structural occupancy calculation, not a guessed dense-vs-MHA
    crossover table.  It searches the chunks-per-CTA choices and minimizes the
    last-wave tail; ties choose fewer CTAs.
    """
    num_chunks = max(1, int(num_chunks))
    independent_tiles = max(1, int(independent_tiles))
    sm_count = max(1, int(sm_count))
    best_chunks_per_cta = num_chunks
    best_gap = 2.0
    for chunks_per_cta in range(1, num_chunks + 1):
        splits = (num_chunks + chunks_per_cta - 1) // chunks_per_cta
        active = independent_tiles * splits
        ceil_waves = (active + sm_count - 1) // sm_count
        if ceil_waves > _MAX_CEIL_WAVES:
            continue
        gap = ceil_waves - active / sm_count
        if gap < best_gap - 1e-6 or (
            gap < best_gap + 1e-6 and chunks_per_cta > best_chunks_per_cta
        ):
            best_gap = gap
            best_chunks_per_cta = chunks_per_cta
    return (num_chunks + best_chunks_per_cta - 1) // best_chunks_per_cta


def choose_num_splits(
    *,
    max_cache_tokens: int,
    max_total_q: int,
    num_q_heads: int,
    query_tile: int,
    sm_count: int,
    budget: Budget | None,
) -> int:
    """Return the capture-static split count for the planned capacity."""
    num_chunks = (
        max(int(max_cache_tokens), 1) + _CANDIDATES_PER_CHUNK - 1
    ) // _CANDIDATES_PER_CHUNK
    query_tiles = (max(int(max_total_q), 1) + int(query_tile) - 1) // int(query_tile)
    head_tiles = max(1, (int(num_q_heads) + 7) // 8)
    splits = _wave_balanced_splits(
        num_chunks=num_chunks,
        independent_tiles=query_tiles * head_tiles,
        sm_count=sm_count,
    )
    if budget is not None and budget.max_splits is not None:
        splits = min(splits, int(budget.max_splits))
    if budget is not None and budget.max_partial_rows is not None:
        split_capacity = int(budget.max_partial_rows) // max(
            int(max_total_q),
            1,
        )
        # A one-split launch writes the final output directly and consumes no
        # partial rows, so even a zero-row merge budget remains executable.
        splits = 1 if split_capacity < 2 else min(splits, split_capacity)
    return max(1, min(splits, num_chunks))


__all__ = [
    "Budget",
    "Mode",
    "choose_num_splits",
    "infer_dense_mla_mode",
]
