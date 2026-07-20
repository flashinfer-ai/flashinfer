# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/paged/tuning/registry.py @ 7dc97259 (2026-03-31) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

Ladder = tuple[tuple[int, int], ...]

DECODE_GRAPH_POLICY: Dict[tuple[str, str], Dict[int, "DecodeGraphPolicy"]] = {}


@dataclass(frozen=True)
class DecodeGraphPolicy:
    graph_ctas_per_sm: int
    chunk_ladder: Ladder
    capture_fixed_split_pages: int | None = None
    capture_page_count: int | None = None
    page_size: int | None = None


def _validate_ladder(
    *,
    ladder: Ladder,
    value_name: str,
) -> None:
    if not ladder:
        raise ValueError("ladder must be non-empty")
    previous_end = 0
    for end_page, value in ladder:
        if end_page <= previous_end:
            raise ValueError("ladder end_page values must be strictly increasing")
        if value <= 0:
            raise ValueError(f"{value_name} must be positive")
        previous_end = end_page


def normalize_kv_dtype_key(kv_dtype: str) -> str:
    return {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "fp8": "fp8",
        "fp8_e4m3fn": "fp8",
        "float8_e4m3fn": "fp8",
    }.get(kv_dtype, kv_dtype)


def register_decode_graph_policy(
    *,
    kv_dtype: str,
    regime: str,
    batch: int,
    graph_ctas_per_sm: int,
    chunk_ladder: Ladder,
    capture_fixed_split_pages: int | None = None,
    capture_page_count: int | None = None,
    page_size: int | None = None,
) -> None:
    if batch <= 0:
        raise ValueError("batch must be positive")
    if graph_ctas_per_sm <= 0:
        raise ValueError("graph_ctas_per_sm must be positive")
    if capture_fixed_split_pages is not None and capture_fixed_split_pages <= 0:
        raise ValueError("capture_fixed_split_pages must be positive when provided")
    if capture_page_count is not None and capture_page_count <= 0:
        raise ValueError("capture_page_count must be positive when provided")
    if page_size is not None and page_size <= 0:
        raise ValueError("page_size must be positive when provided")
    _validate_ladder(ladder=chunk_ladder, value_name="chunk_pages")
    DECODE_GRAPH_POLICY.setdefault((normalize_kv_dtype_key(kv_dtype), regime), {})[
        batch
    ] = DecodeGraphPolicy(
        graph_ctas_per_sm=int(graph_ctas_per_sm),
        chunk_ladder=tuple(chunk_ladder),
        capture_fixed_split_pages=(
            None
            if capture_fixed_split_pages is None
            else int(capture_fixed_split_pages)
        ),
        capture_page_count=None
        if capture_page_count is None
        else int(capture_page_count),
        page_size=None if page_size is None else int(page_size),
    )


def get_decode_graph_policy(
    *,
    kv_dtype: str,
    regime: str,
    batch: int,
) -> DecodeGraphPolicy:
    family = DECODE_GRAPH_POLICY[(normalize_kv_dtype_key(kv_dtype), regime)]
    available_batches = sorted(family)
    snapped_batch = next(
        (candidate for candidate in available_batches if candidate >= batch),
        available_batches[-1],
    )
    return family[snapped_batch]


def lookup_decode_graph_chunk_pages(
    *,
    kv_dtype: str,
    regime: str,
    batch: int,
    page_count: int,
) -> int:
    if page_count <= 0:
        raise ValueError("page_count must be positive")
    ladder = get_decode_graph_policy(
        kv_dtype=kv_dtype, regime=regime, batch=batch
    ).chunk_ladder
    for end_page, chunk_pages in ladder:
        if page_count <= end_page:
            return chunk_pages
    return ladder[-1][1]
