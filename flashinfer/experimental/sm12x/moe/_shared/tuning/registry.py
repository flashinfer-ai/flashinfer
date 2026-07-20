# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/tuning/registry.py @ 3f7ff225 (2026-06-30) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

Ladder = tuple[tuple[int, int], ...]

MAX_ACTIVE_CLUSTERS_POLICY: Dict[tuple[str, str], "MaxActiveClustersPolicy"] = {}


@dataclass(frozen=True)
class MaxActiveClustersPolicy:
    ladder: Ladder


def _validate_ladder(*, ladder: Ladder) -> None:
    if not ladder:
        raise ValueError("ladder must be non-empty")
    previous_end = 0
    for end_routed_rows, max_active_clusters in ladder:
        if end_routed_rows <= previous_end:
            raise ValueError(
                "ladder end_routed_rows values must be strictly increasing"
            )
        if max_active_clusters <= 0:
            raise ValueError("ladder max_active_clusters values must be positive")
        previous_end = end_routed_rows


def register_max_active_clusters_policy(
    *,
    regime: str,
    backend: str,
    ladder: Ladder,
) -> None:
    if not regime:
        raise ValueError("regime must be non-empty")
    if backend not in {"micro", "dynamic", "dynamic_w4a8_decode"}:
        raise ValueError(f"unsupported backend {backend!r}")
    _validate_ladder(ladder=ladder)
    MAX_ACTIVE_CLUSTERS_POLICY[(str(regime), str(backend))] = MaxActiveClustersPolicy(
        ladder=tuple(
            (int(end_routed_rows), int(max_active_clusters))
            for end_routed_rows, max_active_clusters in ladder
        )
    )


def get_max_active_clusters_policy(
    *,
    regime: str,
    backend: str,
) -> MaxActiveClustersPolicy:
    return MAX_ACTIVE_CLUSTERS_POLICY[(str(regime), str(backend))]


def lookup_max_active_clusters(
    *,
    regime: str,
    backend: str,
    routed_rows: int,
) -> int | None:
    if routed_rows <= 0:
        raise ValueError("routed_rows must be positive")
    policy = MAX_ACTIVE_CLUSTERS_POLICY.get((str(regime), str(backend)))
    if policy is None:
        return None
    for end_routed_rows, max_active_clusters in policy.ladder:
        if routed_rows <= end_routed_rows:
            return int(max_active_clusters)
    return None
