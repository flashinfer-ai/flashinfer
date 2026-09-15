# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Offline stage winners exposed as at most four compound MoE tactics.

Tables contain artifact identities, not hardware timings or compiled objects.
An unmeasured token count uses the next measured count (the largest above the
table's range). Geometry, activation and top-k must match exactly.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

_FILENAME = "moe_shortlists.json"


@functools.lru_cache(maxsize=8)
def _read(roots: tuple[Path, ...]) -> dict:
    tables: dict[
        tuple[str, str, int, int, int, int],
        dict[int, tuple[tuple[str, ...], tuple[str, ...]]],
    ] = {}
    for root in roots:
        path = root / _FILENAME
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload.get("version") != 1:
            raise ValueError("unsupported Frost MoE shortlist version")
        for entry in payload["entries"]:
            key = (
                entry["arch"],
                entry["activation"],
                entry["experts"],
                entry["hidden"],
                entry["intermediate"],
                entry["top_k"],
            )
            tokens = entry["tokens"]
            if not isinstance(tokens, int) or tokens <= 0:
                raise ValueError("Frost shortlist tokens must be positive")
            for stage in ("fc1", "fc2"):
                ids = entry[stage]
                if (
                    not isinstance(ids, list)
                    or not 1 <= len(ids) <= 2
                    or any(not isinstance(identity, str) for identity in ids)
                    or len(set(ids)) != len(ids)
                ):
                    raise ValueError(
                        "Frost shortlists require one or two distinct stage IDs"
                    )
            profiles = tables.setdefault(key, {})
            if tokens in profiles:
                raise ValueError("duplicate Frost MoE shortlist profile")
            profiles[tokens] = (tuple(entry["fc1"]), tuple(entry["fc2"]))
    return tables


def select(
    roots, arch, activation, tokens, hidden, intermediate, experts, topk, first, second
):
    """Filter matching kernels before either compilation or plan construction.

    Unlisted geometries retain the explicit legacy artifact path. Automatic
    admission remains independently controlled by the backend support policy.
    """
    profiles = _read(tuple(roots)).get(
        (arch, activation, experts, hidden, intermediate, topk)
    )
    if profiles is None:
        return first, second
    bucket = min((n for n in profiles if n >= tokens), default=max(profiles))
    choices = []
    for pool, identities in zip((first, second), profiles[bucket], strict=True):
        available = {kernel.artifact_id: kernel for kernel in pool}
        if not all(identity in available for identity in identities):
            raise ValueError(
                "Frost shortlist references missing or incompatible kernels"
            )
        choices.append(tuple(available[identity] for identity in identities))
    return tuple(choices)
