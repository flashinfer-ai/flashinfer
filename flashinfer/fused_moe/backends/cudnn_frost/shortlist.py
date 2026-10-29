# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Offline stage winners exposed as at most four compound MoE tactics.

Tables contain artifact identities, not hardware timings or compiled objects.
An unmeasured token count uses the next measured count (the largest above the
table's range). Geometry, activation and top-k must match exactly.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from .manifest import MANIFEST_FILENAME, read_manifest

_FILENAME = "moe_shortlists.json"
_TableKey = tuple[str, str, int, int, int, int]
_ProfileChoices = tuple[tuple[str, ...], tuple[str, ...]]
_ProfileMap = Mapping[int, _ProfileChoices]


@functools.lru_cache(maxsize=8)
def _read(roots: tuple[Path, ...]) -> dict:
    manifests = [root / MANIFEST_FILENAME for root in roots]
    manifests = [path for path in manifests if path.is_file()]
    catalog_ids = (
        {
            record["id"]
            for path in manifests
            for record in read_manifest(path)["kernels"]
        }
        if manifests
        else None
    )
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
                if catalog_ids is not None and any(
                    identity not in catalog_ids for identity in ids
                ):
                    raise ValueError("Frost shortlist references missing kernels")
            profiles = tables.setdefault(key, {})
            if tokens in profiles:
                raise ValueError("duplicate Frost MoE shortlist profile")
            profiles[tokens] = (tuple(entry["fc1"]), tuple(entry["fc2"]))
    return tables


def capture_profiles(roots: tuple[Path, ...]) -> Mapping[_TableKey, _ProfileMap]:
    """Detach an immutable shortlist snapshot from the shared reader cache."""
    tables: dict[_TableKey, _ProfileMap] = {}
    for key, profiles in _read(tuple(roots)).items():
        tables[key] = MappingProxyType(
            {
                tokens: (tuple(first), tuple(second))
                for tokens, (first, second) in profiles.items()
            }
        )
    return MappingProxyType(tables)


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
    return select_profiles(profiles, tokens, first, second)


def select_profiles(profiles: _ProfileMap | None, tokens: int, first, second):
    """Select stage identities using only the supplied shortlist snapshot."""
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
