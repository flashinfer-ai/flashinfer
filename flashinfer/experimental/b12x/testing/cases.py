"""Workload metadata shared by numerical fixtures and native benchmarks."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from b12x.preparation import FrozenMapping


@dataclass(frozen=True, kw_only=True)
class WorkloadCase:
    """One concrete fixture scenario, independent of any search implementation."""

    case_id: str
    group_id: str
    query: FrozenMapping
    scenario: str = "default"
    metadata: FrozenMapping = FrozenMapping()

    @classmethod
    def create(
        cls, *, group_id: str, query: Mapping[str, object], scenario: str = "default",
        metadata: Mapping[str, object] | None = None, label: str | None = None,
    ) -> WorkloadCase:
        identity = {
            "group_id": group_id, "query": dict(query), "scenario": scenario,
            "metadata": dict(metadata or {}),
        }
        # Preserve the existing fixture IDs while retiring the offline sweep.
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        suffix = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
        return cls(
            case_id=f"{label or group_id}-{suffix}", group_id=group_id,
            query=FrozenMapping(query), scenario=scenario, metadata=FrozenMapping(metadata),
        )

    def __post_init__(self) -> None:
        if not self.case_id or not self.group_id or not self.scenario:
            raise ValueError("workload case identifiers must be non-empty")
        if not self.query:
            raise ValueError("workload cases require a non-empty query")


__all__ = ["WorkloadCase"]
