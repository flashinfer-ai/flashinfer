from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SCHEMA_VERSION = 3
ALGORITHM_VERSION = "lpt-ms-v3"
DEFAULT_CHECKPOINT_SECONDS = 1_000_000
DEFAULT_TARGET_UNIT_SECONDS = 1_000_000


def source_file_for_nodeid(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def base_function_for_nodeid(nodeid: str) -> str:
    parts = nodeid.split("::")
    if len(parts) == 1:
        return nodeid
    parts[-1] = parts[-1].split("[", 1)[0]
    return "::".join(parts)


@dataclass(frozen=True)
class CollectedNode:
    nodeid: str
    source_file: str
    base_function: str
    order: int
    shard_group: str | None = None
    solo: bool = False

    @classmethod
    def from_nodeid(
        cls,
        nodeid: str,
        order: int,
        shard_group: str | None = None,
        *,
        solo: bool = False,
    ) -> "CollectedNode":
        return cls(
            nodeid=nodeid,
            source_file=source_file_for_nodeid(nodeid),
            base_function=base_function_for_nodeid(nodeid),
            order=order,
            shard_group=shard_group,
            solo=solo,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodeid": self.nodeid,
            "source_file": self.source_file,
            "base_function": self.base_function,
            "order": self.order,
            "shard_group": self.shard_group,
            "solo": self.solo,
        }


@dataclass(frozen=True)
class PlanningOptions:
    profile: str
    checkpoint_seconds: int = DEFAULT_CHECKPOINT_SECONDS
    target_unit_seconds: int = DEFAULT_TARGET_UNIT_SECONDS
    unknown_case_seconds: int = 5
    shard_count: int = 1

    def __post_init__(self) -> None:
        if self.checkpoint_seconds <= 0:
            raise ValueError("checkpoint_seconds must be positive")
        if self.target_unit_seconds <= 0:
            raise ValueError("target_unit_seconds must be positive")
        if self.unknown_case_seconds <= 0:
            raise ValueError("unknown_case_seconds must be positive")
        if self.shard_count <= 0:
            raise ValueError("shard_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "checkpoint_seconds": self.checkpoint_seconds,
            "target_unit_seconds": self.target_unit_seconds,
            "unknown_case_seconds": self.unknown_case_seconds,
            "shard_count": self.shard_count,
        }


@dataclass(frozen=True)
class Batch:
    id: str
    source_file: str
    nodeids: tuple[str, ...]
    estimated_ms: int
    overhead_ms: int
    oversized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_file": self.source_file,
            "nodeids": list(self.nodeids),
            "estimated_ms": self.estimated_ms,
            "overhead_ms": self.overhead_ms,
            "oversized": self.oversized,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Batch":
        return cls(
            id=value["id"],
            source_file=value["source_file"],
            nodeids=tuple(value["nodeids"]),
            estimated_ms=int(value["estimated_ms"]),
            overhead_ms=int(value["overhead_ms"]),
            oversized=bool(value["oversized"]),
        )


@dataclass(frozen=True)
class Unit:
    id: str
    batches: tuple[Batch, ...]
    estimated_ms: int
    oversized: bool
    shard_index: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "batches": [batch.to_dict() for batch in self.batches],
            "estimated_ms": self.estimated_ms,
            "oversized": self.oversized,
            "shard_index": self.shard_index,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Unit":
        return cls(
            id=value["id"],
            batches=tuple(Batch.from_dict(batch) for batch in value["batches"]),
            estimated_ms=int(value["estimated_ms"]),
            oversized=bool(value["oversized"]),
            shard_index=int(value["shard_index"]),
        )


@dataclass(frozen=True)
class Plan:
    options: PlanningOptions
    nodes: tuple[CollectedNode, ...]
    units: tuple[Unit, ...]
    fallback_counts: dict[str, int] = field(default_factory=dict)
    fallback_sources: dict[str, str] = field(default_factory=dict)

    def fallback_index_groups(self) -> dict[str, list[int]]:
        nodeids = [node.nodeid for node in self.nodes]
        node_indexes = {nodeid: index for index, nodeid in enumerate(nodeids)}
        if len(node_indexes) != len(nodeids):
            raise ValueError("plan contains duplicate node IDs")
        fallback_groups: dict[str, list[int]] = {}
        for nodeid, source in self.fallback_sources.items():
            fallback_groups.setdefault(source, []).append(node_indexes[nodeid])
        return {
            source: sorted(indexes)
            for source, indexes in sorted(fallback_groups.items())
        }

    def to_dict(self) -> dict[str, Any]:
        nodeids = [node.nodeid for node in self.nodes]
        node_indexes = {nodeid: index for index, nodeid in enumerate(nodeids)}
        if len(node_indexes) != len(nodeids):
            raise ValueError("plan contains duplicate node IDs")
        source_file_groups: dict[str, list[int]] = {}
        for index, node in enumerate(self.nodes):
            source_file_groups.setdefault(node.source_file, []).append(index)
        shard_groups = {
            str(index): node.shard_group
            for index, node in enumerate(self.nodes)
            if node.shard_group is not None
        }
        solo_sources = sorted(
            {node.source_file for node in self.nodes if node.solo},
            key=lambda value: value.encode("utf-8"),
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "options": self.options.to_dict(),
            "nodeids": nodeids,
            "source_files": [
                {"path": path, "node_indexes": indexes}
                for path, indexes in sorted(source_file_groups.items())
            ],
            "shard_groups": shard_groups,
            "solo_sources": solo_sources,
            "units": [
                {
                    "id": unit.id,
                    "batches": [
                        {
                            "id": batch.id,
                            "node_indexes": [
                                node_indexes[nodeid] for nodeid in batch.nodeids
                            ],
                            "estimated_ms": batch.estimated_ms,
                            "overhead_ms": batch.overhead_ms,
                            "oversized": batch.oversized,
                        }
                        for batch in unit.batches
                    ],
                    "estimated_ms": unit.estimated_ms,
                    "oversized": unit.oversized,
                    "shard_index": unit.shard_index,
                }
                for unit in self.units
            ],
            "fallback_counts": dict(sorted(self.fallback_counts.items())),
            "fallbacks": [
                {
                    "source": source,
                    "node_indexes": indexes,
                }
                for source, indexes in self.fallback_index_groups().items()
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Plan":
        option_value = value["options"]
        options = PlanningOptions(
            profile=option_value["profile"],
            checkpoint_seconds=int(option_value["checkpoint_seconds"]),
            target_unit_seconds=int(option_value["target_unit_seconds"]),
            unknown_case_seconds=int(option_value["unknown_case_seconds"]),
            shard_count=int(option_value["shard_count"]),
        )
        if "nodeids" not in value:
            return cls(
                options=options,
                nodes=tuple(
                    CollectedNode(
                        nodeid=node["nodeid"],
                        source_file=node["source_file"],
                        base_function=node["base_function"],
                        order=int(node["order"]),
                        shard_group=node.get("shard_group"),
                        solo=bool(node.get("solo", False)),
                    )
                    for node in value["nodes"]
                ),
                units=tuple(Unit.from_dict(unit) for unit in value["units"]),
                fallback_counts={
                    str(name): int(count)
                    for name, count in value.get("fallback_counts", {}).items()
                },
                fallback_sources={
                    str(nodeid): str(source)
                    for nodeid, source in value.get("fallback_sources", {}).items()
                },
            )
        nodeids = tuple(str(nodeid) for nodeid in value["nodeids"])
        shard_groups = {
            int(index): str(group)
            for index, group in value.get("shard_groups", {}).items()
        }
        source_files = {
            int(index): str(group["path"])
            for group in value.get("source_files", [])
            for index in group["node_indexes"]
        }
        solo_sources = {str(source) for source in value.get("solo_sources", [])}
        nodes = tuple(
            CollectedNode(
                nodeid,
                source_files.get(index, source_file_for_nodeid(nodeid)),
                base_function_for_nodeid(nodeid),
                index,
                shard_groups.get(index),
                source_files.get(index, source_file_for_nodeid(nodeid)) in solo_sources,
            )
            for index, nodeid in enumerate(nodeids)
        )
        units = tuple(
            Unit(
                id=unit["id"],
                batches=tuple(
                    Batch(
                        id=batch["id"],
                        source_file=nodes[int(batch["node_indexes"][0])].source_file,
                        nodeids=tuple(
                            nodeids[int(index)] for index in batch["node_indexes"]
                        ),
                        estimated_ms=int(batch["estimated_ms"]),
                        overhead_ms=int(batch["overhead_ms"]),
                        oversized=bool(batch["oversized"]),
                    )
                    for batch in unit["batches"]
                ),
                estimated_ms=int(unit["estimated_ms"]),
                oversized=bool(unit["oversized"]),
                shard_index=int(unit["shard_index"]),
            )
            for unit in value["units"]
        )
        fallback_sources = {
            nodeids[int(index)]: str(group["source"])
            for group in value.get("fallbacks", [])
            for index in group["node_indexes"]
        }
        return cls(
            options=options,
            nodes=nodes,
            units=units,
            fallback_counts={
                str(name): int(count)
                for name, count in value.get("fallback_counts", {}).items()
            },
            fallback_sources=fallback_sources,
        )
