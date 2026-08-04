from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict, cast

from .io import atomic_write_json, atomic_write_text
from .junit import validate_batch_xml
from .models import Batch, Plan, Unit
from .planner import CapacityMetrics, capacity_metrics
from .state import AttemptRecord, list_attempts, load_attempt, state_lock, units_dir


TIMING_HEADER = [
    "nodeid",
    "source_file",
    "base_function",
    "outcome",
    "setup_seconds",
    "call_seconds",
    "teardown_seconds",
    "wall_clock_seconds",
    "synthetic",
    "batch_id",
    "unit_id",
    "shard_index",
    "attempt_id",
]


class AttemptSummary(AttemptRecord, total=False):
    unit_timeout_events: list[dict[str, Any]]
    closed: bool
    closure: dict[str, Any]


class ShardSummary(TypedDict):
    complete: bool
    planned_nodes: int
    finalized_nodes: int
    pending_nodes: int
    outcomes: dict[str, int]


class FallbackSummary(TypedDict):
    source: str
    node_indexes: list[int]


class RunSummary(TypedDict):
    schema_version: int
    complete: bool
    planned_nodes: int
    finalized_nodes: int
    pending_nodes: list[str]
    outcomes: dict[str, int]
    shards: dict[str, ShardSummary]
    synthetic: int
    fallback_counts: dict[str, int]
    fallback_node_table: str
    fallbacks: list[FallbackSummary]
    oversized_batches: int
    oversized_units: int
    estimated_shard_load_ms: dict[str, int]
    estimated_makespan_ms: int
    estimated_total_overhead_ms: int
    capacity: CapacityMetrics
    attempts: list[AttemptSummary]
    infrastructure_errors: list[str]


def batch_directory(junit_dir: Path, unit: Unit) -> Path:
    return units_dir(junit_dir) / unit.id / "batches"


def batch_xml_path(junit_dir: Path, unit: Unit, batch: Batch) -> Path:
    return batch_directory(junit_dir, unit) / f"{batch.id}.xml"


def batch_is_final(junit_dir: Path, unit: Unit, batch: Batch) -> bool:
    path = batch_xml_path(junit_dir, unit, batch)
    return path.exists() and validate_batch_xml(path, batch.nodeids).valid


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix)


def _attempt_for_batch(path: Path) -> str:
    try:
        value = json.loads(_sidecar(path, ".meta.json").read_text(encoding="utf-8"))
        return str(value.get("attempt_id", ""))
    except (OSError, json.JSONDecodeError):
        return ""


def _phase_results(path: Path) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(_sidecar(path, ".results.json").read_text(encoding="utf-8"))
        return {item["nodeid"]: item for item in value.get("results", [])}
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {}


def _memory_samples(path: Path) -> list[dict[str, float]]:
    sample_path = _sidecar(path, ".memory.csv")
    if not sample_path.exists():
        return []
    samples: list[dict[str, float]] = []
    try:
        with sample_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                samples.append(
                    {
                        "timestamp": float(row["timestamp"]),
                        "host_rss_mib": float(row["host_rss_mib"]),
                        "gpu_memory_mib": float(row["gpu_memory_mib"]),
                    }
                )
    except (OSError, KeyError, TypeError, ValueError):
        return []
    return samples


def _max_memory(samples: list[dict[str, float]]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    return (
        max(sample["host_rss_mib"] for sample in samples),
        max(sample["gpu_memory_mib"] for sample in samples),
    )


@dataclass
class _SummaryScan:
    rows: list[dict[str, Any]] = field(default_factory=list)
    outcomes: Counter[str] = field(default_factory=Counter)
    shard_outcomes: dict[int, Counter[str]] = field(default_factory=dict)
    shard_planned: Counter[int] = field(default_factory=Counter)
    shard_pending: Counter[int] = field(default_factory=Counter)
    shard_finalized: Counter[int] = field(default_factory=Counter)
    pending: list[str] = field(default_factory=list)
    synthetic_count: int = 0
    batch_memory: dict[str, tuple[float, float]] = field(default_factory=dict)
    node_memory: dict[str, tuple[float, float]] = field(default_factory=dict)
    unit_memory: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    shard_memory: dict[int, list[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @classmethod
    def for_plan(cls, plan: Plan) -> _SummaryScan:
        return cls(
            shard_outcomes={
                index: Counter() for index in range(plan.options.shard_count)
            }
        )


def _record_pending(scan: _SummaryScan, unit: Unit, batch: Batch) -> None:
    scan.pending.extend(batch.nodeids)
    scan.shard_pending[unit.shard_index] += len(batch.nodeids)


def _case_memory(
    samples: list[dict[str, float]], phase: dict[str, Any]
) -> tuple[float, float]:
    started_at = phase.get("started_at")
    finished_at = phase.get("finished_at")
    if started_at is None or finished_at is None:
        return 0.0, 0.0
    return _max_memory(
        [
            sample
            for sample in samples
            if float(started_at) <= sample["timestamp"] <= float(finished_at)
        ]
    )


def _record_final_batch(
    scan: _SummaryScan,
    *,
    unit: Unit,
    batch: Batch,
    path: Path,
) -> None:
    validation = validate_batch_xml(path, batch.nodeids)
    if not validation.valid:
        _record_pending(scan, unit, batch)
        return
    attempt_id = _attempt_for_batch(path)
    phases = _phase_results(path)
    samples = _memory_samples(path)
    memory = _max_memory(samples)
    scan.batch_memory[batch.id] = memory
    scan.unit_memory[unit.id].append(memory)
    scan.shard_memory[unit.shard_index].append(memory)
    for case in validation.cases:
        phase = phases.get(case.nodeid, {})
        scan.node_memory[case.nodeid] = _case_memory(samples, phase)
        scan.outcomes[case.outcome] += 1
        scan.shard_outcomes[unit.shard_index][case.outcome] += 1
        scan.shard_finalized[unit.shard_index] += 1
        scan.synthetic_count += int(case.synthetic)
        scan.rows.append(
            {
                "nodeid": case.nodeid,
                "source_file": case.source_file,
                "base_function": case.base_function,
                "outcome": case.outcome,
                "setup_seconds": phase.get("setup", 0.0),
                "call_seconds": phase.get("call", case.seconds),
                "teardown_seconds": phase.get("teardown", 0.0),
                "wall_clock_seconds": case.seconds,
                "synthetic": str(case.synthetic).lower(),
                "batch_id": batch.id,
                "unit_id": unit.id,
                "shard_index": unit.shard_index,
                "attempt_id": attempt_id,
            }
        )


def _scan_batches(junit_dir: Path, plan: Plan) -> _SummaryScan:
    scan = _SummaryScan.for_plan(plan)

    for unit in plan.units:
        for batch in unit.batches:
            scan.shard_planned[unit.shard_index] += len(batch.nodeids)
            path = batch_xml_path(junit_dir, unit, batch)
            if not path.exists():
                _record_pending(scan, unit, batch)
                continue
            _record_final_batch(scan, unit=unit, batch=batch, path=path)
    scan.rows.sort(key=lambda row: row["nodeid"].encode("utf-8"))
    for outcome in ("passed", "failed", "skipped"):
        scan.outcomes.setdefault(outcome, 0)
        for shard in scan.shard_outcomes.values():
            shard.setdefault(outcome, 0)
    return scan


def _write_timing_csv(junit_dir: Path, rows: list[dict[str, Any]]) -> None:
    timing_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        timing_stream, fieldnames=TIMING_HEADER, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(junit_dir / "test_timings.csv", timing_stream.getvalue())


def _write_memory_csv(junit_dir: Path, scan: _SummaryScan) -> None:
    memory_stream = io.StringIO(newline="")
    memory_writer = csv.writer(memory_stream, lineterminator="\n")
    memory_writer.writerow(["level", "id", "max_host_rss_mib", "max_gpu_memory_mib"])
    for nodeid, memory in sorted(scan.node_memory.items()):
        memory_writer.writerow(["node", nodeid, f"{memory[0]:.3f}", f"{memory[1]:.3f}"])
    for batch_id, memory in sorted(scan.batch_memory.items()):
        memory_writer.writerow(
            ["batch", batch_id, f"{memory[0]:.3f}", f"{memory[1]:.3f}"]
        )
    for unit_id, memories in sorted(scan.unit_memory.items()):
        memory_writer.writerow(
            [
                "unit",
                unit_id,
                f"{max(value[0] for value in memories):.3f}",
                f"{max(value[1] for value in memories):.3f}",
            ]
        )
    for shard_index, memories in sorted(scan.shard_memory.items()):
        memory_writer.writerow(
            [
                "shard",
                shard_index,
                f"{max(value[0] for value in memories):.3f}",
                f"{max(value[1] for value in memories):.3f}",
            ]
        )
    atomic_write_text(junit_dir / "memory_summary.csv", memory_stream.getvalue())


def _attempt_history(junit_dir: Path) -> tuple[list[AttemptSummary], list[Path]]:
    attempts: list[AttemptSummary] = []
    attempt_paths = list_attempts(junit_dir)
    for path in attempt_paths:
        attempt = cast(AttemptSummary, load_attempt(path))
        timeout_events = []
        for event_path in sorted((path / "timed-out").glob("*.json")):
            try:
                timeout_events.append(
                    json.loads(event_path.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError):
                timeout_events.append({"path": str(event_path), "malformed": True})
        attempt["unit_timeout_events"] = timeout_events
        closed_path = path / "closed.json"
        attempt["closed"] = closed_path.exists()
        if closed_path.exists():
            try:
                attempt["closure"] = json.loads(closed_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                attempt["closure"] = {"malformed": True}
        attempts.append(attempt)
    return attempts, attempt_paths


def _capacity_for_latest_attempt(
    plan: Plan, attempt_paths: list[Path]
) -> CapacityMetrics:
    workers_by_shard: dict[int, int] = {}
    deadline_seconds = 0
    if attempt_paths:
        latest_path = attempt_paths[-1]
        latest_attempt = load_attempt(latest_path)
        deadline_seconds = int(latest_attempt["settings"]["deadline_seconds"])
        for settings_path in (latest_path / "shards").glob("shard-*.settings.json"):
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
                workers_by_shard[int(settings["shard_index"])] = int(
                    settings["workers"]
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return capacity_metrics(
        plan,
        workers_by_shard,
        deadline_seconds=deadline_seconds,
    )


def _latest_infrastructure_errors(attempts: list[AttemptSummary]) -> list[str]:
    if not attempts:
        return []
    closure = attempts[-1].get("closure", {})
    errors = closure.get("infrastructure_errors", [])
    if isinstance(errors, list):
        return [str(error) for error in errors]
    return ["latest attempt has malformed infrastructure error state"]


def publish_summary_under_lock(junit_dir: Path, plan: Plan) -> RunSummary:
    """Publish derived artifacts while the caller owns ``state_lock(junit_dir)``."""

    scan = _scan_batches(junit_dir, plan)
    _write_timing_csv(junit_dir, scan.rows)
    _write_memory_csv(junit_dir, scan)
    attempts, attempt_paths = _attempt_history(junit_dir)
    capacity = _capacity_for_latest_attempt(plan, attempt_paths)
    summary: RunSummary = {
        "schema_version": 2,
        "complete": not scan.pending,
        "planned_nodes": len(plan.nodes),
        "finalized_nodes": len(scan.rows),
        "pending_nodes": sorted(scan.pending, key=lambda value: value.encode("utf-8")),
        "outcomes": dict(sorted(scan.outcomes.items())),
        "shards": {
            str(index): {
                "complete": scan.shard_pending[index] == 0,
                "planned_nodes": scan.shard_planned[index],
                "finalized_nodes": scan.shard_finalized[index],
                "pending_nodes": scan.shard_pending[index],
                "outcomes": dict(sorted(scan.shard_outcomes[index].items())),
            }
            for index in range(plan.options.shard_count)
        },
        "synthetic": scan.synthetic_count,
        "fallback_counts": dict(sorted(plan.fallback_counts.items())),
        "fallback_node_table": "manifest.json#/plan/nodeids",
        "fallbacks": [
            {"source": source, "node_indexes": indexes}
            for source, indexes in plan.fallback_index_groups().items()
        ],
        "oversized_batches": sum(
            batch.oversized for unit in plan.units for batch in unit.batches
        ),
        "oversized_units": sum(unit.oversized for unit in plan.units),
        "estimated_shard_load_ms": capacity["estimated_shard_load_ms"],
        "estimated_makespan_ms": capacity["estimated_makespan_ms"],
        "estimated_total_overhead_ms": capacity["total_estimated_overhead_ms"],
        "capacity": capacity,
        "attempts": attempts,
        "infrastructure_errors": _latest_infrastructure_errors(attempts),
    }
    atomic_write_json(junit_dir / "run-summary.json", summary)
    return summary


def publish_summary(junit_dir: Path, plan: Plan) -> RunSummary:
    """Regenerate all derived artifacts from one serialized batch snapshot."""

    with state_lock(junit_dir):
        return publish_summary_under_lock(junit_dir, plan)


def exit_code_for_summary(summary: RunSummary) -> int:
    if summary["infrastructure_errors"]:
        return 3
    if not summary["complete"]:
        return 2
    return 1 if summary["outcomes"].get("failed", 0) else 0
