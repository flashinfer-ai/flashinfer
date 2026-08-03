from __future__ import annotations

import csv
import io
import json
import math
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

SOURCE_SUMMARY_HEADER = [
    "shard_index",
    "source_file",
    "status",
    "planned_nodes",
    "finalized_nodes",
    "passed",
    "failed",
    "skipped",
    "unknown",
    "pending_nodes",
    "process_seconds",
    "max_host_rss_mib",
    "max_gpu_memory_mib",
    "memory_samples",
    "partial_resources",
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


@dataclass
class _SourceSummary:
    shard_index: int
    source_file: str
    planned_nodes: int = 0
    finalized_nodes: int = 0
    pending_nodes: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    process_seconds: float = 0.0
    has_process_timing: bool = False
    max_host_rss_mib: float = 0.0
    max_gpu_memory_mib: float = 0.0
    memory_samples: int = 0
    partial_resources: bool = False

    @property
    def status(self) -> str:
        if self.pending_nodes:
            if self.outcomes["failed"]:
                return "failed-partial"
            if self.outcomes["unknown"]:
                return "unknown-partial"
            return "no-result"
        if self.outcomes["failed"]:
            return "failed"
        if self.outcomes["unknown"]:
            return "unknown"
        return "passed"


@dataclass(frozen=True)
class _FailedNode:
    nodeid: str
    reason: str
    shard_index: int


def batch_directory(junit_dir: Path, unit: Unit) -> Path:
    return units_dir(junit_dir) / unit.id / "batches"


def batch_xml_path(junit_dir: Path, unit: Unit, batch: Batch) -> Path:
    return batch_directory(junit_dir, unit) / f"{batch.id}.xml"


def batch_is_final(junit_dir: Path, unit: Unit, batch: Batch) -> bool:
    path = batch_xml_path(junit_dir, unit, batch)
    return path.exists() and validate_batch_xml(path, batch.nodeids).valid


def _sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(path.stem + suffix)


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _batch_metadata(path: Path) -> dict[str, Any]:
    return _json_object(_sidecar(path, ".meta.json")) or {}


def _attempt_for_batch(metadata: dict[str, Any]) -> str:
    return str(metadata.get("attempt_id", ""))


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


def _process_seconds(metadata: dict[str, Any]) -> float | None:
    try:
        launched_at = float(metadata["launched_at"])
        exited_at = float(metadata["exited_at"])
    except (KeyError, TypeError, ValueError):
        return None
    duration = exited_at - launched_at
    return duration if math.isfinite(duration) and duration >= 0 else None


def _batch_memory_monitoring(metadata: dict[str, Any]) -> bool:
    configured = metadata.get("monitor_memory")
    return configured if isinstance(configured, bool) else True


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
    sources: dict[tuple[int, str], _SourceSummary] = field(default_factory=dict)
    failed_nodes: list[_FailedNode] = field(default_factory=list)

    @classmethod
    def for_plan(cls, plan: Plan) -> _SummaryScan:
        scan = cls(
            shard_outcomes={
                index: Counter() for index in range(plan.options.shard_count)
            }
        )
        for unit in plan.units:
            for batch in unit.batches:
                key = (unit.shard_index, batch.source_file)
                source = scan.sources.setdefault(
                    key,
                    _SourceSummary(
                        shard_index=unit.shard_index,
                        source_file=batch.source_file,
                    ),
                )
                source.planned_nodes += len(batch.nodeids)
        return scan


def _record_pending(scan: _SummaryScan, unit: Unit, batch: Batch) -> None:
    scan.pending.extend(batch.nodeids)
    scan.shard_pending[unit.shard_index] += len(batch.nodeids)
    source = scan.sources[(unit.shard_index, batch.source_file)]
    source.pending_nodes += len(batch.nodeids)
    source.partial_resources = True


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


def _recorded_outcome(phase: dict[str, Any], fallback: str) -> str:
    outcome = phase.get("outcome")
    return (
        outcome if outcome in {"passed", "failed", "skipped", "unknown"} else fallback
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
    metadata = _batch_metadata(path)
    attempt_id = _attempt_for_batch(metadata)
    phases = _phase_results(path)
    samples = _memory_samples(path)
    memory = _max_memory(samples)
    source = scan.sources[(unit.shard_index, batch.source_file)]
    source.finalized_nodes += len(validation.cases)
    duration = _process_seconds(metadata)
    if duration is None:
        source.partial_resources = True
    else:
        source.process_seconds += duration
        source.has_process_timing = True
    if samples:
        source.max_host_rss_mib = max(source.max_host_rss_mib, memory[0])
        source.max_gpu_memory_mib = max(source.max_gpu_memory_mib, memory[1])
        source.memory_samples += len(samples)
    elif _batch_memory_monitoring(metadata):
        source.partial_resources = True
    scan.batch_memory[batch.id] = memory
    scan.unit_memory[unit.id].append(memory)
    scan.shard_memory[unit.shard_index].append(memory)
    for case in validation.cases:
        phase = phases.get(case.nodeid, {})
        outcome = _recorded_outcome(phase, case.outcome)
        scan.node_memory[case.nodeid] = _case_memory(samples, phase)
        source.outcomes[outcome] += 1
        scan.outcomes[outcome] += 1
        scan.shard_outcomes[unit.shard_index][outcome] += 1
        scan.shard_finalized[unit.shard_index] += 1
        scan.synthetic_count += int(case.synthetic)
        if outcome == "failed":
            scan.failed_nodes.append(
                _FailedNode(
                    nodeid=case.nodeid,
                    reason=(
                        case.failure_reason or "failure reason unavailable in JUnit XML"
                    ),
                    shard_index=unit.shard_index,
                )
            )
        scan.rows.append(
            {
                "nodeid": case.nodeid,
                "source_file": case.source_file,
                "base_function": case.base_function,
                "outcome": outcome,
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
    scan.failed_nodes.sort(
        key=lambda failed: (failed.nodeid.encode("utf-8"), failed.shard_index)
    )
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


def _source_csv_rows(scan: _SummaryScan) -> list[dict[str, Any]]:
    rows = []
    for source in sorted(
        scan.sources.values(),
        key=lambda item: (item.shard_index, item.source_file.encode("utf-8")),
    ):
        rows.append(
            {
                "shard_index": source.shard_index,
                "source_file": source.source_file,
                "status": source.status,
                "planned_nodes": source.planned_nodes,
                "finalized_nodes": source.finalized_nodes,
                "passed": source.outcomes["passed"],
                "failed": source.outcomes["failed"],
                "skipped": source.outcomes["skipped"],
                "unknown": source.outcomes["unknown"],
                "pending_nodes": source.pending_nodes,
                "process_seconds": f"{source.process_seconds:.6f}",
                "max_host_rss_mib": f"{source.max_host_rss_mib:.3f}",
                "max_gpu_memory_mib": f"{source.max_gpu_memory_mib:.3f}",
                "memory_samples": source.memory_samples,
                "partial_resources": str(source.partial_resources).lower(),
            }
        )
    return rows


def _write_source_csv(junit_dir: Path, scan: _SummaryScan) -> None:
    source_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        source_stream, fieldnames=SOURCE_SUMMARY_HEADER, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(_source_csv_rows(scan))
    atomic_write_text(junit_dir / "source_summary.csv", source_stream.getvalue())


def _merged_sources(
    scan: _SummaryScan,
    shard_index: int | None,
) -> list[_SourceSummary]:
    selected = [
        source
        for source in scan.sources.values()
        if shard_index is None or source.shard_index == shard_index
    ]
    if shard_index is not None:
        return sorted(selected, key=lambda item: item.source_file.encode("utf-8"))
    merged: dict[str, _SourceSummary] = {}
    for source in selected:
        target = merged.setdefault(
            source.source_file,
            _SourceSummary(shard_index=-1, source_file=source.source_file),
        )
        target.planned_nodes += source.planned_nodes
        target.finalized_nodes += source.finalized_nodes
        target.pending_nodes += source.pending_nodes
        target.outcomes.update(source.outcomes)
        target.process_seconds += source.process_seconds
        target.has_process_timing = (
            target.has_process_timing or source.has_process_timing
        )
        target.max_host_rss_mib = max(target.max_host_rss_mib, source.max_host_rss_mib)
        target.max_gpu_memory_mib = max(
            target.max_gpu_memory_mib, source.max_gpu_memory_mib
        )
        target.memory_samples += source.memory_samples
        target.partial_resources = target.partial_resources or source.partial_resources
    return sorted(merged.values(), key=lambda item: item.source_file.encode("utf-8"))


def _memory_monitoring_by_shard(
    junit_dir: Path,
    plan: Plan,
) -> dict[int, str]:
    observed: dict[int, set[bool]] = defaultdict(set)
    unknown: set[int] = set()
    for attempt_path in list_attempts(junit_dir):
        for shard_index in range(plan.options.shard_count):
            settings_path = (
                attempt_path / "shards" / f"shard-{shard_index:04d}.settings.json"
            )
            if not settings_path.exists():
                continue
            settings = _json_object(settings_path)
            if settings is None:
                unknown.add(shard_index)
                continue
            monitor_memory = settings.get("monitor_memory")
            if isinstance(monitor_memory, bool):
                observed[shard_index].add(monitor_memory)
            elif "monitor_memory" not in settings:
                observed[shard_index].add(True)
            else:
                unknown.add(shard_index)

    for unit in plan.units:
        for batch in unit.batches:
            path = batch_xml_path(junit_dir, unit, batch)
            if not path.exists():
                continue
            metadata_path = _sidecar(path, ".meta.json")
            metadata = _json_object(metadata_path)
            if metadata is None:
                unknown.add(unit.shard_index)
                continue
            if metadata.get("synthetic") is True:
                continue
            configured = metadata.get("monitor_memory")
            if isinstance(configured, bool):
                observed[unit.shard_index].add(configured)
            elif "monitor_memory" not in metadata:
                observed[unit.shard_index].add(True)
            else:
                unknown.add(unit.shard_index)

    states: dict[int, str] = {}
    for shard_index in range(plan.options.shard_count):
        values = observed[shard_index]
        if shard_index in unknown:
            states[shard_index] = "unknown"
        elif values == {False}:
            states[shard_index] = "disabled"
        elif values == {True}:
            states[shard_index] = "enabled"
        elif values:
            states[shard_index] = "mixed"
        else:
            states[shard_index] = "unknown"
    return states


def _memory_monitoring_message(
    junit_dir: Path,
    plan: Plan,
    shard_index: int | None,
) -> str | None:
    states = _memory_monitoring_by_shard(junit_dir, plan)
    selected = (
        [shard_index]
        if shard_index is not None
        else list(range(plan.options.shard_count))
    )
    disabled = [index for index in selected if states[index] == "disabled"]
    mixed = [index for index in selected if states[index] == "mixed"]
    unknown = [index for index in selected if states[index] == "unknown"]
    if unknown:
        return (
            "Memory monitoring: mixed or unknown (shards: "
            + ", ".join(str(index) for index in sorted({*disabled, *mixed, *unknown}))
            + ")"
        )
    if disabled and len(disabled) == len(selected):
        return "Memory monitoring: disabled"
    affected = sorted({*disabled, *mixed})
    if affected:
        return (
            "Memory monitoring: partially disabled (shards: "
            + ", ".join(str(index) for index in affected)
            + ")"
        )
    return None


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _format_mib(mib: float) -> str:
    if mib >= 1024 * 1024:
        return f"{mib / (1024 * 1024):.1f} TiB"
    if mib >= 1024:
        return f"{mib / 1024:.1f} GiB"
    return f"{int(round(mib))} MiB"


def _format_resource_row(rank: int, source: _SourceSummary) -> str:
    suffix = " (partial)" if source.partial_resources else ""
    return (
        f"  {rank:2d}. {source.source_file} - "
        f"duration {_format_duration(source.process_seconds)}, "
        f"peak RSS {_format_mib(source.max_host_rss_mib)}, "
        f"peak GPU {_format_mib(source.max_gpu_memory_mib)}, "
        f"samples {source.memory_samples}{suffix}"
    )


def format_test_run_summary(
    junit_dir: Path,
    plan: Plan,
    *,
    shard_index: int | None = None,
) -> str:
    """Render node outcomes and source-level resources from finalized batch data."""

    scan = _scan_batches(junit_dir, plan)
    sources = _merged_sources(scan, shard_index)
    failed_nodes = [
        failed
        for failed in scan.failed_nodes
        if shard_index is None or failed.shard_index == shard_index
    ]
    outcomes: Counter[str] = Counter()
    for source in sources:
        outcomes.update(source.outcomes)
    pending_nodes = sum(source.pending_nodes for source in sources)
    separator = "=" * 42
    title = (
        "TEST SUMMARY" if shard_index is None else f"TEST SUMMARY - SHARD {shard_index}"
    )
    lines = [
        separator,
        title,
        separator,
    ]
    if failed_nodes:
        lines.append("Failed test nodes:")
        lines.extend(
            f"  {index}. {failed.nodeid} - {failed.reason}"
            for index, failed in enumerate(failed_nodes, start=1)
        )
        lines.append("")
    lines.extend(
        [
            f"Total test files: {len(sources)}",
            f"Total test nodes: {sum(source.planned_nodes for source in sources)}",
            f"Passed: {outcomes['passed']}",
            f"Failed: {outcomes['failed']}",
            f"Skipped: {outcomes['skipped']}",
            f"Unknown: {outcomes['unknown']}",
            f"No result: {pending_nodes}",
        ]
    )
    failed = [source for source in sources if source.outcomes["failed"]]
    if failed:
        lines.extend(["", "Failed test files:"])
        lines.extend(
            f"  - {source.source_file} - "
            f"{source.outcomes['failed']}/{source.planned_nodes} failed"
            for source in failed
        )
    pending = [source for source in sources if source.pending_nodes]
    if pending:
        lines.extend(["", "Test files with no result:"])
        lines.extend(
            f"  - {source.source_file} - "
            f"{source.pending_nodes}/{source.planned_nodes} pending"
            for source in pending
        )

    lines.extend(["", separator, "TEST RUN RESOURCE SUMMARY", separator])
    monitoring_message = _memory_monitoring_message(junit_dir, plan, shard_index)
    if monitoring_message is not None:
        lines.append(monitoring_message)
    duration_sources = [source for source in sources if source.has_process_timing]
    memory_sources = [source for source in sources if source.memory_samples]
    if not duration_sources and not memory_sources:
        lines.append("No resource reports found for test-run resource summary.")
        return "\n".join(lines)
    rankings = (
        ("Top 10 longest-running test files:", duration_sources, "process_seconds"),
        ("Top 10 highest host RSS test files:", memory_sources, "max_host_rss_mib"),
        (
            "Top 10 highest GPU memory test files:",
            memory_sources,
            "max_gpu_memory_mib",
        ),
    )
    for title, candidates, attribute in rankings:
        if not candidates:
            continue
        lines.extend([title])
        ranked = sorted(
            candidates,
            key=lambda source: (
                -float(getattr(source, attribute)),
                source.source_file.encode("utf-8"),
            ),
        )[:10]
        lines.extend(
            _format_resource_row(rank, source)
            for rank, source in enumerate(ranked, start=1)
        )
    return "\n".join(lines)


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
    _write_source_csv(junit_dir, scan)
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
