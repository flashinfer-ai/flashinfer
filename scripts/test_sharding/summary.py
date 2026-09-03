from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict, cast

from .io import atomic_write_json, atomic_write_text
from .junit import validate_batch_xml
from .models import Batch, Plan, Unit
from .planner import CapacityMetrics, capacity_metrics
from .state import AttemptRecord, list_attempts, load_attempt, state_lock


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
_TERMINAL_SEPARATOR = "=" * 42


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


class FailedNodeSummary(TypedDict):
    shard_index: int
    source_file: str
    nodeid: str
    diagnostic: str
    batch_id: str
    unit_id: str
    log_path: str
    results_path: str
    junit_path: str
    synthetic: bool


class SourceSummary(TypedDict):
    shard_index: int
    source_file: str
    planned_nodes: int
    finalized_nodes: int
    pending_nodes: int
    passed: int
    failed: int
    skipped: int
    unknown: int
    synthetic: int
    process_seconds: float
    max_host_rss_mib: float
    max_gpu_memory_mib: float
    memory_samples: int
    partial_resources: bool
    status: str


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
    shard_infrastructure_errors: dict[str, list[str]]
    failed_nodes: list[FailedNodeSummary]
    sources: list[SourceSummary]


def batch_directory(junit_dir: Path, unit: Unit) -> Path:
    return (
        junit_dir
        / "shards"
        / f"shard-{unit.shard_index:04d}"
        / "units"
        / unit.id
        / "batches"
    )


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


def _json_sidecar(path: Path, suffix: str) -> dict[str, Any]:
    try:
        value = json.loads(_sidecar(path, suffix).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


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


SOURCE_HEADER = [
    "shard_index",
    "source_file",
    "status",
    "planned_nodes",
    "finalized_nodes",
    "pending_nodes",
    "passed",
    "failed",
    "skipped",
    "unknown",
    "synthetic",
    "process_seconds",
    "max_host_rss_mib",
    "max_gpu_memory_mib",
    "memory_samples",
    "partial_resources",
]


def _source_status(row: SourceSummary) -> str:
    if row["failed"]:
        return "failed"
    if row["finalized_nodes"] == 0:
        return "no result"
    if row["pending_nodes"]:
        return "incomplete"
    if row["unknown"]:
        return "unknown"
    return "passed"


def _source_and_failure_summaries(
    junit_dir: Path, plan: Plan
) -> tuple[list[SourceSummary], list[FailedNodeSummary]]:
    by_source: dict[tuple[int, str], SourceSummary] = {}
    failed_nodes: list[FailedNodeSummary] = []
    for unit in plan.units:
        for batch in unit.batches:
            key = (unit.shard_index, batch.source_file)
            row = by_source.setdefault(
                key,
                {
                    "shard_index": unit.shard_index,
                    "source_file": batch.source_file,
                    "planned_nodes": 0,
                    "finalized_nodes": 0,
                    "pending_nodes": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "unknown": 0,
                    "synthetic": 0,
                    "process_seconds": 0.0,
                    "max_host_rss_mib": 0.0,
                    "max_gpu_memory_mib": 0.0,
                    "memory_samples": 0,
                    "partial_resources": False,
                    "status": "",
                },
            )
            row["planned_nodes"] += len(batch.nodeids)
            path = batch_xml_path(junit_dir, unit, batch)
            validation = (
                validate_batch_xml(path, batch.nodeids) if path.exists() else None
            )
            if validation is None or not validation.valid:
                row["pending_nodes"] += len(batch.nodeids)
                row["partial_resources"] = True
                continue
            phases = _phase_results(path)
            meta = _json_sidecar(path, ".meta.json")
            telemetry = _json_sidecar(path, ".telemetry.json")
            samples = _memory_samples(path)
            memory = _max_memory(samples)
            row["max_host_rss_mib"] = max(row["max_host_rss_mib"], memory[0])
            row["max_gpu_memory_mib"] = max(row["max_gpu_memory_mib"], memory[1])
            row["memory_samples"] += len(samples)
            launched = telemetry.get("process_launch", meta.get("launched_at"))
            exited = telemetry.get("process_exit", meta.get("exited_at"))
            try:
                row["process_seconds"] += max(0.0, float(exited) - float(launched))
            except (TypeError, ValueError):
                if not bool(meta.get("synthetic", False)):
                    row["partial_resources"] = True
            if not bool(meta.get("synthetic", False)) and (
                not bool(meta.get("monitor_memory", True)) or not samples
            ):
                row["partial_resources"] = True
            for case in validation.cases:
                phase = phases.get(case.nodeid, {})
                outcome = str(phase.get("outcome", case.outcome))
                if outcome not in {"passed", "failed", "skipped", "unknown"}:
                    outcome = "unknown"
                row["finalized_nodes"] += 1
                cast(dict[str, Any], row)[outcome] += 1
                row["synthetic"] += int(case.synthetic)
                if outcome == "failed":
                    failed_nodes.append(
                        {
                            "shard_index": unit.shard_index,
                            "source_file": batch.source_file,
                            "nodeid": case.nodeid,
                            "diagnostic": str(
                                phase.get("longrepr")
                                or case.diagnostic
                                or "pytest failure"
                            ),
                            "batch_id": batch.id,
                            "unit_id": unit.id,
                            "log_path": str(_sidecar(path, ".log")),
                            "results_path": str(_sidecar(path, ".results.json")),
                            "junit_path": str(path),
                            "synthetic": case.synthetic,
                        }
                    )
    rows: list[SourceSummary] = []
    for key in sorted(by_source, key=lambda item: (item[0], item[1].encode("utf-8"))):
        row = by_source[key]
        row["status"] = _source_status(row)
        row["process_seconds"] = round(float(row["process_seconds"]), 6)
        row["max_host_rss_mib"] = round(float(row["max_host_rss_mib"]), 3)
        row["max_gpu_memory_mib"] = round(float(row["max_gpu_memory_mib"]), 3)
        rows.append(row)
    failed_nodes.sort(
        key=lambda item: (item["shard_index"], item["nodeid"].encode("utf-8"))
    )
    return rows, failed_nodes


def _write_source_csv(junit_dir: Path, rows: list[SourceSummary]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SOURCE_HEADER, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {name: cast(dict[str, Any], row)[name] for name in SOURCE_HEADER}
        for row in rows
    )
    atomic_write_text(junit_dir / "source_summary.csv", stream.getvalue())


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


def _latest_shard_infrastructure_errors(
    attempt_paths: list[Path], shard_count: int
) -> dict[str, list[str]]:
    errors_by_shard: dict[str, list[str]] = {
        str(index): [] for index in range(shard_count)
    }
    if not attempt_paths:
        return errors_by_shard
    for marker in sorted((attempt_paths[-1] / "shards").glob("shard-*.done.json")):
        try:
            shard_index = int(marker.name.removeprefix("shard-").split(".", 1)[0])
        except ValueError:
            continue
        key = str(shard_index)
        if key not in errors_by_shard:
            continue
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
            marker_errors = value.get("infrastructure_errors", [])
            if not isinstance(marker_errors, list):
                raise TypeError("infrastructure_errors is not a list")
            errors_by_shard[key].extend(str(error) for error in marker_errors)
        except (OSError, AttributeError, TypeError, json.JSONDecodeError) as error:
            errors_by_shard[key].append(
                f"invalid shard completion marker {marker}: {error}"
            )
    return errors_by_shard


def publish_summary_under_lock(junit_dir: Path, plan: Plan) -> RunSummary:
    """Publish derived artifacts while the caller owns ``state_lock(junit_dir)``."""

    scan = _scan_batches(junit_dir, plan)
    _write_timing_csv(junit_dir, scan.rows)
    _write_memory_csv(junit_dir, scan)
    sources, failed_nodes = _source_and_failure_summaries(junit_dir, plan)
    _write_source_csv(junit_dir, sources)
    attempts, attempt_paths = _attempt_history(junit_dir)
    shard_infrastructure_errors = _latest_shard_infrastructure_errors(
        attempt_paths, plan.options.shard_count
    )
    capacity = _capacity_for_latest_attempt(plan, attempt_paths)
    summary: RunSummary = {
        "schema_version": 3,
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
        "infrastructure_errors": [
            f"shard-{index}: {error}"
            for index, errors in shard_infrastructure_errors.items()
            for error in errors
        ],
        "shard_infrastructure_errors": shard_infrastructure_errors,
        "failed_nodes": failed_nodes,
        "sources": sources,
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
    if summary["outcomes"].get("failed", 0):
        return 1
    return 2 if not summary["complete"] else 0


def exit_code_for_shard(summary: RunSummary, shard_index: int) -> int:
    if summary["shard_infrastructure_errors"].get(str(shard_index), []):
        return 3
    shard = summary["shards"][str(shard_index)]
    if shard["outcomes"].get("failed", 0):
        return 1
    return 2 if not shard["complete"] else 0


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{seconds:.1f}s"


def _format_timestamp(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _format_test_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
    except Exception:
        return "unknown"


def _torch_cuda_version() -> str:
    try:
        import torch  # type: ignore[import-not-found]

        return str(torch.version.cuda or "none")
    except Exception:
        return "unavailable"


@lru_cache(maxsize=1)
def _runtime_version_line() -> str:
    versions = [
        ("python", sys.version.split()[0]),
        ("flashinfer", _package_version("flashinfer-python")),
        ("torch", _package_version("torch")),
        ("CUDA", _torch_cuda_version()),
        ("CuTE-DSL", _package_version("nvidia-cutlass-dsl")),
        ("cuda-python", _package_version("cuda-python")),
        ("cuda-tile", _package_version("cuda-tile")),
        ("cuDNN-frontend", _package_version("nvidia-cudnn-frontend")),
        ("triton", _package_version("triton")),
        ("nccl4py", _package_version("nccl4py")),
    ]
    return " ".join(f"{name}={version}" for name, version in versions)


def _format_memory(mib: float) -> str:
    return f"{mib / 1024:.1f} GiB" if mib >= 1024 else f"{mib:.0f} MiB"


def _top_source_lines(
    rows: list[SourceSummary], field: str, *, limit: int = 10
) -> list[str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(cast(dict[str, Any], row)[field]),
            row["source_file"].encode("utf-8"),
        ),
    )[:limit]
    lines = []
    for index, row in enumerate(ordered, 1):
        partial = " (partial)" if row["partial_resources"] else ""
        lines.append(
            f"  {index:2d}. {row['source_file']} - "
            f"duration {_format_duration(float(row['process_seconds']))}, "
            f"peak RSS {_format_memory(float(row['max_host_rss_mib']))}, "
            f"peak GPU {_format_memory(float(row['max_gpu_memory_mib']))}, "
            f"samples {row['memory_samples']}{partial}"
        )
    return lines or ["  No finalized source data."]


def _failure_detail_lines(
    failed_nodes: list[FailedNodeSummary], diagnostic_limit: int
) -> list[str]:
    real_failures = [failure for failure in failed_nodes if not failure["synthetic"]]
    if not real_failures:
        return []
    lines = [_TERMINAL_SEPARATOR, "FAILED TEST NODES", _TERMINAL_SEPARATOR]
    for failure in real_failures:
        diagnostic = str(failure["diagnostic"])
        encoded = diagnostic.encode("utf-8")
        truncated = len(encoded) > diagnostic_limit
        if truncated:
            diagnostic = encoded[:diagnostic_limit].decode("utf-8", errors="replace")
        lines.extend(
            [
                f"  - {failure['nodeid']}",
                "    " + diagnostic.replace("\n", "\n    "),
            ]
        )
        if truncated:
            lines.append(f"    [diagnostic truncated at {diagnostic_limit} bytes]")
        lines.append(f"    log: {failure['log_path']}")
    return lines


def _synthetic_timeout_failure_lines(
    failed_nodes: list[FailedNodeSummary],
) -> list[str]:
    synthetic_failures = [failure for failure in failed_nodes if failure["synthetic"]]
    if not synthetic_failures:
        return []
    return [
        _TERMINAL_SEPARATOR,
        "SYNTHETIC FAIL DUE TO TIMEOUT",
        _TERMINAL_SEPARATOR,
        *(f"  - {failure['nodeid']}" for failure in synthetic_failures),
    ]


def terminal_summary_lines(
    summary: RunSummary | None,
    *,
    shard_index: int | None,
    runner_exit_code: int,
    shell_exit_code: int | None = None,
    status: str,
    test_started_at: float,
    test_ended_at: float,
    cause: str | None = None,
    diagnostic_limit: int = 32 * 1024,
) -> list[str]:
    lines = [
        _TERMINAL_SEPARATOR,
        "TEST SUMMARY",
        _TERMINAL_SEPARATOR,
        f"Start time: {_format_timestamp(test_started_at)}",
        f"End time: {_format_timestamp(test_ended_at)}",
        f"Time elapsed: {_format_test_elapsed(test_ended_at - test_started_at)}",
        f"Versions: {_runtime_version_line()}",
    ]
    if summary is None:
        lines.extend(
            [
                "Scope: unavailable",
                "Planned nodes: unavailable",
                "Finalized nodes: unavailable",
            ]
        )
        failed_nodes: list[FailedNodeSummary] = []
        sources: list[SourceSummary] = []
        infrastructure_errors: list[str] = []
    else:
        if shard_index is None:
            outcomes = summary["outcomes"]
            sources = summary["sources"]
            failed_nodes = summary["failed_nodes"]
            scope_lines = [
                "Scope: shared run",
                f"Planned nodes: {summary['planned_nodes']}",
                f"Finalized nodes: {summary['finalized_nodes']}",
            ]
            pending_count = len(summary["pending_nodes"])
            infrastructure_errors = summary["infrastructure_errors"]
        else:
            shard = summary["shards"][str(shard_index)]
            outcomes = shard["outcomes"]
            sources = [
                row for row in summary["sources"] if row["shard_index"] == shard_index
            ]
            failed_nodes = [
                row
                for row in summary["failed_nodes"]
                if row["shard_index"] == shard_index
            ]
            scope_lines = [
                f"Shard: {shard_index}",
                f"Planned nodes: {shard['planned_nodes']}",
                f"Finalized nodes: {shard['finalized_nodes']}",
            ]
            pending_count = shard["pending_nodes"]
            infrastructure_errors = summary["shard_infrastructure_errors"].get(
                str(shard_index), []
            )
        source_statuses = Counter(row["status"] for row in sources)
        synthetic = sum(int(row["synthetic"]) for row in sources)
        lines.extend(
            [
                *scope_lines,
                f"Passed: {outcomes.get('passed', 0)}",
                f"Failed: {outcomes.get('failed', 0)}",
                f"Skipped: {outcomes.get('skipped', 0)}",
                f"Unknown: {outcomes.get('unknown', 0)}",
                f"Pending: {pending_count}",
                f"Synthetic: {synthetic}",
                f"Source files: passed={source_statuses['passed']} "
                f"failed={source_statuses['failed']} "
                f"incomplete={source_statuses['incomplete']} "
                f"unknown={source_statuses['unknown']} "
                f"no-result={source_statuses['no result']}",
            ]
        )
        if shard_index is not None:
            lines.append(
                f"Shared run: finalized={summary['finalized_nodes']}/{summary['planned_nodes']} "
                f"pending={len(summary['pending_nodes'])}"
            )
    if cause:
        lines.extend(["", "STOP CAUSE", f"  {cause}"])
    if infrastructure_errors:
        lines.extend(["", "INFRASTRUCTURE ERRORS"])
        lines.extend(f"  - {error}" for error in infrastructure_errors)
    real_failed_nodes = [
        failure for failure in failed_nodes if not failure["synthetic"]
    ]
    synthetic_failed_nodes = [
        failure for failure in failed_nodes if failure["synthetic"]
    ]
    if real_failed_nodes:
        failed_sources = Counter(
            failure["source_file"] for failure in real_failed_nodes
        )
        ordered_failed_sources = sorted(
            failed_sources,
            key=lambda source: source.encode("utf-8"),
        )
        lines.extend(["", "Failed test files:"])
        lines.extend(
            f"  - {source} ({failed_sources[source]} failed "
            f"{'node' if failed_sources[source] == 1 else 'nodes'})"
            for source in ordered_failed_sources
        )
    if synthetic_failed_nodes:
        timeout_sources = Counter(
            failure["source_file"] for failure in synthetic_failed_nodes
        )
        ordered_timeout_sources = sorted(
            timeout_sources,
            key=lambda source: source.encode("utf-8"),
        )
        lines.extend(["", "Timeout test files:"])
        lines.extend(
            f"  - {source} ({timeout_sources[source]} timeout "
            f"{'node' if timeout_sources[source] == 1 else 'nodes'})"
            for source in ordered_timeout_sources
        )
    lines.extend(
        ["", "TEST RUN RESOURCE SUMMARY", "Top 10 longest-running source files:"]
    )
    lines.extend(_top_source_lines(sources, "process_seconds"))
    lines.append("Top 10 highest host RSS source files:")
    lines.extend(_top_source_lines(sources, "max_host_rss_mib"))
    lines.append("Top 10 highest GPU memory source files:")
    lines.extend(_top_source_lines(sources, "max_gpu_memory_mib"))
    lines.extend(
        [
            _TERMINAL_SEPARATOR,
            f"Result: status={status} python_exit_code={runner_exit_code} "
            + (
                f"shell_exit_code={shell_exit_code}"
                if shell_exit_code is not None
                else "shell_exit_code=pending"
            ),
            _TERMINAL_SEPARATOR,
        ]
    )
    return [
        *_failure_detail_lines(failed_nodes, diagnostic_limit),
        *_synthetic_timeout_failure_lines(failed_nodes),
        *lines,
    ]
