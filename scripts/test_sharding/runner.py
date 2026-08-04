from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .estimates import EstimateBook
from .io import atomic_write_json
from .junit import (
    SyntheticBatchMetadata,
    create_synthetic_batch_xml,
    finalized_batch_outcomes,
)
from .models import CollectedNode, Plan, PlanningOptions, Unit
from .planner import build_plan, capacity_metrics, source_affine_unit_bins
from .processes import terminate_process_group
from .state import (
    AttemptRecord,
    AttemptSettings,
    ManifestBuild,
    RunnerStateError,
    active_attempt_collection_timeout,
    attempts_dir,
    build_manifest,
    claims_dir,
    collection_fingerprint,
    create_or_join_attempt,
    lease_is_live,
    lease_value,
    list_attempts,
    load_attempt,
    load_manifest,
    recover_unit_elapsed,
    sha256_file,
    source_git_sha_from_env,
    state_lock,
    units_dir,
    verify_manifest,
    write_unit_elapsed,
    write_manifest,
)
from .summary import (
    RunSummary,
    batch_is_final,
    batch_xml_path,
    exit_code_for_summary,
    publish_summary,
    publish_summary_under_lock,
)
from .workers import BatchExecutionRequest, execute_batch, write_console


_COLLECTION_HEARTBEAT_SECONDS = 30.0
_LEASE_HEARTBEAT_SECONDS = 10.0


@dataclass(frozen=True)
class DeadlineClock:
    started_at: float
    limit_seconds: int

    @classmethod
    def from_attempt(cls, attempt: AttemptRecord) -> DeadlineClock:
        return cls(
            started_at=float(attempt["started_at"]),
            limit_seconds=int(attempt["settings"]["deadline_seconds"]),
        )

    def status_fields(self) -> str:
        elapsed = max(0.0, time.time() - self.started_at)
        limit = f"{self.limit_seconds}s" if self.limit_seconds > 0 else "disabled"
        return f"deadline_elapsed={elapsed:.1f}s deadline={limit}"


class CollectionTimeoutError(RunnerStateError):
    def __init__(self, termination_signal: str, elapsed_seconds: float) -> None:
        super().__init__("pytest collection exceeded the remaining attempt deadline")
        self.termination_signal = termination_signal
        self.elapsed_seconds = elapsed_seconds
        self.result_scope = "current-process"
        self.finalized_nodes = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.pending_nodes: int | None = None
        self.deadline_clock: DeadlineClock | None = None

    def record_existing_results(self, summary: RunSummary) -> None:
        outcomes = summary["outcomes"]
        self.result_scope = "suite"
        self.finalized_nodes = int(summary["finalized_nodes"])
        self.passed = int(outcomes.get("passed", 0))
        self.failed = int(outcomes.get("failed", 0))
        self.skipped = int(outcomes.get("skipped", 0))
        self.pending_nodes = len(summary["pending_nodes"])

    def record_deadline_clock(self, deadline_clock: DeadlineClock) -> None:
        self.deadline_clock = deadline_clock


@dataclass(frozen=True)
class SelectionSettings:
    test_path: Path
    sanity_test: bool
    sample_rate: int
    sample_offset: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sanity_test": self.sanity_test,
            "sample_rate": self.sample_rate,
            "sample_offset": self.sample_offset,
        }


def _reject_runner_path_collisions(junit_dir: Path, *, include_attempts: bool) -> None:
    runner_paths = [
        junit_dir / "test_timings.csv",
        junit_dir / "memory_summary.csv",
        junit_dir / "run-summary.json",
        claims_dir(junit_dir),
        units_dir(junit_dir),
    ]
    if include_attempts:
        runner_paths.append(attempts_dir(junit_dir))
    collisions = [path for path in runner_paths if path.exists()]
    if collisions:
        raise RunnerStateError(
            "new run would collide with runner-owned paths: "
            + ", ".join(str(path) for path in collisions[:5])
        )


@dataclass(frozen=True)
class ExecutionSettings:
    workers: int
    shard_index: int
    attempt: AttemptSettings
    monitor_memory: bool = True
    memory_interval: float = 2.0
    pytest_command_prefix: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManifestPreparation:
    repo_root: Path
    junit_dir: Path
    selection: SelectionSettings
    planning: PlanningOptions
    collection_timeout_seconds: float | None = None
    collection_grace_seconds: float = 5.0
    attempt_settings: AttemptSettings | None = None
    operation_started_at: float | None = None
    pytest_command_prefix: tuple[str, ...] = ()


def _report_shard_status(
    summary: RunSummary,
    shard_index: int,
    *,
    state: str,
    deadline_clock: DeadlineClock,
    reason: str | None = None,
) -> None:
    shard = summary["shards"][str(shard_index)]
    outcomes = shard["outcomes"]
    reason_field = f" reason={reason}" if reason else ""
    print(
        f"RUNNER STATUS: shard={shard_index} state={state}{reason_field} "
        f"finalized={shard['finalized_nodes']}/{shard['planned_nodes']} "
        f"passed={outcomes.get('passed', 0)} failed={outcomes.get('failed', 0)} "
        f"skipped={outcomes.get('skipped', 0)} pending={shard['pending_nodes']} "
        f"suite_complete={str(summary['complete']).lower()} "
        f"{deadline_clock.status_fields()}",
        flush=True,
    )


def _estimate_paths(repo_root: Path) -> tuple[Path, Path]:
    data = repo_root / "tests" / "data"
    return (
        data / "unit_test_duration_estimates.csv.gz",
        data / "unit_test_overhead_estimates.csv",
    )


def _estimate_checksums(repo_root: Path) -> dict[str, str | None]:
    duration, overhead = _estimate_paths(repo_root)
    return {
        "duration": sha256_file(duration) if duration.exists() else None,
        "overhead": sha256_file(overhead) if overhead.exists() else None,
    }


def _wait_for_collection(
    process: subprocess.Popen[Any],
    *,
    test_path: Path,
    timeout_seconds: float | None,
    grace_seconds: float,
) -> None:
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds if timeout_seconds is not None else None
    while process.poll() is None:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            termination_signal = terminate_process_group(process, grace_seconds)
            raise CollectionTimeoutError(
                termination_signal, time.monotonic() - started_at
            )
        wait_seconds = _COLLECTION_HEARTBEAT_SECONDS
        if remaining is not None:
            wait_seconds = min(wait_seconds, remaining)
        try:
            process.wait(timeout=max(0.001, wait_seconds))
        except subprocess.TimeoutExpired:
            if deadline is not None and time.monotonic() >= deadline:
                termination_signal = terminate_process_group(process, grace_seconds)
                raise CollectionTimeoutError(
                    termination_signal, time.monotonic() - started_at
                ) from None
            print(
                "RUNNER HEARTBEAT: state=collecting "
                f"elapsed={time.monotonic() - started_at:.1f}s "
                f"collected=unknown test_path={test_path}",
                flush=True,
            )


def _collect_nodes(
    repo_root: Path,
    test_path: Path,
    timeout_seconds: float | None,
    grace_seconds: float,
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="flashinfer-test-collection-") as directory:
        output = Path(directory) / "collection.json"
        log_path = Path(directory) / "collection.log"
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
        )
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-q",
            "--continue-on-collection-errors",
            "-p",
            "scripts.test_sharding.pytest_plugin",
            f"--flashinfer-collection-json={output}",
            str(test_path),
        ]
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_collection(
                process,
                test_path=test_path,
                timeout_seconds=timeout_seconds,
                grace_seconds=grace_seconds,
            )
        if process.returncode != 0 or not output.exists():
            diagnostic = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
            raise RunnerStateError(
                f"pytest collection failed with exit code {process.returncode}:\n{diagnostic}"
            )
        try:
            value = json.loads(output.read_text(encoding="utf-8"))
            nodes = value["nodes"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RunnerStateError(f"invalid collection metadata: {error}") from error
        if not nodes:
            raise RunnerStateError(f"pytest collected no tests below {test_path}")
        return nodes


def _validate_selection(selection: SelectionSettings) -> None:
    if not selection.test_path.exists():
        raise RunnerStateError(f"test path does not exist: {selection.test_path}")
    if selection.sample_rate <= 0:
        raise RunnerStateError("SAMPLE_RATE must be positive")
    if not 0 <= selection.sample_offset < selection.sample_rate:
        raise RunnerStateError("SAMPLE_OFFSET must be in [0, SAMPLE_RATE)")


def _selected_nodes(
    raw_nodes: list[dict[str, Any]],
    selection: SelectionSettings,
) -> list[CollectedNode]:
    if selection.sanity_test:
        raw_nodes = [
            node
            for index, node in enumerate(raw_nodes)
            if index % selection.sample_rate == selection.sample_offset
        ]
        if not raw_nodes:
            raise RunnerStateError("sanity sampling selected no tests")
    return [
        CollectedNode(
            nodeid=node["nodeid"],
            source_file=node["source_file"],
            base_function=node["base_function"],
            order=int(node["order"]),
            shard_group=node.get("shard_group"),
            solo=bool(node.get("solo", False)),
        )
        for node in raw_nodes
    ]


def _verify_collection(manifest: dict[str, Any], nodes: list[CollectedNode]) -> None:
    current = collection_fingerprint(nodes)
    saved = manifest.get("collection_fingerprint")
    if saved != current:
        raise RunnerStateError(
            "existing run is incompatible; use a different --junit-dir:\n  "
            f"collection_fingerprint: saved={saved!r}, current={current!r}"
        )


def _write_new_manifest(
    *,
    request: ManifestPreparation,
    source_git_sha: str | None,
    nodes: list[CollectedNode],
    manifest: dict[str, Any],
    plan: Plan,
) -> tuple[dict[str, Any], Plan, bool]:
    with state_lock(request.junit_dir):
        concurrent = load_manifest(request.junit_dir)
        if concurrent is not None:
            verify_manifest(
                concurrent,
                source_git_sha=source_git_sha,
                test_path=request.selection.test_path,
                selection=request.selection.to_dict(),
                planning_options=request.planning.to_dict(),
                pytest_command_prefix=request.pytest_command_prefix,
            )
            _verify_collection(concurrent, nodes)
            return concurrent, Plan.from_dict(concurrent["plan"]), False
        _reject_runner_path_collisions(
            request.junit_dir,
            include_attempts=request.attempt_settings is None,
        )
        write_manifest(request.junit_dir, manifest)
    return manifest, plan, True


def prepare_manifest(
    request: ManifestPreparation,
) -> tuple[dict[str, Any], Plan, bool]:
    repo_root = request.repo_root
    junit_dir = request.junit_dir
    selection = request.selection
    planning = request.planning
    collection_timeout_seconds = request.collection_timeout_seconds
    attempt_settings = request.attempt_settings
    operation_started_at = request.operation_started_at
    test_path = selection.test_path
    _validate_selection(selection)
    junit_dir.mkdir(parents=True, exist_ok=True)
    source_git_sha = source_git_sha_from_env()
    selection_value = selection.to_dict()
    existing = load_manifest(junit_dir)
    existing_plan: Plan | None = None
    if existing is not None:
        verify_manifest(
            existing,
            source_git_sha=source_git_sha,
            test_path=test_path,
            selection=selection_value,
            planning_options=planning.to_dict(),
            pytest_command_prefix=request.pytest_command_prefix,
        )
        existing_plan = Plan.from_dict(existing["plan"])
    if existing is None:
        _reject_runner_path_collisions(
            junit_dir, include_attempts=attempt_settings is None
        )
    attempts = list_attempts(junit_dir)
    active_attempt = bool(attempts and not (attempts[-1] / "closed.json").exists())
    outputs_complete = bool(
        existing_plan is not None
        and all(
            batch_is_final(junit_dir, unit, batch)
            for unit in existing_plan.units
            for batch in unit.batches
        )
    )
    deadline_clock = (
        DeadlineClock(
            started_at=(
                time.time() if operation_started_at is None else operation_started_at
            ),
            limit_seconds=attempt_settings.deadline_seconds,
        )
        if attempt_settings is not None
        else None
    )
    if attempt_settings is not None and (
        existing is None or not outputs_complete or active_attempt
    ):
        _, attempt = create_or_join_attempt(
            junit_dir,
            attempt_settings,
            started_at=(
                time.time() if operation_started_at is None else operation_started_at
            ),
        )
        deadline_clock = DeadlineClock.from_attempt(attempt)
    collection_timeout_seconds = active_attempt_collection_timeout(
        junit_dir, collection_timeout_seconds
    )
    if existing is not None:
        assert existing_plan is not None
        return existing, existing_plan, False
    try:
        raw_nodes = _collect_nodes(
            repo_root,
            test_path,
            collection_timeout_seconds,
            request.collection_grace_seconds,
        )
    except CollectionTimeoutError as error:
        if existing_plan is not None:
            error.record_existing_results(publish_summary(junit_dir, existing_plan))
        if deadline_clock is not None:
            error.record_deadline_clock(deadline_clock)
        raise
    nodes = _selected_nodes(raw_nodes, selection)
    duration_path, overhead_path = _estimate_paths(repo_root)
    estimates = EstimateBook.from_files(duration_path, overhead_path)
    plan = build_plan(nodes, estimates, planning)
    manifest = build_manifest(
        ManifestBuild(
            repo_root=repo_root,
            test_path=test_path,
            source_git_sha=source_git_sha,
            plan=plan,
            selection=selection_value,
            estimate_files=_estimate_checksums(repo_root),
            pytest_command_prefix=request.pytest_command_prefix,
        )
    )
    return _write_new_manifest(
        request=request,
        source_git_sha=source_git_sha,
        nodes=nodes,
        manifest=manifest,
        plan=plan,
    )


def plan_description(plan: Plan, *, workers: int = 1, deadline_seconds: int = 0) -> str:
    metrics = capacity_metrics(
        plan,
        {index: workers for index in range(plan.options.shard_count)},
        deadline_seconds=deadline_seconds,
    )
    shard_loads = metrics["estimated_shard_load_ms"]
    batch_count = 0
    for unit in plan.units:
        batch_count += len(unit.batches)
    lines = [
        f"Collected nodes: {len(plan.nodes)}",
        f"Checkpoint batches: {batch_count}",
        f"Logical units: {len(plan.units)}",
        f"External shards: {plan.options.shard_count}",
        "Oversized: "
        f"{sum(batch.oversized for unit in plan.units for batch in unit.batches)} batches, "
        f"{sum(unit.oversized for unit in plan.units)} units",
    ]
    for index in range(plan.options.shard_count):
        load = shard_loads[str(index)]
        lines.append(f"  shard {index}: {load / 1000:.1f}s estimated")
    lines.append(
        f"Estimated makespan at {workers} worker(s) per shard: "
        f"{metrics['estimated_makespan_ms'] / 1000:.1f}s"
    )
    if deadline_seconds > 0:
        required = metrics["required_total_worker_slots"]
        lines.append(
            "Required worker slots for deadline: "
            + (str(required) if required is not None else "not feasible")
        )
        lines.append(
            "Required workers by shard: "
            + ", ".join(
                f"{index}={value if value is not None else 'not feasible'}"
                for index, value in metrics["required_workers_by_shard"].items()
            )
        )
    lines.append(
        f"Estimated process/warm-up overhead: "
        f"{metrics['total_estimated_overhead_ms'] / 1000:.1f}s"
    )
    lines.append(
        "Fallbacks: "
        + ", ".join(
            f"{name}={count}" for name, count in sorted(plan.fallback_counts.items())
        )
    )
    return "\n".join(lines)


def visible_devices() -> list[str | None]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        values = [value.strip() for value in configured.split(",") if value.strip()]
        if not values or values == ["-1"]:
            return [None]
        return values
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return [None]
    count = sum(line.startswith("GPU ") for line in result.stdout.splitlines())
    return [str(index) for index in range(count)] if count else [None]


def _synthesize_batch(
    junit_dir: Path,
    unit: Unit,
    batch: Any,
    metadata: SyntheticBatchMetadata,
) -> None:
    output = batch_xml_path(junit_dir, unit, batch)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and batch_is_final(junit_dir, unit, batch):
        return
    create_synthetic_batch_xml(
        output,
        batch.nodeids,
        metadata,
    )
    atomic_write_json(
        output.with_name(f"{batch.id}.meta.json"),
        {
            "attempt_id": metadata.attempt_id,
            "batch_id": batch.id,
            "unit_id": unit.id,
            "shard_index": unit.shard_index,
            "synthetic": True,
            "timeout_policy": metadata.policy,
        },
    )


def _completed_node_count(junit_dir: Path, plan: Plan) -> int:
    """Count batches whose final XML and last-written metadata marker are visible."""

    completed = 0
    for unit in plan.units:
        for batch in unit.batches:
            xml_path = batch_xml_path(junit_dir, unit, batch)
            metadata_path = xml_path.with_name(f"{batch.id}.meta.json")
            if xml_path.exists() and metadata_path.exists():
                completed += len(batch.nodeids)
    return completed


def _live_attempt_leases(attempt_path: Path) -> list[Path]:
    root = attempt_path / "leases"
    if not root.exists():
        return []
    return [path for path in root.glob("*.json") if lease_is_live(path)]


def _attempt_is_eligible(
    attempt_path: Path,
    plan: Plan,
    summary: RunSummary,
    *,
    explicit_fan_in: bool = False,
) -> bool:
    attempt = load_attempt(attempt_path)
    deadline_at = attempt.get("deadline_at")
    deadline_reached = deadline_at is not None and time.time() >= deadline_at
    done_shards = len(list((attempt_path / "shards").glob("shard-*.done.json")))
    all_shards_done = done_shards >= plan.options.shard_count
    return (
        explicit_fan_in
        or deadline_reached
        or all_shards_done
        or bool(summary["complete"])
    )


def _attempt_infrastructure_errors(attempt_path: Path) -> list[str]:
    errors: list[str] = []
    for marker in sorted((attempt_path / "shards").glob("shard-*.done.json")):
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("completion marker is not an object")
            marker_errors = value.get("infrastructure_errors", [])
            if not isinstance(marker_errors, list):
                raise TypeError("infrastructure_errors is not a list")
            errors.extend(f"{marker.stem}: {error}" for error in marker_errors)
        except (OSError, TypeError, json.JSONDecodeError) as error:
            errors.append(f"{marker.stem}: invalid shard completion marker: {error}")
    return errors


def finalize_attempt(
    *,
    junit_dir: Path,
    plan: Plan,
    attempt_path: Path,
    explicit_fan_in: bool = False,
) -> RunSummary:
    with state_lock(junit_dir):
        closed_path = attempt_path / "closed.json"
        if closed_path.exists():
            return publish_summary_under_lock(junit_dir, plan)
        attempt = load_attempt(attempt_path)
        summary = publish_summary_under_lock(junit_dir, plan)
        if not _attempt_is_eligible(
            attempt_path,
            plan,
            summary,
            explicit_fan_in=explicit_fan_in,
        ) or _live_attempt_leases(attempt_path):
            return summary
        policy = attempt["settings"]["timeout_policy"]
        deadline_at = attempt.get("deadline_at")
        deadline_reached = deadline_at is not None and time.time() >= deadline_at
        infrastructure_errors = _attempt_infrastructure_errors(attempt_path)
        if (
            policy in {"skip", "fail"}
            and (deadline_reached or explicit_fan_in)
            and not infrastructure_errors
            and not summary["complete"]
        ):
            for unit in plan.units:
                for batch in unit.batches:
                    if not batch_is_final(junit_dir, unit, batch):
                        _synthesize_batch(
                            junit_dir,
                            unit,
                            batch,
                            SyntheticBatchMetadata(
                                policy=policy,
                                batch_id=batch.id,
                                unit_id=unit.id,
                                shard_index=unit.shard_index,
                                attempt_id=attempt["id"],
                                profile=plan.options.profile,
                            ),
                        )
            summary = publish_summary_under_lock(junit_dir, plan)
        closure = {
            "ended_at": time.time(),
            "complete": summary["complete"],
            "timeout_policy": policy,
            "infrastructure_error": bool(infrastructure_errors),
            "infrastructure_errors": infrastructure_errors,
            "pending_nodes": len(summary["pending_nodes"]),
            "reason": (
                "deadline"
                if deadline_reached
                else "explicit-fan-in"
                if explicit_fan_in
                else "all-work-finished"
            ),
        }
        atomic_write_json(closed_path, closure)
    return publish_summary(junit_dir, plan)


class _LeaseHeartbeat:
    def __init__(self, attempt: AttemptRecord, shard_index: int) -> None:
        self.attempt = attempt
        self.shard_index = shard_index
        self.paths: dict[Path, str] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.failed = threading.Event()
        self.failure: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def add(self, path: Path, worker: str) -> None:
        self.raise_if_failed()
        with self.lock:
            self.paths[path] = worker
        self._renew(path, worker)
        self.raise_if_failed()

    def remove(self, path: Path) -> None:
        with self.lock:
            self.paths.pop(path, None)
        path.unlink(missing_ok=True)

    def _renew(self, path: Path, worker: str) -> None:
        atomic_write_json(
            path,
            lease_value(
                attempt_id=self.attempt["id"],
                shard_index=self.shard_index,
                worker=worker,
            ),
        )

    def _run(self) -> None:
        while not self.stop.wait(_LEASE_HEARTBEAT_SECONDS):
            try:
                with self.lock:
                    for path, worker in self.paths.items():
                        self._renew(path, worker)
            except Exception as error:
                with self.lock:
                    self.failure = error
                self.failed.set()
                self.stop.set()
                return

    def raise_if_failed(self) -> None:
        if not self.failed.is_set():
            return
        with self.lock:
            failure = self.failure
        raise RunnerStateError(f"lease heartbeat failed: {failure}") from failure

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        with self.lock:
            paths = list(self.paths)
            self.paths.clear()
        for path in paths:
            path.unlink(missing_ok=True)
        self.raise_if_failed()


@dataclass(frozen=True)
class _UnitTimer:
    prior_elapsed: float
    started_monotonic: float

    def elapsed(self) -> float:
        return self.prior_elapsed + (time.monotonic() - self.started_monotonic)


@dataclass
class _ExecutionGate:
    condition: threading.Condition = field(default_factory=threading.Condition)
    regular_active: int = 0
    solo_active: bool = False
    solo_waiting: int = 0

    def acquire(self, *, solo: bool) -> None:
        with self.condition:
            if solo:
                self.solo_waiting += 1
                try:
                    self.condition.wait_for(
                        lambda: not self.solo_active and self.regular_active == 0
                    )
                    self.solo_active = True
                finally:
                    self.solo_waiting -= 1
                return
            self.condition.wait_for(
                lambda: not self.solo_active and self.solo_waiting == 0
            )
            self.regular_active += 1

    def release(self, *, solo: bool) -> None:
        with self.condition:
            if solo:
                self.solo_active = False
            else:
                self.regular_active -= 1
            self.condition.notify_all()


def _next_worker_unit(
    work_by_worker: list[queue.Queue[Unit]],
    worker_index: int,
) -> tuple[queue.Queue[Unit], Unit]:
    preferred = work_by_worker[worker_index]
    try:
        return preferred, preferred.get_nowait()
    except queue.Empty:
        pass
    donors = sorted(
        (
            (work.qsize(), index, work)
            for index, work in enumerate(work_by_worker)
            if index != worker_index
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for _, _, work in donors:
        try:
            return work, work.get_nowait()
        except queue.Empty:
            continue
    raise queue.Empty


@dataclass
class _ShardExecutor:
    repo_root: Path
    junit_dir: Path
    plan: Plan
    execution: ExecutionSettings
    attempt_path: Path
    attempt: AttemptRecord
    devices: list[str | None]
    heartbeat: _LeaseHeartbeat
    solo_sources: frozenset[str]
    work_by_worker: list[queue.Queue[Unit]] = field(default_factory=list)
    execution_gate: _ExecutionGate = field(default_factory=_ExecutionGate)
    infrastructure_errors: list[str] = field(default_factory=list)
    interruption_reasons: set[str] = field(default_factory=set)
    errors_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_workers: threading.Event = field(default_factory=threading.Event)

    def add_units(self, units: list[Unit]) -> None:
        pending = []
        for unit in units:
            timed_out = (self.attempt_path / "timed-out" / f"{unit.id}.json").exists()
            if not timed_out and any(
                not batch_is_final(self.junit_dir, unit, batch)
                for batch in unit.batches
            ):
                pending.append(unit)
        self.work_by_worker = []
        for worker_units in source_affine_unit_bins(pending, self.execution.workers):
            work: queue.Queue[Unit] = queue.Queue()
            for unit in sorted(
                worker_units,
                key=lambda item: (-item.estimated_ms, item.id.encode("utf-8")),
            ):
                work.put(unit)
            self.work_by_worker.append(work)

    def run(self) -> None:
        with ThreadPoolExecutor(
            max_workers=self.execution.workers,
            thread_name_prefix="test-worker",
        ) as executor:
            futures = {
                executor.submit(self._worker, index): index
                for index in range(self.execution.workers)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                except Exception as error:
                    self.stop_workers.set()
                    self._record_infrastructure_error(
                        f"worker-{index}: {type(error).__name__}: {error}"
                    )

    def _record_infrastructure_error(self, diagnostic: str) -> None:
        with self.errors_lock:
            self.infrastructure_errors.append(diagnostic)

    def _worker(self, index: int) -> None:
        while not self.stop_workers.is_set():
            try:
                work, unit = _next_worker_unit(self.work_by_worker, index)
            except queue.Empty:
                return
            solo = self._unit_is_solo(unit)
            self.execution_gate.acquire(solo=solo)
            try:
                self._execute_unit(index, unit, solo=solo)
            finally:
                self.execution_gate.release(solo=solo)
                work.task_done()

    def _unit_is_solo(self, unit: Unit) -> bool:
        return any(batch.source_file in self.solo_sources for batch in unit.batches)

    def _execute_unit(self, worker_index: int, unit: Unit, *, solo: bool) -> None:
        claim_path = claims_dir(self.junit_dir) / f"{unit.id}.json"
        prior_elapsed = recover_unit_elapsed(
            self.attempt_path,
            unit.id,
            stale_claim_path=claim_path,
        )
        self.heartbeat.add(claim_path, f"worker-{worker_index}:{unit.id}")
        timer = _UnitTimer(prior_elapsed, time.monotonic())
        write_unit_elapsed(
            self.attempt_path,
            unit.id,
            elapsed_seconds=prior_elapsed,
            active_started_at=time.time(),
        )
        try:
            completed = self._execute_unit_batches(worker_index, unit, timer, solo=solo)
            if completed:
                self._report_unit_complete(worker_index, unit)
        finally:
            write_unit_elapsed(
                self.attempt_path,
                unit.id,
                elapsed_seconds=timer.elapsed(),
                active_started_at=None,
            )
            self.heartbeat.remove(claim_path)

    def _execute_unit_batches(
        self,
        worker_index: int,
        unit: Unit,
        timer: _UnitTimer,
        *,
        solo: bool,
    ) -> bool:
        for batch_position, batch in enumerate(unit.batches):
            if batch_is_final(self.junit_dir, unit, batch):
                continue
            remaining, timeout_reason = self._remaining_budget(timer)
            if remaining is not None and remaining <= 0:
                status = "timeout"
            else:
                result = execute_batch(
                    BatchExecutionRequest(
                        repo_root=self.repo_root,
                        junit_dir=self.junit_dir,
                        unit=unit,
                        batch=batch,
                        attempt_id=self.attempt["id"],
                        profile=self.plan.options.profile,
                        timeout_seconds=remaining,
                        timeout_reason=timeout_reason,
                        grace_seconds=self.execution.attempt.timeout_grace_seconds,
                        worker_index=worker_index,
                        device=None if solo else self.devices[worker_index],
                        monitor_memory=self.execution.monitor_memory,
                        memory_interval=self.execution.memory_interval,
                        pytest_command_prefix=self.execution.pytest_command_prefix,
                        abort_event=self.heartbeat.failed,
                    )
                )
                self.heartbeat.raise_if_failed()
                status = result.status
                if status == "infrastructure":
                    self._record_infrastructure_error(
                        f"{batch.id}: {result.diagnostic}"
                    )
            if status == "finalized":
                continue
            if status == "infrastructure":
                return False
            completed = self._handle_timeout(
                unit,
                batch_position,
                timer,
                timeout_reason or "timeout",
            )
            if completed:
                return True
            return False
        return True

    def _report_unit_complete(self, worker_index: int, unit: Unit) -> None:
        outcomes: Counter[str] = Counter()
        for batch in unit.batches:
            path = batch_xml_path(self.junit_dir, unit, batch)
            batch_outcomes, diagnostics = finalized_batch_outcomes(path, batch.nodeids)
            if diagnostics:
                raise RunnerStateError(
                    f"completed unit {unit.id} has invalid batch {batch.id}: "
                    + "; ".join(diagnostics)
                )
            outcomes.update(batch_outcomes)
        completed_nodes = _completed_node_count(self.junit_dir, self.plan)
        write_console(
            f"PYTEST UNIT COMPLETE worker={worker_index} unit={unit.id} "
            f"finalized={sum(outcomes.values())} "
            f"passed={outcomes['passed']} failed={outcomes['failed']} "
            f"skipped={outcomes['skipped']} unknown={outcomes['unknown']} "
            f"completed_nodes={completed_nodes}/{len(self.plan.nodes)}"
        )

    def _remaining_budget(self, timer: _UnitTimer) -> tuple[float | None, str | None]:
        limits: list[tuple[float, str]] = []
        unit_timeout = self.execution.attempt.unit_timeout_seconds
        if unit_timeout > 0:
            limits.append((unit_timeout - timer.elapsed(), "unit-timeout"))
        deadline_at = self.attempt.get("deadline_at")
        if deadline_at is not None:
            limits.append((deadline_at - time.time(), "attempt-deadline"))
        return min(limits, key=lambda value: value[0]) if limits else (None, None)

    def _handle_timeout(
        self,
        unit: Unit,
        batch_position: int,
        timer: _UnitTimer,
        reason: str,
    ) -> bool:
        with self.errors_lock:
            self.interruption_reasons.add(reason)
        timeout = self.execution.attempt.unit_timeout_seconds
        unit_timeout = timeout > 0 and timer.elapsed() >= timeout
        deadline_at = self.attempt.get("deadline_at")
        deadline_timeout = deadline_at is not None and time.time() >= deadline_at
        if not unit_timeout or deadline_timeout:
            return False
        policy = self.execution.attempt.timeout_policy
        atomic_write_json(
            self.attempt_path / "timed-out" / f"{unit.id}.json",
            {
                "unit_id": unit.id,
                "timed_out_at": time.time(),
                "elapsed_seconds": timer.elapsed(),
                "timeout_policy": policy,
            },
        )
        if policy not in {"skip", "fail"}:
            return False
        for pending in unit.batches[batch_position:]:
            if not batch_is_final(self.junit_dir, unit, pending):
                _synthesize_batch(
                    self.junit_dir,
                    unit,
                    pending,
                    SyntheticBatchMetadata(
                        policy=policy,
                        batch_id=pending.id,
                        unit_id=unit.id,
                        shard_index=unit.shard_index,
                        attempt_id=self.attempt["id"],
                        profile=self.plan.options.profile,
                    ),
                )
        return True


@dataclass
class _ShardLeases:
    heartbeat: _LeaseHeartbeat
    process: Path
    workers: list[Path]

    def close(self) -> None:
        for path in self.workers:
            self.heartbeat.remove(path)
        self.heartbeat.remove(self.process)
        self.heartbeat.close()


def _claim_shard_leases(
    junit_dir: Path,
    attempt_path: Path,
    attempt: AttemptRecord,
    execution: ExecutionSettings,
) -> _ShardLeases:
    lease_root = attempt_path / "leases"
    lease_root.mkdir(parents=True, exist_ok=True)
    process_lease = lease_root / f"shard-{execution.shard_index:04d}.json"
    with state_lock(junit_dir):
        if lease_is_live(process_lease):
            raise RunnerStateError(
                f"external shard {execution.shard_index} already has a live owner"
            )
        atomic_write_json(
            process_lease,
            lease_value(
                attempt_id=attempt["id"],
                shard_index=execution.shard_index,
                worker="coordinator",
            ),
        )
    heartbeat = _LeaseHeartbeat(attempt, execution.shard_index)
    heartbeat.add(process_lease, "coordinator")
    worker_leases = [
        lease_root / f"shard-{execution.shard_index:04d}-worker-{index:03d}.json"
        for index in range(execution.workers)
    ]
    for index, path in enumerate(worker_leases):
        heartbeat.add(path, f"worker-{index}")
    heartbeat.start()
    return _ShardLeases(heartbeat, process_lease, worker_leases)


def _existing_shard_result(
    *,
    junit_dir: Path,
    plan: Plan,
    execution: ExecutionSettings,
    attempt_path: Path,
    deadline_clock: DeadlineClock,
) -> int | None:
    shard_done = (
        attempt_path / "shards" / f"shard-{execution.shard_index:04d}.done.json"
    )
    if not shard_done.exists():
        return None
    summary = finalize_attempt(
        junit_dir=junit_dir,
        plan=plan,
        attempt_path=attempt_path,
    )
    shard_summary = summary["shards"][str(execution.shard_index)]
    _report_shard_status(
        summary,
        execution.shard_index,
        state="completed" if shard_summary["complete"] else "incomplete",
        deadline_clock=deadline_clock,
    )
    return exit_code_for_summary(summary)


def _report_shard_result(
    shard_executor: _ShardExecutor,
    deadline_clock: DeadlineClock,
) -> int:
    junit_dir = shard_executor.junit_dir
    plan = shard_executor.plan
    execution = shard_executor.execution
    attempt_path = shard_executor.attempt_path
    summary = finalize_attempt(
        junit_dir=junit_dir,
        plan=plan,
        attempt_path=attempt_path,
    )
    if shard_executor.infrastructure_errors:
        for error in shard_executor.infrastructure_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        _report_shard_status(
            summary,
            execution.shard_index,
            state="failed",
            deadline_clock=deadline_clock,
            reason="infrastructure-error",
        )
        return 3
    shard_summary = summary["shards"][str(execution.shard_index)]
    if shard_summary["complete"]:
        state, reason = "completed", None
    elif shard_executor.interruption_reasons:
        state = "interrupted"
        reason = ",".join(sorted(shard_executor.interruption_reasons))
    else:
        state, reason = "incomplete", None
    _report_shard_status(
        summary,
        execution.shard_index,
        state=state,
        deadline_clock=deadline_clock,
        reason=reason,
    )
    return exit_code_for_summary(summary)


def execute_shard(
    *,
    repo_root: Path,
    junit_dir: Path,
    plan: Plan,
    execution: ExecutionSettings,
    operation_started_at: float,
) -> int:
    if not 0 <= execution.shard_index < plan.options.shard_count:
        raise RunnerStateError(
            f"shard index {execution.shard_index} is outside [0, {plan.options.shard_count})"
        )
    initial_summary = publish_summary(junit_dir, plan)
    deadline_clock = DeadlineClock(
        started_at=operation_started_at,
        limit_seconds=execution.attempt.deadline_seconds,
    )
    if initial_summary["complete"]:
        attempts = list_attempts(junit_dir)
        if attempts and not (attempts[-1] / "closed.json").exists():
            deadline_clock = DeadlineClock.from_attempt(load_attempt(attempts[-1]))
            initial_summary = finalize_attempt(
                junit_dir=junit_dir,
                plan=plan,
                attempt_path=attempts[-1],
            )
        _report_shard_status(
            initial_summary,
            execution.shard_index,
            state="completed",
            deadline_clock=deadline_clock,
        )
        return exit_code_for_summary(initial_summary)
    devices = visible_devices()
    if execution.workers <= 0:
        raise RunnerStateError("workers must be positive")
    if execution.workers > len(devices):
        raise RunnerStateError(
            f"requested {execution.workers} workers but only {len(devices)} visible GPU slot(s)"
        )
    attempt_path, attempt = create_or_join_attempt(
        junit_dir, execution.attempt, started_at=operation_started_at
    )
    deadline_clock = DeadlineClock.from_attempt(attempt)
    existing_result = _existing_shard_result(
        junit_dir=junit_dir,
        plan=plan,
        execution=execution,
        attempt_path=attempt_path,
        deadline_clock=deadline_clock,
    )
    if existing_result is not None:
        return existing_result
    leases = _claim_shard_leases(junit_dir, attempt_path, attempt, execution)

    atomic_write_json(
        attempt_path / "shards" / f"shard-{execution.shard_index:04d}.settings.json",
        {
            "shard_index": execution.shard_index,
            "workers": execution.workers,
            "recorded_at": time.time(),
        },
    )

    shard_units = sorted(
        (unit for unit in plan.units if unit.shard_index == execution.shard_index),
        key=lambda unit: (-unit.estimated_ms, unit.id.encode("utf-8")),
    )
    shard_executor = _ShardExecutor(
        repo_root=repo_root,
        junit_dir=junit_dir,
        plan=plan,
        execution=execution,
        attempt_path=attempt_path,
        attempt=attempt,
        devices=devices,
        heartbeat=leases.heartbeat,
        solo_sources=frozenset(node.source_file for node in plan.nodes if node.solo),
    )
    shard_executor.add_units(shard_units)
    try:
        shard_executor.run()
    finally:
        try:
            leases.close()
        except RunnerStateError as error:
            shard_executor._record_infrastructure_error(str(error))

    shard_done = (
        attempt_path / "shards" / f"shard-{execution.shard_index:04d}.done.json"
    )
    atomic_write_json(
        shard_done,
        {
            "finished_at": time.time(),
            "infrastructure_errors": shard_executor.infrastructure_errors,
        },
    )
    return _report_shard_result(
        shard_executor,
        deadline_clock,
    )


def finalize_latest(junit_dir: Path, plan: Plan, *, wait: bool = True) -> int:
    attempts = list_attempts(junit_dir)
    if not attempts:
        return exit_code_for_summary(publish_summary(junit_dir, plan))
    attempt_path = attempts[-1]
    if wait and not (attempt_path / "closed.json").exists():
        attempt = load_attempt(attempt_path)
        grace = int(attempt["settings"]["timeout_grace_seconds"])
        wait_until = time.time() + grace + 31
        while _live_attempt_leases(attempt_path) and time.time() < wait_until:
            time.sleep(1)
    summary = finalize_attempt(
        junit_dir=junit_dir,
        plan=plan,
        attempt_path=attempt_path,
        explicit_fan_in=True,
    )
    if _live_attempt_leases(attempt_path):
        raise RunnerStateError("cannot finalize while worker leases are still live")
    return exit_code_for_summary(summary)
