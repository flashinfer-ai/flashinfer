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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .estimates import EstimateBook
from .io import atomic_write_json
from .junit import (
    SyntheticBatchMetadata,
    create_synthetic_batch_xml,
    finalized_batch_outcomes,
    validate_batch_xml,
)
from .models import RUNTIME_TIMING_PROFILE, CollectedNode, Plan, PlanningOptions, Unit
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
    exit_code_for_shard,
    publish_summary,
    publish_summary_under_lock,
)
from .workers import BatchExecutionRequest, execute_batch, write_console


_COLLECTION_HEARTBEAT_SECONDS = 30.0
MAX_TEST_PATHS = 16
_LEASE_HEARTBEAT_SECONDS = 10.0
_LEASE_CLOSE_SECONDS = 2.0
_CONTROLLER_PROGRESS_SECONDS = 60.0

# The SM90 pull-style, SM120 swap-AB, and SM100 MegaMoE drops all import
# vendored modules such as ``common`` as top-level packages, so no two of them
# can be imported by one pytest collection process. Keep the SM100 family in
# the primary scope and give each smaller arch family its own partition. Long
# term, namespace the vendored trees and use package-relative imports; once
# they can coexist in ``sys.modules``, remove these collection partitions.
_COLLECTION_ISOLATION_GROUPS = (
    (
        "sm90-pull-style-cutedsl-megakernel",
        ("test_moe_ep_sm90_pull_*_mega_multirank.py",),
    ),
    (
        "sm120-swapab-cutedsl-megakernel",
        ("test_moe_ep_sm120_*_mega_multirank.py",),
    ),
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
    test_paths: tuple[Path, ...]
    sanity_test: bool
    sample_rate: int
    sample_offset: int

    @property
    def test_path(self) -> Path:
        return self.test_paths[0]

    def display_test_path(self) -> str:
        return " ".join(str(path) for path in self.test_paths)

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
        junit_dir / "shards",
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
    timing_profile: str = RUNTIME_TIMING_PROFILE


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
    duration_estimates: Path | None = None
    overhead_estimates: Path | None = None


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


def _estimate_checksums(
    duration: Path | None, overhead: Path | None
) -> dict[str, str | None]:
    for label, path in (("duration", duration), ("overhead", overhead)):
        if path is not None and not path.is_file():
            raise RunnerStateError(f"{label} estimate file does not exist: {path}")
    return {
        "duration": sha256_file(duration) if duration is not None else None,
        "overhead": sha256_file(overhead) if overhead is not None else None,
    }


def _as_test_paths(test_path: Path | Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(test_path, Path):
        return (test_path,)
    return tuple(test_path)


def _path_is_under(child: Path, parent: Path) -> bool:
    if not parent.is_dir():
        return False
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return child.resolve() != parent.resolve()


def collapse_test_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    if len(unique) > MAX_TEST_PATHS:
        raise RunnerStateError(
            f"too many test paths ({len(unique)}); maximum is {MAX_TEST_PATHS}"
        )
    kept = [
        path
        for path in unique
        if not any(_path_is_under(path, other) for other in unique)
    ]
    return tuple(kept)


def _wait_for_collection(
    process: subprocess.Popen[Any],
    *,
    test_path: Path | str,
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


def _isolated_collection_groups(
    test_paths: Sequence[Path],
) -> list[tuple[str, list[Path]]]:
    groups = []
    for name, patterns in _COLLECTION_ISOLATION_GROUPS:
        matches: set[Path] = set()
        for test_path in test_paths:
            for pattern in patterns:
                if test_path.is_file():
                    if test_path.match(pattern):
                        matches.add(test_path)
                else:
                    matches.update(
                        path for path in test_path.rglob(pattern) if path.is_file()
                    )
        if matches:
            groups.append(
                (name, sorted(matches, key=lambda path: str(path).encode("utf-8")))
            )
    return groups


def _pytest_root(repo_root: Path, test_path: Path | Sequence[Path]) -> Path:
    """Return the stable pytest root used for collection and execution."""
    resolved_repo = repo_root.resolve()
    for path in _as_test_paths(test_path):
        resolved_test = path.resolve()
        try:
            resolved_test.relative_to(resolved_repo)
        except ValueError:
            return resolved_test if resolved_test.is_dir() else resolved_test.parent
    return resolved_repo


def _collect_partition(
    repo_root: Path,
    test_paths: Sequence[Path],
    targets: list[Path],
    ignored_paths: list[Path],
    timeout_seconds: float | None,
    grace_seconds: float,
    directory: Path,
    partition_index: int,
    partition_name: str,
    *,
    allow_empty: bool,
) -> list[dict[str, Any]]:
    output = directory / f"collection-{partition_index:02d}.json"
    log_path = directory / f"collection-{partition_index:02d}.log"
    pytest_root = _pytest_root(repo_root, test_paths)
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        f"--rootdir={pytest_root}",
        "--collect-only",
        "-q",
        "-q",
        "--continue-on-collection-errors",
        "-p",
        "scripts.test_sharding.pytest_plugin",
        f"--flashinfer-collection-json={output}",
        *(f"--ignore={path}" for path in ignored_paths),
        *(str(path) for path in targets),
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
            test_path=" ".join(str(path) for path in test_paths),
            timeout_seconds=timeout_seconds,
            grace_seconds=grace_seconds,
        )
    if allow_empty and process.returncode == 5:
        return []
    if process.returncode != 0 or not output.exists():
        diagnostic = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
        raise RunnerStateError(
            f"pytest collection failed for {partition_name} "
            f"with exit code {process.returncode}:\n{diagnostic}"
        )
    try:
        value = json.loads(output.read_text(encoding="utf-8"))
        nodes = value["nodes"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise RunnerStateError(
            f"invalid collection metadata for {partition_name}: {error}"
        ) from error
    if not nodes and not allow_empty:
        raise RunnerStateError(f"pytest collected no tests for {partition_name}")
    return nodes


def _collect_nodes(
    repo_root: Path,
    test_path: Path | Sequence[Path],
    timeout_seconds: float | None,
    grace_seconds: float,
) -> list[dict[str, Any]]:
    started_at = time.monotonic()
    test_paths = _as_test_paths(test_path)
    display_test_path = " ".join(str(path) for path in test_paths)
    isolated_groups = _isolated_collection_groups(test_paths)
    isolated_paths = [path for _, paths in isolated_groups for path in paths]
    primary_targets = [path for path in test_paths if path not in isolated_paths]

    def remaining_timeout() -> float | None:
        if timeout_seconds is None:
            return None
        return max(0.0, timeout_seconds - (time.monotonic() - started_at))

    try:
        with tempfile.TemporaryDirectory(
            prefix="flashinfer-test-collection-"
        ) as temporary_directory:
            directory = Path(temporary_directory)
            partitions = []
            if primary_targets:
                partitions.append(
                    _collect_partition(
                        repo_root,
                        test_paths,
                        primary_targets,
                        isolated_paths,
                        remaining_timeout(),
                        grace_seconds,
                        directory,
                        0,
                        "primary test scope",
                        allow_empty=bool(isolated_paths),
                    )
                )
            for partition_index, (name, paths) in enumerate(isolated_groups, start=1):
                partitions.append(
                    _collect_partition(
                        repo_root,
                        test_paths,
                        paths,
                        [],
                        remaining_timeout(),
                        grace_seconds,
                        directory,
                        partition_index,
                        name,
                        allow_empty=False,
                    )
                )
    except CollectionTimeoutError as error:
        error.elapsed_seconds = time.monotonic() - started_at
        raise

    nodes = []
    seen_nodeids = set()
    for partition in partitions:
        for node in partition:
            nodeid = node["nodeid"]
            if nodeid in seen_nodeids:
                raise RunnerStateError(
                    f"duplicate pytest node ID across collection partitions: {nodeid}"
                )
            seen_nodeids.add(nodeid)
            merged = dict(node)
            merged["order"] = len(nodes)
            nodes.append(merged)
    if not nodes:
        raise RunnerStateError(f"pytest collected no tests below {display_test_path}")
    return nodes


def _validate_selection(selection: SelectionSettings) -> None:
    missing = [path for path in selection.test_paths if not path.exists()]
    if missing:
        raise RunnerStateError(
            "test path does not exist: " + ", ".join(str(path) for path in missing)
        )
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
            long_running=bool(node.get("long_running", False)),
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
                test_paths=request.selection.test_paths,
                selection=request.selection.to_dict(),
                planning_options=request.planning.to_dict(),
                pytest_command_prefix=request.pytest_command_prefix,
                estimate_files=_estimate_checksums(
                    request.duration_estimates, request.overhead_estimates
                ),
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
    test_paths = selection.test_paths
    _validate_selection(selection)
    junit_dir.mkdir(parents=True, exist_ok=True)
    source_git_sha = source_git_sha_from_env()
    selection_value = selection.to_dict()
    estimate_files = _estimate_checksums(
        request.duration_estimates, request.overhead_estimates
    )
    existing = load_manifest(junit_dir)
    existing_plan: Plan | None = None
    if existing is not None:
        verify_manifest(
            existing,
            source_git_sha=source_git_sha,
            test_path=test_paths[0],
            test_paths=test_paths,
            selection=selection_value,
            planning_options=planning.to_dict(),
            pytest_command_prefix=request.pytest_command_prefix,
            estimate_files=estimate_files,
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
            test_paths,
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
    estimates = EstimateBook.from_files(
        request.duration_estimates,
        request.overhead_estimates,
    )
    plan = build_plan(nodes, estimates, planning)
    manifest = build_manifest(
        ManifestBuild(
            repo_root=repo_root,
            test_path=test_paths[0],
            test_paths=test_paths,
            source_git_sha=source_git_sha,
            plan=plan,
            selection=selection_value,
            estimate_files=estimate_files,
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
                                profile=RUNTIME_TIMING_PROFILE,
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
        self.started = False

    def add(self, path: Path, worker: str) -> None:
        self.raise_if_failed()
        with self.lock:
            self.paths[path] = worker
        self._renew(path, worker)
        self.raise_if_failed()

    def remove(self, path: Path) -> None:
        with self.lock:
            worker = self.paths.pop(path, None)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            if worker is not None:
                with self.lock:
                    self.paths[path] = worker
            raise

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
        self.started = True

    def close(self) -> None:
        self.stop.set()
        if self.started:
            self.thread.join(timeout=_LEASE_CLOSE_SECONDS)
            if self.thread.is_alive():
                raise RunnerStateError(
                    "lease heartbeat did not stop within "
                    f"{_LEASE_CLOSE_SECONDS:g} seconds"
                )
        with self.lock:
            paths = list(self.paths)
            self.paths.clear()
        errors: list[str] = []
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                errors.append(f"cannot remove lease {path}: {error}")
        try:
            self.raise_if_failed()
        except RunnerStateError as error:
            errors.append(str(error))
        if errors:
            raise RunnerStateError("; ".join(errors))


@dataclass(frozen=True)
class _UnitTimer:
    prior_elapsed: float
    started_monotonic: float

    def elapsed(self) -> float:
        return self.prior_elapsed + (time.monotonic() - self.started_monotonic)


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
class _ShardProgress:
    shard_index: int
    planned: int
    finalized: int
    outcomes: Counter[str]
    synthetic: int = 0
    finalized_this_invocation: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            name=f"shard-{self.shard_index}-progress",
            daemon=True,
        )
        self.thread.start()
        self.report()

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.report()

    def _run(self) -> None:
        while not self.stop.wait(_CONTROLLER_PROGRESS_SECONDS):
            self.report()

    def record(self, outcomes: Counter[str], *, synthetic: int = 0) -> None:
        count = sum(outcomes.values())
        with self.lock:
            self.finalized += count
            self.finalized_this_invocation += count
            self.outcomes.update(outcomes)
            self.synthetic += synthetic
        self.report()

    def report(self) -> None:
        with self.lock:
            finalized = self.finalized
            current = self.finalized_this_invocation
            outcomes = Counter(self.outcomes)
            synthetic = self.synthetic
        write_console(
            f"PROGRESS shard={self.shard_index} finalized={finalized}/{self.planned} "
            f"finalized_this_invocation={current} passed={outcomes['passed']} "
            f"failed={outcomes['failed']} skipped={outcomes['skipped']} "
            f"unknown={outcomes['unknown']} pending={max(0, self.planned - finalized)} "
            f"synthetic={synthetic}"
        )


@dataclass
class _ShardExecutor:
    repo_root: Path
    pytest_root: Path
    junit_dir: Path
    plan: Plan
    execution: ExecutionSettings
    attempt_path: Path
    attempt: AttemptRecord
    devices: list[str | None]
    heartbeat: _LeaseHeartbeat
    solo_sources: frozenset[str]
    long_running_sources: frozenset[str]
    progress: _ShardProgress
    long_work_by_worker: list[queue.Queue[Unit]] = field(default_factory=list)
    normal_work_by_worker: list[queue.Queue[Unit]] = field(default_factory=list)
    solo_units: list[Unit] = field(default_factory=list)
    non_solo_units: list[Unit] = field(default_factory=list)
    all_non_solo_units: list[Unit] = field(default_factory=list)
    infrastructure_errors: list[str] = field(default_factory=list)
    interruption_reasons: set[str] = field(default_factory=set)
    errors_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_workers: threading.Event = field(default_factory=threading.Event)

    def add_units(self, units: list[Unit]) -> None:
        self.all_non_solo_units = [
            unit for unit in units if not self._unit_is_solo(unit)
        ]
        pending = []
        for unit in units:
            timed_out = (self.attempt_path / "timed-out" / f"{unit.id}.json").exists()
            if not timed_out and any(
                not batch_is_final(self.junit_dir, unit, batch)
                for batch in unit.batches
            ):
                pending.append(unit)
        self.solo_units = sorted(
            (unit for unit in pending if self._unit_is_solo(unit)),
            key=lambda item: (-item.estimated_ms, item.id.encode("utf-8")),
        )
        self.non_solo_units = [unit for unit in pending if not self._unit_is_solo(unit)]
        self.long_work_by_worker = []
        self.normal_work_by_worker = []
        for worker_units in source_affine_unit_bins(
            self.non_solo_units, self.execution.workers
        ):
            long_work: queue.Queue[Unit] = queue.Queue()
            normal_work: queue.Queue[Unit] = queue.Queue()
            for unit in sorted(
                worker_units,
                key=lambda item: (-item.estimated_ms, item.id.encode("utf-8")),
            ):
                target = long_work if self._unit_is_long_running(unit) else normal_work
                target.put(unit)
            self.long_work_by_worker.append(long_work)
            self.normal_work_by_worker.append(normal_work)

    def run(self) -> None:
        self.progress.start()
        try:
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
            if self.stop_workers.is_set() or not self._non_solo_finalized():
                return
            self._run_solo_phase()
        finally:
            self.progress.close()

    def _record_infrastructure_error(self, diagnostic: str) -> None:
        with self.errors_lock:
            self.infrastructure_errors.append(diagnostic)

    def _worker(self, index: int) -> None:
        started = time.monotonic()
        stats: Counter[str] = Counter()
        device = self.devices[index]
        write_console(
            f"WORKER START worker={index} time={_timestamp()} "
            f"device={device if device is not None else 'unassigned'}"
        )
        try:
            while not self.stop_workers.is_set():
                try:
                    try:
                        work, unit = _next_worker_unit(self.long_work_by_worker, index)
                    except queue.Empty:
                        work, unit = _next_worker_unit(
                            self.normal_work_by_worker, index
                        )
                except queue.Empty:
                    return
                before = {
                    batch.id
                    for batch in unit.batches
                    if batch_is_final(self.junit_dir, unit, batch)
                }
                pending_nodes = sum(
                    len(batch.nodeids)
                    for batch in unit.batches
                    if batch.id not in before
                )
                self._report_worker_task(index, unit, pending_nodes, solo=False)
                try:
                    self._execute_unit(index, unit, solo=False)
                    stats.update(self._new_unit_outcomes(unit, before))
                finally:
                    work.task_done()
        finally:
            self._report_worker_end(index, started, stats, solo=False)

    def _run_solo_phase(self) -> None:
        if not self.solo_units:
            return
        started = time.monotonic()
        stats: Counter[str] = Counter()
        write_console(
            f"WORKER START worker=solo-0 time={_timestamp()} device=all-visible-gpus"
        )
        try:
            for unit in self.solo_units:
                if self.stop_workers.is_set():
                    return
                before = {
                    batch.id
                    for batch in unit.batches
                    if batch_is_final(self.junit_dir, unit, batch)
                }
                pending_nodes = sum(
                    len(batch.nodeids)
                    for batch in unit.batches
                    if batch.id not in before
                )
                self._report_worker_task("solo-0", unit, pending_nodes, solo=True)
                self._execute_unit(0, unit, solo=True)
                stats.update(self._new_unit_outcomes(unit, before))
        finally:
            self._report_worker_end("solo-0", started, stats, solo=True)

    def _report_worker_task(
        self, worker: int | str, unit: Unit, node_count: int, *, solo: bool
    ) -> None:
        source = unit.batches[0].source_file if unit.batches else "unknown"
        first_node = unit.batches[0].nodeids[0] if unit.batches else "unknown"
        write_console(
            f"WORKER TASK worker={worker} time={_timestamp()} solo={str(solo).lower()} "
            f"source={source} unit={unit.id} nodes={node_count} first_node={first_node}"
        )

    def _report_worker_end(
        self,
        worker: int | str,
        started: float,
        outcomes: Counter[str],
        *,
        solo: bool,
    ) -> None:
        handled = sum(outcomes.values())
        write_console(
            f"WORKER END worker={worker} time={_timestamp()} solo={str(solo).lower()} "
            f"elapsed={time.monotonic() - started:.3f}s handled={handled} "
            f"passed={outcomes['passed']} failed={outcomes['failed']} "
            f"skipped={outcomes['skipped']} unknown={outcomes['unknown']}"
        )

    def _new_unit_outcomes(
        self, unit: Unit, previously_final: set[str]
    ) -> Counter[str]:
        outcomes: Counter[str] = Counter()
        for batch in unit.batches:
            if batch.id in previously_final or not batch_is_final(
                self.junit_dir, unit, batch
            ):
                continue
            batch_outcomes, _ = finalized_batch_outcomes(
                batch_xml_path(self.junit_dir, unit, batch), batch.nodeids
            )
            outcomes.update(batch_outcomes)
        return outcomes

    def _unit_is_solo(self, unit: Unit) -> bool:
        return any(batch.source_file in self.solo_sources for batch in unit.batches)

    def _unit_is_long_running(self, unit: Unit) -> bool:
        return any(
            batch.source_file in self.long_running_sources for batch in unit.batches
        )

    def _non_solo_finalized(self) -> bool:
        return all(
            batch_is_final(self.junit_dir, unit, batch)
            for unit in self.all_non_solo_units
            for batch in unit.batches
        )

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
                        pytest_root=self.pytest_root,
                        junit_dir=self.junit_dir,
                        unit=unit,
                        batch=batch,
                        attempt_id=self.attempt["id"],
                        profile=self.execution.timing_profile,
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
                diagnostic = (
                    f"source={batch.source_file} batch={batch.id} "
                    f"nodes={len(batch.nodeids)} first_node={batch.nodeids[0]} "
                    f"cause={result.diagnostic}"
                )
                write_console(f"ERROR: {diagnostic}")
                self._record_infrastructure_error(diagnostic)
            if status == "finalized":
                self._record_batch_progress(unit, batch)
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

    def _record_batch_progress(self, unit: Unit, batch: Any) -> None:
        path = batch_xml_path(self.junit_dir, unit, batch)
        outcomes, diagnostics = finalized_batch_outcomes(path, batch.nodeids)
        if diagnostics:
            raise RunnerStateError(
                f"cannot record progress for {batch.source_file} {batch.id}: "
                + "; ".join(diagnostics)
            )
        validation = validate_batch_xml(path, batch.nodeids)
        self.progress.record(
            outcomes,
            synthetic=sum(case.synthetic for case in validation.cases),
        )

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
                        profile=self.execution.timing_profile,
                    ),
                )
                self._record_batch_progress(unit, pending)
        return True


def _claim_shard_leases(
    junit_dir: Path,
    attempt_path: Path,
    attempt: AttemptRecord,
    execution: ExecutionSettings,
) -> _LeaseHeartbeat:
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
    worker_leases = [
        lease_root / f"shard-{execution.shard_index:04d}-worker-{index:03d}.json"
        for index in range(execution.workers)
    ]
    try:
        heartbeat.add(process_lease, "coordinator")
        for index, path in enumerate(worker_leases):
            heartbeat.add(path, f"worker-{index}")
        heartbeat.start()
    except Exception as error:
        try:
            heartbeat.close()
        except Exception as cleanup_error:
            raise RunnerStateError(
                f"cannot claim shard leases: {error}; "
                f"lease rollback also failed: {cleanup_error}"
            ) from error
        raise
    return heartbeat


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
    return exit_code_for_shard(summary, execution.shard_index)


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
            print(f"ERROR: {error}")
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
    return exit_code_for_shard(summary, execution.shard_index)


def execute_shard(
    *,
    repo_root: Path,
    junit_dir: Path,
    plan: Plan,
    execution: ExecutionSettings,
    operation_started_at: float,
    test_path: Path | Sequence[Path],
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
        return exit_code_for_shard(initial_summary, execution.shard_index)
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
    lease_heartbeat = _claim_shard_leases(junit_dir, attempt_path, attempt, execution)
    shard_executor: _ShardExecutor | None = None

    try:
        atomic_write_json(
            attempt_path
            / "shards"
            / f"shard-{execution.shard_index:04d}.settings.json",
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
            pytest_root=_pytest_root(repo_root, test_path),
            junit_dir=junit_dir,
            plan=plan,
            execution=execution,
            attempt_path=attempt_path,
            attempt=attempt,
            devices=devices,
            heartbeat=lease_heartbeat,
            solo_sources=frozenset(
                node.source_file for node in plan.nodes if node.solo
            ),
            long_running_sources=frozenset(
                node.source_file for node in plan.nodes if node.long_running
            ),
            progress=_ShardProgress(
                shard_index=execution.shard_index,
                planned=initial_summary["shards"][str(execution.shard_index)][
                    "planned_nodes"
                ],
                finalized=initial_summary["shards"][str(execution.shard_index)][
                    "finalized_nodes"
                ],
                outcomes=Counter(
                    initial_summary["shards"][str(execution.shard_index)]["outcomes"]
                ),
                synthetic=sum(
                    int(row["synthetic"])
                    for row in initial_summary["sources"]
                    if int(row["shard_index"]) == execution.shard_index
                ),
            ),
        )
        shard_executor.add_units(shard_units)
        shard_executor.run()
    finally:
        primary_error = sys.exc_info()[1]
        try:
            lease_heartbeat.close()
        except Exception as cleanup_error:
            if shard_executor is not None:
                shard_executor._record_infrastructure_error(str(cleanup_error))
            if primary_error is not None:
                print(f"ERROR: lease cleanup also failed: {cleanup_error}", flush=True)
            elif shard_executor is None:
                raise

    assert shard_executor is not None

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
