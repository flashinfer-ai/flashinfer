from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .io import atomic_write_json, atomic_write_text
from .junit import annotate_batch_xml, validate_batch_xml
from .models import Batch, Unit
from .processes import terminate_process_group
from .progress import PYTEST_EVENT_PREFIX, decode_pytest_event
from .summary import batch_directory, batch_xml_path


@dataclass(frozen=True)
class BatchExecution:
    status: str
    diagnostic: str = ""


@dataclass(frozen=True)
class BatchExecutionRequest:
    repo_root: Path
    junit_dir: Path
    unit: Unit
    batch: Batch
    attempt_id: str
    profile: str
    timeout_seconds: float | None
    timeout_reason: str | None
    grace_seconds: int
    worker_index: int
    device: str | None
    monitor_memory: bool
    memory_interval: float
    pytest_command_prefix: tuple[str, ...] = ()
    abort_event: threading.Event | None = None


@dataclass
class _BatchProgress:
    completed: int = 0
    current_nodeid: str | None = None
    current_started_at: float | None = None
    last_function: str | None = None
    outcomes: Counter[str] = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start(self, nodeid: str, started_at: float) -> tuple[str, bool]:
        function = nodeid.split("[", 1)[0]
        with self._lock:
            self.current_nodeid = nodeid
            self.current_started_at = started_at
            should_print = function != self.last_function
            if should_print:
                self.last_function = function
        return function, should_print

    def finish(self, nodeid: str, outcome: str) -> None:
        with self._lock:
            self.completed += 1
            self.outcomes[outcome] += 1
            if self.current_nodeid == nodeid:
                self.current_nodeid = None
                self.current_started_at = None

    def current(self) -> tuple[str | None, float | None, int]:
        with self._lock:
            return self.current_nodeid, self.current_started_at, self.completed

    def totals(self) -> tuple[int, Counter[str]]:
        with self._lock:
            return self.completed, Counter(self.outcomes)


_CONSOLE_LOCK = threading.Lock()
_PROGRESS_INTERVAL_SECONDS = 30.0


def write_console(message: str) -> None:
    with _CONSOLE_LOCK:
        print(message, flush=True)


def _descendant_pids(root_pid: int) -> set[int]:
    pids = {root_pid}
    changed = True
    while changed:
        changed = False
        for status_path in Path("/proc").glob("[0-9]*/status"):
            try:
                content = status_path.read_text(encoding="utf-8")
                pid = int(status_path.parent.name)
                parent_line = next(
                    line for line in content.splitlines() if line.startswith("PPid:")
                )
                parent = int(parent_line.split()[1])
            except (OSError, StopIteration, ValueError):
                continue
            if parent in pids and pid not in pids:
                pids.add(pid)
                changed = True
    return pids


def _rss_mib(pids: set[int]) -> float:
    total_kib = 0
    for pid in pids:
        try:
            for line in (
                Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
            ):
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (OSError, ValueError):
            pass
    return total_kib / 1024


def _gpu_mib(pids: set[int]) -> float:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    total = 0.0
    for line in result.stdout.splitlines():
        try:
            pid_text, memory_text = line.split(",", 1)
            if int(pid_text.strip()) in pids:
                total += float(memory_text.strip())
        except ValueError:
            continue
    return total


def _monitor_memory(
    pid: int,
    stop: threading.Event,
    samples: list[tuple[float, float, float]],
    interval: float,
) -> None:
    while not stop.is_set():
        pids = _descendant_pids(pid)
        samples.append((time.time(), _rss_mib(pids), _gpu_mib(pids)))
        stop.wait(interval)


def _forward_pytest_output(
    stream: TextIO,
    log: TextIO,
    progress: _BatchProgress,
    *,
    worker_index: int,
    batch_id: str,
) -> None:
    for line in stream:
        event = decode_pytest_event(line)
        if event is None:
            log.write(line)
            log.flush()
            continue
        leading_output = line.split(PYTEST_EVENT_PREFIX, 1)[0]
        if leading_output:
            log.write(leading_output)
            log.flush()
        nodeid = str(event.get("nodeid", ""))
        if event.get("event") == "start":
            function, should_print = progress.start(
                nodeid, float(event.get("started_at", time.time()))
            )
            if should_print:
                write_console(
                    f"PYTEST START worker={worker_index} batch={batch_id} "
                    f"function={function} node={nodeid}"
                )
        elif event.get("event") == "finish":
            outcome = str(event.get("outcome", "unknown"))
            duration = float(event.get("duration_seconds", 0.0))
            progress.finish(nodeid, outcome)
            if outcome in {"failed", "unknown"}:
                write_console(
                    f"PYTEST RESULT worker={worker_index} batch={batch_id} "
                    f"outcome={outcome} duration={duration:.3f}s node={nodeid}"
                )


def _progress_heartbeat(
    stop: threading.Event,
    progress: _BatchProgress,
    *,
    worker_index: int,
    batch_id: str,
) -> None:
    while not stop.wait(_PROGRESS_INTERVAL_SECONDS):
        nodeid, started_at, completed = progress.current()
        if nodeid is None or started_at is None:
            continue
        write_console(
            f"PYTEST RUNNING worker={worker_index} batch={batch_id} "
            f"elapsed={max(0.0, time.time() - started_at):.1f}s "
            f"completed_in_batch={completed} node={nodeid}"
        )


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def _discard_temporary_artifacts(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _BatchArtifacts:
    final_xml: Path
    selection: Path
    temporary_xml: Path
    temporary_results: Path
    temporary_telemetry: Path
    log: Path

    @property
    def temporary(self) -> tuple[Path, ...]:
        return (
            self.temporary_xml,
            self.temporary_results,
            self.temporary_telemetry,
            self.selection,
        )


@dataclass(frozen=True)
class _ProcessOutcome:
    returncode: int
    launched_at: float
    exited_at: float
    timed_out: bool
    aborted: bool
    termination_signal: str
    samples: tuple[tuple[float, float, float], ...]
    progress: _BatchProgress


def _batch_artifacts(request: BatchExecutionRequest) -> _BatchArtifacts:
    directory = batch_directory(request.junit_dir, request.unit)
    directory.mkdir(parents=True, exist_ok=True)
    final_xml = batch_xml_path(request.junit_dir, request.unit, request.batch)
    batch_id = request.batch.id
    return _BatchArtifacts(
        final_xml=final_xml,
        selection=_temporary(directory / f"{batch_id}.selection.json"),
        temporary_xml=_temporary(final_xml),
        temporary_results=_temporary(directory / f"{batch_id}.results.json"),
        temporary_telemetry=_temporary(directory / f"{batch_id}.telemetry.json"),
        log=directory / f"{batch_id}.log",
    )


def _pytest_command(
    request: BatchExecutionRequest, artifacts: _BatchArtifacts
) -> list[str]:
    return [
        *request.pytest_command_prefix,
        sys.executable,
        "-m",
        "pytest",
        "--continue-on-collection-errors",
        "-p",
        "scripts.test_sharding.pytest_plugin",
        f"--flashinfer-node-file={artifacts.selection}",
        f"--flashinfer-result-json={artifacts.temporary_results}",
        f"--flashinfer-telemetry-json={artifacts.temporary_telemetry}",
        f"--junitxml={artifacts.temporary_xml}",
        request.batch.source_file,
    ]


def _pytest_environment(request: BatchExecutionRequest) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{request.repo_root}{os.pathsep}{pythonpath}"
        if pythonpath
        else str(request.repo_root)
    )
    if request.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = request.device
    return env


def _run_pytest(
    request: BatchExecutionRequest,
    artifacts: _BatchArtifacts,
    command: list[str],
) -> _ProcessOutcome:
    launched_at = time.time()
    samples: list[tuple[float, float, float]] = []
    stop_monitor = threading.Event()
    monitor: threading.Thread | None = None
    termination_signal = ""
    progress = _BatchProgress()
    progress_stop = threading.Event()
    timed_out = False
    aborted = False
    with artifacts.log.open("a", encoding="utf-8") as log:
        log.write(f"command: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=request.repo_root,
            env=_pytest_environment(request),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert process.stdout is not None
        output_thread = threading.Thread(
            target=_forward_pytest_output,
            args=(process.stdout, log, progress),
            kwargs={
                "worker_index": request.worker_index,
                "batch_id": request.batch.id,
            },
            daemon=True,
        )
        heartbeat_thread = threading.Thread(
            target=_progress_heartbeat,
            args=(progress_stop, progress),
            kwargs={
                "worker_index": request.worker_index,
                "batch_id": request.batch.id,
            },
            daemon=True,
        )
        output_thread.start()
        heartbeat_thread.start()
        if request.monitor_memory:
            monitor = threading.Thread(
                target=_monitor_memory,
                args=(process.pid, stop_monitor, samples, request.memory_interval),
                daemon=True,
            )
            monitor.start()
        try:
            timeout_at = (
                time.monotonic() + request.timeout_seconds
                if request.timeout_seconds is not None
                else None
            )
            while process.poll() is None:
                if request.abort_event is not None and request.abort_event.is_set():
                    aborted = True
                    termination_signal = terminate_process_group(
                        process, request.grace_seconds
                    )
                    break
                remaining = (
                    timeout_at - time.monotonic() if timeout_at is not None else None
                )
                if remaining is not None and remaining <= 0:
                    timed_out = True
                    termination_signal = terminate_process_group(
                        process, request.grace_seconds
                    )
                    break
                wait_seconds = 0.25 if remaining is None else min(0.25, remaining)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=wait_seconds)
            process.wait()
        finally:
            stop_monitor.set()
            progress_stop.set()
            if monitor is not None:
                monitor.join(timeout=max(1.0, request.memory_interval + 1))
            output_thread.join(timeout=5)
            heartbeat_thread.join(timeout=2)
    return _ProcessOutcome(
        returncode=int(process.returncode),
        launched_at=launched_at,
        exited_at=time.time(),
        timed_out=timed_out,
        aborted=aborted,
        termination_signal=termination_signal,
        samples=tuple(samples),
        progress=progress,
    )


def _memory_csv(samples: tuple[tuple[float, float, float], ...]) -> str:
    memory_stream = io.StringIO(newline="")
    memory_writer = csv.writer(memory_stream, lineterminator="\n")
    memory_writer.writerow(["timestamp", "host_rss_mib", "gpu_memory_mib"])
    for timestamp, rss_mib, gpu_mib in samples:
        memory_writer.writerow([f"{timestamp:.6f}", f"{rss_mib:.3f}", f"{gpu_mib:.3f}"])
    return memory_stream.getvalue()


def _timeout_result(
    request: BatchExecutionRequest,
    artifacts: _BatchArtifacts,
    outcome: _ProcessOutcome,
) -> BatchExecution:
    active_nodeid, _, _ = outcome.progress.current()
    completed, outcomes = outcome.progress.totals()
    write_console(
        f"PYTEST KILLED worker={request.worker_index} batch={request.batch.id} "
        f"reason={request.timeout_reason or 'timeout'} "
        f"signal={outcome.termination_signal} "
        f"completed_in_batch={completed} passed={outcomes['passed']} "
        f"failed={outcomes['failed']} skipped={outcomes['skipped']} "
        f"node={active_nodeid or 'unknown'}"
    )
    _discard_temporary_artifacts(*artifacts.temporary)
    return BatchExecution("timeout", "pytest batch exceeded its time budget")


def _promote_batch_artifacts(
    request: BatchExecutionRequest,
    artifacts: _BatchArtifacts,
    outcome: _ProcessOutcome,
) -> None:
    batch = request.batch
    unit = request.unit
    annotate_batch_xml(
        artifacts.temporary_xml,
        {
            "batch_id": batch.id,
            "unit_id": unit.id,
            "shard_index": str(unit.shard_index),
            "attempt_id": request.attempt_id,
            "timing_profile": request.profile,
            "synthetic": "false",
        },
    )
    if artifacts.temporary_telemetry.exists():
        try:
            telemetry = json.loads(
                artifacts.temporary_telemetry.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            telemetry = {}
        telemetry.update(
            {
                "process_launch": outcome.launched_at,
                "process_exit": outcome.exited_at,
                "batch_id": batch.id,
            }
        )
        atomic_write_json(artifacts.temporary_telemetry, telemetry)
    os.replace(artifacts.temporary_xml, artifacts.final_xml)
    if artifacts.temporary_results.exists():
        os.replace(
            artifacts.temporary_results,
            artifacts.final_xml.with_name(f"{batch.id}.results.json"),
        )
    if artifacts.temporary_telemetry.exists():
        os.replace(
            artifacts.temporary_telemetry,
            artifacts.final_xml.with_name(f"{batch.id}.telemetry.json"),
        )
    atomic_write_text(
        artifacts.final_xml.with_name(f"{batch.id}.memory.csv"),
        _memory_csv(outcome.samples),
    )
    atomic_write_json(
        artifacts.final_xml.with_name(f"{batch.id}.meta.json"),
        {
            "attempt_id": request.attempt_id,
            "batch_id": batch.id,
            "unit_id": unit.id,
            "shard_index": unit.shard_index,
            "pytest_exit_code": outcome.returncode,
            "launched_at": outcome.launched_at,
            "exited_at": outcome.exited_at,
            "synthetic": False,
        },
    )
    artifacts.selection.unlink(missing_ok=True)


def execute_batch(request: BatchExecutionRequest) -> BatchExecution:
    artifacts = _batch_artifacts(request)
    atomic_write_json(artifacts.selection, list(request.batch.nodeids))
    outcome = _run_pytest(request, artifacts, _pytest_command(request, artifacts))
    if outcome.aborted:
        _discard_temporary_artifacts(*artifacts.temporary)
        return BatchExecution(
            "infrastructure",
            "pytest stopped because the runner lease heartbeat failed",
        )
    if outcome.timed_out:
        return _timeout_result(request, artifacts, outcome)
    if outcome.returncode not in {0, 1}:
        _discard_temporary_artifacts(*artifacts.temporary)
        return BatchExecution(
            "infrastructure",
            f"pytest exited with infrastructure exit code {outcome.returncode}",
        )
    if not artifacts.temporary_xml.exists():
        _discard_temporary_artifacts(*artifacts.temporary)
        return BatchExecution("infrastructure", "pytest did not produce JUnit XML")
    validation = validate_batch_xml(artifacts.temporary_xml, request.batch.nodeids)
    if not validation.valid:
        _discard_temporary_artifacts(*artifacts.temporary)
        return BatchExecution("infrastructure", "; ".join(validation.diagnostics))
    _promote_batch_artifacts(request, artifacts, outcome)
    outcomes = Counter(case.outcome for case in validation.cases)
    write_console(
        f"PYTEST BATCH COMPLETE worker={request.worker_index} "
        f"batch={request.batch.id} finalized={len(validation.cases)} "
        f"passed={outcomes['passed']} failed={outcomes['failed']} "
        f"skipped={outcomes['skipped']}"
    )
    return BatchExecution("finalized")
