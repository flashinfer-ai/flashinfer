# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Host-only process and GPU-lock primitives for sweep workers."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence
import json


class WorkerStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class FailureKind(str, Enum):
    NONE = "none"
    COMPILE_ERROR = "compile_error"
    COMPILE_TIMEOUT = "compile_timeout"
    LOCK_TIMEOUT = "lock_timeout"
    RUN_ERROR = "run_error"
    RUN_TIMEOUT = "run_timeout"
    CUDA_FAULT = "cuda_fault"
    WORKER_CRASH = "worker_crash"


@dataclass(frozen=True)
class ProcessResult:
    status: WorkerStatus
    failure: FailureKind
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float
    termination: str | None = None
    phase: str | None = None
    phase_unit: str | None = None
    phase_elapsed_seconds: float | None = None

    @property
    def ok(self) -> bool:
        return self.status is WorkerStatus.SUCCEEDED


@dataclass(frozen=True)
class WorkerActivity:
    """Snapshot of an in-flight request, replaced atomically by the poller."""

    phase: str
    unit: str
    phase_started: float
    request_started: float


class GPULockTimeout(TimeoutError):
    """Raised when a worker cannot acquire its GPU's exclusive timing lock."""


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists even if another user owns a process in the group.
        return True
    return True


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    grace_seconds: float = 1.0,
) -> str:
    """Terminate a session leader and all descendants, escalating to SIGKILL."""

    process_group = process.pid
    prior_returncode = process.poll()
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return (
            f"exit-{prior_returncode}"
            if prior_returncode is not None
            else "already-exited"
        )

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while _process_group_exists(process_group):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=min(0.05, remaining))
        # The leader may exit while a descendant in the process group remains.
        time.sleep(min(0.01, max(0.0, remaining)))

    if not _process_group_exists(process_group):
        process.wait()
        return "SIGTERM"

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        process.wait()
        return "SIGTERM"
    process.wait()
    return "SIGKILL"


def _failure_for_phase(phase: str, *, timeout: bool) -> FailureKind:
    normalized = phase.lower()
    if timeout:
        if normalized == "compile":
            return FailureKind.COMPILE_TIMEOUT
        if normalized == "lock_wait":
            return FailureKind.LOCK_TIMEOUT
        return FailureKind.RUN_TIMEOUT
    return (
        FailureKind.COMPILE_ERROR if normalized == "compile" else FailureKind.RUN_ERROR
    )


class PersistentProcessWorker:
    """Long-lived isolated worker, hard-killed on faults or phase timeouts."""

    def __init__(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        cwd: str | os.PathLike[str] | None,
        env: Mapping[str, str],
        log_path: str | os.PathLike[str],
        restart_after: int = 20,
        grace_seconds: float = 1.0,
    ) -> None:
        if restart_after <= 0:
            raise ValueError("restart_after must be positive")
        self.command = tuple(os.fspath(part) for part in command)
        self.cwd = cwd
        self.env = dict(env)
        self.log_path = Path(log_path)
        self.restart_after = restart_after
        self.grace_seconds = grace_seconds
        self.process: subprocess.Popen[str] | None = None
        self.jobs = 0
        self.activity: WorkerActivity | None = None
        self._log_stream = None

    def _start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_stream = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.jobs = 0

    def _stop(self) -> str | None:
        termination = None
        if self.process is not None:
            termination = terminate_process_group(
                self.process, grace_seconds=self.grace_seconds
            )
        self.process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
        return termination

    def _kill(self) -> str:
        if self.process is not None:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        self.process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
        return "SIGKILL"

    def close(self) -> None:
        self._stop()

    def _log_size(self) -> int:
        try:
            return self.log_path.stat().st_size
        except OSError:
            return 0

    def _log_tail(self, limit: int = 4000, *, since: int = 0) -> str:
        try:
            with self.log_path.open(encoding="utf-8", errors="replace") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                stream.seek(max(since, size - limit))
                return stream.read()
        except OSError:
            return ""

    def run_request(
        self,
        payload_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        progress_path: str | os.PathLike[str],
        *,
        compile_timeout_seconds: float,
        run_timeout_seconds: float,
        hard_timeout_seconds: float,
    ) -> ProcessResult:
        started = time.monotonic()
        self.activity = WorkerActivity("compile", "", started, started)
        try:
            return self._run_request(
                payload_path,
                output_path,
                progress_path,
                started=started,
                compile_timeout_seconds=compile_timeout_seconds,
                run_timeout_seconds=run_timeout_seconds,
                hard_timeout_seconds=hard_timeout_seconds,
            )
        finally:
            self.activity = None

    def _run_request(
        self,
        payload_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        progress_path: str | os.PathLike[str],
        *,
        started: float,
        compile_timeout_seconds: float,
        run_timeout_seconds: float,
        hard_timeout_seconds: float,
    ) -> ProcessResult:
        if self.process is None or self.process.poll() is not None:
            self._stop()
            self._start()
        assert self.process is not None and self.process.stdin is not None
        # The log is shared by every request this process serves; only the
        # current request's output is relevant to its failure record.
        log_start = self._log_size()
        output = Path(output_path)
        progress = Path(progress_path)
        output.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)
        request = {
            "payload": os.fspath(payload_path),
            "output": os.fspath(output_path),
            "progress": os.fspath(progress_path),
        }
        try:
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            return ProcessResult(
                status=WorkerStatus.FAILED,
                failure=FailureKind.WORKER_CRASH,
                returncode=self.process.poll(),
                stdout="",
                stderr=(
                    f"{type(error).__name__}: {error}\n"
                    f"{self._log_tail(since=log_start)}"
                ),
                elapsed_seconds=time.monotonic() - started,
                termination=self._stop(),
            )

        phase = "compile"
        unit = ""
        phase_started = started
        last_progress = None
        while True:
            if output.exists():
                self.jobs += 1
                elapsed = time.monotonic() - started
                try:
                    output_record = json.loads(output.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    output_record = {}
                if (
                    output_record.get("failure") == FailureKind.CUDA_FAULT.value
                    or self.jobs >= self.restart_after
                ):
                    self._stop()
                return ProcessResult(
                    status=WorkerStatus.SUCCEEDED,
                    failure=FailureKind.NONE,
                    returncode=0,
                    stdout="",
                    stderr="",
                    elapsed_seconds=elapsed,
                )
            returncode = self.process.poll()
            if returncode is not None:
                log = self._log_tail(since=log_start)
                self._stop()
                now = time.monotonic()
                return ProcessResult(
                    status=WorkerStatus.FAILED,
                    failure=FailureKind.WORKER_CRASH,
                    returncode=returncode,
                    stdout="",
                    stderr=log,
                    elapsed_seconds=now - started,
                    phase=phase,
                    phase_unit=unit,
                    phase_elapsed_seconds=now - phase_started,
                )
            try:
                current_progress = progress.read_text(encoding="utf-8")
            except OSError:
                current_progress = None
            if current_progress and current_progress != last_progress:
                last_progress = current_progress
                try:
                    progress_record = json.loads(current_progress)
                    phase = str(progress_record["phase"])
                    unit = str(progress_record.get("unit", ""))
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    phase = "run"
                    unit = ""
                phase_started = time.monotonic()
                self.activity = WorkerActivity(phase, unit, phase_started, started)
            now = time.monotonic()
            phase_timeout = (
                compile_timeout_seconds if phase == "compile" else run_timeout_seconds
            )
            phase_timed_out = (
                phase != "lock_wait" and now - phase_started >= phase_timeout
            )
            hard_timed_out = now - started >= hard_timeout_seconds
            if phase_timed_out or hard_timed_out:
                termination = self._kill()
                return ProcessResult(
                    status=WorkerStatus.TIMED_OUT,
                    failure=_failure_for_phase(phase, timeout=True),
                    returncode=None,
                    stdout="",
                    stderr=self._log_tail(since=log_start),
                    elapsed_seconds=now - started,
                    termination=termination,
                    phase=phase,
                    phase_unit=unit,
                    phase_elapsed_seconds=now - phase_started,
                )
            time.sleep(0.05)


def gpu_lock_path(
    gpu: str | int,
    *,
    lock_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return a stable lock path without trusting GPU text as a filename."""

    root = (
        Path(lock_dir)
        if lock_dir is not None
        else Path(tempfile.gettempdir()) / "flashinfer-prims-ts-moe-gpu-locks"
    )
    identity = str(gpu)
    readable = "".join(
        character if character.isalnum() else "-" for character in identity
    )
    readable = readable.strip("-")[:32] or "gpu"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return root / f"{readable}-{digest}.lock"


@contextmanager
def gpu_lock(
    gpu: str | int,
    *,
    lock_dir: str | os.PathLike[str] | None = None,
    timeout_seconds: float | None = None,
    poll_interval_seconds: float = 0.05,
) -> Iterator[Path]:
    """Hold the fair, exclusive per-GPU advisory lock for timing and real IO.

    The descriptor remains open for the context lifetime.  ``flock`` is
    released by the kernel if the worker crashes or is killed. A turnstile
    prevents a worker that just released the GPU lock from bypassing waiters.
    """

    if timeout_seconds is not None and timeout_seconds < 0:
        raise ValueError("timeout_seconds must be non-negative or None")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    path = gpu_lock_path(gpu, lock_dir=lock_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    turnstile = path.with_suffix(f"{path.suffix}.turnstile").open(
        "a+", encoding="utf-8"
    )
    acquired = False
    started = time.monotonic()

    def acquire(descriptor: int) -> None:
        if timeout_seconds is None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            return
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    raise GPULockTimeout(
                        f"timed out waiting for GPU {gpu!r} lock {path}"
                    ) from None
                time.sleep(
                    min(poll_interval_seconds, max(0.0, timeout_seconds - elapsed))
                )

    try:
        acquire(turnstile.fileno())
        try:
            acquire(stream.fileno())
            acquired = True
        finally:
            fcntl.flock(turnstile.fileno(), fcntl.LOCK_UN)
        # Diagnostic only; ownership is the flock, not this content.
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        yield path
    finally:
        if acquired:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        turnstile.close()
        stream.close()


def resolve_gpu_lock_identity(fallback: str | int = "0") -> str:
    """Return the current physical GPU UUID, or a stable caller fallback."""

    try:
        import torch

        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        uuid = getattr(properties, "uuid", None)
        if uuid is not None:
            return str(uuid)
    except (AttributeError, ImportError, RuntimeError):
        pass
    return str(fallback)
