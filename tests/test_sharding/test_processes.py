from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress

import pytest

from scripts.test_sharding import processes
from scripts.test_sharding.processes import (
    FORCED_TERMINATION,
    start_process_group,
    terminate_process_group,
)

# The graceful request is SIGTERM on POSIX and CTRL_BREAK_EVENT (SIGBREAK) on
# Windows. The leader exits on it; its child ignores it so only the forced
# group kill can end the child.
_GRACEFUL_SIGNAL = "SIGBREAK" if sys.platform == "win32" else "SIGTERM"
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _spawn_process_group() -> subprocess.Popen[str]:
    return start_process_group(
        [
            sys.executable,
            "-c",
            (
                "import os, signal, subprocess, sys, time; "
                f"signal.signal(signal.{_GRACEFUL_SIGNAL}, lambda *_: sys.exit(0)); "
                "child = subprocess.Popen([sys.executable, '-c', "
                f"'import signal,time; signal.signal(signal.{_GRACEFUL_SIGNAL}, signal.SIG_IGN); "
                'print("ready", flush=True); time.sleep(30)\'], '
                "stdout=subprocess.PIPE, text=True); "
                "child.stdout.readline(); "
                "print(child.pid, flush=True); "
                "time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _child_pid(process: subprocess.Popen[str]) -> int:
    assert process.stdout is not None
    return int(process.stdout.readline().strip())


def _kill_if_alive(pid: int) -> None:
    # Windows raises a generic OSError for a process that already exited.
    with suppress(OSError):
        os.kill(pid, _SIGKILL)


def test_termination_grace_applies_to_descendants_after_leader_exits() -> None:
    process = _spawn_process_group()
    child_pid = _child_pid(process)
    started = time.monotonic()
    try:
        termination_signal = terminate_process_group(process, grace_seconds=0.25)
    finally:
        _kill_if_alive(child_pid)
        _kill_if_alive(process.pid)
        process.wait(timeout=5)

    elapsed = time.monotonic() - started
    assert elapsed >= 0.20
    assert elapsed < 2
    assert termination_signal == FORCED_TERMINATION


def test_termination_cleans_group_after_leader_already_exited() -> None:
    process = _spawn_process_group()
    child_pid = _child_pid(process)
    # On Windows this is TerminateProcess on the leader alone; the child stays
    # in the leader's job, which is what the group cleanup must still reach.
    os.kill(process.pid, signal.SIGTERM)
    process.wait(timeout=5)
    try:
        termination_signal = terminate_process_group(process, grace_seconds=0.05)
    finally:
        _kill_if_alive(child_pid)

    assert termination_signal == FORCED_TERMINATION


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object setup")
def test_launch_fails_when_job_object_cannot_be_assigned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[subprocess.Popen[str]] = []

    def fail_assignment(process: subprocess.Popen[str]) -> None:
        started.append(process)
        raise OSError(5, "AssignProcessToJobObject failed: Access is denied.")

    monkeypatch.setattr(processes, "_assign_job", fail_assignment)

    with pytest.raises(OSError, match="AssignProcessToJobObject"):
        start_process_group(
            [sys.executable, "-c", "import time; time.sleep(30)"], text=True
        )

    assert len(started) == 1
    assert started[0].returncode is not None
