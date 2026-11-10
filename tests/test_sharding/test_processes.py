from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress

from scripts.test_sharding.processes import terminate_process_group


def _spawn_process_group() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, signal, subprocess, sys, time; "
                "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                'print("ready", flush=True); time.sleep(30)\'], '
                "stdout=subprocess.PIPE, text=True); "
                "child.stdout.readline(); "
                "print(child.pid, flush=True); "
                "time.sleep(30)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def _child_pid(process: subprocess.Popen[str]) -> int:
    assert process.stdout is not None
    return int(process.stdout.readline().strip())


def _kill_if_alive(pid: int) -> None:
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


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
    assert termination_signal == "SIGKILL"


def test_termination_cleans_group_after_leader_already_exited() -> None:
    process = _spawn_process_group()
    child_pid = _child_pid(process)
    os.kill(process.pid, signal.SIGTERM)
    process.wait(timeout=5)
    try:
        termination_signal = terminate_process_group(process, grace_seconds=0.05)
    finally:
        _kill_if_alive(child_pid)

    assert termination_signal == "SIGKILL"
