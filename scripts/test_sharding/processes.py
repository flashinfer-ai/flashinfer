from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(
    process: subprocess.Popen[Any], grace_seconds: float
) -> str:
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
    while True:
        process.poll()
        if not _process_group_exists(process_group):
            process.wait()
            return "SIGTERM"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.05, remaining))

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        process.wait()
        return "SIGTERM"
    process.wait()
    return "SIGKILL"
