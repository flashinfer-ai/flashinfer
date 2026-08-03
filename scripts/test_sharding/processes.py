from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


def terminate_process_group(
    process: subprocess.Popen[Any], grace_seconds: float
) -> str:
    if process.poll() is not None:
        return f"exit-{process.returncode}"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already-exited"
    try:
        process.wait(timeout=grace_seconds)
        return "SIGTERM"
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return "SIGTERM"
    process.wait()
    return "SIGKILL"
