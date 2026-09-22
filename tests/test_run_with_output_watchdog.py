import os
import signal
import subprocess
import sys
import time
from pathlib import Path


WATCHDOG = Path(__file__).parents[1] / "scripts" / "run_with_output_watchdog.py"


def run_watchdog(
    *command: str, timeout_seconds: str = "5"
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--timeout-seconds",
            timeout_seconds,
            "--term-grace-seconds",
            "0.1",
            "--",
            *command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_watchdog_preserves_output_and_exit_status() -> None:
    result = run_watchdog(
        sys.executable,
        "-c",
        "import sys; print('compiler output', flush=True); sys.exit(7)",
    )

    assert result.returncode == 7
    assert result.stdout == "compiler output\n"


def test_watchdog_diagnoses_no_output(tmp_path: Path) -> None:
    diagnostics = tmp_path / "watchdog.txt"
    result = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--timeout-seconds",
            "0.2",
            "--term-grace-seconds",
            "0.1",
            "--diagnostics-file",
            str(diagnostics),
            "--",
            sys.executable,
            "-c",
            "import time; print('started'); time.sleep(30)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
    )

    assert result.returncode == 124
    assert result.stdout == "started\n"
    assert "output watchdog fired" in result.stderr
    assert "===== process tree =====" in result.stderr
    assert "no_output_seconds=" in diagnostics.read_text()


def test_watchdog_kills_descendant_that_ignores_sigterm(tmp_path: Path) -> None:
    descendant_pid_file = tmp_path / "descendant.pid"
    command = """
import pathlib
import signal
import subprocess
import sys
import time

descendant = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
    ]
)
pathlib.Path(sys.argv[1]).write_text(str(descendant.pid))
print("started", flush=True)
time.sleep(30)
"""
    result = run_watchdog(
        sys.executable,
        "-c",
        command,
        str(descendant_pid_file),
        timeout_seconds="0.2",
    )

    descendant_pid = int(descendant_pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        stat_path = Path("/proc") / str(descendant_pid) / "stat"
        if (
            stat_path.is_file()
            and stat_path.read_text().rpartition(")")[2].split()[0] == "Z"
        ):
            break
        time.sleep(0.05)
    else:
        os.kill(descendant_pid, signal.SIGKILL)
        raise AssertionError("watchdog left a live descendant behind")

    assert result.returncode == 124


def test_watchdog_terminates_command_when_diagnostics_write_fails(
    tmp_path: Path,
) -> None:
    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("file")
    result = subprocess.run(
        [
            sys.executable,
            str(WATCHDOG),
            "--timeout-seconds",
            "0.2",
            "--term-grace-seconds",
            "0.1",
            "--diagnostics-file",
            str(invalid_parent / "watchdog.txt"),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=tmp_path,
    )

    assert result.returncode == 124
    assert "watchdog could not write" in result.stderr
