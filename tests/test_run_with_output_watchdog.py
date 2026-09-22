import subprocess
import sys
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
        timeout=20,
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
            "import time; print('started', flush=True); time.sleep(30)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 124
    assert result.stdout == "started\n"
    assert "output watchdog fired" in result.stderr
    assert "===== process tree =====" in result.stderr
    assert "no_output_seconds=" in diagnostics.read_text()
