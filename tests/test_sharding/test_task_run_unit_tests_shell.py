from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "task_run_unit_tests.sh"


def _run_wrapper(
    tmp_path: Path, **environment: str
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        """\
#!/bin/bash
if [ "${2:-}" = "__shell-settings" ]; then
    settings_rc="${SHELL_SETTINGS_RC:-0}"
    if [ "${settings_rc}" -ne 0 ]; then
        echo "fake argparse error" >&2
        exit "${settings_rc}"
    fi
    printf '%s\\n%s\\n' "${RUNNER_OPERATION:-plan}" "tests/"
    exit 0
fi
if [ "${1:-}" = "-c" ]; then
    printf '%s' "${PYTHON_IMPORT_OUTPUT:-}"
    exit "${PYTHON_IMPORT_RC:-0}"
fi
if [ -n "${RUNNER_SIGNAL:-}" ]; then
    kill -s "${RUNNER_SIGNAL}" "$$"
fi
exit "${RUNNER_RC:-0}"
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    pip = fake_bin / "pip"
    pip.write_text('#!/bin/bash\nexit "${PIP_RC:-0}"\n', encoding="utf-8")
    pip.chmod(0o755)
    mktemp = fake_bin / "mktemp"
    mktemp.write_text(
        """\
#!/bin/bash
if [ "${MKTEMP_RC:-0}" -ne 0 ]; then
    exit "${MKTEMP_RC}"
fi
exec /usr/bin/mktemp "$@"
""",
        encoding="utf-8",
    )
    mktemp.chmod(0o755)
    env = os.environ.copy()
    for name in (
        "PIP_RC",
        "MKTEMP_RC",
        "PYTEST_FILE_TIMEOUT_KILL_AFTER_SECONDS",
        "PYTEST_FILE_TIMEOUT_SECONDS",
        "PYTHON_IMPORT_RC",
        "PYTHON_IMPORT_OUTPUT",
        "RUNNER_OPERATION",
        "RUNNER_RC",
        "RUNNER_SIGNAL",
        "SHELL_SETTINGS_RC",
    ):
        env.pop(name, None)
    env.update(environment)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    if "PIP_CONSTRAINT" not in environment:
        env["PIP_CONSTRAINT"] = str(tmp_path / "unused-constraint.txt")
    return subprocess.run(
        [str(SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("runner_code", "runner_status", "wrapper_code"),
    [
        (0, "complete-without-failures", 0),
        (1, "complete-with-failures", 1),
        (2, "incomplete-and-resumable", 0),
        (3, "configuration-collection-or-infrastructure-error", 3),
    ],
)
def test_semantic_runner_codes_use_the_ci_exit_contract(
    tmp_path: Path, runner_code: int, runner_status: str, wrapper_code: int
) -> None:
    result = _run_wrapper(tmp_path, RUNNER_RC=str(runner_code))

    assert result.returncode == wrapper_code, result.stdout + result.stderr
    assert (
        f"UNIT TEST RUNNER RESULT: exit_code={runner_code} "
        f"status={runner_status} wrapper_exit_code={wrapper_code}"
    ) in result.stdout


def test_wrapper_does_not_enable_bash_errexit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "set -e" not in source
    assert "set -eo" not in source


@pytest.mark.parametrize(
    ("signal", "returncode"),
    [("TERM", 143), ("KILL", 137)],
)
def test_runner_signal_exit_remains_nonzero(
    tmp_path: Path, signal: str, returncode: int
) -> None:
    result = _run_wrapper(tmp_path, RUNNER_SIGNAL=signal)

    assert result.returncode == returncode
    assert (
        f"UNIT TEST RUNNER ABNORMAL EXIT: exit_code={returncode} "
        "wrapper_exit_code=unchanged"
    ) in result.stderr


def test_argparse_failure_before_runner_execution_remains_nonzero(
    tmp_path: Path,
) -> None:
    result = _run_wrapper(tmp_path, SHELL_SETTINGS_RC="2")

    assert result.returncode == 2
    assert "fake argparse error" in result.stderr
    assert "UNIT TEST RUNNER RESULT" not in result.stdout


def test_dependency_failure_before_runner_execution_remains_nonzero(
    tmp_path: Path,
) -> None:
    result = _run_wrapper(
        tmp_path,
        RUNNER_OPERATION="run",
        PYTHON_IMPORT_RC="1",
        PIP_RC="47",
    )

    assert result.returncode == 47
    assert "UNIT TEST RUNNER RESULT" not in result.stdout


def test_installation_failure_remains_nonzero_without_errexit(tmp_path: Path) -> None:
    result = _run_wrapper(
        tmp_path,
        RUNNER_OPERATION="run",
        PYTHON_IMPORT_RC="0",
        PIP_RC="41",
    )

    assert result.returncode == 41
    assert "UNIT TEST RUNNER RESULT" not in result.stdout


def test_constraint_file_creation_failure_remains_nonzero_without_errexit(
    tmp_path: Path,
) -> None:
    result = _run_wrapper(
        tmp_path,
        PIP_CONSTRAINT="",
        PYTHON_IMPORT_OUTPUT="torch==2.8.0",
        MKTEMP_RC="42",
    )

    assert result.returncode == 42
    assert "UNIT TEST RUNNER RESULT" not in result.stdout


def test_obsolete_environment_failure_remains_nonzero(tmp_path: Path) -> None:
    result = _run_wrapper(tmp_path, PYTEST_FILE_TIMEOUT_SECONDS="10")

    assert result.returncode == 3
    assert "PYTEST_FILE_TIMEOUT_SECONDS is obsolete" in result.stderr
    assert "UNIT TEST RUNNER RESULT" not in result.stdout
