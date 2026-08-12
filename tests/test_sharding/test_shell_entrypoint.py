from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pytest_timeout_plugin_enforces_marked_deadline(tmp_path: Path) -> None:
    test_file = tmp_path / "test_timeout.py"
    test_file.write_text(
        """\
import time
import pytest

@pytest.mark.timeout(0.05)
def test_hangs():
    time.sleep(1)
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_file)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "Timeout" in output


@pytest.mark.parametrize(
    ("required", "expected_code", "expected_level"),
    [("false", 0, "WARNING"), ("true", 1, "ERROR")],
)
def test_missing_precompiled_kernels_follow_explicit_policy(
    tmp_path: Path, required: str, expected_code: int, expected_level: str
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "JUNIT_DIR": str(tmp_path / "junit"),
            "MAX_JOBS": "1",
            "PIP_CONSTRAINT": os.devnull,
            "UNIT_TEST_REQUIRE_PRECOMPILED_KERNELS": required,
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; DRY_RUN=false; CUDA_VERSION=cu129; '
            "JIT_ARCH=9.0a; install_precompiled_kernels",
            "bash",
            str(REPO_ROOT / "scripts" / "test_utils.sh"),
        ],
        cwd=work,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == expected_code, output
    assert f"{expected_level}: flashinfer-cubin wheel not found" in output
    assert f"{expected_level}: flashinfer-jit-cache wheel not found" in output


@pytest.mark.parametrize(
    ("required", "jit_arch", "expected_code", "expected_message"),
    [
        (
            "sometimes",
            "9.0a",
            2,
            "UNIT_TEST_REQUIRE_PRECOMPILED_KERNELS must be true or false",
        ),
        (
            "true",
            "",
            1,
            "UNIT_TEST_REQUIRE_PRECOMPILED_KERNELS=true requires JIT_ARCH",
        ),
    ],
)
def test_precompiled_kernel_policy_rejects_invalid_configuration(
    tmp_path: Path,
    required: str,
    jit_arch: str,
    expected_code: int,
    expected_message: str,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "JIT_ARCH": jit_arch,
            "JUNIT_DIR": str(tmp_path / "junit"),
            "MAX_JOBS": "1",
            "PIP_CONSTRAINT": os.devnull,
            "UNIT_TEST_REQUIRE_PRECOMPILED_KERNELS": required,
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; DRY_RUN=false; CUDA_VERSION=cu129; '
            "install_precompiled_kernels",
            "bash",
            str(REPO_ROOT / "scripts" / "test_utils.sh"),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == expected_code, output
    assert expected_message in output


def test_optional_precompiled_policy_warns_without_jit_arch(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "JIT_ARCH": "",
            "JUNIT_DIR": str(tmp_path / "junit"),
            "MAX_JOBS": "1",
            "PIP_CONSTRAINT": os.devnull,
            "UNIT_TEST_REQUIRE_PRECOMPILED_KERNELS": "false",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; DRY_RUN=false; CUDA_VERSION=cu129; '
            "install_precompiled_kernels",
            "bash",
            str(REPO_ROOT / "scripts" / "test_utils.sh"),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "JIT_ARCH is unset" in output
    assert "using JIT compilation" in output


@pytest.mark.parametrize(
    ("python_code", "shell_code"),
    [(0, 0), (1, 1), (2, 0), (3, 3)],
)
def test_shell_maps_runner_codes_and_leaves_summary_last(
    tmp_path: Path, python_code: int, shell_code: int
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entrypoint = scripts / "task_run_unit_tests.sh"
    shutil.copy(REPO_ROOT / "scripts" / "task_run_unit_tests.sh", entrypoint)
    (scripts / "test_utils.sh").write_text("", encoding="utf-8")
    (scripts / "unit_test_runner.py").write_text(
        """\
import os
import sys

if sys.argv[1] == "__shell-settings":
    print("plan")
    print("tests/")
    raise SystemExit(0)

code = int(os.environ["FAKE_RUNNER_CODE"])
print("==========================================")
print("TEST SUMMARY")
print("==========================================")
print(f"Result: status=fake python_exit_code={code} shell_exit_code={0 if code in {0, 2} else code}")
print("==========================================")
raise SystemExit(code)
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["FAKE_RUNNER_CODE"] = str(python_code)

    result = subprocess.run(
        ["bash", str(entrypoint), "--dry-run"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == shell_code
    assert result.stderr == ""
    assert result.stdout.count("TEST SUMMARY") == 1
    assert result.stdout.rstrip().endswith("==========================================")


def test_shell_start_time_cannot_be_overridden_by_caller(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entrypoint = scripts / "task_run_unit_tests.sh"
    shutil.copy(REPO_ROOT / "scripts" / "task_run_unit_tests.sh", entrypoint)
    (scripts / "test_utils.sh").write_text("", encoding="utf-8")
    (scripts / "unit_test_runner.py").write_text(
        """\
import sys

if sys.argv[1] == "__shell-settings":
    print("plan")
    print("tests/")
    raise SystemExit(0)

starts = [
    sys.argv[index + 1]
    for index, argument in enumerate(sys.argv)
    if argument == "--wrapper-started-at"
]
print(f"EFFECTIVE WRAPPER START: {starts[-1]}")
print("==========================================")
print("TEST SUMMARY")
print("==========================================")
raise SystemExit(0)
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(entrypoint), "--dry-run", "--wrapper-started-at", "1"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    effective = next(
        line for line in result.stdout.splitlines() if line.startswith("EFFECTIVE ")
    )
    assert float(effective.rsplit(" ", 1)[1]) > 1


def test_shell_prints_fallback_summary_for_argument_preflight_error(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entrypoint = scripts / "task_run_unit_tests.sh"
    shutil.copy(REPO_ROOT / "scripts" / "task_run_unit_tests.sh", entrypoint)
    (scripts / "test_utils.sh").write_text("", encoding="utf-8")
    (scripts / "unit_test_runner.py").write_text(
        "import sys\nprint('ERROR: bad option')\nraise SystemExit(3)\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(entrypoint), "--dry-run"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 3
    assert result.stderr == ""
    assert "ERROR: bad option" in result.stdout
    assert result.stdout.count("TEST SUMMARY") == 1
    assert "Start time: " in result.stdout
    assert "End time: " in result.stdout
    assert "Time elapsed: " in result.stdout
    assert "phase=argument-preflight" in result.stdout
    assert result.stdout.rstrip().endswith("==========================================")


def test_shell_prints_fallback_summary_for_dependency_setup_failure(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entrypoint = scripts / "task_run_unit_tests.sh"
    shutil.copy(REPO_ROOT / "scripts" / "task_run_unit_tests.sh", entrypoint)
    (scripts / "test_utils.sh").write_text(
        "install_and_verify() { return 17; }\n", encoding="utf-8"
    )
    (scripts / "unit_test_runner.py").write_text(
        "import sys\n"
        "if sys.argv[1] == '__shell-settings': print('run\\ntests/'); raise SystemExit(0)\n"
        "raise AssertionError('runner must not start')\n",
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "pip").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (fake_bin / "pip").chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", str(entrypoint)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 17
    assert result.stderr == ""
    assert result.stdout.count("TEST SUMMARY") == 1
    assert "phase=dependency-setup" in result.stdout


def test_shell_preserves_abnormal_runner_exit_and_prints_fallback_summary(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entrypoint = scripts / "task_run_unit_tests.sh"
    shutil.copy(REPO_ROOT / "scripts" / "task_run_unit_tests.sh", entrypoint)
    (scripts / "test_utils.sh").write_text("", encoding="utf-8")
    (scripts / "unit_test_runner.py").write_text(
        "import sys\n"
        "if sys.argv[1] == '__shell-settings': print('plan\\ntests/'); raise SystemExit(0)\n"
        "print('runner crashed without a summary')\n"
        "raise SystemExit(9)\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(entrypoint), "--dry-run"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 9
    assert result.stdout.count("TEST SUMMARY") == 1
    assert "UNIT TEST RUNNER ABNORMAL EXIT" in result.stdout
    assert "phase=python-runner" in result.stdout


def test_shell_prints_fallback_summary_for_catchable_signal(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entrypoint = scripts / "task_run_unit_tests.sh"
    shutil.copy(REPO_ROOT / "scripts" / "task_run_unit_tests.sh", entrypoint)
    (scripts / "test_utils.sh").write_text("", encoding="utf-8")
    (scripts / "unit_test_runner.py").write_text(
        "import sys, time\n"
        "if sys.argv[1] == '__shell-settings': print('plan\\ntests/'); raise SystemExit(0)\n"
        "print('runner-started', flush=True)\n"
        "time.sleep(0.2)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        ["bash", str(entrypoint), "--dry-run"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    prefix = process.stdout.readline()
    while "runner-started" not in prefix:
        prefix += process.stdout.readline()
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=5)
    output = prefix + stdout

    assert process.returncode == 143
    assert stderr == ""
    assert output.count("TEST SUMMARY") == 1
    assert "received SIGTERM" in output
    assert output.rstrip().endswith("==========================================")
