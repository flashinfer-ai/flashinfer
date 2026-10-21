from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.test_run_status import local_worker_identity


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/test_run_status.py"


@pytest.mark.parametrize(
    "count_name,rank_name",
    [
        ("LOCAL_WORLD_SIZE", "LOCAL_RANK"),
        ("OMPI_COMM_WORLD_LOCAL_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK"),
        ("MPI_LOCALNRANKS", "MPI_LOCALRANKID"),
    ],
)
def test_worker_identity_supports_ci_mpi_launchers(monkeypatch, count_name, rank_name):
    for name in [
        "LOCAL_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MPI_LOCALNRANKS",
        "PMI_SIZE",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(count_name, "4")
    for rank in range(4):
        monkeypatch.setenv(rank_name, str(rank))
        assert local_worker_identity() == (4, rank)


def test_global_pmi_size_does_not_count_other_containers(monkeypatch):
    for name in ["LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PMI_SIZE", "4")
    monkeypatch.setenv("PMI_RANK", "3")
    assert local_worker_identity() == (1, 0)


def command(directory: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), "--directory", str(directory), *args],
        text=True,
        capture_output=True,
        check=True,
    )


def test_worker_missing_before_python_starts_prevents_publication(
    tmp_path: Path,
) -> None:
    command(tmp_path, "begin")
    receipt = command(tmp_path, "start").stdout.strip()
    workers = tmp_path / Path(receipt).stem
    workers.mkdir()
    (workers / "worker-one.json").write_text(
        json.dumps(
            {
                "complete": True,
                "exit_code": 1,
                "infrastructure_errors": [],
                "worker_count": 2,
                "worker_rank": 0,
                "tests_executed": 1,
            }
        )
    )
    command(tmp_path, "finish", "--receipt", receipt, "--status", "1")
    command(tmp_path, "complete", "--status", "1")
    report = json.loads((tmp_path / "script-result.json").read_text())
    assert report["complete"] is False
    assert report["infrastructure_errors"]


# Cover assertion failures on all three launch routes, then put each additional
# outcome where it exercises distinct behavior rather than crossing every pair.
@pytest.mark.parametrize(
    "route,outcome",
    [
        ("torchrun", "passed"),
        ("torchrun", "assertion"),
        ("torchrun", "internal"),
        ("torchrun", "crash"),
        ("torchrun", "skip"),
        ("mpi", "assertion"),
        ("multi_node", "passed"),
        ("multi_node", "assertion"),
        ("multi_node", "collection"),
        ("multi_node", "crash"),
    ],
)
def test_communication_wrappers_report_execution_outcomes(
    tmp_path: Path, route: str, outcome: str
) -> None:
    wrapper, test_path = {
        "torchrun": (
            "task_test_multi_gpu_comm_kernels.sh",
            "tests/attention/test_parallel_attention.py",
        ),
        "mpi": (
            "task_test_multi_gpu_comm_kernels.sh",
            "tests/comm/test_allreduce_unified_api.py",
        ),
        "multi_node": (
            "task_test_multi_node_comm_kernels.sh",
            "tests/comm/test_mnnvl_memory.py",
        ),
    }[route]
    # Run the production wrappers and shared launcher, replacing only GPU setup
    # and distributed executables. The fake launchers run real pytest workers
    # and normalize all worker failures to 1, as torchrun does.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in [
        wrapper,
        "test_utils.sh",
        "test_path_scope.sh",
        "test_run_status.sh",
        "test_run_status.py",
    ]:
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    (scripts / "setup_test_env.sh").write_text("install_and_verify() { :; }\n")
    test_file = tmp_path / test_path
    test_file.parent.mkdir(parents=True, exist_ok=True)
    if outcome == "collection":
        test_file.write_text("import nonexistent_cache_contract_module\n")
    elif outcome == "crash":
        test_file.write_text("import os\ndef test_crash():\n    os._exit(139)\n")
    elif outcome == "skip":
        test_file.write_text(
            'import pytest\n@pytest.mark.skip(reason="unsupported")\ndef test_skip():\n    pass\n'
        )
    elif outcome == "passed":
        test_file.write_text("def test_success():\n    pass\n")
    else:
        test_file.write_text("def test_failure():\n    assert False\n")
    if outcome == "internal":
        (tmp_path / "conftest.py").write_text(
            'def pytest_sessionstart(session):\n    raise RuntimeError("internal error")\n'
        )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ["mpirun", "torchrun"]:
        executable = fake_bin / name
        executable.write_text(
            "#!/bin/sh\n"
            + ('shift\npython "$@"\n' if name == "torchrun" else 'shift 2\n"$@"\n')
            + 'status=$?\n[ "$status" -eq 0 ] && exit 0\nexit 1\n'
        )
        executable.chmod(0o755)
    env = dict(os.environ)
    env.update(
        PATH=f"{fake_bin}:{env['PATH']}",
        PIP_CONSTRAINT=os.devnull,
        MAX_JOBS="1",
        MONITOR_TEST_MEMORY="false",
        PARALLEL_TESTS="false",
        JUNIT_DIR=str(tmp_path / "junit"),
        UNIT_TEST_RUN_STATUS_DIR=str(tmp_path / "status"),
        TEST_PATH=test_path,
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
    )
    result = subprocess.run(
        ["bash", str(scripts / wrapper)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode in (0, 1), result.stdout + result.stderr
    report = json.loads((tmp_path / "status/script-result.json").read_text())
    assert report["exit_code"] == result.returncode
    if outcome in ("passed", "assertion"):
        assert report["complete"] is True
        assert report["infrastructure_errors"] == []
        assert result.returncode == (1 if outcome == "assertion" else 0)
        assert report["tests_executed"] == 1
    elif outcome == "skip":
        assert report["complete"] is True
        assert report["tests_executed"] == 0
    else:
        assert report["infrastructure_errors"]
        if outcome == "crash":
            assert report["complete"] is False


def test_setup_failure_leaves_wrapper_report_incomplete(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in [
        "task_test_multi_node_comm_kernels.sh",
        "test_utils.sh",
        "test_path_scope.sh",
        "test_run_status.sh",
        "test_run_status.py",
    ]:
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    (scripts / "setup_test_env.sh").write_text("exit 1\n")
    env = dict(os.environ)
    env.update(
        PIP_CONSTRAINT=os.devnull,
        MAX_JOBS="1",
        JUNIT_DIR=str(tmp_path / "junit"),
        UNIT_TEST_RUN_STATUS_DIR=str(tmp_path / "status"),
        TEST_PATH="",
    )
    result = subprocess.run(
        ["bash", str(scripts / "task_test_multi_node_comm_kernels.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((tmp_path / "status/script-result.json").read_text())
    assert report["complete"] is False
