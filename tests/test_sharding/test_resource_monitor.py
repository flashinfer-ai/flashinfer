from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from scripts.test_sharding import resource_monitor as telemetry, workers


def test_missing_gpu_does_not_lose_host_samples_or_report_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable():
        raise OSError("no GPU driver")

    monkeypatch.setattr(telemetry, "gpu_samples", unavailable)
    monkeypatch.setattr(telemetry, "gpu_process_memory", unavailable)
    monitor = telemetry.ResourceMonitor(tmp_path)
    monitor.stop.set()
    monitor.run()
    host = list(csv.DictReader((tmp_path / "host.csv").open()))
    assert len(host) == 1
    assert float(host[0]["memory_used_bytes"]) > 0
    assert not (tmp_path / "gpu.csv").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete"]
    assert any("no GPU driver" in error for error in summary["errors"])
    snapshot = json.loads((tmp_path / "gpu-processes.json").read_text())
    assert snapshot["process_memory_mib"] is None


def test_nested_shell_reuses_sampler_and_preserves_failure(tmp_path: Path) -> None:
    helper = Path(telemetry.__file__).parents[1] / "test_resource_monitor.sh"
    env = os.environ.copy()
    for key in ("FLASHINFER_RESOURCE_SNAPSHOT", "FLASHINFER_RESOURCE_MONITOR_PID"):
        env.pop(key, None)
    env.update(
        {
            "JUNIT_DIR": str(tmp_path),
            "CUDA_VISIBLE_DEVICES": "",
            "RESOURCE_MONITOR_INTERVAL": "0.05",
            "MONITOR_TEST_MEMORY": "true",
        }
    )
    script = """
source "$1"
trap stop_resource_monitor EXIT
start_resource_monitor
outer=$FLASHINFER_RESOURCE_MONITOR_PID
bash -c 'source "$1"; start_resource_monitor; stop_resource_monitor; kill -0 "$FLASHINFER_RESOURCE_MONITOR_PID"' bash "$1"
kill -0 "$outer"
exit 17
"""
    result = subprocess.run(
        ["bash", "-e", "-c", script, "bash", str(helper)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 17, result.stdout + result.stderr
    outputs = list((tmp_path / "resource-telemetry").iterdir())
    assert len(outputs) == 1
    assert json.loads((outputs[0] / "summary.json").read_text())["complete"]


@pytest.mark.parametrize(
    ("output", "age", "expected"),
    [
        ("123, 12\n123, 8\n", 0, 20),
        ("123, [N/A]\n", 0, None),
        ("123, 12\n123, [N/A]\n", 0, None),
        ("123, [N/A]\n123, 12\n", 0, None),
        ("456, [N/A]\n123, 12\n", 0, 12),
        ("", 0, 0),
        ("123, 12\n", 30, None),
        (None, 0, None),
    ],
)
def test_gpu_snapshot_preserves_unknown_readings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str | None,
    age: float,
    expected: float | None,
) -> None:
    monkeypatch.setattr(telemetry, "_query_gpu", lambda _: output)
    memory = None if output is None else telemetry.gpu_process_memory()
    snapshot = tmp_path / "gpu-processes.json"
    snapshot.write_text(
        json.dumps(
            {
                "timestamp": time.time() - age,
                "interval_seconds": 2,
                "process_memory_mib": memory,
            }
        )
    )
    monkeypatch.setenv("FLASHINFER_RESOURCE_SNAPSHOT", str(snapshot))
    assert workers._gpu_mib({123}) == expected


@pytest.mark.parametrize(
    ("mount_root", "membership", "contains_self", "expected"),
    [
        ("/", "/job", False, ["job", ""]),
        ("/job", "/job", False, [""]),
        ("/host/job", "/", True, [""]),
        ("/host/job", "/child", True, ["child", ""]),
        ("/other", "/job", False, []),
        ("/", "/missing", False, []),
        ("/", "/uncounted", False, [""]),
    ],
)
def test_cgroup_paths_require_membership_evidence(
    tmp_path: Path,
    mount_root: str,
    membership: str,
    contains_self: bool,
    expected: list[str],
) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    mount = tmp_path / "cgroup"
    for relative in ("", "job", "child"):
        path = mount / relative
        path.mkdir(exist_ok=True)
        (path / "memory.current").write_text("1")
        (path / "cgroup.procs").write_text(
            str(os.getpid())
            if contains_self and relative == membership.lstrip("/")
            else ""
        )
    # A valid leaf need not have memory accounting delegated to it.
    (mount / "uncounted").mkdir()
    (mount / "uncounted/cgroup.procs").write_text(str(os.getpid()))
    (proc / "self/cgroup").write_text(f"0::{membership}\n")
    (proc / "self/mountinfo").write_text(
        f"1 0 0:1 {mount_root} {mount} rw - cgroup2 cgroup rw\n"
    )
    assert telemetry.cgroup_paths(proc) == [(2, mount / path) for path in expected]


@pytest.mark.parametrize("owner", ["alive", "dead", "missing", "invalid"])
def test_inherited_sampler_requires_live_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
) -> None:
    monkeypatch.setenv("MONITOR_TEST_MEMORY", "true")
    monkeypatch.setenv("FLASHINFER_RESOURCE_SNAPSHOT", str(tmp_path / "old.json"))
    monkeypatch.delenv("RESOURCE_MONITOR_INTERVAL", raising=False)
    if owner == "missing":
        monkeypatch.delenv("FLASHINFER_RESOURCE_MONITOR_PID", raising=False)
    else:
        monkeypatch.setenv(
            "FLASHINFER_RESOURCE_MONITOR_PID",
            "invalid" if owner == "invalid" else "123",
        )

    def check_owner(pid: int, signum: int) -> None:
        assert pid == 123 and signum == 0
        if owner == "dead":
            raise ProcessLookupError()

    monkeypatch.setattr(telemetry.os, "kill", check_owner)
    monitor = telemetry.monitor_from_env(tmp_path)
    if owner == "alive":
        assert monitor is None
    else:
        assert monitor is not None
        assert os.environ["FLASHINFER_RESOURCE_SNAPSHOT"] != str(tmp_path / "old.json")
        assert os.environ["FLASHINFER_RESOURCE_MONITOR_PID"] == str(os.getpid())


def test_host_sample_is_flushed_before_gpu_query_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted():
        raise RuntimeError("GPU query interrupted")

    monkeypatch.setattr(telemetry, "gpu_samples", interrupted)
    with pytest.raises(RuntimeError, match="GPU query interrupted"):
        telemetry.ResourceMonitor(tmp_path).run()
    host = list(csv.DictReader((tmp_path / "host.csv").open()))
    assert len(host) == 1
    assert float(host[0]["memory_used_bytes"]) > 0
