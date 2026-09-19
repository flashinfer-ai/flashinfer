from __future__ import annotations

import os
import signal
import time
from contextlib import suppress
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest

from scripts.test_sharding import workers as workers_module
from scripts.test_sharding.models import Batch, Unit
from scripts.test_sharding.progress import encode_pytest_event
from scripts.unit_test_runner import _configure_jit_parallelism
from scripts.test_sharding.workers import (
    BatchExecutionRequest,
    _BatchProgress,
    _HostCpuTimes,
    _ProcessCpuStat,
    _ResourceSample,
    _forward_pytest_output,
    _host_cpu_percentages,
    _parse_process_cpu_stat,
    _pytest_environment,
    _resource_csv,
    _summarize_process_tree,
    _worker_master_port,
    execute_batch,
)


def test_abnormal_pytest_exit_does_not_finalize_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_pythonpath = tmp_path / "fake-pythonpath"
    pytest_package = fake_pythonpath / "pytest"
    pytest_package.mkdir(parents=True)
    (pytest_package / "__init__.py").write_text("", encoding="utf-8")
    (pytest_package / "__main__.py").write_text(
        """\
import json
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

arguments = sys.argv[1:]
selection = next(value.split("=", 1)[1] for value in arguments if value.startswith("--flashinfer-node-file="))
xml_path = Path(next(value.split("=", 1)[1] for value in arguments if value.startswith("--junitxml=")))
nodeid = json.loads(Path(selection).read_text(encoding="utf-8"))[0]
xml_path.write_text(
    "<testsuites><testsuite tests=\\"1\\"><testcase name=\\"case\\" time=\\"0\\">"
    "<properties><property name=\\"pytest_nodeid\\" value=" + quoteattr(nodeid) + "/>"
    "</properties></testcase></testsuite></testsuites>",
    encoding="utf-8",
)
raise SystemExit(3)
""",
        encoding="utf-8",
    )
    existing_pythonpath = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        (
            f"{fake_pythonpath}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(fake_pythonpath)
        ),
    )
    nodeid = "tests/test_sample.py::test_case"
    batch = Batch(
        id="batch-1",
        source_file="tests/test_sample.py",
        nodeids=(nodeid,),
        estimated_ms=1000,
        overhead_ms=0,
        oversized=False,
    )
    unit = Unit(
        id="unit-1",
        batches=(batch,),
        estimated_ms=1000,
        oversized=False,
        shard_index=0,
    )

    result = execute_batch(
        BatchExecutionRequest(
            repo_root=Path(__file__).resolve().parents[2],
            pytest_root=Path(__file__).resolve().parents[2],
            junit_dir=tmp_path / "junit",
            unit=unit,
            batch=batch,
            attempt_id="attempt-1",
            profile="synthetic",
            timeout_seconds=10,
            timeout_reason=None,
            grace_seconds=0,
            worker_index=0,
            device=None,
            monitor_memory=False,
            memory_interval=1,
        )
    )

    assert result.status == "infrastructure"
    assert "exit code 3" in result.diagnostic
    assert not list((tmp_path / "junit").glob("shards/*/units/*/batches/*.xml"))


def test_failure_event_is_printed_with_source_node_diagnostic_and_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stream = StringIO(
        encode_pytest_event(
            "failure",
            nodeid="tests/test_sample.py::test_case",
            phase="call",
            diagnostic="assert expected == actual",
            diagnostic_truncated=False,
        )
        + "\n"
    )
    log = StringIO()

    _forward_pytest_output(
        stream,
        log,
        _BatchProgress(),
        worker_index=2,
        batch_id="batch-1",
        source_file="tests/test_sample.py",
        log_path=tmp_path / "batch.log",
        results_path=tmp_path / "batch.results.json",
        junit_path=tmp_path / "batch.xml",
    )

    output = capsys.readouterr().out
    assert "source=tests/test_sample.py" in output
    assert "node=tests/test_sample.py::test_case" in output
    assert "assert expected == actual" in output
    assert f"results={tmp_path / 'batch.results.json'}" in output
    assert f"junit={tmp_path / 'batch.xml'}" in output


def _fake_batch_request(tmp_path: Path) -> BatchExecutionRequest:
    nodeid = "tests/test_sample.py::test_case"
    batch = Batch(
        id="batch-1",
        source_file="tests/test_sample.py",
        nodeids=(nodeid,),
        estimated_ms=1000,
        overhead_ms=0,
        oversized=False,
    )
    unit = Unit(
        id="unit-1",
        batches=(batch,),
        estimated_ms=1000,
        oversized=False,
        shard_index=0,
    )
    return BatchExecutionRequest(
        repo_root=Path(__file__).resolve().parents[2],
        pytest_root=Path(__file__).resolve().parents[2],
        junit_dir=tmp_path / "junit",
        unit=unit,
        batch=batch,
        attempt_id="attempt-1",
        profile="synthetic",
        timeout_seconds=10,
        timeout_reason=None,
        grace_seconds=0,
        worker_index=0,
        device=None,
        monitor_memory=False,
        memory_interval=1,
    )


@pytest.mark.parametrize("worker_count", [1, 2, 4, 8])
def test_pytest_workers_get_isolated_default_master_port_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_count: int
) -> None:
    monkeypatch.delenv("MASTER_PORT", raising=False)
    request = _fake_batch_request(tmp_path)

    ports = [
        int(
            _pytest_environment(replace(request, worker_index=worker_index))[
                "MASTER_PORT"
            ]
        )
        for worker_index in range(worker_count)
    ]

    assert ports == [29500 + worker_index * 100 for worker_index in range(worker_count)]
    blocks = [set(range(port, port + 100)) for port in ports]
    assert all(
        left.isdisjoint(right)
        for left_index, left in enumerate(blocks)
        for right in blocks[left_index + 1 :]
    )


def test_explicit_master_port_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MASTER_PORT", "31234")

    environment = _pytest_environment(
        replace(_fake_batch_request(tmp_path), worker_index=360)
    )

    assert environment["MASTER_PORT"] == "31234"


def test_worker_master_port_defines_a_valid_block() -> None:
    assert _worker_master_port(0) == "29500"
    assert _worker_master_port(3) == "29800"
    assert _worker_master_port(359) == "65400"

    with pytest.raises(ValueError, match="non-negative"):
        _worker_master_port(-1)
    with pytest.raises(ValueError, match="no valid rendezvous port block"):
        _worker_master_port(360)


def test_automatic_jit_parallelism_preserves_host_budget_for_prebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_JOBS", "163")
    monkeypatch.setenv("FLASHINFER_AUTO_MAX_JOBS", "1")
    monkeypatch.delenv("FLASHINFER_JIT_PREBUILD_MAX_JOBS", raising=False)

    _configure_jit_parallelism(workers=4)

    assert os.environ["MAX_JOBS"] == "40"
    assert os.environ["FLASHINFER_JIT_PREBUILD_MAX_JOBS"] == "163"
    assert "FLASHINFER_AUTO_MAX_JOBS" not in os.environ


def test_host_cpu_percentages_separate_busy_from_iowait() -> None:
    busy, iowait = _host_cpu_percentages(
        _HostCpuTimes(total=100, idle=30, iowait=10),
        _HostCpuTimes(total=200, idle=50, iowait=20),
    )

    assert busy == pytest.approx(70.0)
    assert iowait == pytest.approx(10.0)
    assert _host_cpu_percentages(None, None) == (None, None)


def test_resource_csv_includes_host_and_process_tree_diagnostics() -> None:
    output = _resource_csv(
        (
            _ResourceSample(
                timestamp=123.0,
                host_rss_mib=456.0,
                gpu_memory_mib=789.0,
                host_cpu_percent=80.0,
                host_iowait_percent=2.5,
                load1=10.0,
                load5=8.0,
                load15=6.0,
                worker_cpu_seconds=12.0,
                descendant_cpu_seconds=34.0,
                descendant_process_count=5,
                running_process_count=3,
                disk_sleep_process_count=1,
            ),
        )
    )

    header, row = output.splitlines()
    assert "host_cpu_percent" in header
    assert "host_iowait_percent" in header
    assert "descendant_cpu_seconds" in header
    assert "disk_sleep_process_count" in header
    assert (
        row
        == "123.000000,456.000,789.000,80.000,2.500,10.000,8.000,6.000,12.000,34.000,5,3,1"
    )


def test_process_cpu_stat_handles_parentheses_in_process_name() -> None:
    # Fields after the process name begin at state (field 3). CPU counters are
    # utime, stime, cutime and cstime (fields 14 through 17).
    stat = "123 (compiler worker (sm90)) R " + " ".join(
        ["0"] * 10 + ["100", "50", "25", "5"]
    )

    parsed = _parse_process_cpu_stat(stat, ticks_per_second=10)

    assert parsed.state == "R"
    assert parsed.own_cpu_seconds == 15
    assert parsed.children_cpu_seconds == 3


def test_process_tree_cpu_includes_reaped_and_live_descendants() -> None:
    stats = {
        10: _ProcessCpuStat("S", own_cpu_seconds=2, children_cpu_seconds=7),
        11: _ProcessCpuStat("R", own_cpu_seconds=3, children_cpu_seconds=5),
    }

    summary = _summarize_process_tree(10, stats)

    assert summary.worker_cpu_seconds == 2
    assert summary.descendant_cpu_seconds == 15
    assert summary.descendant_process_count == 1
    assert summary.running_process_count == 1
    assert _summarize_process_tree(99, stats).worker_cpu_seconds == 0


def _install_fake_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    fake_pythonpath = tmp_path / "fake-pythonpath"
    pytest_package = fake_pythonpath / "pytest"
    pytest_package.mkdir(parents=True)
    (pytest_package / "__init__.py").write_text("", encoding="utf-8")
    (pytest_package / "__main__.py").write_text(source, encoding="utf-8")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        (
            f"{fake_pythonpath}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(fake_pythonpath)
        ),
    )


def test_inherited_stdout_cannot_block_batch_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_pid_path = tmp_path / "escaped-child.pid"
    monkeypatch.setenv("ESCAPED_CHILD_PID_FILE", str(child_pid_path))
    monkeypatch.setattr(workers_module, "_OUTPUT_DRAIN_SECONDS", 0.05, raising=False)
    _install_fake_pytest(
        tmp_path,
        monkeypatch,
        """\
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

arguments = sys.argv[1:]
selection = next(value.split("=", 1)[1] for value in arguments if value.startswith("--flashinfer-node-file="))
xml_path = Path(next(value.split("=", 1)[1] for value in arguments if value.startswith("--junitxml=")))
nodeid = json.loads(Path(selection).read_text(encoding="utf-8"))[0]
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(30)"],
    start_new_session=True,
)
Path(os.environ["ESCAPED_CHILD_PID_FILE"]).write_text(str(child.pid), encoding="utf-8")
xml_path.write_text(
    "<testsuites><testsuite tests=\\"1\\"><testcase name=\\"case\\" time=\\"0\\">"
    "<properties><property name=\\"pytest_nodeid\\" value=" + quoteattr(nodeid) + "/>"
    "</properties></testcase></testsuite></testsuites>",
    encoding="utf-8",
)
""",
    )
    started = time.monotonic()
    child_pid: int | None = None
    try:
        result = execute_batch(_fake_batch_request(tmp_path))
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    finally:
        if child_pid is None and child_pid_path.exists():
            child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)

    assert time.monotonic() - started < 3
    assert result.status == "infrastructure"
    assert "output did not reach EOF" in result.diagnostic


def test_output_reader_failure_is_an_infrastructure_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_pytest(
        tmp_path,
        monkeypatch,
        """\
import json
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

arguments = sys.argv[1:]
selection = next(value.split("=", 1)[1] for value in arguments if value.startswith("--flashinfer-node-file="))
xml_path = Path(next(value.split("=", 1)[1] for value in arguments if value.startswith("--junitxml=")))
nodeid = json.loads(Path(selection).read_text(encoding="utf-8"))[0]
print(
    '@@flashinfer-pytest-event@@ ' + json.dumps(
        {"event": "start", "nodeid": nodeid, "started_at": "not-a-time"}
    ),
    flush=True,
)
xml_path.write_text(
    "<testsuites><testsuite tests=\\"1\\"><testcase name=\\"case\\" time=\\"0\\">"
    "<properties><property name=\\"pytest_nodeid\\" value=" + quoteattr(nodeid) + "/>"
    "</properties></testcase></testsuite></testsuites>",
    encoding="utf-8",
)
""",
    )

    result = execute_batch(_fake_batch_request(tmp_path))

    assert result.status == "infrastructure"
    assert "pytest output reader failed" in result.diagnostic
