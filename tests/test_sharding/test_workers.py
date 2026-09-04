from __future__ import annotations

import os
import signal
import time
from contextlib import suppress
from io import StringIO
from pathlib import Path

import pytest

from scripts.test_sharding import workers as workers_module
from scripts.test_sharding.models import Batch, Unit
from scripts.test_sharding.progress import encode_pytest_event
from scripts.test_sharding.workers import (
    BatchExecutionRequest,
    _BatchProgress,
    _forward_pytest_output,
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
