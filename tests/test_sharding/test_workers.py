from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.test_sharding.models import Batch, Unit
from scripts.test_sharding.workers import BatchExecutionRequest, execute_batch


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
    assert not list((tmp_path / "junit").glob("units/*/batches/*.xml"))
