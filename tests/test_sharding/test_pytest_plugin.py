from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.test_sharding.progress import decode_pytest_event


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pytest(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "scripts.test_sharding.pytest_plugin",
            *args,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_plugin_collects_groups_and_selects_exact_nodeids(tmp_path: Path) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        """\
import pytest

@pytest.mark.shard_group("shared-compile")
@pytest.mark.parametrize("value", [1, 2])
def test_grouped(value):
    assert value > 0

def test_unselected():
    raise AssertionError("must not execute")
""",
        encoding="utf-8",
    )
    collection_path = tmp_path / "collection.json"

    collected = _pytest(
        tmp_path,
        "--collect-only",
        f"--flashinfer-collection-json={collection_path}",
        str(test_file),
    )

    assert collected.returncode == 0, collected.stdout + collected.stderr
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    grouped = [item for item in collection["nodes"] if "test_grouped" in item["nodeid"]]
    assert len(grouped) == 2
    assert {item["shard_group"] for item in grouped} == {"shared-compile"}

    selected_node = grouped[1]["nodeid"]
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps([selected_node]), encoding="utf-8")
    junit_path = tmp_path / "batch.xml"
    executed = _pytest(
        tmp_path,
        f"--flashinfer-node-file={selection_path}",
        f"--junitxml={junit_path}",
        str(test_file),
    )

    assert executed.returncode == 0, executed.stdout + executed.stderr
    testcases = ET.parse(junit_path).getroot().findall(".//testcase")
    assert len(testcases) == 1
    properties = {
        prop.attrib["name"]: prop.attrib.get("value", "")
        for prop in testcases[0].findall("./properties/property")
    }
    assert properties["pytest_nodeid"] == selected_node


def test_plugin_promotes_scheduling_markers_to_their_sources(
    tmp_path: Path,
) -> None:
    node_exclusive_file = tmp_path / "test_node_exclusive.py"
    node_exclusive_file.write_text(
        """\
import pytest

@pytest.mark.node_exclusive
def test_marked():
    pass

def test_same_source():
    pass
""",
        encoding="utf-8",
    )
    long_file = tmp_path / "test_long.py"
    long_file.write_text(
        """\
import pytest

@pytest.mark.long_running
def test_marked():
    pass

def test_same_source():
    pass
""",
        encoding="utf-8",
    )
    collection_path = tmp_path / "collection.json"
    gpu_file = tmp_path / "test_gpu.py"
    gpu_file.write_text(
        "import pytest\n"
        "@pytest.mark.gpu_exclusive\n"
        "def test_marked(): pass\n"
        "def test_same_source(): pass\n",
        encoding="utf-8",
    )
    host_file = tmp_path / "test_host.py"
    host_file.write_text(
        "import pytest\n"
        "@pytest.mark.host_memory_heavy\n"
        "def test_marked(): pass\n"
        "def test_same_source(): pass\n",
        encoding="utf-8",
    )

    collected = _pytest(
        tmp_path,
        "--strict-markers",
        "--collect-only",
        f"--flashinfer-collection-json={collection_path}",
        str(node_exclusive_file),
        str(long_file),
        str(gpu_file),
        str(host_file),
    )

    assert collected.returncode == 0, collected.stdout + collected.stderr
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    node_exclusive_nodes = [
        node
        for node in collection["nodes"]
        if "test_node_exclusive.py::" in node["nodeid"]
    ]
    long_nodes = [
        node for node in collection["nodes"] if "test_long.py::" in node["nodeid"]
    ]
    assert len(node_exclusive_nodes) == 2
    assert len(long_nodes) == 2
    assert all(node["node_exclusive"] is True for node in node_exclusive_nodes)
    assert all(node["long_running"] is False for node in node_exclusive_nodes)
    assert all(node["long_running"] is True for node in long_nodes)
    assert all(node["node_exclusive"] is False for node in long_nodes)
    gpu_nodes = [
        node for node in collection["nodes"] if "test_gpu.py::" in node["nodeid"]
    ]
    assert len(gpu_nodes) == 2
    assert all(node["gpu_exclusive"] for node in gpu_nodes)
    assert all(not node["node_exclusive"] for node in gpu_nodes)
    host_nodes = [
        node for node in collection["nodes"] if "test_host.py::" in node["nodeid"]
    ]
    assert len(host_nodes) == 2
    assert all(node["host_memory_heavy"] for node in host_nodes)
    assert all(
        not node["node_exclusive"] and not node["gpu_exclusive"] for node in host_nodes
    )
    assert "PytestUnknownMarkWarning" not in collected.stdout + collected.stderr


def test_plugin_emits_detailed_failure_event_as_soon_as_report_is_available(
    tmp_path: Path,
) -> None:
    test_file = tmp_path / "test_failure.py"
    test_file.write_text(
        "def test_failure():\n    assert 1 == 2, 'specific failure detail'\n",
        encoding="utf-8",
    )

    result = _pytest(tmp_path, str(test_file))

    assert result.returncode == 1
    events = [
        event
        for line in result.stdout.splitlines()
        if (event := decode_pytest_event(line)) is not None
    ]
    failure = next(event for event in events if event["event"] == "failure")
    assert failure["nodeid"].endswith("test_failure.py::test_failure")
    assert failure["phase"] == "call"
    assert "specific failure detail" in failure["diagnostic"]
