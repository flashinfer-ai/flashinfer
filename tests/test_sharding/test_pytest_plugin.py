from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


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


def test_plugin_marks_every_node_from_a_solo_source(tmp_path: Path) -> None:
    test_file = tmp_path / "test_solo.py"
    test_file.write_text(
        """\
import pytest

@pytest.mark.solo
def test_marked():
    pass

def test_same_source():
    pass
""",
        encoding="utf-8",
    )
    collection_path = tmp_path / "collection.json"

    collected = _pytest(
        tmp_path,
        "--strict-markers",
        "--collect-only",
        f"--flashinfer-collection-json={collection_path}",
        str(test_file),
    )

    assert collected.returncode == 0, collected.stdout + collected.stderr
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    assert len(collection["nodes"]) == 2
    assert all(node["solo"] is True for node in collection["nodes"])
    assert "PytestUnknownMarkWarning" not in collected.stdout + collected.stderr
