from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from scripts.test_sharding.junit import (
    SyntheticBatchMetadata,
    annotate_batch_xml,
    create_synthetic_batch_xml,
    finalized_batch_outcomes,
    validate_batch_xml,
)


def _write_batch(path: Path, nodeid: str, *, failed: bool = False) -> None:
    failure = '<failure message="boom">details</failure>' if failed else ""
    path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" tests="1" failures="{int(failed)}" errors="0" skipped="0" time="1.25">
  <testcase classname="sample" name="case" time="1.25"><properties><property name="pytest_nodeid" value="{nodeid}"/></properties>{failure}</testcase>
</testsuite></testsuites>
""",
        encoding="utf-8",
    )


def test_batch_is_final_only_when_exact_node_coverage_matches(tmp_path: Path) -> None:
    report = tmp_path / "batch.xml"
    _write_batch(report, "tests/test_sample.py::test_case")

    valid = validate_batch_xml(report, ["tests/test_sample.py::test_case"])
    missing = validate_batch_xml(report, ["tests/test_sample.py::test_other"])

    assert valid.valid is True
    assert valid.cases[0].outcome == "passed"
    assert valid.cases[0].seconds == 1.25
    assert missing.valid is False
    assert "missing" in missing.diagnostics[0]


def test_finalized_batch_outcomes_preserve_unknown_plugin_results(
    tmp_path: Path,
) -> None:
    report = tmp_path / "batch.xml"
    nodeid = "tests/test_sample.py::test_case"
    _write_batch(report, nodeid)
    report.with_name("batch.results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "nodeid": nodeid,
                        "outcome": "unknown",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    outcomes, diagnostics = finalized_batch_outcomes(report, [nodeid])

    assert diagnostics == ()
    assert outcomes == Counter({"unknown": 1})


def test_synthetic_timeout_report_is_self_describing(tmp_path: Path) -> None:
    output = tmp_path / "synthetic.xml"
    nodes = ["tests/test_sample.py::test_a", "tests/test_sample.py::test_b"]

    create_synthetic_batch_xml(
        output,
        nodes,
        SyntheticBatchMetadata(
            policy="skip",
            batch_id="batch-1",
            unit_id="unit-1",
            shard_index=0,
            attempt_id="attempt-1",
            profile="profile",
        ),
    )

    validation = validate_batch_xml(output, nodes)
    assert validation.valid is True
    assert [case.outcome for case in validation.cases] == ["skipped", "skipped"]
    root = ET.parse(output).getroot()
    testcase_properties = {
        prop.attrib["name"]: prop.attrib["value"]
        for prop in root.findall(".//testcase/properties/property")
    }
    assert testcase_properties["synthetic"] == "true"


def test_annotation_normalizes_a_bare_testsuite_root(tmp_path: Path) -> None:
    report = tmp_path / "batch.xml"
    report.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="1" failures="0" errors="0" skipped="0" time="1">
  <testcase classname="sample" name="case" time="1"><properties><property name="pytest_nodeid" value="tests/test_sample.py::test_case"/></properties></testcase>
</testsuite>
""",
        encoding="utf-8",
    )

    annotate_batch_xml(report, {"batch_id": "batch-1"})

    root = ET.parse(report).getroot()
    assert root.tag == "testsuites"
    assert root.find("./testsuite") is not None
    properties = {
        prop.attrib["name"]: prop.attrib["value"]
        for prop in root.findall("./testsuite/properties/property")
    }
    assert properties["batch_id"] == "batch-1"
