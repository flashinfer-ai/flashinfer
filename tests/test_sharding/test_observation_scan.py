from __future__ import annotations

import json
from pathlib import Path
import zipfile

from scripts.test_sharding.estimates import DurationEstimate, EstimateBook
from scripts.test_sharding.models import CollectedNode, PlanningOptions
from scripts.test_sharding.planner import build_plan
from scripts.test_sharding.scanner import (
    scan_cleaned_artifact_dirs,
    scan_observation_inputs,
)
from scripts.test_sharding.summary import batch_xml_path


def test_scanner_reads_authoritative_batch_xml(
    tmp_path: Path,
) -> None:
    nodes = [
        CollectedNode.from_nodeid("tests/test_sample.py::test_case[0]", 0),
        CollectedNode.from_nodeid("tests/test_sample.py::test_case[1]", 1),
    ]
    estimates = EstimateBook(
        [DurationEstimate("profile", node.nodeid, 1, 1) for node in nodes]
    )
    plan = build_plan(
        nodes,
        estimates,
        PlanningOptions(
            profile="profile",
            checkpoint_seconds=10,
            target_unit_seconds=10,
            unknown_case_seconds=5,
            shard_count=1,
        ),
    )
    run_dir = tmp_path / "junit"
    manifest_path = run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source_git_sha": "run-identity",
                "selection": {"sanity_test": False},
                "test_path": "tests",
                "plan": plan.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    unit = plan.units[0]
    batch = unit.batches[0]
    raw = batch_xml_path(run_dir, unit, batch)
    raw.parent.mkdir(parents=True)
    testcases = "".join(
        f'<testcase classname="sample" name="case" time="{index + 1}"><properties>'
        f'<property name="pytest_nodeid" value="{node.nodeid}"/>'
        "</properties></testcase>"
        for index, node in enumerate(nodes)
    )
    raw.write_text(
        f'<testsuites><testsuite name="{batch.id}" tests="2" failures="0" '
        f'errors="0" skipped="0" time="3"><properties><property name="batch_id" '
        f'value="{batch.id}"/></properties>{testcases}</testsuite></testsuites>',
        encoding="utf-8",
    )
    raw.with_name(f"{batch.id}.meta.json").write_text(
        json.dumps({"attempt_id": "attempt-1"}), encoding="utf-8"
    )
    raw.with_name(f"{batch.id}.telemetry.json").write_text(
        json.dumps(
            {
                "process_launch": 1,
                "collection_complete": 2,
                "first_case_start": 2.5,
                "process_exit": 6,
            }
        ),
        encoding="utf-8",
    )
    observations, overheads, diagnostics = scan_observation_inputs([run_dir])

    assert diagnostics == []
    assert [item.nodeid for item in observations] == [node.nodeid for node in nodes]
    assert [item.seconds for item in observations] == [1, 2]
    assert len(overheads) == 1
    assert overheads[0].process_startup_seconds == 1
    assert overheads[0].source_warmup_seconds == 0.5


def test_scanner_recovers_cleaned_gitlab_artifacts(
    tmp_path: Path,
) -> None:
    long_node = (
        "tests/test_sample.py::test_case[" + "parameter-value-" * 10 + "complete]"
    )
    log_only_node = "tests/test_log_only.py::test_failure[value]"
    nodes = [
        CollectedNode.from_nodeid(long_node, 0),
        CollectedNode.from_nodeid(log_only_node, 1),
    ]
    estimates = EstimateBook(
        [DurationEstimate("planning-profile", node.nodeid, 1, 1) for node in nodes]
    )
    plan = build_plan(
        nodes,
        estimates,
        PlanningOptions(
            profile="planning-profile",
            checkpoint_seconds=10,
            target_unit_seconds=10,
            unknown_case_seconds=5,
            shard_count=1,
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"plan": plan.to_dict()}),
        encoding="utf-8",
    )
    batch = next(
        batch
        for unit in plan.units
        for batch in unit.batches
        if long_node in batch.nodeids
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    archive_path = artifact_dir / "job.zip"
    truncated_node = long_node[:100]
    xml = (
        '<testsuites><testsuite name="pytest" tests="1" failures="0" '
        'errors="0" skipped="0" time="2.5"><properties>'
        f'<property name="batch_id" value="{batch.id}"/>'
        '<property name="timing_profile" value="artifact-profile"/>'
        '<property name="synthetic" value="false"/>'
        "</properties>"
        '<testcase classname="sample" name="truncated" time="2.5">'
        "<properties>"
        f'<property name="pytest_nodeid" value="{truncated_node}"/>'
        "</properties></testcase></testsuite></testsuites>"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("report/unit/batches/batch.xml", xml)
    (artifact_dir / "job.log").write_text(
        "PYTEST RESULT worker=0 batch=batch-incomplete outcome=failed "
        f"duration=3.250s node={log_only_node}\n",
        encoding="utf-8",
    )

    observations, overheads, diagnostics = scan_cleaned_artifact_dirs(
        [artifact_dir],
        manifest_path,
    )

    assert overheads == []
    assert diagnostics == []
    by_node = {item.nodeid: item for item in observations}
    assert set(by_node) == {long_node, log_only_node}
    assert by_node[long_node].profile == "artifact-profile"
    assert by_node[long_node].seconds == 2.5
    assert by_node[long_node].outcome == "passed"
    assert by_node[log_only_node].seconds == 3.25
    assert by_node[log_only_node].outcome == "failed"


def test_scanner_omits_ambiguous_truncated_artifact_nodeids(
    tmp_path: Path,
) -> None:
    shared_prefix = "tests/test_sample.py::test_case[" + "x" * 80
    nodes = [
        CollectedNode.from_nodeid(f"{shared_prefix}-one]", 0),
        CollectedNode.from_nodeid(f"{shared_prefix}-two]", 1),
    ]
    plan = build_plan(
        nodes,
        EstimateBook(),
        PlanningOptions(
            profile="planning-profile",
            checkpoint_seconds=10,
            target_unit_seconds=10,
            unknown_case_seconds=5,
            shard_count=1,
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"plan": plan.to_dict()}),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    xml = (
        '<testsuites><testsuite name="pytest" tests="1" failures="0" '
        'errors="0" skipped="0" time="1"><properties>'
        '<property name="batch_id" value="batch-from-another-plan"/>'
        '<property name="timing_profile" value="artifact-profile"/>'
        "</properties>"
        '<testcase classname="sample" name="test_case[" time="1">'
        "<properties>"
        f'<property name="pytest_nodeid" value="{shared_prefix}"/>'
        "</properties></testcase></testsuite></testsuites>"
    )
    with zipfile.ZipFile(artifact_dir / "job.zip", "w") as archive:
        archive.writestr("report/unit/batches/batch.xml", xml)

    observations, overheads, diagnostics = scan_cleaned_artifact_dirs(
        [artifact_dir],
        manifest_path,
    )

    assert observations == []
    assert overheads == []
    assert len(diagnostics) == 1
    assert "omitted 1 ambiguous testcase" in diagnostics[0]


def test_scanner_combines_nodeid_and_testcase_name_prefixes(
    tmp_path: Path,
) -> None:
    source = "tests/" + "long_directory/" * 8 + "test_sample.py"
    selected = f"{source}::test_case[alpha-parameter]"
    nodes = [
        CollectedNode.from_nodeid(selected, 0),
        CollectedNode.from_nodeid(f"{source}::test_case[beta-parameter]", 1),
    ]
    plan = build_plan(
        nodes,
        EstimateBook(),
        PlanningOptions(
            profile="planning-profile",
            checkpoint_seconds=10,
            target_unit_seconds=10,
            unknown_case_seconds=5,
            shard_count=1,
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"plan": plan.to_dict()}),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    xml = (
        '<testsuites><testsuite name="pytest" tests="1" failures="0" '
        'errors="0" skipped="0" time="1"><properties>'
        '<property name="batch_id" value="batch-from-another-plan"/>'
        '<property name="timing_profile" value="artifact-profile"/>'
        "</properties>"
        '<testcase classname="sample" name="test_case[alpha" time="1">'
        "<properties>"
        f'<property name="pytest_nodeid" value="{source[:100]}"/>'
        "</properties></testcase></testsuite></testsuites>"
    )
    with zipfile.ZipFile(artifact_dir / "job.zip", "w") as archive:
        archive.writestr("report/unit/batches/batch.xml", xml)

    observations, overheads, diagnostics = scan_cleaned_artifact_dirs(
        [artifact_dir],
        manifest_path,
    )

    assert overheads == []
    assert diagnostics == []
    assert [item.nodeid for item in observations] == [selected]


def test_scanner_omits_prefix_resolution_collisions_across_batches(
    tmp_path: Path,
) -> None:
    nodeid = "tests/test_sample.py::test_case[" + "x" * 100 + "]"
    node = CollectedNode.from_nodeid(nodeid, 0)
    plan = build_plan(
        [node],
        EstimateBook(),
        PlanningOptions(
            profile="planning-profile",
            checkpoint_seconds=10,
            target_unit_seconds=10,
            unknown_case_seconds=5,
            shard_count=1,
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"plan": plan.to_dict()}),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    prefix = nodeid[:100]

    def xml(batch_id: str) -> str:
        return (
            '<testsuites><testsuite name="pytest" tests="1" failures="0" '
            'errors="0" skipped="0" time="1"><properties>'
            f'<property name="batch_id" value="{batch_id}"/>'
            '<property name="timing_profile" value="artifact-profile"/>'
            "</properties>"
            '<testcase classname="sample" name="test_case[" time="1">'
            "<properties>"
            f'<property name="pytest_nodeid" value="{prefix}"/>'
            "</properties></testcase></testsuite></testsuites>"
        )

    with zipfile.ZipFile(artifact_dir / "job.zip", "w") as archive:
        archive.writestr("report/batch-one.xml", xml("batch-from-old-plan-one"))
        archive.writestr("report/batch-two.xml", xml("batch-from-old-plan-two"))

    observations, overheads, diagnostics = scan_cleaned_artifact_dirs(
        [artifact_dir],
        manifest_path,
    )

    assert observations == []
    assert overheads == []
    assert len(diagnostics) == 1
    assert "omitted 2 prefix-resolved observations" in diagnostics[0]
