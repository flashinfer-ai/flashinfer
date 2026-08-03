from __future__ import annotations

import csv
import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pytest

from scripts.test_sharding import summary as summary_module
from scripts.test_sharding.models import (
    Batch,
    CollectedNode,
    Plan,
    PlanningOptions,
    Unit,
)
from scripts.test_sharding.state import state_lock
from scripts.test_sharding.summary import batch_xml_path


def _write_final_batch(
    junit_dir: Path,
    unit: Unit,
    batch: Batch,
    outcomes: dict[str, str],
    *,
    launched_at: float,
    exited_at: float,
    memory: list[tuple[float, float, float]],
    monitor_memory: bool | None = True,
    failure_reasons: dict[str, str] | None = None,
) -> None:
    path = batch_xml_path(junit_dir, unit, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    testcases = []
    for nodeid in batch.nodeids:
        outcome = outcomes[nodeid]
        result = (
            f"<failure message={quoteattr((failure_reasons or {}).get(nodeid, 'failed'))}/>"
            if outcome == "failed"
            else '<skipped message="skipped"/>'
            if outcome == "skipped"
            else ""
        )
        testcases.append(
            '<testcase classname="sample" name="case" time="1">'
            f'<properties><property name="pytest_nodeid" value="{nodeid}"/>'
            f"</properties>{result}</testcase>"
        )
    path.write_text(
        '<testsuites><testsuite name="pytest" tests="'
        f'{len(batch.nodeids)}">{"".join(testcases)}</testsuite></testsuites>',
        encoding="utf-8",
    )
    metadata = {
        "attempt_id": "attempt-1",
        "launched_at": launched_at,
        "exited_at": exited_at,
    }
    if monitor_memory is not None:
        metadata["monitor_memory"] = monitor_memory
    path.with_name(f"{batch.id}.meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    path.with_name(f"{batch.id}.results.json").write_text(
        json.dumps(
            {
                "results": [
                    {"nodeid": nodeid, "outcome": outcomes[nodeid]}
                    for nodeid in batch.nodeids
                ]
            }
        ),
        encoding="utf-8",
    )
    with path.with_name(f"{batch.id}.memory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "host_rss_mib", "gpu_memory_mib"])
        writer.writerows(memory)


def _summary_plan() -> Plan:
    alpha_nodes = tuple(
        CollectedNode.from_nodeid(f"tests/test_alpha.py::test_case[{index}]", index)
        for index in range(3)
    )
    beta_node = CollectedNode.from_nodeid("tests/test_beta.py::test_pending", 3)
    gamma_nodes = tuple(
        CollectedNode.from_nodeid(f"tests/test_gamma.py::test_case[{index}]", index + 4)
        for index in range(3)
    )

    def unit(identifier: str, nodes: tuple[CollectedNode, ...]) -> Unit:
        batch = Batch(
            id=f"batch-{identifier}",
            source_file=nodes[0].source_file,
            nodeids=tuple(node.nodeid for node in nodes),
            estimated_ms=1000,
            overhead_ms=0,
            oversized=False,
        )
        return Unit(
            id=f"unit-{identifier}",
            batches=(batch,),
            estimated_ms=1000,
            oversized=False,
            shard_index=0,
        )

    nodes = (*alpha_nodes, beta_node, *gamma_nodes)
    return Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=nodes,
        units=(
            unit("alpha-0", alpha_nodes[:1]),
            unit("alpha-1", alpha_nodes[1:2]),
            unit("alpha-2", alpha_nodes[2:]),
            unit("beta", (beta_node,)),
            unit("gamma", gamma_nodes),
        ),
    )


def test_summary_publication_waits_for_the_state_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=(),
        units=(),
    )
    lock_attempted = threading.Event()

    @contextmanager
    def observed_state_lock(junit_dir: Path) -> Iterator[None]:
        lock_attempted.set()
        with state_lock(junit_dir):
            yield

    monkeypatch.setattr(summary_module, "state_lock", observed_state_lock)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with state_lock(tmp_path):
            future = executor.submit(summary_module.publish_summary, tmp_path, plan)
            assert lock_attempted.wait(timeout=5)
            assert not future.done()
        summary = future.result(timeout=5)
    finally:
        executor.shutdown()

    assert summary["complete"] is True
    assert summary["schema_version"] == 2


def test_console_summary_counts_nodes_and_aggregates_failed_files(
    tmp_path: Path,
) -> None:
    plan = _summary_plan()
    alpha_first_unit, alpha_second_unit, _, _, gamma_unit = plan.units
    alpha_first_batch = alpha_first_unit.batches[0]
    alpha_second_batch = alpha_second_unit.batches[0]
    gamma_batch = gamma_unit.batches[0]
    _write_final_batch(
        tmp_path,
        alpha_first_unit,
        alpha_first_batch,
        {alpha_first_batch.nodeids[0]: "passed"},
        launched_at=10,
        exited_at=3610,
        memory=[(11, 1024, 2048)],
    )
    _write_final_batch(
        tmp_path,
        alpha_second_unit,
        alpha_second_batch,
        {alpha_second_batch.nodeids[0]: "failed"},
        launched_at=20,
        exited_at=143,
        memory=[(21, 2048, 4096)],
        failure_reasons={
            alpha_second_batch.nodeids[0]: (
                "AssertionError: quantized output element mismatch: "
                "64/256 elements differ"
            )
        },
    )
    _write_final_batch(
        tmp_path,
        gamma_unit,
        gamma_batch,
        {
            gamma_batch.nodeids[0]: "passed",
            gamma_batch.nodeids[1]: "skipped",
            gamma_batch.nodeids[2]: "unknown",
        },
        launched_at=20,
        exited_at=80,
        memory=[],
    )

    report = summary_module.format_test_run_summary(tmp_path, plan, shard_index=0)

    assert "TEST SUMMARY - SHARD 0" in report
    assert "Total test files: 3" in report
    assert "Total test nodes: 7" in report
    assert "Passed: 2" in report
    assert "Failed: 1" in report
    assert "Skipped: 1" in report
    assert "Unknown: 1" in report
    assert "No result: 2" in report
    assert (
        "Failed test nodes:\n"
        "  1. tests/test_alpha.py::test_case[1] - "
        "AssertionError: quantized output element mismatch: "
        "64/256 elements differ" in report
    )
    assert report.index("Failed test nodes:") < report.index("Total test files:")
    assert "  - tests/test_alpha.py - 1/3 failed" in report
    assert "  - tests/test_alpha.py - 1/3 pending" in report
    assert "  - tests/test_beta.py - 1/1 pending" in report
    assert "Top 10 longest-running test files:" in report
    assert (
        "tests/test_alpha.py - duration 1h02m03s, peak RSS 2.0 GiB, "
        "peak GPU 4.0 GiB, samples 2" in report
    )
    longest_section = report.split("Top 10 longest-running test files:", 1)[1].split(
        "Top 10 highest host RSS test files:", 1
    )[0]
    host_rss_section = report.split("Top 10 highest host RSS test files:", 1)[1]
    assert "tests/test_gamma.py" in longest_section
    assert "tests/test_gamma.py" not in host_rss_section

    summary_module.publish_summary(tmp_path, plan)
    with (tmp_path / "source_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = {row["source_file"]: row for row in csv.DictReader(stream)}
    assert rows["tests/test_alpha.py"]["status"] == "failed-partial"
    assert rows["tests/test_alpha.py"]["finalized_nodes"] == "2"
    assert rows["tests/test_alpha.py"]["pending_nodes"] == "1"


def test_publish_summary_writes_shard_source_csv(tmp_path: Path) -> None:
    plan = _summary_plan()

    summary_module.publish_summary(tmp_path, plan)

    with (tmp_path / "source_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["source_file"] for row in rows] == [
        "tests/test_alpha.py",
        "tests/test_beta.py",
        "tests/test_gamma.py",
    ]
    assert rows[0]["shard_index"] == "0"
    assert rows[0]["planned_nodes"] == "3"
    assert rows[1]["pending_nodes"] == "1"


def test_global_summary_merges_a_source_split_across_shards(tmp_path: Path) -> None:
    nodes = tuple(
        CollectedNode.from_nodeid(f"tests/test_split.py::test_case[{index}]", index)
        for index in range(2)
    )
    units = []
    for shard_index, node in enumerate(nodes):
        batch = Batch(
            id=f"batch-{shard_index}",
            source_file=node.source_file,
            nodeids=(node.nodeid,),
            estimated_ms=1000,
            overhead_ms=0,
            oversized=False,
        )
        units.append(
            Unit(
                id=f"unit-{shard_index}",
                batches=(batch,),
                estimated_ms=1000,
                oversized=False,
                shard_index=shard_index,
            )
        )
    plan = Plan(
        options=PlanningOptions(profile="synthetic", shard_count=2),
        nodes=nodes,
        units=tuple(units),
    )
    _write_final_batch(
        tmp_path,
        units[0],
        units[0].batches[0],
        {nodes[0].nodeid: "failed"},
        launched_at=10,
        exited_at=40,
        memory=[(11, 100, 200)],
    )
    _write_final_batch(
        tmp_path,
        units[1],
        units[1].batches[0],
        {nodes[1].nodeid: "passed"},
        launched_at=20,
        exited_at=80,
        memory=[(21, 300, 150)],
    )

    report = summary_module.format_test_run_summary(tmp_path, plan)

    assert "TEST SUMMARY\n" in report
    assert "Total test files: 1" in report
    assert "Total test nodes: 2" in report
    assert "Passed: 1" in report
    assert "Failed: 1" in report
    assert "  - tests/test_split.py - 1/2 failed" in report
    assert (
        "tests/test_split.py - duration 1m30s, peak RSS 300 MiB, "
        "peak GPU 200 MiB, samples 2" in report
    )

    summary_module.publish_summary(tmp_path, plan)
    with (tmp_path / "source_summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["shard_index"], row["source_file"]) for row in rows] == [
        ("0", "tests/test_split.py"),
        ("1", "tests/test_split.py"),
    ]


def test_failed_nodes_are_all_numbered_in_nodeid_order_and_scoped_to_shard(
    tmp_path: Path,
) -> None:
    nodes = tuple(
        CollectedNode.from_nodeid(f"tests/test_failed.py::test_case[{index}]", index)
        for index in (1, 0)
    )
    units = []
    for shard_index, node in enumerate(nodes):
        batch = Batch(
            id=f"batch-failed-{shard_index}",
            source_file=node.source_file,
            nodeids=(node.nodeid,),
            estimated_ms=1000,
            overhead_ms=0,
            oversized=False,
        )
        unit = Unit(
            id=f"unit-failed-{shard_index}",
            batches=(batch,),
            estimated_ms=1000,
            oversized=False,
            shard_index=shard_index,
        )
        units.append(unit)
        _write_final_batch(
            tmp_path,
            unit,
            batch,
            {node.nodeid: "failed"},
            launched_at=10,
            exited_at=20,
            memory=[],
            failure_reasons={node.nodeid: f"reason {node.order}"},
        )
    plan = Plan(
        options=PlanningOptions(profile="synthetic", shard_count=2),
        nodes=nodes,
        units=tuple(units),
    )

    global_report = summary_module.format_test_run_summary(tmp_path, plan)
    shard_report = summary_module.format_test_run_summary(tmp_path, plan, shard_index=0)

    first = "  1. tests/test_failed.py::test_case[0] - reason 0"
    second = "  2. tests/test_failed.py::test_case[1] - reason 1"
    assert first in global_report
    assert second in global_report
    assert global_report.index(first) < global_report.index(second)
    assert "  1. tests/test_failed.py::test_case[1] - reason 1" in shard_report
    assert "test_case[0] - reason 0" not in shard_report


def test_disabled_memory_monitoring_is_reported_and_not_partial(
    tmp_path: Path,
) -> None:
    node = CollectedNode.from_nodeid("tests/test_nomem.py::test_case", 0)
    batch = Batch(
        id="batch-nomem",
        source_file=node.source_file,
        nodeids=(node.nodeid,),
        estimated_ms=1000,
        overhead_ms=0,
        oversized=False,
    )
    unit = Unit(
        id="unit-nomem",
        batches=(batch,),
        estimated_ms=1000,
        oversized=False,
        shard_index=0,
    )
    plan = Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=(node,),
        units=(unit,),
    )
    _write_final_batch(
        tmp_path,
        unit,
        batch,
        {node.nodeid: "passed"},
        launched_at=10,
        exited_at=25,
        memory=[],
        monitor_memory=False,
    )

    report = summary_module.format_test_run_summary(tmp_path, plan, shard_index=0)

    assert "Memory monitoring: disabled" in report
    assert (
        "tests/test_nomem.py - duration 15s, peak RSS 0 MiB, "
        "peak GPU 0 MiB, samples 0" in report
    )
    assert "samples 0 (partial)" not in report

    summary_module.publish_summary(tmp_path, plan)
    with (tmp_path / "source_summary.csv").open(newline="", encoding="utf-8") as stream:
        [row] = list(csv.DictReader(stream))
    assert row["partial_resources"] == "false"


def test_legacy_batch_without_monitoring_metadata_is_conservatively_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = CollectedNode.from_nodeid("tests/test_legacy.py::test_case", 0)
    batch = Batch(
        id="batch-legacy",
        source_file=node.source_file,
        nodeids=(node.nodeid,),
        estimated_ms=1000,
        overhead_ms=0,
        oversized=False,
    )
    unit = Unit(
        id="unit-legacy",
        batches=(batch,),
        estimated_ms=1000,
        oversized=False,
        shard_index=0,
    )
    plan = Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=(node,),
        units=(unit,),
    )
    _write_final_batch(
        tmp_path,
        unit,
        batch,
        {node.nodeid: "passed"},
        launched_at=10,
        exited_at=25,
        memory=[],
        monitor_memory=None,
    )

    monkeypatch.setenv("MONITOR_TEST_MEMORY", "0")
    disabled_environment = summary_module.format_test_run_summary(
        tmp_path, plan, shard_index=0
    )
    monkeypatch.setenv("MONITOR_TEST_MEMORY", "true")
    enabled_environment = summary_module.format_test_run_summary(
        tmp_path, plan, shard_index=0
    )

    assert disabled_environment == enabled_environment
    assert "Memory monitoring: disabled" not in disabled_environment
    assert "samples 0 (partial)" in disabled_environment


@pytest.mark.parametrize("attempt_values", [(False, True), (True, False)])
def test_resumed_attempts_report_mixed_memory_monitoring_deterministically(
    tmp_path: Path,
    attempt_values: tuple[bool, bool],
) -> None:
    node = CollectedNode.from_nodeid("tests/test_pending.py::test_case", 0)
    batch = Batch(
        id="batch-pending",
        source_file=node.source_file,
        nodeids=(node.nodeid,),
        estimated_ms=1000,
        overhead_ms=0,
        oversized=False,
    )
    plan = Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=(node,),
        units=(
            Unit(
                id="unit-pending",
                batches=(batch,),
                estimated_ms=1000,
                oversized=False,
                shard_index=0,
            ),
        ),
    )
    for number, monitor_memory in enumerate(attempt_values, start=1):
        attempt = tmp_path / "attempts" / f"attempt-{number:04d}"
        shard_settings = attempt / "shards" / "shard-0000.settings.json"
        shard_settings.parent.mkdir(parents=True)
        (attempt / "attempt.json").write_text("{}", encoding="utf-8")
        shard_settings.write_text(
            json.dumps({"monitor_memory": monitor_memory}), encoding="utf-8"
        )

    report = summary_module.format_test_run_summary(tmp_path, plan, shard_index=0)

    assert "Memory monitoring: partially disabled (shards: 0)" in report


def test_malformed_memory_monitoring_settings_are_reported_as_unknown(
    tmp_path: Path,
) -> None:
    plan = Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=(),
        units=(),
    )
    attempt = tmp_path / "attempts" / "attempt-0001"
    settings = attempt / "shards" / "shard-0000.settings.json"
    settings.parent.mkdir(parents=True)
    (attempt / "attempt.json").write_text("{}", encoding="utf-8")
    settings.write_text("[]", encoding="utf-8")

    report = summary_module.format_test_run_summary(tmp_path, plan, shard_index=0)

    assert "Memory monitoring: mixed or unknown (shards: 0)" in report


def test_no_monitoring_evidence_is_environment_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _summary_plan()

    monkeypatch.setenv("MONITOR_TEST_MEMORY", "0")
    disabled_environment = summary_module.format_test_run_summary(
        tmp_path, plan, shard_index=0
    )
    monkeypatch.setenv("MONITOR_TEST_MEMORY", "true")
    enabled_environment = summary_module.format_test_run_summary(
        tmp_path, plan, shard_index=0
    )

    assert disabled_environment == enabled_environment
    assert "Memory monitoring: mixed or unknown (shards: 0)" in disabled_environment
