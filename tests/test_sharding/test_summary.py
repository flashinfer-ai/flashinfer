from __future__ import annotations

import csv
import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

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
) -> None:
    path = batch_xml_path(junit_dir, unit, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    testcases = []
    for nodeid in batch.nodeids:
        outcome = outcomes[nodeid]
        result = (
            '<failure message="failed"/>'
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
    path.with_name(f"{batch.id}.meta.json").write_text(
        json.dumps(
            {
                "attempt_id": "attempt-1",
                "launched_at": launched_at,
                "exited_at": exited_at,
            }
        ),
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
