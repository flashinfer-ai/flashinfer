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
from scripts.test_sharding.summary import (
    batch_xml_path,
    exit_code_for_shard,
    exit_code_for_summary,
    terminal_summary_lines,
)


def test_summary_publication_waits_for_the_state_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = Plan(
        options=PlanningOptions(),
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
    assert summary["schema_version"] == 3


def test_failure_exit_code_wins_over_incomplete() -> None:
    summary = {
        "infrastructure_errors": [],
        "outcomes": {"failed": 1},
        "complete": False,
    }

    assert exit_code_for_summary(summary) == 1  # type: ignore[arg-type]


def _summary_plan() -> Plan:
    nodes = tuple(
        CollectedNode.from_nodeid(f"tests/test_sample.py::test_case_{index}", index)
        for index in range(2)
    )
    units = []
    for index, node in enumerate(nodes):
        batch = Batch(
            id=f"batch-{index}",
            source_file=node.source_file,
            nodeids=(node.nodeid,),
            estimated_ms=1000,
            overhead_ms=0,
            oversized=False,
        )
        units.append(
            Unit(
                id=f"unit-{index}",
                batches=(batch,),
                estimated_ms=1000,
                oversized=False,
                shard_index=0,
            )
        )
    return Plan(options=PlanningOptions(), nodes=nodes, units=tuple(units))


def _write_final_batch(
    junit_dir: Path,
    unit: Unit,
    *,
    failed: bool,
    launched_at: float,
    exited_at: float,
    rss_mib: float,
    gpu_mib: float,
    monitor_memory: bool,
) -> None:
    batch = unit.batches[0]
    path = batch_xml_path(junit_dir, unit, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    failure = (
        '<failure message="boom">detailed pytest failure</failure>' if failed else ""
    )
    path.write_text(
        '<testsuites><testsuite><testcase time="1">'
        f'<properties><property name="pytest_nodeid" value="{batch.nodeids[0]}"/>'
        "</properties>"
        f"{failure}</testcase></testsuite></testsuites>",
        encoding="utf-8",
    )
    path.with_name(f"{batch.id}.results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "nodeid": batch.nodeids[0],
                        "outcome": "failed" if failed else "passed",
                        "longrepr": "detailed pytest failure" if failed else "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    path.with_name(f"{batch.id}.meta.json").write_text(
        json.dumps(
            {
                "attempt_id": "attempt-1",
                "launched_at": launched_at,
                "exited_at": exited_at,
                "synthetic": False,
                "monitor_memory": monitor_memory,
            }
        ),
        encoding="utf-8",
    )
    path.with_name(f"{batch.id}.memory.csv").write_text(
        f"timestamp,host_rss_mib,gpu_memory_mib\n1,{rss_mib},{gpu_mib}\n",
        encoding="utf-8",
    )


def test_source_summary_aggregates_local_resources_and_failed_diagnostics(
    tmp_path: Path,
) -> None:
    plan = _summary_plan()
    _write_final_batch(
        tmp_path,
        plan.units[0],
        failed=False,
        launched_at=10,
        exited_at=12,
        rss_mib=100,
        gpu_mib=900,
        monitor_memory=True,
    )
    _write_final_batch(
        tmp_path,
        plan.units[1],
        failed=True,
        launched_at=20,
        exited_at=25,
        rss_mib=300,
        gpu_mib=400,
        monitor_memory=False,
    )

    summary = summary_module.publish_summary(tmp_path, plan)

    source = summary["sources"][0]
    assert source == {
        "shard_index": 0,
        "source_file": "tests/test_sample.py",
        "planned_nodes": 2,
        "finalized_nodes": 2,
        "pending_nodes": 0,
        "passed": 1,
        "failed": 1,
        "skipped": 0,
        "unknown": 0,
        "synthetic": 0,
        "process_seconds": 7.0,
        "max_host_rss_mib": 300.0,
        "max_gpu_memory_mib": 900.0,
        "memory_samples": 2,
        "partial_resources": True,
        "status": "failed",
    }
    failure = summary["failed_nodes"][0]
    assert failure["nodeid"].endswith("::test_case_1")
    assert failure["diagnostic"] == "detailed pytest failure"
    assert failure["results_path"].endswith("batch-1.results.json")
    with (tmp_path / "source_summary.csv").open(newline="", encoding="utf-8") as stream:
        csv_row = next(csv.DictReader(stream))
    assert csv_row["status"] == "failed"
    assert csv_row["partial_resources"] == "True"

    terminal = "\n".join(
        terminal_summary_lines(
            summary,
            shard_index=0,
            runner_exit_code=1,
            status="complete-with-failures",
            test_started_at=100.0,
            test_ended_at=101.0,
            cause="pytest failure",
            diagnostic_limit=8,
        )
    )
    assert "[diagnostic truncated at 8 bytes]" in terminal
    assert "log:" in terminal
    assert "  - tests/test_sample.py::test_case_1" in terminal


def test_terminal_summary_lists_detailed_failures_before_deduplicated_files(
    tmp_path: Path,
) -> None:
    plan = _summary_plan()
    for index, unit in enumerate(plan.units):
        _write_final_batch(
            tmp_path,
            unit,
            failed=True,
            launched_at=10 + index,
            exited_at=11 + index,
            rss_mib=100,
            gpu_mib=200,
            monitor_memory=True,
        )
    summary = summary_module.publish_summary(tmp_path, plan)

    terminal_lines = terminal_summary_lines(
        summary,
        shard_index=0,
        runner_exit_code=1,
        status="complete-with-failures",
        test_started_at=100.0,
        test_ended_at=101.0,
    )
    summary_at = terminal_lines.index("TEST SUMMARY")
    detail_output = "\n".join(terminal_lines[:summary_at])
    summary_output = "\n".join(terminal_lines[summary_at:])

    assert detail_output.count("tests/test_sample.py::test_case_") == 2
    assert detail_output.count("detailed pytest failure") == 2
    assert "Failed test files:" not in detail_output
    assert summary_output.count("  - tests/test_sample.py") == 1
    assert "tests/test_sample.py::test_case_" not in summary_output
    assert "detailed pytest failure" not in summary_output


def test_terminal_summary_skips_failure_sections_when_no_test_failed(
    tmp_path: Path,
) -> None:
    plan = _summary_plan()
    for index, unit in enumerate(plan.units):
        _write_final_batch(
            tmp_path,
            unit,
            failed=False,
            launched_at=10 + index,
            exited_at=11 + index,
            rss_mib=100,
            gpu_mib=200,
            monitor_memory=True,
        )
    summary = summary_module.publish_summary(tmp_path, plan)

    terminal = "\n".join(
        terminal_summary_lines(
            summary,
            shard_index=0,
            runner_exit_code=0,
            status="complete-without-failures",
            test_started_at=100.0,
            test_ended_at=101.0,
        )
    )

    assert "FAILED TEST NODES" not in terminal
    assert "Failed test files:" not in terminal


def test_terminal_summary_reports_test_times() -> None:
    terminal = "\n".join(
        terminal_summary_lines(
            None,
            shard_index=None,
            runner_exit_code=3,
            status="configuration-collection-or-infrastructure-error",
            test_started_at=1_700_000_000.0,
            test_ended_at=1_700_000_005.25,
        )
    )

    assert "Start time: 2023-11-14T22:13:20Z" in terminal
    assert "End time: 2023-11-14T22:13:25Z" in terminal
    assert "Time elapsed: 5s" in terminal


def test_shard_exit_code_ignores_other_shard_infrastructure_errors() -> None:
    summary = {
        "shards": {
            "0": {"complete": True, "outcomes": {}},
            "1": {"complete": False, "outcomes": {}},
        },
        "shard_infrastructure_errors": {"0": [], "1": ["gpu disappeared"]},
    }

    assert exit_code_for_shard(summary, 0) == 0  # type: ignore[arg-type]
    assert exit_code_for_shard(summary, 1) == 3  # type: ignore[arg-type]
