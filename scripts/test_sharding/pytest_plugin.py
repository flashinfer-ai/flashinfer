"""Pytest plugin used by the sharding runner.

The plugin communicates through JSON files so the coordinator never has to parse
pytest's terminal output and never passes hundreds of thousands of node IDs on a
command line.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from .io import atomic_write_json
from .models import CollectedNode
from .progress import encode_pytest_event


def _emit_progress(event: str, **fields: Any) -> None:
    stream = sys.__stdout__
    stream.write(encode_pytest_event(event, **fields) + "\n")
    stream.flush()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flashinfer-sharding")
    group.addoption("--flashinfer-collection-json", metavar="PATH")
    group.addoption("--flashinfer-node-file", metavar="PATH")
    group.addoption("--flashinfer-result-json", metavar="PATH")
    group.addoption("--flashinfer-telemetry-json", metavar="PATH")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "shard_group(name): keep marked nodes from one source in one pytest batch",
    )
    config.addinivalue_line(
        "markers",
        "solo: run every node from this source without overlapping another local unit",
    )
    config._flashinfer_sharding = {  # type: ignore[attr-defined]
        "session_start": time.time(),
        "collection_complete": None,
        "first_case_start": None,
        "report_complete": None,
        "nodes": {},
    }


def _marker_name(item: pytest.Item) -> str | None:
    marker = item.get_closest_marker("shard_group")
    if marker is None:
        return None
    if len(marker.args) != 1 or marker.kwargs or not isinstance(marker.args[0], str):
        raise pytest.UsageError("shard_group requires exactly one string argument")
    if not marker.args[0]:
        raise pytest.UsageError("shard_group name must not be empty")
    return marker.args[0]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    solo_sources = {
        Path(str(item.path)).resolve()
        for item in items
        if item.get_closest_marker("solo") is not None
    }
    collected = []
    for order, item in enumerate(items):
        node = CollectedNode.from_nodeid(item.nodeid, order, _marker_name(item))
        item_path = Path(str(item.path)).resolve()
        try:
            source_file = item_path.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            source_file = str(item_path)
        collected.append(
            CollectedNode(
                nodeid=node.nodeid,
                source_file=source_file,
                base_function=node.base_function,
                order=node.order,
                shard_group=node.shard_group,
                solo=item_path in solo_sources,
            )
        )
    collection_path = config.getoption("--flashinfer-collection-json")
    if collection_path:
        atomic_write_json(
            Path(collection_path),
            {"schema_version": 1, "nodes": [node.to_dict() for node in collected]},
        )

    selection_path = config.getoption("--flashinfer-node-file")
    if selection_path:
        try:
            selected_data = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise pytest.UsageError(
                f"cannot read node selection {selection_path}: {error}"
            ) from error
        if not isinstance(selected_data, list) or not all(
            isinstance(nodeid, str) for nodeid in selected_data
        ):
            raise pytest.UsageError("node selection must be a JSON array of node IDs")
        selected = set(selected_data)
        available = {item.nodeid for item in items}
        missing = sorted(selected - available)
        if missing:
            raise pytest.UsageError(
                "selected node IDs were not collected: " + ", ".join(missing[:5])
            )
        keep = [item for item in items if item.nodeid in selected]
        deselected = [item for item in items if item.nodeid not in selected]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = keep

    for item in items:
        item.user_properties.append(("pytest_nodeid", item.nodeid))


def pytest_collection_finish(session: pytest.Session) -> None:
    state = session.config._flashinfer_sharding  # type: ignore[attr-defined]
    state["collection_complete"] = time.time()


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    del location
    record = _REPORTS.setdefault(
        nodeid,
        {"nodeid": nodeid, "setup": 0.0, "call": 0.0, "teardown": 0.0},
    )
    record["started_at"] = time.time()
    _emit_progress("start", nodeid=nodeid, started_at=record["started_at"])


def pytest_runtest_logfinish(
    nodeid: str, location: tuple[str, int | None, str]
) -> None:
    del location
    record = _REPORTS.setdefault(
        nodeid,
        {"nodeid": nodeid, "setup": 0.0, "call": 0.0, "teardown": 0.0},
    )
    record["finished_at"] = time.time()
    _emit_progress(
        "finish",
        nodeid=nodeid,
        outcome=_final_outcome(record),
        duration_seconds=sum(
            float(record.get(phase, 0.0)) for phase in ("setup", "call", "teardown")
        ),
        finished_at=record["finished_at"],
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    state = item.config._flashinfer_sharding  # type: ignore[attr-defined]
    if state["first_case_start"] is None:
        state["first_case_start"] = time.time()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    config = getattr(report, "config", None)
    del config
    # TestReport does not expose Config. The per-process recorder is attached to
    # the plugin module by pytest_runtest_makereport below.
    record = _REPORTS.setdefault(
        report.nodeid,
        {"nodeid": report.nodeid, "setup": 0.0, "call": 0.0, "teardown": 0.0},
    )
    record[report.when] = float(report.duration)
    record[f"{report.when}_outcome"] = report.outcome
    if report.failed:
        record["longrepr"] = str(report.longrepr)


_REPORTS: dict[str, dict[str, Any]] = {}


def _final_outcome(record: dict[str, Any]) -> str:
    outcomes = [
        record.get(f"{phase}_outcome") for phase in ("setup", "call", "teardown")
    ]
    if "failed" in outcomes:
        return "failed"
    if "skipped" in outcomes:
        return "skipped"
    if record.get("call_outcome") == "passed":
        return "passed"
    return "unknown"


def pytest_sessionfinish(
    session: pytest.Session, exitstatus: int | pytest.ExitCode
) -> None:
    now = time.time()
    state = session.config._flashinfer_sharding  # type: ignore[attr-defined]
    state["report_complete"] = now
    result_path = session.config.getoption("--flashinfer-result-json")
    if result_path:
        results = []
        for nodeid in sorted(_REPORTS, key=lambda value: value.encode("utf-8")):
            record = dict(_REPORTS[nodeid])
            record["outcome"] = _final_outcome(record)
            results.append(record)
        atomic_write_json(
            Path(result_path),
            {
                "schema_version": 1,
                "exitstatus": int(exitstatus),
                "results": results,
            },
        )
    telemetry_path = session.config.getoption("--flashinfer-telemetry-json")
    if telemetry_path:
        atomic_write_json(
            Path(telemetry_path),
            {
                "schema_version": 1,
                "process_id": os.getpid(),
                "session_start": state["session_start"],
                "collection_complete": state["collection_complete"],
                "first_case_start": state["first_case_start"],
                "report_complete": state["report_complete"],
                "process_exit": now,
            },
        )
    _REPORTS.clear()
