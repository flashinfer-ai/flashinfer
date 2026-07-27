from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.test_sharding import runner
from scripts.test_sharding.models import (
    Batch,
    CollectedNode,
    Plan,
    PlanningOptions,
    Unit,
)
from scripts.test_sharding.state import AttemptSettings
from scripts.test_sharding.workers import BatchExecution


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "unit_test_runner.py"


def _run(
    tmp_path: Path,
    command: str,
    test_path: Path,
    *extra: str,
    env_override: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.update(env_override or {})
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            command,
            "--junit-dir",
            str(tmp_path / "junit"),
            "--test-path",
            str(test_path),
            "--timing-profile",
            "synthetic",
            "--unknown-case-seconds",
            "1",
            "--checkpoint-seconds",
            "1",
            "--target-unit-seconds",
            "1",
            "--workers",
            "1",
            "--timeout-grace-seconds",
            "0",
            "--deadline-seconds",
            "120",
            *extra,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def test_run_streams_current_pytest_node_before_it_finishes(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(2)\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    process = subprocess.Popen(
        [
            sys.executable,
            str(RUNNER),
            "run",
            "--junit-dir",
            str(tmp_path / "junit"),
            "--test-path",
            str(suite),
            "--timing-profile",
            "synthetic",
            "--unknown-case-seconds",
            "1",
            "--checkpoint-seconds",
            "1",
            "--target-unit-seconds",
            "1",
            "--workers",
            "1",
            "--timeout-grace-seconds",
            "0",
            "--deadline-seconds",
            "120",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    captured: list[str] = []
    saw_running_node = False
    read_deadline = time.monotonic() + 15
    while process.poll() is None and time.monotonic() < read_deadline:
        readable, _, _ = select.select([process.stdout], [], [], 0.1)
        if not readable:
            continue
        line = process.stdout.readline()
        captured.append(line)
        if "PYTEST START" in line and "test_slow.py::test_slow" in line:
            saw_running_node = process.poll() is None
            break
    remainder, _ = process.communicate(timeout=120)
    captured.append(remainder)

    assert process.returncode == 0, "".join(captured)
    assert saw_running_node, "".join(captured)


def test_slow_collection_reports_a_live_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(
        "import time\ntime.sleep(0.3)\n", encoding="utf-8"
    )
    (suite / "test_sample.py").write_text("def test_passes(): pass\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_COLLECTION_HEARTBEAT_SECONDS", 0.05)

    nodes = runner._collect_nodes(REPO_ROOT, suite, 5)

    assert len(nodes) == 1
    output = capsys.readouterr().out
    assert "RUNNER HEARTBEAT: state=collecting" in output
    assert f"test_path={suite}" in output


def test_final_status_reports_elapsed_against_deadline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = {
        "complete": True,
        "shards": {
            "0": {
                "finalized_nodes": 2,
                "planned_nodes": 2,
                "outcomes": {"passed": 1, "failed": 0, "skipped": 1},
                "pending_nodes": 0,
            }
        },
    }
    monkeypatch.setattr(runner.time, "time", lambda: 135.5)

    runner._report_shard_status(
        summary,
        0,
        state="completed",
        deadline_clock=runner.DeadlineClock(started_at=100.0, limit_seconds=120),
    )

    output = capsys.readouterr().out
    assert "RUNNER STATUS: shard=0 state=completed" in output
    assert "deadline_elapsed=35.5s deadline=120s" in output


@pytest.mark.parametrize("timeout_policy", ["skip", "fail"])
def test_worker_exception_is_reported_as_infrastructure_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout_policy: str,
) -> None:
    node = CollectedNode.from_nodeid("tests/test_sample.py::test_case", 0)
    batch = Batch(
        id="batch-1",
        source_file=node.source_file,
        nodeids=(node.nodeid,),
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
    plan = Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=(node,),
        units=(unit,),
    )
    execution = runner.ExecutionSettings(
        workers=1,
        shard_index=0,
        attempt=AttemptSettings(
            deadline_seconds=0,
            unit_timeout_seconds=0,
            timeout_grace_seconds=0,
            timeout_policy=timeout_policy,
        ),
        monitor_memory=False,
    )
    monkeypatch.setattr(runner, "visible_devices", lambda: [None])

    def fail_batch(_request):
        raise OSError("cannot start pytest")

    monkeypatch.setattr(runner, "execute_batch", fail_batch)

    result = runner.execute_shard(
        repo_root=REPO_ROOT,
        junit_dir=tmp_path / "junit",
        plan=plan,
        execution=execution,
        operation_started_at=time.time(),
    )

    assert result == 3
    done = json.loads(
        (
            tmp_path
            / "junit"
            / "attempts"
            / "attempt-0001"
            / "shards"
            / "shard-0000.done.json"
        ).read_text(encoding="utf-8")
    )
    assert any(
        "cannot start pytest" in error for error in done["infrastructure_errors"]
    )
    assert not list((tmp_path / "junit").glob("units/*/batches/*.xml"))
    summary = json.loads(
        (tmp_path / "junit" / "run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is False
    closure = summary["attempts"][0]["closure"]
    assert closure["infrastructure_error"] is True
    assert closure["infrastructure_errors"]


def test_lease_heartbeat_failure_aborts_work_and_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = CollectedNode.from_nodeid("tests/test_sample.py::test_case", 0)
    batch = Batch(
        id="batch-1",
        source_file=node.source_file,
        nodeids=(node.nodeid,),
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
    plan = Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=(node,),
        units=(unit,),
    )
    execution = runner.ExecutionSettings(
        workers=1,
        shard_index=0,
        attempt=AttemptSettings(
            deadline_seconds=0,
            unit_timeout_seconds=0,
            timeout_grace_seconds=0,
            timeout_policy="fail",
        ),
        monitor_memory=False,
    )
    monkeypatch.setattr(runner, "visible_devices", lambda: [None])
    monkeypatch.setattr(runner, "_LEASE_HEARTBEAT_SECONDS", 0.01)
    original_renew = runner._LeaseHeartbeat._renew

    def fail_background_renewal(
        heartbeat: runner._LeaseHeartbeat, path: Path, worker: str
    ) -> None:
        if threading.current_thread() is heartbeat.thread:
            raise OSError("lease storage unavailable")
        original_renew(heartbeat, path, worker)

    def wait_for_abort(request) -> BatchExecution:
        assert request.abort_event is not None
        assert request.abort_event.wait(timeout=5)
        return BatchExecution("infrastructure", "aborted by runner")

    monkeypatch.setattr(runner._LeaseHeartbeat, "_renew", fail_background_renewal)
    monkeypatch.setattr(runner, "execute_batch", wait_for_abort)

    result = runner.execute_shard(
        repo_root=REPO_ROOT,
        junit_dir=tmp_path / "junit",
        plan=plan,
        execution=execution,
        operation_started_at=time.time(),
    )

    assert result == 3
    done = json.loads(
        (
            tmp_path
            / "junit"
            / "attempts"
            / "attempt-0001"
            / "shards"
            / "shard-0000.done.json"
        ).read_text(encoding="utf-8")
    )
    assert any(
        "lease heartbeat failed: lease storage unavailable" in error
        for error in done["infrastructure_errors"]
    )
    assert not list((tmp_path / "junit").glob("units/*/batches/*.xml"))


def test_collection_deadline_reports_that_pytest_was_killed(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    (suite / "test_sample.py").write_text("def test_passes(): pass\n", encoding="utf-8")

    result = _run(tmp_path, "run", suite, "--deadline-seconds", "1")

    assert result.returncode == 3, result.stdout + result.stderr
    combined = result.stdout + result.stderr
    assert "PYTEST KILLED phase=collection reason=attempt-deadline" in combined
    assert "RUNNER STATUS: state=killed phase=collection" in combined
    assert "signal=SIGTERM" in combined
    assert "deadline_elapsed=" in combined
    assert "deadline=1s" in combined


def test_collection_kill_reports_results_from_a_previous_run(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(
        """\
import os
import time

if os.environ.get("SLOW_COLLECTION"):
    time.sleep(5)
""",
        encoding="utf-8",
    )
    (suite / "test_sample.py").write_text(
        "def test_one(): pass\ndef test_two(): pass\n", encoding="utf-8"
    )
    completed = _run(tmp_path, "run", suite)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    killed = _run(
        tmp_path,
        "run",
        suite,
        "--deadline-seconds",
        "1",
        env_override={"SLOW_COLLECTION": "1"},
    )

    assert killed.returncode == 3, killed.stdout + killed.stderr
    combined = killed.stdout + killed.stderr
    assert "RUNNER STATUS: state=killed phase=collection" in combined
    assert "scope=suite finalized=2 passed=2 failed=0 skipped=0 pending=0" in combined
    assert "deadline_elapsed=" in combined
    assert "deadline=1s" in combined
    assert "deadline=unknown" not in combined


def test_plan_and_run_publish_complete_resumable_artifacts(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_sample.py").write_text(
        """\
def test_passes():
    pass

def test_fails():
    assert False, "intentional"
""",
        encoding="utf-8",
    )

    planned = _run(tmp_path, "plan", suite)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    manifest_path = tmp_path / "junit" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned_nodes = manifest["plan"]["nodeids"]
    assert len(planned_nodes) == 2

    executed = _run(
        tmp_path,
        "run",
        suite,
        "--unit-timeout-seconds",
        "60",
        "--timeout-grace-seconds",
        "0",
    )
    assert executed.returncode == 1, executed.stdout + executed.stderr
    assert "RUNNER STATUS: shard=0 state=completed" in executed.stdout
    assert "finalized=2/2 passed=1 failed=1 skipped=0 pending=0" in executed.stdout
    summary = json.loads(
        (tmp_path / "junit" / "run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is True
    assert summary["outcomes"] == {"failed": 1, "passed": 1, "skipped": 0}
    assert summary["shards"]["0"] == {
        "complete": True,
        "finalized_nodes": 2,
        "outcomes": {"failed": 1, "passed": 1, "skipped": 0},
        "pending_nodes": 0,
        "planned_nodes": 2,
    }
    assert summary["fallback_counts"] == {"suite-other-profile": 2}
    assert summary["fallbacks"] == [
        {"node_indexes": [0, 1], "source": "suite-other-profile"}
    ]
    assert summary["capacity"]["estimated_makespan_ms"] > 0
    assert not list((tmp_path / "junit").glob("unit-*.xml"))
    batch_xml = sorted((tmp_path / "junit").glob("units/*/batches/batch-*.xml"))
    assert len(batch_xml) == 1
    assert all(ET.parse(path).getroot().tag == "testsuites" for path in batch_xml)
    assert not (tmp_path / "junit" / ".state").exists()

    repeated = _run(
        tmp_path,
        "run",
        suite,
        "--unit-timeout-seconds",
        "60",
        "--timeout-grace-seconds",
        "0",
    )
    assert repeated.returncode == 1
    assert len(list((tmp_path / "junit" / "attempts").glob("attempt-*"))) == 1


def test_run_records_the_shared_attempt_before_pytest_collection(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "conftest.py").write_text(
        """\
import os
from pathlib import Path

attempts = Path(os.environ["ATTEMPT_PROBE"]) / "attempts"
assert list(attempts.glob("attempt-*/attempt.json")), "attempt was not reserved"
""",
        encoding="utf-8",
    )
    (suite / "test_sample.py").write_text("def test_passes(): pass\n", encoding="utf-8")

    result = _run(
        tmp_path,
        "run",
        suite,
        "--unit-timeout-seconds",
        "5",
        env_override={"ATTEMPT_PROBE": str(tmp_path / "junit")},
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("policy", "expected_code", "outcome"),
    [("skip", 0, "skipped"), ("fail", 1, "failed")],
)
def test_terminal_timeout_policies_synthesize_junit(
    tmp_path: Path,
    policy: str,
    expected_code: int,
    outcome: str,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(5)\n",
        encoding="utf-8",
    )

    result = _run(
        tmp_path,
        "run",
        suite,
        "--unit-timeout-seconds",
        "1",
        "--timeout-grace-seconds",
        "0",
        "--timeout-policy",
        policy,
    )

    assert result.returncode == expected_code, result.stdout + result.stderr
    summary = json.loads(
        (tmp_path / "junit" / "run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is True
    assert summary["synthetic"] == 1
    assert summary["outcomes"][outcome] == 1
    assert len(summary["attempts"][0]["unit_timeout_events"]) == 1
    assert summary["attempts"][0]["unit_timeout_events"][0]["timeout_policy"] == policy


def test_timeout_policy_resume_starts_a_new_attempt_without_fake_results(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(5)\n",
        encoding="utf-8",
    )
    options = (
        "--unit-timeout-seconds",
        "3",
        "--timeout-grace-seconds",
        "0",
        "--timeout-policy",
        "resume",
    )

    first = _run(tmp_path, "run", suite, *options)
    second = _run(tmp_path, "run", suite, *options)

    assert first.returncode == 2, first.stdout + first.stderr
    assert second.returncode == 2, second.stdout + second.stderr
    assert "PYTEST KILLED" in first.stdout
    assert "node=test_slow.py::test_slow" in first.stdout
    assert "RUNNER STATUS: shard=0 state=interrupted" in first.stdout
    assert "reason=unit-timeout" in first.stdout
    assert not list((tmp_path / "junit").glob("unit-*.xml"))
    attempts = list((tmp_path / "junit" / "attempts").glob("attempt-*"))
    assert len(attempts) == 2


def test_timed_out_shard_is_not_retried_inside_the_same_attempt(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(5)\n",
        encoding="utf-8",
    )
    options = (
        "--shard-count",
        "2",
        "--shard-index",
        "0",
        "--unit-timeout-seconds",
        "1",
        "--timeout-grace-seconds",
        "0",
        "--timeout-policy",
        "resume",
    )

    first = _run(tmp_path, "run", suite, *options)
    repeated = _run(tmp_path, "run", suite, *options)

    assert first.returncode == 2, first.stdout + first.stderr
    assert repeated.returncode == 2, repeated.stdout + repeated.stderr
    attempts = list((tmp_path / "junit" / "attempts").glob("attempt-*"))
    assert len(attempts) == 1
    logs = list((tmp_path / "junit" / "units").glob("**/*.log"))
    assert len(logs) == 1
    assert logs[0].read_text(encoding="utf-8").count("command:") == 1


def test_resume_rejects_environment_dependent_collection_changes(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_dynamic.py").write_text(
        """\
import os
import pytest

@pytest.mark.parametrize("value", range(int(os.environ["DYNAMIC_CASES"])))
def test_dynamic(value):
    pass
""",
        encoding="utf-8",
    )

    first = _run(
        tmp_path,
        "plan",
        suite,
        env_override={"DYNAMIC_CASES": "1"},
    )
    changed = _run(
        tmp_path,
        "plan",
        suite,
        env_override={"DYNAMIC_CASES": "2"},
    )

    assert first.returncode == 0, first.stdout + first.stderr
    assert changed.returncode == 3
    assert "collection_fingerprint" in changed.stderr
    assert "RUNNER STATUS: state=failed" in changed.stderr
    assert "deadline_elapsed=" in changed.stderr
    assert "deadline=120s" in changed.stderr


def test_finalize_fan_in_closes_an_attempt_after_all_leases_are_gone(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_one.py").write_text("def test_one(): pass\n", encoding="utf-8")
    (suite / "test_two.py").write_text("def test_two(): pass\n", encoding="utf-8")
    partial = _run(
        tmp_path,
        "run",
        suite,
        "--shard-count",
        "2",
        "--shard-index",
        "0",
        "--timeout-policy",
        "skip",
        "--deadline-seconds",
        "0",
    )
    finalize_env = os.environ.copy()
    finalize_env["PYTHONPATH"] = str(REPO_ROOT)
    finalized = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "finalize",
            "--junit-dir",
            str(tmp_path / "junit"),
        ],
        cwd=REPO_ROOT,
        env=finalize_env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert partial.returncode == 2, partial.stdout + partial.stderr
    assert finalized.returncode == 0, finalized.stdout + finalized.stderr
    attempt = tmp_path / "junit" / "attempts" / "attempt-0001"
    closure = json.loads((attempt / "closed.json").read_text(encoding="utf-8"))
    assert closure["reason"] == "explicit-fan-in"
    summary = json.loads(
        (tmp_path / "junit" / "run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is True
    assert summary["synthetic"] == 1
