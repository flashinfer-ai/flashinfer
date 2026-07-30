from __future__ import annotations

import json
import os
import queue
import select
import shlex
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import unit_test_runner
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


def _queue_test_unit(identifier: str) -> Unit:
    return Unit(
        id=identifier,
        batches=(),
        estimated_ms=1,
        oversized=False,
    )


def test_worker_unit_selection_prefers_local_queue_then_lowest_tied_donor() -> None:
    work_by_worker = [queue.Queue() for _ in range(3)]
    work_by_worker[0].put(_queue_test_unit("donor-0-first"))
    work_by_worker[0].put(_queue_test_unit("donor-0-second"))
    work_by_worker[1].put(_queue_test_unit("local"))
    work_by_worker[2].put(_queue_test_unit("donor-2-first"))
    work_by_worker[2].put(_queue_test_unit("donor-2-second"))

    local_queue, local_unit = runner._next_worker_unit(work_by_worker, 1)

    assert local_queue is work_by_worker[1]
    assert local_unit.id == "local"

    donor_queue, donor_unit = runner._next_worker_unit(work_by_worker, 1)

    assert donor_queue is work_by_worker[0]
    assert donor_unit.id == "donor-0-first"


def test_worker_unit_stealing_drains_queued_units_exactly_once() -> None:
    work_by_worker = [queue.Queue() for _ in range(4)]
    expected = [f"unit-{index}" for index in range(40)]
    for index, identifier in enumerate(expected):
        work_by_worker[index % 2].put(_queue_test_unit(identifier))

    def drain(worker_index: int) -> list[str]:
        completed = []
        while True:
            try:
                work, unit = runner._next_worker_unit(work_by_worker, worker_index)
            except queue.Empty:
                return completed
            completed.append(unit.id)
            work.task_done()

    with ThreadPoolExecutor(max_workers=4) as executor:
        completed = [
            identifier
            for worker_units in executor.map(drain, range(4))
            for identifier in worker_units
        ]

    assert sorted(completed) == sorted(expected)
    assert all(work.unfinished_tasks == 0 for work in work_by_worker)


def test_compatibility_defaults_use_file_sized_units_without_time_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNIT_TEST_DEADLINE_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_UNKNOWN_CASE_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_CHECKPOINT_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_TARGET_SECONDS", raising=False)

    args = unit_test_runner._parser().parse_args(["run"])

    assert args.deadline_seconds == 0
    assert args.unit_timeout_seconds == 0
    assert args.unknown_case_seconds == 1
    assert args.checkpoint_seconds == 1_000_000
    assert args.target_unit_seconds == 1_000_000
    assert PlanningOptions(profile="test").checkpoint_seconds == 1_000_000
    assert PlanningOptions(profile="test").target_unit_seconds == 1_000_000


def test_explicit_zero_deadline_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNIT_TEST_DEADLINE_SECONDS", "60")

    args = unit_test_runner._parser().parse_args(["run", "--deadline-seconds", "0"])

    assert args.deadline_seconds == 0
    clock = runner.DeadlineClock(started_at=100.0, limit_seconds=args.deadline_seconds)
    assert clock.status_fields().endswith("deadline=disabled")


def test_empty_test_path_environment_uses_default_suite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_PATH", "")

    args = unit_test_runner._parser().parse_args(["run"])

    assert args.test_path == Path("tests/")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("UNIT_TEST_CHECKPOINT_SECONDS", "0"),
        ("UNIT_TEST_SHARD_COUNT", "0"),
        ("UNIT_TEST_WORKERS", "0"),
        ("UNIT_TEST_DEADLINE_SECONDS", "-1"),
        ("UNIT_TEST_TIMEOUT_POLICY", "bogus"),
    ],
)
def test_invalid_cli_environment_defaults_fail_argument_preflight(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit) as exit_info:
        unit_test_runner._parser().parse_args(["run"])

    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SAMPLE_RATE", "0"),
        ("SAMPLE_OFFSET", "5"),
        ("PYTEST_COMMAND_PREFIX", "'unterminated"),
        ("MEMORY_MONITOR_INTERVAL", "invalid"),
        ("MEMORY_MONITOR_INTERVAL", "0"),
        ("MEMORY_MONITOR_INTERVAL", "-1"),
        ("MEMORY_MONITOR_INTERVAL", "nan"),
        ("MEMORY_MONITOR_INTERVAL", "inf"),
    ],
)
def test_invalid_runtime_environment_fails_shell_settings_preflight(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit) as exit_info:
        unit_test_runner._shell_settings(["--dry-run"])

    assert exit_info.value.code == 2


def test_help_defines_global_zero_based_sanity_sampling(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        unit_test_runner._parser().parse_args(["run", "--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "globally select every SAMPLE_RATE-th collected node" in help_text
    assert "SAMPLE_RATE=5, SAMPLE_OFFSET=0" in help_text


def test_help_reports_effective_argument_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UNIT_TEST_TARGET_SECONDS", "123")
    monkeypatch.delenv("UNIT_TEST_TIMING_PROFILE", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        unit_test_runner._parser().parse_args(["run", "--help"])

    assert exit_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert (
        "soft logical-unit target (UNIT_TEST_TARGET_SECONDS) (default: 123)"
        in help_text
    )
    assert (
        "stable timing profile (UNIT_TEST_TIMING_PROFILE; default: auto-detected)"
    ) in help_text
    assert "exit codes: 0=complete without failures;" in help_text


def test_help_reports_configured_timing_profile_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("UNIT_TEST_TIMING_PROFILE", "sm103-cuda13")

    with pytest.raises(SystemExit) as exit_info:
        unit_test_runner._parser().parse_args(["run", "--help"])

    assert exit_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert (
        "stable timing profile (UNIT_TEST_TIMING_PROFILE) (default: sm103-cuda13)"
    ) in help_text


def test_pytest_command_prefix_wraps_batches_and_is_frozen_in_manifest(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_sample.py").write_text("def test_passes(): pass\n", encoding="utf-8")
    record = tmp_path / "prefix-arguments.json"
    wrapper = tmp_path / "prefix wrapper.py"
    wrapper.write_text(
        """\
import json
import os
import sys
from pathlib import Path

Path(os.environ["PREFIX_RECORD"]).write_text(
    json.dumps(sys.argv[1:]), encoding="utf-8"
)
os.execv(sys.argv[1], sys.argv[1:])
""",
        encoding="utf-8",
    )
    prefix = shlex.join([sys.executable, str(wrapper)])
    environment = {
        "PREFIX_RECORD": str(record),
        "PYTEST_COMMAND_PREFIX": prefix,
    }

    created = _run(tmp_path, "run", suite, env_override=environment)

    assert created.returncode == 0, created.stdout + created.stderr
    forwarded = json.loads(record.read_text(encoding="utf-8"))
    assert forwarded[:3] == [sys.executable, "-m", "pytest"]
    manifest = json.loads(
        (tmp_path / "junit" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["pytest_command_prefix"] == [sys.executable, str(wrapper)]

    changed = _run(
        tmp_path,
        "run",
        suite,
        env_override={
            **environment,
            "PYTEST_COMMAND_PREFIX": shlex.join([sys.executable, "-u", str(wrapper)]),
        },
    )

    assert changed.returncode == 3
    assert "pytest_command_prefix" in changed.stderr


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

    nodes = runner._collect_nodes(REPO_ROOT, suite, 5, 5)

    assert len(nodes) == 1
    output = capsys.readouterr().out
    assert "RUNNER HEARTBEAT: state=collecting" in output
    assert f"test_path={suite}" in output


def test_collection_termination_uses_configured_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RunningProcess:
        def poll(self) -> None:
            return None

    grace_values: list[float] = []

    def terminate(_process, grace_seconds: float) -> str:
        grace_values.append(grace_seconds)
        return "SIGTERM"

    monkeypatch.setattr(runner, "terminate_process_group", terminate)

    with pytest.raises(runner.CollectionTimeoutError):
        runner._wait_for_collection(
            RunningProcess(),  # type: ignore[arg-type]
            test_path=tmp_path,
            timeout_seconds=0,
            grace_seconds=17,
        )

    assert grace_values == [17]


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
    assert "signal=SIGKILL" in combined
    assert "deadline_elapsed=" in combined
    assert "deadline=1s" in combined


def test_completed_manifest_skips_slow_recollection(tmp_path: Path) -> None:
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

    reused = _run(
        tmp_path,
        "run",
        suite,
        "--deadline-seconds",
        "1",
        env_override={"SLOW_COLLECTION": "1"},
    )

    assert reused.returncode == 0, reused.stdout + reused.stderr
    assert "Using plan" in reused.stdout
    assert "RUNNER STATUS: shard=0 state=completed" in reused.stdout
    assert "finalized=2/2 passed=2 failed=0 skipped=0 pending=0" in reused.stdout
    assert "PYTEST KILLED phase=collection" not in reused.stdout + reused.stderr


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
    assert summary["fallback_counts"] == {"suite-mean-other-profile": 2}
    assert summary["fallbacks"] == [
        {"node_indexes": [0, 1], "source": "suite-mean-other-profile"}
    ]
    assert summary["capacity"]["estimated_makespan_ms"] > 0
    assert not list((tmp_path / "junit").glob("unit-*.xml"))
    batch_xml = sorted((tmp_path / "junit").glob("units/*/batches/batch-*.xml"))
    planned_batches = sum(len(unit["batches"]) for unit in manifest["plan"]["units"])
    assert len(batch_xml) == planned_batches
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


def test_unit_progress_omits_skipped_results_and_reports_all_outcomes(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_sample.py").write_text(
        """\
import pytest

def test_passes():
    pass

def test_fails():
    assert False, "intentional"

def test_skips():
    pytest.skip("intentional")
""",
        encoding="utf-8",
    )

    result = _run(
        tmp_path,
        "run",
        suite,
        "--checkpoint-seconds",
        "100",
        "--target-unit-seconds",
        "100",
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "PYTEST RESULT" in result.stdout
    assert "outcome=failed" in result.stdout
    assert not any(
        "PYTEST RESULT" in line and "outcome=skipped" in line
        for line in result.stdout.splitlines()
    )
    unit_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("PYTEST UNIT COMPLETE")
    ]
    assert len(unit_lines) == 1
    assert (
        "finalized=3 passed=1 failed=1 skipped=1 unknown=0 completed_nodes=3/3"
    ) in unit_lines[0]


def test_completed_node_progress_observes_finalized_batches_from_every_shard(
    tmp_path: Path,
) -> None:
    nodes = tuple(
        CollectedNode.from_nodeid(f"tests/test_{index}.py::test_case", index)
        for index in range(2)
    )
    units = tuple(
        Unit(
            id=f"unit-{index}",
            batches=(
                Batch(
                    id=f"batch-{index}",
                    source_file=node.source_file,
                    nodeids=(node.nodeid,),
                    estimated_ms=1000,
                    overhead_ms=0,
                    oversized=False,
                ),
            ),
            estimated_ms=1000,
            oversized=False,
            shard_index=index,
        )
        for index, node in enumerate(nodes)
    )
    plan = Plan(
        options=PlanningOptions(profile="synthetic", shard_count=2),
        nodes=nodes,
        units=units,
    )
    metadata = runner.SyntheticBatchMetadata(
        policy="skip",
        batch_id=units[0].batches[0].id,
        unit_id=units[0].id,
        shard_index=0,
        attempt_id="attempt-1",
        profile="synthetic",
    )

    assert runner._completed_node_count(tmp_path / "junit", plan) == 0
    runner._synthesize_batch(
        tmp_path / "junit", units[0], units[0].batches[0], metadata
    )
    assert runner._completed_node_count(tmp_path / "junit", plan) == 1
    runner._synthesize_batch(
        tmp_path / "junit",
        units[1],
        units[1].batches[0],
        runner.SyntheticBatchMetadata(
            policy="fail",
            batch_id=units[1].batches[0].id,
            unit_id=units[1].id,
            shard_index=1,
            attempt_id="attempt-1",
            profile="synthetic",
        ),
    )
    assert runner._completed_node_count(tmp_path / "junit", plan) == 2


def test_solo_source_runs_exclusively_with_full_gpu_visibility(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    state = tmp_path / "solo-state"
    state.mkdir()
    (suite / "test_solo.py").write_text(
        """\
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.solo

@pytest.fixture(scope="module", autouse=True)
def exclusive_source():
    state = Path(os.environ["SOLO_STATE"])
    solo = state / "solo-active"
    regular = state / "regular-active"
    assert not regular.exists(), "regular batch overlapped solo setup"
    solo.write_text("active", encoding="utf-8")
    (state / "solo-devices").write_text(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), encoding="utf-8"
    )
    time.sleep(1)
    yield
    time.sleep(1)
    assert not regular.exists(), "regular batch overlapped solo teardown"
    solo.unlink()

def test_solo_one():
    pass

def test_solo_two():
    pass
""",
        encoding="utf-8",
    )
    (suite / "test_regular.py").write_text(
        """\
import os
import time
from pathlib import Path

def test_regular():
    state = Path(os.environ["SOLO_STATE"])
    solo = state / "solo-active"
    regular = state / "regular-active"
    assert not solo.exists(), "solo batch overlapped regular setup"
    regular.write_text("active", encoding="utf-8")
    (state / "regular-devices").write_text(
        os.environ.get("CUDA_VISIBLE_DEVICES", ""), encoding="utf-8"
    )
    try:
        time.sleep(2)
        assert not solo.exists(), "solo batch overlapped regular execution"
    finally:
        regular.unlink()
""",
        encoding="utf-8",
    )

    result = _run(
        tmp_path,
        "run",
        suite,
        "--workers",
        "2",
        env_override={
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SOLO_STATE": str(state),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (state / "solo-devices").read_text(encoding="utf-8") == "0,1"
    assert (state / "regular-devices").read_text(encoding="utf-8") in {"0", "1"}


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


def test_manifest_compares_source_git_sha_only_when_both_values_are_available(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_sample.py").write_text("def test_passes(): pass\n", encoding="utf-8")

    created = _run(
        tmp_path,
        "plan",
        suite,
        env_override={"SOURCE_GIT_SHA": "source-a"},
    )
    mismatch = _run(
        tmp_path,
        "plan",
        suite,
        env_override={"SOURCE_GIT_SHA": "source-b"},
    )
    unavailable = _run(
        tmp_path,
        "plan",
        suite,
        env_override={"SOURCE_GIT_SHA": ""},
    )

    assert created.returncode == 0, created.stdout + created.stderr
    manifest = json.loads(
        (tmp_path / "junit" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source_git_sha"] == "source-a"
    assert mismatch.returncode == 3
    assert "source_git_sha" in mismatch.stderr
    assert unavailable.returncode == 0, unavailable.stdout + unavailable.stderr


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
    expected_counts = (
        "passed=0 failed=1 skipped=0 unknown=0"
        if policy == "fail"
        else "passed=0 failed=0 skipped=1 unknown=0"
    )
    assert expected_counts in result.stdout
    assert "completed_nodes=1/1" in result.stdout
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


def test_existing_manifest_skips_environment_dependent_recollection(
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
    assert changed.returncode == 0, changed.stdout + changed.stderr
    assert "Using plan" in changed.stdout
    manifest = json.loads(
        (tmp_path / "junit" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["plan"]["nodeids"]) == 1


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
