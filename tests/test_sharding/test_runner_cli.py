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


def test_worker_unit_selection_prefers_local_queue_before_stealing() -> None:
    work_by_worker = [queue.Queue() for _ in range(2)]
    work_by_worker[0].put(_queue_test_unit("donor"))
    work_by_worker[1].put(_queue_test_unit("local"))

    local_queue, local_unit = runner._next_worker_unit(work_by_worker, 1)

    assert local_queue is work_by_worker[1]
    assert local_unit.id == "local"

    donor_queue, donor_unit = runner._next_worker_unit(work_by_worker, 1)

    assert donor_queue is work_by_worker[0]
    assert donor_unit.id == "donor"


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


def test_compatibility_defaults_use_file_sized_units_with_failure_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNIT_TEST_DEADLINE_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_TIMEOUT_POLICY", raising=False)
    monkeypatch.delenv("UNIT_TEST_DEFAULT_CASE_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_DEFAULT_SOURCE_OVERHEAD_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_DURATION_ESTIMATES", raising=False)
    monkeypatch.delenv("UNIT_TEST_OVERHEAD_ESTIMATES", raising=False)
    monkeypatch.delenv("UNIT_TEST_SHARD_COUNT", raising=False)
    monkeypatch.delenv("UNIT_TEST_CHECKPOINT_SECONDS", raising=False)
    monkeypatch.delenv("UNIT_TEST_TARGET_SECONDS", raising=False)

    args = unit_test_runner._parser().parse_args(["run"])

    assert args.deadline_seconds == 0
    assert args.unit_timeout_seconds == 7_200
    assert args.timeout_policy == "fail"
    assert args.default_case_seconds == 1
    assert args.default_source_overhead_seconds == 30
    assert args.duration_estimates is None
    assert args.overhead_estimates is None
    assert args.shard_count == 1
    assert args.checkpoint_seconds == 1_000_000
    assert args.target_unit_seconds == 1_000_000
    assert PlanningOptions().checkpoint_seconds == 1_000_000
    assert PlanningOptions().target_unit_seconds == 1_000_000


def test_shard_count_precedence_is_cli_then_environment_then_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("UNIT_TEST_SHARD_COUNT", raising=False)
    assert unit_test_runner._parser().parse_args(["run"]).shard_count == 1

    monkeypatch.setenv("UNIT_TEST_SHARD_COUNT", "4")
    assert unit_test_runner._parser().parse_args(["run"]).shard_count == 4
    assert (
        unit_test_runner._parser().parse_args(["run", "--shard-count", "2"]).shard_count
        == 2
    )


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

    assert args.test_path == [Path("tests/")]


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
    monkeypatch.setenv("UNIT_TEST_DURATION_ESTIMATES", "/tmp/durations.csv.gz")
    monkeypatch.setenv("UNIT_TEST_OVERHEAD_ESTIMATES", "/tmp/overheads.csv")

    with pytest.raises(SystemExit) as exit_info:
        unit_test_runner._parser().parse_args(["run", "--help"])

    assert exit_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert (
        "soft logical-unit target (UNIT_TEST_TARGET_SECONDS) (default: 123)"
        in help_text
    )
    assert "optional per-node duration CSV or CSV.gz" in help_text
    assert "default estimate for a node missing timing data" in help_text
    assert "default per-source process overhead" in help_text
    assert "--dry-run" in help_text
    assert "--wrapper-started-at" in help_text
    assert "--timing-profile" not in help_text
    assert "--unknown-case-seconds" not in help_text
    assert "exit codes: 0=complete without failures;" in help_text
    assert "(default: /tmp/durations.csv.gz)" in help_text
    assert "(default: /tmp/overheads.csv)" in help_text


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
    assert "pytest_command_prefix" in changed.stdout


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
            "--default-case-seconds",
            "1",
            "--default-source-overhead-seconds",
            "0",
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


def test_cli_summary_uses_scoped_shell_invocation_start_time(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_sample.py").write_text("def test_case(): pass\n", encoding="utf-8")

    result = _run(
        tmp_path,
        "plan",
        suite,
        "--wrapper-started-at",
        "1700000000",
    )

    assert result.returncode == 0, result.stdout
    assert "Start time: 2023-11-14T22:13:20Z" in result.stdout
    assert "End time: " in result.stdout
    assert "Time elapsed: " in result.stdout


def test_cli_summary_ignores_unscoped_start_time_environment(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_sample.py").write_text("def test_case(): pass\n", encoding="utf-8")

    result = _run(
        tmp_path,
        "plan",
        suite,
        env_override={"UNIT_TEST_RUN_STARTED_AT": "1700000000"},
    )

    assert result.returncode == 0, result.stdout
    assert "Start time: 2023-11-14T22:13:20Z" not in result.stdout


def test_optional_timing_files_use_first_matching_rows(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = suite / "test_sample.py"
    source.write_text("def test_case(): pass\n", encoding="utf-8")
    nodeid = f"{source.name}::test_case"
    source_file = str(source.resolve())
    duration = tmp_path / "durations.csv"
    duration.write_text(
        f"nodeid,estimated_seconds\n{nodeid},7\n{nodeid},99\n",
        encoding="utf-8",
    )
    overhead = tmp_path / "overheads.csv"
    overhead.write_text(
        "source_file,process_startup_seconds,source_warmup_seconds\n"
        f"{source_file},3,7\n"
        f"{source_file},99,99\n",
        encoding="utf-8",
    )

    result = _run(
        tmp_path,
        "plan",
        suite,
        "--duration-estimates",
        str(duration),
        "--overhead-estimates",
        str(overhead),
    )

    assert result.returncode == 0, result.stdout
    manifest = json.loads(
        (tmp_path / "junit" / "manifest.json").read_text(encoding="utf-8")
    )
    batch = manifest["plan"]["units"][0]["batches"][0]
    assert batch["estimated_ms"] == 17_000
    assert set(manifest["estimate_files"]) == {"duration", "overhead"}


def test_manifest_freezes_timing_content_not_input_path(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "test_sample.py").write_text("def test_case(): pass\n", encoding="utf-8")
    content = "nodeid,estimated_seconds\ntest_sample.py::test_case,7\n"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")

    created = _run(tmp_path, "plan", suite, "--duration-estimates", str(first))
    reused = _run(tmp_path, "plan", suite, "--duration-estimates", str(second))
    second.write_text(content.replace(",7", ",8"), encoding="utf-8")
    changed = _run(tmp_path, "plan", suite, "--duration-estimates", str(second))

    assert created.returncode == 0, created.stdout
    assert reused.returncode == 0, reused.stdout
    assert "Using plan" in reused.stdout
    assert changed.returncode == 3
    assert "estimate_files" in changed.stdout


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
            "--default-case-seconds",
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


def test_collection_isolates_sm90_pull_multirank_modules(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    sm100 = suite / "sm100"
    sm100.mkdir()
    (sm100 / "common.py").write_text("BACKEND = 'sm100'\n", encoding="utf-8")
    sm90 = suite / "sm90"
    sm90.mkdir()
    (sm90 / "common.py").write_text("BACKEND = 'sm90'\n", encoding="utf-8")
    (suite / "test_aaa_sm100.py").write_text(
        """\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "sm100"))
import common

assert common.BACKEND == "sm100"

def test_sm100():
    pass
""",
        encoding="utf-8",
    )
    isolated = suite / "test_moe_ep_sm90_pull_fp8_mega_multirank.py"
    isolated.write_text(
        """\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "sm90"))
import common

if common.BACKEND != "sm90":
    raise RuntimeError("SM100 and SM90 common modules cannot share one process")

def test_sm90():
    pass
""",
        encoding="utf-8",
    )

    nodes = runner._collect_nodes(REPO_ROOT, suite, 15, 0)

    assert [node["nodeid"] for node in nodes] == [
        "test_aaa_sm100.py::test_sm100",
        f"{isolated.name}::test_sm90",
    ]
    assert [node["order"] for node in nodes] == [0, 1]

    isolated_nodes = runner._collect_nodes(REPO_ROOT, isolated, 15, 0)

    assert [node["nodeid"] for node in isolated_nodes] == [
        f"{isolated.name}::test_sm90"
    ]
    assert [node["order"] for node in isolated_nodes] == [0]


def test_collection_isolates_sm120_swapab_multirank_modules(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    for backend in ("sm100", "sm90", "sm120"):
        tree = suite / backend
        tree.mkdir()
        (tree / "common.py").write_text(f"BACKEND = {backend!r}\n", encoding="utf-8")

    def _write(name: str, backend: str, test_name: str) -> Path:
        path = suite / name
        path.write_text(
            f"""\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "{backend}"))
import common

if common.BACKEND != "{backend}":
    raise RuntimeError("vendored common modules cannot share one process")

def {test_name}():
    pass
""",
            encoding="utf-8",
        )
        return path

    _write("test_aaa_sm100.py", "sm100", "test_sm100")
    sm90 = _write("test_moe_ep_sm90_pull_fp8_mega_multirank.py", "sm90", "test_sm90")
    sm120 = _write(
        "test_moe_ep_sm120_mxfp8_cutedsl_mega_multirank.py", "sm120", "test_sm120"
    )

    nodes = runner._collect_nodes(REPO_ROOT, suite, 20, 0)

    assert [node["nodeid"] for node in nodes] == [
        "test_aaa_sm100.py::test_sm100",
        f"{sm90.name}::test_sm90",
        f"{sm120.name}::test_sm120",
    ]
    assert [node["order"] for node in nodes] == [0, 1, 2]

    isolated_nodes = runner._collect_nodes(REPO_ROOT, sm120, 15, 0)

    assert [node["nodeid"] for node in isolated_nodes] == [f"{sm120.name}::test_sm120"]
    assert [node["order"] for node in isolated_nodes] == [0]


def test_pytest_root_is_stable_for_repository_and_external_scopes(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    test_file = suite / "test_sample.py"
    test_file.write_text("def test_case(): pass\n", encoding="utf-8")

    assert runner._pytest_root(REPO_ROOT, REPO_ROOT / "tests") == REPO_ROOT.resolve()
    assert runner._pytest_root(REPO_ROOT, suite) == suite.resolve()
    assert runner._pytest_root(REPO_ROOT, test_file) == suite.resolve()


def test_collection_preserves_external_pytest_config(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text(
        "[pytest]\npython_files = check_*.py\n",
        encoding="utf-8",
    )
    (suite / "check_sample.py").write_text(
        "def test_case(): pass\n",
        encoding="utf-8",
    )

    nodes = runner._collect_nodes(REPO_ROOT, suite, 15, 0)

    assert [node["nodeid"] for node in nodes] == ["check_sample.py::test_case"]


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
        options=PlanningOptions(),
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
        test_path=REPO_ROOT / "tests",
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
    assert not list((tmp_path / "junit").glob("shards/*/units/*/batches/*.xml"))
    summary = json.loads(
        (tmp_path / "junit" / "run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is False
    closure = summary["attempts"][0]["closure"]
    assert closure["infrastructure_error"] is True
    assert closure["infrastructure_errors"]


def test_setup_error_is_not_masked_when_lease_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    plan = Plan(
        options=PlanningOptions(),
        nodes=(node,),
        units=(
            Unit(
                id="unit-1",
                batches=(batch,),
                estimated_ms=1000,
                oversized=False,
                shard_index=0,
            ),
        ),
    )
    execution = runner.ExecutionSettings(
        workers=1,
        shard_index=0,
        attempt=AttemptSettings(
            deadline_seconds=0,
            unit_timeout_seconds=0,
            timeout_grace_seconds=0,
            timeout_policy="resume",
        ),
        monitor_memory=False,
    )
    monkeypatch.setattr(runner, "visible_devices", lambda: [None])
    original_write = runner.atomic_write_json

    def fail_settings(path: Path, value) -> None:
        if path.name.endswith(".settings.json"):
            raise OSError("cannot write shard settings")
        original_write(path, value)

    class FailingLeases:
        def close(self) -> None:
            raise runner.RunnerStateError("cannot close shard leases")

    monkeypatch.setattr(runner, "atomic_write_json", fail_settings)
    monkeypatch.setattr(
        runner,
        "_claim_shard_leases",
        lambda *_args, **_kwargs: FailingLeases(),
    )

    with pytest.raises(OSError, match="cannot write shard settings"):
        runner.execute_shard(
            repo_root=REPO_ROOT,
            junit_dir=tmp_path / "junit",
            plan=plan,
            execution=execution,
            operation_started_at=time.time(),
            test_path=REPO_ROOT / "tests",
        )

    assert "cannot close shard leases" in capsys.readouterr().out


def test_partial_shard_lease_claim_is_rolled_back(
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
    plan = Plan(
        options=PlanningOptions(),
        nodes=(node,),
        units=(
            Unit(
                id="unit-1",
                batches=(batch,),
                estimated_ms=1000,
                oversized=False,
                shard_index=0,
            ),
        ),
    )
    execution = runner.ExecutionSettings(
        workers=1,
        shard_index=0,
        attempt=AttemptSettings(
            deadline_seconds=0,
            unit_timeout_seconds=0,
            timeout_grace_seconds=0,
            timeout_policy="resume",
        ),
        monitor_memory=False,
    )
    monkeypatch.setattr(runner, "visible_devices", lambda: [None])
    original_add = runner._LeaseHeartbeat.add

    def fail_worker_lease(
        heartbeat: runner._LeaseHeartbeat, path: Path, worker: str
    ) -> None:
        if worker == "worker-0":
            raise OSError("cannot create worker lease")
        original_add(heartbeat, path, worker)

    monkeypatch.setattr(runner._LeaseHeartbeat, "add", fail_worker_lease)

    with pytest.raises(OSError, match="cannot create worker lease"):
        runner.execute_shard(
            repo_root=REPO_ROOT,
            junit_dir=tmp_path / "junit",
            plan=plan,
            execution=execution,
            operation_started_at=time.time(),
            test_path=REPO_ROOT / "tests",
        )

    leases = tmp_path / "junit" / "attempts" / "attempt-0001" / "leases"
    assert list(leases.glob("*.json")) == []


def test_heartbeat_close_is_bounded_when_storage_write_is_stuck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = {
        "id": "attempt-1",
        "started_at": time.time(),
        "deadline_at": None,
        "settings": {
            "deadline_seconds": 0,
            "unit_timeout_seconds": 0,
            "timeout_grace_seconds": 0,
            "timeout_policy": "resume",
        },
    }
    heartbeat = runner._LeaseHeartbeat(attempt, 0)  # type: ignore[arg-type]
    lease = tmp_path / "lease.json"
    heartbeat.add(lease, "coordinator")
    entered = threading.Event()
    release = threading.Event()
    original_renew = heartbeat._renew

    def block_background_write(path: Path, worker: str) -> None:
        if threading.current_thread() is heartbeat.thread:
            entered.set()
            release.wait(timeout=5)
            return
        original_renew(path, worker)

    monkeypatch.setattr(heartbeat, "_renew", block_background_write)
    monkeypatch.setattr(runner, "_LEASE_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(runner, "_LEASE_CLOSE_SECONDS", 0.05)
    heartbeat.start()
    assert entered.wait(timeout=2)
    started = time.monotonic()
    try:
        with pytest.raises(runner.RunnerStateError, match="did not stop"):
            heartbeat.close()
    finally:
        release.set()
        heartbeat.thread.join(timeout=2)
        lease.unlink(missing_ok=True)

    assert time.monotonic() - started < 0.5


def test_heartbeat_close_attempts_every_lease_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = {
        "id": "attempt-1",
        "started_at": time.time(),
        "deadline_at": None,
        "settings": {
            "deadline_seconds": 0,
            "unit_timeout_seconds": 0,
            "timeout_grace_seconds": 0,
            "timeout_policy": "resume",
        },
    }
    heartbeat = runner._LeaseHeartbeat(attempt, 0)  # type: ignore[arg-type]
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    heartbeat.add(first, "first")
    heartbeat.add(second, "second")
    original_unlink = Path.unlink

    def fail_first(path: Path, *, missing_ok: bool = False) -> None:
        if path == first:
            raise OSError("first lease is busy")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_first)
    try:
        with pytest.raises(runner.RunnerStateError, match="first lease is busy"):
            heartbeat.close()
        assert not second.exists()
    finally:
        original_unlink(first, missing_ok=True)


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
        options=PlanningOptions(),
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
        test_path=REPO_ROOT / "tests",
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
    assert not list((tmp_path / "junit").glob("shards/*/units/*/batches/*.xml"))


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


def test_plan_run_and_completed_reuse_publish_resumable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pytest 9 may otherwise inherit the repository root for this external
    # temporary suite, which used to leak pytest-of-*/... prefixes into node IDs.
    monkeypatch.setenv("PYTEST_ADDOPTS", f"--rootdir={REPO_ROOT}")
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
    failure_detail_at = executed.stdout.index("FAILED TEST NODES")
    summary_at = executed.stdout.index("TEST SUMMARY")
    assert failure_detail_at < summary_at
    assert "test_sample.py::test_fails" in executed.stdout[:summary_at]
    assert "intentional" in executed.stdout[:summary_at]
    summary_output = executed.stdout[summary_at:]
    assert f"Failed test files:\n  - {suite / 'test_sample.py'}" in summary_output
    assert "test_sample.py::test_fails" not in summary_output
    assert "intentional" not in summary_output
    assert "Top 10 longest-running source files:" in executed.stdout
    assert "Top 10 highest host RSS source files:" in executed.stdout
    assert "Top 10 highest GPU memory source files:" in executed.stdout
    assert "RUNNER STATUS: shard=0 state=completed" in executed.stdout
    assert "finalized=2/2 passed=1 failed=1 skipped=0 pending=0" in executed.stdout
    summary = json.loads(
        (tmp_path / "junit" / "run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is True
    assert summary["failed_nodes"][0]["nodeid"] == "test_sample.py::test_fails"
    assert "intentional" in summary["failed_nodes"][0]["diagnostic"]
    assert summary["outcomes"] == {"failed": 1, "passed": 1, "skipped": 0}
    assert summary["shards"]["0"] == {
        "complete": True,
        "finalized_nodes": 2,
        "outcomes": {"failed": 1, "passed": 1, "skipped": 0},
        "pending_nodes": 0,
        "planned_nodes": 2,
    }
    assert summary["fallback_counts"] == {"default-case": 2}
    assert summary["fallbacks"] == [{"node_indexes": [0, 1], "source": "default-case"}]
    assert summary["capacity"]["estimated_makespan_ms"] > 0
    assert not list((tmp_path / "junit").glob("unit-*.xml"))
    batch_xml = sorted(
        (tmp_path / "junit").glob("shards/shard-0000/units/*/batches/batch-*.xml")
    )
    planned_batches = sum(len(unit["batches"]) for unit in manifest["plan"]["units"])
    assert len(batch_xml) == planned_batches
    assert all(ET.parse(path).getroot().tag == "testsuites" for path in batch_xml)
    assert not (tmp_path / "junit" / ".state").exists()

    # A completed run must use its frozen plan without repeating collection.
    repeated = _run(
        tmp_path,
        "run",
        suite,
        "--unit-timeout-seconds",
        "60",
        "--timeout-grace-seconds",
        "0",
        "--deadline-seconds",
        "1",
        env_override={"SLOW_COLLECTION": "1"},
    )
    assert repeated.returncode == 1
    assert "Using plan" in repeated.stdout
    assert "RUNNER STATUS: shard=0 state=completed" in repeated.stdout
    assert "finalized=2/2 passed=1 failed=1 skipped=0 pending=0" in repeated.stdout
    assert "PYTEST KILLED phase=collection" not in repeated.stdout + repeated.stderr
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
        options=PlanningOptions(shard_count=2),
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


def test_long_running_dispatches_first_and_solo_runs_after_non_solo_finalized(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    state = tmp_path / "phase-state"
    state.mkdir()
    for name in ("a", "b"):
        (suite / f"test_long_{name}.py").write_text(
            f"""\
import os
import time
from pathlib import Path
import pytest

pytestmark = pytest.mark.long_running

def test_long_{name}():
    state = Path(os.environ["PHASE_STATE"])
    (state / "long-{name}-started").write_text("1", encoding="utf-8")
    time.sleep(0.5)
    (state / "long-{name}-done").write_text("1", encoding="utf-8")
""",
            encoding="utf-8",
        )
    (suite / "test_normal.py").write_text(
        """\
import os
from pathlib import Path

def test_normal():
    state = Path(os.environ["PHASE_STATE"])
    assert (state / "long-a-started").exists()
    assert (state / "long-b-started").exists()
    (state / "normal-done").write_text("1", encoding="utf-8")
""",
        encoding="utf-8",
    )
    (suite / "test_solo.py").write_text(
        """\
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.solo

def test_solo():
    state = Path(os.environ["PHASE_STATE"])
    assert (state / "long-a-done").exists()
    assert (state / "long-b-done").exists()
    assert (state / "normal-done").exists()
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
            "PHASE_STATE": str(state),
        },
    )

    assert result.returncode == 0, result.stdout
    assert "WORKER START worker=0" in result.stdout
    assert "WORKER TASK worker=0" in result.stdout
    assert "WORKER END worker=0" in result.stdout
    assert "PROGRESS shard=0 finalized=4/4" in result.stdout
    assert "PYTEST RUNNING" not in result.stdout


def test_pending_non_solo_work_blocks_the_solo_phase(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    solo_started = tmp_path / "solo-started"
    (suite / "test_regular.py").write_text(
        "import time\ndef test_regular(): time.sleep(5)\n",
        encoding="utf-8",
    )
    (suite / "test_solo.py").write_text(
        f"""\
from pathlib import Path
import pytest

pytestmark = pytest.mark.solo

def test_solo():
    Path({str(solo_started)!r}).write_text("started", encoding="utf-8")
""",
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
        "resume",
    )

    assert result.returncode == 2, result.stdout
    assert not solo_started.exists()
    assert "WORKER TASK worker=solo-0" not in result.stdout
    assert "Pending: 2" in result.stdout


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
    assert "source_git_sha" in mismatch.stdout
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
        "2",
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
    logs = list((tmp_path / "junit" / "shards" / "shard-0000").glob("**/*.log"))
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

    assert partial.returncode == 0, partial.stdout + partial.stderr
    assert "Shared run: finalized=1/2 pending=1" in partial.stdout
    assert finalized.returncode == 0, finalized.stdout + finalized.stderr
    assert "Scope: shared run" in finalized.stdout
    assert "Finalized nodes: 2" in finalized.stdout
    attempt = tmp_path / "junit" / "attempts" / "attempt-0001"
    closure = json.loads((attempt / "closed.json").read_text(encoding="utf-8"))
    assert closure["reason"] == "explicit-fan-in"
    summary = json.loads(
        (tmp_path / "junit" / "run-summary.json").read_text(encoding="utf-8")
    )
    assert summary["complete"] is True
    assert summary["synthetic"] == 1


def test_test_path_environment_splits_multiple_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_PATH", "tests/moe tests/gdn")

    args = unit_test_runner._parser().parse_args(["run"])

    assert args.test_path == [Path("tests/moe"), Path("tests/gdn")]


def test_shell_settings_prints_space_separated_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TEST_PATH", "tests/moe tests/gdn")

    assert unit_test_runner._shell_settings([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["run", "tests/moe tests/gdn"]


def test_cli_test_path_accepts_multiple_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_PATH", raising=False)

    args = unit_test_runner._parser().parse_args(
        ["run", "--test-path", "tests/moe", "tests/gdn"]
    )

    assert args.test_path == [Path("tests/moe"), Path("tests/gdn")]


def test_collapse_drops_nested_file(tmp_path: Path) -> None:
    parent = tmp_path / "suite"
    parent.mkdir()
    child = parent / "test_sample.py"
    child.write_text("def test_case(): pass\n", encoding="utf-8")

    assert runner.collapse_test_paths([parent, child]) == (parent.resolve(),)
    assert runner.collapse_test_paths([child, parent]) == (parent.resolve(),)


def test_missing_test_path_fails_closed(tmp_path: Path) -> None:
    present = tmp_path / "suite"
    present.mkdir()
    missing = tmp_path / "missing"
    selection = runner.SelectionSettings(
        test_paths=(present, missing),
        sanity_test=False,
        sample_rate=5,
        sample_offset=0,
    )

    with pytest.raises(runner.RunnerStateError, match="missing"):
        runner._validate_selection(selection)


def test_collect_nodes_unions_multiple_directories(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "test_a.py").write_text("def test_a(): pass\n", encoding="utf-8")
    (second / "test_b.py").write_text("def test_b(): pass\n", encoding="utf-8")

    nodes = runner._collect_nodes(REPO_ROOT, (first, second), 15, 0)
    nodeids = {node["nodeid"] for node in nodes}

    assert "test_a.py::test_a" in nodeids
    assert "test_b.py::test_b" in nodeids
