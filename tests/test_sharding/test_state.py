from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts.test_sharding.state import (
    AttemptSettings,
    RunnerStateError,
    active_attempt_collection_timeout,
    create_or_join_attempt,
    fingerprint_paths,
    recover_unit_elapsed,
    state_lock,
    write_unit_elapsed,
)


def test_unit_elapsed_recovers_only_the_stale_claim_window(tmp_path: Path) -> None:
    attempt_path = tmp_path / "attempt-0001"
    claim_path = tmp_path / "claim.json"
    write_unit_elapsed(
        attempt_path,
        "unit-a",
        elapsed_seconds=12.0,
        active_started_at=100.0,
    )
    claim_path.write_text(json.dumps({"expires_at": 130.0}), encoding="utf-8")

    elapsed = recover_unit_elapsed(
        attempt_path,
        "unit-a",
        stale_claim_path=claim_path,
        now=200.0,
    )

    assert elapsed == 42.0
    saved = json.loads(
        (attempt_path / "unit-elapsed" / "unit-a.json").read_text(encoding="utf-8")
    )
    assert saved == {"active_started_at": None, "elapsed_seconds": 42.0}


def test_repository_fingerprint_ignores_only_timing_refresh_artifacts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    test_dir = repo / "tests"
    data_dir = test_dir / "data"
    data_dir.mkdir(parents=True)
    (test_dir / "test_sample.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    duration = data_dir / "unit_test_duration_estimates.csv.gz"
    duration.write_text("first", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "tests@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test Runner"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    initial = fingerprint_paths(repo, test_dir)

    duration.write_text("refreshed", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "refresh timing"], check=True
    )
    assert fingerprint_paths(repo, test_dir) == initial

    (data_dir / "test_fixture.json").write_text("{}\n", encoding="utf-8")
    assert fingerprint_paths(repo, test_dir) != initial
    assert fingerprint_paths(repo, test_dir, excluded_roots=(data_dir,)) == initial


def test_repository_fingerprint_includes_staged_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    test_dir = repo / "tests"
    test_dir.mkdir(parents=True)
    test_file = test_dir / "test_sample.py"
    original = "def test_ok(): pass\n"
    test_file.write_text(original, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "tests@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test Runner"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "initial"], check=True)
    initial = fingerprint_paths(repo, test_dir)

    test_file.write_text("def test_ok(): assert False\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", str(test_file)], check=True)
    assert fingerprint_paths(repo, test_dir) != initial

    subprocess.run(
        ["git", "-C", str(repo), "restore", "--staged", str(test_file)], check=True
    )
    test_file.write_text(original, encoding="utf-8")
    assert fingerprint_paths(repo, test_dir) == initial

    test_file.unlink()
    subprocess.run(["git", "-C", str(repo), "add", "--update"], check=True)
    assert fingerprint_paths(repo, test_dir) != initial


def test_collection_timeout_uses_active_attempt_absolute_deadline(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempts" / "attempt-0001"
    attempt.mkdir(parents=True)
    (attempt / "attempt.json").write_text(
        json.dumps({"deadline_at": 125.0}), encoding="utf-8"
    )

    assert active_attempt_collection_timeout(tmp_path, 100.0, now=100.0) == 25.0
    assert active_attempt_collection_timeout(tmp_path, 10.0, now=100.0) == 10.0
    assert active_attempt_collection_timeout(tmp_path, None, now=100.0) == 25.0


def test_state_lock_rejects_foreign_lock_file(tmp_path: Path) -> None:
    (tmp_path / "lock").write_text("foreign\n", encoding="utf-8")

    with pytest.raises(RunnerStateError, match="lock"), state_lock(tmp_path):
        pass


def test_state_lock_supports_simultaneous_first_start(tmp_path: Path) -> None:
    worker_count = 8
    start = threading.Barrier(worker_count)

    def acquire_lock() -> None:
        start.wait()
        with state_lock(tmp_path):
            pass

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(acquire_lock) for _ in range(worker_count)]
        for future in futures:
            future.result()


def test_create_attempt_rejects_empty_reserved_attempts_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "attempts").mkdir()
    settings = AttemptSettings(
        deadline_seconds=60,
        unit_timeout_seconds=30,
        timeout_grace_seconds=5,
        timeout_policy="resume",
    )

    with pytest.raises(RunnerStateError, match="attempts"):
        create_or_join_attempt(tmp_path, settings, started_at=100.0)
