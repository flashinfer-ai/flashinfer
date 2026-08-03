from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from scripts.test_sharding.models import Plan, PlanningOptions
from scripts.test_sharding.state import (
    AttemptSettings,
    ManifestBuild,
    RunnerStateError,
    active_attempt_collection_timeout,
    build_manifest,
    create_or_join_attempt,
    recover_unit_elapsed,
    source_git_sha_from_env,
    state_lock,
    verify_manifest,
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


def _manifest(tmp_path: Path, source_git_sha: str | None) -> dict[str, Any]:
    test_path = tmp_path / "tests"
    plan = Plan(options=PlanningOptions(profile="test"), nodes=(), units=())
    return build_manifest(
        ManifestBuild(
            repo_root=tmp_path,
            test_path=test_path,
            source_git_sha=source_git_sha,
            plan=plan,
            selection={"sanity_test": False},
            estimate_files={"duration": None, "overhead": None},
        )
    )


def test_source_git_sha_from_env_normalizes_missing_and_blank_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOURCE_GIT_SHA", raising=False)
    assert source_git_sha_from_env() is None

    monkeypatch.setenv("SOURCE_GIT_SHA", "  ")
    assert source_git_sha_from_env() is None

    monkeypatch.setenv("SOURCE_GIT_SHA", " abc123 \n")
    assert source_git_sha_from_env() == "abc123"


def test_manifest_rejects_different_available_source_git_shas(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, "saved-sha")
    verify_manifest(
        manifest,
        source_git_sha="saved-sha",
        test_path=tmp_path / "tests",
        selection={"sanity_test": False},
        planning_options=PlanningOptions(profile="test").to_dict(),
    )

    with pytest.raises(RunnerStateError, match="source_git_sha"):
        verify_manifest(
            manifest,
            source_git_sha="current-sha",
            test_path=tmp_path / "tests",
            selection={"sanity_test": False},
            planning_options=PlanningOptions(profile="test").to_dict(),
        )


@pytest.mark.parametrize(
    ("saved_sha", "current_sha"),
    [(None, "current-sha"), ("saved-sha", None), (None, None)],
)
def test_manifest_assumes_source_matches_when_either_sha_is_unavailable(
    tmp_path: Path,
    saved_sha: str | None,
    current_sha: str | None,
) -> None:
    manifest = _manifest(tmp_path, saved_sha)

    verify_manifest(
        manifest,
        source_git_sha=current_sha,
        test_path=tmp_path / "tests",
        selection={"sanity_test": False},
        planning_options=PlanningOptions(profile="test").to_dict(),
    )


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
