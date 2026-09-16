from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence, TypedDict

from .io import atomic_write_json
from .models import ALGORITHM_VERSION, SCHEMA_VERSION, CollectedNode, Plan


class RunnerStateError(RuntimeError):
    pass


_LOCK_MARKER = "flashinfer-unit-test-runner-lock-v1\n"


def _publish_runner_lock(lock_path: Path) -> None:
    """Publish a complete lock marker without exposing an empty lock file."""

    if os.path.lexists(lock_path):
        return

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=lock_path.parent, prefix=".lock-"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(_LOCK_MARKER)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, lock_path)
        except FileExistsError:
            pass
        except OSError as error:
            raise RunnerStateError(
                f"cannot create runner-owned lock path {lock_path}: {error}"
            ) from error
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def state_lock(junit_dir: Path) -> Iterator[None]:
    junit_dir.mkdir(parents=True, exist_ok=True)
    lock_path = junit_dir / "lock"
    _publish_runner_lock(lock_path)
    try:
        stream = lock_path.open("r+")
    except OSError as error:
        raise RunnerStateError(
            f"cannot open runner-owned lock path {lock_path}: {error}"
        ) from error
    with stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            stream.seek(0)
            marker = stream.read()
            if marker != _LOCK_MARKER:
                raise RunnerStateError(
                    f"runner-owned lock path has foreign content: {lock_path}"
                )
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_source_git_sha(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def source_git_sha_from_env() -> str | None:
    """Return the optional CI archive identity used to validate resume."""

    # SOURCE_GIT_SHA is injected by callers that want plan and run jobs to prove
    # they are using the same source archive. Local runs and older wrappers may
    # omit it, so absence is treated as "no SHA guard available".
    return _normalize_source_git_sha(os.environ.get("SOURCE_GIT_SHA"))


def collection_fingerprint(plan_or_nodes: Plan | Sequence[CollectedNode]) -> str:
    digest = hashlib.sha256()
    nodes = plan_or_nodes.nodes if isinstance(plan_or_nodes, Plan) else plan_or_nodes
    for node in nodes:
        digest.update(
            json.dumps(node.to_dict(), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _unit_elapsed_path(attempt_path: Path, unit_id: str) -> Path:
    return attempt_path / "unit-elapsed" / f"{unit_id}.json"


def write_unit_elapsed(
    attempt_path: Path,
    unit_id: str,
    *,
    elapsed_seconds: float,
    active_started_at: float | None,
) -> None:
    atomic_write_json(
        _unit_elapsed_path(attempt_path, unit_id),
        {
            "active_started_at": active_started_at,
            "elapsed_seconds": max(0.0, elapsed_seconds),
        },
    )


def recover_unit_elapsed(
    attempt_path: Path,
    unit_id: str,
    *,
    stale_claim_path: Path,
    now: float | None = None,
) -> float:
    """Recover an attempt's unit budget, bounded by its last claim lease."""

    path = _unit_elapsed_path(attempt_path, unit_id)
    if not path.exists():
        return 0.0
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        elapsed = float(value.get("elapsed_seconds", 0.0))
        active_started_at = value.get("active_started_at")
        active_started_at = (
            float(active_started_at) if active_started_at is not None else None
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerStateError(f"invalid unit elapsed state {path}: {error}") from error
    if active_started_at is not None:
        current = time.time() if now is None else now
        try:
            claim = json.loads(stale_claim_path.read_text(encoding="utf-8"))
            cutoff = min(current, float(claim["expires_at"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            cutoff = active_started_at
        elapsed += max(0.0, cutoff - active_started_at)
        write_unit_elapsed(
            attempt_path,
            unit_id,
            elapsed_seconds=elapsed,
            active_started_at=None,
        )
    return elapsed


def manifest_path(junit_dir: Path) -> Path:
    return junit_dir / "manifest.json"


def claims_dir(junit_dir: Path) -> Path:
    return junit_dir / "claims"


def units_dir(junit_dir: Path) -> Path:
    return junit_dir / "units"


def load_manifest(junit_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(junit_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerStateError(
            f"cannot read runner manifest {path}: {error}"
        ) from error


def write_manifest(junit_dir: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(manifest_path(junit_dir), manifest)


@dataclass(frozen=True)
class ManifestBuild:
    repo_root: Path
    test_path: Path
    source_git_sha: str | None
    plan: Plan
    selection: dict[str, Any]
    estimate_files: dict[str, str | None]
    pytest_command_prefix: tuple[str, ...] = ()
    test_paths: tuple[Path, ...] | None = None


def _resolved_test_paths(
    test_path: Path, test_paths: Sequence[Path] | None = None
) -> tuple[Path, ...]:
    if test_paths:
        return tuple(path.resolve() for path in test_paths)
    return (test_path.resolve(),)


def build_manifest(request: ManifestBuild) -> dict[str, Any]:
    repo_root = request.repo_root
    test_paths = _resolved_test_paths(request.test_path, request.test_paths)
    plan = request.plan
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "created_at": time.time(),
        "repository_root": str(repo_root.resolve()),
        "source_git_sha": request.source_git_sha,
        "collection_fingerprint": collection_fingerprint(plan),
        "test_path": " ".join(str(path) for path in test_paths),
        "test_paths": [str(path) for path in test_paths],
        "selection": request.selection,
        "pytest_command_prefix": list(request.pytest_command_prefix),
        "estimate_files": request.estimate_files,
        "plan": plan.to_dict(),
    }


def verify_manifest(
    manifest: dict[str, Any],
    *,
    source_git_sha: str | None,
    test_path: Path,
    selection: dict[str, Any],
    planning_options: dict[str, Any],
    pytest_command_prefix: tuple[str, ...] = (),
    test_paths: Sequence[Path] | None = None,
) -> None:
    mismatches: list[str] = []
    resolved_paths = _resolved_test_paths(test_path, test_paths)
    expected_test_path = " ".join(str(path) for path in resolved_paths)
    checks = {
        "schema_version": (manifest.get("schema_version"), SCHEMA_VERSION),
        "algorithm_version": (
            manifest.get("algorithm_version"),
            ALGORITHM_VERSION,
        ),
        "test_path": (manifest.get("test_path"), expected_test_path),
        "selection": (manifest.get("selection"), selection),
        "pytest_command_prefix": (
            tuple(manifest.get("pytest_command_prefix", ())),
            pytest_command_prefix,
        ),
        "planning_options": (
            manifest.get("plan", {}).get("options"),
            planning_options,
        ),
    }
    if "test_paths" in manifest:
        checks["test_paths"] = (
            manifest.get("test_paths"),
            [str(path) for path in resolved_paths],
        )
    for name, (saved, current) in checks.items():
        if saved != current:
            mismatches.append(f"{name}: saved={saved!r}, current={current!r}")
    saved_source_git_sha = _normalize_source_git_sha(manifest.get("source_git_sha"))
    current_source_git_sha = _normalize_source_git_sha(source_git_sha)
    # The source SHA is an optional compatibility guard. When both sides provide
    # it, a mismatch means the saved plan belongs to a different source state.
    # When either side omits it, rely on the mandatory manifest fields above.
    if (
        saved_source_git_sha is not None
        and current_source_git_sha is not None
        and saved_source_git_sha != current_source_git_sha
    ):
        mismatches.append(
            "source_git_sha: "
            f"saved={saved_source_git_sha!r}, current={current_source_git_sha!r}"
        )
    if mismatches:
        raise RunnerStateError(
            "existing run is incompatible; use a different --junit-dir:\n  "
            + "\n  ".join(mismatches)
        )


class AttemptSettingsRecord(TypedDict):
    deadline_seconds: int
    unit_timeout_seconds: int
    timeout_grace_seconds: int
    timeout_policy: str


class AttemptRecord(TypedDict):
    schema_version: int
    id: str
    started_at: float
    deadline_at: float | None
    settings: AttemptSettingsRecord


@dataclass(frozen=True)
class AttemptSettings:
    deadline_seconds: int
    unit_timeout_seconds: int
    timeout_grace_seconds: int
    timeout_policy: str

    def to_dict(self) -> AttemptSettingsRecord:
        return {
            "deadline_seconds": self.deadline_seconds,
            "unit_timeout_seconds": self.unit_timeout_seconds,
            "timeout_grace_seconds": self.timeout_grace_seconds,
            "timeout_policy": self.timeout_policy,
        }


def attempts_dir(junit_dir: Path) -> Path:
    return junit_dir / "attempts"


def _attempt_number(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _attempt_paths(junit_dir: Path, *, require_complete: bool) -> list[Path]:
    root = attempts_dir(junit_dir)
    if not root.exists():
        return []
    if not root.is_dir():
        raise RunnerStateError(f"runner-owned attempts path is not a directory: {root}")
    entries = list(root.iterdir())
    invalid = {
        path
        for path in entries
        if not path.is_dir()
        or _attempt_number(path) < 0
        or path.name != f"attempt-{_attempt_number(path):04d}"
    }
    incomplete = [path for path in entries if not (path / "attempt.json").is_file()]
    if require_complete:
        invalid.update(incomplete)
    if invalid:
        raise RunnerStateError(
            "runner-owned attempts path contains invalid entries: "
            + ", ".join(str(path) for path in sorted(invalid)[:5])
        )
    return sorted(
        (path for path in entries if path not in incomplete), key=_attempt_number
    )


def list_attempts(junit_dir: Path) -> list[Path]:
    return _attempt_paths(junit_dir, require_complete=False)


def load_attempt(path: Path) -> AttemptRecord:
    return json.loads((path / "attempt.json").read_text(encoding="utf-8"))


def active_attempt_collection_timeout(
    junit_dir: Path,
    requested_timeout: float | None,
    *,
    now: float | None = None,
) -> float | None:
    """Cap re-collection by the shared deadline of an active attempt."""

    attempts = list_attempts(junit_dir)
    if not attempts or (attempts[-1] / "closed.json").exists():
        return requested_timeout
    attempt = load_attempt(attempts[-1])
    deadline_at = attempt.get("deadline_at")
    if deadline_at is None:
        return requested_timeout
    remaining = max(0.001, float(deadline_at) - (time.time() if now is None else now))
    return remaining if requested_timeout is None else min(requested_timeout, remaining)


def create_or_join_attempt(
    junit_dir: Path,
    settings: AttemptSettings,
    *,
    started_at: float,
) -> tuple[Path, AttemptRecord]:
    with state_lock(junit_dir):
        attempts_root_exists = attempts_dir(junit_dir).exists()
        attempts = _attempt_paths(junit_dir, require_complete=True)
        if attempts_root_exists and not attempts:
            raise RunnerStateError(
                "runner-owned attempts path is empty: " + str(attempts_dir(junit_dir))
            )
        if attempts and not (attempts[-1] / "closed.json").exists():
            path = attempts[-1]
            active_attempt = load_attempt(path)
            saved_settings = active_attempt["settings"]
            if saved_settings != settings.to_dict():
                raise RunnerStateError(
                    "active attempt settings differ: "
                    f"saved={saved_settings!r}, current={settings.to_dict()!r}"
                )
            return path, active_attempt
        number = _attempt_number(attempts[-1]) + 1 if attempts else 1
        attempt_id = f"attempt-{number:04d}"
        path = attempts_dir(junit_dir) / attempt_id
        path.mkdir(parents=True, exist_ok=False)
        deadline_at = (
            started_at + settings.deadline_seconds
            if settings.deadline_seconds > 0
            else None
        )
        new_attempt: AttemptRecord = {
            "schema_version": SCHEMA_VERSION,
            "id": attempt_id,
            "started_at": started_at,
            "deadline_at": deadline_at,
            "settings": settings.to_dict(),
        }
        atomic_write_json(path / "attempt.json", new_attempt)
        return path, new_attempt


def lease_is_live(path: Path, now: float | None = None) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return float(value["expires_at"]) > (time.time() if now is None else now)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def lease_value(
    *, attempt_id: str, shard_index: int, worker: str, ttl_seconds: int = 30
) -> dict[str, Any]:
    now = time.time()
    return {
        "attempt_id": attempt_id,
        "shard_index": shard_index,
        "worker": worker,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "renewed_at": now,
        "expires_at": now + ttl_seconds,
    }
