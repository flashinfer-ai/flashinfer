# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Byte-preserving importer for generated CUDA and cubin programs.

The producer owns generation and writes an immutable manifest.  This module
only verifies that manifest, copies the declared bytes without rewriting them,
and verifies the installed files again.  Keeping this layer workload-neutral
lets different generated program families share one promotion boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = 1
MANIFEST_KIND = "flashinfer.generated_program_promotion"
MODES = ("cuda", "cubin")
_ROOT_KEYS = {"artifacts", "kind", "mode", "name", "schema_version"}
_ARTIFACT_KEYS = {
    "destination",
    "executable",
    "sha256",
    "size_bytes",
    "source",
}
_NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PromotionIntegrityError(ValueError):
    """A promotion manifest or one of its declared artifacts is invalid."""


@dataclass(frozen=True)
class PromotionArtifact:
    source: PurePosixPath
    destination: PurePosixPath
    sha256: str
    size_bytes: int
    executable: bool


@dataclass(frozen=True)
class PromotionManifest:
    name: str
    mode: str
    artifacts: tuple[PromotionArtifact, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionIntegrityError(message)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionIntegrityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _relative_path(value: object, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    _require("\\" not in value, f"{label} must use POSIX separators")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and relative.as_posix() == value
        and value not in (".", "..")
        and ".." not in relative.parts,
        f"{label} must be a normalized relative path: {value!r}",
    )
    return relative


def _parse_artifact(value: object, index: int) -> PromotionArtifact:
    label = f"artifacts[{index}]"
    _require(isinstance(value, dict), f"{label} must be an object")
    keys = set(value)
    _require(
        keys == _ARTIFACT_KEYS,
        f"{label} keys must be exactly {sorted(_ARTIFACT_KEYS)}; got {sorted(keys)}",
    )
    sha256 = value["sha256"]
    _require(
        isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256) is not None,
        f"{label}.sha256 must be one full lowercase SHA-256",
    )
    size_bytes = value["size_bytes"]
    _require(
        isinstance(size_bytes, int)
        and not isinstance(size_bytes, bool)
        and size_bytes >= 0,
        f"{label}.size_bytes must be a non-negative integer",
    )
    executable = value["executable"]
    _require(isinstance(executable, bool), f"{label}.executable must be boolean")
    return PromotionArtifact(
        source=_relative_path(value["source"], f"{label}.source"),
        destination=_relative_path(value["destination"], f"{label}.destination"),
        sha256=sha256,
        size_bytes=size_bytes,
        executable=executable,
    )


def parse_manifest(payload: object) -> PromotionManifest:
    """Validate and normalize a decoded promotion manifest."""

    _require(isinstance(payload, dict), "manifest must be a JSON object")
    keys = set(payload)
    _require(
        keys == _ROOT_KEYS,
        f"manifest keys must be exactly {sorted(_ROOT_KEYS)}; got {sorted(keys)}",
    )
    _require(
        payload["schema_version"] == SCHEMA_VERSION,
        f"schema_version must be {SCHEMA_VERSION}",
    )
    _require(payload["kind"] == MANIFEST_KIND, f"kind must be {MANIFEST_KIND!r}")
    name = payload["name"]
    _require(
        isinstance(name, str) and _NAME_RE.fullmatch(name) is not None,
        "name must be a lowercase promotion identifier",
    )
    mode = payload["mode"]
    _require(mode in MODES, f"mode must be one of {MODES}")
    artifact_payloads = payload["artifacts"]
    _require(
        isinstance(artifact_payloads, list) and bool(artifact_payloads),
        "artifacts must be a non-empty list",
    )
    artifacts = tuple(
        _parse_artifact(value, index) for index, value in enumerate(artifact_payloads)
    )
    destinations = [item.destination.as_posix() for item in artifacts]
    sources = [item.source.as_posix() for item in artifacts]
    _require(
        destinations == sorted(destinations),
        "artifacts must be sorted by destination",
    )
    _require(len(destinations) == len(set(destinations)), "destinations repeat")
    _require(len(sources) == len(set(sources)), "sources repeat")
    destination_parts = [item.destination.parts for item in artifacts]
    for index, left in enumerate(destination_parts):
        for right in destination_parts[index + 1 :]:
            _require(
                left != right[: len(left)] and right != left[: len(right)],
                "one artifact destination is a parent of another",
            )
    return PromotionManifest(name=name, mode=mode, artifacts=artifacts)


def load_manifest(path: Path | str) -> PromotionManifest:
    """Load a manifest while rejecting duplicate JSON keys."""

    manifest_path = Path(path)
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionIntegrityError(
            f"could not read promotion manifest {manifest_path}: {exc}"
        ) from exc
    return parse_manifest(payload)


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _assert_no_symlink_components(path: Path, root: Path, label: str) -> None:
    relative = path.relative_to(root)
    cursor = root
    _require(not root.is_symlink(), f"{label} root must not be a symlink: {root}")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            _require(not cursor.is_symlink(), f"{label} traverses a symlink: {cursor}")


def _verify_file(path: Path, artifact: PromotionArtifact, label: str) -> None:
    try:
        file_stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PromotionIntegrityError(f"could not stat {label} {path}: {exc}") from exc
    _require(stat.S_ISREG(file_stat.st_mode), f"{label} is not a regular file: {path}")
    size, digest = _sha256_file(path)
    _require(
        size == artifact.size_bytes and digest == artifact.sha256,
        f"{label} identity mismatch for {path}: bytes={size} sha256={digest}; "
        f"expected bytes={artifact.size_bytes} sha256={artifact.sha256}",
    )


def _verify_destination(path: Path, artifact: PromotionArtifact) -> None:
    _verify_file(path, artifact, "destination")
    executable = bool(path.stat(follow_symlinks=False).st_mode & 0o111)
    _require(
        executable == artifact.executable,
        f"destination executable bit mismatch for {path}",
    )


def _existing_root(path: Path | str, label: str) -> Path:
    root = Path(path).absolute()
    _require(root.is_dir(), f"{label} root is not a directory: {root}")
    _require(
        root.resolve(strict=True) == root,
        f"{label} root must not traverse a symlink: {root}",
    )
    return root


def _require_mode(manifest: PromotionManifest, mode: str) -> None:
    _require(mode in MODES, f"requested mode must be one of {MODES}")
    _require(
        mode == manifest.mode,
        f"requested mode {mode!r} does not match manifest mode {manifest.mode!r}",
    )


def _verify_payload_inventory(
    root: Path, artifacts: Iterable[PromotionArtifact]
) -> None:
    expected = {artifact.source.as_posix() for artifact in artifacts}
    actual: set[str] = set()
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            path = directory_path / name
            _require(
                not path.is_symlink(),
                f"payload inventory contains a directory symlink: {path}",
            )
        for name in filenames:
            path = directory_path / name
            _require(
                not path.is_symlink(),
                f"payload inventory contains a file symlink: {path}",
            )
            file_stat = path.stat(follow_symlinks=False)
            _require(
                stat.S_ISREG(file_stat.st_mode),
                f"payload inventory contains a non-regular file: {path}",
            )
            actual.add(path.relative_to(root).as_posix())
    extras = sorted(actual - expected)
    missing = sorted(expected - actual)
    _require(
        not extras and not missing,
        f"payload inventory mismatch: extra={extras} missing={missing}",
    )


def _materialize_paths(
    root: Path, artifacts: Iterable[PromotionArtifact]
) -> list[tuple[PromotionArtifact, Path]]:
    return [
        (artifact, root.joinpath(*artifact.destination.parts)) for artifact in artifacts
    ]


def verify_promotion(
    manifest: PromotionManifest,
    *,
    payload_root: Path | str,
    output_root: Path | str,
    mode: str,
) -> None:
    """Rehash all producer inputs and imported outputs without writing."""

    source_root = _existing_root(payload_root, "payload")
    target_root = _existing_root(output_root, "output")
    _require_mode(manifest, mode)
    _verify_payload_inventory(source_root, manifest.artifacts)
    for artifact in manifest.artifacts:
        source = source_root.joinpath(*artifact.source.parts)
        destination = target_root.joinpath(*artifact.destination.parts)
        _assert_no_symlink_components(source, source_root, "source")
        _assert_no_symlink_components(destination, target_root, "destination")
        _verify_file(source, artifact, "source")
        _verify_destination(destination, artifact)


def import_promotion(
    manifest: PromotionManifest,
    *,
    payload_root: Path | str,
    output_root: Path | str,
    mode: str,
    replace: bool = False,
) -> None:
    """Verify and atomically copy every declared artifact into ``output_root``.

    Each file replacement is atomic.  All source identities and destination
    conflicts are checked before the first destination is replaced.
    """

    source_root = _existing_root(payload_root, "payload")
    _require_mode(manifest, mode)
    _verify_payload_inventory(source_root, manifest.artifacts)
    target_root = Path(output_root).absolute()
    if target_root.exists():
        target_root = _existing_root(target_root, "output")
    else:
        parent = target_root.parent
        while not parent.exists():
            parent = parent.parent
        _existing_root(parent, "output ancestor")
        target_root.mkdir(parents=True)
        target_root = _existing_root(target_root, "output")

    sources: list[tuple[PromotionArtifact, Path]] = []
    destinations = _materialize_paths(target_root, manifest.artifacts)
    for artifact, destination in destinations:
        source = source_root.joinpath(*artifact.source.parts)
        _assert_no_symlink_components(source, source_root, "source")
        _assert_no_symlink_components(destination, target_root, "destination")
        _verify_file(source, artifact, "source")
        sources.append((artifact, source))
        if destination.exists() or destination.is_symlink():
            _require(
                not destination.is_symlink(), f"destination is a symlink: {destination}"
            )
            _require(destination.is_file(), f"destination is not a file: {destination}")
            if not replace:
                _verify_destination(destination, artifact)

    staged: list[tuple[PromotionArtifact, Path, Path]] = []
    try:
        for (artifact, source), (_, destination) in zip(
            sources, destinations, strict=True
        ):
            if destination.exists() and not replace:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            _assert_no_symlink_components(destination, target_root, "destination")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".promotion",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_:
                    while chunk := input_.read(1024 * 1024):
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.chmod(0o755 if artifact.executable else 0o644)
                _verify_file(temporary, artifact, "staged destination")
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            staged.append((artifact, temporary, destination))

        for _, temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for _, temporary, _ in staged:
            temporary.unlink(missing_ok=True)

    verify_promotion(
        manifest,
        payload_root=source_root,
        output_root=target_root,
        mode=mode,
    )
