# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Durable, host-only persistence helpers for the Prims-TS MoE sweep.

Only the driver may write the JSONL journal or checkpoint.  Workers return
plain data to the driver and never open either file for writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping


CHECKPOINT_SCHEMA_VERSION = 2
_RESULTS_V3_TO_V5_CHECKPOINT_SCHEMA_VERSION = 1


class JournalCorruptionError(ValueError):
    """Raised when corruption occurs before the journal's trailing fragment."""


class DriverOnlyWriterError(RuntimeError):
    """Raised when a forked/non-owner process tries to use a driver writer."""


@dataclass(frozen=True)
class JSONLRecovery:
    """Result of scanning and, optionally, repairing a JSONL journal."""

    records: tuple[dict[str, Any], ...]
    valid_bytes: int
    truncated_bytes: int = 0
    added_newline: bool = False


def canonical_json(value: Any) -> str:
    """Return the stable JSON encoding used for IDs and on-disk records."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def deterministic_id(value: Any, *, length: int | None = None) -> str:
    """Hash JSON-safe data without depending on dict insertion order."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def _decode_record(raw: bytes, *, line_number: int, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JournalCorruptionError(
            f"{path}: invalid JSON on complete line {line_number}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise JournalCorruptionError(
            f"{path}: line {line_number} is {type(value).__name__}, not an object"
        )
    return value


def recover_jsonl(
    path: str | os.PathLike[str],
    *,
    repair: bool = True,
    fsync: bool = False,
) -> JSONLRecovery:
    """Read a JSONL journal and recover an interrupted final append.

    An invalid final line without a newline is treated as a torn append and may
    be truncated.  Invalid newline-terminated lines and invalid lines before
    the end are corruption and are never silently discarded.  A valid final
    object missing only its newline is retained and repaired.
    """

    journal = Path(path)
    if not journal.exists():
        return JSONLRecovery(records=(), valid_bytes=0)

    data = journal.read_bytes()
    if not data:
        return JSONLRecovery(records=(), valid_bytes=0)

    records: list[dict[str, Any]] = []
    valid_bytes = 0
    truncated_bytes = 0
    add_newline = False
    lines = data.splitlines(keepends=True)

    for index, line in enumerate(lines, start=1):
        complete = line.endswith(b"\n")
        raw = line[:-1] if complete else line
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        is_last = index == len(lines)

        if complete or not is_last:
            records.append(_decode_record(raw, line_number=index, path=journal))
            valid_bytes += len(line)
            continue

        # A non-newline-terminated final fragment may be either a complete JSON
        # value whose newline was lost, or an interrupted write.
        try:
            records.append(_decode_record(raw, line_number=index, path=journal))
        except JournalCorruptionError:
            truncated_bytes = len(line)
        else:
            valid_bytes += len(line)
            add_newline = True

    if repair and (truncated_bytes or add_newline):
        with journal.open("r+b") as stream:
            if truncated_bytes:
                stream.truncate(valid_bytes)
                stream.seek(0, os.SEEK_END)
            if add_newline:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                valid_bytes += 1
            stream.flush()
            if fsync:
                os.fsync(stream.fileno())

    return JSONLRecovery(
        records=tuple(records),
        valid_bytes=valid_bytes,
        truncated_bytes=truncated_bytes,
        added_newline=add_newline,
    )


def read_jsonl(
    path: str | os.PathLike[str], *, recover_trailing: bool = True
) -> list[dict[str, Any]]:
    """Read journal records, optionally repairing a torn trailing append."""

    return list(recover_jsonl(path, repair=recover_trailing).records)


class JSONLWriter:
    """Append-only JSONL writer bound to the creating driver process."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        flush: bool = True,
        fsync: bool = False,
        recover: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if recover:
            recover_jsonl(self.path, repair=True, fsync=fsync)
        self._stream = self.path.open("ab")
        self._flush = flush or fsync
        self._fsync = fsync
        self._owner_pid = os.getpid()
        self._closed = False

    def _check_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise DriverOnlyWriterError(
                "the JSONL writer belongs to the driver process "
                f"{self._owner_pid}, not process {os.getpid()}"
            )
        if self._closed:
            raise ValueError("write to closed JSONL writer")

    def append(
        self,
        record: Mapping[str, Any],
        *,
        flush: bool | None = None,
        fsync: bool | None = None,
    ) -> None:
        """Append exactly one object as one compact line."""

        self._check_owner()
        encoded = (canonical_json(dict(record)) + "\n").encode("utf-8")
        self._stream.write(encoded)
        should_fsync = self._fsync if fsync is None else fsync
        should_flush = self._flush if flush is None else flush
        if should_flush or should_fsync:
            self._stream.flush()
        if should_fsync:
            os.fsync(self._stream.fileno())

    def flush(self, *, fsync: bool | None = None) -> None:
        self._check_owner()
        self._stream.flush()
        if self._fsync if fsync is None else fsync:
            os.fsync(self._stream.fileno())

    def close(self) -> None:
        if self._closed:
            return
        self._check_owner()
        self._stream.flush()
        if self._fsync:
            os.fsync(self._stream.fileno())
        self._stream.close()
        self._closed = True

    def __enter__(self) -> JSONLWriter:
        self._check_owner()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def atomic_write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    fsync: bool = True,
) -> None:
    """Atomically replace a JSON file and optionally sync file and directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write((canonical_json(value) + "\n").encode("utf-8"))
            stream.flush()
            if fsync:
                os.fsync(stream.fileno())
        os.replace(temporary, destination)
        if fsync:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def encode_rng_state(state: object) -> dict[str, Any]:
    """Encode ``random.Random.getstate()`` without lossy tuple conversion."""

    if not isinstance(state, tuple) or len(state) != 3:
        raise ValueError("unsupported random state")
    version, internal, gauss_next = state
    if not isinstance(version, int) or not isinstance(internal, tuple):
        raise ValueError("unsupported random state")
    encoded = {
        "version": version,
        "internal": list(internal),
        "gauss_next": gauss_next,
    }
    # Validate finite numbers and JSON compatibility now, not at checkpoint time.
    canonical_json(encoded)
    return encoded


def decode_rng_state(value: Mapping[str, Any]) -> tuple[Any, ...]:
    """Decode a state produced by :func:`encode_rng_state`."""

    try:
        version = int(value["version"])
        internal = tuple(int(item) for item in value["internal"])
        gauss_next = value.get("gauss_next")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid encoded random state") from error
    state = (version, internal, gauss_next)
    # Let random validate implementation-specific details.
    validator = random.Random()
    try:
        validator.setstate(state)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid encoded random state") from error
    return state


def capture_rng_state(rng: random.Random) -> dict[str, Any]:
    return encode_rng_state(rng.getstate())


def restore_rng_state(rng: random.Random, state: Mapping[str, Any]) -> None:
    rng.setstate(decode_rng_state(state))


@dataclass
class SweepCheckpoint:
    """Complete driver search state needed for deterministic continuation."""

    rng_state: dict[str, Any]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    sequence: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        rng: random.Random,
        *,
        sequence: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> SweepCheckpoint:
        return cls(
            rng_state=capture_rng_state(rng),
            sequence=sequence,
            extra=dict(extra or {}),
        )

    def restore_rng(self, rng: random.Random | None = None) -> random.Random:
        restored = rng or random.Random()
        restore_rng_state(restored, self.rng_state)
        return restored

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "rng_state": self.rng_state,
            "sequence": self.sequence,
            "extra": self.extra,
        }
        canonical_json(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SweepCheckpoint:
        version = int(value.get("schema_version", -1))
        if version not in (
            _RESULTS_V3_TO_V5_CHECKPOINT_SCHEMA_VERSION,
            CHECKPOINT_SCHEMA_VERSION,
        ):
            raise ValueError(
                f"unsupported checkpoint schema {version}; "
                f"expected {_RESULTS_V3_TO_V5_CHECKPOINT_SCHEMA_VERSION} or "
                f"{CHECKPOINT_SCHEMA_VERSION}"
            )
        rng_state = value.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise ValueError("checkpoint rng_state is missing or invalid")
        # Validate before accepting a checkpoint.
        decode_rng_state(rng_state)
        return cls(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            rng_state=dict(rng_state),
            sequence=int(value.get("sequence", 0)),
            extra=dict(value.get("extra", {})),
        )

    def save(self, path: str | os.PathLike[str], *, fsync: bool = True) -> None:
        atomic_write_json(path, self.to_dict(), fsync=fsync)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> SweepCheckpoint:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("checkpoint root must be an object")
        return cls.from_dict(value)
