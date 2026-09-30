"""Integrity checks and durable publication for the CuTe object cache."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import suppress
from functools import lru_cache
from pathlib import Path


def _file_identity(path: Path) -> tuple[int, ...]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("cache entry is not a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@lru_cache(maxsize=8192)
def _validate_object(
    object_path: Path,
    manifest_path: Path,
    cache_key: str,
    object_identity: tuple[int, ...],
    manifest_identity: tuple[int, ...],
) -> bool:
    # The identities participate in the memo key. Replacing or rewriting either
    # file invalidates the result, including same-size edits and repaired files.
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("cache_key") != cache_key:
        return False
    size = manifest.get("object_bytes")
    digest = manifest.get("object_sha256")
    if type(size) is not int or size <= 0 or size != object_identity[2]:
        return False
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    checksum = hashlib.sha256()
    with object_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest() == digest


def valid_object(object_path: Path, manifest_path: Path, cache_key: str) -> bool:
    """Treat incomplete or corrupt object/manifest pairs as cache misses.

    This checks stored bytes, not CUDA loadability or kernel correctness. A
    concurrent publisher may cause a miss; the compiler rechecks under its
    existing per-key lock before rebuilding. No cache files are removed here.
    """
    try:
        object_identity = _file_identity(object_path)
        manifest_identity = _file_identity(manifest_path)
        valid = _validate_object(
            object_path, manifest_path, cache_key, object_identity, manifest_identity,
        )
        return (valid and object_identity == _file_identity(object_path)
                and manifest_identity == _file_identity(manifest_path))
    except (OSError, ValueError, UnicodeError):
        return False


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    if path.is_dir():
        return
    _mkdir_durable(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise
    # Persist new shard directories as well as files within those shards.
    _fsync_directory(path.parent)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Publish a complete, fsynced file, then persist its directory entry.

    Writers must hold the compiler's per-key lock across object and manifest
    publication. Publish the manifest last; readers reject incomplete pairs.
    """
    _mkdir_durable(path.parent)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            with suppress(OSError):
                os.unlink(temporary)
