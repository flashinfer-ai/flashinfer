"""Validated, file-backed CuTe artifacts for preparation peer transfers."""
from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from b12x._lib.cache_integrity import _fsync_directory, _mkdir_durable, valid_object


class CuTeArtifactCache:
    def __init__(self, root: Path | None = None):
        if root is None:
            from b12x._lib.compiler import _cute_compile_cache_dir
            root = _cute_compile_cache_dir()
        self.root = Path(root)

    def path(self, key: str, suffix: str) -> Path:
        if (len(key) != 64 or any(c not in "0123456789abcdef" for c in key)
                or suffix not in (".o", ".json", ".lock")):
            raise ValueError("invalid CuTe artifact path")
        return self.root / key[:2] / (key + suffix)

    def has(self, key: str) -> bool:
        return valid_object(self.path(key, ".o"), self.path(key, ".json"), key)

    @contextmanager
    def stage(self, key: str):
        directory = self.path(key, ".o").parent
        _mkdir_durable(directory)
        with tempfile.TemporaryDirectory(dir=directory, prefix=".peer-") as temporary:
            yield Path(temporary)

    def publish(self, key: str, staged: Path) -> bool:
        """Publish a verified pair without waiting for an active local compiler."""
        object_path, manifest_path = (staged / (key + suffix) for suffix in (".o", ".json"))
        if not valid_object(object_path, manifest_path, key):
            return False
        with self.path(key, ".lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return self.has(key)
            try:
                if self.has(key):
                    return True
                for path in (object_path, manifest_path):
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    os.replace(path, self.path(key, path.suffix))
                _fsync_directory(self.path(key, ".o").parent)
                return True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
