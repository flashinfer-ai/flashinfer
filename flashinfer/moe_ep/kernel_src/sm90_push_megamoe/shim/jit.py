"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .....jit import env as jit_env
from .....jit.core import JitSpec, gen_jit_spec, sm90a_nvcc_flags

__all__ = ["gen_sm90_push_a2a_module", "sm90_push_a2a_uri"]

_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "a2a"
_SOURCE_NAMES = ("sm90_push_a2a_ops.cu", "sm90_push_a2a.cuh")


def _canonical_source(source: bytes) -> bytes:
    return source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _source_blobs() -> dict[str, bytes]:
    return {
        name: _canonical_source((_SOURCE_DIR / name).read_bytes())
        for name in _SOURCE_NAMES
    }


def _snapshot_matches(path: Path, content: str) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            return source.read() == content
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return False


def _write_snapshot_atomic(path: Path, content: str) -> None:
    if _snapshot_matches(path, content):
        return

    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as snapshot:
            snapshot.write(content)
        os.replace(temp_path, path)
        temp_path = None
    except PermissionError:
        if not _snapshot_matches(path, content):
            raise
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()


def sm90_push_a2a_uri(cuda_flags: tuple[str, ...] | None = None) -> str:
    """Return the content-addressed module name for sources and passed CUDA flags."""
    flags = tuple(sm90a_nvcc_flags) if cuda_flags is None else cuda_flags
    digest = hashlib.sha256()
    for name, source in _source_blobs().items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(source)
        digest.update(b"\0")
    digest.update(json.dumps(flags, separators=(",", ":")).encode())
    return f"sm90_push_a2a_{digest.hexdigest()[:20]}"


def gen_sm90_push_a2a_module() -> JitSpec:
    """Snapshot package sources into the writable JIT cache and return a spec."""
    flags = tuple(sm90a_nvcc_flags)
    uri = sm90_push_a2a_uri(flags)
    output_dir = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    sources = _source_blobs()
    for name, source in sources.items():
        snapshot = source.decode()
        _write_snapshot_atomic(output_dir / name, snapshot)
    return gen_jit_spec(
        uri,
        [output_dir / "sm90_push_a2a_ops.cu"],
        extra_cuda_cflags=list(flags),
        extra_include_paths=[output_dir],
    )
