from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts.test_sharding import io


def test_concurrent_atomic_writes_use_distinct_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "state.json"
    replace_barrier = threading.Barrier(2)
    real_replace = os.replace

    def synchronized_replace(source: Path, target: Path) -> None:
        replace_barrier.wait(timeout=5)
        real_replace(source, target)

    monkeypatch.setattr(io.os, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(io.atomic_write_bytes, destination, content)
            for content in (b"first", b"second")
        ]
        for future in futures:
            future.result(timeout=5)

    assert destination.read_bytes() in {b"first", b"second"}
