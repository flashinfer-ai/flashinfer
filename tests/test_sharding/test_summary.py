from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from scripts.test_sharding import summary as summary_module
from scripts.test_sharding.models import Plan, PlanningOptions
from scripts.test_sharding.state import state_lock


def test_summary_publication_waits_for_the_state_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = Plan(
        options=PlanningOptions(profile="synthetic"),
        nodes=(),
        units=(),
    )
    lock_attempted = threading.Event()

    @contextmanager
    def observed_state_lock(junit_dir: Path) -> Iterator[None]:
        lock_attempted.set()
        with state_lock(junit_dir):
            yield

    monkeypatch.setattr(summary_module, "state_lock", observed_state_lock)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with state_lock(tmp_path):
            future = executor.submit(summary_module.publish_summary, tmp_path, plan)
            assert lock_attempted.wait(timeout=5)
            assert not future.done()
        summary = future.result(timeout=5)
    finally:
        executor.shutdown()

    assert summary["complete"] is True
    assert summary["schema_version"] == 2
