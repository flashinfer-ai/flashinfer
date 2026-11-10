from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from scripts.test_sharding.models import Batch, Unit
from scripts.test_sharding.scheduling import ResourceQueue, _next_worker_unit


def _queue_test_unit(identifier: str) -> Unit:
    return Unit(
        id=identifier,
        batches=(
            Batch(
                identifier, f"{identifier}.py", (f"{identifier}.py::test",), 1, 0, False
            ),
        ),
        estimated_ms=1,
        oversized=False,
    )


def test_worker_unit_selection_prefers_local_queue_before_stealing() -> None:
    work_by_worker = [[_queue_test_unit("donor")], [_queue_test_unit("local")]]
    assert _next_worker_unit(work_by_worker, 1).id == "local"
    assert _next_worker_unit(work_by_worker, 1).id == "donor"


def test_worker_unit_stealing_drains_queued_units_exactly_once() -> None:
    work_by_worker = [[] for _ in range(4)]
    expected = [f"unit-{index}" for index in range(40)]
    for index, identifier in enumerate(expected):
        work_by_worker[index % 2].append(_queue_test_unit(identifier))
    resources = ResourceQueue(
        devices=["4", "6", "4", "6"],
        long_work_by_worker=[[] for _ in range(4)],
        normal_work_by_worker=work_by_worker,
        gpu_exclusive_sources=frozenset(f"unit-{i}.py" for i in range(0, 40, 5)),
        host_memory_heavy_sources=frozenset(f"unit-{i}.py" for i in range(0, 40, 7)),
    )

    def drain(worker_index: int) -> list[str]:
        completed = []
        while True:
            with resources.reserve(worker_index) as unit:
                if unit is None:
                    return completed
                completed.append(unit.id)
                time.sleep(0.001)

    with ThreadPoolExecutor(max_workers=4) as executor:
        completed = [
            identifier
            for worker_units in executor.map(drain, range(4))
            for identifier in worker_units
        ]

    assert sorted(completed) == sorted(expected)
    assert not any(work_by_worker)
    assert not resources.active
    assert not resources.draining


def test_resource_reservation_releases_after_failure() -> None:
    resources = ResourceQueue(
        devices=["4"],
        long_work_by_worker=[[_queue_test_unit("heavy"), _queue_test_unit("next")]],
        normal_work_by_worker=[[]],
        gpu_exclusive_sources=frozenset({"heavy.py"}),
        host_memory_heavy_sources=frozenset({"heavy.py", "next.py"}),
    )
    with pytest.raises(RuntimeError), resources.reserve(0) as unit:
        assert unit.id == "heavy"
        raise RuntimeError("batch failed")
    assert not resources.active
    with resources.reserve(0) as unit:
        assert unit.id == "next"


def test_shutdown_wakes_worker_waiting_for_host_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources = ResourceQueue(
        devices=["4", "6"],
        long_work_by_worker=[
            [_queue_test_unit("active")],
            [_queue_test_unit("blocked")],
        ],
        normal_work_by_worker=[[], []],
        host_memory_heavy_sources=frozenset({"active.py", "blocked.py"}),
    )
    waiting = threading.Event()
    original_wait = resources.condition.wait

    def wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(resources.condition, "wait", wait)

    def reserve_blocked():
        with resources.reserve(1) as unit:
            return unit

    # Release the reservation before executor shutdown if an assertion fails.
    with ThreadPoolExecutor(max_workers=1) as executor, resources.reserve(0):
        future = executor.submit(reserve_blocked)
        try:
            assert waiting.wait(5), "worker did not wait for host capacity"
        finally:
            resources.request_stop()
        # The active reservation is still held: only stop+notify can wake it.
        assert future.result(timeout=5) is None
        assert resources.long_work_by_worker[1][0].id == "blocked"


def test_gpu_wait_does_not_reserve_host_capacity() -> None:
    resources = ResourceQueue(
        devices=["4", "4", "6"],
        long_work_by_worker=[[], [], []],
        normal_work_by_worker=[
            [_queue_test_unit("light")],
            [_queue_test_unit("both")],
            [],
        ],
        gpu_exclusive_sources=frozenset({"both.py"}),
        host_memory_heavy_sources=frozenset({"both.py", "heavy.py"}),
    )
    with resources.reserve(0):
        with resources.condition:
            assert resources._take(1) is None
            resources.long_work_by_worker[2].append(_queue_test_unit("heavy"))
        with resources.reserve(2) as unit:
            assert unit.id == "heavy"
            with resources.condition:
                resources.normal_work_by_worker[1].append(
                    _queue_test_unit("next-light")
                )
            with resources.reserve(1) as light:
                assert light.id == "next-light"
    with resources.reserve(1) as unit:
        assert unit.id == "both"


def test_exclusive_file_drains_one_gpu_and_can_be_stolen() -> None:
    resources = ResourceQueue(
        devices=["4", "4", "6", "6"],
        long_work_by_worker=[[], [], [], []],
        normal_work_by_worker=[
            [_queue_test_unit("active")],
            [_queue_test_unit("exclusive"), _queue_test_unit("waiting")],
            [_queue_test_unit("other")],
            [],
        ],
        gpu_exclusive_sources=frozenset({"exclusive.py"}),
    )
    with resources.reserve(0), resources.reserve(2):
        with resources.condition:
            assert resources._take(1) is None
            assert set(resources.draining) == {"4"}
        with resources.reserve(3) as unit:
            assert unit.id == "waiting"
    # GPU 6 takes the exclusive file and clears GPU 4's drain request.
    with resources.reserve(2) as unit:
        assert unit.id == "exclusive"
        assert not resources.draining
