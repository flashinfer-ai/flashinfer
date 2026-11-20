from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .models import Unit
from .workers import write_console


def _next_worker_unit(
    work_by_worker: list[list[Unit]],
    worker_index: int,
    can_run: Callable[[Unit], bool] = lambda unit: True,
) -> Unit | None:
    # The resource queue holds its condition while selecting and reserving work.
    donors = sorted(
        (
            (len(work), index, work)
            for index, work in enumerate(work_by_worker)
            if index != worker_index
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for work in [work_by_worker[worker_index], *(work for _, _, work in donors)]:
        for position, unit in enumerate(work):
            if can_run(unit):
                return work.pop(position)
    return None


@dataclass
class ResourceQueue:
    """Dispatch queued units under categorical host/GPU isolation rules.

    Reservations cover setup through process cleanup. These rules prevent
    selected overlaps; they do not measure or enforce memory capacity.
    """

    devices: list[str | None]
    long_work_by_worker: list[list[Unit]]
    normal_work_by_worker: list[list[Unit]]
    gpu_exclusive_sources: frozenset[str] = frozenset()
    host_memory_heavy_sources: frozenset[str] = frozenset()
    stop: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)
    active: dict[int, Unit] = field(default_factory=dict)
    draining: dict[str | None, Unit] = field(default_factory=dict)

    def request_stop(self) -> None:
        """Wake idle workers so shutdown does not depend on active work finishing."""
        with self.condition:
            self.stop.set()
            self.condition.notify_all()

    def _gpu_exclusive(self, unit: Unit) -> bool:
        return any(b.source_file in self.gpu_exclusive_sources for b in unit.batches)

    def _host_heavy(self, unit: Unit) -> bool:
        return any(
            b.source_file in self.host_memory_heavy_sources for b in unit.batches
        )

    def _take(self, worker: int) -> Unit | None:
        device = self.devices[worker]
        gpu_active = [
            unit for index, unit in self.active.items() if self.devices[index] == device
        ]
        host_busy = any(self._host_heavy(unit) for unit in self.active.values())
        self.draining = {
            gpu: unit
            for gpu, unit in self.draining.items()
            if not (host_busy and self._host_heavy(unit))
        }
        if any(self._gpu_exclusive(unit) for unit in gpu_active):
            return None

        def can_run(unit: Unit) -> bool:
            if host_busy and self._host_heavy(unit):
                return False
            if self._gpu_exclusive(unit):
                if not gpu_active:
                    return True
                # Drain one GPU for this file; other GPUs can still take light work.
                if unit not in self.draining.values():
                    self.draining.setdefault(device, unit)
                return False
            return device not in self.draining

        for queues in (self.long_work_by_worker, self.normal_work_by_worker):
            unit = _next_worker_unit(queues, worker, can_run)
            if unit is not None:
                self.active[worker] = unit
                self.draining = {
                    gpu: waiting
                    for gpu, waiting in self.draining.items()
                    if waiting.id != unit.id
                }
                self.condition.notify_all()
                return unit
        return None

    @contextmanager
    def reserve(self, worker: int) -> Iterator[Unit | None]:
        unit = None
        waited = 0.0
        with self.condition:
            while not self.stop.is_set():
                unit = self._take(worker)
                if unit is not None or not any(
                    self.long_work_by_worker + self.normal_work_by_worker
                ):
                    break
                started = time.monotonic()
                self.condition.wait()
                waited += time.monotonic() - started
        try:
            if waited >= 1:
                write_console(
                    f"RESOURCE WAIT worker={worker} device={self.devices[worker]} "
                    f"elapsed={waited:.3f}s"
                )
            yield unit
        finally:
            if unit is not None:
                with self.condition:
                    del self.active[worker]
                    self.condition.notify_all()
