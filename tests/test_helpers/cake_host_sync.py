# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""Host-synchronization census of one call: does it copy from pageable memory or wait for queued device work?

Four signals per window, each with a positive control in the same window:

* an upload counter that needs no device tracing: a ``TorchDispatchMode`` counts the ``copy_`` / ``_to_copy``
  operations that move a non-empty CPU tensor to a CUDA device (pinned and pageable sources apart), and
  ``torch.tensor`` / ``torch.as_tensor`` with a CUDA ``device=`` are counted as pageable copies (their own copy
  runs below the dispatch modes).  Calibration: one pageable ``copy_``, one ``torch.tensor(..., device=)`` and one
  pinned ``copy_`` must count exactly (2 pageable, 1 pinned), else the census fails;
* the profiler's host-to-device copies by kind (``Memcpy HtoD (Pageable -> Device)`` / ``(Pinned -> Device)``),
  calibrated by the same three copies.  Where the profiler cannot trace the device (no CUPTI support) its counts are
  ``None`` (``prof_counted`` False) and the census says so; elsewhere they must equal the counter's;
* the caching allocator's device allocations and frees (``torch.cuda.memory_stats`` deltas, no tracing needed);
  :func:`allocator_counts` is their positive control (a fresh allocation must count);
* the synchronizing operations ``torch.cuda.set_sync_debug_mode("warn")`` reports, and a queue probe: a
  ``torch.cuda._sleep`` enqueued before the window must still run when the call returns (an event recorded right
  after the sleep, not the stream: work the call enqueues after its last wait may still run), and the host must
  return well before the sleep ends.
"""

from __future__ import annotations

import contextlib
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

SLEEP_MS = 1000.0  # queued device work of the queue probe
HOST_BOUND = 0.25  # the host returns within this fraction of the probe's sleep
PAGEABLE_KIND = "Memcpy HtoD (Pageable -> Device)"
PINNED_KIND = "Memcpy HtoD (Pinned -> Device)"

_CYCLES_PER_MS: dict[int, float] = {}
_ALLOC_COUNTS: dict[int, bool] = {}


def sleep_cycles(device: torch.device, ms: float) -> int:
    """``torch.cuda._sleep`` cycles for about ``ms`` milliseconds (calibrated once per device, before any probe:
    the calibration drains the current stream)."""
    rate = _CYCLES_PER_MS.get(device.index)
    if rate is None:
        n = 50_000_000
        torch.cuda.synchronize(device)
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        torch.cuda._sleep(n)
        b.record()
        b.synchronize()
        rate = _CYCLES_PER_MS[device.index] = n / max(a.elapsed_time(b), 1e-3)
    return int(rate * ms)


class _Mode(TorchDispatchMode):
    def __init__(self, counter: HostToDeviceCounter) -> None:
        super().__init__()
        self.counter = counter

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        c = self.counter
        if not c.inside_factory:
            if func is torch.ops.aten.copy_.default:
                dst, src = args[0], args[1]
                if (
                    isinstance(src, torch.Tensor)
                    and dst.device.type == "cuda"
                    and src.device.type == "cpu"
                    and src.numel()
                ):
                    c.note(src.is_pinned())
            elif func is torch.ops.aten._to_copy.default:
                src, dev = args[0], kwargs.get("device")
                if (
                    src.device.type == "cpu"
                    and dev is not None
                    and torch.device(dev).type == "cuda"
                    and src.numel()
                ):
                    c.note(src.is_pinned())
        return func(*args, **kwargs)


class HostToDeviceCounter:
    """Counts host-to-device copies (see the module docstring) while :meth:`active`; thread-safe.  ``mode()`` enters
    the dispatch half in the calling thread (dispatch modes are per thread); the factory half is process-wide."""

    def __init__(self) -> None:
        self.pageable = 0
        self.pinned = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def inside_factory(self) -> bool:
        return getattr(self._local, "factory", False)

    def note(self, pinned: bool) -> None:
        with self._lock:
            if pinned:
                self.pinned += 1
            else:
                self.pageable += 1

    def reset(self) -> None:
        with self._lock:
            self.pageable = self.pinned = 0

    def mode(self) -> _Mode:
        return _Mode(self)

    def _wrap(self, factory: Callable) -> Callable:
        def counted(*args, **kwargs):
            self._local.factory = True
            try:
                t = factory(*args, **kwargs)
            finally:
                self._local.factory = False
            data = args[0] if args else kwargs.get("data")
            on_device = isinstance(data, torch.Tensor) and data.device.type == "cuda"
            if (
                isinstance(t, torch.Tensor)
                and t.device.type == "cuda"
                and t.numel()
                and not on_device
            ):
                self.note(isinstance(data, torch.Tensor) and data.is_pinned())
            return t

        return counted

    @contextlib.contextmanager
    def active(self):
        """The factory wrappers (process-wide) and the dispatch mode of the calling thread."""
        saved = (torch.tensor, torch.as_tensor)
        torch.tensor = self._wrap(saved[0])  # type: ignore[assignment]
        torch.as_tensor = self._wrap(saved[1])  # type: ignore[assignment]
        try:
            with self.mode():
                yield self
        finally:
            torch.tensor, torch.as_tensor = saved  # type: ignore[assignment]


def _alloc_stats(device: torch.device) -> tuple[int, int, int]:
    s = torch.cuda.memory_stats(device)
    return (
        int(s.get("num_device_alloc", 0)),
        int(s.get("num_device_free", 0)),
        int(s.get("num_alloc_retries", 0)),
    )


def allocator_counts(device: torch.device) -> bool:
    """Positive control of the allocator counts (once per device): after ``empty_cache`` a fresh 256 MiB allocation
    must raise ``num_device_alloc`` and its release by ``empty_cache`` ``num_device_free``.  False under an allocator
    whose statistics do not count device allocations (for example ``backend:cudaMallocAsync``)."""
    ok = _ALLOC_COUNTS.get(device.index)
    if ok is None:
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        a0, f0, _ = _alloc_stats(device)
        t = torch.empty(256 << 20, dtype=torch.uint8, device=device)
        a1, _, _ = _alloc_stats(device)
        del t
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        _, f2, _ = _alloc_stats(device)
        ok = _ALLOC_COUNTS[device.index] = a1 > a0 and f2 > f0
    return ok


def census(
    fn: Callable[[], Any],
    device: torch.device,
    *,
    probe: bool = True,
    sleep_ms: float = SLEEP_MS,
    in_thread: bool = False,
    setup: Callable[[], None] | None = None,
) -> dict:
    """Run ``fn`` once inside the signals of the module docstring; ``in_thread`` runs it on a new thread (the host
    time is then the thread's start to join); ``setup`` runs after the opening synchronization and the calibration,
    right before the probe (for example work queued on a side stream that must still be queued when ``fn`` runs).  Returns ``pageable`` / ``pinned`` (the counter), ``prof_pageable`` /
    ``prof_pinned`` (the profiler; ``None`` without device tracing), ``device_alloc`` / ``device_free`` /
    ``alloc_retries`` (allocator deltas), ``sync_warn``, ``host_s``, ``busy`` and ``fn``'s result as ``out``."""
    torch.cuda.synchronize(device)
    cycles = sleep_cycles(device, sleep_ms) if probe else 0
    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    dst = torch.empty(64, dtype=torch.uint8, device=device)
    src = torch.ones(64, dtype=torch.uint8)
    src_pinned = src.pin_memory()
    slept = torch.cuda.Event()
    counter = HostToDeviceCounter()
    result: dict[str, Any] = {}

    def body() -> None:
        with counter.mode():
            result["out"] = fn()

    with torch.profiler.profile(activities=acts) as prof, counter.active():
        dst.copy_(src)
        torch.tensor([1, 2, 3], dtype=torch.uint8, device=device)
        dst.copy_(src_pinned, non_blocking=True)
        torch.cuda.synchronize(device)
        calibration = (counter.pageable, counter.pinned)
        if calibration != (2, 1):
            raise AssertionError(
                f"the host-to-device counter saw {calibration} of the calibration copies (2 pageable, 1 pinned)"
            )
        counter.reset()
        if setup is not None:
            setup()
        stats0 = _alloc_stats(device)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                if probe:
                    torch.cuda._sleep(cycles)
                    slept.record()
                t0 = time.perf_counter()
                if in_thread:
                    th = threading.Thread(target=body)
                    th.start()
                    th.join()
                    if "out" not in result:
                        raise RuntimeError("census: the thread of fn raised")
                else:
                    result["out"] = fn()
                host = time.perf_counter() - t0
            finally:
                torch.cuda.set_sync_debug_mode(0)
        busy = probe and not slept.query()
        stats1 = _alloc_stats(device)
        torch.cuda.synchronize(device)
    names = [e.name for e in prof.events()]
    prof_pageable = sum(PAGEABLE_KIND in n for n in names) - 2
    prof_pinned = sum(PINNED_KIND in n for n in names) - 1
    counted = prof_pageable >= 0 and prof_pinned >= 0
    if not counted:
        print(
            "CENSUS_PROFILER_UNAVAILABLE: the profiler traced no calibration copy on",
            device,
            "(the counter, allocator, sync-debug and probe signals still apply)",
        )
    return {
        "pageable": counter.pageable,
        "pinned": counter.pinned,
        "prof_counted": counted,
        "prof_pageable": prof_pageable if counted else None,
        "prof_pinned": prof_pinned if counted else None,
        "prof_malloc": sum(n == "cudaMalloc" for n in names) if counted else None,
        "device_alloc": stats1[0] - stats0[0],
        "device_free": stats1[1] - stats0[1],
        "alloc_retries": stats1[2] - stats0[2],
        "sync_warn": sum(
            "synchronizing CUDA operation" in str(w.message) for w in caught
        ),
        "host_s": host,
        "busy": busy,
        "out": result["out"],
    }


def counts_agree(c: dict) -> bool:
    """The profiler's copy counts equal the counter's (vacuously true without device tracing: ``prof_counted``)."""
    return not c["prof_counted"] or (
        c["prof_pageable"] == c["pageable"] and c["prof_pinned"] == c["pinned"]
    )


def nonblocking(c: dict, sleep_ms: float = SLEEP_MS) -> bool:
    """No pageable copy (counter and profiler), no synchronizing operation, the probe still ran when the call
    returned, and the host returned within :data:`HOST_BOUND` of the sleep."""
    return (
        c["pageable"] == 0
        and counts_agree(c)
        and c["sync_warn"] == 0
        and c["busy"]
        and c["host_s"] < HOST_BOUND * sleep_ms / 1e3
    )


def summary(c: dict) -> dict:
    """``c`` without its result (for reports)."""
    return {k: v for k, v in c.items() if k != "out"}
