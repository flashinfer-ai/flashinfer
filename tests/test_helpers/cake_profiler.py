# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""CUDA launch records of one call under ``torch.profiler``, robust to a session that loses its device records.

The Cake tests read launch structure (one kernel per call, no cast kernel, no device-to-host copy) from the CUDA
activity records of ``torch.profiler.profile``.  Some torch / CUPTI builds hand back a session with no device
records at all, or with its first records missing: in the flashinfer-ci nightlies of 2026-10-24..26 the GB200 and
GB300 jobs of the cu129 image (torch 2.13.0+cu129, CUPTI 12.9 on aarch64 nodes) lost the records of a profiled
Cake call in one to four sessions per night, while the same tests passed in the other images' jobs on the same
nodes and in every B200 job.  A missing record cannot be told from a kernel that was never launched, so every
capture here brackets the call with sentinel kernels (``torch.cuda._sleep``, the ``spin_kernel`` of ATen) whose
records prove the session captured the whole window.  A session that misses a sentinel is discarded and the
capture is retried; a process in which no session captures the sentinels skips the assertion with a reason that
starts with ``CUPTI-LOST-RECORDS`` instead of reporting a launch it cannot see.  A session that has device records
but none of the sentinel is a misconfigured helper (the sentinel kernel's name changed), reported as a failure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest
import torch

SENTINEL_KERNEL = (
    "spin_kernel"  # ATen's at::cuda::sleep kernel, launched by torch.cuda._sleep
)
ATTEMPTS = 4
LOST_RECORDS = "CUPTI-LOST-RECORDS"
_COPY_PREFIXES = ("Memcpy", "Memset")


def launch_sentinel() -> None:
    """Launch the sentinel kernel (a one-cycle device spin)."""
    torch.cuda._sleep(1)


def is_sentinel(name: str) -> bool:
    return SENTINEL_KERNEL in name


def cuda_event_names(profile: torch.profiler.profile) -> list[str]:
    """Names of the session's CUDA activity records in start order."""
    events = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    events.sort(key=lambda event: event.time_range.start)
    return [event.name for event in events]


def between_sentinels(names: list[str]) -> list[str] | None:
    """The device records of a window ``sentinel, sentinel, <call>, sentinel``.

    ``None`` when the session lost a sentinel (no record, a leading record or the trailing record missing), so
    the caller profiles again.  A session with device records but no sentinel among them is not a loss: the
    sentinel kernel's name changed and the helper must be fixed, so that raises.
    """
    if not names:
        return None
    marks = [i for i, name in enumerate(names) if is_sentinel(name)]
    if not marks:
        raise AssertionError(
            f"no CUDA record of the sentinel kernel {SENTINEL_KERNEL!r} among {names}: "
            "the sentinel's name changed; fix tests.test_helpers.cake_profiler"
        )
    if len(marks) < 2 or marks[-1] != len(names) - 1:
        return None
    leading = marks[:-1]
    if leading != list(range(len(leading))):
        return None  # a record before the window's first sentinel: not the window that was launched
    return names[leading[-1] + 1 : marks[-1]]


def skip_lost_records(attempts: int) -> None:
    pytest.skip(
        f"{LOST_RECORDS}: torch.profiler returned no CUDA record of the sentinel kernel in {attempts} "
        "sessions, so this process's launch structure cannot be read"
    )


def cuda_kernel_names(
    fn: Callable[[], object],
    *,
    exclude: Iterable[str] = _COPY_PREFIXES,
    attempts: int = ATTEMPTS,
) -> list[str]:
    """Names of the CUDA kernels one call of ``fn`` launches, in launch order.

    Records whose names start with ``exclude`` (memcpy / memset activities by default) are dropped.  ``fn`` runs
    once per session: the profiler's warm-up step (CUDA activity collection starts asynchronously once the
    profiler is entered) holds only a sentinel, and the measured call sits behind two leading sentinels and
    before a trailing one.  A session that misses the leading or the trailing sentinel is retried, and after
    ``attempts`` such sessions the test is skipped (``LOST_RECORDS``).
    """
    exclude = tuple(exclude)
    for _ in range(attempts):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
        ) as profile:
            launch_sentinel()
            torch.cuda.synchronize()
            profile.step()
            launch_sentinel()
            launch_sentinel()
            fn()
            launch_sentinel()
            torch.cuda.synchronize()
        inner = between_sentinels(cuda_event_names(profile))
        if inner is not None:
            return [name for name in inner if not name.startswith(exclude)]
    skip_lost_records(attempts)
    raise AssertionError("unreachable")  # pragma: no cover
