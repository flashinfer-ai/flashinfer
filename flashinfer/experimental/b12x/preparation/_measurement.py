"""Representative activation-producing races owned by preparation."""

from __future__ import annotations

import ctypes
import gc
import math
import os
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field

from b12x._lib.compile_plan import (
    ProgramKey,
    forbid_lowering,
    observe_programs,
    record_program,
)
from .types import _prime, _close_all

# Surviving candidates complete three scored rounds within the leader margin.
SURVIVOR_ROUNDS = 3
ELIMINATION_MARGIN = 1.10
DEFAULT_SAMPLES = 2
ROUND_BUDGET_US = 256.0


class _ParentCompilations:
    """Scoped actual-compilation counters and program keys, not launch tracing."""

    def __init__(self, *, cache_only):
        self.cache_only = cache_only
        self.cute = self.triton = 0

    def __enter__(self):
        from triton import knobs
        from b12x._lib import compiler

        self.compiler, self.knobs = compiler, knobs
        self.before = int(compiler.compile_cache_info()["compile_misses"])
        self.original_compile = compiler._call_cute_compile
        self.previous_listener = knobs.compilation.listener
        self.observation = observe_programs()
        self.programs = self.observation.__enter__()

        def listener(*args, **kwargs):
            metadata = kwargs.get("metadata", args[1] if len(args) > 1 else {})
            record_program(
                ProgramKey("triton", metadata["hash"], metadata.get("name", ""))
            )
            if not kwargs.get("cache_hit", args[4] if len(args) > 4 else False):
                self.triton += 1
            if self.previous_listener is not None:
                self.previous_listener(*args, **kwargs)

        def reject_compile(*_args, **_kwargs):
            name = getattr(_kwargs.get("compile_spec"), "kernel_id", "unknown")
            raise RuntimeError(
                f"no-compilation phase encountered an unplanned CuTe program: {name} {_kwargs.get('cache_key')}"
            )

        knobs.compilation.listener = listener
        if self.cache_only:
            compiler._call_cute_compile = reject_compile
        return self

    def check(self):
        self.cute = (
            int(self.compiler.compile_cache_info()["compile_misses"]) - self.before
        )
        if self.cache_only and (self.cute or self.triton):
            raise RuntimeError(
                f"no-compilation phase compiled CuTe={self.cute}, Triton={self.triton}"
            )

    def __exit__(self, kind, value, traceback):
        self.cute = (
            int(self.compiler.compile_cache_info()["compile_misses"]) - self.before
        )
        self.knobs.compilation.listener = self.previous_listener
        if self.cache_only:
            self.compiler._call_cute_compile = self.original_compile
        self.observation.__exit__(kind, value, traceback)


@contextmanager
def no_compilation():
    """Permit already-built object loads, but no new CuTe/Triton compilation."""
    with _ParentCompilations(cache_only=True) as observed, forbid_lowering():
        yield observed
        observed.check()


class _StreamGate:
    """Hold queued device work until the host finishes submitting a sample."""

    def __init__(self):
        if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
            raise RuntimeError(
                "stream-gated autotuning requires CUDA_LAUNCH_BLOCKING to be disabled"
            )
        from cuda.bindings import driver

        self.driver = driver
        self.pointer = self._check(
            driver.cuMemHostAlloc(4, driver.CU_MEMHOSTALLOC_DEVICEMAP)
        )
        try:
            self.device_pointer = self._check(
                driver.cuMemHostGetDevicePointer(self.pointer, 0)
            )
            self.flag = ctypes.c_uint32.from_address(int(self.pointer))
            self.flag.value = 0
        except BaseException:
            driver.cuMemFreeHost(self.pointer)
            raise
        self.sequence = 0
        self.streams = {}

    def _check(self, result):
        status, *values = result
        if status != self.driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA autotuning stream gate failed: {status}")
        return values[0] if values else None

    @contextmanager
    def hold(self, stream):
        self.streams[stream.cuda_stream] = stream
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        target = self.sequence
        self._check(
            self.driver.cuStreamWaitValue32(
                stream.cuda_stream,
                self.device_pointer,
                target,
                int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ),
            )
        )
        try:
            yield
        finally:
            # A later release must also satisfy an earlier, still queued wait.
            self.flag.value = target

    def close(self):
        if self.pointer is None:
            return
        self.flag.value = self.sequence
        for stream in self.streams.values():
            stream.synchronize()
        self._check(self.driver.cuMemFreeHost(self.pointer))
        self.pointer = None
        self.streams.clear()


@contextmanager
def _collection_paused():
    """Keep automatic garbage collection out of a gated sample.

    A collection can finalize an unreferenced CuTe module, whose
    ``cudaLibraryUnload`` waits for queued device work. Work behind the stream
    gate waits for this thread to release it, so neither would proceed.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class _TimedCall:
    def __init__(self, call, eviction, samples, gate):
        import torch

        self.call, self.eviction = call, eviction
        self.gate = gate
        self.producers = call.benchmark_producers or (call.produce,)
        self.events = tuple(
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(samples * len(self.producers))
        )
        for pair in self.events:
            for event in pair:
                event.record()

    def replay(self):
        import torch
        from .types import call_scope

        stream = torch.cuda.current_stream()
        with call_scope():
            for index, (start, end) in enumerate(self.events):
                self.eviction()
                if self.call.reset is not None:
                    self.call.reset()
                self.producers[index % len(self.producers)]()
                with _collection_paused(), self.gate.hold(stream):
                    start.record(stream)
                    self.call.invoke()
                    end.record(stream)

    def samples(self):
        return tuple(start.elapsed_time(end) * 1000.0 for start, end in self.events)

    def close(self):
        self.call = self.eviction = self.gate = None
        self.events = ()
        self.producers = ()


@dataclass
class PreparedRace:
    timers: tuple[_TimedCall, ...]
    eviction: object
    sample_count: int
    completed_rounds: int = 0
    planned_rounds: int = 0
    active_count: int = 0
    latest_round_us: tuple[float, ...] = ()
    gate: object = None
    _closed: bool = field(default=False, init=False)

    def close(self):
        if self._closed:
            return
        self._closed = True
        closers = [] if self.gate is None else [self.gate.close]
        closers.extend(timer.close for timer in self.timers)
        _close_all(closers)


@dataclass(frozen=True)
class RaceMeasurements:
    latencies_us: tuple[float, ...]
    overlapped_samples: int


def _l2_flush_fn(device: object, *, enabled: bool):
    if not enabled:
        return None
    import torch

    properties = torch.cuda.get_device_properties(device)
    flush_bytes = max(2 * int(properties.L2_cache_size), 64 << 20)
    buffer = torch.ones(
        (flush_bytes + 3) // 4,
        dtype=torch.float32,
        device=device,
    )
    reduction = torch.empty((), dtype=torch.float32, device=device)

    def flush() -> None:
        torch.sum(buffer, dim=0, out=reduction)

    return flush


def prepare_race_steps(
    calls,
    *,
    device_ordinal,
    samples=DEFAULT_SAMPLES,
    primed=False,
    eviction=None,
):
    """Prepare event pairs for asynchronous calls on the current CUDA stream."""
    import torch

    if not calls or type(samples) is not int or samples <= 0:
        raise ValueError("a race requires candidates and positive samples")
    if any(call.produce is None for call in calls):
        raise ValueError("candidate races require an activation-producing context")
    workload_counts = {len(call.benchmark_producers) or 1 for call in calls}
    if len(workload_counts) != 1:
        raise ValueError(
            "candidate races require the same workload count for every candidate"
        )
    sample_count = samples * workload_counts.pop()
    timers = []
    gate = None
    completed = False
    try:
        with torch.cuda.device(device_ordinal), no_compilation():
            if eviction is None:
                eviction = _l2_flush_fn(
                    torch.device("cuda", device_ordinal), enabled=True
                )
            try:
                eviction()
                if not primed:
                    for call in calls:
                        _prime(call)
            finally:
                torch.cuda.current_stream(device_ordinal).synchronize()
            gate = _StreamGate()
        yield
        for call in calls:
            with torch.cuda.device(device_ordinal), no_compilation():
                timers.append(_TimedCall(call, eviction, samples, gate))
            yield
        completed = True
        return PreparedRace(tuple(timers), eviction, sample_count, gate=gate)
    finally:
        if not completed:
            PreparedRace(tuple(timers), None, sample_count, gate=gate).close()


def _replay_timers(timers, *, device_ordinal, sample_count=0, compilation_active=None):
    import torch

    overlaps = 0
    with torch.cuda.device(device_ordinal), no_compilation():
        try:
            for timer in timers:
                if compilation_active is not None and compilation_active():
                    overlaps += sample_count
                timer.replay()
        finally:
            torch.cuda.current_stream(device_ordinal).synchronize()
    return overlaps


def measure_race_steps(
    prepared,
    *,
    device_ordinal,
    rounds=7,
    compilation_active=None,
    eliminate=False,
    champion=False,
    sample_observer=None,
    adaptive_repeats=True,
):
    """Balanced comparison; cancellation discards this generator's result.

    A selection race sets ``eliminate``: a timer whose best round so far trails
    the leader by more than ELIMINATION_MARGIN stops being re-timed and keeps the
    median of the rounds it completed, survivors run at most SURVIVOR_ROUNDS
    rounds, and ``champion`` exempts timer 0, which carries the previous batch's
    winner. Left unset, every timer completes every round.
    """
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("race rounds must be positive")
    if eliminate:
        rounds = min(rounds, SURVIVOR_ROUNDS)
    prepared.planned_rounds = rounds
    prepared.active_count = len(prepared.timers)
    values = [[] for _ in prepared.timers]
    active = list(range(len(prepared.timers)))
    overlaps = 0
    repeats = [1] * len(prepared.timers)
    for turn in range(rounds):
        order = list(active)
        if turn % 2:
            order.reverse()
        offset = (turn // 2) % len(order)
        order = order[offset:] + order[:offset]
        totals = [0.0] * len(prepared.timers)
        repetition = 0
        while repetition < max(repeats[index] for index in order):
            indices = tuple(
                index
                for index in (order if repetition % 2 == 0 else reversed(order))
                if repetition < repeats[index]
            )
            overlaps += _replay_timers(
                tuple(prepared.timers[index] for index in indices),
                device_ordinal=device_ordinal,
                sample_count=prepared.sample_count,
                compilation_active=compilation_active,
            )
            for index in indices:
                samples = prepared.timers[index].samples()
                if sample_observer is not None:
                    sample_observer(index, turn, repetition, samples)
                latency = statistics.fmean(samples)
                if not math.isfinite(latency) or latency <= 0:
                    raise RuntimeError("candidate race produced an invalid latency")
                totals[index] += latency
                if adaptive_repeats and turn == 0 and repetition == 0:
                    # The first scored replay also sizes the remaining work.
                    repeats[index] = max(
                        1,
                        math.ceil(ROUND_BUDGET_US / (prepared.sample_count * latency)),
                    )
            repetition += 1
            yield
        for index in order:
            values[index].append(totals[index] / repeats[index])
        # Timers that sat out this round report no latency, so a reader of the
        # round feed does not mistake a stale entry for a fresh measurement.
        timed = frozenset(order)
        prepared.latest_round_us = tuple(
            series[-1] if index in timed else math.nan
            for index, series in enumerate(values)
        )
        prepared.completed_rounds = turn + 1
        if eliminate:
            best = [min(series) for series in values]
            leader = min(best[index] for index in active)
            # A timer outside the leader's margin stops being re-timed and keeps
            # the median of the rounds it completed; the champion at position 0
            # is re-timed against every batch.
            active = [
                index
                for index in active
                if best[index] <= ELIMINATION_MARGIN * leader
                or (champion and index == 0)
            ]
            prepared.active_count = len(active)
    latencies = tuple(statistics.median(series) for series in values)
    if any(not math.isfinite(value) or value <= 0 for value in latencies):
        raise RuntimeError("candidate race produced an invalid latency")
    return RaceMeasurements(latencies, overlaps)


def _consume(steps):
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return finished.value


def _prepare_race(calls, *, device_ordinal, samples=DEFAULT_SAMPLES, primed=False):
    return _consume(
        prepare_race_steps(
            calls,
            device_ordinal=device_ordinal,
            samples=samples,
            primed=primed,
        )
    )


def _measure_race(prepared, *, device_ordinal, rounds=7, compilation_active=None):
    return _consume(
        measure_race_steps(
            prepared,
            device_ordinal=device_ordinal,
            rounds=rounds,
            compilation_active=compilation_active,
        )
    )
