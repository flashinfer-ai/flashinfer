"""Prepared-call benchmarks using the preparation engine's GPU sampler."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import statistics

from b12x.preparation import PreparedCall
from b12x.preparation._measurement import (
    _consume,
    measure_race_steps,
    no_compilation,
    prepare_race_steps,
)
from b12x.preparation.types import _prime

METHOD = "stream_gated_events_v1"


def _nothing():
    pass


@dataclass(frozen=True)
class Sample:
    name: str
    round: int
    repetition: int
    samples_us: tuple[float, ...]


@dataclass(frozen=True)
class Measurement:
    latencies_us: dict[str, float]
    samples: tuple[Sample, ...]
    rounds: int
    samples_per_workload: int
    method: str = METHOD

    def raw_samples(self, name: str) -> list[float]:
        return [
            value
            for row in self.samples
            if row.name == name
            for value in row.samples_us
        ]


def measure_calls(
    calls: Mapping[str, PreparedCall],
    *,
    device=None,
    warmup=3,
    samples=8,
    rounds=9,
    eviction: Callable[[], None] | None = None,
    checks: Mapping[str, Callable[[PreparedCall], None]] | None = None,
    collective=False,
    adaptive_repeats=False,
) -> Measurement:
    """Measure every arm in balanced rounds with compilation excluded.

    Calls and their tensor owners remain owned by the caller. Correctness checks
    run after priming and before measurement; exceptions prevent a result.
    """
    import torch

    if not calls or any(not name for name in calls):
        raise ValueError("benchmarks require named prepared calls")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if (
        type(samples) is not int
        or samples <= 0
        or type(rounds) is not int
        or rounds <= 0
    ):
        raise ValueError("samples and rounds must be positive integers")
    if checks is not None and set(checks) != set(calls):
        raise ValueError("correctness checks must cover every comparison arm")
    if collective and adaptive_repeats:
        raise ValueError("collectives require identical launch counts on every rank")
    names, values = tuple(calls), tuple(calls.values())
    ordinal = (
        device
        if isinstance(device, int)
        else (
            torch.cuda.current_device()
            if device is None
            else torch.device(device).index
        )
    )
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    with torch.cuda.device(ordinal):
        for name, call in calls.items():
            for _ in range(warmup):
                _prime(call)
            if checks is not None:
                with no_compilation():
                    _prime(call)
                    checks[name](call)
        torch.cuda.synchronize(ordinal)
        prepared = _consume(
            prepare_race_steps(
                values,
                device_ordinal=ordinal,
                samples=samples,
                primed=True,
                eviction=eviction,
            )
        )
        records = []

        def observe(index, turn, repetition, timings):
            if any(not math.isfinite(value) or value <= 0 for value in timings):
                raise RuntimeError("benchmark produced an invalid raw sample")
            records.append(Sample(names[index], turn, repetition, tuple(timings)))

        try:
            measured = _consume(
                measure_race_steps(
                    prepared,
                    device_ordinal=ordinal,
                    rounds=rounds,
                    eliminate=False,
                    sample_observer=observe,
                    adaptive_repeats=adaptive_repeats,
                )
            )
            if measured.overlapped_samples:
                raise RuntimeError("benchmark timing overlapped compilation")
            result = Measurement(
                dict(zip(names, measured.latencies_us, strict=True)),
                tuple(records),
                rounds,
                samples,
            )
            return result
        finally:
            prepared.close()


def measure_call(
    run, *, warmup=3, samples=8, rounds=9, produce=None, reset=None, eviction=None
):
    """Measure a pre-bound workload with explicit preparation outside timing."""
    run = getattr(run, "_b12x_benchmark_call", run)
    owner = getattr(run, "__self__", None)
    if owner is not None and hasattr(owner, "_b12x_benchmark_call"):
        run = owner._b12x_benchmark_call
    call = PreparedCall(
        run=run, produce=produce or _nothing, reset=reset, owners=(run,)
    )
    return measure_calls(
        {"workload": call},
        warmup=warmup,
        samples=samples,
        rounds=rounds,
        eviction=eviction,
    )


def samples_ms(run, *, warmup, iters, l2_flush=None, produce=None, reset=None):
    measured = measure_call(
        run,
        warmup=warmup,
        samples=iters,
        produce=produce,
        reset=reset,
        eviction=l2_flush or _nothing,
    )
    return [value / 1000.0 for value in measured.raw_samples("workload")]


def median_ms(run, *, warmup, iters, l2_flush=None):
    return statistics.median(
        samples_ms(run, warmup=warmup, iters=iters, l2_flush=l2_flush)
    )


def transaction_samples(run, *, samples, warmup=0, prepare=None, l2_flush=None):
    """Report kernel and complete-step scopes using the same gated sampler."""
    kernel = PreparedCall(run=run, produce=prepare or _nothing)
    calls = {"replay_us": kernel}
    if prepare is not None:

        def step():
            prepare()
            return run()

        calls["metadata_us"] = PreparedCall(run=prepare, produce=_nothing)
        calls["step_us"] = PreparedCall(run=step, produce=_nothing)
    result = measure_calls(
        calls, warmup=warmup, samples=samples, eviction=l2_flush or _nothing
    )
    raw = {name: result.raw_samples(name) for name in calls}
    if prepare is None:
        raw["metadata_us"] = [0.0] * len(raw["replay_us"])
        raw["step_us"] = raw["replay_us"]
    return raw


def graph_samples(graph, *, replays, prepare=None, l2_flush=None):
    """Measure the bound invocation retained alongside a qualification graph."""
    if not hasattr(graph, "_b12x_benchmark_call"):
        raise ValueError("qualification graph must retain its uncaptured invocation")
    return transaction_samples(
        graph._b12x_benchmark_call, samples=replays, prepare=prepare, l2_flush=l2_flush
    )
