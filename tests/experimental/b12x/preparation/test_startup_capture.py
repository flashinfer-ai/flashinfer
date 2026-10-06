"""Prepared launchers remain valid after compiler caches release their copies."""

import gc
import statistics
import time

import pytest
import torch

from b12x._lib import compiler
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x


def test_graph_replay_survives_compiler_cache_eviction(monkeypatch):
    from b12x.gemm import bf16_gemv

    device = require_b12x()
    source = torch.randn(2, 2048, device=device, dtype=torch.bfloat16)
    weight = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
    plan = bf16_gemv.plan(bf16_gemv.query_from_call(source, weight))
    request = plan.request(
        name="gemv",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, weight)
        ),
    )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((request,))
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                output = bf16_gemv.mm(source, weight, plan=plan)

            compiler.clear_compile_cache()
            gc.collect()

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "prepared graph replay reached compiler or loader"
                )

            monkeypatch.setattr(compiler, "compile", forbidden)
            monkeypatch.setattr(
                compiler, "_load_cute_compile_from_disk", forbidden
            )
            changed = torch.randn_like(source)
            source.copy_(changed)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)

            expected = changed.float() @ weight.float().T
            assert torch.isfinite(output).all()
            assert torch.count_nonzero(output) > 0
            torch.testing.assert_close(
                output, expected.bfloat16(), rtol=2e-2, atol=2e-2
            )
        finally:
            graph.reset()


def test_candidate_timing_preserves_outputs_without_capture_or_allocator_flushes(monkeypatch):
    from b12x.gemm import bf16_gemv
    from b12x.preparation import _measurement

    device = require_b12x()
    weight = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
    sources = [torch.ones(rows, 2048, device=device, dtype=torch.bfloat16) for rows in (2, 3)]
    plans = [bf16_gemv.plan(bf16_gemv.query_from_call(source, weight)) for source in sources]
    requests = [
        plan.request(
            name=f"candidate-{index}",
            prepare_call=lambda state, source=source: PreparedCall(
                run=lambda: state.run(source, weight),
            ),
        )
        for index, (source, plan) in enumerate(zip(sources, plans, strict=True))
    ]
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(requests)
        outputs = [torch.empty(source.shape[0], weight.shape[0], device=device, dtype=source.dtype)
                   for source in sources]
        calls = [
            PreparedCall(
                run=lambda source=source, plan=plan, output=output: bf16_gemv.mm(source, weight, plan=plan, out=output),
                produce=lambda source=source: source.add_(0.125),
                owners=(source, weight),
            )
            for source, plan, output in zip(sources, plans, outputs, strict=True)
        ]

        def forbidden(*_args, **_kwargs):
            raise AssertionError("candidate timing captured a graph or flushed the process allocator")

        with monkeypatch.context() as guards:
            guards.setattr(torch.cuda, "empty_cache", forbidden)
            guards.setattr(torch._C, "_host_emptyCache", forbidden)
            guards.setattr(torch.cuda, "CUDAGraph", forbidden)
            race = _measurement._prepare_race(calls, device_ordinal=torch.cuda.current_device(), samples=8)
        try:
            pointers = [call.output.data_ptr() for call in calls]
            assert len(set(pointers)) == len(calls)
            for initial in (1.0, 3.0):
                for source, call in zip(sources, calls, strict=True):
                    source.fill_(initial)
                    with torch.inference_mode():
                        call.output.fill_(float("nan"))
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_allocated(device)
                for timer in race.timers:
                    timer.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                assert [call.output.data_ptr() for call in calls] == pointers
                for source, call, timer in zip(sources, calls, race.timers, strict=True):
                    torch.testing.assert_close(source, torch.full_like(source, initial + 1.0))
                    expected = source.float() @ weight.float().T
                    assert torch.isfinite(call.output).all()
                    assert torch.count_nonzero(call.output) > 0
                    torch.testing.assert_close(call.output, expected.bfloat16(), rtol=2e-2, atol=2e-2)
                    assert all(value > 0 for value in timer.samples())
        finally:
            race.close()


def test_candidate_samples_visit_the_complete_workload_mix():
    from b12x.preparation import _measurement

    device = require_b12x()
    source = torch.empty(1, device=device)
    output = torch.empty_like(source)
    observed = torch.empty(6, device=device)
    recording = False
    index = 0

    def run():
        nonlocal index
        output.copy_(source)
        if recording:
            observed[index:index + 1].copy_(output)
            index += 1
        return output

    producers = tuple(lambda value=value: source.fill_(value) for value in (1, 2, 3))
    call = PreparedCall(
        run=run, produce=producers[0], benchmark_producers=producers,
        reset=lambda: source.fill_(float("nan")),
    )
    race = _measurement._prepare_race(
        [call], device_ordinal=torch.cuda.current_device(), samples=2,
    )
    try:
        recording = True
        race.timers[0].replay()
        torch.cuda.synchronize(device)
        assert race.sample_count == 6
        assert len(race.timers[0].samples()) == 6
        torch.testing.assert_close(observed, torch.tensor([1, 2, 3, 1, 2, 3], device=device).float())
    finally:
        race.close()


def test_candidate_races_reject_incomparable_workload_counts():
    from b12x.preparation import _measurement

    def producer():
        pass
    calls = [
        PreparedCall(run=lambda: None, produce=producer),
        PreparedCall(run=lambda: None, produce=producer, benchmark_producers=(producer,) * 3),
    ]
    with pytest.raises(ValueError, match="same workload count"):
        _measurement._prepare_race(calls, device_ordinal=0)


def test_retired_candidate_timers_do_not_accumulate_or_release_live_graphs(monkeypatch):
    from b12x.preparation import _measurement

    device = require_b12x()
    ordinal = torch.cuda.current_device()
    source = torch.ones(1024, 1024, device=device)
    live_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(live_graph):
        live_output = source + 11

    # Keep a reusable default-pool allocation larger than candidate temporaries.
    cached = torch.empty(16 << 20, device=device)
    cached_pointer = cached.data_ptr()
    del cached

    def forbidden(*_args, **_kwargs):
        raise AssertionError("candidate cleanup flushed the process allocator")

    reserved = []
    try:
        with monkeypatch.context() as guards:
            guards.setattr(torch.cuda, "empty_cache", forbidden)
            guards.setattr(torch._C, "_host_emptyCache", forbidden)
            for batch in range(8):
                calls = [PreparedCall(run=source.clone, produce=lambda: source.add_(0.125))
                         for _ in range(2)]
                race = _measurement._prepare_race(calls, device_ordinal=ordinal, samples=8)
                try:
                    for timer in race.timers:
                        timer.replay()
                    torch.cuda.synchronize(device)
                    for call in calls:
                        assert torch.isfinite(call.output).all()
                        call.output = None
                finally:
                    race.close()
                reserved.append(torch.cuda.memory_reserved(device))
                del race, calls
                source.fill_(batch)
                live_graph.replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(live_output, torch.full_like(source, batch + 11))

            cached = torch.empty(16 << 20, device=device)
            assert cached.data_ptr() == cached_pointer
            assert len(set(reserved[2:])) == 1, reserved
    finally:
        live_graph.reset()
        del live_output


def test_candidate_timing_preserves_carried_outputs_and_bounds_residency(monkeypatch):
    from b12x.preparation import _measurement
    from b12x.preparation.types import _prime

    device = require_b12x()
    ordinal = torch.cuda.current_device()
    source = torch.ones(1024, 1024, device=device)
    reserved = []
    champion = None
    live_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(live_graph):
        live_output = source + 11

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a candidate batch evicted allocator storage")

    try:
        with monkeypatch.context() as guards:
            guards.setattr(torch.cuda, "empty_cache", forbidden)
            guards.setattr(torch._C, "_host_emptyCache", forbidden)
            guards.setattr(_measurement, "_prime", forbidden)
            for batch in range(8):
                calls = [] if champion is None else [champion]
                while len(calls) < 2:
                    output = torch.empty_like(source)
                    call = PreparedCall(
                        run=lambda output=output: output.copy_(source),
                        produce=lambda: source.add_(0.125),
                    )
                    _prime(call)
                    calls.append(call)
                race = _measurement._prepare_race(
                    calls, device_ordinal=ordinal, samples=2, primed=True,
                )
                pointers = [call.output.data_ptr() for call in calls]
                assert len(set(pointers)) == 2
                source.fill_(batch)
                with torch.inference_mode():
                    for call in calls:
                        call.output.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated(device)
                order = (0, 1) if batch % 2 == 0 else (1, 0)
                for index in order:
                    race.timers[index].replay()
                torch.cuda.current_stream(device).synchronize()
                assert torch.cuda.memory_allocated(device) == allocated
                assert [call.output.data_ptr() for call in calls] == pointers
                for position, index in enumerate(order):
                    torch.testing.assert_close(calls[index].output, torch.full_like(source, batch + 0.25 * (position + 1)))
                race.close()
                champion = calls[batch % 2]
                calls[1 - batch % 2].output = None
                del race, calls, call
                reserved.append(torch.cuda.memory_reserved(device))
                source.fill_(batch)
                live_graph.replay()
                torch.cuda.current_stream(device).synchronize()
                torch.testing.assert_close(live_output, torch.full_like(source, batch + 11))
            assert len(set(reserved[3:])) == 1, reserved
    finally:
        if champion is not None:
            champion.output = None
        live_graph.reset()
        del live_output


@pytest.mark.parametrize("capture_safe", [False, True])
def test_candidate_events_exclude_python_gaps_without_capture(monkeypatch, capture_safe):
    from b12x.preparation import _measurement

    device = require_b12x()
    output = torch.empty(1024, device=device)

    def run():
        output.mul_(2)
        time.sleep(0.02)
        output.add_(3)

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate timing attempted CUDA graph capture")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", forbidden)
    call = PreparedCall(run=run, produce=lambda: output.fill_(1), capture_safe=capture_safe)
    race = _measurement._prepare_race([call], device_ordinal=torch.cuda.current_device(), samples=4)
    try:
        _measurement._replay_timers(race.timers, device_ordinal=torch.cuda.current_device())
        torch.testing.assert_close(output, torch.full_like(output, 5))
        assert 0 < statistics.median(race.timers[0].samples()) < 5000
    finally:
        race.close()


def test_stream_gate_releases_queued_work_when_a_call_raises():
    from b12x.preparation._measurement import _StreamGate

    device = require_b12x()
    output = torch.zeros(1, device=device)
    output.add_(1)
    output.zero_()
    torch.cuda.synchronize(device)
    gate = _StreamGate()
    stream = torch.cuda.current_stream()
    try:
        torch.cuda._sleep(100_000_000)
        with gate.hold(stream):
            output.add_(1)
        with pytest.raises(RuntimeError, match="launch failed"), gate.hold(stream):
            output.add_(2)
            raise RuntimeError("launch failed")
        gate.close()
        torch.testing.assert_close(output, torch.full_like(output, 3))
    finally:
        gate.close()
