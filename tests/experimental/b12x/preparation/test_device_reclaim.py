"""A finished preparation reclaims the device memory its losing candidates used."""
import weakref

import pytest

from b12x._lib import program_cache
from b12x.preparation import PreparedCall
from ..conftest import require_b12x
from .test_defaults import contract
from .test_session import _deterministic_timer, request, session


def test_race_reclaims_device_memory_once_after_every_trial_closes(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    trial_closed, calls, reclaimed = [], [], []

    def benchmark(state):
        return PreparedCall(
            run=lambda: state.value, produce=lambda: None,
            close=lambda: trial_closed.append(state.value),
        )

    def reclaim(keep, *, stack_limit):
        reclaimed.append((frozenset(keep), stack_limit, sorted(trial_closed)))

    monkeypatch.setattr(program_cache, "reclaim_device_memory", reclaim)
    with session(tmp_path) as engine:
        result = engine.prepare((request(name="first", tuning=contract(), calls=calls, benchmark=benchmark),))
        assert result.benchmarked_candidates == 3
        kept = frozenset(result.plans["first"].prepared.programs)
    # A host device has no stack limit; every trial closed before the reclaim ran.
    assert reclaimed == [(kept, None, [3, 6, 12])]
    with session(tmp_path, autotune=False) as engine:
        result = engine.prepare((request(name="cached", tuning=contract(), calls=calls),))
        assert result.selections["cached"].source == "cached"
        assert result.benchmarked_candidates == 0
    # A preparation without a race launched no losing candidate.
    assert len(reclaimed) == 1


def test_host_reclaim_collects_cycles_without_touching_a_device():
    class Node:
        pass

    node = Node()
    node.cycle = node
    ref = weakref.ref(node)
    del node
    program_cache.reclaim_device_memory(frozenset(), stack_limit=None)
    assert ref() is None


def test_gpu_reclaim_restores_the_prior_stack_limit_and_keeps_prepared_plans_runnable():
    device = require_b12x()
    import torch
    from cuda.bindings import runtime
    from b12x.norm import hyperconnection as hc
    from b12x.norm.hyperconnection import _impl
    from b12x.preparation import FrozenMapping, PreparationSession

    source = torch.randn((1, 12), device=device, dtype=torch.bfloat16)
    output = torch.empty((1, 6), device=device, dtype=torch.bfloat16)
    declaration = hc.plan(
        hc.Caps(device=device, max_tokens=1, hidden_size=6),
        invocation=FrozenMapping({"operation": "swiglu", "limit": 2.0}),
    )

    def call(state):
        return PreparedCall(run=lambda: _impl.run_swiglu_impl(source, limit=2.0, out=output, plan=state))

    with PreparationSession(device=device, autotune=False, compile_workers=0) as engine:
        engine.prepare((declaration.request(name="swiglu", prepare_call=call),))
        before = program_cache.stack_limit_bytes()
        expected = output.clone()
        program_cache.reclaim_device_memory(frozenset(declaration.prepared.programs), stack_limit=before)
        assert program_cache.stack_limit_bytes() == before
        assert runtime.cudaDeviceSetLimit(runtime.cudaLimit.cudaLimitStackSize, before + 512)[0] == 0
        program_cache.reclaim_device_memory(frozenset(declaration.prepared.programs), stack_limit=before)
        assert program_cache.stack_limit_bytes() == before
        output.fill_(float("nan"))
        hc.run_swiglu(source, limit=2.0, out=output, plan=declaration)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_allocator_counter_tracks_live_storage_after_graph_pool_release():
    device = require_b12x()
    import torch
    from b12x.preparation import PreparationSession

    with PreparationSession(device=device, autotune=False) as engine:
        baseline = engine._allocated()
        storage = torch.empty((1024, 1024), device=device, dtype=torch.float32)
        assert engine._allocated() - baseline == storage.numel() * storage.element_size()
        for _ in range(512):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                storage.fill_(1)
            graph.replay()
            graph.reset()
            del graph
        torch.cuda.synchronize(device)
        assert engine._allocated() == torch.cuda.memory_allocated(device)
        assert engine._allocated() - baseline == storage.numel() * storage.element_size()
        del storage
        torch.cuda.synchronize(device)
        assert engine._allocated() == baseline
