"""Serial warmup uses native preparation hooks without search or compile planning."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x._lib.compile_plan import ProgramKey, record_program
from b12x.preparation import DetectedDevice, MemoryRequirements, Plan, PreparedCall
from b12x.preparation.tuning import TuningContract
from .test_defaults import Config, Query, contract
from .test_session import session, request


def forbidden(*_args, **_kwargs):
    raise AssertionError('warmup entered autotuning infrastructure')


@pytest.mark.parametrize('disable', ['session', 'job', 'environment', 'cancel'])
def test_warmup_uses_only_default_hooks(tmp_path, monkeypatch, disable):
    monkeypatch.setattr(TuningContract, 'parameter_space', forbidden)
    monkeypatch.setattr(TuningContract, 'iterate', forbidden)
    monkeypatch.delenv('B12X_AUTOTUNE', raising=False)
    if disable == 'environment':
        monkeypatch.setenv('B12X_AUTOTUNE', '0')
    events = []
    caller_program = ProgramKey('cute', 'd' * 64, 'direct-prime')
    engine = session(tmp_path, autotune=disable != 'session', compile_workers=16)
    engine._selection_cache = forbidden
    engine._compiler = forbidden
    if disable == 'cancel':
        engine.cancel_tuning()
    # A configured shard must have no effect on local heuristic preparation.
    engine.configure_tuning_shard(2, (0, 1, 2, 3))
    def materialize(selection, device):
        events.append(('materialize', selection.config.width))
        assert plan.prepared is None
        return SimpleNamespace(value=0)
    def prepare(state):
        def run():
            assert plan.prepared is None
            record_program(caller_program)
            state.value = 7
            events.append('prime')
        return PreparedCall(run=run, restore=lambda: events.append('restore'), close=lambda: events.append('close'))
    plan = Plan(
        contract=replace(contract(), materialize=forbidden), query=Query(3),
        _compile_jobs=forbidden, _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )
    with engine:
        snapshots = []
        result = engine.prepare(
            (plan.request(name='local', prepare_call=prepare),),
            autotune=False if disable == 'job' else True if disable == 'environment' else None,
            progress=snapshots.append,
        )
        assert snapshots
        # A normal warmup-only batch must not be presented as stopped tuning.
        assert all(p.tuning_stopped == (disable in {'environment', 'cancel'}) for p in snapshots)
        assert plan.selection.config == Config(7)
        assert plan.prepared.state.value == 7
        assert plan.prepared.programs == frozenset({caller_program})
        assert result.benchmarked_candidates == result.cache_hits == 0
        assert engine._pool is None
        engine.freeze()
        engine.prepare((plan.request(name='ready', prepare_call=prepare),))
        assert events == [('materialize', 7), 'prime', 'restore']
    assert events[-1] == 'close'


def test_warmup_pin_and_dependency_programs_survive(tmp_path, monkeypatch):
    monkeypatch.setattr(TuningContract, 'parameter_space', forbidden)
    key = ProgramKey('cute', 'e' * 64, 'dependency')
    first = request(name='first', tuning=contract(), pin=Config(9))
    first = replace(first, prepare_call=lambda state: PreparedCall(run=lambda: record_program(key)))
    second = request(name='second', tuning=contract(), dependencies=('first',))
    with session(tmp_path, autotune=False) as engine:
        engine.prepare((first, second))
        assert first.plan.selection.config == Config(9)
        assert second.plan.prepared.programs == frozenset({key})


def test_cancelled_compiler_wait_terminates_pool_and_primes_default(tmp_path, monkeypatch):
    from b12x.preparation import session as module
    from b12x._lib.compile_pool import CompileJob
    events = []
    key = ProgramKey('cute', 'f' * 64, 'queued')
    class Pool:
        pending = True
        active_compilations = 1
        def summary(self):
            return SimpleNamespace(cute_compilations=0, triton_compilations=0)
        def submit_plans(self, *_args, **_kwargs):
            events.append('queued')
        def ready(self, *_args):
            return False
        def cancel_optional(self):
            events.append('cancel-queued')
        def wake(self):
            pass
        def close(self, *, terminate=False):
            assert terminate
            events.append('terminate')
    monkeypatch.setattr(module, 'compiled_program_available', lambda program: False)
    monkeypatch.setattr(module, 'describe_compilation', lambda job: SimpleNamespace(programs=(key,)))
    with session(tmp_path) as engine:
        pool = Pool()
        def compiler():
            engine._pool = pool
            return pool
        engine._compiler = compiler
        req = request(name='waiting', tuning=contract(), pin=Config(9), calls=events)
        req = replace(req, plan=replace(req.plan, _compile_jobs=lambda config, device: (
            CompileJob(factory='tests.preparation.test_heuristic_warmup:forbidden'),
        )))
        job = engine.begin((req,))
        assert job.advance().pending_compilation
        engine.cancel_tuning()
        while not job.advance().done:
            pass
        assert req.plan.selection.config == Config(9)
        assert events.index('terminate') < events.index(27)
        assert engine._pool is None
        assert job.result().benchmarked_candidates == 0


def test_cancellation_during_multi_job_planning_uses_default(tmp_path, monkeypatch):
    from b12x.preparation import session as module
    from b12x._lib.compile_pool import CompileJob
    calls = []
    with session(tmp_path) as engine:
        def describe(job):
            engine.cancel_tuning()
            return SimpleNamespace(programs=())
        monkeypatch.setattr(module, 'describe_compilation', describe)
        engine._compiler = forbidden
        req = request(name='planning', tuning=contract(), calls=calls, benchmark=forbidden)
        req = replace(req, plan=replace(req.plan, _compile_jobs=lambda config, device: (
            CompileJob(factory='tests.preparation.test_heuristic_warmup:forbidden'),
            CompileJob(factory='tests.preparation.test_heuristic_warmup:forbidden'),
        )))
        result = engine.prepare((req,))
        assert result.selections['planning'].config == Config(7)
        assert calls == [21]
        assert engine._cache.records == {}


def test_failed_heuristic_prime_never_publishes_readiness(tmp_path):
    closed, restored = [], []
    req = request(name='failed', tuning=contract())
    def fail():
        raise ValueError('native prime failed')
    req = replace(req, prepare_call=lambda state: PreparedCall(
        run=fail, close=lambda: closed.append(1), restore=lambda: restored.append(1),
    ))
    with session(tmp_path, autotune=False) as engine:
        with pytest.raises(ValueError, match='native prime failed'):
            engine.prepare((req,))
        assert req.plan.prepared is None
    assert closed == restored == [1]


@pytest.mark.parametrize('already_prepared', [False, True])
def test_cancelled_collective_wait_keeps_prepared_state_or_uses_default(tmp_path, already_prepared):
    from b12x.preparation import CollectiveRequirement
    calls = []
    req = request(name='collective', tuning=contract(values=(2,)), calls=calls)
    with session(tmp_path) as engine:
        if already_prepared:
            engine.prepare((req,), autotune=False)
        initial = req.plan.prepared
        req = replace(req, collective=CollectiveRequirement('comm', (0,)))
        job = engine.begin((req,))
        while not job.advance().ready_collectives:
            pass
        engine.cancel_tuning()
        progress = job.advance(collective_key='comm')
        while not progress.done:
            progress = job.advance()
        assert req.plan.selection.config == Config(7)
        assert calls == [21]
        if initial is not None:
            assert req.plan.prepared is initial
