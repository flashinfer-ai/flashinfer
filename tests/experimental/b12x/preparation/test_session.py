"""Host state-machine boundaries, without substituting a serving kernel."""
import gc
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.preparation import (
    CollectiveRequirement, DetectedDevice, MemoryRequirements,
    PersistentMemory, Plan, PreparationSession, PreparedCall, current_plan,
    current_prepared_state, plan_from_handle, require_prepared,
)
from b12x.preparation._cache import SelectionCache
from b12x.preparation.types import DeviceIdentity, _CompositePlan
from b12x._lib.scratch import ScratchBufferSpec
from b12x._lib.runtime_control import KernelResolutionFrozenError, kernel_resolution_guard
from .test_defaults import Config, Query, contract


def session(tmp_path, **kwargs):
    value = PreparationSession(device=DetectedDevice(None, None), **kwargs)
    value._cache = SelectionCache(tmp_path, {"schema_version": 6, "tuning_cache_version": 1})
    return value


@pytest.mark.parametrize("workers", [0, 1, 4])
def test_compiler_process_budget_can_be_limited_without_disabling_tuning(
    tmp_path, monkeypatch, workers
):
    monkeypatch.setenv("B12X_COMPILE_WORKERS", str(workers))
    with session(tmp_path) as engine:
        assert engine.compile_workers == workers
        assert engine.autotune
    with session(tmp_path, compile_workers=2) as engine:
        assert engine.compile_workers == 2


@pytest.mark.parametrize("identity, workers", [
    (DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10"), 4),
    (DeviceIdentity("nvidia", (12, 1), 48, "GB10"), 4),
    (DeviceIdentity("nvidia", (12, 0), 170, "NVIDIA GeForce RTX 5090"), 8),
    (None, 8),
])
def test_compiler_default_limits_spark_memory_across_stages(monkeypatch, identity, workers):
    monkeypatch.delenv("B12X_COMPILE_WORKERS", raising=False)
    with PreparationSession(device=DetectedDevice(None, identity)) as engine:
        assert engine.compile_workers == workers
        engine.configure_compile_workers(2)
        engine.configure_compile_workers()
        assert engine.compile_workers == workers


def test_spark_compiler_budget_preserves_explicit_overrides(monkeypatch):
    monkeypatch.setenv("B12X_COMPILE_WORKERS", "6")
    device = DetectedDevice(None, DeviceIdentity("nvidia", (12, 1), 48, "NVIDIA GB10"))
    with PreparationSession(device=device, compile_workers=1) as engine:
        assert engine.compile_workers == 1
        engine.configure_compile_workers()
        assert engine.compile_workers == 6
        engine.configure_compile_workers(0)
        assert engine.compile_workers == 0
    with PreparationSession(device=device) as engine:
        assert engine.compile_workers == 6


def declaration(*, tuning=None, pin=None, shared=False):
    tuning = contract(values=(2,)) if tuning is None else tuning
    return Plan(
        contract=tuning, query=Query(3), override=pin, shared=shared,
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=lambda selection, device: SimpleNamespace(value=selection.config.width * 3),
    )


def test_compiler_budget_changes_between_jobs_preserve_prepared_plans(tmp_path):
    with session(tmp_path) as engine:
        first = request(name="weights")
        engine.configure_compile_workers(16)
        engine.prepare((first,))
        prepared = first.plan.prepared
        engine.configure_compile_workers(4)
        job = engine.begin((request(name="state"),))
        with pytest.raises(RuntimeError, match="between jobs"):
            engine.configure_compile_workers(2)
        job.close()
        assert engine.compile_workers == 4
        assert first.plan.prepared is prepared
        assert require_prepared(first.plan, "test.arithmetic").value == 6


def request(*, name, tuning=None, pin=None, calls=None, close=None, benchmark=None, dependencies=(), collective=None, shared=False):
    calls = [] if calls is None else calls
    return declaration(tuning=tuning, pin=pin, shared=shared).request(
        name=name,
        prepare_call=lambda state: PreparedCall(run=lambda: calls.append(state.value), close=close),
        benchmark_call=benchmark, dependencies=dependencies, collective=collective,
    )


def test_prepare_fills_plans_in_place_and_release_runs_closers(tmp_path):
    calls, closed = [], []
    a = request(name="a", calls=calls, close=lambda: closed.append("a"))
    b = request(name="b", calls=calls, close=lambda: closed.append("b"))
    engine = session(tmp_path)
    result = engine.prepare((a, b))
    assert calls == [6, 6]
    assert result.plans["a"] is a.plan
    assert require_prepared(a.plan, "test.arithmetic").value == 6
    assert a.plan.selection.source == "fixed"
    engine.freeze()
    engine.prepare((a, b))
    assert calls == [6, 6]
    engine.release(a.plan)
    assert closed == ["a"]
    assert a.plan.prepared is None
    with pytest.raises(RuntimeError, match="frozen"):
        require_prepared(a.plan, "test.arithmetic")
    engine.close()
    assert sorted(closed) == ["a", "b"]
    assert b.plan.prepared is None


def test_job_bundle_memo_ends_at_completion_but_plan_programs_survive(tmp_path):
    import weakref
    from b12x._lib import compile_plan, program_cache

    built = []
    key = compile_plan.ProgramKey("cute", "7" * 64, "job-owned")

    @program_cache.program_cache
    def executable():
        return compile_plan.CompiledCuTeProgram(lambda: 23, key)

    @program_cache.program_cache(scope="preparation")
    def factory():
        built.append(1)
        return executable()

    def make(name):
        plan = replace(declaration(), _materialize=lambda *_: compile_plan.load_programs(factory()))
        return plan.request(name=name, prepare_call=lambda state: PreparedCall(run=state))

    with session(tmp_path, autotune=False) as engine:
        requests = (make("a"), make("b"))
        job = engine.begin(requests)
        while not job.advance().done:
            pass
        assert not job._program_cache._caches
        assert built == [1]
        owner = weakref.ref(requests[0].plan.prepared.state)
        assert owner() is requests[1].plan.prepared.state
        engine.release_many((requests[0].plan,))
        assert requests[1].plan.prepared.state() == 23
        assert executable.cache_info().currsize == 1
        engine.release_many((requests[1].plan,))
        assert executable.cache_info().currsize == 0
        assert owner() is None
        engine.prepare((make("later"),))
        assert built == [1, 1]


def test_bulk_release_attempts_all_closers_and_reclaims_once(tmp_path, monkeypatch):
    closed, reclaimed = [], []

    def close(name):
        closed.append(name)
        if name == "a":
            raise RuntimeError("closer failed")

    with session(tmp_path) as engine:
        requests = tuple(request(name=name, close=lambda name=name: close(name)) for name in ("a", "b"))
        engine.prepare(requests)
        reclaim = engine._reclaim_programs
        monkeypatch.setattr(engine, "_reclaim_programs", lambda: (reclaimed.append(1), reclaim()))
        with pytest.raises(RuntimeError, match="closer failed"):
            engine.release_many(item.plan for item in requests)
        assert closed == ["a", "b"]
        assert reclaimed == [1]
        assert all(item.plan.prepared is None for item in requests)
        engine.release_many(item.plan for item in requests)
        assert reclaimed == [1]


def test_closing_one_session_preserves_another_sessions_executable(tmp_path):
    from b12x._lib import compile_plan, program_cache

    @program_cache.program_cache
    def executable():
        return compile_plan.CompiledCuTeProgram(lambda: 41, compile_plan.ProgramKey("cute", "6" * 64))

    def make(name):
        plan = replace(declaration(), _materialize=lambda *_: compile_plan.load_programs(executable()))
        return plan.request(name=name, prepare_call=lambda state: PreparedCall(run=state))

    with session(tmp_path, autotune=False) as first, session(tmp_path, autotune=False) as second:
        a, b = make("a"), make("b")
        first.prepare((a,))
        second.prepare((b,))
        first.close()
        assert b.plan.prepared.state() == 41
        assert executable.cache_info().currsize == 1
        second.close()
        assert executable.cache_info().currsize == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_job_factory_scope_clears_on_failure_or_early_close(tmp_path, cancel):
    from b12x._lib.program_cache import PreparationProgramCache, program_cache

    @program_cache(scope="preparation")
    def factory():
        return SimpleNamespace(value=3)

    def fail(state):
        raise RuntimeError("binding failed")

    plan = replace(declaration(), _materialize=lambda *_: factory())
    with session(tmp_path, autotune=False) as engine:
        job = engine.begin((plan.request(name="a", prepare_call=fail),))
        if cancel:
            with job._program_cache.activate():
                factory()
            job.close()
        else:
            with pytest.raises(RuntimeError, match="binding failed"):
                while not job.advance().done:
                    pass
        assert not job._program_cache._caches
        with PreparationProgramCache().activate():
            assert factory().value == 3


def test_plan_scoped_persistent_owners_reserve_independent_buffers(tmp_path):
    buffers = {}

    def memory(config, device):
        return MemoryRequirements(persistent=(
            PersistentMemory(("scratch-owner", current_plan()), 16),
            PersistentMemory("shared-readonly", 8, 8),
        ))

    def materialize(selection, device):
        buffers[current_plan()] = bytearray(16)
        return SimpleNamespace(buffer=buffers[current_plan()])

    def make(name):
        plan = Plan(
            contract=contract(values=(2,)), query=Query(3),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=memory, _materialize=materialize,
        )
        return plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(run=lambda: state.buffer.__setitem__(0, 1)),
        )

    requests = (make("target"), make("draft"))
    with session(tmp_path) as engine:
        assert engine.candidate_memory_envelope(requests).pending_persistent_nbytes == 32
        job = engine.begin(requests)
        while True:
            progress = job.advance()
            with pytest.raises(RuntimeError):
                current_plan()
            if progress.done:
                break
        job.result()
        target = require_prepared(requests[0].plan, "test.arithmetic").buffer
        draft = require_prepared(requests[1].plan, "test.arithmetic").buffer
        target[0] = 9
        assert draft[0] == 1


def test_sticky_stop_before_enumeration_prepares_default_without_winner(tmp_path):
    def no_optional(query, device, assignment):
        raise AssertionError("stopped session enumerated an optional candidate")

    tuning = replace(contract(), materialize=no_optional)
    calls = []
    with session(tmp_path) as engine:
        engine.cancel_tuning()
        result = engine.prepare((request(name="a", tuning=tuning, calls=calls),))
        assert result.selections["a"].source == "default"
        result = engine.prepare((request(name="b", tuning=tuning, calls=calls),))
        assert result.selections["b"].source == "default"
        assert engine._cache.records == {}
    assert calls == [21, 21]


def test_cache_only_effective_singleton_and_explicit_pin_need_no_selection_record(tmp_path):
    tuning = replace(contract(), equivalence_key=lambda query, device, config: {"same": True})
    calls = []
    with session(tmp_path, cache_only=True) as engine:
        result = engine.prepare((request(name="singleton", tuning=tuning, calls=calls),))
        assert result.selections["singleton"].source == "fixed"
        result = engine.prepare((request(name="pinned", tuning=contract(), pin=Config(9), calls=calls),))
        assert result.selections["pinned"].source == "override"
        with pytest.raises(LookupError):
            engine.prepare((request(name="missing", tuning=contract()),))
    assert calls == [3, 27]


def test_collective_requires_explicit_matching_authorization(tmp_path):
    calls = []
    requirement = CollectiveRequirement("group/prime", (0, 1))
    req = request(name="collective", calls=calls, collective=requirement)
    with session(tmp_path) as engine:
        with pytest.raises(ValueError):
            engine.prepare((req,))
        job = engine.begin((req,))
        progress = job.advance()
        assert progress.ready_collectives == (requirement,)
        assert calls == []
        progress = job.advance(collective_key="wrong/group")
        assert progress.ready_collectives == (requirement,)
        assert calls == []
        progress = job.advance(collective_key=requirement.key)
        while not progress.done:
            progress = job.advance()
        assert calls == [6]
        job.result()


def test_new_obligation_fails_after_freeze(tmp_path):
    with session(tmp_path) as engine:
        engine.prepare((request(name="ready"),))
        engine.freeze()
        with pytest.raises(KernelResolutionFrozenError):
            engine.prepare((request(name="not-ready"),))


def test_second_prepare_is_incremental_for_prepared_plans(tmp_path):
    calls = []
    a, b = request(name="a", calls=calls), request(name="b", calls=calls)
    with session(tmp_path) as engine:
        engine.prepare((a, b))
        first = a.plan.prepared
        c = request(name="c", calls=calls)
        engine.prepare((a, b, c))
        assert calls == [6, 6, 6]
        assert a.plan.prepared is first
        assert c.plan.prepared is not None


def test_failure_restores_and_closes_all_while_preserving_primary_error(tmp_path):
    restored = []

    def factory(state):
        def fail():
            raise ValueError("primary failure")

        def restore():
            restored.append("restore")
            raise RuntimeError("cleanup failure")

        return PreparedCall(run=fail, restore=restore, close=lambda: restored.append("close"))

    req = request(name="failed")
    req = replace(req, prepare_call=factory)
    with session(tmp_path) as engine:
        with pytest.raises(ValueError, match="primary failure"):
            engine.prepare((req,))
    assert restored == ["restore", "close"]
    assert req.plan.prepared is None


def test_persistent_memory_counts_shared_keys_once_and_rejects_conflicts():
    requirements = MemoryRequirements(persistent=(
        PersistentMemory("weights-side-state", 40, 10),
        PersistentMemory("weights-side-state", 40, 10),
    ))
    assert requirements.pending_persistent_nbytes == 30
    assert MemoryRequirements(persistent=(PersistentMemory("state", 40, 40),)).pending_persistent_nbytes == 0
    with pytest.raises(ValueError):
        MemoryRequirements.sequential((requirements, MemoryRequirements(persistent=(
            PersistentMemory("weights-side-state", 40, 20),
        ))))


def _deterministic_timer(monkeypatch, *, stop=None, batches=None):
    from b12x.preparation import _measurement

    def prepare(calls, **kwargs):
        if batches is not None:
            batches.append(tuple(call.output for call in calls))
        yield
        return SimpleNamespace(calls=calls, close=lambda: None, completed_rounds=0,
                               planned_rounds=0, active_count=len(calls), latest_round_us=())

    def measure(race, **kwargs):
        if stop is not None:
            stop()
        yield
        return _measurement.RaceMeasurements(
            tuple(abs(call.output - 6) + 1 for call in race.calls), 0,
        )

    monkeypatch.setattr(_measurement, "prepare_race_steps", prepare)
    monkeypatch.setattr(_measurement, "measure_race_steps", measure)


def test_complete_race_cached_restart_and_disabled_precedence(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    trial_closed, calls = [], []

    def benchmark(state):
        return PreparedCall(
            run=lambda: state.value, produce=lambda: None,
            close=lambda: trial_closed.append(state.value),
        )

    with session(tmp_path) as engine:
        result = engine.prepare((request(name="first", tuning=contract(), calls=calls, benchmark=benchmark),))
        assert result.selections["first"].source == "tuned"
        assert result.selections["first"].config.width == 2
        assert result.benchmarked_candidates == 3
        assert sorted(trial_closed) == [3, 6, 12]
    with session(tmp_path) as engine:
        result = engine.prepare((request(name="different-owner", tuning=contract(), calls=calls),))
        assert result.selections["different-owner"].source == "cached"
    with session(tmp_path, autotune=False) as engine:
        result = engine.prepare((request(name="heuristic", tuning=contract(), calls=calls),))
        assert result.selections["heuristic"].source == "default"
        assert result.selections["heuristic"].config == Config(7)
        result = engine.prepare((request(name="pin", tuning=contract(), pin=Config(9), calls=calls),))
        assert result.selections["pin"].source == "override"
    assert calls == [6, 6, 21, 27]


def test_race_batches_bound_residency_and_carry_the_champion(tmp_path, monkeypatch):
    batches, trial_closed = [], []
    _deterministic_timer(monkeypatch, batches=batches)

    def benchmark(state):
        return PreparedCall(
            run=lambda: state.value, produce=lambda: None,
            close=lambda: trial_closed.append(state.value),
        )

    tuning = contract(values=(1, 2, 4, 8))
    with session(tmp_path, race_batch=2) as engine:
        result = engine.prepare((request(name="batched", tuning=tuning, benchmark=benchmark),))
        assert result.selections["batched"].config.width == 2
        assert result.benchmarked_candidates == 4
        assert result.coverage["batched"]["measured_count"] == 4
    assert batches == [(3, 6), (6, 12, 24)]
    assert sorted(trial_closed) == [3, 6, 12, 24]


def _cuda_error(monkeypatch, code=720):
    from cutlass.base_dsl import common

    # Construct the actual DSL exception without querying a CUDA context.
    monkeypatch.setattr(common, "_get_friendly_cuda_error_message",
                        lambda code, name: (f"{name} ({code})", "", ""))
    return common.DSLCudaRuntimeError(code, "test CUDA launch error")


@pytest.mark.parametrize("rejected", ((3,), (12,), (3, 6)))
def test_unlaunchable_trials_release_scratch_and_race_survivors(
    tmp_path, monkeypatch, caplog, rejected,
):
    batches, events = [], []
    _deterministic_timer(monkeypatch, batches=batches)
    error = _cuda_error(monkeypatch)

    def benchmark(state):
        def run():
            events.append(("run", state.value))
            if state.value in rejected:
                raise error
            return state.value

        return PreparedCall(
            run=run, produce=lambda: None,
            restore=lambda: events.append(("restore", state.value)),
            close=lambda: events.append(("close", state.value)),
        )

    with session(tmp_path, race_batch=1) as engine:
        monkeypatch.setattr(engine, "_synchronize", lambda: events.append("sync"))
        result = engine.prepare((request(name="race", tuning=contract(), benchmark=benchmark),))
        survivors = {3, 6, 12} - set(rejected)
        assert result.selections["race"].config.width * 3 in survivors
        assert result.benchmarked_candidates == len(survivors)
        assert result.coverage["race"]["measured_count"] == len(survivors)
        assert all(set(batch) <= survivors for batch in batches)
        # The cache accepts only fully measured races.
        assert not engine._cache.records
    for value in rejected:
        start = events.index(("run", value))
        assert events[start + 1:start + 4] == ["sync", ("restore", value), ("close", value)]
        assert events.count(("close", value)) == 1
    assert caplog.text.count("Skipping race candidate") == len(rejected)


def test_all_trials_rejected_fail_without_installing_a_default(tmp_path, monkeypatch):
    error = _cuda_error(monkeypatch)
    closed, prepared = [], []

    def benchmark(state):
        def run():
            raise error
        return PreparedCall(run=run, close=lambda: closed.append(state.value))

    req = request(name="race", tuning=contract(), calls=prepared, benchmark=benchmark)
    with session(tmp_path) as engine:
        with pytest.raises(RuntimeError, match="no launchable candidates for race: all 3"):
            engine.prepare((req,))
        assert req.plan.prepared is None
        assert not engine._cache.records
    assert closed == [3, 6, 12]
    assert not prepared


@pytest.mark.parametrize("failure", ("cuda", "text", "factory", "restore", "sync", "selected"))
def test_candidate_recovery_keeps_other_failures_fatal(tmp_path, monkeypatch, failure):
    _deterministic_timer(monkeypatch)
    error = (_cuda_error(monkeypatch, 700 if failure == "cuda" else 720)
             if failure != "text" else RuntimeError("CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (720)"))
    attempted = []

    def fail():
        raise error

    def cleanup_failure():
        raise RuntimeError("cleanup failed")

    def benchmark(state):
        attempted.append(state.value)
        if failure == "factory":
            fail()
        return PreparedCall(
            run=(lambda: state.value) if failure == "selected" else fail,
            restore=cleanup_failure if failure == "restore" else None,
        )

    req = request(name="race", tuning=contract(), benchmark=benchmark)
    if failure == "selected":
        req = replace(req, prepare_call=lambda state: PreparedCall(run=fail))
    with session(tmp_path) as engine:
        if failure == "sync":
            monkeypatch.setattr(engine, "_synchronize", cleanup_failure)
        with pytest.raises(RuntimeError, match="cleanup failed" if failure in {"restore", "sync"} else "failed to prepare"):
            engine.prepare((req,))
        assert req.plan.prepared is None
    assert attempted == ([3, 6, 12] if failure == "selected" else [3])


def test_rejected_shard_accepts_a_peer_winner_without_claiming_full_measurement(
    tmp_path, monkeypatch,
):
    from b12x.preparation import TuningRequirement

    error = _cuda_error(monkeypatch)

    def benchmark(state):
        def run():
            raise error
        return PreparedCall(run=run)

    with session(tmp_path) as engine:
        engine.configure_tuning_shard(0, (0, 1))
        req = request(name="race", tuning=contract(values=(1, 2)), benchmark=benchmark)
        job = engine.begin((req,))
        snapshot = job.advance().ready_cache
        progress = job.advance(cache=(snapshot, snapshot))
        while not progress.ready_tuning:
            progress = job.advance()
        contribution, = progress.ready_tuning
        assert contribution.assignment is None and contribution.rejected_count == 1
        winner = TuningRequirement(contribution.key, (0, 1), {"width": 2}, 1.0, 1, 1)
        progress = job.advance(tuning=(winner,))
        while not progress.done:
            progress = job.advance()
        result = job.result()
        assert req.plan.selection.config.width == 2
        assert result.coverage["race"]["measured_count"] == 1
        assert result.benchmarked_candidates == 0
        assert not engine._cache.records


@pytest.mark.parametrize("cached_ranks", ((), (0,), (1,), (0, 1)))
def test_two_ranks_agree_on_cached_choices_and_shard_remaining_races(
    tmp_path, monkeypatch, cached_ranks,
):
    _deterministic_timer(monkeypatch)
    tuning = contract(values=(1, 2, 4, 8))
    for rank in cached_ranks:
        with session(tmp_path / f"rank-{rank}") as engine:
            engine.prepare((request(
                name="cached", tuning=tuning,
                benchmark=lambda state: PreparedCall(run=lambda: state.value, produce=lambda: None),
            ),))
            # Simulate another completed race choosing a different winner.
            if rank == 1 and len(cached_ranks) == 2:
                key, record = next(iter(engine._cache.records.items()))
                engine._cache.save(key, assignment={"width": 4}, config={"width": 4},
                                   coverage=record["coverage"], programs=())
    engines = [session(tmp_path / f"rank-{rank}") for rank in range(2)]
    requests = []
    jobs = []
    for rank, engine in enumerate(engines):
        engine.configure_tuning_shard(rank, (0, 1))
        req = request(
            name="shared",
            tuning=tuning,
            benchmark=lambda state: PreparedCall(
                run=lambda: state.value,
                produce=lambda: None,
            ),
        )
        fresh = replace(req, name="fresh", dependencies=("shared",),
                        plan=replace(req.plan, query=Query(5)))
        requests.append((req, fresh))
        jobs.append(engine.begin((req, fresh)))

    authorizations = [None, None]
    caches = [None, None]
    progress = [None, None]
    for _ in range(100):
        progress = [
            job.advance(tuning=authorization, cache=cache)
            for job, authorization, cache in zip(jobs, authorizations, caches, strict=True)
        ]
        authorizations = [None, None]
        caches = [None, None]
        if any(state.ready_cache is not None for state in progress):
            snapshots = tuple(state.ready_cache for state in progress)
            assert all(snapshot is not None for snapshot in snapshots)
            caches = [snapshots, snapshots]
        contributions = [
            item
            for state in progress
            for item in state.ready_tuning
        ]
        if contributions:
            assert len(contributions) == 2
            assert {item.candidate_index for item in contributions} == {0, 1}
            winner = min(
                contributions,
                key=lambda item: (item.latency_us, item.candidate_index),
            )
            authorizations = [winner, winner]
        if all(state.done for state in progress):
            break
    else:
        pytest.fail("distributed tuning did not complete")

    results = [job.result() for job in jobs]
    try:
        assert [result.benchmarked_candidates for result in results] == ([2, 2] if cached_ranks else [4, 4])
        assert [result.selections["shared"].config.width for result in results] == [2, 2]
        assert [result.selections["shared"].source for result in results] == (["cached"] * 2 if cached_ranks else ["tuned"] * 2)
        assert all(result.selections["fresh"].source == "tuned" for result in results)
        assert [result.coverage["shared"]["measured_count"] for result in results] == [4, 4]
        assert [req[0].plan.selection.config.width for req in requests] == [2, 2]
        # Saving the fresh choice must not restore a conflicting local winner.
        for engine in engines:
            result = engine.prepare((request(name="again", tuning=tuning),))
            assert result.selections["again"].config.width == 2
    finally:
        for engine in engines:
            engine.close()


def test_job_autotune_override_upgrades_live_default_and_reuses_winner(
    tmp_path, monkeypatch
):
    _deterministic_timer(monkeypatch)
    calls = []

    def benchmark(state):
        return PreparedCall(
            run=lambda: state.value,
            produce=lambda: None,
        )

    with session(tmp_path) as engine:
        early = request(
            name="early",
            tuning=contract(),
            calls=calls,
            benchmark=benchmark,
        )
        result = engine.prepare((early,), autotune=False)
        assert result.selections["early"].source == "default"
        assert early.plan.selection.source == "default"

        result = engine.prepare((early,))
        assert result.selections["early"].source == "tuned"
        assert result.selections["early"].config.width == 2
        assert early.plan.selection.source == "tuned"

        result = engine.prepare((early,))
        assert result.selections["early"].source == "tuned"

    assert calls == [21, 6]


def test_candidate_memory_envelope_covers_every_legal_config(tmp_path):
    tuning = contract(default=1, values=(256, 512, 1024))
    plan = Plan(
        contract=tuning,
        query=Query(3),
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(
            scratch=(
                ScratchBufferSpec(
                    name="candidate",
                    shape=(config.width,),
                    dtype=torch.uint8,
                    device=torch.device("cpu"),
                ),
            )
        ),
        _materialize=lambda selection, device: object(),
    )
    candidate = plan.request(
        name="candidate",
        prepare_call=lambda state: PreparedCall(run=lambda: None),
    )

    with session(tmp_path) as engine:
        envelope = engine.candidate_memory_envelope((candidate,))

    assert envelope.scratch[0].shape == (1024,)
    assert plan.scratch_specs()[0].shape == (1,)


def test_prepared_scratch_reuses_selection_and_preserves_live_residency(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    monkeypatch.setattr("b12x.preparation.device.detect_device", lambda device=None: DetectedDevice(None, None))
    sized = []

    def memory(config, device):
        state = current_prepared_state()
        sized.append((current_plan(), state))
        return MemoryRequirements(
            scratch=(ScratchBufferSpec(
                name="workspace", shape=(config.width,),
                dtype=torch.uint8, device=torch.device("cpu"),
            ),),
            persistent=(PersistentMemory(
                current_plan(), 16, 0 if state is None else state.resident,
            ),),
        )

    plan = Plan(
        contract=contract(), query=Query(3),
        _compile_jobs=lambda config, device: (), _memory_requirements=memory,
        _materialize=lambda selection, device: SimpleNamespace(
            value=selection.config.width * 3, resident=8,
        ),
    )
    req = plan.request(
        name="workspace",
        prepare_call=lambda state: PreparedCall(run=lambda: state.value),
        benchmark_call=lambda state: PreparedCall(run=lambda: state.value, produce=lambda: None),
    )
    assert plan.scratch_specs()[0].shape == (7,)
    with session(tmp_path) as engine:
        engine.prepare((req,), autotune=False)
        default_specs = plan.scratch_specs()
        assert default_specs[0].shape == (7,)
        engine.prepare((req,))
        selected_specs = plan.scratch_specs()
        assert selected_specs[0].shape == (2,)
        assert selected_specs is not default_specs
        assert sized[-1] == (plan, plan.prepared.state)
        assert plan.memory_requirements().pending_persistent_nbytes == 8
        plan.prepared.state.resident = 16
        assert plan.memory_requirements().pending_persistent_nbytes == 0
        count = len(sized)
        engine.freeze()
        with engine.capture():
            for _ in range(10):
                assert plan.scratch_specs() is selected_specs
        assert len(sized) == count
    assert plan.scratch_specs()[0].shape == (7,)
    assert plan.memory_requirements().pending_persistent_nbytes == 16
    with session(tmp_path) as engine:
        engine.prepare((req,), autotune=False)
        assert plan.scratch_specs()[0].shape == (7,)
        assert plan.scratch_specs() is not selected_specs


def test_composite_scratch_retains_all_exact_variants_without_replanning(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    monkeypatch.setattr("b12x.preparation.device.detect_device", lambda device=None: DetectedDevice(None, None))
    sized = []
    counts = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 4096)

    def child(rows):
        def memory(config, device):
            sized.append(rows)
            return MemoryRequirements(scratch=(ScratchBufferSpec(
                name="workspace", shape=(rows * config.width,),
                dtype=torch.uint8, device=torch.device("cpu"),
            ),))

        return Plan(
            contract=contract(), query=Query(rows),
            _compile_jobs=lambda config, device: (), _memory_requirements=memory,
            _materialize=lambda selection, device: SimpleNamespace(value=selection.config.width * 3),
        )

    children = {rows: child(rows) for rows in counts}
    plan = _CompositePlan(
        component_id="test.arithmetic", capacity_metadata={}, variants=children,
        _assemble=lambda states, device: dict(states),
    )
    req = plan.request(
        name="workspace",
        prepare_calls={rows: lambda state: PreparedCall(run=lambda: state.value) for rows in counts},
        benchmark_calls={rows: lambda state: PreparedCall(run=lambda: state.value, produce=lambda: None) for rows in counts},
    )
    assert plan.scratch_specs()[0].shape == (4096 * 7,)
    with session(tmp_path) as engine:
        engine.prepare((req,), autotune=False)
        assert plan.scratch_specs()[0].shape == (4096 * 7,)
        engine.prepare((req,))
        specs = plan.scratch_specs()
        assert specs[0].shape == (4096 * 2,)
        assert tuple(plan.prepared.state) == counts
        assert all(state.value == 6 for state in plan.prepared.state.values())
        count = len(sized)
        engine.freeze()
        with engine.capture():
            for _ in range(80):
                assert plan.scratch_specs() is specs
        assert len(sized) == count
    assert plan.scratch_specs()[0].shape == (4096 * 7,)


def test_failed_prepared_scratch_snapshot_preserves_previous_payload(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    closed = []

    def memory(config, device):
        state = current_prepared_state()
        if state is not None and state.value == 6:
            raise ValueError("selected workspace is invalid")
        return MemoryRequirements()

    plan = replace(declaration(tuning=contract()), _memory_requirements=memory)
    req = plan.request(
        name="workspace",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.value, close=lambda: closed.append(state.value),
        ),
        benchmark_call=lambda state: PreparedCall(run=lambda: state.value, produce=lambda: None),
    )
    with session(tmp_path) as engine:
        engine.prepare((req,), autotune=False)
        previous = plan.prepared
        with pytest.raises(ValueError, match="selected workspace is invalid"):
            engine.prepare((req,))
        assert plan.prepared is previous
        assert require_prepared(plan, "test.arithmetic").value == 21
        assert plan.scratch_specs() == ()
        assert closed == [6]
    assert closed == [6, 21]


def test_stop_mid_race_discards_partial_winner_and_restores_trials(tmp_path, monkeypatch):
    calls, restored = [], []
    with session(tmp_path) as engine:
        _deterministic_timer(monkeypatch, stop=engine.cancel_tuning)

        def benchmark(state):
            return PreparedCall(
                run=lambda: state.value, produce=lambda: None,
                restore=lambda: restored.append(state.value),
            )

        result = engine.prepare((request(name="stopped", tuning=contract(), calls=calls, benchmark=benchmark),))
        assert result.selections["stopped"].source == "default"
        assert result.benchmarked_candidates == 0
        assert engine._cache.records == {}
    assert calls == [21]
    assert sorted(restored) == [3, 6, 12]


def test_equal_declarations_enumerate_their_candidates_once(tmp_path, monkeypatch):
    from b12x.preparation.tuning import TuningContract

    enumerated = []
    original = TuningContract.iterate

    def counting(self, configuration):
        enumerated.append(configuration.encoded_query)
        return original(self, configuration)

    monkeypatch.setattr(TuningContract, "iterate", counting)
    _deterministic_timer(monkeypatch)
    tuning = contract()

    def benchmark(state):
        return PreparedCall(run=lambda: state.value, produce=lambda: None)

    def duplicate(name, rows):
        plan = Plan(
            contract=tuning, query=Query(rows),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(value=selection.config.width),
        )
        return plan.request(
            name=name, prepare_call=lambda state: PreparedCall(run=lambda: None),
            benchmark_call=benchmark,
        )

    with session(tmp_path) as engine:
        result = engine.prepare((
            duplicate("first", 3), duplicate("second", 3), duplicate("wider", 5),
        ))
    assert [query["rows"] for query in enumerated] == [3, 5]
    assert result.coverage["first"]["effective_count"] == 3
    assert result.coverage["second"]["effective_count"] == 3
    assert result.selections["second"].config == result.selections["first"].config


def test_duplicate_choice_dependency_order_still_prepares_both_plans(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    calls = []
    producer = request(name="producer", tuning=contract(), pin=Config(2))

    def benchmark(state):
        return PreparedCall(run=lambda: state.value, produce=lambda: None)

    a = request(name="a", tuning=contract(), calls=calls,
                benchmark=benchmark, dependencies=("producer",))
    b = request(name="b", tuning=contract(), calls=calls,
                benchmark=benchmark, dependencies=("producer",))
    with session(tmp_path) as engine:
        result = engine.prepare((b, a, producer))
        assert result.benchmarked_candidates == 3
        assert a.plan.prepared is not b.plan.prepared
    assert calls == [6, 6]


def test_shared_declarations_alias_one_prepared_state(tmp_path):
    calls, closed = [], []
    a = request(name="a", calls=calls, shared=True, close=lambda: closed.append("closed"))
    b = request(name="b", calls=calls, shared=True, close=lambda: closed.append("closed"))
    c = request(name="c", calls=calls, shared=True)
    with session(tmp_path) as engine:
        engine.prepare((a, b))
        assert calls == [6]
        assert a.plan.prepared is b.plan.prepared
        engine.prepare((c,))
        assert calls == [6]
        assert c.plan.prepared is a.plan.prepared
        engine.release(a.plan)
        assert a.plan.prepared is None
        assert b.plan.prepared is not None
        assert closed == []
        engine.release(b.plan)
        engine.release(c.plan)
        assert closed == ["closed"]
    with pytest.raises(ValueError, match="shared plan"):
        with session(tmp_path) as engine:
            engine.prepare((request(name="d", shared=True, dependencies=("e",)), request(name="e")))


def test_composite_prepares_children_and_assembles_their_states(tmp_path):
    assembled = []

    def child(width):
        return Plan(
            contract=contract(), query=Query(3), override=Config(width),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(value=selection.config.width * 3),
        )

    children = {1: child(1), 2: child(2)}
    root = _CompositePlan(
        component_id="test.arithmetic", capacity_metadata={}, variants=children,
        _assemble=lambda states, device: assembled.append(dict(states)) or SimpleNamespace(states=dict(states)),
    )
    calls = []
    req = root.request(
        name="root",
        prepare_calls={count: (lambda state: PreparedCall(run=lambda: calls.append(state.value))) for count in (1, 2)},
    )
    with session(tmp_path) as engine:
        result = engine.prepare((req,))
        assert result.plans["root"] is root
        assert sorted(calls) == [3, 6]
        state = require_prepared(root, "test.arithmetic")
        assert state.states == {1: children[1].prepared.state, 2: children[2].prepared.state}
        assert root.prepared.variants[2] is children[2]
        assert root.token_counts == (1, 2)
        engine.prepare((req,))
        assert len(assembled) == 1
        engine.release(root)
        assert root.prepared is None and children[1].prepared is None


def test_plan_handles_are_stable_and_resolve_only_live_plans():
    req = request(name="handle")
    handle = req.plan.handle
    assert plan_from_handle(handle) is req.plan
    with pytest.raises(RuntimeError):
        plan_from_handle(handle + 1_000_000)
    del req
    gc.collect()
    with pytest.raises(RuntimeError):
        plan_from_handle(handle)


def test_unprepared_plan_materializes_its_default_with_a_warning_before_freeze_only(
    tmp_path, caplog,
):
    from b12x.preparation.session import _warn_unprepared_declaration

    _warn_unprepared_declaration.cache_clear()
    calls = []
    req = request(name="lazy", calls=calls)
    with caplog.at_level("WARNING", logger="b12x"):
        state = require_prepared(req.plan, "test.arithmetic")
    assert state.value == 21
    assert req.plan.selection.source == "default"
    assert calls == []
    messages = [r.getMessage() for r in caplog.records if "not prepared before its first use" in r.getMessage()]
    assert len(messages) == 1 and "test.arithmetic" in messages[0]
    with pytest.raises(ValueError, match="belongs to"):
        require_prepared(req.plan, "test.other")
    other = request(name="later")
    with session(tmp_path) as engine:
        engine.prepare((request(name="ready"),))
        engine.freeze()
        with pytest.raises(RuntimeError, match="frozen"):
            require_prepared(other.plan, "test.arithmetic")


def test_priming_closures_release_transients_and_restore_before_readiness(tmp_path):
    import weakref

    class Temporary:
        pass

    references, lifetime = [], []

    def factory(state):
        source = Temporary()
        references.append(weakref.ref(source))

        def run():
            assert source is not None
            output = Temporary()
            references.append(weakref.ref(output))
            return output

        return PreparedCall(
            run=run, restore=lambda: lifetime.append("restored"),
            close=lambda: lifetime.append("resources closed"),
        )

    req = replace(request(name="temporary"), prepare_call=factory)
    with session(tmp_path) as engine:
        engine.prepare((req,))
        gc.collect()
        assert lifetime == ["restored"]
        assert all(reference() is None for reference in references)
    assert lifetime == ["restored", "resources closed"]


def test_prepared_resources_remain_mutable_after_inference_mode_priming(tmp_path):
    observed = {}

    def materialize(selection, device):
        observed["materialize_inference"] = torch.is_inference_mode_enabled()
        return SimpleNamespace(buffer=torch.zeros(1))

    def factory(state):
        observed["factory_inference"] = torch.is_inference_mode_enabled()
        observed["buffer"] = state.buffer

        def produce():
            observed["prime_inference"] = torch.is_inference_mode_enabled()
            state.buffer.fill_(1)

        return PreparedCall(
            run=lambda: state.buffer.add_(1),
            produce=produce,
            owners=(state.buffer,),
        )

    plan = Plan(
        contract=contract(values=(2,)),
        query=Query(3),
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )
    prepared = plan.request(
        name="inference-profile",
        prepare_call=factory,
    )

    with session(tmp_path) as engine, torch.inference_mode():
        engine.prepare((prepared,))
        assert observed["buffer"].item() == 2

    assert observed == {
        "materialize_inference": False,
        "factory_inference": False,
        "buffer": observed["buffer"],
        "prime_inference": True,
    }
    assert not observed["buffer"].is_inference()
    observed["buffer"].add_(1)
    assert observed["buffer"].item() == 3


def test_frozen_reuse_primes_independent_benchmark_trial(tmp_path):
    production, trials, closed = [], [], []

    def benchmark(state):
        token = object()
        return PreparedCall(
            run=lambda: trials.append((token, state.value)),
            close=lambda: closed.append(token),
        )

    req = replace(
        request(name="retained", calls=production, benchmark=benchmark),
        retain_benchmark_call=True,
    )
    with session(tmp_path) as engine:
        with engine.prepare((req,)) as first:
            first.benchmark_calls["retained"].invoke()
        assert len(closed) == 1
        engine.freeze()
        with engine.prepare((req,)) as second:
            second.benchmark_calls["retained"].invoke()
            assert production == [6]
            assert [value for _, value in trials] == [6, 6, 6, 6]
            assert trials[0][0] is not trials[2][0]
            assert len(closed) == 1
        assert len(closed) == 2


def test_frozen_reuse_rejects_changed_declared_device(tmp_path):
    req = request(name="device")
    with session(tmp_path) as engine:
        engine.prepare((req,))
        engine.freeze()
        changed = replace(req, plan=replace(req.plan, _device="cuda:1"))
        with pytest.raises(KernelResolutionFrozenError):
            engine.prepare((changed,))



def test_plan_state_omits_the_prepared_payload(tmp_path):
    """Compiler caches serialize closed-over plans; only the declaration travels."""
    req = request(name="pickled")
    plan = req.plan
    prepared_session = session(tmp_path, autotune=False)
    prepared_session.prepare((req,))
    assert plan.prepared is not None
    state = plan.__getstate__()
    assert state["_prepared"] is None
    copy = object.__new__(type(plan))
    copy.__setstate__(state)
    assert copy.prepared is None
    assert copy.handle == plan.handle
    assert copy.query == plan.query
    assert plan_from_handle(plan.handle) is plan
    assert plan.prepared is not None
    prepared_session.close()


def test_prepare_default_primes_with_the_request_call_and_refuses_after_freeze_or_under_capture(
    tmp_path, monkeypatch,
):
    import torch

    from b12x.preparation import prepare_default
    from b12x.preparation import session as session_module

    monkeypatch.setitem(session_module._LAZY_SESSIONS, None, session(tmp_path, autotune=False))
    calls = []
    req = request(name="on-demand", tuning=contract(values=(1, 2, 4)), calls=calls)
    prepared = prepare_default(req)
    assert prepared is req.plan.prepared and prepared is not None
    assert calls == [prepared.selection.config.width * 3]
    assert req.plan.selection.source == "default"
    with kernel_resolution_guard("frozen"), pytest.raises(RuntimeError, match="frozen"):
        prepare_default(request(name="late"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="under CUDA graph capture"):
        prepare_default(request(name="captured"))


def test_gpu_steps_share_a_bounded_advance_without_changing_order(tmp_path, monkeypatch):
    from b12x.preparation import session as implementation
    from b12x.preparation.types import PreparationResult

    clock = [implementation.time.monotonic()]
    monkeypatch.setattr(implementation.time, "monotonic", lambda: clock[0])
    seen = []

    def steps():
        for index in range(10):
            seen.append(index)
            clock[0] += 0.03
            yield "gpu"
        return PreparationResult(plans={})

    with session(tmp_path) as engine:
        job = engine.begin(())
        job._steps = steps()
        assert not job.advance().done
        assert seen == list(range(4))
        assert not job.advance().done
        assert seen == list(range(8))
        assert job.advance().done
        assert seen == list(range(10))
        job.result()


def test_tuning_contribution_preserves_rejected_candidate_count():
    from b12x.preparation import TuningRequirement

    winner = TuningRequirement("query", (0, 1), {"width": 2}, 1.0, 0)
    empty = TuningRequirement("query", (0, 1), None, None, None, rejected_count=2)
    consolidated = replace(winner, rejected_count=empty.rejected_count + 1)
    assert winner.rejected_count == 0
    assert consolidated.rejected_count == 3
    assert consolidated.assignment == winner.assignment
    assert consolidated.latency_us == winner.latency_us
    assert consolidated.candidate_index == winner.candidate_index


@pytest.mark.parametrize("count", (-1, True, 1.5, "2"))
def test_tuning_contribution_rejects_invalid_rejection_counts(count):
    from b12x.preparation import TuningRequirement

    with pytest.raises(ValueError, match="rejected candidate count"):
        TuningRequirement("query", (0, 1), None, None, None, rejected_count=count)


@pytest.mark.parametrize("boundary", ("compile", "collective", "tuning", "cache"))
def test_batched_gpu_steps_stop_before_unready_work(tmp_path, boundary):
    from b12x.preparation.types import PreparationResult, TuningCacheRequirement, TuningRequirement

    seen = []
    collective = CollectiveRequirement("ready/collective", (0, 1))
    tuning = TuningRequirement("ready/tuning", (0, 1), {"width": 2}, 1.0, 0)
    cache = TuningCacheRequirement((0, 1), {}, {})
    signals = {"compile": "compile", "collective": collective, "tuning": tuning, "cache": cache}

    def steps():
        for index in range(2):
            seen.append(index)
            yield "gpu"
        authorization = yield signals[boundary]
        seen.append(authorization)
        yield "gpu"
        return PreparationResult(plans={})

    with session(tmp_path) as engine:
        job = engine.begin(())
        job._steps = steps()
        progress = job.advance()
        assert seen == [0, 1]
        assert not progress.done
        if boundary == "collective":
            assert progress.ready_collectives == (collective,)
            assert job.advance().ready_collectives == (collective,)
            assert seen == [0, 1]
            progress = job.advance(collective_key=collective.key)
        elif boundary == "tuning":
            assert progress.ready_tuning == (tuning,)
            assert job.advance().ready_tuning == (tuning,)
            assert seen == [0, 1]
            progress = job.advance(tuning=tuning)
        elif boundary == "cache":
            assert progress.ready_cache == cache
            assert job.advance().ready_cache == cache
            assert seen == [0, 1]
            progress = job.advance(cache=(cache, cache))
        else:
            assert progress.pending_compilation
            progress = job.advance()
        assert progress.done
        expected = tuning if boundary == "tuning" else (cache, cache) if boundary == "cache" else None
        assert seen == [0, 1, expected]
        job.result()


def test_cancelled_cache_agreement_continues_with_required_defaults(tmp_path):
    with session(tmp_path) as engine:
        engine.configure_tuning_shard(0, (0, 1))
        req = request(name="cancelled-cache", tuning=contract())
        job = engine.begin((req,))
        assert job.advance().ready_cache is not None
        engine.cancel_tuning()
        progress = job.advance(cache=())
        while not progress.done:
            progress = job.advance()
        assert req.plan.selection.source == "default"
        assert req.plan.selection.config.width == 7
        assert engine._cache.records == {}
        job.result()


def test_closed_trials_release_storage_before_the_next_batch(tmp_path, monkeypatch):
    import weakref
    from b12x.preparation import _measurement

    class TrialStorage:
        pass

    references, alive, allocation_peaks = [], [], []
    _deterministic_timer(monkeypatch)
    prepare = _measurement.prepare_race_steps

    def observe(calls, **kwargs):
        alive.append(sum(reference() is not None for reference in references))
        return (yield from prepare(calls, **kwargs))

    def benchmark(state):
        storage = TrialStorage()
        references.append(weakref.ref(storage))
        allocation_peaks.append(sum(reference() is not None for reference in references))

        def run():
            _ = storage
            return state.value

        return PreparedCall(run=run, produce=lambda: None)

    monkeypatch.setattr(_measurement, "prepare_race_steps", observe)
    with session(tmp_path, race_batch=2) as engine:
        result = engine.prepare((request(
            name="trial-lifetime", tuning=contract(values=(1, 2, 4, 8, 16, 32)),
            benchmark=benchmark,
        ),))
        assert result.benchmarked_candidates == 6
        assert result.selections["trial-lifetime"].config.width == 2
        assert alive == [2, 3, 3]
        assert allocation_peaks == [1, 2, 2, 3, 2, 3]
        assert all(reference() is None for reference in references)


@pytest.mark.parametrize("shared", [False, True])
def test_fifty_layer_bindings_are_one_preparation_request(tmp_path, monkeypatch, shared):
    from b12x.preparation.tuning import TuningContract

    _deterministic_timer(monkeypatch)
    configured, compiled, primed, closed = [], [], [], []
    configure = TuningContract.configure

    def count_configuration(self, query, **kwargs):
        configured.append(query)
        return configure(self, query, **kwargs)

    monkeypatch.setattr(TuningContract, "configure", count_configuration)
    weights = [index + 1 for index in range(50)]

    def make(index):
        def compile_jobs(config, device):
            compiled.append(config.width)
            return ()

        plan = Plan(
            contract=contract(), query=Query(3), shared=shared,
            _compile_jobs=compile_jobs,
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(value=3 * selection.config.width),
        )
        return plan.request(
            name=f"layer.{index}",
            prepare_call=lambda state: PreparedCall(
                run=lambda: primed.append((index, state.value * weights[index])),
                close=lambda: closed.append(index),
            ),
            benchmark_call=lambda state: PreparedCall(run=lambda: state.value, produce=lambda: None),
        )

    requests = tuple(make(index) for index in range(50))
    progress = []
    with session(tmp_path) as engine:
        result = engine.prepare(requests, progress=progress.append)
        assert len(configured) == 1
        assert sorted(compiled) == [1, 2, 4, 7]
        assert result.benchmarked_candidates == 3 and result.cache_hits == 0
        assert progress[-1].completed_requests == progress[-1].total_requests == 1
        assert len(result.plans) == 50
        assert len(primed) == (1 if shared else 50)
        assert len({id(item.plan.prepared) for item in requests}) == (1 if shared else 50)
        for index, item in enumerate(requests):
            assert require_prepared(item.plan, "test.arithmetic").value * weights[index] == 6 * (index + 1)
        weights[27] = -3
        assert require_prepared(requests[27].plan, "test.arithmetic").value * weights[27] == -18
        engine.release(requests[0].plan)
        extra = make(0)
        if shared:
            engine.prepare((extra,))
            assert extra.plan.prepared is requests[1].plan.prepared
            assert len(primed) == 1 and closed == []
        engine.freeze()
        ready = tuple(item for item in requests if item.plan.prepared is not None)
        engine.prepare(ready, progress=progress.append)
        assert progress[-1].total_requests == progress[-1].completed_requests == 1
    assert len(closed) == (1 if shared else 50)


def test_coalesced_collectives_authorize_each_resource_binding(tmp_path):
    calls, authorizations, progress = [], [], []
    requests = tuple(request(
        name=name, calls=calls, collective=CollectiveRequirement(name, (0, 1)),
    ) for name in ("first-channel", "second-channel"))

    def coordinate(state):
        if not state.ready_collectives:
            return None
        key = state.ready_collectives[0].key
        authorizations.append(key)
        return key

    with session(tmp_path) as engine:
        engine.prepare(requests, coordinator=coordinate, progress=progress.append)
        assert calls == [6, 6]
        assert authorizations == ["first-channel", "second-channel"]
        assert progress[-1].completed_requests == progress[-1].total_requests == 1
        assert requests[0].plan.prepared is not requests[1].plan.prepared


@pytest.mark.parametrize("dependent", (False, True))
def test_sharded_races_exchange_after_independent_work_and_release_trials(
    tmp_path, monkeypatch, dependent,
):
    _deterministic_timer(monkeypatch)
    closed = []
    with session(tmp_path) as engine:
        engine.configure_tuning_shard(0, (0, 1))
        requests = []
        for i in range(2):
            req = request(
                name=f"query-{i}", tuning=contract(values=(1, 2, 4, 8)),
                dependencies=("query-0",) if dependent and i else (),
                benchmark=lambda state: PreparedCall(
                    run=lambda: state.value, produce=lambda: None,
                    close=lambda: closed.append(state.value),
                ),
            )
            req = replace(req, plan=replace(req.plan, query=Query(i + 3)))
            requests.append(req)
        job = engine.begin(requests)
        snapshot = job.advance().ready_cache
        job.advance(cache=(snapshot, snapshot))
        exchanges = []
        tuning = None
        for _ in range(100):
            progress = job.advance(tuning=tuning)
            tuning = None
            if progress.ready_tuning:
                exchanges.append(len(progress.ready_tuning))
                assert len(closed) == sum(exchanges) * 2
                assert requests[1].plan.prepared is None
                tuning = progress.ready_tuning
            if progress.done:
                break
        else:
            pytest.fail("sharded job did not finish")
        assert exchanges == ([1, 1] if dependent else [2])
        assert progress.total_candidates == progress.measured_candidates == 4
        assert progress.global_candidate_count == 4
        assert progress.candidate_count == 2
        assert all(req.plan.selection.source == "tuned" for req in requests)
        job.result()


def test_cancelled_pending_races_prepare_defaults_without_caching(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    with session(tmp_path) as engine:
        engine.configure_tuning_shard(0, (0, 1))
        requests = tuple(
            replace(
                req := request(name=f"query-{i}", tuning=contract(), benchmark=lambda state: PreparedCall(
                    run=lambda: state.value, produce=lambda: None,
                )), plan=replace(req.plan, query=Query(i + 3)),
            )
            for i in range(2)
        )
        job = engine.begin(requests)
        snapshot = job.advance().ready_cache
        job.advance(cache=(snapshot, snapshot))
        for _ in range(100):
            progress = job.advance()
            if progress.ready_tuning:
                break
        assert len(progress.ready_tuning) == 2
        keys = [item.key for item in progress.ready_tuning]
        with pytest.raises(ValueError, match="consolidation"):
            job.advance(tuning=(progress.ready_tuning[0],))
        assert all(req.plan.prepared is None for req in requests)
        engine.cancel_tuning()
        progress = job.advance(tuning=())
        while not progress.done:
            progress = job.advance()
        assert all(req.plan.selection.source == "default" for req in requests)
        assert all(engine._cache.get(key) is None for key in keys)
        job.result()


def test_fixed_collective_without_dependents_does_not_split_race_results(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)
    with session(tmp_path) as engine:
        engine.configure_tuning_shard(0, (0, 1))
        races = []
        for i in range(2):
            req = request(name=f"race-{i}", tuning=contract(), benchmark=lambda state: PreparedCall(
                run=lambda: state.value, produce=lambda: None,
            ))
            races.append(replace(req, plan=replace(req.plan, query=Query(i + 3))))
        collective = request(name="comm", collective=CollectiveRequirement("comm", (0, 1)))
        collective = replace(collective, plan=replace(collective.plan, query=Query(99)))
        job = engine.begin((races[0], collective, races[1]))
        snapshot = job.advance().ready_cache
        job.advance(cache=(snapshot, snapshot))
        tuning = key = None
        boundaries = []
        for _ in range(100):
            progress = job.advance(tuning=tuning, collective_key=key)
            tuning = key = None
            if progress.ready_tuning:
                boundaries.append(("tuning", len(progress.ready_tuning)))
                tuning = progress.ready_tuning
            if progress.ready_collectives:
                boundaries.append(("collective", 1))
                key = progress.ready_collectives[0].key
            if progress.done:
                break
        assert progress.done
        assert boundaries == [("tuning", 2), ("collective", 1)]
        job.result()


def test_candidate_progress_counts_races_and_excludes_fixed_or_cached_choices(tmp_path, monkeypatch):
    _deterministic_timer(monkeypatch)

    def requests():
        return (
            request(name="race", tuning=contract(values=(1, 2, 4, 8, 16)),
                    benchmark=lambda state: PreparedCall(run=lambda: state.value, produce=lambda: None)),
            request(name="fixed", tuning=contract(values=(1,))),
        )

    for expected in (5, 0):
        snapshots = []
        with session(tmp_path, race_batch=2) as engine:
            result = engine.prepare(requests(), progress=snapshots.append)
        assert snapshots[-1].total_candidates == expected
        assert snapshots[-1].measured_candidates == expected
        assert result.benchmarked_candidates == expected
        assert all(p.measured_candidates <= p.total_candidates for p in snapshots if p.total_candidates is not None)


def test_first_use_warnings_group_equal_declarations_without_hiding_other_shapes(caplog):
    from b12x.preparation.session import _LAZY_SESSIONS, _warn_unprepared_declaration

    _warn_unprepared_declaration.cache_clear()
    plans = [declaration(shared=True) for _ in range(20)]
    plans.append(replace(declaration(shared=True), query=Query(5)))
    try:
        with caplog.at_level("DEBUG", logger="b12x.preparation"):
            states = [require_prepared(plan, "test.arithmetic") for plan in plans]
        assert all(state is states[0] for state in states[:20])
        assert states[-1] is not states[0]
        assert all(plan.prepared is not None for plan in plans)
        messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(messages) == 2
        assert "rows=3" in messages[0] and "rows=5" in messages[1]
        details = [r.getMessage() for r in caplog.records if "unprepared plan Query#" in r.getMessage()]
        assert len(details) == 21
    finally:
        # Lazy sessions are keyed by device ordinal: None without CUDA, else the GPU.
        from b12x.preparation.device import detect_device
        session = _LAZY_SESSIONS[detect_device(None).ordinal]
        for plan in plans:
            session.release(plan)
        _warn_unprepared_declaration.cache_clear()


def test_progress_counts_prepared_selection_sources_once_per_shared_group(tmp_path):
    fixed = request(name="fixed", shared=True)
    alias = request(name="alias", shared=True)
    pinned = replace(request(name="pinned"), plan=declaration(pin=Config(9)))

    def finish(engine, requests):
        job = engine.begin(requests)
        try:
            while True:
                progress = job.advance()
                if progress.done:
                    break
            assert sum(dict(progress.selection_counts).values()) == progress.completed_requests
            job.result().close()
            return dict(progress.selection_counts)
        finally:
            job.close()

    with session(tmp_path) as engine:
        assert finish(engine, (fixed, alias, pinned)) == {"fixed": 1, "override": 1}
        engine.cancel_tuning()
        default = replace(request(name="default"), plan=replace(declaration(), query=Query(5)))
        assert finish(engine, (fixed, default)) == {"fixed": 1, "default": 1}
        engine.freeze()
        assert finish(engine, (fixed, alias, pinned, default)) == {
            "fixed": 1, "override": 1, "default": 1,
        }


def test_missing_wo_warning_reports_shape_and_geometry_without_codegen_dump(caplog):
    from b12x.gemm import wo_projection
    from b12x.preparation import FrozenMapping
    from b12x.preparation.session import _prepare_default, _warn_unprepared_declaration
    from unittest.mock import patch

    plan = wo_projection.plan(
        wo_projection.Caps(
            device="cuda:0", max_tokens=1797, groups=2,
            group_width=4096, rank=1024, hidden=5120,
        ),
        invocation=FrozenMapping({
            "operation": "inv_rope", "heads_per_group": 8, "nope_dim": 448, "rope_dim": 64,
            "positions_dtype": "int64", "cos_sin_dtype": "bfloat16",
        }),
    )
    _warn_unprepared_declaration.cache_clear()
    try:
        with patch("b12x.preparation.session.prepare_default"), caplog.at_level("WARNING"):
            _prepare_default(plan)
        (message,) = [record.getMessage() for record in caplog.records]
        for detail in (
            "gemm.wo_projection", "max_tokens=1797", "operation=inv_rope",
            "dtype=bfloat16", "groups=2", "group_width=4096", "rank=1024", "hidden=5120",
        ):
            assert detail in message
        assert "codegen" not in message and "positions_dtype" not in message
    finally:
        _warn_unprepared_declaration.cache_clear()


def test_missing_plan_warning_uses_component_query_fields(caplog):
    from dataclasses import dataclass
    from unittest.mock import patch
    from b12x.preparation.session import _prepare_default, _warn_unprepared_declaration

    @dataclass(frozen=True)
    class ShapeQuery:
        image_shape: tuple[int, ...]
        window_width: int
        max_rows: int

    tuning = replace(
        contract(), component_id="test.image", query_fields=frozenset(ShapeQuery.__dataclass_fields__),
        encode_query=lambda query: vars(query),
    )
    plan = replace(declaration(), contract=tuning, query=ShapeQuery((3, 16, 16), 7, 19))
    _warn_unprepared_declaration.cache_clear()
    try:
        with patch("b12x.preparation.session.prepare_default"), caplog.at_level("WARNING"):
            _prepare_default(plan)
        (message,) = [record.getMessage() for record in caplog.records]
        assert "max_rows=19, image_shape=(3, 16, 16), window_width=7" in message
    finally:
        _warn_unprepared_declaration.cache_clear()
