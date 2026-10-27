# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""CPU contracts for optional process-based MLA prebuilding."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_helpers import prims_ts_mla_prebuild as prebuild


def test_prebuild_selection_preserves_pytest_skip_and_fixture_ownership():
    """Workers must not execute skipped tests or invent fixture arguments."""

    def parameterized(value, optional=True):
        """Stand-in for a fixture-free numerical test."""

    def with_fixture(monkeypatch):
        """Stand-in for a test whose setup belongs to pytest."""

    def item(function, parameters, skipif=()):
        """Supply the pytest item metadata consumed by the prebuilder."""
        return SimpleNamespace(
            function=function,
            path=Path("test_mla.py"),
            originalname=function.__name__,
            callspec=SimpleNamespace(params=parameters),
            get_closest_marker=lambda name: None,
            iter_markers=lambda name: iter(skipif),
        )

    selected = item(parameterized, {"value": 7})
    skipped = item(
        parameterized,
        {"value": 8},
        (SimpleNamespace(args=(True,), kwargs={}),),
    )
    fixture = item(with_fixture, {})
    assert prebuild.selected_cases([selected, skipped, fixture]) == [
        ("test_mla.py", "parameterized", {"value": 7})
    ]


def test_prebuild_respects_worker_and_host_budgets(monkeypatch):
    """A host prebuild allowance cannot override a smaller per-worker limit."""
    monkeypatch.setattr(prebuild.os, "sched_getaffinity", lambda pid: set(range(64)))
    monkeypatch.setenv("MAX_JOBS", "3")
    monkeypatch.setenv("FLASHINFER_JIT_PREBUILD_MAX_JOBS", "8")
    assert prebuild.prebuild_workers() == 3
    monkeypatch.setenv("FLASHINFER_JIT_PREBUILD_MAX_JOBS", "2")
    assert prebuild.prebuild_workers() == 2
    monkeypatch.setattr(prebuild.os, "sched_getaffinity", lambda pid: {0})
    assert prebuild.prebuild_workers() == 1


@pytest.mark.parametrize("disabled", ["cache", "jit"])
def test_prebuild_preserves_disabled_jit_modes(monkeypatch, disabled):
    """No prebuild worker may bypass a caller's cache/JIT opt-out."""
    from flashinfer.jit import cute_dsl_core

    monkeypatch.setattr(
        cute_dsl_core, "cute_dsl_cache_disabled", lambda: disabled == "cache"
    )
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    if disabled == "jit":
        monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")

    def unexpected_selection(items):
        """Fail if disabled prebuilding reaches case selection."""
        pytest.fail("disabled prebuilding selected work")

    monkeypatch.setattr(prebuild, "selected_cases", unexpected_selection)
    assert prebuild.prebuild_selected_mla_tests([object()]) == (0, [], 0)


def test_prebuild_defers_test_failures_but_preserves_interrupts(monkeypatch):
    """A setup failure belongs to normal pytest, while cancellation propagates."""

    def failure():
        """Produce pytest's BaseException-derived failure outcome."""
        pytest.fail("normal test must report this")

    def interrupt():
        """Simulate cancellation during worker setup."""
        raise KeyboardInterrupt

    monkeypatch.setattr(
        prebuild,
        "load_test_module",
        lambda path: SimpleNamespace(failure=failure, interrupt=interrupt),
    )
    count, error = prebuild.precompile_case(("unused", "failure", {}))
    assert count == 0
    assert "normal test must report this" in error
    with pytest.raises(KeyboardInterrupt):
        prebuild.precompile_case(("unused", "interrupt", {}))


def test_validation_only_mla_schedule_keeps_exhaustive_checking(monkeypatch):
    """The JIT-time optimization must preserve explicit offline verification."""
    pytest.importorskip("cutlass", minversion="4.7.0")
    from cutlass.experimental.task_scheduling.task_manager import TaskManager
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.config import (
        make_mla_decode_config,
    )
    from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
        build_mla_decode_task_manager,
    )

    checked = []

    def record_check(manager):
        """Observe the requested exhaustive check without enumerating its states."""
        checked.append(manager)

    monkeypatch.setattr(TaskManager, "_exhaustive_aliasing_races", record_check)
    cfg = make_mla_decode_config()
    build_mla_decode_task_manager(cfg, domain=1)
    assert len(checked) == 1
    build_mla_decode_task_manager(cfg, domain=1, exhaustive_deadlock_race_check=False)
    assert len(checked) == 1


def test_prebuild_limits_child_thread_pools_and_restores_environment(monkeypatch):
    """Spawned compilers get one BLAS/OMP thread without changing parent settings."""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")

    class Pool:
        """Observe the environment at pool construction and task submission."""

        def __init__(self, **kwargs):
            """Check limits before any worker could import numerical libraries."""
            assert prebuild.os.environ["OPENBLAS_NUM_THREADS"] == "1"
            assert prebuild.os.environ["OMP_NUM_THREADS"] == "1"

        def __enter__(self):
            """Match the executor context protocol."""
            return self

        def __exit__(self, *args):
            """No subprocesses are created by this contract test."""

        def map(self, function, cases):
            """Simulate two completed warmup cases."""
            return [(1, None) for _ in cases]

    monkeypatch.setattr(prebuild, "ProcessPoolExecutor", Pool)
    assert prebuild.precompile_cases([object(), object()], 2) == (2, [])
    assert prebuild.os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert prebuild.os.environ["OMP_NUM_THREADS"] == "3"
