# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for benchmark provider evidence, independent of FE engines."""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


_BENCHMARK = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "bench_cudnn_linear_attention.py"
    )
)
observe_route = _BENCHMARK["observe_route"]
check_route = _BENCHMARK["check_route"]


@pytest.fixture
def adapter(monkeypatch):
    adapter = SimpleNamespace(_run_la_graph=lambda: None)
    importer = SimpleNamespace(import_module=lambda name: adapter)
    monkeypatch.setitem(observe_route.__globals__, "importlib", importer)
    return adapter


def test_route_observes_success_without_frontend_private_frames(adapter):
    original = adapter._run_la_graph
    previous_profile = sys.getprofile()
    result, route = observe_route(lambda: adapter._run_la_graph())
    assert result is None  # A graph can execute without returning a final state.
    assert route["cudnn_executions"] == 1
    assert not route["frontend_plans"]
    check_route("cudnn", route)
    assert adapter._run_la_graph is original
    assert sys.getprofile() is previous_profile


def test_build_decline_followed_by_native_fallback_is_not_cudnn(adapter):
    def decline():
        raise NotImplementedError("unsupported build")

    adapter._run_la_graph = decline

    def auto():
        try:
            adapter._run_la_graph()
        except NotImplementedError:
            return "native result"

    result, route = observe_route(auto)
    assert result == "native result"
    assert route["cudnn_executions"] == 0
    check_route("auto", route)
    with pytest.raises(AssertionError, match="no successful execution"):
        check_route("cudnn", route)
    assert adapter._run_la_graph is decline


def test_execute_error_propagates_and_restores_observers(adapter):
    error = RuntimeError("execution failed")

    def fail():
        raise error

    adapter._run_la_graph = fail
    previous_profile = sys.getprofile()
    with pytest.raises(RuntimeError) as caught:
        observe_route(lambda: adapter._run_la_graph())
    assert caught.value is error
    assert adapter._run_la_graph is fail
    assert sys.getprofile() is previous_profile


@pytest.mark.parametrize("backend", ["native-auto", "cake_gdn"])
def test_native_controls_reject_cudnn_execution(backend):
    with pytest.raises(AssertionError, match="unexpectedly executed cuDNN"):
        check_route(backend, dict(cudnn_executions=1, frontend_plans=[]))


def test_frontend_plan_cannot_hide_missing_adapter_observation():
    with pytest.raises(AssertionError, match="bypassed"):
        check_route("auto", dict(cudnn_executions=0, frontend_plans=[{}]))


@pytest.mark.parametrize("eager", [0, 1])
@pytest.mark.parametrize("captured", [0, 1])
def test_eager_and_capture_must_execute_the_same_provider(eager, captured):
    # Missing optional plan details must not make distinct routes look equal.
    route = dict(cudnn_executions=captured, frontend_plans=[])
    eager_route = dict(cudnn_executions=eager, frontend_plans=[])
    if eager == captured:
        check_route("auto", route, eager_route=eager_route)
    else:
        with pytest.raises(AssertionError, match="different providers"):
            check_route("auto", route, eager_route=eager_route)
