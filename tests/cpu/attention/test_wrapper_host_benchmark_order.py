# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for benchmark scheduling and resolved route reporting."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def scheduler():
    path = (
        Path(__file__).resolve().parents[3] / "benchmarks/bench_cudnn_wrapper_host.py"
    )
    module = ast.parse(path.read_text(), str(path))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_interleave_measurements"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


def test_all_backends_prepare_before_sampling_and_keep_separate_state():
    events, order, samples = [], [], {"a": [], "b": []}

    def backend(name):
        events.append(("prepare", name))
        for phase in ("host", "graph"):
            events.append(("ready", phase, name))

            def sample():
                assert ("ready", phase, "a") in events
                assert ("ready", phase, "b") in events
                samples[name].append(phase)
                events.append(("sample", phase, name))

            yield phase, 4, sample
        events.append(("complete", name))

    scheduler()([("a", backend("a")), ("b", backend("b"))], order)
    assert samples == {name: ["host"] * 4 + ["graph"] * 4 for name in ("a", "b")}
    assert [entry["backends"] for entry in order] == [["a", "b"], ["b", "a"]] * 4
    assert events.index(("ready", "graph", "a")) > max(
        index for index, event in enumerate(events) if event[:2] == ("sample", "host")
    )


def test_failed_sample_is_recorded_without_shortening_other_backend():
    order, errors, successful = [], [], []

    def backend(name):
        calls = 0
        try:

            def sample():
                nonlocal calls
                calls += 1
                if name == "bad" and calls == 2:
                    raise ValueError("numerical failure")
                if name == "good":
                    successful.append(calls)

            yield "host", 4, sample
        except ValueError as exc:
            errors.append((name, str(exc)))

    scheduler()([("bad", backend("bad")), ("good", backend("good"))], order)
    assert errors == [("bad", "numerical failure")]
    assert successful == [1, 2, 3, 4]
    assert [entry["backends"] for entry in order] == [
        ["bad", "good"],
        ["good", "bad"],
        ["good"],
        ["good"],
    ]


def test_mismatched_metric_phases_are_not_compared():
    def backend(phase):
        yield phase, 2, lambda: pytest.fail("mismatched phase was measured")

    with pytest.raises(RuntimeError, match="phases differ"):
        scheduler()([("a", backend("host")), ("b", backend("graph"))], [])


def reported_backend(wrapper, requested):
    path = (
        Path(__file__).resolve().parents[3] / "benchmarks/bench_cudnn_wrapper_host.py"
    )
    module = ast.parse(path.read_text(), str(path))
    assignments = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "row"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "actual_backend"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    return eval(
        compile(ast.Expression(assignments[0].value), str(path), "eval"),
        {"w": wrapper, "backend": requested},
    )


@pytest.mark.parametrize(
    "backend,cudnn_auto,tensor_cores,expected",
    [
        ("auto", True, False, "cudnn"),
        ("auto", False, False, "fa2"),
        ("fa3", False, True, "fa3"),
        ("cudnn", False, False, "cudnn"),
    ],
)
def test_decode_report_uses_real_resolved_property(
    backend, cudnn_auto, tensor_cores, expected
):
    import torch

    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    wrapper = BatchDecodeWithPagedKVCacheWrapper.__new__(
        BatchDecodeWithPagedKVCacheWrapper
    )
    wrapper._batch_invariant = None
    wrapper._backend = backend
    wrapper._cudnn_auto = cudnn_auto
    wrapper._use_tensor_cores = tensor_cores
    assert reported_backend(wrapper, "auto") == expected
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "wrapper,requested,expected",
    [(SimpleNamespace(_backend="cudnn"), "auto", "cudnn"), (object(), "fa2", "fa2")],
)
def test_prefill_report_and_requested_fallback_are_preserved(
    wrapper, requested, expected
):
    assert reported_backend(wrapper, requested) == expected
