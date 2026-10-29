# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest
import torch
from flashinfer.autotuner import AutoTuner, TuningConfig
from flashinfer.autotune_cache import MeasurementPolicy


def test_policy_route(monkeypatch):
    spec = spec_from_file_location(
        "accuracy_harness",
        Path(__file__).resolve().parents[2] / "benchmarks/bench_autotuner_accuracy.py",
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    seen = []
    monkeypatch.setattr(
        tuner, "_prepare_input_tensors_with_batches", lambda inputs, config: [inputs]
    )
    monkeypatch.setattr(
        tuner,
        "_profile_single_kernel",
        lambda runner, inputs, tactic, config, **kwargs: seen.append(
            (
                config.use_cuda_graph,
                config.use_cold_l2_cache,
                tuner.repeat,
                tuner._effective_measure_policy,
            )
        )
        or 1.0,
    )
    for mode in ("eager", "cuda_graph"):
        policy = MeasurementPolicy(execution_mode=mode, cold_l2=False)
        assert (
            module.measure_candidate(
                tuner, object(), [torch.ones(2)], 0, TuningConfig(), {}, policy, 7
            )
            == 1.0
        )
        assert seen[-1] == (mode == "cuda_graph", False, 7, policy)
        assert tuner._effective_measure_policy is None
        assert not tuner.is_tuning_mode


def _harness():
    spec = spec_from_file_location(
        "accuracy_harness",
        Path(__file__).resolve().parents[2] / "benchmarks/bench_autotuner_accuracy.py",
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retained_inputs_require_matching_metadata():
    module = _harness()
    inputs = [torch.arange(4.0)]
    config = module.retained_input_config(TuningConfig(), inputs)
    assert config.inputs_pre_hook([torch.zeros(4)])[0] is inputs[0]
    with pytest.raises(ValueError, match="metadata"):
        config.inputs_pre_hook([torch.zeros(8)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("profile_inputs", ["retained", "synthesized"])
def test_graph_study_real_selection_and_numerics(monkeypatch, profile_inputs):
    from types import SimpleNamespace
    from flashinfer.autotuner import DynamicTensorSpec, TunableRunner

    module = _harness()
    original = AutoTuner._instance

    class Runner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0, 1]

        def forward(self, inputs, tactic=-1, **kwargs):
            value = inputs[0].clone() if tactic == 1 else inputs[0]
            return value + 1

    args = SimpleNamespace(
        cold_l2=False,
        profile_inputs=profile_inputs,
        seeds=1,
        graph_policies=["auto", "eager", "cuda_graph"],
        seed=42,
        warmup=2,
        tuner_repeat=3,
        oracle_rounds=2,
        validation_replays=3,
        prequeue_cycles=1_000_000,
        rtol=0,
        atol=0,
    )
    report = module.graph_selection_study(
        "test.accuracy",
        [Runner()],
        TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=(0,),
                    dim_idx=(0,),
                    gen_tuning_buckets=(1, 3),
                    map_to_tuning_buckets=lambda n: 1 if n == 1 else 4,
                ),
            ),
        ),
        [torch.ones(3, device="cuda")],
        {},
        args,
    )
    assert AutoTuner._instance is original
    assert len(report["trials"]) == 3
    assert report["validation"]["output_refresh"] == "unavailable"
    for row in report["trials"]:
        assert row["cache_hit"]
        assert row["profile_count"] == (3 if profile_inputs == "retained" else 6)
        assert row["input_shapes"] == ((3,),)
        assert row["cache_profile"] == ((3 if profile_inputs == "retained" else 4,),)
        assert {m["input_shapes"] for m in row["measurements"]} == (
            {((3,),)} if profile_inputs == "retained" else {((1,),), ((3,),)}
        )
        assert {m["cache_profile"] for m in row["measurements"]} == (
            {((3,),)} if profile_inputs == "retained" else {((1,),), ((4,),)}
        )
        assert row["numerics"] == "passed"
        assert all(
            m["use_cuda_graph"] == (row["policy"] == "cuda_graph")
            for m in row["measurements"]
        )
        assert all(len(values) == 2 for values in row["validation_ms"].values())


def _study_args(**overrides):
    from types import SimpleNamespace

    values = dict(
        cold_l2=False,
        profile_inputs="retained",
        seeds=1,
        graph_policies=["cuda_graph"],
        seed=42,
        warmup=1,
        tuner_repeat=2,
        oracle_rounds=1,
        validation_replays=1,
        prequeue_cycles=1_000_000,
        rtol=0,
        atol=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_graph_study_rejects_all_failed_tactics(monkeypatch):
    from .utils import DummyRunner

    module = _harness()
    original = AutoTuner._instance
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        AutoTuner, "_profile_single_kernel", lambda *a, **kw: float("inf")
    )
    monkeypatch.setattr(
        module, "capture_tactic_graph", lambda runner, inputs, *a: (None, inputs[0])
    )
    with pytest.raises(RuntimeError, match="no cached winner"):
        module.graph_selection_study(
            "test.failed",
            [DummyRunner()],
            TuningConfig(),
            [torch.ones(4)],
            {},
            _study_args(),
        )
    assert AutoTuner._instance is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("inference_output", [False, True])
@pytest.mark.parametrize("kind", ["correct", "wrong", "no_write"])
def test_graph_study_validates_shared_output(monkeypatch, inference_output, kind):
    from contextlib import nullcontext

    from flashinfer.autotuner import TunableRunner

    module = _harness()
    original = AutoTuner._instance
    with torch.inference_mode(inference_output):
        output = torch.empty(4, device="cuda")
        unrelated = torch.empty_like(output)
    refresh = module.build_bmm_fp8(1, 4, 4).prepare_validation
    refreshes = []

    def prepare(output, inputs):
        assert not torch.cuda.is_current_stream_capturing()
        refreshes.append(output.data_ptr())
        refresh(output, inputs)

    class Runner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0]

        def forward(self, inputs, tactic=-1, **kwargs):
            with torch.inference_mode():
                torch.add(
                    inputs[0],
                    2 if tactic == 0 and kind == "wrong" else 1,
                    out=unrelated if tactic == 0 and kind == "no_write" else output,
                )
            return output

    monkeypatch.setattr(
        AutoTuner,
        "_profile_single_kernel",
        lambda self, runner, inputs, tactic, config, **kw: 1.0 if tactic == 0 else 2.0,
    )
    with nullcontext() if kind == "correct" else pytest.raises(AssertionError):
        report = module.graph_selection_study(
            "test.shared_output",
            [Runner()],
            TuningConfig(),
            [torch.ones(4, device="cuda")],
            {},
            _study_args(),
            prepare_validation=prepare,
        )
        assert report["trials"][0]["cache_hit"]
        assert report["trials"][0]["tactic"] == "0"
        assert report["validation"]["output_refresh"] == "fixture_callback"
    assert refreshes == [output.data_ptr()] * (3 if kind == "correct" else 2)
    assert AutoTuner._instance is original
    assert not torch.is_inference_mode_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_study_rejects_invalidated_retained_control(monkeypatch):
    from flashinfer.autotuner import TunableRunner

    module = _harness()

    class Runner(TunableRunner):
        def __init__(self):
            self.bias = torch.ones(4, device="cuda")
            self.output = torch.empty_like(self.bias)
            self.prepared = False

        def __hash__(self):
            return hash(type(self))

        def get_valid_tactics(self, inputs, profile):
            if not self.prepared:
                # Preparation reuses storage held by the old graph. Later
                # graphs bind the new bias and still produce correct output.
                self.bias.zero_()
                self.bias = torch.ones_like(self.bias)
                self.prepared = True
            return [0]

        def forward(self, inputs, tactic=-1, **kwargs):
            return torch.add(inputs[0], self.bias, out=self.output)

    monkeypatch.setattr(
        AutoTuner,
        "_profile_single_kernel",
        lambda self, runner, inputs, tactic, config, **kw: 1.0 if tactic == 0 else 2.0,
    )
    with pytest.raises(AssertionError):
        module.graph_selection_study(
            "test.invalidated_control",
            [Runner()],
            TuningConfig(),
            [torch.ones(4, device="cuda")],
            {},
            _study_args(),
        )


def test_unavailable_cupti_cannot_be_scored_as_an_oracle(monkeypatch):
    import importlib

    implementation = importlib.import_module("flashinfer.autotuner.autotuner")
    module = _harness()
    tuner = AutoTuner()
    monkeypatch.setattr(implementation, "_load_cupti", lambda: None)
    with pytest.raises(module.MeasurementUnavailable, match="CUPTI oracle unavailable"):
        module.measure_candidate(
            tuner,
            object(),
            [torch.ones(2)],
            0,
            TuningConfig(),
            {},
            MeasurementPolicy(_timer="cupti"),
            3,
        )


def test_exhaustive_report_preserves_unavailable_and_valid_shapes(
    monkeypatch, tmp_path
):
    import json
    import sys
    from types import SimpleNamespace

    module = _harness()
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["study", "--ops", "bmm_fp8", "--grid", "1", "2", "--out", str(output)],
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="test", major=9, minor=0),
    )
    monkeypatch.setitem(module.OPS, "bmm_fp8", {"axis": "m", "build": lambda *a: None})
    monkeypatch.setattr(
        module, "capture_choose_one_calls", lambda _: {"op": ([], None, [], {})}
    )
    visited = []

    def sweep(*args):
        visited.append(args[-1])
        if len(visited) == 1:
            raise module.MeasurementUnavailable("no finite oracle candidates")
        return {"op": "op", "valid": 1}

    monkeypatch.setattr(module, "sweep_captured_op", sweep)
    module.main()
    rows = json.loads(output.read_text())["shapes"]
    assert visited == ["bmm_fp8 m=1", "bmm_fp8 m=2"]
    assert rows[0] == {
        "api": "bmm_fp8",
        "op": "op",
        "m": 1,
        "error": "MeasurementUnavailable: no finite oracle candidates",
    }
    assert rows[1] == {"api": "bmm_fp8", "op": "op", "m": 2, "valid": 1}
