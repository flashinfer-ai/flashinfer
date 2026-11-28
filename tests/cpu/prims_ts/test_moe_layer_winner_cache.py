# Copyright (c) 2026 by FlashInfer team. Licensed under the Apache License, Version 2.0.
"""Layer winner reuse must follow the real Prims-TS tactic menu."""

from collections import OrderedDict
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import MoELayer, MoEWeightPack, PrimsTsRunner
from flashinfer.fused_moe.api import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    PrimsTsConfig,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
)
from flashinfer.fused_moe.runners import _TrtllmLaunchState, _TrtllmPackedInputs
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
from flashinfer.tllm_enums import WeightLayout


@pytest.fixture(autouse=True)
def mock_compute_capability(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.fused_moe.runners.get_compute_capability", lambda device: (10, 7)
    )


@pytest.mark.cpu_only
def test_prims_ts_tactic_cache_shares_only_with_same_architecture(monkeypatch):
    capabilities = ((10, 0), (10, 3), (10, 7), (10, 7))
    monkeypatch.setattr(
        "flashinfer.fused_moe.runners.get_compute_capability",
        lambda device: capabilities[device.index],
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=32, top_k=2),
        quant=QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=512),
        backend=BackendOptions((PrimsTsConfig(),)),
        execution=ExecutionConfig(enable_pdl=False),
    )
    runners = [PrimsTsRunner(config, torch.device("cuda", i)) for i in range(4)]
    keys = [runner.get_cache_key_extras([]) for runner in runners]
    assert len(set(keys)) == 3
    assert keys[2] == keys[3]
    assert hash(runners[2]) == hash(runners[3])
    assert runners[0]._cache_key_extras()[:-1] == runners[1]._cache_key_extras()[:-1]


@pytest.mark.cpu_only
@pytest.mark.parametrize("minor", (0, 3, 7))
def test_layer_reselects_for_the_prepared_weight_layout(monkeypatch, minor):
    monkeypatch.setattr(
        "flashinfer.fused_moe.runners.get_compute_capability", lambda _: (10, minor)
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=8, top_k=2),
        quant=QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=512),
        backend=BackendOptions((PrimsTsConfig(),)),
        execution=ExecutionConfig(enable_pdl=False),
    )
    runner = PrimsTsRunner(config, torch.device("cpu"))
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.runners, layer._winners = config, [runner], OrderedDict()
    act = MoEActivationPack(
        hidden_states_q=torch.empty(1, 512, device="meta", dtype=torch.uint8),
        hidden_states_scale=torch.empty(1, 64, device="meta", dtype=torch.uint8),
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=torch.empty(1, 8, device="meta"),
    )
    monkeypatch.setattr(runner, "pack_inputs", lambda act, weights: [])
    monkeypatch.setattr(runner, "forward", lambda inputs, tactic, **kw: tactic)
    monkeypatch.setattr(
        layer,
        "_select_winner",
        lambda act, weights, runners: (
            runner,
            runner._weight_layout(weights.get_view("prims_ts")),
        ),
    )
    for block in (False, True, False):
        shape = (8, 4, 1024, 128) if block else (8, 1024, 512)
        weights = MoEWeightPack(
            {
                "prims_ts": {
                    "gemm1_weights": torch.empty(
                        shape, device="meta", dtype=torch.uint8
                    )
                }
            }
        )
        assert layer(act, weights) == (
            WeightLayout.BlockMajorK if block else WeightLayout.MajorK
        )
    assert len(layer._winners) == 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("transition", ["preserve_rows", "deterministic"])
@pytest.mark.parametrize(
    "finalize",
    [
        MoEFinalizeConfig(),
        MoEFinalizeConfig(use_fused_finalize=False),
        MoEFinalizeConfig(do_finalize=False),
    ],
    ids=["fused", "unfused", "unfinalized"],
)
def test_layer_winner_cache_tracks_effective_finalize(
    monkeypatch, transition, finalize
):
    from flashinfer.prims_ts.moe import runner as prims_runner

    monkeypatch.setattr(prims_runner, "_sm_arch_for_tensor", lambda _: 107)
    deterministic = False
    monkeypatch.setattr(
        torch, "are_deterministic_algorithms_enabled", lambda: deterministic
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=32, top_k=2),
        quant=QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=512),
        backend=BackendOptions((PrimsTsConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=8192, enable_pdl=False),
        finalize=finalize,
    )
    runner = PrimsTsRunner(config, torch.device("cpu"))
    monkeypatch.setattr(runner, "_require_built", lambda: None)
    inner = prims_runner.PrimsTsNvfp4MoERunner(
        None,
        top_k=2,
        num_local_experts=32,
        hidden_size=1024,
        intermediate_size=512,
        use_fused_finalize=finalize.use_fused_finalize,
    )
    act = MoEActivationPack(
        hidden_states_q=torch.empty(8192, 512, device="meta", dtype=torch.uint8),
        hidden_states_scale=torch.empty(8192, 64, device="meta", dtype=torch.uint8),
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=torch.empty(8192, 32, device="meta"),
        output=torch.empty(8192, 1024, device="meta", dtype=torch.bfloat16),
    )
    weights = MoEWeightPack(
        {
            "prims_ts": {
                "gemm1_weights": torch.empty(
                    (32, 1024, 512), device="meta", dtype=torch.uint8
                )
            }
        }
    )
    packed_calls, selected, launched = [], [], []

    def pack(act, weights):
        # Preserve lazy inner-runner initialization and per-call launch metadata,
        # but avoid GPU allocation/preparation in this host cache regression.
        runner._inner = inner
        packed_calls.append(act)
        tensors = MoeRunnerInputs(
            act.output,
            act.routing_logits,
            None,
            None,
            act.hidden_states_q,
            act.hidden_states_scale,
            None,
            None,
        ).to_list()
        return _TrtllmPackedInputs(
            tensors,
            tuning_config=object(),
            launch_state=_TrtllmLaunchState(
                dict(
                    weight_layout=int(WeightLayout.MajorK),
                    do_finalize=finalize.do_finalize,
                    zero_rows_without_local_expert=act.zero_rows_without_local_expert,
                )
            ),
        )

    def choose(*, runners, inputs, **kwargs):
        assert runners == [runner]
        choices = runner.get_valid_tactics(inputs, None)
        tactic = [512, 14] if [512, 14] in choices else [512, 0]
        assert tactic in choices
        selected.append(tactic)
        return runner, tactic

    def forward(inputs, tactic, **kwargs):
        assert tactic in runner.get_valid_tactics(inputs, None), tactic
        launched.append(tactic)
        return inputs[0]

    monkeypatch.setattr(runner, "pack_inputs", pack)
    monkeypatch.setattr(runner, "forward", forward)
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.runners = config, [runner]
    layer._winners = OrderedDict()
    layer.tuner = SimpleNamespace(choose_one=choose)

    fused = finalize.do_finalize and finalize.use_fused_finalize
    initial = [512, 14] if fused else [512, 0]
    assert layer(act, weights) is act.output
    assert launched[-1] == initial
    layer(act, weights)
    assert selected == [initial]

    preserved = replace(act, zero_rows_without_local_expert=False)
    deterministic = transition == "deterministic"
    changed = act if deterministic else preserved
    layer(changed, weights)
    layer(changed, weights)
    assert launched[-2:] == [[512, 0], [512, 0]]
    expected = [initial, [512, 0]] if fused else [initial]
    assert selected == expected

    # Different reasons for the same effective mode must retain cache hits.
    deterministic = True
    layer(preserved, weights)
    deterministic = False
    layer(preserved, weights)
    layer(act, weights)
    assert launched[-1] == initial
    assert selected == expected
    assert len(layer._winners) == (2 if fused else 1)
    assert len(packed_calls) == len(launched) + len(selected)


@pytest.mark.cpu_only
@pytest.mark.parametrize("operation", ["choose", "rank"])
@pytest.mark.parametrize("fused", [False, True])
def test_profile_cache_keeps_candidate_options_across_interleaved_forward(
    monkeypatch, operation, fused
):
    from flashinfer.autotuner import AutoTuner, TuningConfig, autotune
    from flashinfer.prims_ts.moe.epilogue_policy import fused_finalize_requested

    monkeypatch.setattr(AutoTuner, "_instance", None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    monkeypatch.setattr(
        "flashinfer.autotuner._collect_metadata", lambda: {"gpu": "test"}
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=32, top_k=2),
        quant=QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=512),
        backend=BackendOptions((PrimsTsConfig(),)),
        execution=ExecutionConfig(enable_pdl=False),
    )
    runner = PrimsTsRunner(config, torch.device("cpu"))
    inner = SimpleNamespace(_cache_key_static_extras=())
    inner.set_cache_key_static_extras = lambda **kw: setattr(
        inner, "_cache_key_static_extras", tuple(kw.items())
    )

    def valid_tactics(*args):
        assert inner.weight_layout == WeightLayout.BlockMajorK
        return [[512, 14 if fused_finalize_requested(inner) else 0]]

    inner.get_valid_tactics = valid_tactics
    runner._inner = inner
    monkeypatch.setattr(runner, "_require_built", lambda: None)
    monkeypatch.setattr(runner, "_forward_inner", lambda *args: None)
    monkeypatch.setattr(runner, "precompile_tactics", lambda *args, **kwargs: False)
    state = _TrtllmLaunchState(
        dict(
            weight_layout=int(WeightLayout.BlockMajorK),
            zero_rows_without_local_expert=fused,
        )
    )
    # Also interleave after the hook binds A but before tactic enumeration.
    tuning = runner._bind_tuning_config(
        TuningConfig(
            inputs_pre_hook=lambda tensors: (runner.forward(other), tensors)[1]
        ),
        state,
    )
    inputs = _TrtllmPackedInputs(
        [torch.zeros(4, 8)], tuning_config=tuning, launch_state=state
    )
    other = inputs.with_launch_overrides(
        weight_layout=int(WeightLayout.MajorK), zero_rows_without_local_expert=not fused
    )
    tuner = AutoTuner.get()
    monkeypatch.setattr(
        tuner, "_prepare_input_tensors", lambda profile, inputs: list(inputs)
    )
    expected = [512, 14 if fused else 0]
    profiled = []

    def profile(runner, tensors, tactic, *args, **kwargs):
        assert tensors.launch_state is state
        assert tactic == expected
        runner.forward(other)  # Mutates inner static extras before the cache write.
        profiled.append(tactic)
        return 1.0

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    with autotune(True):
        for _ in range(2):
            if operation == "choose":
                assert tuner.choose_one(
                    "prims_profile_options",
                    [runner],
                    tuning,
                    inputs,
                    launch_state=state,
                ) == (runner, expected)
            else:
                assert tuner.rank_tactics(
                    "prims_profile_options",
                    [runner],
                    tuning,
                    inputs,
                    k=2,
                    launch_state=state,
                ) == [expected]
    assert profiled == [expected]  # Second call must hit the original candidate's key.
    assert runner.get_cache_key_extras(inputs) != runner.get_cache_key_extras(other)


@pytest.fixture
def packed_nvfp4_layer(monkeypatch):
    from flashinfer.autotuner import TuningConfig
    from flashinfer.prims_ts.moe import runner as impl

    def make(minor):
        monkeypatch.setattr(
            "flashinfer.fused_moe.runners.get_compute_capability", lambda _: (10, minor)
        )
        monkeypatch.setattr(impl, "_sm_arch_for_tensor", lambda _: 100 + minor)
        config = MoEConfig(
            routing=RoutingConfig(num_experts=8, top_k=2),
            quant=QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4),
            experts=ExpertConfig(intermediate_size=512),
            backend=BackendOptions((PrimsTsConfig(),)),
            execution=ExecutionConfig(enable_pdl=False),
        )
        runner = PrimsTsRunner(config, torch.device("cpu"))
        runner._inner = impl.PrimsTsNvfp4MoERunner(
            None, top_k=2, num_local_experts=8, hidden_size=1024, intermediate_size=512
        )
        monkeypatch.setattr(runner, "_require_built", lambda: None)
        act = MoEActivationPack(
            hidden_states_q=torch.empty(4096, 512, device="meta", dtype=torch.uint8),
            hidden_states_scale=torch.empty(4096, 64, device="meta", dtype=torch.uint8),
            routing_logits=torch.empty(4096, 8, device="meta"),
            routing_input_mode=RoutingInputMode.FromLogits,
            output=torch.empty(4096, 1024, device="meta", dtype=torch.bfloat16),
        )
        weights = MoEWeightPack(
            {"prims_ts": {"gemm1_weights": torch.empty(8, 1024, 512, device="meta")}}
        )

        def pack(act, weights):
            tensors = MoeRunnerInputs(
                act.output,
                act.routing_logits,
                None,
                None,
                act.hidden_states_q,
                act.hidden_states_scale,
                None,
                None,
            ).to_list()
            state = _TrtllmLaunchState(
                dict(
                    weight_layout=int(WeightLayout.MajorK),
                    **weights.get_view("prims_ts"),
                )
            )
            return _TrtllmPackedInputs(
                tensors, tuning_config=TuningConfig(), launch_state=state
            )

        monkeypatch.setattr(runner, "pack_inputs", pack)
        return runner, act, weights

    return make


@pytest.mark.cpu_only
def test_rubin_bias_separates_tuning_and_revalidates_layer_winners(
    monkeypatch, packed_nvfp4_layer
):
    runner, act, weights = packed_nvfp4_layer(7)
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.runners, layer._winners = runner.config, [runner], OrderedDict()
    selections = []

    def select(act, weights, runners):
        choices = runner.get_valid_tactics(runner.pack_inputs(act, weights), None)
        winner = max(t for t in choices if t != -1)
        selections.append(winner)
        return runner, winner

    monkeypatch.setattr(layer, "_select_winner", select)
    monkeypatch.setattr(runner, "forward", lambda inputs, tactic, **kw: tactic)
    keys = {}
    view = weights.get_view("prims_ts")
    for fc1, fc2 in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
        (False, False),
        (True, False),
    ):
        for name, present in (("gemm1_bias", fc1), ("gemm2_bias", fc2)):
            view[name] = torch.empty(8, 1024, device="meta") if present else None
        inputs = runner.pack_inputs(act, weights)
        key = runner.get_cache_key_extras(inputs)
        assert keys.setdefault((fc1, fc2), key) == key
        tile, _ = layer(act, weights)
        assert (tile > 256) == (not fc1 and not fc2)
    assert len(set(keys.values())) == len(selections) == 4
    assert not runner.validate_tactic(inputs, (512, 0))
    assert runner.validate_tactic(inputs, selections[-1])
    # A stale explicit BatchM winner under the biased key must also be rejected.
    layer._winners[next(reversed(layer._winners))] = (runner, [512, 0])
    assert layer(act, weights)[0] <= 256
    assert len(selections) == 5


@pytest.mark.cpu_only
@pytest.mark.parametrize("minor", [0, 3, 7])
@pytest.mark.parametrize("source", ["file", "strict_managed"])
def test_replay_pre_architecture_tactic(
    tmp_path, monkeypatch, packed_nvfp4_layer, minor, source
):
    from flashinfer.autotune_cache import autotune_v2
    from flashinfer.autotuner import AutoTuner, TuningConfig
    from flashinfer.fused_moe.runners import MoERunner

    runner, act, weights = packed_nvfp4_layer(minor)
    inputs = runner.pack_inputs(act, weights)
    monkeypatch.setattr(AutoTuner, "_instance", None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "flashinfer.autotuner._collect_metadata", lambda: {"gpu": "test"}
    )
    tuner, tuning, op = AutoTuner.get(), TuningConfig(), "moe_prims_ts"
    shapes = tuner._get_input_sizes(inputs)
    old = tuner._get_cache_key(
        op, runner, shapes, tuning, MoERunner._cache_key_extras(runner)
    )
    new = tuner._get_cache_key(
        op, runner, shapes, tuning, runner.get_cache_key_extras(inputs)
    )
    assert new.file_key != old.file_key
    # A previously tuned tile may be executable outside the current shortlist.
    tactic = (8, 0)

    def choose():
        return tuner.choose_one(op, [runner], tuning, inputs)[1]

    if source == "file":
        tuner.profiling_cache[old] = (tactic, None)
        path = tmp_path / "old.json"
        tuner.save_configs(str(path))
        tuner.clear_cache()
        assert tuner.load_configs(str(path))
        assert choose() == (tactic if minor != 7 else -1)
        return
    with autotune_v2(
        mode="replay", cache_root=tmp_path, cache_scope="context", strict_replay=True
    ):
        store = tuner._active_managed_store
        policy = tuner._profiling_policy(tuning)
        store.publish(
            old.file_key, type(runner).__name__, [9999, 0], profiling_policy=policy
        )
        with pytest.raises(
            RuntimeError, match="not_found" if minor == 7 else "invalid_tactic"
        ):
            choose()
        store.publish(
            old.file_key, type(runner).__name__, list(tactic), profiling_policy=policy
        )
    tuner.clear_cache()
    with autotune_v2(
        mode="replay", cache_root=tmp_path, cache_scope="context", strict_replay=True
    ):
        if minor == 7:
            with pytest.raises(RuntimeError, match="not_found"):
                choose()
        else:
            assert choose() == tactic
            blocklist = tmp_path / "blocklist.json"
            blocklist.write_text(
                json.dumps({"invalid_tactics": {f"{op}::PrimsTsRunner": [tactic]}})
            )
            assert tuner._blocklist.load(str(blocklist))
            # The promoted memory winner and the original managed entry must
            # both defer to a blocklist loaded after the first replay.
            with pytest.raises(RuntimeError, match="blocklisted_tactic"):
                choose()
