# Copyright (c) 2026 by FlashInfer team. Licensed under the Apache License, Version 2.0.
"""Layer winner reuse must follow the real Prims-TS tactic menu."""

from collections import OrderedDict
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
    weights = MoEWeightPack({"prims_ts": {}})
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
    inner.get_valid_tactics = lambda *args: [
        [512, 14 if fused_finalize_requested(inner) else 0]
    ]
    runner._inner = inner
    monkeypatch.setattr(runner, "_require_built", lambda: None)
    monkeypatch.setattr(runner, "_forward_inner", lambda *args: None)
    monkeypatch.setattr(runner, "precompile_tactics", lambda *args, **kwargs: False)
    state = _TrtllmLaunchState(
        dict(weight_layout=0, zero_rows_without_local_expert=fused)
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
    other = inputs.with_launch_overrides(zero_rows_without_local_expert=not fused)
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
