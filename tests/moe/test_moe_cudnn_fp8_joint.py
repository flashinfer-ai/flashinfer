# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint FP8 tactics share prepared storage and preserve captured operands."""

from dataclasses import replace
from itertools import product
import json

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.autotuner.autotuner import _json_to_tactic, _tactic_to_json
from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFp8PerTensorConfig,
    MoELayer,
    SiTU,
    SwiGLU,
)
from tests.moe.test_moe_cudnn_fp8 import (
    KEY,
    _check,
    _poison,
    _prepare_weights,
    _prepared,
    _reference,
)
from tests.moe.test_unified_moe_cudnn import _require_b200


@pytest.mark.parametrize(
    "domain",
    [
        [],
        ((1, ((6, 128),)),),
        ((20400, ()),),
        ((20400, ((6, 128), (6, 64))),),
        ((20400, ((6, 128.0),)),),
    ],
)
def test_joint_domain_rejects_mutable_or_invalid_records(domain):
    with pytest.raises(ValueError):
        CudnnFp8PerTensorConfig(fc1_tactics=domain)


def test_joint_domains_are_canonical_and_exclusive():
    record = (20400, ((9, 1), (6, 128)))
    canonical = (20400, ((6, 128), (9, 1)))
    config = CudnnFp8PerTensorConfig(
        fc1_tactics=(record, canonical), fc2_tactics=(canonical,)
    )
    assert config.fc1_tactics == (canonical,)
    assert hash(config) == hash(
        CudnnFp8PerTensorConfig(fc1_tactics=(canonical,), fc2_tactics=(canonical,))
    )
    for stage in ("fc1", "fc2"):
        with pytest.raises(ValueError, match="not both"):
            CudnnFp8PerTensorConfig(
                **{stage + "_tactic": canonical, stage + "_tactics": (canonical,)}
            )


def _tactic(name, policy):
    from cudnn.gemm.frost.knobs import GemmKnobs
    from cudnn.gemm.frost.tile_config import by_name

    knobs = replace(GemmKnobs.from_config(by_name(name)), moe_sched_policy=policy)
    return 20400, tuple(sorted((int(k), int(v)) for k, v in knobs.to_public().items()))


def _joint_backend():
    small = "CONFIG_sm100_64x64x128_64x64x32_cluster1x4_1ctamma"
    wide = "CONFIG_sm100_128x128x128_128x128x32_cluster1x4_1ctamma"
    pair = "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
    return CudnnFp8PerTensorConfig(
        fc1_tactics=(_tactic(small, 1), _tactic(pair, 0)),
        fc2_tactics=(_tactic(wide, 1), _tactic(pair, 0)),
    )


@pytest.mark.parametrize("activation", [SwiGLU(), SiTU()], ids=["swiglu", "situ"])
def test_joint_all_pairs_capture_workspace_and_current_packs(activation, monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, first, w1, w2 = _prepared(activation, True)
    backend = _joint_backend()
    config = replace(config, backend=BackendOptions((backend,)))
    layer = MoELayer(config)
    runner = layer.runners[0]
    inputs = runner.pack_inputs(act, first)
    state = runner._resources(inputs)
    pairs = runner.get_valid_tactics(inputs, None)
    assert pairs == list(product(backend.fc1_tactics, backend.fc2_tactics))
    assert len(layer.runners) == 1 and len(pairs) == 4
    for stage in ("fc1", "fc2"):
        prepared = state[stage]
        assert len(prepared.tactic_indices) == 2
        sizes = [
            prepared.graph.get_workspace_size_plan_at_index(i)
            for i in prepared.tactic_indices.values()
        ]
        assert prepared.workspace.numel() == max(sizes)
        assert sizes[1] > sizes[0], (stage, sizes)

    second = _prepare_weights(
        w1 * 0.75, w2 * -0.5, activation, input_scale=32.0, intermediate_scale=64.0
    )
    second_inputs = runner.pack_inputs(act, second)
    assert runner._resources(second_inputs) is state
    expected = [
        _reference(act, pack.get_view(KEY), activation) for pack in (first, second)
    ]
    captures = []
    for index, pair in enumerate(pairs):
        selected_inputs = (inputs, second_inputs)[index % 2]
        _check(runner.forward(selected_inputs, tactic=pair), expected[index % 2])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = runner.forward(selected_inputs, tactic=pair)
        torch.cuda.current_stream().wait_stream(stream)
        captures.append((graph, output, index % 2))

    act.hidden_states_q.copy_((-act.hidden_states_q.float()).to(torch.float8_e4m3fn))
    act.topk_ids.add_(2).remainder_(config.routing.num_experts)
    act.topk_weights.mul_(0.5)
    changed = [
        _reference(act, pack.get_view(KEY), activation) for pack in (first, second)
    ]
    for graph, output, pack_index in captures:
        _poison(state)
        graph.replay()
        _check(output, changed[pack_index])
        assert not torch.equal(output, expected[pack_index])

    act.hidden_states_q.copy_((-act.hidden_states_q.float()).to(torch.float8_e4m3fn))
    act.topk_ids.add_(config.routing.num_experts - 2).remainder_(
        config.routing.num_experts
    )
    act.topk_weights.mul_(2)
    for pair in pairs:
        # Exercise FI's persisted nested-tuple format with actual execution.
        restored = _json_to_tactic(json.loads(json.dumps(_tactic_to_json(pair))))
        _check(runner.forward(inputs, tactic=restored), expected[0])

    # Reject the entire pair before either GEMM executes.
    def unexpected_stage(*args, **kwargs):
        pytest.fail("An invalid joint tactic launched FC1")

    with monkeypatch.context() as patch:
        patch.setattr(state["fc1"], "run", unexpected_stage)
        with pytest.raises(ValueError, match="unavailable"):
            runner.forward(inputs, tactic=(pairs[0][0], (20400, ())))

    # After eager preparation, no plan builds or GPU allocations are allowed
    # even when switching candidates while constructing another capture.
    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Joint tactic replay tried to prepare storage or a plan")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", unexpected_preparation)
        for stage in ("fc1", "fc2"):
            patch.setattr(
                state[stage].graph, "build_plan_at_index", unexpected_preparation
            )
            patch.setattr(state[stage].graph, "_build_plan_at", unexpected_preparation)
        with torch.cuda.graph(graph, stream=stream):
            output = runner.forward(inputs, tactic=pairs[-1])
    torch.cuda.current_stream().wait_stream(stream)
    _poison(state)
    graph.replay()
    _check(output, expected[0])


def test_joint_ordinary_autotune_and_domain_cache_identity(monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, weights, _, _ = _prepared(SwiGLU(), True, tokens=65)
    backend = _joint_backend()
    config = replace(config, backend=BackendOptions((backend,)))
    layer = MoELayer(config)
    runner = layer.runners[0]
    inputs = runner.pack_inputs(act, weights)
    pairs = runner.get_valid_tactics(inputs, None)
    expected = _reference(act, weights.get_view(KEY), config.activation)
    for pair in pairs:
        _check(runner.forward(inputs, tactic=pair), expected)
    observed = set()
    forward = runner.forward

    def tracked(*args, tactic=-1, **kwargs):
        if tactic != -1:
            observed.add(tactic)
        return forward(*args, tactic=tactic, **kwargs)

    monkeypatch.setattr(runner, "forward", tracked)
    with autotune():
        _check(layer(act, weights), expected)
    assert observed == set(pairs)
    selected_runner, selected = next(iter(layer._winners.values()))
    assert selected_runner is runner and selected in pairs
    _check(layer(act, weights), expected)
    reverse = replace(backend, fc2_tactics=backend.fc2_tactics[::-1])
    other = MoELayer(replace(config, backend=BackendOptions((reverse,)))).runners[0]
    assert runner.get_cache_key_extras(inputs) != other.get_cache_key_extras(inputs)


def test_joint_default_has_concrete_frost_identity(monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, weights, _, _ = _prepared(SwiGLU(), True, default_tactics=True)
    layer = MoELayer(config)
    _check(
        layer(act, weights), _reference(act, weights.get_view(KEY), config.activation)
    )
    runner = layer.runners[0]
    inputs = runner.pack_inputs(act, weights)
    pairs = runner.get_valid_tactics(inputs, None)
    assert len(pairs) == 1 and all(
        engine == 20400 and knobs for engine, knobs in pairs[0]
    )
    pinned = replace(
        config,
        backend=BackendOptions(
            (CudnnFp8PerTensorConfig(fc1_tactic=pairs[0][0], fc2_tactic=pairs[0][1]),)
        ),
    )
    replay = MoELayer(pinned)
    _check(
        replay(act, weights), _reference(act, weights.get_view(KEY), config.activation)
    )
