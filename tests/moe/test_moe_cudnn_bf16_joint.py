# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public route and paired-stage tuning with live inputs and retained captures."""

from dataclasses import replace
from itertools import product
import json

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.autotuner.autotuner import _json_to_tactic, _tactic_to_json
from flashinfer.fused_moe import BackendOptions, CudnnMoeConfig, MoELayer, MoEWeightPack
from flashinfer.tllm_enums import RoutingInputMode
from tests.moe.test_unified_moe_cudnn import _case, _reference


def _check(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert (
        float(
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)
        )
        <= 0.01
    )


def _poison(state):
    for name in ("routed", "intermediate", "projected", "output", "fc1_output"):
        if name in state:
            state[name].fill_(float("nan"))
    state["offsets"].fill_(-777)
    for value in state["native_sort"].values():
        if isinstance(value, torch.Tensor):
            value.fill_(-777)


def _kernel_names(graph):
    from cuda.bindings import driver as drv

    def checked(result):
        error, *values = result
        assert int(error) == 0
        return values

    handle = drv.CUgraph(graph.raw_cuda_graph())
    _, count = checked(drv.cuGraphGetNodes(handle, 0))
    nodes, actual_count = checked(drv.cuGraphGetNodes(handle, count))
    assert count == actual_count
    names = []
    for node in nodes:
        (kind,) = checked(drv.cuGraphNodeGetType(node))
        assert kind == drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
        (params,) = checked(drv.cuGraphKernelNodeGetParams(node))
        (name,) = checked(drv.cuFuncGetName(params.func))
        names.append(name.decode())
    return names


def _track(forward, observed):
    def tracked(*args, tactic=-1, **kwargs):
        if tactic != -1:
            observed.add(tactic)
        return forward(*args, tactic=tactic, **kwargs)

    return tracked


@pytest.mark.parametrize(
    "mode", [RoutingInputMode.PackedPrecomputed, RoutingInputMode.UnpackedPrecomputed]
)
def test_bf16_joint_routes_autotune_capture_and_weight_interleave(mode, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU required")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    from cudnn import knob_type
    from cudnn.gemm.frost.knobs import GemmKnobs
    from cudnn.gemm.frost.tile_config import by_name

    tile = by_name("CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4")
    records = tuple(
        (
            20400,
            tuple(
                sorted(
                    (int(k), int(v))
                    for k, v in replace(
                        GemmKnobs.from_config(tile), moe_sched_policy=policy
                    )
                    .to_public()
                    .items()
                )
            ),
        )
        for policy in (0, 1)
    )
    backends = tuple(
        CudnnMoeConfig(
            use_native_routing=True,
            fc1_tactics=records,
            fc2_tactics=records[::-1],
            fc1_fusion=fused,
        )
        for fused in (True, False)
    )
    config, act, weights, w1, w2 = _case(
        mode, tokens=257, experts=40, hidden=272, inter=144
    )
    layer = MoELayer(replace(config, backend=BackendOptions(backends)))
    assert (
        len(layer.runners) == 2
        and tuple(r.backend_config for r in layer.runners) == backends
    )
    second = MoEWeightPack()
    changed_w1, changed_w2 = w1 * 0.75, w2 * -0.5
    second.prepare_for(
        "cudnn",
        CudnnMoeConfig.prepare_weights(
            changed_w1,
            changed_w2,
            num_local_experts=40,
            hidden_size=272,
            intermediate_size=144,
        ),
    )
    references = [
        _reference(act, first, second)
        for first, second in ((w1, w2), (changed_w1, changed_w2))
    ]
    captures, states, observed = [], [], [set(), set()]
    for runner_index, runner in enumerate(layer.runners):
        first_inputs, second_inputs = [
            runner.pack_inputs(act, pack) for pack in (weights, second)
        ]
        state = runner._resources(first_inputs)
        states.append(state)
        assert (
            runner._resources(second_inputs) is state
            and state["fused"] is backends[runner_index].fc1_fusion
        )
        pairs = runner.get_valid_tactics(first_inputs, None)
        assert pairs == list(product(records, records[::-1]))
        for stage in ("fc1", "fc2"):
            prepared = state[stage]
            assert len(prepared.tactic_indices) == 2
            assert prepared.workspace.numel() == max(
                prepared.graph.get_workspace_size_plan_at_index(index)
                for index in prepared.tactic_indices.values()
            )
            for plan in prepared.graph._compiled_plans.values():
                assert plan._compiled.chain.num_gemms == (
                    2 if stage == "fc1" and state["fused"] else 1
                )
        for index, pair in enumerate(pairs):
            selected_inputs = (first_inputs, second_inputs)[index % 2]
            restored = _json_to_tactic(json.loads(json.dumps(_tactic_to_json(pair))))
            _check(
                runner.forward(selected_inputs, tactic=restored), references[index % 2]
            )
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=side):
                actual = runner.forward(selected_inputs, tactic=pair)
            torch.cuda.current_stream().wait_stream(side)
            names = _kernel_names(graph)
            # Dynamic policy is omitted from the canonical public record.
            policies = [
                dict(record[1]).get(int(knob_type.SCHED_POLICY), 0) for record in pair
            ]
            assert len(names) == 8 - sum(policies) - int(state["fused"])
            assert sum("reset_moe_sched_counter" in name for name in names) == 2 - sum(
                policies
            )
            captures.append((graph, actual, state, index % 2))

        monkeypatch.setattr(
            runner, "forward", _track(runner.forward, observed[runner_index])
        )
        with monkeypatch.context() as patch:
            from flashinfer.fused_moe.cute_dsl import moe_utils

            patch.setattr(
                moe_utils,
                "_get_moe_utils_module",
                lambda: pytest.fail("Invalid FC2 launched routing"),
            )
            patch.setattr(
                runner,
                "_run_stages",
                lambda *args: pytest.fail("Invalid FC2 launched a GEMM"),
            )
            with pytest.raises(ValueError, match="unavailable"):
                runner.forward(first_inputs, tactic=(pairs[0][0], (999999, ())))
        observed[runner_index].clear()
    assert states[0] is not states[1]
    assert layer.runners[0].get_cache_key_extras(first_inputs) != layer.runners[
        1
    ].get_cache_key_extras(first_inputs)
    with autotune():
        _check(layer(act, weights), references[0])
    assert all(seen == set(product(records, records[::-1])) for seen in observed)
    selected_runner, selected = next(iter(layer._winners.values()))
    assert selected_runner in layer.runners and selected in set(
        product(records, records[::-1])
    )
    act.hidden_states_q.neg_()
    act.topk_ids.add_(2).remainder_(40)
    act.topk_weights.mul_(0.5)
    changed = [
        _reference(act, first, second)
        for first, second in ((w1, w2), (changed_w1, changed_w2))
    ]
    for previous, current in zip(references, changed, strict=True):
        assert not torch.equal(previous, current)
    for graph, actual, state, pack_index in captures:
        _poison(state)
        graph.replay()
        _check(actual, changed[pack_index])
    inputs = selected_runner.pack_inputs(act, weights)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:

        def unexpected(*args, **kwargs):
            pytest.fail("Prepared joint replay attempted allocation or plan building")

        patch.setattr(torch, "empty", unexpected)
        for state in states:
            for stage in ("fc1", "fc2"):
                patch.setattr(state[stage].graph, "build_plan_at_index", unexpected)
                patch.setattr(state[stage].graph, "_build_plan_at", unexpected)
        with torch.cuda.graph(graph, stream=side):
            actual = selected_runner.forward(inputs, tactic=selected)
    torch.cuda.current_stream().wait_stream(side)
    _poison(selected_runner._resources(inputs))
    graph.replay()
    _check(actual, changed[0])
