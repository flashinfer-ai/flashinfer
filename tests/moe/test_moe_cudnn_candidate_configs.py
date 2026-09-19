# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared same-backend candidates must reach independent FC1/FC2 graph plans."""

from dataclasses import replace
from itertools import product

import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import BackendOptions, CudnnMoeConfig, MoELayer
from flashinfer.tllm_enums import RoutingInputMode
from tests.moe.test_unified_moe_cudnn import _case, _reference, _require_b200


def test_declared_fc1_fc2_scheduler_candidates_remain_distinct(monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    import cudnn
    from cudnn.gemm.frost.knobs import GemmKnobs
    from cudnn.gemm.frost.tile_config import by_name

    tile = by_name("CONFIG_sm100_64x64x128_64x64x32_cluster1x4_1ctamma")

    def tactic(policy):
        knobs = replace(GemmKnobs.from_config(tile), moe_sched_policy=policy)
        return 20400, tuple(
            sorted((int(k), int(v)) for k, v in knobs.to_public().items())
        )

    candidates = tuple(
        CudnnMoeConfig(
            use_native_routing=True, fc1_tactic=tactic(first), fc2_tactic=tactic(second)
        )
        for first, second in product((0, 1), repeat=2)
    )
    config, act, weights, w1, w2 = _case(RoutingInputMode.UnpackedPrecomputed)
    layer = MoELayer(replace(config, backend=BackendOptions(candidates)))
    assert tuple(r.backend_config for r in layer.runners) == candidates, (
        "Candidate iteration repeatedly bound the first backend config"
    )
    assert len({r._cache_key_extras() for r in layer.runners}) == 4
    reference = _reference(act, w1, w2)
    for runner, candidate in zip(layer.runners, candidates, strict=True):
        inputs = runner.pack_inputs(act, weights)
        actual = runner.forward(inputs)
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
        state = runner._resources(inputs)
        for stage, selected in (
            ("fc1", candidate.fc1_tactic),
            ("fc2", candidate.fc2_tactic),
        ):
            assert state[stage].tactics == (selected,)
            engine, knobs = state[stage].graph.get_engine_and_knobs_at_index(0)
            assert engine == 20400
            assert int(knobs.get(cudnn.knob_type.SCHED_POLICY, 0)) == dict(
                selected[1]
            ).get(int(cudnn.knob_type.SCHED_POLICY), 0)
    with autotune():
        actual = layer(act, weights)
    assert layer.winner_backend == "cudnn"
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=side):
        actual = layer(act, weights)
    torch.cuda.current_stream().wait_stream(side)
    previous = actual.clone()
    act.hidden_states_q.neg_()
    actual.fill_(float("nan"))
    capture.replay()
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)
    assert not torch.equal(actual, previous)
