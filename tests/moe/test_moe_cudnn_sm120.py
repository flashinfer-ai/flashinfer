# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public SM120 BF16 MoE route without architecture or JIT monkeypatches."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import CudnnMoeConfig, MoELayer, SwiGLU
from flashinfer.tllm_enums import RoutingInputMode
from tests.moe.test_moe_cudnn_activations import ACTIVATIONS, _reference
from tests.moe.test_unified_moe_cudnn import _case

TILES = [
    "CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4",
    "CONFIG_sm120_128x128x128_16x16x32_cluster1x1_warps4x2",
]


def _exercise(mode, native, tile, dimensions, activation, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU required")
    pytest.importorskip("cudnn")
    from cudnn.frost.buffers import cutedsl_requirement_error

    if (error := cutedsl_requirement_error("SM120 cuDNN MoE test")) is not None:
        pytest.skip(error)
    from cudnn.gemm.frost.knobs import GemmKnobs
    from cudnn.gemm.frost.tile_config import by_name

    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    knobs = GemmKnobs.from_config(by_name(tile)).to_public()
    tactic = (20400, tuple(sorted((int(k), int(v)) for k, v in knobs.items())))
    backend = CudnnMoeConfig(
        use_native_routing=native, fc1_tactic=tactic, fc2_tactic=tactic
    )
    tokens, experts, hidden, inter = dimensions
    config, act, weights, w1, w2 = _case(
        mode,
        tokens=tokens,
        experts=experts,
        hidden=hidden,
        inter=inter,
        backend=backend,
    )
    config = replace(config, activation=activation)
    ids = act.topk_ids.clone()
    scales = act.topk_weights.clone()
    x = act.hidden_states_q.clone()

    def set_variant(variant):
        act.hidden_states_q.copy_(-x if variant == "x" else x)
        act.topk_ids.copy_((ids + 2) % experts if variant == "ids" else ids)
        act.topk_weights.copy_(scales * 0.5 if variant == "scales" else scales)

    # Reference GEMMs run before capture/replay. Keep the reference independent
    # of the adapter's maps and preserve each routing mode's rounding contract.
    references = {}
    for variant in ("base", "x", "ids", "scales"):
        set_variant(variant)
        references[variant] = _reference(act, w1, w2, activation)
    set_variant("base")
    for variant in ("x", "ids", "scales"):
        assert not torch.equal(references[variant], references["base"])

    def check(actual, expected):
        actual, expected = actual.float(), expected.float()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        assert relative_l2 < 0.01, float(relative_l2)

    layer = MoELayer(config)
    actual = layer(act, weights)
    check(actual, references["base"])
    assert layer.winner_backend == "cudnn"
    runner = layer.runners[0]
    state = runner._resources(runner.pack_inputs(act, weights))
    assert not state["fused"]
    assert state["fc1_output"].dtype == torch.float32
    for name in ("fc1", "fc2"):
        graph = state[name].graph
        engine, public_knobs = graph.get_engine_and_knobs_at_index(0)
        assert engine == 20400 and public_knobs == knobs
        compiled = graph._compiled_plans[0]._compiled
        assert compiled.config.pipeline == "sm120" and compiled.config.name == tile

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=side):
        actual = layer(act, weights)
    torch.cuda.current_stream().wait_stream(side)
    for variant in ("x", "ids", "scales", "base"):
        set_variant(variant)
        for name in ("routed", "intermediate", "projected", "output", "fc1_output"):
            state[name].fill_(float("nan"))
        actual.fill_(float("nan"))
        capture.replay()
        check(actual, references[variant])
        if variant != "base":
            assert not torch.equal(actual, references["base"])


@pytest.mark.parametrize(
    "mode", [RoutingInputMode.PackedPrecomputed, RoutingInputMode.UnpackedPrecomputed]
)
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("dimensions", [(17, 8, 256, 256), (257, 16, 2048, 1024)])
def test_public_sm120_routing_capture(mode, native, tile, dimensions, monkeypatch):
    _exercise(mode, native, tile, dimensions, SwiGLU(), monkeypatch)


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=repr)
@pytest.mark.parametrize("native", [False, True])
def test_public_sm120_typed_activation_capture(activation, native, monkeypatch):
    _exercise(
        RoutingInputMode.UnpackedPrecomputed,
        native,
        TILES[0],
        (257, 8, 256, 256),
        activation,
        monkeypatch,
    )
