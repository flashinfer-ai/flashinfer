# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host regressions for exact-shape Frost winners and block-scale tactic bounds."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    GeGLU,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner


@pytest.mark.parametrize("first,following", [(8, (5, 6, 7)), (4, (3,))])
@pytest.mark.parametrize("packed", [False, True])
def test_layer_reselects_shape_bounded_frost_tactic(
    first, following, packed, monkeypatch
):
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: (9, 0))
    monkeypatch.setattr(
        "flashinfer.fused_moe.backends.frost_low_latency.runner.version",
        lambda _: "4.8.0",
    )
    config = MoEConfig(
        RoutingConfig(16, 10),
        QuantConfig(
            weight=QuantFormat.FP8PerTensor,
            activation=QuantFormat.BF16,
            output=QuantFormat.BF16,
        ),
        ExpertConfig(512),
        activation=GeGLU(),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    runner = FrostLowLatencyRunner(config, torch.device("cpu"))
    runner.check_support()
    runner.build()
    # Keep the actual candidate enumeration and layer cache; no GPU work is needed
    # to reject a winner that belongs to another token count.
    monkeypatch.setattr(
        runner, "pack_inputs", lambda act, weights: [act.hidden_states_q]
    )
    monkeypatch.setattr(runner, "launch_kwargs_for", lambda _: {})
    launched, selected = [], []

    def forward(inputs, tactic, **kwargs):
        assert tactic in runner.get_valid_tactics(inputs, None), (
            inputs[0].shape,
            tactic,
        )
        launched.append((inputs[0].shape[0], tactic))
        return inputs[0]

    monkeypatch.setattr(runner, "forward", forward)
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.runners, layer._winners = config, [runner], OrderedDict()

    def select(act, weights, runners):
        assert runners == [runner]
        choices = runner.get_valid_tactics([act.hidden_states_q], None)
        tactic = next((t for t in choices if len(t) == 4 and t[-1] == 256), choices[0])
        selected.append(act.num_tokens)
        return runner, tactic

    monkeypatch.setattr(layer, "_select_winner", select)
    weights = MoEWeightPack({"frost_low_latency": {}})
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    for tokens in (first, *following, first, *following):
        act = MoEActivationPack(
            torch.empty(tokens, 2048, dtype=torch.bfloat16),
            None,
            torch.empty(tokens, 10, dtype=torch.int32),
            torch.empty(tokens, 10),
            routing_input_mode=mode,
        )
        assert layer(act, weights) is act.hidden_states_q
    assert selected == [first, *following]
    assert launched[0][1] == ("warp_mma", first, 2, 256)
    assert all(tactic[0] == "ffma" for tokens, tactic in launched if tokens != first)


@pytest.mark.parametrize("group", [4, 8])
def test_deepseek_rejects_k256_before_tensor_access(group):
    pytest.importorskip("cutlass", minversion="4.8")
    from flashinfer.fused_moe.backends.frost_low_latency.tc.deepseek_fp8 import execute

    # A disallowed block-scale tile must fail before touching inputs or CUDA.
    with pytest.raises(ValueError, match="K64/K128"):
        execute([None] * 15, SimpleNamespace(), ("warp_mma", group, 2, 256))
