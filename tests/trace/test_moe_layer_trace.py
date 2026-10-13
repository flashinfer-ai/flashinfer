# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Frost pack traces preserve quantized storage and work through API auto-dump."""

from collections import OrderedDict
import json
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import MoELayer, RoutingInputMode
from tests.trace.example_moe_layer import example_inputs


@pytest.mark.parametrize("dtype", ["bf16", "mxfp8", "nvfp4", "mxfp8_mxfp4"])
def test_frost_trace_metadata_and_golden(dtype, tmp_path):
    layer, act, weights = example_inputs(dtype)
    name = f"moe_layer_cudnn_frost_{dtype}"
    result = MoELayer.__call__.fi_trace(
        self=layer, act_pack=act, weight_pack=weights, save_dir=tmp_path, name=name
    )
    assert result["outputs"]["output"]["dtype"] == "bfloat16"
    assert result["axes"]["hidden_size"]["value"] == 2048
    assert result["inputs"]["hidden_states_q"]["dtype"] == str(
        act.hidden_states_q.dtype
    ).removeprefix("torch.")
    assert all(value["dtype"] != "unknown" for value in result["inputs"].values())
    from pathlib import Path

    expected = Path(__file__).parent / "fi_trace_out" / f"{name}.json"
    assert result == json.loads(expected.read_text())


def test_moe_layer_autodump(tmp_path, monkeypatch):
    layer, act, weights = example_inputs("nvfp4")

    class Runner:
        backend_key = "cudnn_frost_nvfp4"
        supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)

        def pack_inputs(self, act, weights):
            return [act.hidden_states_q]

        def launch_kwargs_for(self, inputs):
            return {}

        def forward(self, inputs, **kwargs):
            return torch.empty(9, 2048, device="meta", dtype=torch.bfloat16)

    runner = Runner()
    layer.runners = [runner]
    layer.tuner = SimpleNamespace(is_tuning_mode=False)
    layer._winners = OrderedDict()
    monkeypatch.setattr(layer, "_additional_candidates", lambda *args: [])
    monkeypatch.setattr(layer, "_select_winner", lambda *args: (runner, -1))
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr("flashinfer.trace.template._DUMPED_NAMES", set())
    assert layer(act, weights).shape == (9, 2048)
    traces = list(tmp_path.glob("*.json"))
    assert len(traces) == 1
    assert json.loads(traces[0].read_text())["op_type"] == "moe_layer"
