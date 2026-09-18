# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC2 preparation and stage binding preserve the declared K64 storage."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from flashinfer.fused_moe import CudnnMoeConfig
from flashinfer.fused_moe import cudnn_backend as backend


@pytest.mark.parametrize("e,h,i", [(1, 128, 64), (3, 256, 128)])
@pytest.mark.parametrize("fc1", [None, "k_blocked_64_v1"])
def test_fc2_k64_preparation_exact_bytes_and_independent_layout(e, h, i, fc1):
    gen = torch.Generator().manual_seed(2348)
    w1 = torch.randn((e, 2 * i, h), generator=gen, dtype=torch.bfloat16)
    w2 = torch.randn((e, h, i), generator=gen, dtype=torch.bfloat16)
    w2.flatten()[0] = -0.0
    args = dict(
        num_local_experts=e, hidden_size=h, intermediate_size=i, fc1_weight_layout=fc1
    )
    canonical = CudnnMoeConfig.prepare_weights(w1, w2, **args)
    prepared = CudnnMoeConfig.prepare_weights(
        w1, w2, **args, fc2_weight_layout="k_blocked_64_v1"
    )
    down = prepared["down"]
    assert tuple(down.shape) == (e, i // 64, h, 64)
    assert tuple(down.stride()) == (h * i, h * 64, 64, 1)
    restored = down.transpose(1, 2).reshape(e, h, i)
    assert torch.equal(
        restored.contiguous().view(torch.uint8), w2.contiguous().view(torch.uint8)
    )
    for name in ("up", "gate", "gate_up"):
        assert torch.equal(
            prepared[name].contiguous().view(torch.uint8),
            canonical[name].contiguous().view(torch.uint8),
        )
    cfg = CudnnMoeConfig(fc1_weight_layout=fc1, fc2_weight_layout="k_blocked_64_v1")
    assert cfg != CudnnMoeConfig(fc1_weight_layout=fc1)


@pytest.mark.parametrize("h,i", [(64, 64), (128, 32)])
def test_fc2_k64_preparation_rejects_unsupported_geometry(h, i):
    with pytest.raises(ValueError, match="K64 FC2"):
        CudnnMoeConfig.prepare_weights(
            torch.empty(1, 2 * i, h, dtype=torch.bfloat16),
            torch.empty(1, h, i, dtype=torch.bfloat16),
            num_local_experts=1,
            hidden_size=h,
            intermediate_size=i,
            fc2_weight_layout="k_blocked_64_v1",
        )


def test_fc2_k64_config_rejects_unknown_layout():
    with pytest.raises(ValueError, match="FC2 weight layout"):
        CudnnMoeConfig(fc2_weight_layout="unknown")


def test_fc2_k64_stage_rebinds_the_original_rank4_operand(monkeypatch):
    import cudnn

    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=17)
    )
    monkeypatch.setattr(cudnn, "set_stream", lambda **kwargs: None)
    stage = object.__new__(backend._Stage)
    stage.a, stage.offsets, stage.out, weight_desc = [object() for _ in range(4)]
    stage.handle, stage.workspace, stage.graph = object(), object(), Mock()
    stage.weights, stage.weights_parent = [weight_desc], None
    stage.weight_layout, stage.scalar_bindings = "k_blocked_64_v1", {}
    stage.plan_index = lambda tactic: -1
    x, out = torch.empty(2, 128), torch.empty(2, 256)
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    weights = [torch.empty(1, 2, 256, 64, dtype=torch.bfloat16) for _ in range(2)]
    for weight in weights:
        stage.run(x, [weight], offsets, out)
        bound = stage.graph.execute.call_args.args[0][weight_desc]
        assert bound is weight and bound.data_ptr() == weight.data_ptr()
        assert tuple(bound.shape) == (1, 2, 256, 64)


def test_fc2_layout_reaches_only_fc2_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(
        backend, "_Stage", lambda *a, **kw: calls.append(kw) or object()
    )
    runner = object.__new__(backend.CudnnMoeRunner)
    runner.backend_config = CudnnMoeConfig(
        fc1_fusion=True, fc2_weight_layout="k_blocked_64_v1"
    )
    runner.config = SimpleNamespace(
        routing=SimpleNamespace(top_k=2), activation=object()
    )
    x, up, down = (
        torch.empty(3, 128),
        torch.empty(2, 64, 128),
        torch.empty(2, 1, 128, 64),
    )
    state = {
        name: object() for name in ("routed", "offsets", "intermediate", "projected")
    }
    runner._prepare_stages(state, [x, None, None, up, up, down, up], lambda *a: None)
    assert len(calls) == 2
    assert (
        calls[0]["weight_layout"] is None
        and calls[1]["weight_layout"] == "k_blocked_64_v1"
    )
