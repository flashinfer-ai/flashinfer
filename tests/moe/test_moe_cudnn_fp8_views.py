# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical split weights, mixed-layout plan identity and retained captures."""

import pytest
import torch

from flashinfer.fused_moe import MoELayer, MoEWeightPack, SiTU, SwiGLU
from tests.moe.test_moe_cudnn_fp8 import KEY, _check, _poison, _prepared, _reference
from tests.moe.test_unified_moe_cudnn import _require_b200


def _pack(view):
    pack = MoEWeightPack()
    pack.prepare_for(KEY, view)
    return pack


@pytest.mark.parametrize("activation", [SwiGLU(), SiTU()], ids=["swiglu", "situ"])
@pytest.mark.parametrize("wide", [True, False], ids=["1cta-static", "2cta-dynamic"])
def test_split_and_contiguous_captures_remain_independent(
    activation, wide, monkeypatch
):
    _require_b200()
    config, act, weights, _, _ = _prepared(activation, wide, tokens=65)
    original = weights.get_view(KEY)
    # Construct the canonical view independently of prepare_weights, so this
    # is a RED test of the old adapter's layout rejection as well.
    joined = torch.cat((original["up"], original["gate"]), dim=1)
    up, gate = joined.split(joined.shape[1] // 2, dim=1)
    split = dict(original, up=up, gate=gate)
    compact = {name: tensor.clone().contiguous() for name, tensor in split.items()}
    assert not up.is_contiguous() and not gate.is_contiguous()
    packs = [_pack(split), _pack(compact)]
    layer = MoELayer(config)
    layer(act, packs[0])
    runner = layer.runners[0]
    inputs = [runner.pack_inputs(act, pack) for pack in packs]
    states = [runner._resources(inp) for inp in inputs]
    assert states[0] is not states[1]
    assert runner.get_cache_key_extras(inputs[0]) != runner.get_cache_key_extras(
        inputs[1]
    )
    assert len(runner._resources_cache) == 2
    for inp, state, view in zip(inputs, states, (split, compact), strict=True):
        _check(runner.forward(inp), _reference(act, view, activation))
        for stage in ("fc1", "fc2"):
            assert state[stage].graph.get_engine_and_knobs_at_index(0)[0] == 20400

    graphs = []
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    def forbidden(*args, **kwargs):
        raise AssertionError("Prepared layout execution must not allocate or build")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        patch.setattr(runner, "_prepare_stages", forbidden)
        for inp in inputs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side):
                runner.forward(inp)
            graphs.append(graph)
    torch.cuda.current_stream().wait_stream(side)
    # The two packs share no changed operands. The old split graph must keep
    # its own pointers after the compact pack has built another set of plans.
    split["up"].copy_((-split["up"].float()).to(torch.float8_e4m3fn))
    split["fc1_scale"].mul_(0.5)
    compact["gate"].copy_((compact["gate"].float() * 0.5).to(torch.float8_e4m3fn))
    compact["fc2_scale"].mul_(0.75)
    act.hidden_states_q.copy_((-act.hidden_states_q.float()).to(torch.float8_e4m3fn))
    act.topk_ids.add_(2).remainder_(8)
    act.topk_weights.mul_(0.5)
    expected = [_reference(act, view, activation) for view in (split, compact)]
    for index in (0, 1, 0):
        for state in states:
            _poison(state)
        graphs[index].replay()
        _check(inputs[index][7], expected[index])
    assert not torch.equal(expected[0], expected[1])

    # Equal shapes with inner-strided or overlapping expert storage remain
    # unsupported; this extension is deliberately limited to canonical pitch.
    for invalid in (up.transpose(1, 2), up.as_strided(up.shape, (0, *up.stride()[1:]))):
        malformed = _pack(dict(split, up=invalid))
        with pytest.raises(ValueError, match="up"):
            runner.pack_inputs(act, malformed)


def test_prepared_fc1_views_share_quantized_storage():
    from flashinfer.fused_moe import CudnnFp8PerTensorConfig

    w1 = torch.linspace(-1, 1, 2 * 64 * 32).reshape(2, 64, 32).to(torch.bfloat16)
    w2 = torch.ones(2, 32, 32, dtype=torch.bfloat16)
    options = dict(
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=32,
        hidden_states_scale_global=64.0,
        intermediate_scale_global=128.0,
    )
    compact = CudnnFp8PerTensorConfig.prepare_weights(w1, w2, **options)
    view = CudnnFp8PerTensorConfig.prepare_weights(
        w1, w2, copy_fc1_weights=False, **options
    )
    assert compact["up"].is_contiguous() and compact["gate"].is_contiguous()
    assert (
        compact["up"].untyped_storage().data_ptr()
        != compact["gate"].untyped_storage().data_ptr()
    )
    for name, tensor in compact.items():
        assert torch.equal(
            tensor.reshape(-1).view(torch.uint8),
            view[name].reshape(-1).view(torch.uint8),
        )
    with pytest.raises(ValueError, match="copy_fc1_weights"):
        CudnnFp8PerTensorConfig.prepare_weights(w1, w2, copy_fc1_weights=1, **options)
    up, gate = view["up"], view["gate"]
    assert up.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()
    assert up.untyped_storage().nbytes() == w1.numel()
    assert up.stride() == gate.stride() == (64 * 32, 32, 1)
    assert gate.storage_offset() - up.storage_offset() == 32 * 32
    assert view["down"].is_contiguous()


def test_public_preparation_choice_runs_both_layouts():
    from flashinfer.fused_moe import CudnnFp8PerTensorConfig

    _require_b200()
    config, act, compact, w1, w2 = _prepared(SwiGLU(), True, tokens=65)
    view = CudnnFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        hidden_states_scale_global=64.0,
        intermediate_scale_global=128.0,
        copy_fc1_weights=False,
    )
    assert not view["up"].is_contiguous()
    layer = MoELayer(config)
    for pack in (compact, _pack(view), compact):
        _check(layer(act, pack), _reference(act, pack.get_view(KEY), SwiGLU()))
    assert len(layer.runners[0]._resources_cache) == 2
