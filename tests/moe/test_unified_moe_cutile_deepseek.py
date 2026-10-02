# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CuTileDeepSeekFp8Bf16Config,
    CuTileDeepSeekFp8Bf16Runner,
    CuTileDeepSeekFp8Config,
    CuTileDeepSeekFp8Runner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    ReLU2,
    RoutingConfig,
    SiTU,
    SwiGLU,
)

from .test_unified_moe_cutile import _CUTILE_ACTIVATIONS, _require_cutile_fp8
from .utils import compute_reference_activation


def _quantize_weight(weight):
    experts, n, k = weight.shape
    tiles = weight.float().reshape(experts, n // 128, 128, k // 128, 128)
    scale = (tiles.abs().amax((2, 4)) / 448).clamp_min(1e-12)
    expanded = scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
    return (weight.float() / expanded).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def _block_matmul(x, weight, weight_scale, a8):
    if a8:
        tiles = x.float().reshape(x.shape[0], -1, 128)
        scale = (tiles.abs().amax(-1) / 448).clamp_min(1e-12)
        x = (tiles / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        x = x.flatten(1)
    result = torch.zeros(x.shape[0], weight.shape[1], device=x.device)
    for block in range(x.shape[1] // 128):
        columns = slice(block * 128, (block + 1) * 128)
        partial = torch.bmm(
            weight[:, :, columns].float(), x[:, columns].float().unsqueeze(-1)
        ).squeeze(-1)
        if a8:
            partial = partial * scale[:, block, None]
        result += partial * weight_scale[:, :, block].repeat_interleave(128, 1)
    return result.bfloat16()


def _reference(act, checkpoint, activation, a8):
    q1, s1, q2, s2 = checkpoint
    tokens, top_k = act.topk_ids.shape
    ids = act.topk_ids.long().flatten()
    x = act.hidden_states_q.repeat_interleave(top_k, 0)
    g1 = _block_matmul(x, q1.view(torch.uint8)[ids].view(q1.dtype), s1[ids], a8)
    mid = compute_reference_activation(g1, activation, q2.shape[-1]).bfloat16()
    g2 = _block_matmul(mid, q2.view(torch.uint8)[ids].view(q2.dtype), s2[ids], a8)
    return (
        (g2.float().reshape(tokens, top_k, -1) * act.topk_weights[..., None])
        .sum(1)
        .bfloat16()
    )


def _runner(a8, activation, act, checkpoint, cached=False):
    cfg_type = CuTileDeepSeekFp8Config if a8 else CuTileDeepSeekFp8Bf16Config
    runner_type = CuTileDeepSeekFp8Runner if a8 else CuTileDeepSeekFp8Bf16Runner
    _require_cutile_fp8(cfg_type)
    q1, _, q2, _ = checkpoint
    experts, hidden, inter = q1.shape[0], q1.shape[-1], q2.shape[-1]
    config = MoEConfig(
        routing=RoutingConfig(experts, act.topk_ids.shape[1]),
        quant=QuantConfig(
            QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8 if a8 else QuantFormat.BF16
        ),
        activation=activation,
        experts=ExpertConfig(inter),
        backend=BackendOptions((cfg_type(),)),
        execution=ExecutionConfig(tune_max_num_tokens=max(257, act.num_tokens)),
    )
    view = cfg_type.prepare_weights(
        *checkpoint,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size=inter,
        activation=activation,
        cache_bf16_weights=cached,
    )
    weights = MoEWeightPack()
    weights.prepare_for(runner_type.backend_key, view)
    runner = runner_type(config, act.hidden_states_q.device)
    runner.check_support()
    runner.build()
    return runner, weights


def _case(a8, activation, tokens=7, hidden=256, inter=256):
    _require_cutile_fp8(CuTileDeepSeekFp8Config)
    torch.manual_seed(41)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.rand(tokens, 4, device="cuda").topk(2).indices.int()
    scores = torch.softmax(torch.randn(tokens, 2, device="cuda"), -1)
    w1 = (
        torch.randn(
            4,
            inter * (2 if activation.is_gated else 1),
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.04
    )
    w2 = torch.randn(4, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.04
    q1, s1 = _quantize_weight(w1)
    q2, s2 = _quantize_weight(w2)
    checkpoint = q1, s1, q2, s2
    act = MoEActivationPack(x, None, ids, scores)
    runner, weights = _runner(a8, activation, act, checkpoint)
    return runner, act, weights, checkpoint


@pytest.mark.parametrize("a8", [False, True])
@pytest.mark.parametrize("activation", _CUTILE_ACTIVATIONS, ids=repr)
@pytest.mark.parametrize("tokens", [7, 257])
def test_deepseek_moe_reference(a8, activation, tokens):
    runner, act, weights, checkpoint = _case(a8, activation, tokens)
    actual = runner.forward(runner.pack_inputs(act, weights))
    torch.testing.assert_close(
        actual, _reference(act, checkpoint, activation, a8), atol=0.015, rtol=0.015
    )


@pytest.mark.parametrize("a8", [False, True])
@pytest.mark.parametrize(
    "activation",
    [
        SwiGLU(alpha=1.702, beta=1.0, limit=7.0),
        SiTU(gate_scale=2.0, linear_scale=6.0, clamp_limit=4.0),
        ReLU2(),
    ],
    ids=repr,
)
def test_deepseek_graph_changes_inputs_and_routing(a8, activation, monkeypatch):
    from flashinfer.fused_moe.cutile import deepseek_fp8

    runner, act, weights, checkpoint = _case(a8, activation, 129, 384, 640)
    act.hidden_states_q.mul_(8)
    layer = MoELayer(runner.config, runner.device)
    layer(act, weights)
    monkeypatch.setattr(deepseek_fp8, "needs_int64_indexing", lambda *_: True)
    layer(act, weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(act, weights)
    for shift in (1, 3):
        act.hidden_states_q.mul_(0.75)
        act.topk_ids.copy_((act.topk_ids + shift) % 4)
        act.topk_weights.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(
            actual, _reference(act, checkpoint, activation, a8), atol=0.015, rtol=0.015
        )


def test_deepseek_quantization_finite_values_and_zero():
    _require_cutile_fp8(CuTileDeepSeekFp8Config)
    from flashinfer.fused_moe.cutile.deepseek_fp8 import quantize

    values = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = values.view(torch.bfloat16)
    x = values[torch.isfinite(values)].reshape(-1, 128)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.shape[0], 1, device=x.device)
    quantize(x, q, scales)
    expected_scale = (x.float().abs().amax(-1, keepdim=True) * (1.0 / 448.0)).clamp_min(
        1e-12
    )
    expected_q = (x.float() / expected_scale).clamp(-448, 448).to(q.dtype)
    torch.testing.assert_close(scales, expected_scale, atol=0, rtol=1e-6)
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))


def test_deepseek_shared_weights_and_bf16_cache():
    runner, act, weights, checkpoint = _case(False, SwiGLU())
    expected = runner.forward(runner.pack_inputs(act, weights)).clone()
    cached_runner, cached_weights = _runner(False, SwiGLU(), act, checkpoint, True)
    other, other_weights = _runner(True, SwiGLU(), act, checkpoint)
    view = weights.get_view(runner.backend_key)
    other_view = other_weights.get_view(other.backend_key)
    cached_view = cached_weights.get_view(cached_runner.backend_key)
    assert view["w2"] is other_view["w2"]
    for name in ("w1", "w2"):
        torch.testing.assert_close(
            view[name].float(), other_view[name].float(), atol=0, rtol=0
        )
        torch.testing.assert_close(
            cached_view[f"{name}_bf16"].float(), view[name].float(), atol=0, rtol=0
        )
    plain = runner.pack_inputs(act, weights)
    cached = cached_runner.pack_inputs(act, cached_weights)
    assert runner._stage_cache_key(
        plain, stage=1, block_size=32
    ) != cached_runner._stage_cache_key(cached, stage=1, block_size=32)
    assert runner._stage_cache_key(
        plain, stage=1, block_size=32
    ) != other._stage_cache_key(
        other.pack_inputs(act, other_weights), stage=1, block_size=32
    )
    torch.testing.assert_close(cached_runner.forward(cached), expected, atol=0, rtol=0)


@pytest.mark.parametrize("a8", [False, True])
def test_deepseek_weights_beyond_int32(a8):
    _require_cutile_fp8(CuTileDeepSeekFp8Config)
    from flashinfer.fused_moe.cutile import deepseek_fp8

    experts, hidden, inter = 129, 4096, 2048
    q1 = torch.zeros(
        experts, 2 * inter, hidden, device="cuda", dtype=torch.float8_e4m3fn
    )
    q2 = torch.zeros(experts, hidden, inter, device="cuda", dtype=torch.float8_e4m3fn)
    q1[-1].fill_(0.03125)
    q2[-1].fill_(0.015625)
    s1 = torch.full((experts, 2 * inter // 128, hidden // 128), 1.25, device="cuda")
    s2 = torch.full((experts, hidden // 128, inter // 128), 0.75, device="cuda")
    checkpoint = q1, s1, q2, s2
    x = torch.full((1, hidden), 0.01, device="cuda", dtype=torch.bfloat16)
    act = MoEActivationPack(
        x,
        None,
        torch.full((1, 1), experts - 1, device=x.device, dtype=torch.int32),
        torch.ones(1, 1, device=x.device),
    )
    runner, weights = _runner(a8, SwiGLU(), act, checkpoint)
    torch.testing.assert_close(
        runner.forward(runner.pack_inputs(act, weights)),
        _reference(act, checkpoint, SwiGLU(), a8),
        atol=0.25,
        rtol=0.01,
    )
    packed = x.expand(128, hidden).contiguous()
    scales = torch.empty(128, hidden // 128, device=x.device)
    if a8:
        quantized = torch.empty_like(packed, dtype=torch.float8_e4m3fn)
        deepseek_fp8.quantize(packed, quantized, scales)
        packed = quantized
    view = weights.get_view(runner.backend_key)
    output = torch.empty(128, 2 * inter, device=x.device, dtype=torch.bfloat16)
    offsets = torch.zeros(experts + 1, device=x.device, dtype=torch.int32)
    offsets[-1] = 128
    slots = torch.arange(128, device=x.device, dtype=torch.int32)
    expert_ids = torch.full((1,), experts - 1, device=x.device, dtype=torch.int32)
    live = torch.full((1,), 128, device=x.device, dtype=torch.int32)

    def run():
        deepseek_fp8.grouped_gemm(
            packed,
            scales,
            view["w1"],
            view["w1_scale"],
            slots,
            expert_ids,
            live,
            output,
            top_k=1,
            block_size=128,
            config=deepseek_fp8.GemmConfig(128, 128, 1),
            activation_fp8=a8,
            input_sorted=True,
            output_sorted=True,
            persistent=True,
            expert_offsets=offsets,
        )

    run()
    expected = _block_matmul(x, q1[-1:], s1[-1:], a8)
    torch.testing.assert_close(
        output, expected.expand_as(output), atol=0.015, rtol=0.015
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    expert_ids.zero_()
    offsets[1:].fill_(128)
    graph.replay()
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


@pytest.mark.parametrize("scale_group", [32, 128])
def test_sorted_pack_scale_groups(scale_group):
    _require_cutile_fp8(CuTileDeepSeekFp8Config)
    from flashinfer.fused_moe.cutile.fp8 import _pack_sorted_input

    x = torch.randn(5, 256, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(5, 256 // scale_group, device=x.device)
    slots = torch.tensor([8, 2, 0, 6, 4], device=x.device, dtype=torch.int32)
    out, out_scale = torch.empty_like(x), torch.empty_like(scales)
    _pack_sorted_input(
        x,
        scales,
        slots,
        out,
        out_scale,
        top_k=2,
        block_scaled=True,
        quantize_input=False,
        scale_group_size=scale_group,
    )
    torch.testing.assert_close(
        out.float(), x.float()[slots.long() // 2], atol=0, rtol=0
    )
    torch.testing.assert_close(out_scale, scales[slots.long() // 2], atol=0, rtol=0)
