"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fused_moe import (
    ActivationConfig,
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
    TrtllmFp8PerTensorConfig,
    TrtllmFp8PerTensorRunner,
)
from flashinfer.utils import is_sm100a_supported
from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy

TOKENS = 64
HIDDEN = 256
INTERMEDIATE = 256
NUM_EXPERTS = 8
TOP_K = 2


def _sm100_required():
    if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("TRTLLM per-tensor FP8 MoE requires SM100/SM103")


def _global_scale(x: torch.Tensor) -> torch.Tensor:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = x.float().abs().amax()
    return torch.where(amax > 0, fp8_max / amax, torch.ones_like(amax))


def _quant_dequant_per_expert(
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = weights.float().abs().amax(dim=(-1, -2))
    scales = torch.where(amax > 0, fp8_max / amax, torch.ones_like(amax))
    quantized = (weights.float() * scales[:, None, None]).clamp(-fp8_max, fp8_max)
    return quantized.to(torch.float8_e4m3fn).float() / scales[:, None, None], scales


def _reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    input_scale: torch.Tensor,
    intermediate_scale: torch.Tensor,
    expert_offset: int = 0,
) -> torch.Tensor:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    x_q = (x.float() * input_scale).clamp(-fp8_max, fp8_max)
    x_deq = x_q.to(torch.float8_e4m3fn).float() / input_scale
    w1_deq, _ = _quant_dequant_per_expert(w1)
    w2_deq, _ = _quant_dequant_per_expert(w2)

    out = torch.zeros_like(x_deq)
    for local_expert in range(w1.shape[0]):
        token, slot = torch.where(selected_experts == local_expert + expert_offset)
        if token.numel() == 0:
            continue
        gemm1 = x_deq[token] @ w1_deq[local_expert].t()
        up = gemm1[:, :INTERMEDIATE]
        gate = gemm1[:, INTERMEDIATE:]
        intermediate = F.silu(gate) * up
        intermediate_q = (intermediate * intermediate_scale).clamp(-fp8_max, fp8_max)
        intermediate_deq = (
            intermediate_q.to(torch.float8_e4m3fn).float() / intermediate_scale
        )
        expert_out = (
            (intermediate_deq @ w2_deq[local_expert].t()).to(torch.bfloat16).float()
        )
        out[token] += routing_weights[token, slot, None] * expert_out
    return out


def _make_case():
    torch.manual_seed(42)
    device = torch.device("cuda")
    x = torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            NUM_EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            device=device,
            dtype=torch.bfloat16,
        )
        / HIDDEN**0.5
    )
    w2 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            device=device,
            dtype=torch.bfloat16,
        )
        / INTERMEDIATE**0.5
    )
    logits = torch.randn(TOKENS, NUM_EXPERTS, device=device, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(
        torch.softmax(logits, dim=-1), TOP_K, dim=-1
    )
    selected_experts = selected_experts.to(torch.int32)

    input_scale = _global_scale(x)
    intermediate_scale = torch.tensor(64.0, device=device)
    x_q, x_scale = TrtllmFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=input_scale
    )
    view = TrtllmFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        hidden_states_scale_global=input_scale,
        intermediate_scale_global=intermediate_scale,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        device=device,
    )
    act = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_scale,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
    )
    weights = MoEWeightPack()
    weights.prepare_for("trtllm_fp8_per_tensor", view)
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            method=RoutingMethodType.Default,
        ),
        quant=QuantConfig(variant=QuantVariant.FP8PerTensor),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_num_experts=NUM_EXPERTS,
        ),
        activation=ActivationConfig.swiglu,
        backend=BackendOptions((TrtllmFp8PerTensorConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=TOKENS),
    )
    ref = _reference(
        x,
        w1,
        w2,
        selected_experts,
        routing_weights,
        input_scale,
        intermediate_scale,
    )
    return act, weights, config, ref, selected_experts


def _assert_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    check_accuracy(out.float(), ref.float(), atol=0.1, rtol=0.85, percent=0.92)


def test_fp8_per_tensor_layer_and_direct_runner_match_reference():
    _sm100_required()
    act, weights, config, ref, _ = _make_case()
    layer_out = MoELayer(config)(act, weights)
    _assert_close(layer_out, ref)

    runner = TrtllmFp8PerTensorRunner(config, torch.device("cuda"))
    inputs = runner.pack_inputs(act, weights)
    direct_out = runner.forward(inputs)
    _assert_close(direct_out, ref)


def test_fp8_per_tensor_routing_replay_matches_reference():
    _sm100_required()
    act, weights, config, _, selected_experts = _make_case()
    runner = TrtllmFp8PerTensorRunner(config, torch.device("cuda"))
    inputs = runner.pack_inputs(act, weights)
    replay = torch.full(
        (TOKENS, TOP_K), -1, dtype=torch.int16, device=torch.device("cuda")
    )
    runner._static_kwargs["routing_replay_out"] = replay
    runner.forward(inputs)
    torch.testing.assert_close(
        replay.to(torch.int32).sort(dim=-1).values,
        selected_experts.sort(dim=-1).values,
        rtol=0,
        atol=0,
    )


def test_fp8_per_tensor_cuda_graph_replay():
    _sm100_required()
    act, weights, config, ref, _ = _make_case()
    runner = TrtllmFp8PerTensorRunner(config, torch.device("cuda"))
    inputs = runner.pack_inputs(act, weights)
    runner.forward(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs)
    graph.replay()
    _assert_close(inputs[0], ref)
