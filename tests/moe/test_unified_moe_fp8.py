"""Unified TRTLLM block-FP8 conformance tests."""

from __future__ import annotations

import dataclasses

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
    TrtllmFp8BlockConfig,
)
from flashinfer.quantization.fp8_quantization import (
    mxfp8_dequantize_host,
    mxfp8_quantize,
)
from flashinfer.utils import is_sm100a_supported
from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy


def _is_trtllm_fp8_arch() -> bool:
    return torch.cuda.is_available() and is_sm100a_supported(torch.device("cuda"))


pytestmark = pytest.mark.skipif(
    not _is_trtllm_fp8_arch(), reason="TRTLLM block-FP8 MoE requires SM100/103"
)

HIDDEN = 256
INTERMEDIATE = 256
NUM_EXPERTS = 8
TOP_K = 2
TOKENS = 64


def _deepseek_dequant_activations(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.transpose(0, 1).repeat_interleave(128, dim=1)


def _deepseek_dequant_weights(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)


def _mxfp8_dequant_matrix(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return mxfp8_dequantize_host(
        q.detach().cpu().view(torch.uint8),
        scale.detach().cpu().view(torch.uint8).reshape(-1),
        False,
    ).to(q.device)


def _mxfp8_quant_matrix(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one logical matrix without applying the MoE weight shuffle."""
    q, scale = mxfp8_quantize(x, is_sf_swizzled_layout=False)
    return q, scale.view(torch.uint8).reshape(x.shape[0], x.shape[1] // 32)


def _dequant_view(variant, x_q, x_scale, view, canonical_w1, canonical_w2):
    if variant is QuantVariant.DeepSeekFp8:
        x = _deepseek_dequant_activations(x_q, x_scale)
        w1 = _deepseek_dequant_weights(
            view["gemm1_weights"], view["gemm1_weights_scale"]
        )
        w2 = _deepseek_dequant_weights(
            view["gemm2_weights"], view["gemm2_weights_scale"]
        )
    else:
        x = _mxfp8_dequant_matrix(x_q, x_scale)
        w1 = torch.stack(
            [
                _mxfp8_dequant_matrix(*_mxfp8_quant_matrix(expert))
                for expert in canonical_w1
            ]
        )
        w2 = torch.stack(
            [
                _mxfp8_dequant_matrix(*_mxfp8_quant_matrix(expert))
                for expert in canonical_w2
            ]
        )
    return x.float(), w1.float(), w2.float()


def _requant_intermediate(inter: torch.Tensor, variant) -> torch.Tensor:
    if variant is QuantVariant.DeepSeekFp8:
        q, sf = TrtllmFp8BlockConfig.prepare_activations(
            inter.to(torch.bfloat16), variant=variant
        )
        return _deepseek_dequant_activations(q, sf)
    q, sf = _mxfp8_quant_matrix(inter.to(torch.bfloat16))
    return _mxfp8_dequant_matrix(q, sf)


def _reference(x, w1, w2, ids, weights, variant, expert_offset=0):
    weights = weights.to(torch.bfloat16).float()
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.float32)
    for local_expert in range(w1.shape[0]):
        token, slot = torch.where(ids == local_expert + expert_offset)
        if token.numel() == 0:
            continue
        up = x[token] @ w1[local_expert, :INTERMEDIATE].t()
        gate = x[token] @ w1[local_expert, INTERMEDIATE:].t()
        inter = _requant_intermediate(F.silu(gate) * up, variant)
        expert_out = inter @ w2[local_expert].t()
        out[token] += weights[token, slot, None] * expert_out
    return out


def _assert_fp8_close(actual, expected):
    # Calibrated on the deterministic SM100 cases below: DeepSeek FP8 reached
    # 100% and MXFP8 99.68% within this bound. Recalibrate when shapes expand.
    check_accuracy(expected.float(), actual.float(), atol=0.05, rtol=0.3, percent=0.99)


def _make_case(variant, *, expert_offset=0, local_experts=NUM_EXPERTS):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260717)
    x = torch.randn(
        TOKENS, HIDDEN, device=device, dtype=torch.bfloat16, generator=generator
    )
    w1 = (
        torch.randn(
            local_experts,
            2 * INTERMEDIATE,
            HIDDEN,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            local_experts,
            HIDDEN,
            INTERMEDIATE,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.02
    )
    ids = torch.randint(
        expert_offset,
        expert_offset + local_experts,
        (TOKENS, TOP_K),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    route_weights = torch.softmax(
        torch.randn(
            TOKENS, TOP_K, device=device, dtype=torch.float32, generator=generator
        ),
        dim=-1,
    )

    x_q, x_scale = TrtllmFp8BlockConfig.prepare_activations(x, variant=variant)
    view = TrtllmFp8BlockConfig.prepare_weights(
        w1,
        w2,
        variant=variant,
        num_local_experts=local_experts,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        device=device,
    )
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("trtllm_fp8_block", view)
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=expert_offset + local_experts,
            top_k=TOP_K,
        ),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_expert_offset=expert_offset,
            local_num_experts=local_experts,
        ),
        activation=ActivationConfig.swiglu,
        backend=BackendOptions(candidates=(TrtllmFp8BlockConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=TOKENS),
    )
    pack = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_scale,
        topk_ids=ids,
        topk_weights=route_weights,
    )
    dequant = _dequant_view(variant, x_q, x_scale, view, w1, w2)
    return pack, weight_pack, config, dequant


@pytest.mark.parametrize("variant", [QuantVariant.DeepSeekFp8, QuantVariant.MxFp8])
def test_block_fp8_layer_and_direct_runner_match_reference(variant):
    pack, weights, config, (x, w1, w2) = _make_case(variant)
    reference = _reference(x, w1, w2, pack.topk_ids, pack.topk_weights, variant)
    layer = MoELayer(config)
    runner = layer.runners[0]
    direct = runner.forward(runner.pack_inputs(pack, weights), tactic=-1)
    _assert_fp8_close(direct, reference)
    _assert_fp8_close(layer(pack, weights), reference)


def test_mxfp8_prepared_weight_layout_matches_expected_permutation():
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260718)
    w1 = torch.randn(
        1,
        2 * INTERMEDIATE,
        HIDDEN,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    w2 = torch.randn(
        1,
        HIDDEN,
        INTERMEDIATE,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    view = TrtllmFp8BlockConfig.prepare_weights(
        w1,
        w2,
        variant=QuantVariant.MxFp8,
        num_local_experts=1,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        device=torch.device("cuda"),
    )

    cache = {}
    w1_q, w1_sf = _mxfp8_quant_matrix(w1[0])
    w1_perm = _maybe_get_cached_w3_w1_permute_indices(
        cache, w1_q.view(torch.uint8), 128, is_gated_act_gemm=True
    )
    w1_sf_perm = _maybe_get_cached_w3_w1_permute_indices(
        cache,
        w1_sf,
        128,
        num_elts_per_sf=32,
        is_gated_act_gemm=True,
    )
    torch.testing.assert_close(view["gemm1_weights"][0], w1_q[w1_perm], rtol=0, atol=0)
    torch.testing.assert_close(
        view["gemm1_weights_scale"][0], w1_sf[w1_sf_perm], rtol=0, atol=0
    )

    w2_q, w2_sf = _mxfp8_quant_matrix(w2[0])
    w2_perm = get_w2_permute_indices_with_cache(cache, w2_q.view(torch.uint8), 128)
    w2_sf_perm = get_w2_permute_indices_with_cache(
        cache, w2_sf, 128, num_elts_per_sf=32
    )
    torch.testing.assert_close(view["gemm2_weights"][0], w2_q[w2_perm], rtol=0, atol=0)
    torch.testing.assert_close(
        view["gemm2_weights_scale"][0], w2_sf[w2_sf_perm], rtol=0, atol=0
    )


def _run_from_logits_with_replay(layer, act_pack, weights, expected_ids):
    """Run in-kernel routing and assert its selected expert set exactly."""
    runner = layer.runners[0]
    inputs = runner.pack_inputs(act_pack, weights)
    routing_replay = torch.empty_like(expected_ids, dtype=torch.int16)
    runner._static_kwargs["routing_replay_out"] = routing_replay
    actual = runner.forward(inputs, tactic=-1)
    torch.testing.assert_close(
        torch.sort(routing_replay.to(torch.int32), dim=-1).values,
        torch.sort(expected_ids.to(torch.int32), dim=-1).values,
        rtol=0,
        atol=0,
    )
    return actual


@pytest.mark.parametrize("variant", [QuantVariant.DeepSeekFp8, QuantVariant.MxFp8])
def test_block_fp8_from_logits_matches_prerouted(variant):
    pack, weights, config, _ = _make_case(variant)
    logits = torch.randn(TOKENS, NUM_EXPERTS, device="cuda", dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probabilities, TOP_K, dim=-1)
    prerouted = MoEActivationPack(
        hidden_states_q=pack.hidden_states_q,
        hidden_states_scale=pack.hidden_states_scale,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights,
    )
    from_logits = MoEActivationPack(
        hidden_states_q=pack.hidden_states_q,
        hidden_states_scale=pack.hidden_states_scale,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
    )
    layer = MoELayer(config)
    expected = layer(prerouted, weights).clone()
    actual = _run_from_logits_with_replay(layer, from_logits, weights, topk_ids)
    _assert_fp8_close(actual, expected)


def _deepseek_v3_route(logits, bias, *, top_k, n_group, topk_group, scale):
    scores = torch.sigmoid(logits.float())
    selection_scores = scores + bias.float()
    grouped = selection_scores.view(logits.shape[0], n_group, -1)
    group_scores = torch.topk(grouped, k=2, dim=-1).values.sum(dim=-1)
    selected_groups = torch.topk(group_scores, k=topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
        -1, selected_groups, True
    )
    expert_mask = (
        group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(selection_scores)
    )
    selected = torch.topk(
        selection_scores.masked_fill(~expert_mask, float("-inf")),
        k=top_k,
        dim=-1,
    ).indices
    weights = torch.gather(scores, -1, selected)
    weights = weights / weights.sum(dim=-1, keepdim=True) * scale
    return selected.to(torch.int32), weights


@pytest.mark.parametrize("variant", [QuantVariant.DeepSeekFp8, QuantVariant.MxFp8])
def test_block_fp8_deepseek_v3_from_logits_matches_prerouted(variant):
    num_experts = 64
    pack, weights, config, _ = _make_case(variant, local_experts=num_experts)
    generator = torch.Generator(device="cuda").manual_seed(20260719)
    logits = torch.randn(
        TOKENS,
        num_experts,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    bias = torch.randn(
        num_experts,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    n_group, topk_group, routed_scale = 8, 4, 2.5
    topk_ids, topk_weights = _deepseek_v3_route(
        logits,
        bias,
        top_k=TOP_K,
        n_group=n_group,
        topk_group=topk_group,
        scale=routed_scale,
    )
    config = dataclasses.replace(
        config,
        routing=RoutingConfig(
            num_experts=num_experts,
            top_k=TOP_K,
            method=RoutingMethodType.DeepSeekV3,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scale,
        ),
    )
    prerouted = MoEActivationPack(
        hidden_states_q=pack.hidden_states_q,
        hidden_states_scale=pack.hidden_states_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    from_logits = MoEActivationPack(
        hidden_states_q=pack.hidden_states_q,
        hidden_states_scale=pack.hidden_states_scale,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )
    layer = MoELayer(config)
    expected = layer(prerouted, weights).clone()
    actual = _run_from_logits_with_replay(layer, from_logits, weights, topk_ids)
    _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("variant", [QuantVariant.DeepSeekFp8, QuantVariant.MxFp8])
def test_block_fp8_nonzero_expert_offset(variant):
    offset = 8
    pack, weights, config, (x, w1, w2) = _make_case(
        variant, expert_offset=offset, local_experts=8
    )
    layer = MoELayer(config)
    actual = layer.runners[0].forward(
        layer.runners[0].pack_inputs(pack, weights), tactic=-1
    )
    expected = _reference(
        x,
        w1,
        w2,
        pack.topk_ids,
        pack.topk_weights,
        variant,
        expert_offset=offset,
    )
    assert actual.float().abs().max().item() > 0
    _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("variant", [QuantVariant.DeepSeekFp8, QuantVariant.MxFp8])
def test_block_fp8_prerouted_cuda_graph(variant):
    pack, weights, config, _ = _make_case(variant)
    layer = MoELayer(config)
    for _ in range(3):
        layer(pack, weights)
    eager = layer(pack, weights).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(pack, weights)
    graph.replay()
    torch.cuda.synchronize()
    _assert_fp8_close(captured, eager)
