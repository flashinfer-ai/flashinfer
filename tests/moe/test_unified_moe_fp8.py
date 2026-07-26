"""Unified TRTLLM block-scale and per-tensor FP8 conformance tests."""

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
    TrtllmFp8PerTensorConfig,
    TrtllmFp8PerTensorRunner,
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


def _block_fp8_reference(x, w1, w2, ids, weights, variant, expert_offset=0):
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


def _make_block_fp8_case(variant, *, expert_offset=0, local_experts=NUM_EXPERTS):
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
    pack, weights, config, (x, w1, w2) = _make_block_fp8_case(variant)
    reference = _block_fp8_reference(
        x, w1, w2, pack.topk_ids, pack.topk_weights, variant
    )
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
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

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
        view["gemm1_weights_scale"][0],
        block_scale_interleave(w1_sf[w1_sf_perm].contiguous()).reshape_as(w1_sf),
        rtol=0,
        atol=0,
    )

    w2_q, w2_sf = _mxfp8_quant_matrix(w2[0])
    w2_perm = get_w2_permute_indices_with_cache(cache, w2_q.view(torch.uint8), 128)
    w2_sf_perm = get_w2_permute_indices_with_cache(
        cache, w2_sf, 128, num_elts_per_sf=32
    )
    torch.testing.assert_close(view["gemm2_weights"][0], w2_q[w2_perm], rtol=0, atol=0)
    torch.testing.assert_close(
        view["gemm2_weights_scale"][0],
        block_scale_interleave(w2_sf[w2_sf_perm].contiguous()).reshape_as(w2_sf),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("hidden_size", "intermediate_size"),
    [(64, 128), (128, 64)],
)
def test_mxfp8_preparation_rejects_unshufflable_dimensions(
    hidden_size, intermediate_size
):
    w1 = torch.zeros(
        1,
        2 * intermediate_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w2 = torch.zeros(
        1,
        hidden_size,
        intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="divisible by 128"):
        TrtllmFp8BlockConfig.prepare_weights(
            w1,
            w2,
            variant=QuantVariant.MxFp8,
            num_local_experts=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=torch.device("cuda"),
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
    pack, weights, config, _ = _make_block_fp8_case(variant)
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
    pack, weights, config, _ = _make_block_fp8_case(variant, local_experts=num_experts)
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
    pack, weights, config, (x, w1, w2) = _make_block_fp8_case(
        variant, expert_offset=offset, local_experts=8
    )
    layer = MoELayer(config)
    actual = layer.runners[0].forward(
        layer.runners[0].pack_inputs(pack, weights), tactic=-1
    )
    expected = _block_fp8_reference(
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
    pack, weights, config, _ = _make_block_fp8_case(variant)
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


# ---------------------------------------------------------------------------
# Per-tensor FP8 — calibrated E4M3 activations/weights, FromLogits only
# ---------------------------------------------------------------------------


def _per_tensor_global_scale(x: torch.Tensor) -> torch.Tensor:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = x.float().abs().amax()
    return torch.where(amax > 0, fp8_max / amax, torch.ones_like(amax))


def _per_tensor_quant_dequant_experts(
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    amax = weights.float().abs().amax(dim=(-1, -2))
    scales = torch.where(amax > 0, fp8_max / amax, torch.ones_like(amax))
    quantized = (weights.float() * scales[:, None, None]).clamp(-fp8_max, fp8_max)
    return quantized.to(torch.float8_e4m3fn).float() / scales[:, None, None], scales


def _per_tensor_fp8_reference(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    input_scale: torch.Tensor,
    intermediate_scale: torch.Tensor,
    expert_offset: int = 0,
    routing_scales_on_input: bool = False,
) -> torch.Tensor:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    x_q = (x.float() * input_scale).clamp(-fp8_max, fp8_max)
    x_deq = x_q.to(torch.float8_e4m3fn).float() / input_scale
    w1_deq, _ = _per_tensor_quant_dequant_experts(w1)
    w2_deq, _ = _per_tensor_quant_dequant_experts(w2)

    out = torch.zeros_like(x_deq)
    for local_expert in range(w1.shape[0]):
        token, slot = torch.where(selected_experts == local_expert + expert_offset)
        if token.numel() == 0:
            continue
        routed_x = x_deq[token]
        if routing_scales_on_input:
            routed_x = routed_x * routing_weights[token, slot, None]
        gemm1 = routed_x @ w1_deq[local_expert].t()
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
        if routing_scales_on_input:
            out[token] += expert_out
        else:
            out[token] += routing_weights[token, slot, None] * expert_out
    return out


def _make_per_tensor_fp8_case(
    *,
    routing_method: RoutingMethodType = RoutingMethodType.Default,
    top_k: int = TOP_K,
    num_experts: int = NUM_EXPERTS,
    local_num_experts: int = NUM_EXPERTS,
    local_expert_offset: int = 0,
):
    torch.manual_seed(42)
    device = torch.device("cuda")
    x = torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    w1 = (
        torch.randn(
            local_num_experts,
            2 * INTERMEDIATE,
            HIDDEN,
            device=device,
            dtype=torch.bfloat16,
        )
        / HIDDEN**0.5
    )
    w2 = (
        torch.randn(
            local_num_experts,
            HIDDEN,
            INTERMEDIATE,
            device=device,
            dtype=torch.bfloat16,
        )
        / INTERMEDIATE**0.5
    )
    logits = torch.randn(TOKENS, num_experts, device=device, dtype=torch.float32)
    if routing_method is RoutingMethodType.Llama4:
        routing_weights, selected_experts = torch.topk(
            torch.sigmoid(logits), top_k, dim=-1
        )
    else:
        routing_weights, selected_experts = torch.topk(
            torch.softmax(logits, dim=-1), top_k, dim=-1
        )
    selected_experts = selected_experts.to(torch.int32)

    input_scale = _per_tensor_global_scale(x)
    intermediate_scale = torch.tensor(64.0, device=device)
    x_q, x_scale = TrtllmFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=input_scale
    )
    view = TrtllmFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        hidden_states_scale_global=input_scale,
        intermediate_scale_global=intermediate_scale,
        num_local_experts=local_num_experts,
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
            num_experts=num_experts,
            top_k=top_k,
            method=routing_method,
        ),
        quant=QuantConfig(variant=QuantVariant.FP8PerTensor),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_num_experts=local_num_experts,
            local_expert_offset=local_expert_offset,
        ),
        activation=ActivationConfig.swiglu,
        backend=BackendOptions((TrtllmFp8PerTensorConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=TOKENS),
    )
    ref = _per_tensor_fp8_reference(
        x,
        w1,
        w2,
        selected_experts,
        routing_weights,
        input_scale,
        intermediate_scale,
        expert_offset=local_expert_offset,
        routing_scales_on_input=(routing_method is RoutingMethodType.Llama4),
    )
    return act, weights, config, ref, selected_experts


def _assert_per_tensor_fp8_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    check_accuracy(out.float(), ref.float(), atol=0.05, rtol=0.3, percent=0.99)


def test_fp8_per_tensor_layer_and_direct_runner_match_reference():
    act, weights, config, ref, _ = _make_per_tensor_fp8_case()
    layer_out = MoELayer(config)(act, weights)
    _assert_per_tensor_fp8_close(layer_out, ref)

    runner = TrtllmFp8PerTensorRunner(config, torch.device("cuda"))
    inputs = runner.pack_inputs(act, weights)
    direct_out = runner.forward(inputs)
    _assert_per_tensor_fp8_close(direct_out, ref)


def test_fp8_per_tensor_llama4_routes_scale_on_input():
    act, weights, config, ref, _ = _make_per_tensor_fp8_case(
        routing_method=RoutingMethodType.Llama4,
        top_k=1,
    )
    _assert_per_tensor_fp8_close(MoELayer(config)(act, weights), ref)

    runner = TrtllmFp8PerTensorRunner(config, torch.device("cuda"))
    _assert_per_tensor_fp8_close(runner.forward(runner.pack_inputs(act, weights)), ref)

    invalid_config = dataclasses.replace(
        config,
        routing=dataclasses.replace(config.routing, top_k=2),
    )
    invalid_runner = TrtllmFp8PerTensorRunner.__new__(TrtllmFp8PerTensorRunner)
    invalid_runner.config = invalid_config
    with pytest.raises(ValueError, match="top_k=1"):
        invalid_runner.check_support()


def test_fp8_per_tensor_nonzero_expert_offset():
    act, weights, config, ref, _ = _make_per_tensor_fp8_case(
        num_experts=NUM_EXPERTS,
        local_num_experts=NUM_EXPERTS // 2,
        local_expert_offset=NUM_EXPERTS // 2,
    )
    assert torch.count_nonzero(ref)
    _assert_per_tensor_fp8_close(MoELayer(config)(act, weights), ref)

    runner = TrtllmFp8PerTensorRunner(config, torch.device("cuda"))
    _assert_per_tensor_fp8_close(runner.forward(runner.pack_inputs(act, weights)), ref)


def test_fp8_per_tensor_routing_replay_matches_reference():
    act, weights, config, _, selected_experts = _make_per_tensor_fp8_case()
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
    act, weights, config, ref, _ = _make_per_tensor_fp8_case()
    runner = TrtllmFp8PerTensorRunner(config, torch.device("cuda"))
    inputs = runner.pack_inputs(act, weights)
    runner.forward(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_per_tensor_fp8_close(inputs[0], ref)
