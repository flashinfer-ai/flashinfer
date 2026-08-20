"""Unified TRTLLM MxInt4 MoE preparation and runner tests."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    RoutingInputMode,
    TrtllmMxInt4Config,
    TrtllmMxInt4RoutedRunner,
)
from flashinfer.fused_moe.core import (
    MoeRunnerInputs,
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from flashinfer.fused_moe.prepare import _mxint4_quantize
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import get_compute_capability


def _build_mxint4_runner(config):
    runner = TrtllmMxInt4RoutedRunner(config, torch.device("cuda"))
    runner.check_support()
    runner.build()
    return runner


def _is_mxint4_arch() -> bool:
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda")
    ) in ((10, 0), (10, 3), (10, 7))


mxint4_required = pytest.mark.skipif(
    not _is_mxint4_arch(),
    reason="Unified TRTLLM MxInt4 MoE requires SM100/SM103/SM107",
)


def _dequant_mxint4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    values = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    values = torch.where(values < 8, values, values - 16).float()
    return values * scales.float().repeat_interleave(32, dim=-1)


def _mxint4_reference(
    x,
    w1,
    w2,
    selected_experts,
    final_scales,
    intermediate_size,
    expert_offset=0,
):
    out = torch.zeros_like(x.float())
    final_scales = final_scales.to(torch.bfloat16).float()
    for local_e in range(w1.shape[0]):
        token, slot = torch.where(selected_experts == local_e + expert_offset)
        if token.numel() == 0:
            continue
        fc1 = x[token].float() @ w1[local_e].float().t()
        inter = F.silu(fc1[:, intermediate_size:]) * fc1[:, :intermediate_size]
        inter = inter.to(torch.bfloat16).float()
        expert_out = (inter @ w2[local_e].float().t()).to(torch.bfloat16).float()
        out[token] += final_scales[token, slot, None] * expert_out
    return out


def _make_case(
    *,
    num_tokens=16,
    hidden_size=256,
    intermediate_size=256,
    num_experts=8,
    top_k=2,
    local_num_experts=None,
    local_expert_offset=0,
    routing_input_mode=RoutingInputMode.PackedPrecomputed,
    routing_method=None,
):
    routing_method = routing_method or RoutingMethodType.Default
    device = torch.device("cuda")
    local_num_experts = local_num_experts or num_experts
    generator = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1 = (
        torch.randn(
            local_num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 8
    )
    w2 = (
        torch.randn(
            local_num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        / 8
    )
    logits = torch.randn(
        num_tokens,
        num_experts,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    if local_num_experts != num_experts:
        logits[:, :local_expert_offset] = -20
        logits[:, local_expert_offset + local_num_experts :] = -20
    routing_bias = None
    n_group = topk_group = None
    routed_scaling_factor = None
    if routing_method is RoutingMethodType.DeepSeekV3:
        from tests.moe.trtllm_gen_fused_moe_utils import noaux_tc_ref

        n_group, topk_group, routed_scaling_factor = 4, 2, 1.0
        routing_bias = torch.randn(
            num_experts,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        scores = noaux_tc_ref(
            logits.float(),
            routing_bias.float(),
            n_group=n_group,
            topk_group=topk_group,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
        )
        scales, selected = torch.topk(scores, top_k, dim=-1)
    else:
        scales, selected = torch.topk(
            torch.softmax(logits.float(), dim=-1), top_k, dim=-1
        )
    selected = selected.to(torch.int32)

    view = TrtllmMxInt4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    q1, s1 = _mxint4_quantize(w1)
    q2, s2 = _mxint4_quantize(w2)
    w1_dequant = _dequant_mxint4(
        q1, s1.reshape(local_num_experts, 2 * intermediate_size, hidden_size // 32)
    )
    w2_dequant = _dequant_mxint4(
        q2, s2.reshape(local_num_experts, hidden_size, intermediate_size // 32)
    )
    reference = _mxint4_reference(
        x,
        w1_dequant,
        w2_dequant,
        selected,
        scales,
        intermediate_size,
        expert_offset=local_expert_offset,
    )

    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=num_experts,
            top_k=top_k,
            method=routing_method,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
        ),
        quant=QuantConfig(variant=QuantVariant.MxInt4),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions((TrtllmMxInt4Config(),)),
    )
    if routing_input_mode is RoutingInputMode.FromLogits:
        act = MoEActivationPack(
            hidden_states_q=x,
            hidden_states_scale=None,
            routing_input_mode=routing_input_mode,
            routing_logits=logits,
            routing_bias=routing_bias,
        )
    else:
        act = MoEActivationPack(
            hidden_states_q=x,
            hidden_states_scale=None,
            topk_ids=selected,
            topk_weights=scales,
        )
    weights = MoEWeightPack()
    weights.prepare_for("trtllm_mxint4_routed", view)
    return act, weights, config, reference, (w1, w2)


def _assert_mxint4_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.05, rtol=0.2)


@pytest.mark.parametrize("bad_dtype", [torch.float16, torch.float32])
def test_mxint4_prepare_rejects_non_bf16(bad_dtype):
    w1 = torch.empty(2, 512, 256, dtype=bad_dtype)
    w2 = torch.empty(2, 256, 256, dtype=bad_dtype)
    with pytest.raises(ValueError, match="BF16"):
        TrtllmMxInt4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=256,
            intermediate_size=256,
        )


def test_mxint4_prepare_rejects_unaligned_geometry():
    w1 = torch.empty(2, 768, 256, dtype=torch.bfloat16)
    w2 = torch.empty(2, 256, 384, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible by 256"):
        TrtllmMxInt4Config.prepare_weights(
            w1,
            w2,
            num_local_experts=2,
            hidden_size=256,
            intermediate_size=384,
        )


@pytest.mark.parametrize(
    ("compute_capability", "supported"),
    [((10, 0), True), ((10, 3), True), ((10, 7), True), ((12, 0), False)],
)
def test_mxint4_runner_arch_support(monkeypatch, compute_capability, supported):
    import flashinfer.utils as utils

    config = MoEConfig(
        routing=RoutingConfig(num_experts=8, top_k=2),
        quant=QuantConfig(variant=QuantVariant.MxInt4),
        experts=ExpertConfig(intermediate_size=256),
        backend=BackendOptions((TrtllmMxInt4Config(),)),
    )
    runner = TrtllmMxInt4RoutedRunner.__new__(TrtllmMxInt4RoutedRunner)
    runner.config = config
    runner.device = torch.device("cuda")
    monkeypatch.setattr(utils, "get_compute_capability", lambda _: compute_capability)
    if supported:
        runner.check_support()
    else:
        with pytest.raises(NotImplementedError, match="SM100/SM103/SM107"):
            runner.check_support()


@mxint4_required
def test_mxint4_prepare_matches_flat_test_layout():
    from tests.moe.trtllm_gen_fused_moe_utils import MxInt4BlockScaleMoe

    act, _, _, _, canonical = _make_case(num_experts=2)
    w1, w2 = canonical
    actual = TrtllmMxInt4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=2,
        hidden_size=256,
        intermediate_size=256,
        device=act.hidden_states_q.device,
        permute_cache={},
    )
    implementation = MxInt4BlockScaleMoe()
    implementation._cache_permute_indices = {}
    quantized = implementation.quantize_weights(w1, w2, act.hidden_states_q)
    args = SimpleNamespace(
        gemm1_weights=quantized["gemm1_weights"],
        gemm2_weights=quantized["gemm2_weights"],
        gemm1_scales=quantized["gemm1_scales"],
        gemm2_scales=quantized["gemm2_scales"],
    )
    expected = implementation.prepare_static_weights_for_kernel(
        None, args, w1, w2, 256, 256, 2, None
    )
    assert torch.equal(actual["gemm1_weights"], expected["gemm1_weights"])
    assert torch.equal(actual["gemm1_weights_scale"], expected["gemm1_scales"])
    assert torch.equal(actual["gemm2_weights"], expected["gemm2_weights"])
    assert torch.equal(actual["gemm2_weights_scale"], expected["gemm2_scales"])


def test_w2_permute_cache_key_includes_epilogue_tile_m():
    weight = torch.empty(128, 64, dtype=torch.uint8)
    cache = {}
    tile_64 = get_w2_permute_indices_with_cache(cache, weight, 64)
    actual_tile_128 = get_w2_permute_indices_with_cache(cache, weight, 128)
    expected_tile_128 = get_w2_permute_indices_with_cache({}, weight, 128)

    assert not torch.equal(tile_64, expected_tile_128)
    assert torch.equal(actual_tile_128, expected_tile_128)


def test_w3_w1_permute_cache_key_includes_gated_activation_mode():
    weight = torch.empty(256, 64, dtype=torch.uint8)
    cache = {}
    gated = _maybe_get_cached_w3_w1_permute_indices(
        cache, weight, 128, is_gated_act_gemm=True
    )
    actual_ungated = _maybe_get_cached_w3_w1_permute_indices(
        cache, weight, 128, is_gated_act_gemm=False
    )
    expected_ungated = _maybe_get_cached_w3_w1_permute_indices(
        {}, weight, 128, is_gated_act_gemm=False
    )

    assert not torch.equal(gated, expected_ungated)
    assert torch.equal(actual_ungated, expected_ungated)


@mxint4_required
@pytest.mark.parametrize(
    "routing_input_mode",
    [RoutingInputMode.PackedPrecomputed, RoutingInputMode.FromLogits],
    ids=["packed", "from-logits"],
)
def test_mxint4_layer_and_direct_runner_match_reference(routing_input_mode):
    act, weights, config, reference, _ = _make_case(
        routing_input_mode=routing_input_mode
    )
    layer_output = MoELayer(config)(act, weights)
    _assert_mxint4_close(layer_output, reference)

    runner = _build_mxint4_runner(config)
    direct_output = runner.forward(runner.pack_inputs(act, weights))
    _assert_mxint4_close(direct_output, reference)


@mxint4_required
def test_mxint4_from_logits_deepseek_routing_bias():
    act, weights, config, reference, _ = _make_case(
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_method=RoutingMethodType.DeepSeekV3,
    )
    _assert_mxint4_close(MoELayer(config)(act, weights), reference)


@mxint4_required
def test_mxint4_nonzero_expert_offset():
    act, weights, config, reference, _ = _make_case(
        num_experts=8, local_num_experts=4, local_expert_offset=4
    )
    _assert_mxint4_close(MoELayer(config)(act, weights), reference)


@mxint4_required
def test_mxint4_explicit_autotune_matches_reference():
    act, weights, config, reference, _ = _make_case()
    layer = MoELayer(config)
    with autotune(True):
        output = layer(act, weights)
    assert layer.winner_backend == "trtllm_mxint4_routed"
    _assert_mxint4_close(output, reference)


@mxint4_required
def test_mxint4_from_logits_supports_fp32():
    act, weights, config, reference, _ = _make_case(
        routing_input_mode=RoutingInputMode.FromLogits
    )
    act.routing_logits = act.routing_logits.float()
    runner = _build_mxint4_runner(config)
    inputs = runner.pack_inputs(act, weights)
    packed = MoeRunnerInputs.from_list(inputs)
    assert packed.topk_ids.numel() == 0
    assert packed.expert_weights.numel() == 0
    output = runner.forward(inputs)
    _assert_mxint4_close(output, reference)


@mxint4_required
def test_mxint4_from_logits_supports_fp32_bias():
    act, weights, config, reference, _ = _make_case(
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_method=RoutingMethodType.DeepSeekV3,
    )
    act.routing_logits = act.routing_logits.float()
    act.routing_bias = act.routing_bias.float()
    runner = _build_mxint4_runner(config)
    output = runner.forward(runner.pack_inputs(act, weights))
    _assert_mxint4_close(output, reference)


@mxint4_required
@pytest.mark.parametrize("field", ["hidden_states_q", "routing_logits", "routing_bias"])
def test_mxint4_runner_rejects_noncontiguous_runtime_inputs(field):
    act, weights, config, _, _ = _make_case(
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_method=RoutingMethodType.DeepSeekV3,
    )
    tensor = getattr(act, field)
    if tensor.dim() == 2:
        tensor = tensor.T.contiguous().T
    else:
        tensor = tensor.repeat_interleave(2)[::2]
    assert not tensor.is_contiguous()
    setattr(act, field, tensor)

    runner = _build_mxint4_runner(config)
    with pytest.raises(ValueError, match=rf"{field} must be contiguous"):
        runner.pack_inputs(act, weights)


@mxint4_required
@pytest.mark.parametrize(
    "key",
    [
        "gemm1_weights",
        "gemm1_weights_scale",
        "gemm2_weights",
        "gemm2_weights_scale",
    ],
)
def test_mxint4_runner_rejects_malformed_prepared_view(key):
    act, weights, config, _, _ = _make_case()
    view = weights.get_view("trtllm_mxint4_routed")
    view[key] = view[key][..., :-1].contiguous()
    runner = _build_mxint4_runner(config)
    with pytest.raises(ValueError, match=rf"{key} shape"):
        runner.pack_inputs(act, weights)


@mxint4_required
@pytest.mark.parametrize("dimension", ["hidden", "intermediate"])
def test_mxint4_runner_rejects_unaligned_runtime_geometry(dimension):
    act, weights, config, _, _ = _make_case()
    if dimension == "hidden":
        act.hidden_states_q = torch.empty(
            act.hidden_states_q.shape[0],
            384,
            dtype=torch.bfloat16,
            device=act.hidden_states_q.device,
        )
    else:
        config = dataclasses.replace(
            config,
            experts=dataclasses.replace(config.experts, intermediate_size=384),
        )
    runner = _build_mxint4_runner(config)
    with pytest.raises(ValueError, match="divisible by 256"):
        runner.pack_inputs(act, weights)


@mxint4_required
@pytest.mark.parametrize(
    ("mutation", "error_type", "match"),
    [
        ("device", ValueError, "is on"),
        ("dtype", TypeError, "must be float32"),
        ("shape", ValueError, "shape"),
        ("contiguous", ValueError, "must be contiguous"),
    ],
)
def test_mxint4_runner_validates_optional_gemm1_params(mutation, error_type, match):
    act, weights, config, _, _ = _make_case()
    view = weights.get_view("trtllm_mxint4_routed")
    num_local_experts = config.experts.local_num_experts or config.routing.num_experts
    if mutation == "device":
        value = torch.ones(num_local_experts, dtype=torch.float32, device="meta")
    elif mutation == "dtype":
        value = torch.ones(
            num_local_experts, dtype=torch.float16, device=act.hidden_states_q.device
        )
    elif mutation == "shape":
        value = torch.ones(
            num_local_experts - 1,
            dtype=torch.float32,
            device=act.hidden_states_q.device,
        )
    else:
        value = torch.ones(
            1, dtype=torch.float32, device=act.hidden_states_q.device
        ).expand(num_local_experts)
        assert not value.is_contiguous()
    view["gemm1_alpha"] = value

    runner = _build_mxint4_runner(config)
    with pytest.raises(error_type, match=match):
        runner.pack_inputs(act, weights)


@mxint4_required
@pytest.mark.parametrize(
    "routing_input_mode",
    [RoutingInputMode.PackedPrecomputed, RoutingInputMode.FromLogits],
    ids=["packed", "from-logits"],
)
def test_mxint4_cuda_graph_replay(routing_input_mode):
    act, weights, config, reference, _ = _make_case(
        routing_input_mode=routing_input_mode
    )
    runner = _build_mxint4_runner(config)
    inputs = runner.pack_inputs(act, weights)
    eager = runner.forward(inputs).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs)
    inputs[0].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(inputs[0]).all()
    torch.testing.assert_close(inputs[0], eager, atol=0.05, rtol=0.01)
    _assert_mxint4_close(inputs[0], reference)
