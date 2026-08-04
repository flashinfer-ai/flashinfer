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

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fp4_quantization import fp4_quantize
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
    RoutingInputMode,
    RoutingConfig,
    TrtllmFp4Config,
    TrtllmFp4RoutedRunner,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from flashinfer.quantization.fp4_quantization import block_scale_interleave
from flashinfer.utils import get_compute_capability
from tests.moe.test_cute_dsl_fused_moe import (
    check_accuracy,
    compute_reference_moe_fp4,
)


def _sm100_family() -> bool:
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda")
    ) in ((10, 0), (10, 3), (10, 7))


pytestmark = pytest.mark.skipif(
    not _sm100_family(),
    reason="TRTLLM MXFP4 unified tests require SM100, SM103, or SM107",
)


def _xfail_w4a16_sm103(variant: QuantVariant) -> None:
    if variant is QuantVariant.W4A16 and get_compute_capability(
        torch.device("cuda")
    ) == (10, 3):
        pytest.xfail("TRTLLM MXFP4×BF16 is currently disabled on SM103")


def _expected_mxfp4_weight_view(w1_bf16, w2_bf16):
    """Independent composition of the TRTLLM shuffled-MajorK MXFP4 layout."""
    num_experts, gemm1_rows, hidden_size = w1_bf16.shape
    intermediate_size = w2_bf16.shape[2]
    one = torch.ones(1, device=w1_bf16.device, dtype=torch.float32)
    w1_q, w1_sf = fp4_quantize(
        w1_bf16.reshape(num_experts * gemm1_rows, hidden_size),
        global_scale=one,
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,
    )
    w2_q, w2_sf = fp4_quantize(
        w2_bf16.reshape(num_experts * hidden_size, intermediate_size),
        global_scale=one,
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=False,
    )
    w1_q = w1_q.reshape(num_experts, gemm1_rows, hidden_size // 2).view(torch.uint8)
    w2_q = w2_q.reshape(num_experts, hidden_size, intermediate_size // 2).view(
        torch.uint8
    )
    w1_sf = w1_sf.view(torch.uint8).reshape(num_experts, gemm1_rows, hidden_size // 32)
    w2_sf = w2_sf.view(torch.uint8).reshape(
        num_experts, hidden_size, intermediate_size // 32
    )

    cache = {}
    expected_w1, expected_w1_sf, expected_w2, expected_w2_sf = [], [], [], []
    for expert in range(num_experts):
        w1_permute = _maybe_get_cached_w3_w1_permute_indices(
            cache, w1_q[expert], 128, is_gated_act_gemm=True
        )
        w1_sf_permute = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            w1_sf[expert],
            128,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        w2_permute = get_w2_permute_indices_with_cache(cache, w2_q[expert], 128)
        w2_sf_permute = get_w2_permute_indices_with_cache(
            cache, w2_sf[expert], 128, num_elts_per_sf=16
        )
        expected_w1.append(w1_q[expert][w1_permute.to(w1_q.device)].contiguous())
        expected_w1_sf.append(
            block_scale_interleave(
                w1_sf[expert][w1_sf_permute.to(w1_sf.device)].contiguous()
            )
        )
        expected_w2.append(w2_q[expert][w2_permute.to(w2_q.device)].contiguous())
        expected_w2_sf.append(
            block_scale_interleave(
                w2_sf[expert][w2_sf_permute.to(w2_sf.device)].contiguous()
            )
        )
    return {
        "gemm1_weights": torch.stack(expected_w1),
        "gemm1_weights_scale": torch.stack(expected_w1_sf)
        .reshape(num_experts, gemm1_rows, hidden_size // 32)
        .view(torch.float8_e4m3fn),
        "gemm2_weights": torch.stack(expected_w2),
        "gemm2_weights_scale": torch.stack(expected_w2_sf)
        .reshape(num_experts, hidden_size, intermediate_size // 32)
        .view(torch.float8_e4m3fn),
    }


def _make_runtime_case(
    variant: QuantVariant,
    *,
    num_tokens: int = 8,
    global_experts: int = 8,
    local_experts: int = 8,
    expert_offset: int = 0,
):
    generator = torch.Generator(device="cuda").manual_seed(20260727)
    device = torch.device("cuda")
    hidden_size, intermediate_size, top_k = 1024, 512, 2
    x = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    w1 = (
        torch.randn(
            local_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    w2 = (
        torch.randn(
            local_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.1
    )
    ids = (
        torch.arange(num_tokens, device=device, dtype=torch.int32)[:, None]
        + torch.arange(top_k, device=device, dtype=torch.int32)[None, :]
    ) % local_experts + expert_offset
    route_weights = torch.full(
        (num_tokens, top_k),
        1.0 / top_k,
        device=device,
        dtype=torch.float32,
    )
    x_q, x_sf = TrtllmFp4Config.prepare_activations(x, variant=variant)
    act = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_sf,
        topk_ids=ids,
        topk_weights=route_weights,
    )
    weights = MoEWeightPack()
    weights.prepare_for(
        "trtllm_fp4_routed",
        TrtllmFp4Config.prepare_weights(
            w1,
            w2,
            variant=variant,
            num_local_experts=local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        ),
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=global_experts, top_k=top_k),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_expert_offset=expert_offset,
            local_num_experts=local_experts,
        ),
        activation=ActivationConfig.swiglu,
        backend=BackendOptions(candidates=(TrtllmFp4Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=num_tokens),
    )
    return act, weights, config


@pytest.mark.parametrize("variant", [QuantVariant.MXFP4, QuantVariant.W4A16])
def test_trtllm_mxfp4_unified_matches_reference(variant: QuantVariant):
    _xfail_w4a16_sm103(variant)

    torch.manual_seed(42)
    device = torch.device("cuda")
    num_tokens = 8
    hidden_size = 1024
    intermediate_size = 512
    num_experts = 8
    top_k = 2

    hidden_states_bf16 = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) * 0.1
    )
    w1_bf16 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    w2_bf16 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    topk_ids = (
        torch.arange(num_tokens, device=device, dtype=torch.int32)[:, None]
        + torch.arange(top_k, device=device, dtype=torch.int32)[None, :]
    ) % num_experts
    topk_weights = torch.full(
        (num_tokens, top_k),
        1.0 / top_k,
        device=device,
        dtype=torch.float32,
    )

    hidden_states_q, hidden_states_scale = TrtllmFp4Config.prepare_activations(
        hidden_states_bf16,
        variant=variant,
    )
    act_pack = MoEActivationPack(
        hidden_states_q=hidden_states_q,
        hidden_states_scale=hidden_states_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    weight_pack = MoEWeightPack()
    prepared_weights = TrtllmFp4Config.prepare_weights(
        w1_bf16,
        w2_bf16,
        variant=variant,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
    )
    expected_weights = _expected_mxfp4_weight_view(w1_bf16, w2_bf16)
    for name, expected in expected_weights.items():
        torch.testing.assert_close(
            prepared_weights[name].view(torch.uint8),
            expected.view(torch.uint8),
            rtol=0,
            atol=0,
        )
    weight_pack.prepare_for(
        "trtllm_fp4_routed",
        prepared_weights,
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=ActivationConfig.swiglu,
        backend=BackendOptions(candidates=(TrtllmFp4Config(),)),
    )

    output = MoELayer(config, device=device)(act_pack, weight_pack)
    ones = torch.ones(num_experts, device=device, dtype=torch.float32)
    reference = compute_reference_moe_fp4(
        hidden_states=hidden_states_bf16.float(),
        gemm1_weights=w1_bf16.float(),
        gemm2_weights=w2_bf16.float(),
        gemm1_alpha=ones,
        gemm2_alpha=ones,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fc2_input_scale=torch.ones(1, device=device, dtype=torch.float32),
    )
    passed, pct, atol = check_accuracy(output, reference)
    assert passed, (
        f"{variant.name}: only {pct * 100:.2f}% values within tolerance "
        f"(atol={atol:.4f})"
    )


@pytest.mark.parametrize("variant", [QuantVariant.MXFP4, QuantVariant.W4A16])
def test_trtllm_mxfp4_from_logits_matches_prerouted(variant: QuantVariant):
    _xfail_w4a16_sm103(variant)
    act, weights, config = _make_runtime_case(variant)
    logits = torch.randn(
        act.num_tokens,
        config.routing.num_experts,
        device="cuda",
        dtype=torch.bfloat16,
    )
    topk_weights, topk_ids = torch.topk(
        torch.softmax(logits.float(), dim=-1), config.routing.top_k, dim=-1
    )
    prerouted = MoEActivationPack(
        hidden_states_q=act.hidden_states_q,
        hidden_states_scale=act.hidden_states_scale,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights,
    )
    from_logits = MoEActivationPack(
        hidden_states_q=act.hidden_states_q,
        hidden_states_scale=act.hidden_states_scale,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
    )
    layer = MoELayer(config)
    expected = layer(prerouted, weights).clone()
    actual = layer(from_logits, weights)
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("variant", [QuantVariant.MXFP4, QuantVariant.W4A16])
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
def test_trtllm_mxfp4_unpacked_matches_packed(
    variant: QuantVariant, weights_dtype: torch.dtype
):
    _xfail_w4a16_sm103(variant)
    packed, weights, config = _make_runtime_case(variant)
    unpacked = MoEActivationPack(
        hidden_states_q=packed.hidden_states_q,
        hidden_states_scale=packed.hidden_states_scale,
        topk_ids=packed.topk_ids,
        topk_weights=packed.topk_weights.to(weights_dtype),
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    layer = MoELayer(config)
    expected = layer(packed, weights).clone()
    actual = layer(unpacked, weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", [QuantVariant.MXFP4, QuantVariant.W4A16])
def test_trtllm_mxfp4_nonzero_expert_offset(variant: QuantVariant):
    _xfail_w4a16_sm103(variant)
    baseline_act, baseline_weights, baseline_config = _make_runtime_case(
        variant,
        global_experts=4,
        local_experts=4,
    )
    offset_act, offset_weights, offset_config = _make_runtime_case(
        variant,
        global_experts=12,
        local_experts=4,
        expert_offset=8,
    )
    baseline = MoELayer(baseline_config)(baseline_act, baseline_weights)
    actual = MoELayer(offset_config)(offset_act, offset_weights)
    assert torch.count_nonzero(actual)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("variant", [QuantVariant.MXFP4, QuantVariant.W4A16])
def test_trtllm_mxfp4_cuda_graph_and_autotune(variant: QuantVariant):
    _xfail_w4a16_sm103(variant)
    act, weights, config = _make_runtime_case(variant)
    with autotune(True):
        layer = MoELayer(config)
        for _ in range(3):
            layer(act, weights)
    assert layer.winner_backend == "trtllm_fp4_routed"
    eager = layer(act, weights).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(act, weights)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@pytest.mark.parametrize("variant", [QuantVariant.MXFP4, QuantVariant.W4A16])
@pytest.mark.parametrize(
    "hidden_size,intermediate_size",
    [(160, 128), (128, 160)],
)
def test_trtllm_mxfp4_rejects_unaligned_weights(
    variant: QuantVariant, hidden_size: int, intermediate_size: int
):
    w1 = torch.zeros(
        1, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.zeros(
        1, hidden_size, intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(ValueError, match="divisible by 128"):
        TrtllmFp4Config.prepare_weights(
            w1,
            w2,
            variant=variant,
            num_local_experts=1,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )


@pytest.mark.parametrize(
    "variant,compute_capability,supported",
    [
        (QuantVariant.NVFP4, (10, 0), True),
        (QuantVariant.NVFP4, (10, 3), True),
        (QuantVariant.NVFP4, (10, 7), True),
        (QuantVariant.NVFP4, (12, 0), False),
        (QuantVariant.MXFP4, (10, 0), True),
        (QuantVariant.MXFP4, (10, 3), True),
        (QuantVariant.MXFP4, (10, 7), True),
        (QuantVariant.MXFP4, (12, 0), False),
        (QuantVariant.W4A16, (10, 0), True),
        (QuantVariant.W4A16, (10, 3), False),
        (QuantVariant.W4A16, (10, 7), True),
        (QuantVariant.W4A16, (12, 0), False),
    ],
)
def test_trtllm_fp4_variant_architecture_gates(
    monkeypatch, variant, compute_capability, supported
):
    monkeypatch.setattr(
        "flashinfer.utils.get_compute_capability",
        lambda _device: compute_capability,
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=8, top_k=2),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(intermediate_size=128),
        activation=ActivationConfig.swiglu,
    )
    runner = TrtllmFp4RoutedRunner.__new__(TrtllmFp4RoutedRunner)
    runner.config = config
    runner.device = torch.device("cuda")
    if supported:
        runner.check_support()
    else:
        with pytest.raises(NotImplementedError, match="unsupported"):
            runner.check_support()
