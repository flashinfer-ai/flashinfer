"""Unified TRTLLM block-scale and per-tensor FP8 conformance tests."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fused_moe import (
    # Typed activation values
    GeGLU,
    ReLU2,
    SwiGLU,
    # Unified configs, packs, and runners
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
from flashinfer.utils import get_compute_capability
from tests.moe.trtllm_gen_fused_moe_utils import check_accuracy
from tests.moe.utils import assert_trtllm_packed_call_contract


def _build_per_tensor_fp8_runner(config):
    runner = TrtllmFp8PerTensorRunner(config, torch.device("cuda"))
    runner.check_support()
    runner.build()
    return runner


def _is_trtllm_fp8_arch() -> bool:
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda")
    ) in ((10, 0), (10, 3))


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


def _block_fp8_reference(
    x,
    w1,
    w2,
    ids,
    weights,
    variant,
    expert_offset=0,
    gemm1_alpha=None,
    gemm1_beta=None,
    gemm1_clamp_limit=None,
    activation=None,
):
    """Dequantized block-FP8 MoE reference.

    ``gemm1_alpha`` / ``gemm1_beta`` / ``gemm1_clamp_limit`` are the optional
    per-expert SwiGLU OA controls; leaving all three unset reproduces plain SwiGLU.
    """
    activation = activation or SwiGLU()
    weights = weights.to(torch.bfloat16).float()
    has_oa = (
        gemm1_alpha is not None
        or gemm1_beta is not None
        or gemm1_clamp_limit is not None
    )
    out = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.float32)
    for local_expert in range(w1.shape[0]):
        token, slot = torch.where(ids == local_expert + expert_offset)
        if token.numel() == 0:
            continue
        fc1 = x[token] @ w1[local_expert].t()
        if isinstance(activation, ReLU2):
            act = F.relu(fc1) ** 2
        else:
            up, gate = fc1[:, :INTERMEDIATE], fc1[:, INTERMEDIATE:]
            if isinstance(activation, GeGLU):
                act = F.gelu(gate) * up
            else:
                if gemm1_clamp_limit is not None:
                    limit = gemm1_clamp_limit[local_expert].float()
                    up = up.clamp(min=-limit, max=limit)
                    gate = gate.clamp(max=limit)
                if has_oa:
                    alpha = (
                        1.0
                        if gemm1_alpha is None
                        else gemm1_alpha[local_expert].float()
                    )
                    beta = (
                        0.0 if gemm1_beta is None else gemm1_beta[local_expert].float()
                    )
                    act = gate * torch.sigmoid(alpha * gate) * (up + beta)
                else:
                    act = F.silu(gate) * up
        inter = _requant_intermediate(act, variant)
        expert_out = inter @ w2[local_expert].t()
        out[token] += weights[token, slot, None] * expert_out
    return out


def _assert_fp8_close(actual, expected):
    # Calibrated on the deterministic SM100 cases below: DeepSeek FP8 reached
    # 100% and MXFP8 99.68% within this bound. Recalibrate when shapes expand.
    check_accuracy(expected.float(), actual.float(), atol=0.05, rtol=0.3, percent=0.99)


def _assert_closer_to(actual, expected, wrong, *, margin=1.25):
    """Assert the output is closer to the requested formula than a wrong one.

    The standard FP8 absolute tolerance exceeds this fixture's output scale, so
    it cannot distinguish GeGLU from SwiGLU. Compare mean errors instead; this
    fixture gives about 1.5x separation, while scaling it up amplifies
    quantization error more than the activation difference.
    """
    correct_error = (actual.float() - expected.float()).abs().mean().item()
    wrong_error = (actual.float() - wrong.float()).abs().mean().item()
    assert correct_error * margin < wrong_error, (
        f"output is not measurably closer to the expected activation: "
        f"correct_error={correct_error:.6f}, wrong_error={wrong_error:.6f} "
        f"(need correct * {margin} < wrong). The runner may be evaluating a "
        f"different activation formula than the one requested."
    )


def _make_block_fp8_case(
    variant, *, activation=None, expert_offset=0, local_experts=NUM_EXPERTS
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260717)
    x = torch.randn(
        TOKENS, HIDDEN, device=device, dtype=torch.bfloat16, generator=generator
    )
    activation = activation or SwiGLU()
    gemm1_rows = INTERMEDIATE * (2 if activation.is_gated else 1)
    w1 = (
        torch.randn(
            local_experts,
            gemm1_rows,
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
        activation=activation,
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
        activation=activation,
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
    inputs = runner.pack_inputs(pack, weights)
    assert_trtllm_packed_call_contract(runner, inputs)
    direct = runner.forward(inputs, tactic=-1)
    _assert_fp8_close(direct, reference)
    _assert_fp8_close(layer(pack, weights), reference)


@pytest.mark.parametrize("activation", (GeGLU(), ReLU2()))
def test_mxfp8_new_activation_layer_and_direct_match_reference(activation):
    pack, weights, config, (x, w1, w2) = _make_block_fp8_case(
        QuantVariant.MxFp8, activation=activation
    )

    def _reference(act):
        return _block_fp8_reference(
            x,
            w1,
            w2,
            pack.topk_ids,
            pack.topk_weights,
            QuantVariant.MxFp8,
            activation=act,
        )

    reference = _reference(activation)
    layer = MoELayer(config)
    runner = layer.runners[0]
    direct = runner.forward(runner.pack_inputs(pack, weights), tactic=-1)
    _assert_fp8_close(direct, reference)
    _assert_fp8_close(layer(pack, weights), reference)

    if activation.is_gated:
        # Compare formulas with matching weight geometry. ReLU2 uses I-row
        # weights, so a gated dispatch is structurally invalid instead.
        _assert_closer_to(direct, reference, _reference(SwiGLU()))


@pytest.mark.parametrize("variant", [QuantVariant.DeepSeekFp8, QuantVariant.MxFp8])
def test_block_fp8_swiglu_oa_params_reach_the_kernel(variant):
    """The unified runner forwards the SwiGLU OA params from the weight view.

    Both block-scale variants consume them, by different routes: MxFp8 in the fused
    FC1 epilogue of the cubins, DeepSeekFp8 in its separate activation kernel.
    """
    pack, weights, config, (x, w1, w2) = _make_block_fp8_case(variant)
    device = pack.hidden_states_q.device
    view = weights.get_view("trtllm_fp8_block")

    def per_expert(value):
        return torch.full((NUM_EXPERTS,), value, device=device, dtype=torch.float32)

    layer = MoELayer(config)
    runner = layer.runners[0]
    baseline = runner.forward(runner.pack_inputs(pack, weights), tactic=-1).clone()

    # FC1 outputs have std ~0.32 for this generator (w1 is 0.02*randn over
    # HIDDEN=256), so 0.3 clamps ~35% of the linear half and ~17% of the gate
    # half rather than being a silent no-op.
    alpha, beta, clamp_limit = per_expert(1.702), per_expert(1.0), per_expert(0.3)
    view["gemm1_alpha"] = alpha
    view["gemm1_beta"] = beta
    view["gemm1_clamp_limit"] = clamp_limit

    reference = _block_fp8_reference(
        x,
        w1,
        w2,
        pack.topk_ids,
        pack.topk_weights,
        variant,
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=clamp_limit,
    )
    actual = runner.forward(runner.pack_inputs(pack, weights), tactic=-1)
    _assert_fp8_close(actual, reference)
    # Guard against the params being accepted and then dropped on the way down.
    assert not torch.allclose(actual.float(), baseline.float(), atol=1e-2, rtol=1e-2)

    # An explicit no-op set (alpha=1, beta=0, limit above the FC1 range) must
    # reproduce the baseline, which pins the neutral-value semantics.
    view["gemm1_alpha"] = per_expert(1.0)
    view["gemm1_beta"] = per_expert(0.0)
    view["gemm1_clamp_limit"] = per_expert(1.0e9)
    noop = runner.forward(runner.pack_inputs(pack, weights), tactic=-1)
    torch.testing.assert_close(noop.float(), baseline.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("key", ["gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"])
def test_block_fp8_swiglu_oa_params_rejected_when_malformed(key):
    """Malformed OA params fail at the runner boundary, naming the offending key."""
    pack, weights, config, _ = _make_block_fp8_case(QuantVariant.DeepSeekFp8)
    runner = MoELayer(config).runners[0]
    view = weights.get_view("trtllm_fp8_block")
    device = pack.hidden_states_q.device

    view[key] = torch.ones(NUM_EXPERTS, device=device, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match=f"{key} must be float32"):
        runner.pack_inputs(pack, weights)

    view[key] = torch.ones(NUM_EXPERTS + 1, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match=f"{key} shape"):
        runner.pack_inputs(pack, weights)


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
    inputs = inputs.with_launch_overrides(routing_replay_out=routing_replay)
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
# Per-tensor FP8 — calibrated E4M3 activations/weights
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
    activation=None,
    wrong_formula: bool = False,
) -> torch.Tensor:
    activation = activation or SwiGLU()
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    routing_weights = routing_weights.to(torch.bfloat16).float()
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
        if wrong_formula:
            # Negative-control path: a genuinely different non-gated formula
            # (plain ReLU instead of ReLU^2) over the same weights and scales.
            intermediate = F.relu(gemm1)
        elif isinstance(activation, ReLU2):
            intermediate = F.relu(gemm1) ** 2
        else:
            up = gemm1[:, :INTERMEDIATE]
            gate = gemm1[:, INTERMEDIATE:]
            intermediate = (
                F.gelu(gate) * up
                if isinstance(activation, GeGLU)
                else F.silu(gate) * up
            )
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
    routing_input_mode: RoutingInputMode = RoutingInputMode.FromLogits,
    routing_method: RoutingMethodType = RoutingMethodType.Default,
    top_k: int = TOP_K,
    num_experts: int = NUM_EXPERTS,
    local_num_experts: int = NUM_EXPERTS,
    local_expert_offset: int = 0,
    activation=None,
    with_wrong_formula_reference: bool = False,
):
    torch.manual_seed(42)
    device = torch.device("cuda")
    x = torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16)
    activation = activation or SwiGLU()
    gemm1_rows = INTERMEDIATE * (2 if activation.is_gated else 1)
    w1 = (
        torch.randn(
            local_num_experts,
            gemm1_rows,
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
        activation=activation,
        device=device,
    )
    if routing_input_mode is RoutingInputMode.FromLogits:
        act = MoEActivationPack(
            hidden_states_q=x_q,
            hidden_states_scale=x_scale,
            routing_input_mode=routing_input_mode,
            routing_logits=logits,
        )
    else:
        assert routing_input_mode in (
            RoutingInputMode.PackedPrecomputed,
            RoutingInputMode.UnpackedPrecomputed,
        )
        act = MoEActivationPack(
            hidden_states_q=x_q,
            hidden_states_scale=x_scale,
            routing_input_mode=routing_input_mode,
            topk_ids=selected_experts,
            topk_weights=routing_weights,
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
        activation=activation,
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
        activation=activation,
    )
    if not with_wrong_formula_reference:
        return act, weights, config, ref, selected_experts
    # Same inputs and scales, deliberately wrong activation formula: the
    # negative control for the permissive tolerance in
    # _assert_per_tensor_fp8_close.
    wrong_ref = _per_tensor_fp8_reference(
        x,
        w1,
        w2,
        selected_experts,
        routing_weights,
        input_scale,
        intermediate_scale,
        expert_offset=local_expert_offset,
        routing_scales_on_input=(routing_method is RoutingMethodType.Llama4),
        activation=activation,
        wrong_formula=True,
    )
    return act, weights, config, ref, selected_experts, wrong_ref


def _assert_per_tensor_fp8_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    check_accuracy(out.float(), ref.float(), atol=0.05, rtol=0.3, percent=0.99)


def _assert_per_tensor_fp8_discriminates(
    out: torch.Tensor, wrong_ref: torch.Tensor
) -> None:
    """Verify FP8 tolerance rejects a genuinely wrong activation formula.

    A rescaled reference is ineffective because these outputs are smaller than
    the absolute tolerance.
    """
    try:
        check_accuracy(
            out.float(), wrong_ref.float(), atol=0.05, rtol=0.3, percent=0.99
        )
    except Exception as exc:
        # Only a tolerance rejection counts as discrimination. check_accuracy
        # also raises on non-finite inputs, and swallowing that would let a NaN
        # or Inf output masquerade as a passing negative control.
        if "Mismatch percentage" not in str(exc):
            raise
        return
    pytest.fail(
        "output also matched a deliberately wrong activation reference; the "
        "tolerance cannot detect a wrong-formula regression."
    )


@pytest.mark.parametrize(
    "routing_input_mode",
    [
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    ],
    ids=["from-logits", "packed", "unpacked"],
)
def test_fp8_per_tensor_layer_and_direct_runner_match_reference(routing_input_mode):
    act, weights, config, ref, _ = _make_per_tensor_fp8_case(
        routing_input_mode=routing_input_mode
    )
    layer_out = MoELayer(config)(act, weights)
    _assert_per_tensor_fp8_close(layer_out, ref)

    runner = _build_per_tensor_fp8_runner(config)
    inputs = runner.pack_inputs(act, weights)
    assert_trtllm_packed_call_contract(runner, inputs)
    direct_out = runner.forward(inputs)
    _assert_per_tensor_fp8_close(direct_out, ref)


@pytest.mark.parametrize("activation", (ReLU2(),))
def test_fp8_per_tensor_new_activation_matches_reference(activation):
    act, weights, config, ref, _, wrong_ref = _make_per_tensor_fp8_case(
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        activation=activation,
        with_wrong_formula_reference=True,
    )
    layer_out = MoELayer(config)(act, weights)
    _assert_per_tensor_fp8_close(layer_out, ref)
    _assert_per_tensor_fp8_discriminates(layer_out, wrong_ref)

    runner = _build_per_tensor_fp8_runner(config)
    direct = runner.forward(runner.pack_inputs(act, weights))
    _assert_per_tensor_fp8_close(direct, ref)


@pytest.mark.parametrize(
    "routing_input_mode",
    [
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    ],
    ids=["from-logits", "packed", "unpacked"],
)
def test_fp8_per_tensor_llama4_routes_scale_on_input(routing_input_mode):
    act, weights, config, ref, _ = _make_per_tensor_fp8_case(
        routing_input_mode=routing_input_mode,
        routing_method=RoutingMethodType.Llama4,
        top_k=1,
    )
    _assert_per_tensor_fp8_close(MoELayer(config)(act, weights), ref)

    runner = _build_per_tensor_fp8_runner(config)
    _assert_per_tensor_fp8_close(runner.forward(runner.pack_inputs(act, weights)), ref)

    invalid_config = dataclasses.replace(
        config,
        routing=dataclasses.replace(config.routing, top_k=2),
    )
    invalid_runner = TrtllmFp8PerTensorRunner.__new__(TrtllmFp8PerTensorRunner)
    invalid_runner.config = invalid_config
    with pytest.raises(ValueError, match="top_k=1"):
        invalid_runner.check_support()


@pytest.mark.parametrize(
    "routing_input_mode",
    [
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    ],
    ids=["from-logits", "packed", "unpacked"],
)
def test_fp8_per_tensor_nonzero_expert_offset(routing_input_mode):
    act, weights, config, ref, _ = _make_per_tensor_fp8_case(
        routing_input_mode=routing_input_mode,
        num_experts=NUM_EXPERTS,
        local_num_experts=NUM_EXPERTS // 2,
        local_expert_offset=NUM_EXPERTS // 2,
    )
    assert torch.count_nonzero(ref)
    _assert_per_tensor_fp8_close(MoELayer(config)(act, weights), ref)

    runner = _build_per_tensor_fp8_runner(config)
    _assert_per_tensor_fp8_close(runner.forward(runner.pack_inputs(act, weights)), ref)


def test_fp8_per_tensor_packed_ids_keep_global_ids_and_weight_bits():
    from flashinfer.fused_moe.core import MoeRunnerInputs

    act, weights, config, _, _ = _make_per_tensor_fp8_case(
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
        num_experts=NUM_EXPERTS,
        local_num_experts=NUM_EXPERTS // 2,
        local_expert_offset=NUM_EXPERTS // 2,
    )
    act.topk_weights[0, 0] = -act.topk_weights[0, 0]
    expected_ids = act.topk_ids.clone()
    expected_bits = (
        act.topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    )

    runner = _build_per_tensor_fp8_runner(config)
    moe_inputs = MoeRunnerInputs.from_list(runner.pack_inputs(act, weights))
    packed = moe_inputs.topk_ids
    assert moe_inputs.expert_weights is not None
    assert moe_inputs.expert_weights.numel() == 0
    assert torch.equal(packed >> 16, expected_ids)
    assert torch.equal(packed & 0xFFFF, expected_bits)


def test_fp8_per_tensor_noncontiguous_packed_routing_matches_reference():
    from flashinfer.fused_moe.core import MoeRunnerInputs

    act, weights, config, ref, _ = _make_per_tensor_fp8_case(
        routing_input_mode=RoutingInputMode.PackedPrecomputed
    )
    act.topk_ids = act.topk_ids.T.contiguous().T
    act.topk_weights = act.topk_weights.T.contiguous().T
    assert not act.topk_ids.is_contiguous()
    assert not act.topk_weights.is_contiguous()

    runner = _build_per_tensor_fp8_runner(config)
    inputs = runner.pack_inputs(act, weights)
    assert MoeRunnerInputs.from_list(inputs).topk_ids.is_contiguous()
    _assert_per_tensor_fp8_close(runner.forward(inputs), ref)


def test_fp8_per_tensor_routing_replay_matches_reference():
    act, weights, config, _, selected_experts = _make_per_tensor_fp8_case()
    runner = _build_per_tensor_fp8_runner(config)
    inputs = runner.pack_inputs(act, weights)
    replay = torch.full(
        (TOKENS, TOP_K), -1, dtype=torch.int16, device=torch.device("cuda")
    )
    inputs = inputs.with_launch_overrides(routing_replay_out=replay)
    runner.forward(inputs)
    torch.testing.assert_close(
        replay.to(torch.int32).sort(dim=-1).values,
        selected_experts.sort(dim=-1).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "routing_input_mode",
    [
        RoutingInputMode.FromLogits,
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    ],
    ids=["from-logits", "packed", "unpacked"],
)
def test_fp8_per_tensor_cuda_graph_replay(routing_input_mode):
    act, weights, config, ref, _ = _make_per_tensor_fp8_case(
        routing_input_mode=routing_input_mode
    )
    runner = _build_per_tensor_fp8_runner(config)
    inputs = runner.pack_inputs(act, weights)
    runner.forward(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_per_tensor_fp8_close(inputs[0], ref)


# ---------------------------------------------------------------------------
# Fused shared experts (non-EP, DeepSeekV3 + FromLogits only)
# ---------------------------------------------------------------------------
#
# Comparing with the legacy flat API isolates the unified plumbing; legacy
# kernel correctness is covered elsewhere. A self-fused pre-routed oracle
# cannot cover checkpoint S=1/S=2 because its declared E+S must be divisible by 4.
# Replay uses top_k stride while the kernel writes top_k + S, so these tests
# compare outputs rather than selected expert ids.

SHARED_EXPERTS_E = 64
SHARED_N_GROUP = 8
SHARED_TOPK_GROUP = 4
SHARED_ROUTED_SCALE = 2.5


def _make_shared_expert_case(
    variant,
    *,
    num_shared,
    activation=None,
    shared_scale=4.0,
    num_experts=SHARED_EXPERTS_E,
):
    """Build a DSv3 case with distinguishable shared rows.

    ``shared_scale=0`` removes only their contribution.
    """
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(20260801)
    rows = num_experts + num_shared

    activation = activation or SwiGLU()
    gemm1_rows = INTERMEDIATE * (2 if activation.is_gated else 1)
    x = torch.randn(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16, generator=gen)
    w1 = (
        torch.randn(
            rows,
            gemm1_rows,
            HIDDEN,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            rows,
            HIDDEN,
            INTERMEDIATE,
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.02
    )
    w1[num_experts:] *= shared_scale
    w2[num_experts:] *= shared_scale

    x_q, x_scale = TrtllmFp8BlockConfig.prepare_activations(x, variant=variant)
    # prepare_weights takes the *physical* row count, which includes the shared
    # experts; the routed-only count lives in RoutingConfig/ExpertConfig.
    view = TrtllmFp8BlockConfig.prepare_weights(
        w1,
        w2,
        variant=variant,
        num_local_experts=rows,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        activation=activation,
        device=device,
    )
    weights = MoEWeightPack()
    weights.prepare_for("trtllm_fp8_block", view)

    logits = torch.randn(
        TOKENS, num_experts, device=device, dtype=torch.bfloat16, generator=gen
    )
    bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16, generator=gen)

    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=num_experts,
            top_k=TOP_K,
            method=RoutingMethodType.DeepSeekV3,
            n_group=SHARED_N_GROUP,
            topk_group=SHARED_TOPK_GROUP,
            routed_scaling_factor=SHARED_ROUTED_SCALE,
        ),
        quant=QuantConfig(variant=variant),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            num_fused_shared_experts=num_shared,
        ),
        activation=activation,
        backend=BackendOptions(candidates=(TrtllmFp8BlockConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=TOKENS),
    )
    pack = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_scale,
        routing_input_mode=RoutingInputMode.FromLogits,
        routing_logits=logits,
        routing_bias=bias,
    )
    return pack, weights, config, view, (logits, bias)


def _legacy_block_fp8_shared(pack, view, logits, bias, config, num_shared):
    """Same launch through the legacy flat API, for cross-checking the unified path."""
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
    from flashinfer.tllm_enums import Fp8QuantizationType

    return trtllm_fp8_block_scale_moe(
        logits,
        bias,
        pack.hidden_states_q,
        pack.hidden_states_scale,
        view["gemm1_weights"],
        view["gemm1_weights_scale"],
        view["gemm2_weights"],
        view["gemm2_weights_scale"],
        config.routing.num_experts,
        config.routing.top_k,
        config.routing.n_group,
        config.routing.topk_group,
        config.experts.intermediate_size,
        0,
        config.routing.num_experts,
        config.routing.routed_scaling_factor,
        routing_method_type=int(RoutingMethodType.DeepSeekV3),
        # The flat API defaults to DeepSeekFp8; MXFP8 uses a different
        # activation-scale layout ([M, H/32] vs [H/128, M]) and the shuffled
        # weight view, so both must be passed for the cross-check to compare
        # the same launch the unified runner performs.
        fp8_quantization_type=(
            Fp8QuantizationType.MxFp8
            if config.quant.variant is QuantVariant.MxFp8
            else Fp8QuantizationType.DeepSeekFp8
        ),
        use_shuffled_weight=config.quant.variant is QuantVariant.MxFp8,
        num_fused_shared_experts=num_shared,
        activation_type=int(config.activation.type),
        gemm1_alpha=view.get("gemm1_alpha"),
        gemm1_beta=view.get("gemm1_beta"),
        gemm1_clamp_limit=view.get("gemm1_clamp_limit"),
    )


@pytest.mark.parametrize(
    "variant,num_shared",
    [
        pytest.param(QuantVariant.DeepSeekFp8, 1, id="deepseek-s1"),
        pytest.param(QuantVariant.MxFp8, 2, id="mxfp8-s2"),
    ],
)
def test_block_fp8_fused_shared_experts_match_legacy(variant, num_shared):
    pack, weights, config, view, (logits, bias) = _make_shared_expert_case(
        variant, num_shared=num_shared
    )
    actual = MoELayer(config)(pack, weights).clone()
    expected = _legacy_block_fp8_shared(pack, view, logits, bias, config, num_shared)
    _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("activation", (GeGLU(), ReLU2()))
def test_mxfp8_new_activations_match_flat_launcher(activation):
    pack, weights, config, view, (logits, bias) = _make_shared_expert_case(
        QuantVariant.MxFp8,
        num_shared=0,
        activation=activation,
    )
    actual = MoELayer(config)(pack, weights).clone()
    expected = _legacy_block_fp8_shared(pack, view, logits, bias, config, 0)
    _assert_fp8_close(actual, expected)


@pytest.mark.parametrize("key", ["gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"])
def test_block_fp8_shared_expert_oa_params_use_physical_rows(key):
    num_shared = 1
    pack, weights, config, view, _ = _make_shared_expert_case(
        QuantVariant.DeepSeekFp8, num_shared=num_shared
    )
    runner = MoELayer(config).runners[0]
    device = pack.hidden_states_q.device

    view[key] = torch.ones(
        SHARED_EXPERTS_E + num_shared, device=device, dtype=torch.float32
    )
    runner.pack_inputs(pack, weights)

    view[key] = torch.ones(SHARED_EXPERTS_E, device=device, dtype=torch.float32)
    with pytest.raises(
        ValueError, match=rf"{key} shape.*expected \({SHARED_EXPERTS_E + num_shared},\)"
    ):
        runner.pack_inputs(pack, weights)


def test_block_fp8_fused_shared_experts_contribute():
    """Verify shared rows contribute independently of the legacy cross-check."""
    variant = QuantVariant.DeepSeekFp8
    num_shared = 1
    pack, weights, config, _, _ = _make_shared_expert_case(
        variant, num_shared=num_shared
    )
    live = MoELayer(config)(pack, weights).clone()

    pack0, weights0, config0, _, _ = _make_shared_expert_case(
        variant, num_shared=num_shared, shared_scale=0.0
    )
    muted = MoELayer(config0)(pack0, weights0).clone()

    rel = (live.float() - muted.float()).abs().max() / (
        muted.float().abs().max() + 1e-6
    )
    assert rel > 0.1, (
        f"S={num_shared}: zeroing the shared rows moved the output by only "
        f"{rel:.4f}; the shared experts are not being applied."
    )


def test_block_fp8_fused_shared_experts_cuda_graph_replay():
    """Verify that static S reaches routing during CUDA graph replay."""
    pack, weights, config, _, _ = _make_shared_expert_case(
        QuantVariant.DeepSeekFp8, num_shared=1
    )
    from flashinfer.fused_moe.runners import TrtllmFp8BlockRunner

    runner = TrtllmFp8BlockRunner(config, torch.device("cuda"))
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(pack, weights)
    eager = runner.forward(inputs).clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_fp8_close(inputs[0], eager)
