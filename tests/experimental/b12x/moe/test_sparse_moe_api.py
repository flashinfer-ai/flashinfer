from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import b12x.moe.fused_moe._impl as tp_moe
from b12x.moe.fused_moe._impl import (
    B12XFP4ExpertWeights,
    B12XTopKRouting,
    TPMoEFP4Binding,
    b12x_route_experts_fast,
    b12x_sparse_moe_fp4,
    build_tp_moe_route_binding,
    build_tp_moe_sparse_fp4_binding,
    plan_b12x_fp4_moe_weights,
    _PreparedWeightRepresentation,
)


def _make_experts(
    hidden_size: int,
    num_experts: int = 3,
    *,
    source_format: str = "modelopt_nvfp4",
    activation: str = "silu",
    quant_mode: str | None = None,
) -> B12XFP4ExpertWeights:
    from types import SimpleNamespace

    quant_mode = quant_mode or (
        "w4a16" if source_format == "compressed_tensors" else "nvfp4"
    )
    w1_fp4 = torch.zeros(
        num_experts, 4, max(1, hidden_size // 2), dtype=torch.uint8
    )
    w2_fp4 = torch.zeros(num_experts, hidden_size, 1, dtype=torch.uint8)
    w1_alphas = torch.ones(num_experts, dtype=torch.float32)
    w2_alphas = torch.ones(num_experts, dtype=torch.float32)
    plan = plan_b12x_fp4_moe_weights(
        quant_modes=quant_mode,
        source_format=source_format,
        activation=activation,
        params_dtype=torch.float32,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=2,
    )
    layout = plan.required_weight_layout(quant_mode)
    w1_blockscale = torch.zeros(num_experts, 1, dtype=torch.float32)
    w2_blockscale = torch.zeros(num_experts, 1, dtype=torch.float32)
    representation = None
    if layout is not None:
        payload = SimpleNamespace(
            w13=w1_fp4,
            w13_scale=w1_blockscale,
            w13_global_scale=w1_alphas,
            w2=w2_fp4,
            w2_scale=w2_blockscale,
            w2_global_scale=w2_alphas,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=2,
        )
        representation = _PreparedWeightRepresentation(
            quant_mode=quant_mode,
            layout=layout,
            value=payload,
        )
    return B12XFP4ExpertWeights(
        plan=plan,
        a1_gscale=torch.ones(num_experts, dtype=torch.float32),
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a2_gscale=torch.ones(num_experts, dtype=torch.float32),
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        representation=representation,
    )


def _make_scratch():
    return tp_moe.TPMoEWorkspacePool()


def _make_fp4_binding_from_kwargs(**kwargs) -> TPMoEFP4Binding:
    a = kwargs["a"]
    experts = kwargs["experts"]
    topk_ids = kwargs["topk_ids"]
    mode = kwargs.get("quant_mode") or next(iter(experts.plan.quant_modes))
    return TPMoEFP4Binding(
        a=a,
        experts=experts,
        topk_weights=kwargs["topk_weights"],
        topk_ids=topk_ids,
        implementation="test",
        state_E=experts.num_experts,
        weight_E=experts.num_experts,
        max_rows=int(a.shape[0]),
        k=int(a.shape[1]),
        n=experts.intermediate_size,
        num_topk=int(topk_ids.shape[1]),
        device=a.device,
        dtype=a.dtype,
        apply_router_weight_on_input=bool(
            kwargs.get("apply_router_weight_on_input", False)
        ),
        output=kwargs.get("output"),
        input_scales_static=bool(kwargs.get("input_scales_static", False)),
        fast_math=kwargs.get("fast_math"),
        quant_mode=mode,
        unit_scale_contract=bool(kwargs.get("unit_scale_contract", False)),
        swiglu_limit=kwargs.get("swiglu_limit"),
        swiglu_alpha=kwargs.get("swiglu_alpha"),
        swiglu_beta=kwargs.get("swiglu_beta"),
    )


def _make_fp4_binding(
    hidden_states: torch.Tensor,
    experts: B12XFP4ExpertWeights,
    routing: B12XTopKRouting,
    **kwargs,
) -> TPMoEFP4Binding:
    binding_kwargs = {
        "a": hidden_states,
        "experts": experts,
        "topk_weights": routing.topk_weights,
        "topk_ids": routing.topk_ids,
    }
    binding_kwargs.update(kwargs)
    return _make_fp4_binding_from_kwargs(**binding_kwargs)


def _make_sparse_binding(
    hidden_states: torch.Tensor,
    experts: B12XFP4ExpertWeights,
    **kwargs,
):
    return build_tp_moe_sparse_fp4_binding(
        scratch=_make_scratch(),
        hidden_states=hidden_states,
        experts=experts,
        **kwargs,
    )


def test_route_experts_fast_from_gate_weight_renormalizes() -> None:
    hidden_states = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    gate_weight = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 10.0],
            [-1.0, -1.0],
        ],
        dtype=torch.float32,
    )

    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        gate_weight=gate_weight,
    )
    routing = b12x_route_experts_fast(binding=binding)

    assert routing.router_logits is not None
    assert routing.topk_ids.dtype == torch.int32
    assert routing.flat_ids is not None
    assert routing.flat_weights is not None
    assert routing.topk_ids.tolist() == [[0, 1], [1, 0]]
    expected = torch.softmax(
        torch.tensor(
            [
                [10.0, 0.0],
                [10.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    torch.testing.assert_close(routing.topk_weights, expected)
    torch.testing.assert_close(routing.flat_ids, routing.topk_ids.view(-1))
    torch.testing.assert_close(routing.flat_weights, routing.topk_weights.view(-1))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ],
)
def test_route_experts_fast_without_renormalize_returns_topk_logits(device: str) -> None:
    hidden_states = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device=device)
    router_logits = torch.tensor([[0.5, 3.0, -4.0]], dtype=torch.float32, device=device)

    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=2,
        router_logits=router_logits,
        renormalize=False,
    )
    routing = b12x_route_experts_fast(binding=binding)

    assert routing.topk_ids.tolist() == [[1, 0]]
    torch.testing.assert_close(
        routing.topk_weights,
        torch.tensor([[3.0, 0.5]], dtype=torch.float32, device=device),
    )


def test_route_experts_fast_applies_gate_bias() -> None:
    hidden_states = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    gate_weight = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    gate_bias = torch.tensor([5.0, 0.0], dtype=torch.float32)

    binding = build_tp_moe_route_binding(
        hidden_states=hidden_states,
        top_k=1,
        gate_weight=gate_weight,
        gate_bias=gate_bias,
    )
    routing = b12x_route_experts_fast(binding=binding)

    assert routing.topk_ids.tolist() == [[0]]
    torch.testing.assert_close(routing.router_logits, torch.tensor([[5.0, 1.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_experts,top_k", [(384, 6), (128, 3)])
@pytest.mark.parametrize("renormalize", [False, True])
def test_sqrtsoftplus_routing_mixed_modality_unbiased_weights(
    num_experts: int, top_k: int, renormalize: bool
) -> None:
    from b12x.moe import fused_moe

    # Both modalities share logits, but disjoint correction peaks select
    # different experts; neither peak is allowed to contaminate the weights.
    logits = torch.linspace(-8.0, 8.0, num_experts, device="cuda").repeat(4, 1)
    text_bias = torch.zeros(num_experts, device="cuda")
    text_bias[:top_k] = 10.0
    image_bias = torch.zeros_like(text_bias)
    image_bias[num_experts // 2 : num_experts // 2 + top_k] = 10.0
    image_mask = torch.tensor([False, True, True, False], device="cuda")
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    selection = scores + torch.where(image_mask[:, None], image_bias, text_bias)
    expected_ids = selection.topk(top_k, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True) + 1e-20
    expected_weights *= 1.5
    assert not torch.equal(expected_ids[0], scores[0].topk(top_k).indices)
    assert not torch.equal(expected_ids[0], expected_ids[1])

    routing = fused_moe.route(
        binding=fused_moe.bind_route(
            hidden_states=torch.empty(4, 1, device="cuda"),
            router_logits=logits,
            top_k=top_k,
            renormalize=renormalize,
            score_func="sqrtsoftplus",
            correction_bias=text_bias,
            image_correction_bias=image_bias,
            image_mask=image_mask,
            routed_scaling_factor=1.5,
        )
    )
    torch.testing.assert_close(routing.topk_ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(routing.topk_weights, expected_weights, rtol=2e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sqrtsoftplus_top1_and_extreme_logits() -> None:
    from b12x.moe.fused_moe import route_topk

    logits = torch.tensor(
        [[-80.0, -25.0, 0.0, 25.0, 80.0, 1000.0]], device="cuda"
    )
    # Force selection across softplus's tiny-value, transition, and linear regions.
    for expert in range(6):
        bias = torch.zeros(6, device="cuda")
        bias[expert] = 100.0
        selected_logits = torch.empty(1, 1, device="cuda")
        ids = torch.empty(1, 1, dtype=torch.int32, device="cuda")
        weights = torch.empty_like(selected_logits)
        route_topk(
            logits, selected_logits, ids, weights,
            renormalize=True,
            score_func="sqrtsoftplus",
            correction_bias=bias,
            routed_scaling_factor=1.5,
        )
        expected = torch.nn.functional.softplus(logits[:, expert : expert + 1]).sqrt() * 1.5
        torch.testing.assert_close(ids, torch.full_like(ids, expert))
        torch.testing.assert_close(selected_logits, logits[:, expert : expert + 1])
        torch.testing.assert_close(weights, expected, rtol=2e-6, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("score_func", ["softmax", "sqrtsoftplus"])
def test_route_topk_ties_exclude_padded_and_selected_lanes(score_func: str) -> None:
    from b12x.moe.fused_moe import route_topk

    # A 384-expert row is padded to 512. Real scores tie at -inf for the legacy
    # route and zero for sqrtsoftplus; padding must never win either selection.
    logits = torch.full((2, 384), -float("inf"), device="cuda")
    selected_logits = torch.empty(2, 6, device="cuda")
    ids = torch.empty(2, 6, dtype=torch.int32, device="cuda")
    weights = torch.empty_like(selected_logits)
    route_topk(
        logits, selected_logits, ids, weights,
        renormalize=score_func == "sqrtsoftplus",
        score_func=score_func,
        routed_scaling_factor=1.5,
    )
    expected_ids = torch.arange(383, 377, -1, dtype=torch.int32, device="cuda").expand(2, -1)
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(selected_logits, logits[:, :6])
    expected_weights = torch.zeros_like(weights) if score_func == "sqrtsoftplus" else logits[:, :6]
    torch.testing.assert_close(weights, expected_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sqrtsoftplus_routing_graph_replays_modality_and_strided_outputs() -> None:
    from b12x.moe.fused_moe import route_topk

    logits = torch.linspace(-6.0, 6.0, 384, device="cuda").repeat(7, 1)
    text_bias = torch.zeros(384, device="cuda")
    text_bias[:6] = 10.0
    image_bias = text_bias.flip(0)
    image_mask = torch.zeros(7, dtype=torch.bool, device="cuda")
    # Distinct row strides are valid caller-owned scratch layouts.
    selected_logits = torch.empty(7, 9, device="cuda")[:, :6]
    ids = torch.empty(7, 8, dtype=torch.int32, device="cuda")[:, :6]
    weights = torch.empty(7, 10, device="cuda")[:, :6]

    def launch(rows: int) -> None:
        route_topk(
            logits[:rows], selected_logits[:rows], ids[:rows], weights[:rows],
            renormalize=True,
            score_func="sqrtsoftplus",
            correction_bias=text_bias,
            image_correction_bias=image_bias,
            image_mask=image_mask[:rows],
            routed_scaling_factor=1.5,
        )

    launch(1)
    launch(7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(7)
    image_mask[1::2] = True
    logits.add_(0.25)
    graph.replay()
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    selection = scores + torch.where(image_mask[:, None], image_bias, text_bias)
    expected_ids = selection.topk(6, dim=-1).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / (expected_weights.sum(-1, keepdim=True) + 1e-20) * 1.5
    torch.testing.assert_close(ids, expected_ids.to(torch.int32))
    torch.testing.assert_close(selected_logits, logits.gather(1, expected_ids))
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=1e-7)


def test_sparse_moe_fp4_accepts_precomputed_router_logits() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    captured: dict[str, torch.Tensor | object] = {}

    def fake_build_tp_moe_fp4_binding(**kwargs):
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_b12x_moe_fp4(*, binding):
        captured["a"] = binding.a
        captured["topk_weights"] = binding.topk_weights
        captured["topk_ids"] = binding.topk_ids
        if binding.output is None:
            return torch.full_like(binding.a, 7.0)
        binding.output.fill_(7.0)
        return binding.output

    router_logits = torch.tensor(
        [
            [0.5, 3.0, -1.0],
            [2.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    binding = _make_sparse_binding(
        hidden_states,
        experts,
        top_k=2,
        router_logits=router_logits,
        return_routing=True,
    )
    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4),
    ):
        out, routing = b12x_sparse_moe_fp4(binding=binding)

    assert captured["a"] is hidden_states
    assert out.shape == hidden_states.shape
    assert routing.topk_ids.tolist() == [[1, 0], [0, 2]]
    torch.testing.assert_close(captured["topk_ids"], routing.topk_ids)
    torch.testing.assert_close(captured["topk_weights"], routing.topk_weights)


def test_sparse_moe_fp4_forwards_prepared_contract_and_launch_options() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(
        hidden_size=4,
        source_format="compressed_tensors",
        activation="swigluoai_uninterleave",
        quant_mode="w4a16",
    )
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 2, dtype=torch.float32),
        topk_ids=torch.zeros(2, 2, dtype=torch.int64),
    )
    captured: dict[str, object] = {}

    def fake_build_tp_moe_fp4_binding(**kwargs):
        captured["output"] = kwargs.get("output")
        captured["input_scales_static"] = kwargs.get("input_scales_static")
        captured["fast_math"] = kwargs.get("fast_math")
        captured["experts"] = kwargs.get("experts")
        captured["quant_mode"] = kwargs.get("quant_mode")
        captured["swiglu_limit"] = kwargs.get("swiglu_limit")
        captured["swiglu_alpha"] = kwargs.get("swiglu_alpha")
        captured["swiglu_beta"] = kwargs.get("swiglu_beta")
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_b12x_moe_fp4(*, binding):
        if binding.output is None:
            return torch.ones_like(hidden_states)
        binding.output.fill_(1.0)
        return binding.output

    output = torch.empty_like(hidden_states)
    binding = _make_sparse_binding(
        hidden_states,
        experts,
        routing=routing,
        output=output,
        input_scales_static=True,
        fast_math=False,
        quant_mode="w4a16",
        swiglu_limit=5.0,
        swiglu_alpha=1.5,
        swiglu_beta=0.25,
    )
    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4),
    ):
        actual = b12x_sparse_moe_fp4(binding=binding)

    assert actual is output
    assert captured["output"] is output
    assert captured["experts"] is experts
    assert captured["input_scales_static"] is True
    assert captured["fast_math"] is False
    assert captured["quant_mode"] == "w4a16"
    assert captured["swiglu_limit"] == 5.0
    assert captured["swiglu_alpha"] == 1.5
    assert captured["swiglu_beta"] == 0.25




def test_moe_fp4_rejects_compressed_tensors_with_nvfp4() -> None:
    from b12x.moe import fused_moe

    # Reject the incompatible numerical recipe before constructing execution
    # buffers; an incomplete hand-built run binding tests a different error.
    with pytest.raises(ValueError):
        fused_moe.plan_weights(
            quant_modes="nvfp4",
            source_format="compressed_tensors",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=4,
            hidden_size=256,
            intermediate_size=128,
        )


def test_sparse_moe_fp4_rejects_compressed_tensors_with_nvfp4() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4, source_format="compressed_tensors")
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    with pytest.raises(ValueError) as exc_info:
        _make_sparse_binding(
            hidden_states,
            experts,
            routing=routing,
            quant_mode="nvfp4",
        )

    message = str(exc_info.value)
    assert "quant_mode='nvfp4'" in message
    assert "prepared-weight plan ['w4a16']" in message


def test_sparse_moe_fp4_prepared_plan_ignores_runtime_force_env(monkeypatch) -> None:
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    captured: list[object] = []

    def fake_build_tp_moe_fp4_binding(**kwargs):
        captured.append(kwargs.get("quant_mode"))
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_b12x_moe_fp4(*, binding):
        del binding
        return torch.ones_like(hidden_states)

    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4),
    ):
        b12x_sparse_moe_fp4(
            binding=_make_sparse_binding(hidden_states, experts, routing=routing)
        )
        b12x_sparse_moe_fp4(
            binding=_make_sparse_binding(
                hidden_states,
                experts,
                routing=routing,
                quant_mode="nvfp4",
            )
        )

    assert captured == ["nvfp4", "nvfp4"]


def test_sparse_moe_fp4_scales_output_in_place() -> None:
    hidden_states = torch.randn(3, 4)
    experts = _make_experts(hidden_size=4)
    output = torch.empty_like(hidden_states)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(3, 2, dtype=torch.float32),
        topk_ids=torch.zeros(3, 2, dtype=torch.int64),
    )

    def fake_build_tp_moe_fp4_binding(**kwargs):
        return _make_fp4_binding_from_kwargs(**kwargs)

    def fake_b12x_moe_fp4(*, binding):
        assert binding.output is not None
        binding.output.fill_(2.0)
        return binding.output

    binding = _make_sparse_binding(
        hidden_states,
        experts,
        routing=routing,
        output=output,
        routed_scaling_factor=0.25,
    )
    with (
        patch.object(tp_moe, "build_tp_moe_fp4_binding", fake_build_tp_moe_fp4_binding),
        patch.object(tp_moe, "b12x_moe_fp4", fake_b12x_moe_fp4),
    ):
        actual = b12x_sparse_moe_fp4(binding=binding)

    assert actual is output
    torch.testing.assert_close(actual, torch.full_like(hidden_states, 0.5))


def test_sparse_moe_fp4_requires_topk_or_routing() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)

    try:
        _make_sparse_binding(hidden_states, experts)
    except ValueError as exc:
        assert "top_k is required" in str(exc)
    else:
        raise AssertionError("expected missing top_k validation to fire")


def test_sparse_moe_fp4_keeps_routing_path_explicit() -> None:
    hidden_states = torch.randn(2, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )

    try:
        _make_sparse_binding(hidden_states, experts, routing=routing, top_k=1)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("expected routing/top_k exclusivity check to fire")


def test_sparse_moe_fp4_rejects_routing_batch_mismatch() -> None:
    hidden_states = torch.randn(3, 4)
    experts = _make_experts(hidden_size=4)
    routing = B12XTopKRouting(
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int64),
    )
    binding = _make_sparse_binding(hidden_states, experts, routing=routing)

    try:
        b12x_sparse_moe_fp4(binding=binding)
    except ValueError as exc:
        assert "routing batch mismatch" in str(exc)
    else:
        raise AssertionError("expected routing batch validation to fire")
