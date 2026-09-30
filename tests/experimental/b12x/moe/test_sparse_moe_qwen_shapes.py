"""Sparse-block API tests with Qwen-like inputs and real model weights."""

from __future__ import annotations

import functools
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.experimental.b12x.benchmark_moe import (
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    load_expert_weights,
    load_gate_weight,
    make_input_activations,
)
from b12x.moe.fused_moe._impl import B12XFP4ExpertWeights, build_tp_moe_route_binding, build_tp_moe_sparse_fp4_binding, clear_tp_moe_caches, plan_b12x_fp4_moe_weights, prepare_b12x_fp4_moe_weights, b12x_route_experts_fast, b12x_sparse_moe_fp4
import b12x.moe.fused_moe._impl as tp_moe_impl
from b12x.testing.reference.helpers import run_tp_moe_fp4


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


@functools.lru_cache(maxsize=1)
def _load_qwen_case() -> tuple[ModelSpec, object, torch.Tensor]:
    spec = _make_spec()
    return spec, load_expert_weights(MODEL_PATH, spec), load_gate_weight(MODEL_PATH, spec)


def _pack_experts(weights) -> B12XFP4ExpertWeights:
    plan = plan_b12x_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format=weights.source_format,
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=weights.spec.num_experts,
        hidden_size=weights.spec.hidden_size,
        intermediate_size=weights.spec.I_tp,
        w13_layout=weights.w13_layout,
    )
    prepared = prepare_b12x_fp4_moe_weights(
        plan=plan,
        w1_global_scale=(
            weights.g1_alphas_per_expert
            * weights.w13_input_scale_quant_per_expert
        ),
        w2_global_scale=(
            weights.g2_alphas_per_expert
            * weights.w2_input_scale_quant_per_expert
        ),
        w1_fp4=weights.w13_weight,
        w1_blockscale=weights.w13_blockscale_swizzled,
        w2_fp4=weights.w2_weight,
        w2_blockscale=weights.w2_blockscale_swizzled,
        a1_gscale=weights.w13_input_scale_quant_per_expert,
        a2_gscale=weights.w2_input_scale_quant_per_expert,
        params_dtype=torch.bfloat16,
    )
    return prepared


def _make_scratch():
    return tp_moe_impl.TPMoEWorkspacePool()


def _manual_route(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_logits = F.linear(hidden_states, gate_weight)
    topk_logits, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits.to(torch.float32), dim=-1)
    return router_logits, topk_ids, topk_weights


def _selected_logits(router_logits: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(router_logits, 1, topk_ids.to(torch.int64))


@pytest.mark.parametrize("m", [1, 23, 80])
def test_route_experts_fast_matches_manual_qwen_gate_path(m: int) -> None:
    _skip_if_unavailable()
    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec, weights, gate_weight = _load_qwen_case()
    hidden_states = make_input_activations(spec, m, seed=9_000 + m, device=device)

    router_logits, topk_ids, topk_weights = _manual_route(hidden_states, gate_weight, spec.top_k)
    del weights

    routing = b12x_route_experts_fast(
        binding=build_tp_moe_route_binding(
            scratch=_make_scratch(),
            hidden_states=hidden_states,
            top_k=spec.top_k,
            gate_weight=gate_weight,
        )
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(routing.router_logits, router_logits)
    assert routing.topk_ids.dtype == torch.int32
    assert routing.flat_ids is not None
    assert routing.flat_weights is not None
    torch.testing.assert_close(
        _selected_logits(router_logits, routing.topk_ids),
        _selected_logits(router_logits, topk_ids),
    )
    torch.testing.assert_close(routing.topk_weights, topk_weights)
    torch.testing.assert_close(routing.flat_ids, routing.topk_ids.view(-1))
    torch.testing.assert_close(routing.flat_weights, routing.topk_weights.view(-1))


def test_route_experts_fast_reuses_exact_workspace_buffers() -> None:
    _skip_if_unavailable()
    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec, weights, gate_weight = _load_qwen_case()
    hidden_states = make_input_activations(spec, 23, seed=30_023, device=device)
    _, topk_ids, _ = _manual_route(hidden_states, gate_weight, spec.top_k)
    del weights, topk_ids
    scratch = _make_scratch()

    first = b12x_route_experts_fast(
        binding=build_tp_moe_route_binding(
            scratch=scratch,
            hidden_states=hidden_states,
            top_k=spec.top_k,
            gate_weight=gate_weight,
        )
    )
    second = b12x_route_experts_fast(
        binding=build_tp_moe_route_binding(
            scratch=scratch,
            hidden_states=hidden_states,
            top_k=spec.top_k,
            gate_weight=gate_weight,
        )
    )

    assert first.router_logits is second.router_logits
    assert first.topk_ids is second.topk_ids
    assert first.topk_weights is second.topk_weights
    assert first.flat_ids is not None and second.flat_ids is not None
    assert first.flat_weights is not None and second.flat_weights is not None
    assert first.flat_ids.data_ptr() == first.topk_ids.view(-1).data_ptr()
    assert first.flat_weights.data_ptr() == first.topk_weights.view(-1).data_ptr()


@pytest.mark.parametrize("m", [1, 23])
def test_sparse_moe_fp4_matches_manual_qwen_gate_path(m: int) -> None:
    _skip_if_unavailable()
    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec, weights, gate_weight = _load_qwen_case()
    experts = _pack_experts(weights)
    hidden_states = make_input_activations(spec, m, seed=10_000 + m, device=device)

    router_logits, topk_ids, topk_weights = _manual_route(hidden_states, gate_weight, spec.top_k)
    sparse_output, routing = b12x_sparse_moe_fp4(
        binding=build_tp_moe_sparse_fp4_binding(
            scratch=_make_scratch(),
            hidden_states=hidden_states,
            experts=experts,
            top_k=spec.top_k,
            gate_weight=gate_weight,
            return_routing=True,
            input_scales_static=True,
        )
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(routing.router_logits, router_logits)
    torch.testing.assert_close(
        _selected_logits(router_logits, routing.topk_ids),
        _selected_logits(router_logits, topk_ids),
    )
    torch.testing.assert_close(routing.topk_weights, topk_weights)
    routed_manual_output = run_tp_moe_fp4(
        a=hidden_states,
        experts=experts,
        topk_weights=routing.topk_weights,
        topk_ids=routing.topk_ids,
        input_scales_static=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(sparse_output, routed_manual_output, atol=5e-4, rtol=1e-2)


@pytest.mark.parametrize("m", [1, 80])
def test_sparse_moe_fp4_matches_manual_qwen_router_logits(m: int) -> None:
    _skip_if_unavailable()
    clear_tp_moe_caches()

    device = torch.device("cuda")
    spec, weights, gate_weight = _load_qwen_case()
    experts = _pack_experts(weights)
    hidden_states = make_input_activations(spec, m, seed=20_000 + m, device=device)

    router_logits, topk_ids, topk_weights = _manual_route(hidden_states, gate_weight, spec.top_k)
    output = torch.empty_like(hidden_states)
    sparse_output, routing = b12x_sparse_moe_fp4(
        binding=build_tp_moe_sparse_fp4_binding(
            scratch=_make_scratch(),
            hidden_states=hidden_states,
            experts=experts,
            top_k=spec.top_k,
            router_logits=router_logits,
            output=output,
            return_routing=True,
            input_scales_static=True,
        )
    )
    torch.cuda.synchronize()

    assert sparse_output is output
    torch.testing.assert_close(
        _selected_logits(router_logits, routing.topk_ids),
        _selected_logits(router_logits, topk_ids),
    )
    torch.testing.assert_close(routing.topk_weights, topk_weights)
    routed_manual_output = run_tp_moe_fp4(
        a=hidden_states,
        experts=experts,
        topk_weights=routing.topk_weights,
        topk_ids=routing.topk_ids,
        input_scales_static=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(sparse_output, routed_manual_output, atol=5e-4, rtol=1e-2)
