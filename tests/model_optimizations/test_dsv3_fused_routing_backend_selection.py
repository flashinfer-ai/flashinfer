from types import SimpleNamespace

import pytest
import torch

import flashinfer.fused_moe.fused_routing_dsv3 as fused_routing
from flashinfer.fused_moe.fused_routing_dsv3 import (
    _is_cake_dsv3_fused_routing_supported,
)


def _supported(**overrides):
    params = {
        "capability": (10, 0),
        "num_tokens": 1,
        "num_experts": 256,
        "n_group": 8,
        "topk_group": 4,
        "topk": 8,
        "score_dtype": torch.float32,
        "bias_dtype": torch.bfloat16,
    }
    params.update(overrides)
    return _is_cake_dsv3_fused_routing_supported(**params)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
@pytest.mark.parametrize("score_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cake_backend_accepts_contract_dtype_and_arch_union(
    capability, score_dtype, bias_dtype
):
    assert _supported(
        capability=capability, score_dtype=score_dtype, bias_dtype=bias_dtype
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"capability": (9, 0)},
        {"capability": (12, 0)},
        {"num_tokens": 0},
        {"num_experts": 257},
        {"n_group": 9, "num_experts": 252},
        {"topk_group": 5},
        {"n_group": 8, "num_experts": 8, "topk_group": 4},
        {"n_group": 8, "num_experts": 264},
        {"n_group": 8, "num_experts": 256, "topk": 9},
        {"n_group": 3, "num_experts": 96, "topk_group": 1, "topk": 5},
        {"score_dtype": torch.float64},
        {"bias_dtype": torch.float64},
    ],
)
def test_cake_backend_rejects_calls_outside_contract(overrides):
    assert not _supported(**overrides)


@pytest.mark.parametrize("num_experts", [2, 128, 256, 384])
def test_cake_backend_accepts_single_group_boundary(num_experts):
    assert _supported(
        num_experts=num_experts,
        n_group=1,
        topk_group=1,
        topk=1,
    )


def test_cake_backend_rejects_single_group_above_boundary():
    assert not _supported(num_experts=385, n_group=1, topk_group=1)


def test_cake_backend_preserves_source_single_group_topk_constraint():
    assert not _supported(num_experts=256, n_group=1, topk_group=1, topk=8)


def _selected_backend(monkeypatch, backend=None):
    selected = []

    def fake_get_module(backend="default"):
        selected.append(backend)
        return SimpleNamespace(NoAuxTc=lambda *_args, **_kwargs: None)

    monkeypatch.setattr(fused_routing, "get_dsv3_fused_routing_module", fake_get_module)

    scores = torch.empty((1, 256), dtype=torch.bfloat16)
    bias = torch.empty((256,), dtype=torch.bfloat16)
    kwargs = {}
    if backend is not None:
        kwargs["backend"] = backend
    fused_routing.fused_topk_deepseek(
        scores,
        bias,
        n_group=8,
        topk_group=4,
        topk=8,
        routed_scaling_factor=1.0,
        topk_values=torch.empty((1, 8), dtype=torch.bfloat16),
        topk_indices=torch.empty((1, 8), dtype=torch.int32),
        skip_check=True,
        **kwargs,
    )
    return selected


def test_fused_topk_deepseek_preserves_default_backend(monkeypatch):
    def fail_on_capability_query(*_args, **_kwargs):
        raise AssertionError("default backend must not query Cake capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", fail_on_capability_query)
    assert _selected_backend(monkeypatch) == ["default"]


def test_fused_topk_deepseek_allows_explicit_cake_backend(monkeypatch):
    assert _selected_backend(monkeypatch, backend="cake") == ["cake"]


def test_fused_topk_deepseek_backend_capability_metadata():
    assert fused_routing.fused_topk_deepseek.is_backend_supported("default", 90)
    assert fused_routing.fused_topk_deepseek.is_backend_supported("default", 100)
    assert not fused_routing.fused_topk_deepseek.is_backend_supported("cake", 90)
    assert fused_routing.fused_topk_deepseek.is_backend_supported("cake", 100)
    assert fused_routing.fused_topk_deepseek.is_backend_supported("cake", 103)
