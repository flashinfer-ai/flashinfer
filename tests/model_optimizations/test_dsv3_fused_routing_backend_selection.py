from types import SimpleNamespace

import pytest
import torch

import flashinfer.fused_moe.fused_routing_dsv3 as fused_routing
from flashinfer.fused_moe.fused_routing_dsv3 import (
    _check_dsv3_fused_routing_supported,
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


def _check_default(num_experts, n_group, topk_group, topk):
    """Run the real default-backend validator on CPU tensors (no GPU needed)."""
    num_tokens = 2
    scores = torch.zeros((num_tokens, num_experts), dtype=torch.float32)
    bias = torch.zeros((num_experts,), dtype=torch.float32)
    # ``topk`` may be non-positive in guard tests, so size the outputs with a
    # non-negative width; a PyTorch allocation error would otherwise mask the
    # production guard.
    topk_width = max(0, topk)
    topk_values = torch.empty((num_tokens, topk_width), dtype=torch.float32)
    topk_indices = torch.empty((num_tokens, topk_width), dtype=torch.int32)
    return _check_dsv3_fused_routing_supported(
        scores,
        bias,
        n_group,
        topk_group,
        topk,
        1.0,
        topk_values,
        topk_indices,
        False,
    )


@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,topk",
    [
        pytest.param(256, 1, 1, 8, id="issue4867-single-group-k8"),
        pytest.param(384, 1, 1, 8, id="kimi-k2-single-group-k8"),
        pytest.param(256, 1, 1, 1, id="single-group-top1"),
        pytest.param(64, 8, 1, 8, id="topk-equals-reachable"),
        pytest.param(96, 3, 1, 5, id="partial-group-capacity"),
        pytest.param(256, 8, 4, 8, id="deepseek-v3-canonical"),
    ],
)
def test_default_backend_accepts_reachable_capacity(
    num_experts, n_group, topk_group, topk
):
    """``topk`` at or below ``topk_group * num_experts / n_group`` is admitted.

    Covers issue #4867 (``256/1/1/8``), which the old ``topk_group * n_group``
    product rejected.
    """
    assert _check_default(num_experts, n_group, topk_group, topk) is True


@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,topk,match",
    [
        pytest.param(
            16,
            8,
            1,
            4,
            r"topk \(4\) must be <= the number of experts reachable",
            id="reachable-two-topk-four",
        ),
        pytest.param(
            4,
            4,
            4,
            8,
            r"topk \(8\) must be <= the number of experts reachable",
            id="reachable-four-topk-eight",
        ),
        pytest.param(
            384,
            1,
            2,
            8,
            r"topk_group \(2\) must be in \[1, n_group \(1\)\]",
            id="topk-group-above-n-group",
        ),
        pytest.param(
            512,
            1,
            1,
            8,
            r"num_experts \(512\) must be <= 384",
            id="experts-above-384",
        ),
        pytest.param(
            96,
            3,
            1,
            32,
            r"topk \(32\) must be <= 8",
            id="topk-above-8",
        ),
        pytest.param(
            255,
            8,
            4,
            8,
            r"num_experts \(255\) must be divisible by n_group \(8\)",
            id="experts-not-divisible",
        ),
        pytest.param(
            256,
            0,
            1,
            8,
            r"n_group \(0\) must be > 0",
            id="n-group-zero",
        ),
        pytest.param(
            256,
            8,
            4,
            0,
            r"topk \(0\) must be > 0",
            id="topk-zero",
        ),
    ],
)
def test_default_backend_rejects_unreachable_capacity(
    num_experts, n_group, topk_group, topk, match
):
    """Configurations outside the reachable-expert capacity must raise.

    Includes ``16/8/1/4``, which the old ``topk_group * n_group`` product wrongly
    admitted, plus the ``n_group``, divisibility and non-positive ``topk``
    guardrails.  Each case asserts the validator names the offending parameter.
    """
    with pytest.raises(ValueError, match=match):
        _check_default(num_experts, n_group, topk_group, topk)


@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,topk,match",
    [
        pytest.param(
            256,
            16,
            4,
            8,
            r"n_group \(16\) must be <= 8",
            id="n-group-above-warp-count",
        ),
        pytest.param(
            64,
            8,
            5,
            8,
            r"topk_group \(5\) must be <= 4",
            id="topk-group-above-tracked-groups",
        ),
        pytest.param(
            8,
            8,
            4,
            4,
            r"num_experts / n_group \(1\) must be >= 2",
            id="experts-per-group-below-top-2",
        ),
    ],
)
def test_default_backend_rejects_grouped_constraint_violations(
    num_experts, n_group, topk_group, topk, match
):
    """Grouped configurations outside the kernel contract must raise.

    Each case asserts the validator reports the offending parameter.
    """
    with pytest.raises(ValueError, match=match):
        _check_default(num_experts, n_group, topk_group, topk)


@pytest.mark.parametrize(
    "num_experts,n_group,topk_group,topk",
    [
        pytest.param(128, 8, 4, 8, id="eight-groups-at-warp-limit"),
        pytest.param(64, 8, 4, 4, id="four-tracked-groups"),
        pytest.param(16, 8, 4, 8, id="two-experts-per-group"),
    ],
)
def test_default_backend_accepts_grouped_constraint_boundaries(
    num_experts, n_group, topk_group, topk
):
    """The grouped constraints stay inclusive at their exact legal boundary.

    ``128/8/4/8`` sits exactly at ``n_group == 8``, ``64/8/4/4`` at
    ``topk_group == 4`` and ``16/8/4/8`` at ``experts_per_group == 2``, so each
    must still be admitted (contrast with the rejects above).
    """
    assert _check_default(num_experts, n_group, topk_group, topk) is True


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
        {"n_group": 8, "num_experts": 16, "topk_group": 1, "topk": 4},
        {"score_dtype": torch.float64},
        {"bias_dtype": torch.float64},
    ],
)
def test_cake_backend_rejects_calls_outside_contract(overrides):
    """Reject unsupported shapes, dtypes, architectures, and reachable capacities."""
    assert not _supported(**overrides)


@pytest.mark.parametrize("num_experts", [2, 128, 256, 384])
def test_cake_backend_accepts_single_group_boundary(num_experts):
    """Admit single-group top-1 routing through the 384-expert capacity limit."""
    assert _supported(
        num_experts=num_experts,
        n_group=1,
        topk_group=1,
        topk=1,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_group": 3, "num_experts": 96, "topk_group": 1, "topk": 5},
        {"n_group": 4, "num_experts": 64, "topk_group": 1, "topk": 8},
        {"n_group": 8, "num_experts": 64, "topk_group": 1, "topk": 8},
    ],
)
def test_cake_backend_accepts_within_group_capacity(overrides):
    """``topk`` may use every expert reachable from the selected groups.

    ``topk_group * num_experts / n_group`` is the real capacity; a ``topk``
    equal to it (``64/8/1/8``) is the boundary case.
    """
    assert _supported(**overrides)


def test_cake_backend_rejects_single_group_above_boundary():
    """Reject single-group calls exceeding the 384-expert kernel capacity."""
    assert not _supported(num_experts=385, n_group=1, topk_group=1, topk=1)


def test_cake_backend_single_group_requires_topk_one():
    """Keep Cake's single-group schedules limited to their one output per token."""
    assert _supported(num_experts=256, n_group=1, topk_group=1, topk=1)
    assert not _supported(num_experts=256, n_group=1, topk_group=1, topk=2)
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
