# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import flashinfer.norm as norm
from flashinfer.norm import (
    gated_residual_next_rmsnorm,
    supports_gated_residual_next_rmsnorm,
)

HIDDEN_SIZE = 8192
POST_EPS = 1e-8
NEXT_EPS = 1e-5
NEXT_GAIN_CENTER = 1.0
BACKENDS = ("auto", "cake", "cute")


def _skip_unless_supported(inputs, backend, **kwargs):
    if not supports_gated_residual_next_rmsnorm(*inputs, **kwargs):
        pytest.skip("gated residual next RMSNorm requires SM100, SM103, or SM107")
    if not supports_gated_residual_next_rmsnorm(*inputs, backend=backend, **kwargs):
        pytest.skip(f"backend {backend!r} is not available on this device")


def _inputs(rows: int):
    branch = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    next_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.float32).uniform_(-3, 3)
    beta = torch.sigmoid(gate)
    alpha = torch.sqrt(torch.clamp(1.0 - beta.square(), min=1e-3))
    return branch, residual, next_weight, alpha, beta


def _reference(inputs, post_weight=None, post_gain_center=0.0):
    branch, residual, next_weight, alpha, beta = inputs
    branch_fp32 = branch.float()
    if post_weight is not None:
        branch_fp32 *= post_weight.float() + post_gain_center
    branch_fp32 *= torch.rsqrt(
        branch_fp32.square().mean(dim=-1, keepdim=True) + POST_EPS
    )
    residual_out = alpha * residual + beta * branch_fp32
    ffn_input = residual_out * torch.rsqrt(
        residual_out.square().mean(dim=-1, keepdim=True) + NEXT_EPS
    )
    ffn_input *= next_weight.float() + NEXT_GAIN_CENTER
    return residual_out, ffn_input.to(torch.bfloat16)


def _call(inputs, **kwargs):
    return gated_residual_next_rmsnorm(
        *inputs,
        post_eps=POST_EPS,
        next_eps=NEXT_EPS,
        next_gain_center=NEXT_GAIN_CENTER,
        **kwargs,
    )


def _assert_close(actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.015625, atol=0.03125)


@pytest.mark.parametrize("alpha", [1, 1.0, 0.5, -1.0])
@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (10, 7)])
def test_scalar_alpha_support_and_dispatch(monkeypatch, alpha, capability):
    # Check scalar dispatch without a GPU or allocating an alpha vector.
    monkeypatch.setattr(norm, "_HAS_CUTE_DSL_FUSED_NORM", True)
    monkeypatch.setattr(norm, "_HAS_CAKE_FUSED_NORM", False)
    monkeypatch.setattr(norm, "_is_aligned", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
    monkeypatch.setattr(norm, "_cute_dsl_supports_arch", lambda *_: True)
    calls = []
    monkeypatch.setattr(
        norm,
        "_gated_residual_next_rmsnorm_cute",
        lambda *args: calls.append(args),
        raising=False,
    )
    with FakeTensorMode():
        branch = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        residual = torch.empty(1, HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        weight = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        beta = torch.empty(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        inputs = (branch, residual, weight, alpha, beta)
        assert supports_gated_residual_next_rmsnorm(*inputs)
        residual_out, ffn_input = gated_residual_next_rmsnorm(*inputs)
        assert residual_out.shape == branch.shape
        assert residual_out.dtype == torch.float32
        assert ffn_input.shape == branch.shape
        assert ffn_input.dtype == torch.bfloat16
        assert len(calls) == 1
        assert calls[0][2] is beta
        assert calls[0][-1] == float(alpha)
        assert not supports_gated_residual_next_rmsnorm(*inputs, backend="cake")
        with pytest.raises(ValueError, match="backend 'cake' is not available"):
            gated_residual_next_rmsnorm(*inputs, backend="cake")


@pytest.mark.parametrize("alpha", [1, 0.5, None])
@pytest.mark.parametrize("backend", ["auto", "cake"])
@pytest.mark.parametrize("post_input_scale", [False, True])
def test_cake_dispatch_prefers_the_package(
    monkeypatch, alpha, backend, post_input_scale
):
    # The same dispatch check for the Cake route: auto and cake reach the package, cute keeps the kernel.
    monkeypatch.setattr(norm, "_HAS_CUTE_DSL_FUSED_NORM", True)
    monkeypatch.setattr(norm, "_HAS_CAKE_FUSED_NORM", True)
    monkeypatch.setattr(norm, "_is_aligned", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    monkeypatch.setattr(norm, "_cute_dsl_supports_arch", lambda *_: True)
    monkeypatch.setattr(
        norm,
        "_cake_gated_residual_next_rmsnorm_supported",
        lambda *_, **__: True,
        raising=False,
    )
    cake_calls, cute_calls = [], []
    monkeypatch.setattr(
        norm,
        "_cake_gated_residual_next_rmsnorm",
        lambda *args, **kwargs: cake_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        norm,
        "_gated_residual_next_rmsnorm_cute",
        lambda *args: cute_calls.append(args),
        raising=False,
    )
    with FakeTensorMode():
        branch = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        residual = torch.empty(1, HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        weight = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        beta = torch.empty(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        gate = torch.empty(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        kwargs = (
            {"post_weight": weight, "post_gain_center": 1.0} if post_input_scale else {}
        )
        inputs = (branch, residual, weight, gate if alpha is None else alpha, beta)
        assert supports_gated_residual_next_rmsnorm(
            *inputs, backend=backend, post_weight=kwargs.get("post_weight")
        )
        residual_out, ffn_input = gated_residual_next_rmsnorm(
            *inputs, backend=backend, **kwargs
        )
        assert residual_out.shape == branch.shape and ffn_input.shape == branch.shape
        assert len(cake_calls) == 1 and not cute_calls
        args, keywords = cake_calls[0]
        assert args[3] is (gate if alpha is None else beta) and args[4] is beta
        assert keywords["alpha_scalar"] == (None if alpha is None else float(alpha))
        assert keywords["post_weight"] is kwargs.get("post_weight")
        gated_residual_next_rmsnorm(*inputs, backend="cute", **kwargs)
        assert len(cake_calls) == 1 and len(cute_calls) == 1
        with pytest.raises(ValueError, match="backend must be one of"):
            gated_residual_next_rmsnorm(*inputs, backend="fast", **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("rows", [0, 1, 17, 128, 2048])
@pytest.mark.parametrize("post_gain_center", [None, 0.0, 1.0])
def test_gated_residual_next_rmsnorm_matches_reference(rows, post_gain_center, backend):
    torch.manual_seed(1000 + rows)
    inputs = _inputs(rows)
    kwargs = {}
    if post_gain_center is not None:
        kwargs = {
            "post_weight": torch.randn_like(inputs[2]) * 0.1,
            "post_gain_center": post_gain_center,
        }
    _skip_unless_supported(inputs, backend, post_weight=kwargs.get("post_weight"))
    _assert_close(
        _call(inputs, backend=backend, **kwargs), _reference(inputs, **kwargs)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_gated_residual_next_rmsnorm_keeps_intermediate_fp32(backend):
    inputs = list(_inputs(17))
    _skip_unless_supported(inputs, backend)
    inputs[1].mul_(64.0)
    residual_out, ffn_input = _call(inputs, backend=backend)
    expected_residual, expected_ffn = _reference(inputs)
    rounded_ffn = residual_out.to(torch.bfloat16).float()
    rounded_ffn *= torch.rsqrt(
        rounded_ffn.square().mean(dim=-1, keepdim=True) + NEXT_EPS
    )
    rounded_ffn *= inputs[2].float() + NEXT_GAIN_CENTER
    rounded_ffn = rounded_ffn.to(torch.bfloat16)

    torch.testing.assert_close(residual_out, expected_residual, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ffn_input, expected_ffn, rtol=0.015625, atol=0.03125)
    assert not torch.equal(ffn_input, rounded_ffn)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("alpha", [1.0, 0.5, -1.0])
@pytest.mark.parametrize("rows", [0, 1, 17])
@pytest.mark.parametrize("post_input_scale", [False, True])
def test_scalar_alpha_matches_tensor(alpha, rows, post_input_scale, backend):
    torch.manual_seed(1000 + rows)
    inputs = list(_inputs(rows))
    _skip_unless_supported(inputs, backend)
    kwargs = {}
    if post_input_scale:
        kwargs = {"post_weight": inputs[2], "post_gain_center": 1.0}
    inputs[3] = alpha
    assert supports_gated_residual_next_rmsnorm(
        *inputs, post_weight=kwargs.get("post_weight"), backend=backend
    )
    actual = _call(inputs, backend=backend, **kwargs)
    _assert_close(actual, _reference(inputs, **kwargs))
    inputs[3] = torch.full_like(inputs[4], alpha)
    _assert_close(actual, _call(inputs, backend=backend, **kwargs))


def test_gated_residual_next_rmsnorm_rejects_unsafe_layouts():
    inputs = list(_inputs(1))
    if not supports_gated_residual_next_rmsnorm(*inputs):
        pytest.skip("gated residual next RMSNorm requires SM100, SM103, or SM107")
    strided_weight = torch.empty(2 * HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")[
        ::2
    ]
    assert not supports_gated_residual_next_rmsnorm(
        inputs[0], inputs[1], strided_weight, inputs[3], inputs[4]
    )
    misaligned_weight = torch.empty(
        HIDDEN_SIZE + 1, dtype=torch.bfloat16, device="cuda"
    )[1:]
    for post_weight in (
        strided_weight,
        misaligned_weight,
        inputs[2][:-1],
        inputs[2].float(),
        inputs[2].cpu(),
    ):
        assert not supports_gated_residual_next_rmsnorm(
            *inputs, post_weight=post_weight
        )
        with pytest.raises(ValueError, match="contract"):
            _call(inputs, post_weight=post_weight)

    storage = torch.empty(HIDDEN_SIZE + 16, dtype=torch.float32, device="cuda")
    misaligned_residual_out = storage[4 : 4 + HIDDEN_SIZE].view(1, HIDDEN_SIZE)
    assert misaligned_residual_out.data_ptr() % 32 != 0
    with pytest.raises(ValueError, match="residual_out must match"):
        gated_residual_next_rmsnorm(
            *inputs,
            post_eps=POST_EPS,
            next_eps=NEXT_EPS,
            next_gain_center=NEXT_GAIN_CENTER,
            residual_out=misaligned_residual_out,
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("post_input_scale", [False, True])
@pytest.mark.parametrize("alpha", [None, 1.0, 0.5])
def test_gated_residual_next_rmsnorm_cuda_graph_uses_changed_inputs(
    post_input_scale, alpha, backend
):
    inputs = list(_inputs(17))
    _skip_unless_supported(inputs, backend)
    if alpha is not None:
        inputs[3] = alpha
    kwargs = {}
    if post_input_scale:
        kwargs = {
            "post_weight": torch.randn_like(inputs[2]) * 0.1,
            "post_gain_center": 1.0,
        }
    _call(inputs, backend=backend, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _call(inputs, backend=backend, **kwargs)

    for seed in (3001, 4001):
        torch.manual_seed(seed)
        inputs[0].copy_(torch.randn_like(inputs[0]))
        inputs[1].copy_(torch.randn_like(inputs[1]))
        if post_input_scale:
            kwargs["post_weight"].copy_(torch.randn_like(inputs[2]) * 0.1)
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(actual, _reference(inputs, **kwargs))
