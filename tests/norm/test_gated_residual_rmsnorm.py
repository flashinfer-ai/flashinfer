# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import flashinfer.norm as norm
from flashinfer.norm import (
    gated_residual_rmsnorm,
    supports_gated_residual_rmsnorm,
)

HIDDEN_SIZE = 8192
EPS = 1e-5
BACKENDS = ("auto", "cake", "cute")


def _skip_unless_supported(inputs, backend):
    if not supports_gated_residual_rmsnorm(*inputs):
        pytest.skip(
            "gated residual RMSNorm requires BF16/FP32 tensors on SM100, SM103, or SM107"
        )
    if not supports_gated_residual_rmsnorm(*inputs, backend=backend):
        pytest.skip(f"backend {backend!r} is not available on this device")


def _inputs(rows: int):
    branch = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.float32).uniform_(-3, 3)
    beta = torch.sigmoid(gate)
    alpha = torch.sqrt(torch.clamp(1.0 - beta.square(), min=1e-3))
    return branch, residual, weight, alpha, beta


def _misaligned(shape, dtype):
    numel = 1
    for size in shape:
        numel *= size
    tensor = torch.empty(numel + 1, dtype=dtype, device="cuda")[1:].view(shape)
    assert tensor.is_contiguous()
    assert tensor.data_ptr() % 16 != 0
    return tensor


def _reference(inputs, weight_before_norm: bool):
    branch, residual, weight, alpha, beta = inputs
    values = branch.float()
    if weight_before_norm:
        values *= weight.float() + 1.0
    values *= torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPS)
    return alpha * residual + beta * values


def _api_inputs(inputs, weight_before_norm: bool):
    api_inputs = list(inputs)
    if not weight_before_norm:
        api_inputs[2] = None
    return api_inputs


def _kwargs(weight_before_norm: bool):
    return {"eps": EPS, "gain_center": 1.0 if weight_before_norm else 0.0}


def test_gated_residual_rmsnorm_supports_supported_devices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("SM100, SM103, or SM107 is required")
    inputs = _inputs(1)
    assert supports_gated_residual_rmsnorm(*inputs)
    assert supports_gated_residual_rmsnorm(
        *inputs, backend="cute"
    ) or supports_gated_residual_rmsnorm(*inputs, backend="cake")


@pytest.mark.parametrize("weight_before_norm", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_gated_residual_rmsnorm_dispatch(
    monkeypatch, weight_before_norm, backend, enable_pdl
):
    # Dispatch without a GPU: auto and cake reach the Cake package (auto only where the
    # package admits the row count), cute keeps the CuTe DSL kernel, the output is bound.
    monkeypatch.setattr(norm, "_HAS_CUTE_DSL_FUSED_NORM", True)
    monkeypatch.setattr(norm, "_HAS_CAKE_FUSED_NORM", True)
    monkeypatch.setattr(norm, "_is_aligned", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    monkeypatch.setattr(norm, "_cute_dsl_supports_arch", lambda *_: True)
    supported_rows = {None, 1}
    monkeypatch.setattr(norm, "device_support_pdl", lambda *_: True)

    def supported(device, *, weight_before_norm, enable_pdl, rows=None):
        return rows in supported_rows

    monkeypatch.setattr(
        norm, "_cake_gated_residual_rmsnorm_supported", supported, raising=False
    )
    cake_calls, cute_calls = [], []
    monkeypatch.setattr(
        norm,
        "_cake_gated_residual_rmsnorm",
        lambda *args, **kwargs: cake_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        norm,
        "_gated_residual_rmsnorm_cute",
        lambda *args: cute_calls.append(args),
        raising=False,
    )
    with FakeTensorMode():
        branch = torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        residual = torch.empty(1, HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        weight = torch.empty(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        alpha = torch.empty(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        beta = torch.empty(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        inputs = (branch, residual, weight if weight_before_norm else None, alpha, beta)
        kwargs = {**_kwargs(weight_before_norm), "enable_pdl": enable_pdl}
        assert supports_gated_residual_rmsnorm(*inputs, backend=backend)
        out = gated_residual_rmsnorm(*inputs, backend=backend, **kwargs)
        assert out.shape == branch.shape and out.dtype == torch.float32
        if backend == "cute":
            assert not cake_calls and len(cute_calls) == 1
            assert cute_calls[0][7] == int(weight_before_norm)
            assert cute_calls[0][9] is enable_pdl
        else:
            assert len(cake_calls) == 1 and not cute_calls
            args, keywords = cake_calls[0]
            assert args[0] is branch and args[1] is residual
            assert args[2] is (weight if weight_before_norm else None)
            assert args[3] is alpha and args[4] is beta and args[5] is out
            assert keywords == {
                "eps": EPS,
                "gain_center": kwargs["gain_center"],
                "enable_pdl": enable_pdl,
            }
        # Two rows lie in a gap of the package's windows: auto falls to the CuTe DSL
        # kernel; an explicit cake still reaches the package (which raises there).
        two = torch.empty(2, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        two_residual = torch.empty(2, HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        gap_inputs = (two, two_residual, *inputs[2:])
        before = (len(cake_calls), len(cute_calls))
        gated_residual_rmsnorm(*gap_inputs, backend=backend, **kwargs)
        after = (len(cake_calls), len(cute_calls))
        if backend == "cake":
            assert after == (before[0] + 1, before[1])
        else:
            assert after == (before[0], before[1] + 1)
        # The support predicates answer for the device, the variant and the storage
        # (cake) or for the call (auto, cute): all three stay true at the gap rows,
        # which auto and cute run and an explicit cake call rejects.
        for predicate_backend in ("auto", "cake", "cute"):
            assert supports_gated_residual_rmsnorm(
                *gap_inputs, backend=predicate_backend
            )
        with pytest.raises(ValueError, match="backend must be one of"):
            gated_residual_rmsnorm(*inputs, backend="fast", **kwargs)
        monkeypatch.setattr(norm, "_HAS_CAKE_FUSED_NORM", False)
        assert not supports_gated_residual_rmsnorm(*inputs, backend="cake")
        with pytest.raises(ValueError, match="backend 'cake' is not available"):
            gated_residual_rmsnorm(*inputs, backend="cake", **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weight_before_norm", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 17, 128])
def test_gated_residual_rmsnorm_matches_reference(rows, weight_before_norm, backend):
    inputs = _inputs(rows)
    _skip_unless_supported(inputs, backend)
    expected = _reference(inputs, weight_before_norm)
    output = gated_residual_rmsnorm(
        *_api_inputs(inputs, weight_before_norm),
        backend=backend,
        **_kwargs(weight_before_norm),
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_gated_residual_rmsnorm_rejects_wrong_hidden_size():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    inputs = list(_inputs(1))
    inputs[0] = inputs[0][:, :-1]
    with pytest.raises(ValueError, match="contract"):
        gated_residual_rmsnorm(*inputs)


def test_gated_residual_rmsnorm_rejects_unsafe_layouts():
    inputs = list(_inputs(1))
    if not supports_gated_residual_rmsnorm(*inputs):
        pytest.skip(
            "gated residual RMSNorm requires BF16/FP32 tensors on SM100, SM103, or SM107"
        )

    strided_weight = torch.empty(2 * HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")[
        ::2
    ]
    assert not strided_weight.is_contiguous()
    assert not supports_gated_residual_rmsnorm(
        inputs[0], inputs[1], strided_weight, inputs[3], inputs[4]
    )

    misaligned_branch = _misaligned((1, HIDDEN_SIZE), torch.bfloat16)
    assert not supports_gated_residual_rmsnorm(
        misaligned_branch, inputs[1], inputs[2], inputs[3], inputs[4]
    )

    misaligned_out = _misaligned((1, HIDDEN_SIZE), torch.float32)
    with pytest.raises(ValueError, match="out must match"):
        gated_residual_rmsnorm(*inputs, out=misaligned_out)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_gated_residual_rmsnorm_cuda_graph(weight_before_norm, backend):
    inputs = list(_inputs(17))
    _skip_unless_supported(inputs, backend)
    kwargs = _kwargs(weight_before_norm)
    api_inputs = _api_inputs(inputs, weight_before_norm)
    gated_residual_rmsnorm(*api_inputs, backend=backend, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gated_residual_rmsnorm(*api_inputs, backend=backend, **kwargs)
    for _ in range(2):
        inputs[0].normal_()
        inputs[1].normal_()
        expected = _reference(inputs, weight_before_norm)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
