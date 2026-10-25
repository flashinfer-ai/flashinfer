# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import flashinfer.norm as norm
from flashinfer.norm import (
    rmsnorm_sigmoid_gate,
    supports_rmsnorm_sigmoid_gate,
)

HEAD_DIM = 64
NUM_HEADS = 32
NUM_KV_HEADS = 2
ATOL = 0.015625
RTOL = 0.015625
BACKENDS = ("auto", "cake", "cute")


def _skip_unless_supported(backend):
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM):
        pytest.skip("RMSNorm sigmoid gate requires BF16 on SM100, SM103, or SM107")
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM, backend=backend):
        pytest.skip(f"backend {backend!r} is not available on this device")


def _inputs(tokens: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q_size = NUM_HEADS * HEAD_DIM
    kv_size = NUM_KV_HEADS * HEAD_DIM
    projection_width = 2 * q_size + 2 * kv_size
    gate_offset = q_size + 2 * kv_size
    projection = torch.empty(
        (tokens, projection_width), dtype=torch.bfloat16, device="cuda"
    ).uniform_(-8.0, 8.0, generator=generator)
    gate = projection[:, gate_offset : gate_offset + q_size].view(
        tokens, NUM_HEADS, HEAD_DIM
    )
    x = torch.randn(
        (tokens, NUM_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return x, gate, projection


def _reference(x: torch.Tensor, gate: torch.Tensor, eps: float):
    values = x.float()
    inv_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (values * inv_rms * torch.sigmoid(gate.float())).bfloat16()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(1, NUM_HEADS, HEAD_DIM), (1, HEAD_DIM)])
def test_rmsnorm_sigmoid_gate_dispatch(monkeypatch, backend, shape):
    # Dispatch without a GPU: auto and cake reach the Cake package (auto only where the
    # package routes the head count and the token count), cute keeps the CuTe DSL kernel.
    monkeypatch.setattr(norm, "_HAS_CUTE_DSL_FUSED_NORM", True)
    monkeypatch.setattr(norm, "_HAS_CAKE_FUSED_NORM", True)
    monkeypatch.setattr(norm, "_is_aligned", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    monkeypatch.setattr(
        norm, "_cute_supports_rmsnorm_sigmoid_gate", lambda *_, **__: True
    )
    monkeypatch.setattr(norm, "device_support_pdl", lambda *_: True)

    def supported(device, *, heads=None, tokens=None):
        return heads is None or heads == NUM_HEADS

    monkeypatch.setattr(
        norm, "_cake_rmsnorm_sigmoid_gate_supported", supported, raising=False
    )
    cake_calls, cute_calls = [], []
    monkeypatch.setattr(
        norm,
        "_cake_rmsnorm_sigmoid_gate",
        lambda *args, **kwargs: cake_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        norm,
        "_rmsnorm_sigmoid_gate_cute",
        lambda *args, **kwargs: cute_calls.append((args, kwargs)),
        raising=False,
    )
    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        gate = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        assert supports_rmsnorm_sigmoid_gate(
            torch.bfloat16, HEAD_DIM, device=x.device, backend=backend
        )
        rmsnorm_sigmoid_gate(x, gate, 1e-3, backend=backend)
        # A 2-D call is one head, which the package does not route: auto falls to the
        # CuTe DSL kernel; an explicit cake still reaches the package (which raises there).
        cake_expected = backend == "cake" or (backend == "auto" and len(shape) == 3)
        if cake_expected:
            assert len(cake_calls) == 1 and not cute_calls
            args, keywords = cake_calls[0]
            assert args == (x, gate) and keywords == {"eps": 1e-3}
        else:
            assert not cake_calls and len(cute_calls) == 1
            args, keywords = cute_calls[0]
            assert args[:3] == (x, gate, 1e-3) and keywords == {"enable_pdl": True}
        with pytest.raises(ValueError, match="backend must be one of"):
            rmsnorm_sigmoid_gate(x, gate, backend="fast")
        monkeypatch.setattr(norm, "_HAS_CAKE_FUSED_NORM", False)
        assert not supports_rmsnorm_sigmoid_gate(
            torch.bfloat16, HEAD_DIM, device=x.device, backend="cake"
        )
        with pytest.raises(ValueError, match="backend 'cake' is not available"):
            rmsnorm_sigmoid_gate(x, gate, backend="cake")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tokens", [0, 1, 2, 17, 257])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_rmsnorm_sigmoid_gate_matches_strided_reference(
    tokens: int, eps: float, backend: str
):
    _skip_unless_supported(backend)

    x, gate, projection = _inputs(tokens, seed=919 + tokens)
    assert gate.stride()[1:] == (HEAD_DIM, 1)
    if tokens > 1:
        assert not gate.is_contiguous()

    expected = _reference(x, gate, eps)
    output_ptr = x.data_ptr()
    rmsnorm_sigmoid_gate(x, gate, eps, backend=backend)
    torch.cuda.synchronize()

    assert x.data_ptr() == output_ptr
    torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
    assert projection.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("tokens", [8, 64])
def test_rmsnorm_sigmoid_gate_cuda_graph_uses_changed_inputs(tokens: int, backend: str):
    _skip_unless_supported(backend)

    static_x, static_gate, projection = _inputs(tokens, seed=1001)
    rmsnorm_sigmoid_gate(static_x, static_gate, 1e-5, backend=backend)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsnorm_sigmoid_gate(static_x, static_gate, 1e-5, backend=backend)

    for seed in (2003, 3001, 4001):
        next_x, next_gate, next_projection = _inputs(tokens, seed=seed)
        expected = _reference(next_x, next_gate, 1e-5)
        static_x.copy_(next_x)
        projection.copy_(next_projection)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_x, expected, atol=ATOL, rtol=RTOL)


def test_rmsnorm_sigmoid_gate_support_is_narrow():
    for backend in BACKENDS:
        assert not supports_rmsnorm_sigmoid_gate(
            torch.float16, HEAD_DIM, backend=backend
        )
        assert not supports_rmsnorm_sigmoid_gate(torch.bfloat16, 128, backend=backend)
        assert not supports_rmsnorm_sigmoid_gate(
            torch.bfloat16, HEAD_DIM, device=torch.device("cpu"), backend=backend
        )


def test_rmsnorm_sigmoid_gate_rejects_cpu_tensors():
    x = torch.empty(1, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    gate = torch.empty_like(x)
    with pytest.raises(ValueError, match="contract"):
        rmsnorm_sigmoid_gate(x, gate)


def test_rmsnorm_sigmoid_gate_rejects_misaligned_tensors():
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM):
        pytest.skip("RMSNorm sigmoid gate requires BF16 on SM100, SM103, or SM107")
    numel = NUM_HEADS * HEAD_DIM
    x = torch.empty(numel + 1, dtype=torch.bfloat16, device="cuda")[1:].view(
        1, NUM_HEADS, HEAD_DIM
    )
    gate = torch.empty_like(x)
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="contract"):
        rmsnorm_sigmoid_gate(x, gate)
