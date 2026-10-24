# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""GPU tests of the Cake per-head RMSNorm times the sigmoid gate (``flashinfer.cake_muse_spark_norm_backend``)
behind ``rmsnorm_sigmoid_gate(..., backend="cake")``: every routed program against the reference on both sides of
its window edges for the three routed head counts, the Cake result bit for bit against the CuTe DSL kernel's, the
strided gate view of a wider projection, the 2-D call (one head, which the package does not route), the empty
batch, CUDA graph replay, the fall-back at a gap of the windows and the input checks."""

import pytest
import torch

import flashinfer.cake_muse_spark_norm_backend as pkg
from flashinfer.cake_muse_spark_norm_backend import cake_jit as loader
from flashinfer.norm import (
    rmsnorm_sigmoid_gate,
    supports_rmsnorm_sigmoid_gate,
)

HEAD_DIM = pkg.HEAD_DIM
OP = pkg.RMSNORM_SIGMOID_GATE
HEADS = (128, 32, 16)
KV_HEADS = 2
ATOL = RTOL = 0.015625


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _arch():
    return pkg._arch(_device()) if torch.cuda.is_available() else None


def _supported() -> bool:
    return _arch() is not None and all(
        pkg.is_cake_rmsnorm_sigmoid_gate_supported(_device(), heads=heads)
        for heads in HEADS
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the Cake RMSNorm sigmoid gate needs an SM100 / SM103 device with its route exported",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        names = set(loader.route_modules(_arch(), pkg.rmsnorm_sigmoid_gate_route()))
        for spec in loader.specs_for_arch(_arch()):
            if any(spec.name.startswith(name) for name in names):
                spec.build_and_load()
    yield


def _inputs(tokens: int, heads: int, seed: int):
    """``x`` contiguous ``(tokens, heads, 64)``; ``gate`` the head rows of a ``[q, k, v, gate]`` projection of width
    ``2 * heads * 64 + 2 * KV_HEADS * 64`` (a strided view); both drawn on the CPU."""
    generator = torch.Generator().manual_seed(seed)
    q_size = heads * HEAD_DIM
    kv_size = KV_HEADS * HEAD_DIM
    projection = (
        torch.empty((tokens, 2 * q_size + 2 * kv_size), dtype=torch.bfloat16)
        .uniform_(-8.0, 8.0, generator=generator)
        .cuda()
    )
    gate = projection[:, q_size + 2 * kv_size : 2 * q_size + 2 * kv_size].view(
        tokens, heads, HEAD_DIM
    )
    x = torch.randn(
        (tokens, heads, HEAD_DIM), dtype=torch.bfloat16, generator=generator
    ).cuda()
    return x, gate, projection


def _reference(x: torch.Tensor, gate: torch.Tensor, eps: float):
    values = x.float()
    inv_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    return (values * inv_rms * torch.sigmoid(gate.float())).bfloat16()


def _edge_tokens(heads: int) -> list[int]:
    """1, 2, 17 and both sides of every window edge of the head count on this device."""
    tokens = {1, 2, 17}
    for name in loader.route_modules(_arch(), pkg.rmsnorm_sigmoid_gate_route()):
        for count, _above, at_most in loader.MODULES[name]["select_rule"]["windows"][
            _arch()
        ]:
            if int(count) == heads and at_most is not None:
                tokens.update((int(at_most), int(at_most) + 1))
    return sorted(tokens)


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
@pytest.mark.parametrize("heads", HEADS)
def test_every_window_matches_the_reference(heads, eps):
    route = pkg.rmsnorm_sigmoid_gate_route()
    launched = []
    original = pkg._launch

    def spy(plan, args):
        launched.append((plan.name, plan.rows))
        return original(plan, args)

    try:
        pkg._launch = spy
        for tokens in _edge_tokens(heads):
            x, gate, projection = _inputs(tokens, heads, seed=919 + tokens)
            assert gate.stride()[1:] == (HEAD_DIM, 1)
            if tokens > 1:
                # one token's view takes the compact stride; from two on it is the projection's
                assert gate.stride(0) == projection.stride(0)
            expected = _reference(x, gate, eps)
            pointer = x.data_ptr()
            rmsnorm_sigmoid_gate(x, gate, eps, backend="cake")
            torch.cuda.synchronize()
            assert x.data_ptr() == pointer
            torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
            name = loader.selected_module(_arch(), route, tokens, heads)
            assert launched[-1] == (name, tokens)
    finally:
        pkg._launch = original


@pytest.mark.parametrize("heads", HEADS)
def test_matches_the_cute_kernel_bit_for_bit(heads):
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM, backend="cute"):
        pytest.skip("the CuTe DSL kernel is not available on this device")
    for tokens in (1, 8, 9, 17, 256, 257, 1024, 1025):
        x, gate, _ = _inputs(tokens, heads, seed=5000 + tokens)
        cake, cute = x.clone(), x.clone()
        rmsnorm_sigmoid_gate(cake, gate, 1e-5, backend="cake")
        rmsnorm_sigmoid_gate(cute, gate, 1e-5, backend="cute")
        torch.cuda.synchronize()
        assert torch.equal(cake, cute), (heads, tokens)


def test_contiguous_gate_and_differing_token_strides():
    x, gate, _ = _inputs(300, 32, seed=11)
    dense = gate.contiguous()
    expected = _reference(x, dense, 1e-5)
    strided, contiguous = x.clone(), x.clone()
    rmsnorm_sigmoid_gate(strided, gate, 1e-5, backend="cake")
    rmsnorm_sigmoid_gate(contiguous, dense, 1e-5, backend="cake")
    torch.cuda.synchronize()
    torch.testing.assert_close(contiguous, expected, atol=ATOL, rtol=RTOL)
    assert torch.equal(strided, contiguous)


def test_a_two_d_call_is_one_head_and_falls_to_the_cute_kernel(monkeypatch):
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM, backend="cute"):
        pytest.skip("the CuTe DSL kernel is not available on this device")
    x, gate, _ = _inputs(64, 32, seed=13)
    rows, rows_gate = x.view(-1, HEAD_DIM).clone(), gate.reshape(-1, HEAD_DIM).clone()
    expected = _reference(rows, rows_gate, 1e-5)
    launched = []
    monkeypatch.setattr(pkg, "_launch", lambda *a: launched.append(a))
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(_device(), heads=1)
    rmsnorm_sigmoid_gate(rows, rows_gate, 1e-5, backend="auto")
    torch.cuda.synchronize()
    torch.testing.assert_close(rows, expected, atol=ATOL, rtol=RTOL)
    assert launched == []
    with pytest.raises(ValueError, match="admits 2048 tokens of 1 heads"):
        rmsnorm_sigmoid_gate(rows, rows_gate, 1e-5, backend="cake")


def test_empty_batch_launches_nothing():
    launched = []
    original = pkg._launch
    try:
        pkg._launch = lambda *a: launched.append(a)
        x, gate, _ = _inputs(0, 32, seed=1)
        rmsnorm_sigmoid_gate(x, gate, 1e-5, backend="cake")
    finally:
        pkg._launch = original
    assert launched == []


@pytest.mark.parametrize("heads", HEADS)
def test_cuda_graph_replay_uses_changed_inputs(heads):
    static_x, static_gate, projection = _inputs(64, heads, seed=1001)
    rmsnorm_sigmoid_gate(static_x, static_gate, 1e-5, backend="cake")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsnorm_sigmoid_gate(static_x, static_gate, 1e-5, backend="cake")
    for seed in (2003, 3001):
        next_x, next_gate, next_projection = _inputs(64, heads, seed=seed)
        expected = _reference(next_x, next_gate, 1e-5)
        static_x.copy_(next_x)
        projection.copy_(next_projection)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_x, expected, atol=ATOL, rtol=RTOL)


def test_a_gap_of_the_windows_falls_to_the_cute_kernel(monkeypatch):
    """A token count no record admits for the head count: ``auto`` runs the CuTe DSL kernel and launches no Cake
    program; an explicit ``"cake"`` raises."""
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM, backend="cute"):
        pytest.skip("the CuTe DSL kernel is not available on this device")
    route = pkg.rmsnorm_sigmoid_gate_route()
    gapped = {
        "arches": [_arch()],
        "route": route,
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {_arch(): [[32, 0, 8]]},
        },
    }
    monkeypatch.setitem(loader.MODULES, "gapped_record", gapped)
    monkeypatch.setitem(loader.ROUTES[_arch()], route, ["gapped_record"])
    pkg.clear_launch_plans()  # plans prepared under the real registry must not serve the gapped one
    launched = []
    monkeypatch.setattr(pkg, "_launch", lambda *a: launched.append(a))
    x, gate, _ = _inputs(9, 32, seed=9)
    expected = _reference(x, gate, 1e-5)
    assert supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM, backend="cake")
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(_device(), heads=32, tokens=9)
    rmsnorm_sigmoid_gate(x, gate, 1e-5, backend="auto")
    torch.cuda.synchronize()
    torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
    assert launched == []
    with pytest.raises(ValueError, match="admits 9 tokens of 32 heads"):
        rmsnorm_sigmoid_gate(x, gate, 1e-5, backend="cake")


def test_refuses_inputs_outside_the_contract():
    x, gate, _ = _inputs(2, 32, seed=2)
    call = pkg.cake_rmsnorm_sigmoid_gate
    call(x.clone(), gate, eps=1e-5)
    with pytest.raises(ValueError, match="must be a bfloat16"):
        call(x.half(), gate.half(), eps=1e-5)
    with pytest.raises(ValueError, match="gate must have the shape"):
        call(x, gate[:1], eps=1e-5)
    wide = torch.empty(2, 32, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=r"must be \[tokens, 64\]"):
        call(wide, wide, eps=1e-5)
    storage = torch.empty(2 * 32 * HEAD_DIM + 8, dtype=torch.bfloat16, device="cuda")
    misaligned = storage[1 : 1 + 2 * 32 * HEAD_DIM].view(2, 32, HEAD_DIM)
    with pytest.raises(ValueError, match="aligned storage"):
        call(misaligned, gate, eps=1e-5)
    odd = torch.empty(2, 68, dtype=torch.bfloat16, device="cuda")[:, :HEAD_DIM]
    with pytest.raises(ValueError, match="multiple of 8 elements"):
        call(odd, odd, eps=1e-5)
    with pytest.raises(ValueError, match="supports compute capability"):
        call(x.cpu(), gate.cpu(), eps=1e-5)
    assert supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM, backend="cake")
    assert not supports_rmsnorm_sigmoid_gate(
        torch.bfloat16, HEAD_DIM, device=torch.device("cpu"), backend="cake"
    )
