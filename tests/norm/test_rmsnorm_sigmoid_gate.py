# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team

import pytest
import torch

from flashinfer.norm import (
    rmsnorm_sigmoid_gate,
    supports_rmsnorm_sigmoid_gate,
)

HEAD_DIM = 64
NUM_HEADS = 32
NUM_KV_HEADS = 2
ATOL = 0.015625
RTOL = 0.015625


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


@pytest.mark.parametrize("tokens", [0, 1, 2, 17, 257])
@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_rmsnorm_sigmoid_gate_matches_strided_reference(tokens: int, eps: float):
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM):
        pytest.skip("RMSNorm sigmoid gate requires BF16 on SM100, SM103, or SM107")

    x, gate, projection = _inputs(tokens, seed=919 + tokens)
    assert gate.stride()[1:] == (HEAD_DIM, 1)
    if tokens > 1:
        assert not gate.is_contiguous()

    expected = _reference(x, gate, eps)
    output_ptr = x.data_ptr()
    rmsnorm_sigmoid_gate(x, gate, eps)
    torch.cuda.synchronize()

    assert x.data_ptr() == output_ptr
    torch.testing.assert_close(x, expected, atol=ATOL, rtol=RTOL)
    assert projection.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()


@pytest.mark.parametrize("tokens", [8, 64])
def test_rmsnorm_sigmoid_gate_cuda_graph_uses_changed_inputs(tokens: int):
    if not supports_rmsnorm_sigmoid_gate(torch.bfloat16, HEAD_DIM):
        pytest.skip("RMSNorm sigmoid gate requires BF16 on SM100, SM103, or SM107")

    static_x, static_gate, projection = _inputs(tokens, seed=1001)
    rmsnorm_sigmoid_gate(static_x, static_gate, 1e-5)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsnorm_sigmoid_gate(static_x, static_gate, 1e-5)

    for seed in (2003, 3001, 4001):
        next_x, next_gate, next_projection = _inputs(tokens, seed=seed)
        expected = _reference(next_x, next_gate, 1e-5)
        static_x.copy_(next_x)
        projection.copy_(next_projection)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_x, expected, atol=ATOL, rtol=RTOL)


def test_rmsnorm_sigmoid_gate_support_is_narrow():
    assert not supports_rmsnorm_sigmoid_gate(torch.float16, HEAD_DIM)
    assert not supports_rmsnorm_sigmoid_gate(torch.bfloat16, 128)
    assert not supports_rmsnorm_sigmoid_gate(
        torch.bfloat16, HEAD_DIM, device=torch.device("cpu")
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
