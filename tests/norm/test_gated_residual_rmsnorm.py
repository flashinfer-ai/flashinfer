# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team

import pytest
import torch

from flashinfer.norm import (
    gated_residual_rmsnorm,
    supports_gated_residual_rmsnorm,
)

HIDDEN_SIZE = 8192
EPS = 1e-5


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


def test_gated_residual_rmsnorm_supports_blackwell_devices():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100 or SM103 is required")
    assert supports_gated_residual_rmsnorm(*_inputs(1))


@pytest.mark.parametrize("weight_before_norm", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 17, 128])
def test_gated_residual_rmsnorm_matches_reference(rows, weight_before_norm):
    inputs = _inputs(rows)
    if not supports_gated_residual_rmsnorm(*inputs):
        pytest.skip(
            "gated residual RMSNorm requires BF16/FP32 tensors on SM100 or SM103"
        )
    expected = _reference(inputs, weight_before_norm)
    api_inputs = list(inputs)
    if not weight_before_norm:
        api_inputs[2] = None
    output = gated_residual_rmsnorm(
        *api_inputs,
        eps=EPS,
        gain_center=1.0 if weight_before_norm else 0.0,
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
            "gated residual RMSNorm requires BF16/FP32 tensors on SM100 or SM103"
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


@pytest.mark.parametrize("weight_before_norm", [False, True])
def test_gated_residual_rmsnorm_cuda_graph(weight_before_norm):
    inputs = list(_inputs(17))
    if not supports_gated_residual_rmsnorm(*inputs):
        pytest.skip(
            "gated residual RMSNorm requires BF16/FP32 tensors on SM100 or SM103"
        )
    kwargs = {
        "eps": EPS,
        "gain_center": 1.0 if weight_before_norm else 0.0,
    }
    api_inputs = (
        inputs if weight_before_norm else [inputs[0], inputs[1], None, *inputs[3:]]
    )
    gated_residual_rmsnorm(*api_inputs, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gated_residual_rmsnorm(*api_inputs, **kwargs)
    for _ in range(2):
        inputs[0].normal_()
        inputs[1].normal_()
        expected = _reference(inputs, weight_before_norm)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
