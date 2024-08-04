"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy
import pytest
import torch

import flashinfer


def llama_rms_norm(x, w, eps=1e-6):
    def _norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    output = _norm(x.float()).type_as(x)
    return output * w


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return x, residual


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_norm(batch_size, hidden_size, dtype):
    x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    y = flashinfer.norm.rmsnorm(x, w)

    numpy.testing.assert_allclose(
        y_ref.cpu().numpy(), y.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.fused_add_rmsnorm(x_fused, residual_fused, weight, eps)

    torch.testing.assert_close(x_fused, x_native, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-2, atol=1e-2)
