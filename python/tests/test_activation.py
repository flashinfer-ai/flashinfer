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


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_fused_silu_mul(dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(0).to(torch.float16)
    y_ref = x[..., dim:] * torch.nn.functional.silu(x[..., :dim])
    y = flashinfer.activation.silu_and_mul(x)
    numpy.testing.assert_allclose(
        y_ref.cpu().numpy(), y.cpu().numpy(), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_fused_gelu_tanh_mul(dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(0).to(torch.float16)
    y_ref = x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="tanh")
    y = flashinfer.activation.gelu_tanh_and_mul(x)
    numpy.testing.assert_allclose(
        y_ref.cpu().numpy(), y.cpu().numpy(), rtol=1e-3, atol=1e-3
    )
