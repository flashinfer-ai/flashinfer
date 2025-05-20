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

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        [
            flashinfer.activation.gen_act_and_mul_module("silu"),
            flashinfer.activation.gen_act_and_mul_module("gelu"),
            flashinfer.activation.gen_act_and_mul_module("gelu_tanh"),
        ],
        verbose=False,
    )
    yield


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_fused_silu_mul(dim, batch_size, seq_len, enable_pdl):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(0).to(torch.float16)
    major, _ = get_compute_capability(x.device)
    if major < 9 and enable_pdl:
        pytest.skip("PDL is only available for Hopper and later GPUs")
    y_ref = x[..., dim:] * torch.nn.functional.silu(x[..., :dim])
    y = flashinfer.activation.silu_and_mul(x, enable_pdl=enable_pdl)
    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_fused_gelu_tanh_mul(dim, batch_size, seq_len, enable_pdl):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(0).to(torch.float16)
    major, _ = get_compute_capability(x.device)
    if major < 9 and enable_pdl:
        pytest.skip("PDL is only available for Hopper and later GPUs")
    y_ref = x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="tanh")
    y = flashinfer.activation.gelu_tanh_and_mul(x, enable_pdl=enable_pdl)
    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_fused_gelu_mul(dim, batch_size, seq_len, enable_pdl):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(0).to(torch.float16)
    major, _ = get_compute_capability(x.device)
    if major < 9 and enable_pdl:
        pytest.skip("PDL is only available for Hopper and later GPUs")
    y_ref = x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="none")
    y = flashinfer.activation.gelu_and_mul(x, enable_pdl=enable_pdl)
    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_fused_silu_mul(128, 1, 1, True)
