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
from flashinfer.utils import get_compute_capability, has_flashinfer_jit_cache


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
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


@pytest.mark.parametrize(
    "dim", [128, 256, 512, 855, 1710, 2048, 3420, 4096, 11008, 16384]
)
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


@pytest.mark.parametrize(
    "dim", [128, 256, 512, 855, 1710, 2048, 3420, 4096, 11008, 16384]
)
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


@pytest.mark.parametrize(
    "dim", [128, 256, 512, 855, 1710, 2048, 3420, 4096, 11008, 16384]
)
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


_ACT_REFS = {
    "silu": torch.nn.functional.silu,
    "gelu": lambda x: torch.nn.functional.gelu(x, approximate="none"),
    "gelu_tanh": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
}


@pytest.mark.parametrize("act", ["silu", "gelu", "gelu_tanh"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [855, 1710, 3420, 4096, 14336, 28672])
def test_act_and_mul_small_batch_split(act, dtype, dim):
    # Small batches split each row across several CTAs; a row's output must not
    # depend on how many rows are launched together. Large batches use one CTA per row.
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    x = torch.randn(8 * num_sms, 2 * dim, device="cuda", dtype=dtype)
    fn = getattr(flashinfer.activation, f"{act}_and_mul")
    y = fn(x)
    y_ref = (_ACT_REFS[act](x[:, :dim].float()) * x[:, dim:].float()).to(dtype)
    tol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(y, y_ref, rtol=tol, atol=tol)
    sizes = [1, 3, num_sms - 1, num_sms, num_sms + 1, 2 * num_sms - 1, 2 * num_sms]
    for num_tokens in sizes:
        torch.testing.assert_close(fn(x[:num_tokens]), y[:num_tokens], rtol=0, atol=0)


def test_act_and_mul_cuda_graph():
    # Two split launches back to back inside a graph, the second reading the first
    # one's output. On sm90+, where enable_pdl defaults to True, a multi-CTA-per-row
    # kernel then waits on another one through PDL.
    x = torch.randn(4, 2 * 14336, device="cuda", dtype=torch.bfloat16)
    y1 = torch.full((4, 14336), float("nan"), device="cuda", dtype=torch.bfloat16)
    y2 = torch.full((4, 7168), float("nan"), device="cuda", dtype=torch.bfloat16)
    y1_ref = flashinfer.activation.silu_and_mul(x)
    y2_ref = flashinfer.activation.gelu_and_mul(y1_ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashinfer.activation.silu_and_mul(x, out=y1)
        flashinfer.activation.gelu_and_mul(y1, out=y2)
    graph.replay()
    torch.testing.assert_close(y1, y1_ref, rtol=0, atol=0)
    torch.testing.assert_close(y2, y2_ref, rtol=0, atol=0)


if __name__ == "__main__":
    test_fused_silu_mul(128, 1, 1, True)
