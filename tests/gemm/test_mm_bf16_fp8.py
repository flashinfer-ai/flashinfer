# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


_CORRECTNESS_CASES = [
    # m, n, k, output dtype
    (1, 64, 256, torch.bfloat16),
    (7, 192, 320, torch.bfloat16),
    (48, 80, 256, torch.float16),
    (128, 80, 256, torch.bfloat16),
    (1, 10304, 2688, torch.bfloat16),
]

_BMM_FP8_COMPATIBILITY_CASES = [
    # m, n, k
    (1, 64, 256),
    (2, 80, 256),
    (4, 128, 256),
    (8, 192, 320),
    (16, 256, 512),
    (32, 80, 256),
    (48, 80, 256),
    (64, 256, 512),
    (128, 80, 256),
    (1, 1024, 1024),
    (8, 1024, 1024),
    (1, 10304, 2688),
]


def _require_supported_device() -> int:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    major, minor = get_compute_capability(torch.device("cuda"))
    arch = major * 10 + minor
    if not flashinfer.mm_bf16_fp8.is_compute_capability_supported(arch):
        pytest.skip(f"mm_bf16_fp8 is not supported on SM{arch}")
    return arch


def test_mm_bf16_fp8_api_contract() -> None:
    assert "backend" not in inspect.signature(flashinfer.mm_bf16_fp8).parameters


@pytest.mark.parametrize("m,n,k,out_dtype", _CORRECTNESS_CASES)
def test_mm_bf16_fp8_correctness(
    m: int, n: int, k: int, out_dtype: torch.dtype
) -> None:
    _require_supported_device()
    torch.manual_seed(1)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    weight_nt, weight_scale = to_float8(weight_bf16)
    weight = weight_nt.T
    weight_scale = weight_scale.reshape(1)

    assert weight.unsqueeze(0).data_ptr() == weight_nt.data_ptr()
    actual = flashinfer.mm_bf16_fp8(a, weight, weight_scale, dtype=out_dtype)
    quantized_reference = (
        torch.mm(a.float(), weight.to(torch.bfloat16).float()) * weight_scale
    ).to(out_dtype)
    torch.testing.assert_close(actual, quantized_reference, rtol=2e-2, atol=2e-2)

    original_reference = torch.mm(a, weight_bf16.T)
    cosine = F.cosine_similarity(
        actual.float().reshape(-1), original_reference.float().reshape(-1), dim=0
    )
    assert cosine > 0.99


@pytest.mark.parametrize("m,n,k", _BMM_FP8_COMPATIBILITY_CASES)
def test_mm_bf16_fp8_reuses_cudnn_bmm_fp8_weight(m: int, n: int, k: int) -> None:
    arch = _require_supported_device()
    if not flashinfer.bmm_fp8.is_backend_supported("cudnn", arch):
        pytest.skip(f"cudnn bmm_fp8 is not supported on SM{arch}")

    torch.manual_seed(2)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    a_fp8, a_scale = to_float8(a)
    weight_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    weight_nt, weight_scale = to_float8(weight_bf16)
    weight = weight_nt.T
    batched_weight = weight.unsqueeze(0)

    w8a16 = flashinfer.mm_bf16_fp8(a, weight, weight_scale.reshape(1))
    w8a8 = flashinfer.bmm_fp8(
        a_fp8.unsqueeze(0),
        batched_weight,
        a_scale.reshape(1),
        weight_scale.reshape(1),
        torch.bfloat16,
        backend="cudnn",
    ).squeeze(0)

    assert weight_nt.data_ptr() == weight.data_ptr() == batched_weight.data_ptr()
    w8a16_reference = (
        torch.mm(a.float(), weight.to(torch.bfloat16).float()) * weight_scale
    ).to(torch.bfloat16)
    w8a8_reference = (
        torch.mm(a_fp8.float(), weight_nt.float().T) * a_scale * weight_scale
    ).to(torch.bfloat16)
    torch.testing.assert_close(w8a16, w8a16_reference, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(w8a8, w8a8_reference, rtol=5e-2, atol=5e-2)

    cosine = F.cosine_similarity(
        w8a16.float().reshape(-1), w8a8.float().reshape(-1), dim=0
    )
    assert cosine > 0.99


def test_mm_bf16_fp8_rejects_row_major_weight() -> None:
    _require_supported_device()
    a = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(64, 32, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="column-major"):
        flashinfer.mm_bf16_fp8(a, weight, scale)


def test_mm_bf16_fp8_cuda_graph() -> None:
    _require_supported_device()
    torch.manual_seed(2)
    m, n, k = 2, 128, 256
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight_nt = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    weight = weight_nt.T
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    fn = lambda: flashinfer.mm_bf16_fp8(  # noqa: E731
        a, weight, scale, out=out
    )

    eager = fn().clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=2e-2, atol=2e-2)
