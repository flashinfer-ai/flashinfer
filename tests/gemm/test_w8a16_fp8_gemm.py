"""
Copyright (c) 2026 by FlashInfer team.

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
import torch.nn.functional as F

from flashinfer.gemm import w8a16_fp8_gemm


def _quantize(w32: torch.Tensor):
    scale = (w32.abs().amax() / 448.0).to(torch.float32).reshape(1)
    return (w32 / scale).to(torch.float8_e4m3fn), scale


def _cast_copy_reference(act, w_fp8, scale, bias):
    """TRT-LLM W8A16FP8LinearMethod.apply, verbatim semantics."""
    weight = w_fp8.to(act.dtype).mul_(scale.to(act.dtype))
    return F.linear(act, weight, bias)


@pytest.mark.parametrize("m", [1, 7, 128, 1024])
@pytest.mark.parametrize(
    "n,k", [(1024, 5120), (5120, 1024), (640, 25600), (1280, 1536)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
def test_w8a16_fp8_gemm(m, n, k, dtype, bias):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(m, k, device=dev, dtype=torch.float32).to(dtype)
    w32 = torch.randn(n, k, device=dev, dtype=torch.float32) * 0.02
    w_fp8, scale = _quantize(w32)
    b = torch.randn(n, device=dev, dtype=torch.float32).to(dtype) if bias else None

    out = w8a16_fp8_gemm(act, w_fp8, scale, bias=b)
    assert out.shape == (m, n)
    assert out.dtype == dtype

    ref32 = act.float() @ (w_fp8.float() * scale).T
    if b is not None:
        ref32 = ref32 + b.float()
    cast_out = _cast_copy_reference(act, w_fp8, scale, b)

    # Both the kernel and the cast-copy path differ from the fp32 reference
    # only by 16-bit rounding / accumulation order; the kernel must not be
    # worse than the cast-copy path by more than a small margin.
    kernel_err = (out.float() - ref32).abs().max().item()
    cast_err = (cast_out.float() - ref32).abs().max().item()
    floor = 2e-3 * ref32.abs().max().item()
    assert torch.isfinite(out.float()).all()
    assert kernel_err <= max(2.0 * cast_err, floor), (kernel_err, cast_err, floor)


def test_w8a16_fp8_gemm_3d_and_out():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    act = torch.randn(2, 33, 256, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(512, 256, device=dev) * 0.02)
    out = torch.empty(2, 33, 512, device=dev, dtype=torch.bfloat16)
    ret = w8a16_fp8_gemm(act, w_fp8, scale, out=out)
    assert ret is out
    ref = act.float().reshape(-1, 256) @ (w_fp8.float() * scale).T
    torch.testing.assert_close(out.float().reshape(-1, 512), ref, atol=5e-2, rtol=2e-2)


def test_w8a16_fp8_gemm_rejects_bad_inputs():
    dev = torch.device("cuda")
    act = torch.randn(4, 64, device=dev, dtype=torch.bfloat16)
    w_fp8, scale = _quantize(torch.randn(32, 64, device=dev))
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act.float(), w_fp8, scale)
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act, w_fp8.to(torch.bfloat16), scale)
    with pytest.raises(ValueError):
        w8a16_fp8_gemm(act[:, :32], w_fp8, scale)
