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
import torch.nn.functional as F

import flashinfer
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec
from flashinfer.utils import device_support_pdl


def llama_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def llama_rms_norm_quant(x, w, scale, eps=1e-6):
    inv_scale = torch.reciprocal(torch.tensor(scale)).float()
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x * inv_scale
    x = torch.clamp(
        x, torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max
    )
    x = x.to(torch.float8_e4m3fn)
    return x


def gemma_rms_norm(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x


def gemma_fused_add_rms_norm(x, residual, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x + residual
    residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + w.float())
    x = x.to(orig_dtype)
    return x, residual


def fused_add_rms_norm(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(orig_dtype)
    return x, residual


def fused_add_rms_norm_quant(x, residual, weight, scale, eps):
    inv_scale = torch.reciprocal(torch.tensor(scale)).float()
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + residual.to(torch.float32)
    residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * weight.float()
    x = x * inv_scale
    x = torch.clamp(
        x, torch.finfo(torch.float8_e4m3fn).min, torch.finfo(torch.float8_e4m3fn).max
    )
    x = x.to(torch.float8_e4m3fn)
    return x, residual


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_norm(batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous):
    if contiguous:
        x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_norm_quant(
    batch_size, hidden_size, dtype, quant_scale, enable_pdl, contiguous
):
    if contiguous:
        x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = llama_rms_norm_quant(x, w, quant_scale)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.rmsnorm_quant(y, x, w, quant_scale, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref.float(), y.float(), rtol=1, atol=1)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("num_heads", [4, 7, 16])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_qknorm(
    batch_size, num_heads, head_dim, dtype, specify_out, enable_pdl, contiguous
):
    if contiguous:
        x = torch.randn(batch_size, num_heads, head_dim).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, num_heads * 2, head_dim, device="cuda").to(dtype)
        x = x[:, :num_heads, :head_dim]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    w = torch.randn(head_dim).to(0).to(dtype)

    y_ref = llama_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_fused_add_rmsnorm(batch_size, hidden_size, dtype, enable_pdl, contiguous):
    eps = 1e-6

    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_scale", [0.01, 1.0, 10.0])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_fused_add_rmsnorm_quant(
    batch_size, hidden_size, dtype, quant_scale, enable_pdl, contiguous
):
    eps = 1e-6

    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = fused_add_rms_norm_quant(
        x.clone(), residual.clone(), weight, quant_scale, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.fused_add_rmsnorm_quant(
        y, x_fused, residual_fused, weight, quant_scale, eps, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(y.float(), x_native.float(), rtol=1, atol=1)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("specify_out", [True, False])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_norm(
    batch_size, hidden_size, dtype, specify_out, enable_pdl, contiguous
):
    if contiguous:
        x = torch.randn(batch_size, hidden_size).to(0).to(dtype)
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    w = torch.randn(hidden_size).to(0).to(dtype)

    y_ref = gemma_rms_norm(x, w)
    if specify_out:
        y = torch.empty_like(x)
        flashinfer.norm.gemma_rmsnorm(x, w, out=y, enable_pdl=enable_pdl)
    else:
        y = flashinfer.norm.gemma_rmsnorm(x, w, enable_pdl=enable_pdl)

    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("hidden_size", [111, 500, 1024, 3072, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
def test_gemma_fused_add_rmsnorm(
    batch_size, hidden_size, dtype, enable_pdl, contiguous
):
    eps = 1e-6

    if contiguous:
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    else:
        x = torch.randn(batch_size, hidden_size * 2, device="cuda").to(dtype)
        x = x[:, :hidden_size]

    if enable_pdl and not device_support_pdl(x.device):
        pytest.skip("PDL is only available for Hopper and later GPUs")

    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    x_native, residual_native = gemma_fused_add_rms_norm(
        x.clone(), residual.clone(), weight, eps
    )

    x_fused = x.clone()
    residual_fused = residual.clone()
    flashinfer.gemma_fused_add_rmsnorm(
        x_fused, residual_fused, weight, eps, enable_pdl=enable_pdl
    )

    torch.testing.assert_close(x_fused, x_native, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(residual_fused, residual_native, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 128])
@pytest.mark.parametrize("hidden_size", [128, 129, 1024, 16384])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_layernorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    gamma = torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    beta = torch.randn(hidden_size, dtype=torch.float32, device="cuda")

    out = flashinfer.layernorm(x, gamma, beta, eps)
    out_ref = F.layer_norm(x.float(), (hidden_size,), gamma, beta, eps).to(dtype)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)


def test_norm_compilation_without_fp8():
    """Test that norm module compiles successfully without ENABLE_FP8 flag.

    This test verifies the fix for issue #2271 where batchWarpReduceSum in
    reduceKernelUtils.cuh depends on PackType which is only defined when
    ENABLE_FP8 is set. The fix guards batchWarpReduceSum with #ifdef ENABLE_FP8.
    """
    # Create a JIT spec for norm module without ENABLE_FP8 flag
    nvcc_flags = [
        "-DENABLE_BF16",
        # Note: ENABLE_FP8 is intentionally omitted to test compilation without it
    ]
    spec = gen_jit_spec(
        "norm_without_fp8_test",
        [
            jit_env.FLASHINFER_CSRC_DIR / "norm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_norm_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )

    # This should compile successfully without errors
    # If batchWarpReduceSum is not properly guarded, this will fail with:
    # "error: incomplete type is not allowed" for PackType
    module = spec.build_and_load()

    # Verify the module loaded successfully
    assert module is not None


if __name__ == "__main__":
    # test_norm(1, 1024, torch.float16, False, True, True)
    test_norm(19, 1024, torch.float16, False, True, False)
    # test_fused_add_rmsnorm(1, 16384, torch.float16, True, True)
