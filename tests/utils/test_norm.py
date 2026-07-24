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
    flashinfer.norm.rmsnorm_quant(
        y, x, w, torch.tensor(quant_scale, device="cuda"), enable_pdl=enable_pdl
    )

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
        y,
        x_fused,
        residual_fused,
        weight,
        torch.tensor(quant_scale, device="cuda"),
        eps,
        enable_pdl=enable_pdl,
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
@pytest.mark.parametrize("hidden_size", [128, 129, 256, 1024, 3584, 4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_layernorm(batch_size, hidden_size, dtype):
    eps = 1e-6

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    gamma = torch.randn(hidden_size, dtype=torch.float32, device="cuda")
    beta = torch.randn(hidden_size, dtype=torch.float32, device="cuda")

    out = flashinfer.layernorm(x, gamma, beta, eps)
    out_ref = F.layer_norm(x.float(), (hidden_size,), gamma, beta, eps).to(dtype)

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)


# =============================================================================
# Regression tests for int32 stride overflow
# =============================================================================
# These tests verify that rmsnorm kernels accept tensors with strides exceeding
# INT32_MAX.  The 2D tests use M=2 so that is_contiguous() returns False and the
# non-contiguous kernel path (which uses sym_int64 strides) is exercised.  This
# requires a ~4 GB flat buffer so the large stride is actually traversable.
# The 3D qknorm test can use batch=1 because qk_rmsnorm_cute always uses
# symbolic strides regardless of contiguity.

_INT64_STRIDE = 2**31  # just above INT32_MAX = 2**31 - 1
_STRIDE_BUF_BYTES = (_INT64_STRIDE + 128) * 2  # bf16, H=128


def _skip_if_low_vram():
    free, _ = torch.cuda.mem_get_info()
    if free < _STRIDE_BUF_BYTES * 1.2:
        pytest.skip(
            f"Requires ~{_STRIDE_BUF_BYTES / 1024**3:.1f}GB free VRAM, "
            f"only {free / 1024**3:.1f}GB available"
        )


def test_rmsnorm_int64_stride():
    """2D rmsnorm with row stride > INT32_MAX (issue #3005)."""
    _skip_if_low_vram()
    H = 128
    dtype = torch.bfloat16
    buf = torch.randn(_INT64_STRIDE + H, dtype=dtype, device="cuda")
    w = torch.randn(H, dtype=dtype, device="cuda")

    x = torch.as_strided(buf, (2, H), (_INT64_STRIDE, 1))
    assert not x.is_contiguous()
    y = flashinfer.norm.rmsnorm(x, w)

    y_ref = llama_rms_norm(x.contiguous(), w)
    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)


def test_qknorm_int64_stride():
    """3D qk_rmsnorm with batch stride > INT32_MAX (issue #3005).

    qk_rmsnorm_cute always uses symbolic strides (no contiguity check),
    so batch=1 suffices — the large stride is validated by TVM-FFI even
    though it is never traversed.
    """
    num_heads, head_dim = 4, 128
    dtype = torch.bfloat16
    buf = torch.randn(1, num_heads, head_dim, dtype=dtype, device="cuda")
    w = torch.randn(head_dim, dtype=dtype, device="cuda")

    x = torch.as_strided(buf, (1, num_heads, head_dim), (_INT64_STRIDE, head_dim, 1))
    y = flashinfer.norm.rmsnorm(x, w)

    y_ref = llama_rms_norm(buf, w)
    torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)


def test_rmsnorm_quant_int64_stride():
    """rmsnorm_quant with row stride > INT32_MAX (issue #3005)."""
    _skip_if_low_vram()
    H = 128
    dtype = torch.bfloat16
    quant_scale = 1.0
    buf = torch.randn(_INT64_STRIDE + H, dtype=dtype, device="cuda")
    w = torch.randn(H, dtype=dtype, device="cuda")

    x = torch.as_strided(buf, (2, H), (_INT64_STRIDE, 1))
    assert not x.is_contiguous()
    y = torch.empty(2, H, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.rmsnorm_quant(y, x, w, torch.tensor(quant_scale, device="cuda"))

    y_ref = llama_rms_norm_quant(x.contiguous(), w, quant_scale)
    torch.testing.assert_close(y.float(), y_ref.float(), rtol=1, atol=1)


def test_fused_add_rmsnorm_int64_stride():
    """fused_add_rmsnorm with row stride > INT32_MAX (issue #3005)."""
    _skip_if_low_vram()
    H = 128
    dtype = torch.bfloat16
    eps = 1e-6
    buf_x = torch.randn(_INT64_STRIDE + H, dtype=dtype, device="cuda")
    w = torch.randn(H, dtype=dtype, device="cuda")
    # Contiguous residual — only one non-contiguous tensor is needed to
    # trigger the non-contiguous kernel path.
    r = torch.randn(2, H, dtype=dtype, device="cuda")

    x = torch.as_strided(buf_x, (2, H), (_INT64_STRIDE, 1))
    assert not x.is_contiguous()
    x_ref, r_ref = fused_add_rms_norm(x.contiguous().clone(), r.clone(), w, eps)

    flashinfer.fused_add_rmsnorm(x, r, w, eps)

    torch.testing.assert_close(x.contiguous(), x_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(r, r_ref, rtol=1e-3, atol=1e-3)


def test_fused_add_rmsnorm_quant_int64_stride():
    """fused_add_rmsnorm_quant with row stride > INT32_MAX (issue #3005)."""
    _skip_if_low_vram()
    H = 128
    dtype = torch.bfloat16
    eps = 1e-6
    quant_scale = 1.0
    buf_x = torch.randn(_INT64_STRIDE + H, dtype=dtype, device="cuda")
    w = torch.randn(H, dtype=dtype, device="cuda")
    r = torch.randn(2, H, dtype=dtype, device="cuda")

    x = torch.as_strided(buf_x, (2, H), (_INT64_STRIDE, 1))
    assert not x.is_contiguous()
    x_ref, r_ref = fused_add_rms_norm_quant(
        x.contiguous().clone(), r.clone(), w, quant_scale, eps
    )

    y = torch.empty(2, H, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.fused_add_rmsnorm_quant(
        y, x, r, w, torch.tensor(quant_scale, device="cuda"), eps
    )

    torch.testing.assert_close(y.float(), x_ref.float(), rtol=1, atol=1)
    torch.testing.assert_close(r, r_ref, rtol=1e-3, atol=1e-3)


# =============================================================================
# Tests: contiguous tensor with M*H > INT32_MAX
# =============================================================================
# Exercise the contiguous (compact) path of the cute-DSL norm kernels with a
# tensor whose flat element count exceeds 2**31. The compact layout bakes the
# row stride in as a constexpr int, so the offset arithmetic row * H is
# computed in int32 and overflows when M * H > 2**31, producing
# cudaErrorIllegalAddress on the first synchronize.

# Shape (175000, 12288) fp16: 174999 * 12288 > 2**31, while a single row's
# offset fits comfortably in int32.
_OVERFLOW_M = 175000
_OVERFLOW_H = 12288
_OVERFLOW_DTYPE = torch.float16
assert (_OVERFLOW_M - 1) * _OVERFLOW_H > 2**31


def _skip_if_low_vram_for_overflow(extra_bytes_per_elem: float = 4.0):
    """Skip when free VRAM cannot hold ~extra_bytes_per_elem * M * H bytes."""
    need = int(extra_bytes_per_elem * _OVERFLOW_M * _OVERFLOW_H)
    free, _ = torch.cuda.mem_get_info()
    if free < need * 1.15:
        pytest.skip(
            f"Requires ~{need / 1024**3:.1f}GB free VRAM, "
            f"only {free / 1024**3:.1f}GB available"
        )


def _spot_check_rows(y_actual: torch.Tensor, x: torch.Tensor, ref_fn, rtol, atol):
    """Compare the kernel output against a per-row reference on a handful of
    rows that bracket the overflow boundary, plus the first and last rows.
    Avoids materializing a full fp32 reference."""
    boundary_row = (2**31) // _OVERFLOW_H  # first row whose flat offset > INT32_MAX
    rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    for r in rows:
        if r < 0 or r >= _OVERFLOW_M:
            continue
        row_ref = ref_fn(r)
        torch.testing.assert_close(
            y_actual[r : r + 1].float(), row_ref.float(), rtol=rtol, atol=atol
        )


def test_rmsnorm_contiguous_overflow():
    """rmsnorm on a contiguous tensor with M*H > INT32_MAX."""
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)  # input + output fp16
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous()

    y = flashinfer.norm.rmsnorm(x, w)
    torch.cuda.synchronize()  # surface any async illegal-address error

    _spot_check_rows(
        y, x, lambda r: llama_rms_norm(x[r : r + 1], w), rtol=1e-3, atol=1e-3
    )


def test_gemma_rmsnorm_contiguous_overflow():
    """gemma_rmsnorm on a contiguous tensor with M*H > INT32_MAX."""
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous()

    y = flashinfer.norm.gemma_rmsnorm(x, w)
    torch.cuda.synchronize()

    _spot_check_rows(
        y, x, lambda r: gemma_rms_norm(x[r : r + 1], w), rtol=1e-3, atol=1e-3
    )


def test_rmsnorm_quant_contiguous_overflow():
    """rmsnorm_quant on a contiguous tensor with M*H > INT32_MAX."""
    # input fp16 (2 B) + fp8 output (1 B).
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=3.0)
    quant_scale = 1.0
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous()

    y = torch.empty(_OVERFLOW_M, _OVERFLOW_H, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.rmsnorm_quant(y, x, w, torch.tensor(quant_scale, device="cuda"))
    torch.cuda.synchronize()

    _spot_check_rows(
        y,
        x,
        lambda r: llama_rms_norm_quant(x[r : r + 1], w, quant_scale),
        rtol=1,
        atol=1,
    )


def test_fused_add_rmsnorm_contiguous_overflow():
    """fused_add_rmsnorm on a contiguous tensor with M*H > INT32_MAX."""
    # input + residual, both fp16; output is written in-place over input.
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    eps = 1e-6
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    r = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous() and r.is_contiguous()

    # Snapshot the spot-check rows before the in-place op mutates them.
    boundary_row = (2**31) // _OVERFLOW_H
    check_rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    x_snap = {i: x[i : i + 1].clone() for i in check_rows}
    r_snap = {i: r[i : i + 1].clone() for i in check_rows}

    flashinfer.fused_add_rmsnorm(x, r, w, eps)
    torch.cuda.synchronize()

    for i in check_rows:
        x_ref, r_ref = fused_add_rms_norm(x_snap[i], r_snap[i], w, eps)
        torch.testing.assert_close(x[i : i + 1], x_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(r[i : i + 1], r_ref, rtol=1e-3, atol=1e-3)


def test_gemma_fused_add_rmsnorm_contiguous_overflow():
    """gemma_fused_add_rmsnorm on a contiguous tensor with M*H > INT32_MAX."""
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    eps = 1e-6
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    r = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous() and r.is_contiguous()

    boundary_row = (2**31) // _OVERFLOW_H
    check_rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    x_snap = {i: x[i : i + 1].clone() for i in check_rows}
    r_snap = {i: r[i : i + 1].clone() for i in check_rows}

    flashinfer.norm.gemma_fused_add_rmsnorm(x, r, w, eps)
    torch.cuda.synchronize()

    for i in check_rows:
        x_ref, r_ref = gemma_fused_add_rms_norm(x_snap[i], r_snap[i], w, eps)
        torch.testing.assert_close(x[i : i + 1], x_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(r[i : i + 1], r_ref, rtol=1e-3, atol=1e-3)


def test_fused_add_rmsnorm_quant_contiguous_overflow():
    """fused_add_rmsnorm_quant on a contiguous tensor with M*H > INT32_MAX."""
    # input fp16 + residual fp16 + fp8 output.
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=5.0)
    eps = 1e-6
    quant_scale = 1.0
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    r = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    w = torch.randn(_OVERFLOW_H, dtype=_OVERFLOW_DTYPE, device="cuda")
    assert x.is_contiguous() and r.is_contiguous()

    boundary_row = (2**31) // _OVERFLOW_H
    check_rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    x_snap = {i: x[i : i + 1].clone() for i in check_rows}
    r_snap = {i: r[i : i + 1].clone() for i in check_rows}

    y = torch.empty(_OVERFLOW_M, _OVERFLOW_H, dtype=torch.float8_e4m3fn, device="cuda")
    flashinfer.norm.fused_add_rmsnorm_quant(
        y, x, r, w, torch.tensor(quant_scale, device="cuda"), eps
    )
    torch.cuda.synchronize()

    for i in check_rows:
        y_ref, r_ref = fused_add_rms_norm_quant(
            x_snap[i], r_snap[i], w, quant_scale, eps
        )
        torch.testing.assert_close(y[i : i + 1].float(), y_ref.float(), rtol=1, atol=1)
        torch.testing.assert_close(r[i : i + 1], r_ref, rtol=1e-3, atol=1e-3)


def test_layernorm_contiguous_overflow():
    """layernorm on a contiguous tensor with M*H > INT32_MAX.

    layernorm_cute uses 32-bit symbolic M and row strides, so it overflows on
    the contiguous path the same way rmsnorm does.
    """
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    eps = 1e-6
    dtype = torch.bfloat16  # layernorm requires bf16 input
    x = torch.randn(_OVERFLOW_M, _OVERFLOW_H, dtype=dtype, device="cuda")
    gamma = torch.randn(_OVERFLOW_H, dtype=torch.float32, device="cuda")
    beta = torch.randn(_OVERFLOW_H, dtype=torch.float32, device="cuda")
    assert x.is_contiguous()

    y = flashinfer.layernorm(x, gamma, beta, eps)
    torch.cuda.synchronize()

    boundary_row = (2**31) // _OVERFLOW_H
    rows = sorted(
        {0, 1, boundary_row - 1, boundary_row, boundary_row + 1, _OVERFLOW_M - 1}
    )
    for r in rows:
        ref = F.layer_norm(x[r : r + 1].float(), (_OVERFLOW_H,), gamma, beta, eps).to(
            dtype
        )
        torch.testing.assert_close(y[r : r + 1], ref, rtol=1e-2, atol=1e-2)


def test_qk_rmsnorm_contiguous_overflow():
    """3D qk_rmsnorm with B*N*head_dim > INT32_MAX.

    The 3D path computes the flattened row count M = B*N as int32 inside the
    kernel, so any tensor with B*N*head_dim > 2**31 overflows the row offset.
    """
    _skip_if_low_vram_for_overflow(extra_bytes_per_elem=4.0)
    B, N, head_dim = 16800, 1024, 128  # B*N*head_dim = 2_202_009_600 > 2**31
    assert B * N * head_dim > 2**31
    dtype = torch.float16
    x = torch.randn(B, N, head_dim, dtype=dtype, device="cuda")
    w = torch.randn(head_dim, dtype=dtype, device="cuda")
    assert x.is_contiguous()

    y = flashinfer.norm.rmsnorm(x, w)
    torch.cuda.synchronize()

    # Spot-check rows around the boundary in the flattened (B*N, head_dim) view.
    boundary_flat = (2**31) // head_dim
    flat_rows = sorted(
        {0, 1, boundary_flat - 1, boundary_flat, boundary_flat + 1, B * N - 1}
    )
    x_flat = x.view(B * N, head_dim)
    y_flat = y.view(B * N, head_dim)
    for r in flat_rows:
        ref = llama_rms_norm(x_flat[r : r + 1], w)
        torch.testing.assert_close(y_flat[r : r + 1], ref, rtol=1e-3, atol=1e-3)


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
