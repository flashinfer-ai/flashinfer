# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""Dtype-contract tests for the public norm entry points.

The norm kernels dispatch their compute type from ``input.dtype()`` and
reinterpret every other operand at that type. A mismatched operand therefore
has to be rejected instead of silently producing wrong numbers. The host guard
rejects on CPU tensors, so proving it needs no GPU; the native launcher checks
are exercised through the CUDA-JIT module.
"""

import pytest
import torch

import flashinfer.norm as norm


def test_rmsnorm_rejects_weight_dtype_mismatch():
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"weight dtype .* must match input dtype"):
        norm.rmsnorm(x, weight)


def test_rmsnorm_rejects_out_dtype_mismatch():
    x = torch.randn(2, 16, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.float32)
    out = torch.empty(2, 16, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"out dtype .* must match input dtype"):
        norm.rmsnorm(x, weight, out=out)


def test_fused_add_rmsnorm_rejects_residual_dtype_mismatch():
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    residual = torch.randn(2, 16, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"residual dtype .* must match input dtype"):
        norm.fused_add_rmsnorm(x, residual, weight)


def test_fused_add_rmsnorm_rejects_weight_dtype_mismatch():
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    residual = torch.randn(2, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"weight dtype .* must match input dtype"):
        norm.fused_add_rmsnorm(x, residual, weight)


def _fp8_out(x):
    return torch.empty_like(x, dtype=torch.float8_e4m3fn)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda x, w: norm.rmsnorm_quant(_fp8_out(x), x, w, torch.ones(1)),
            id="rmsnorm_quant",
        ),
        pytest.param(
            lambda x, w: norm.fused_add_rmsnorm_quant(
                _fp8_out(x), x, x, w, torch.ones(1)
            ),
            id="fused_add_rmsnorm_quant",
        ),
        pytest.param(lambda x, w: norm.gemma_rmsnorm(x, w), id="gemma_rmsnorm"),
        pytest.param(
            lambda x, w: norm.gemma_fused_add_rmsnorm(x, x, w),
            id="gemma_fused_add_rmsnorm",
        ),
    ],
)
def test_other_entry_points_reject_weight_dtype_mismatch(call):
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    weight = torch.randn(16, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"weight dtype .* must match input dtype"):
        call(x, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize(
    "case",
    [
        "rmsnorm_weight",
        "rmsnorm_out",
        "rmsnorm_quant_weight",
        "rmsnorm_quant_scale",
        "fused_add_rmsnorm_residual",
        "fused_add_rmsnorm_weight",
        "fused_add_rmsnorm_quant_residual",
        "fused_add_rmsnorm_quant_weight",
        "fused_add_rmsnorm_quant_scale",
        "gemma_rmsnorm_weight",
        "gemma_rmsnorm_out",
        "gemma_fused_add_rmsnorm_residual",
        "gemma_fused_add_rmsnorm_weight",
        "layernorm_out",
    ],
)
def test_native_launchers_reject_dtype_mismatch(case):
    module = norm.get_norm_module()
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    out = torch.empty_like(x)
    out_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.ones(1, device="cuda")
    gamma = torch.ones(256, dtype=torch.float32, device="cuda")
    beta = torch.zeros(256, dtype=torch.float32, device="cuda")

    cases = {
        "rmsnorm_weight": (
            lambda: module.rmsnorm(out, x, weight.float(), 1e-6, False),
            "weight dtype must match input dtype",
        ),
        "rmsnorm_out": (
            lambda: module.rmsnorm(out.float(), x, weight, 1e-6, False),
            "output dtype must match input dtype",
        ),
        "rmsnorm_quant_weight": (
            lambda: module.rmsnorm_quant(
                out_fp8, x, weight.float(), scale, 1e-6, False
            ),
            "weight dtype must match input dtype",
        ),
        "rmsnorm_quant_scale": (
            lambda: module.rmsnorm_quant(out_fp8, x, weight, scale.half(), 1e-6, False),
            "Inconsistency of Tensor type: scale",
        ),
        "fused_add_rmsnorm_residual": (
            lambda: module.fused_add_rmsnorm(x, residual.float(), weight, 1e-6, False),
            "residual dtype must match input dtype",
        ),
        "fused_add_rmsnorm_weight": (
            lambda: module.fused_add_rmsnorm(x, residual, weight.float(), 1e-6, False),
            "weight dtype must match input dtype",
        ),
        "fused_add_rmsnorm_quant_residual": (
            lambda: module.fused_add_rmsnorm_quant(
                out_fp8, x, residual.float(), weight, scale, 1e-6, False
            ),
            "residual dtype must match input dtype",
        ),
        "fused_add_rmsnorm_quant_weight": (
            lambda: module.fused_add_rmsnorm_quant(
                out_fp8, x, residual, weight.float(), scale, 1e-6, False
            ),
            "weight dtype must match input dtype",
        ),
        "fused_add_rmsnorm_quant_scale": (
            lambda: module.fused_add_rmsnorm_quant(
                out_fp8, x, residual, weight, scale.half(), 1e-6, False
            ),
            "Inconsistency of Tensor type: scale",
        ),
        "gemma_rmsnorm_weight": (
            lambda: module.gemma_rmsnorm(out, x, weight.float(), 1e-6, False),
            "weight dtype must match input dtype",
        ),
        "gemma_rmsnorm_out": (
            lambda: module.gemma_rmsnorm(out.float(), x, weight, 1e-6, False),
            "output dtype must match input dtype",
        ),
        "gemma_fused_add_rmsnorm_residual": (
            lambda: module.gemma_fused_add_rmsnorm(
                x, residual.float(), weight, 1e-6, False
            ),
            "residual dtype must match input dtype",
        ),
        "gemma_fused_add_rmsnorm_weight": (
            lambda: module.gemma_fused_add_rmsnorm(
                x, residual, weight.float(), 1e-6, False
            ),
            "weight dtype must match input dtype",
        ),
        "layernorm_out": (
            lambda: module.layernorm(out.float(), x, gamma, beta, 1e-6),
            "output dtype must match input dtype",
        ),
    }
    launch, match = cases[case]

    with pytest.raises(RuntimeError, match=match):
        launch()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_accepts_matching_dtype(dtype):
    x = torch.randn(4, 256, device="cuda", dtype=dtype)
    weight = torch.randn(256, device="cuda", dtype=dtype)

    out = norm.rmsnorm(x, weight)

    assert out.shape == x.shape
    assert out.dtype == dtype
