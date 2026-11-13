"""Zero-token normalization is a validated no-op on both backends (#9440)."""

import pytest
import torch

import flashinfer.norm as norm


@pytest.mark.parametrize("cuda_backend", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "operation",
    [
        "rmsnorm",
        "gemma_rmsnorm",
        "fused_add_rmsnorm",
        "gemma_fused_add_rmsnorm",
        "layernorm",
        "rmsnorm_quant",
        "fused_add_rmsnorm_quant",
        "layernorm_quant",
        "rmsnorm_zero_batch",
        "rmsnorm_zero_heads",
    ],
)
def test_empty_norm(operation, dtype, cuda_backend, monkeypatch):
    if operation.startswith("layernorm") and dtype != torch.bfloat16:
        pytest.skip("LayerNorm's input contract is BF16")
    monkeypatch.setattr(norm, "_USE_CUDA_NORM", cuda_backend)
    shape = (
        (2, 0, 128)
        if operation.endswith("zero_heads")
        else ((0, 2, 128) if operation.endswith("zero_batch") else (0, 128))
    )
    x = torch.empty(shape, device="cuda", dtype=dtype)
    residual = torch.empty_like(x)
    weight = torch.ones(128, device="cuda", dtype=dtype)
    weight32 = weight.float()
    bias = torch.zeros_like(weight32)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    out = torch.empty_like(
        x, dtype=torch.float8_e4m3fn if "quant" in operation else dtype
    )
    function = getattr(
        norm, "rmsnorm" if operation.startswith("rmsnorm_zero") else operation
    )
    if operation.startswith("layernorm"):
        result = (
            function(out, x, weight32, bias, scale)
            if "quant" in operation
            else function(x, weight32, bias)
        )
    elif operation == "fused_add_rmsnorm_quant":
        result = function(out, x, residual, weight, scale)
    elif operation.endswith("fused_add_rmsnorm"):
        result = function(x, residual, weight)
    elif "quant" in operation:
        result = function(out, x, weight, scale)
    else:
        result = function(x, weight, out=out)
        assert result is out
    if result is not None:
        assert result.shape == x.shape and result.dtype == dtype
    torch.cuda.synchronize()


@pytest.mark.parametrize("invalid", ["weight", "out", "residual", "hidden"])
def test_empty_norm_rejects_invalid_metadata(invalid):
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=x.dtype)
    with pytest.raises(ValueError):
        if invalid == "weight":
            norm.rmsnorm(x, weight[:-1])
        elif invalid == "out":
            norm.rmsnorm(
                x, weight, out=torch.empty((0, 64), device="cuda", dtype=x.dtype)
            )
        elif invalid == "residual":
            norm.fused_add_rmsnorm(
                x, torch.empty((0, 64), device="cuda", dtype=x.dtype), weight
            )
        else:
            norm.rmsnorm(torch.empty((0, 0), device="cuda", dtype=x.dtype), weight[:0])


def test_empty_cuda_norm_preserves_dtype_validation(monkeypatch):
    monkeypatch.setattr(norm, "_USE_CUDA_NORM", True)
    x = torch.empty((0, 128), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="FP16 or BF16"):
        norm.rmsnorm(x, torch.ones(128, device="cuda", dtype=x.dtype))


def test_empty_quant_norm_preserves_scale_validation():
    x = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="scale"):
        norm.rmsnorm_quant(
            out,
            x,
            torch.ones(128, device="cuda", dtype=x.dtype),
            torch.ones(2, device="cuda"),
        )
