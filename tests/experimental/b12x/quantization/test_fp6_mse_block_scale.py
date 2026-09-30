"""Phase 1.2: MSE-optimal per-block UE8M0 exponent (offline weights)."""
from __future__ import annotations

import pytest
import torch

from b12x.quantization.mxfp6.fp6_checkpoint import (
    build_quantization_config,
    dequantize_linear_from_fp6,
    quantize_linear_to_fp6,
)


def _rel_frobenius_rmse(w: torch.Tensor, w_hat: torch.Tensor) -> float:
    diff = (w.float() - w_hat.float()).reshape(-1)
    ref = w.float().reshape(-1)
    denom = float(torch.linalg.vector_norm(ref).item())
    if denom < 1e-12:
        return 0.0
    return float(torch.linalg.vector_norm(diff).item() / denom)


def test_mse_never_worse_than_ceil_rmse() -> None:
    """Joint MSE encode must not increase reconstruction RMSE vs ceil."""
    torch.manual_seed(0)
    # Mix of well-behaved and outlier-heavy blocks (ceil-1 can win on the latter).
    w = torch.randn(64, 256, dtype=torch.bfloat16) * 0.05
    w[0, 0] = 3.0  # single large outlier in one block
    w[1, 32:40] = torch.linspace(0.01, 0.2, 8).to(torch.bfloat16)

    qt_ceil = quantize_linear_to_fp6(
        w, source_format="mxfp6_w6a8", use_gpu=False, block_scale_rule="ceil"
    )
    qt_mse = quantize_linear_to_fp6(
        w, source_format="mxfp6_w6a8", use_gpu=False, block_scale_rule="mse"
    )
    hat_ceil = dequantize_linear_from_fp6(
        qt_ceil.weight, qt_ceil.weight_scale, fmt=qt_ceil.fmt,
        weight_scale_2=qt_ceil.weight_scale_2,
    )
    hat_mse = dequantize_linear_from_fp6(
        qt_mse.weight, qt_mse.weight_scale, fmt=qt_mse.fmt,
        weight_scale_2=qt_mse.weight_scale_2,
    )
    rmse_ceil = _rel_frobenius_rmse(w, hat_ceil)
    rmse_mse = _rel_frobenius_rmse(w, hat_mse)
    assert rmse_mse <= rmse_ceil + 1e-7, (
        f"mse RMSE {rmse_mse:.6e} worse than ceil {rmse_ceil:.6e}"
    )


def test_mse_can_select_finer_exponent() -> None:
    """When amax just crosses a power-of-two boundary, MSE should pick ceil-1."""
    # E2M3 max=7.5: amax slightly above 7.5 forces ceil scale=2; ceil-1 scale=1
    # only clips the tip while doubling resolution for the rest of the block.
    torch.manual_seed(1)
    block = torch.randn(32, dtype=torch.float32) * 0.4
    block[0] = 7.6
    w = block.view(1, 32).to(torch.bfloat16)

    qt_ceil = quantize_linear_to_fp6(
        w, source_format="mxfp6_w6a8", use_gpu=False, block_scale_rule="ceil"
    )
    qt_mse = quantize_linear_to_fp6(
        w, source_format="mxfp6_w6a8", use_gpu=False, block_scale_rule="mse"
    )
    ue_ceil = int(qt_ceil.weight_scale[0, 0].item())
    ue_mse = int(qt_mse.weight_scale[0, 0].item())
    assert ue_mse == ue_ceil - 1, (
        f"expected mse to pick finer exponent; ceil ue={ue_ceil} mse ue={ue_mse}"
    )
    hat_ceil = dequantize_linear_from_fp6(
        qt_ceil.weight, qt_ceil.weight_scale, fmt=qt_ceil.fmt
    )
    hat_mse = dequantize_linear_from_fp6(
        qt_mse.weight, qt_mse.weight_scale, fmt=qt_mse.fmt
    )
    assert _rel_frobenius_rmse(w, hat_mse) < _rel_frobenius_rmse(w, hat_ceil)


def test_build_config_records_block_scale_rule() -> None:
    cfg = build_quantization_config(block_scale_rule="mse")
    assert cfg["block_scale_rule"] == "mse"
    cfg2 = build_quantization_config(block_scale_rule="ceil")
    assert cfg2["block_scale_rule"] == "ceil"


def test_rejects_bad_block_scale_rule() -> None:
    w = torch.zeros(32, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="block_scale_rule"):
        quantize_linear_to_fp6(w, use_gpu=False, block_scale_rule="round")
