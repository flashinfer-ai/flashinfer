import pytest
import torch

from flashinfer.fused_moe.core import _resolve_fp4_gemm2_per_token_scaling


def test_fp4_gemm2_scaling_mode_preserves_legacy_behavior():
    per_token_scale = torch.ones(2, dtype=torch.float32)

    assert _resolve_fp4_gemm2_per_token_scaling(None, None) is False
    assert _resolve_fp4_gemm2_per_token_scaling(per_token_scale, None) is True


def test_fp4_gemm2_scaling_mode_allows_static_fc2_with_per_token_fc1():
    per_token_scale = torch.ones(2, dtype=torch.float32)

    assert _resolve_fp4_gemm2_per_token_scaling(per_token_scale, False) is False


def test_fp4_gemm2_scaling_mode_rejects_missing_token_scales():
    with pytest.raises(ValueError, match="requires per_token_scale"):
        _resolve_fp4_gemm2_per_token_scaling(None, True)
