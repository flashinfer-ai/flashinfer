"""Numerical references for quantized matrix multiplication."""

import torch

from .helpers import dequantize_grouped_nvfp4


def nvfp4_gemm_reference(lhs, rhs, lhs_scale, rhs_scale, *, k, dtype):
    """Decode local block scales, accumulate in FP32, then apply global scales."""
    a, a_sf = lhs
    b, b_sf = rhs
    one = torch.ones(1, device=a.device, dtype=torch.float32)
    a_ref = dequantize_grouped_nvfp4(a.permute(2, 0, 1), a_sf, k, one)
    b_ref = dequantize_grouped_nvfp4(b.permute(2, 0, 1), b_sf, k, one)
    alpha = 1.0 / (lhs_scale * rhs_scale)
    return ((a_ref @ b_ref.transpose(-1, -2)) * alpha.reshape(-1, 1, 1)).to(dtype)
