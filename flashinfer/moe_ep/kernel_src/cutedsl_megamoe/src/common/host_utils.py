# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0
"""Host utility helpers shared across NVFP4 and MXFP8 runners."""

import argparse  # noqa: F401
import os  # noqa: F401
import sys  # noqa: F401
from typing import List, Optional, Tuple  # noqa: F401
import torch

from common.megamoe_constants import (
    Fp8E5M2Max,  # noqa: F401
    Fp8E4M3FNMax,  # noqa: F401
    Nvfp4E2M1Max,  # noqa: F401
    Nvfp4BlockSize,
    Mxfp8BlockSize,
    SfPaddingBlock,  # noqa: F401
    TmaLeadingDimByteAlign,  # noqa: F401
    Nvfp4E2M1RcpLimit,  # noqa: F401
    Fp8E4M3RcpLimit,
    Fp8E5M2RcpLimit,
)

# ---------------------------------------------------------------------------
# Dtype constants
# ---------------------------------------------------------------------------

_Nvfp4DataDtype: torch.dtype = torch.float4_e2m1fn_x2
_Nvfp4ScaleDtype: torch.dtype = torch.float8_e4m3fn
_Mxfp8DataDtype_e4m3: torch.dtype = torch.float8_e4m3fn
_Mxfp8DataDtype_e5m2: torch.dtype = torch.float8_e5m2
_Mxfp8ScaleDtype: torch.dtype = torch.float8_e8m0fnu

# ---------------------------------------------------------------------------
# kind_* helpers
# ---------------------------------------------------------------------------


def kind_data_dtype(kind: str) -> torch.dtype:
    if kind == "nvfp4":
        return _Nvfp4DataDtype
    if kind == "mxfp8_e4m3":
        return _Mxfp8DataDtype_e4m3
    if kind == "mxfp8_e5m2":
        return _Mxfp8DataDtype_e5m2
    raise ValueError(f"Unknown kind: {kind!r}")


def kind_scale_dtype(kind: str) -> torch.dtype:
    return _Nvfp4ScaleDtype if kind == "nvfp4" else _Mxfp8ScaleDtype


def kind_sf_vec_size(kind: str) -> int:
    return Nvfp4BlockSize if kind == "nvfp4" else Mxfp8BlockSize


# ---------------------------------------------------------------------------
# Mxfp8 quantize function. May move function to mxfp8 folder later
# ---------------------------------------------------------------------------


def mxfp8_quantize_per_block_32(
    c_fp32: torch.Tensor,
    data_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-quantize along the trailing dim with per-32 E8M0 block scales."""
    M, N = c_fp32.shape
    if N % Mxfp8BlockSize != 0:
        raise ValueError(
            f"Trailing dim ({N}) must be a multiple of sf_vec_size ({Mxfp8BlockSize})."
        )
    n_blocks = N // Mxfp8BlockSize
    data_max_rcp_limit = (
        Fp8E4M3RcpLimit if data_dtype == torch.float8_e4m3fn else Fp8E5M2RcpLimit
    )
    blocked = c_fp32.view(M, n_blocks, Mxfp8BlockSize)
    absmax = blocked.abs().amax(dim=-1)
    safe_absmax = torch.clamp(absmax, min=1e-30)
    scale_exp = torch.ceil(torch.log2(safe_absmax * data_max_rcp_limit))
    e_uint8 = torch.clamp(scale_exp + 127.0, min=0.0, max=254.0).to(torch.int32)
    e_uint8 = torch.where(absmax == 0, torch.zeros_like(e_uint8), e_uint8)
    sfc_e8m0 = e_uint8.to(torch.uint8).view(_Mxfp8ScaleDtype)
    scale_fp32 = torch.pow(2.0, e_uint8.float() - 127.0)
    scale_expanded = scale_fp32.unsqueeze(-1).expand_as(blocked).reshape(M, N)
    scaled = c_fp32 / scale_expanded
    c_fp8 = scaled.to(data_dtype)
    return c_fp8, sfc_e8m0


# ---------------------------------------------------------------------------
# referench check helper
# ---------------------------------------------------------------------------


def compare_and_report_mismatches(
    gpu_tensor,
    ref_tensor,
    name="Tensor",
    atol=1e-05,
    rtol=1e-05,
    max_mismatches=8,
    print_first_8=False,
):
    import torch as _torch  # host-only helper, keep out of module-level imports

    """
    Compare two tensors and report the first N mismatched elements.

    Args:
        gpu_tensor: Results computed on GPU
        ref_tensor: Reference results (CPU)
        name: Name of the tensor
        atol: Absolute tolerance
        rtol: Relative tolerance
        max_mismatches: Maximum number of mismatches to report
    """
    # Ensure both are on CPU for comparison
    if gpu_tensor.is_cuda:
        gpu_data = gpu_tensor.cpu()
    else:
        gpu_data = gpu_tensor

    if ref_tensor.is_cuda:
        ref_data = ref_tensor.cpu()
    else:
        ref_data = ref_tensor

    # Ensure shapes match
    assert gpu_data.shape == ref_data.shape, (
        f"Shape mismatch: {gpu_data.shape} vs {ref_data.shape}"
    )

    if print_first_8:
        # Print first 8 elements (regardless of match)
        print(f"\n{name} - First 8 elements:")
        print(
            f"{'Index':<6} {'Coordinate':<30} {'GPU Data':<20} {'CPU Data':<20} {'Abs Error':<20}"
        )
        print("-" * 100)
        print(f"\n")  # noqa: F541

        flat_gpu = gpu_data.flatten()
        flat_ref = ref_data.flatten()
        num_elements = min(8, flat_gpu.numel())

        for i in range(num_elements):
            # Get multi-dimensional coordinate
            idx_tuple = _torch.unravel_index(_torch.tensor(i), gpu_data.shape)
            coord = tuple(idx.item() for idx in idx_tuple)
            gpu_val = flat_gpu[i].item()
            ref_val = flat_ref[i].item()
            abs_error = abs(gpu_val - ref_val)
            print(
                f"{i + 1:<6} {str(coord):<30} {gpu_val:<20.6f} {ref_val:<20.6f} {abs_error:<20.6f}"
            )

    # Compute differences
    diff = _torch.abs(gpu_data.float() - ref_data.float())
    threshold = atol + rtol * _torch.abs(ref_data.float())
    mismatch_mask = (diff > threshold) | _torch.isnan(diff)

    # Find all mismatch indices
    mismatch_indices = _torch.nonzero(mismatch_mask, as_tuple=False)
    num_mismatches = mismatch_indices.shape[0]

    if num_mismatches == 0:
        print(f"✓ {name} passed validation! All elements are within tolerance.")
        return True
    else:
        print(f"✗ {name} failed validation!")
        print(
            f"  Total {num_mismatches} mismatched elements (total elements: {gpu_data.numel()}, mismatch rate: {100.0 * num_mismatches / gpu_data.numel():.4f}%)"
        )
        print(f"  Tolerance settings: atol={atol}, rtol={rtol}")
        print(f"\nFirst {min(max_mismatches, num_mismatches)} mismatched elements:")
        print(
            f"{'Index':<6} {'Coordinate':<30} {'GPU Data':<20} {'CPU Data':<20} {'Abs Error':<20}"
        )
        print("-" * 100)

        for i in range(min(max_mismatches, num_mismatches)):
            idx = mismatch_indices[i]
            coord = tuple(idx.tolist())
            gpu_val = gpu_data[coord].item()
            ref_val = ref_data[coord].item()
            abs_error = diff[coord].item()

            print(
                f"{i + 1:<6} {str(coord):<30} {gpu_val:<20.6f} {ref_val:<20.6f} {abs_error:<20.6f}"
            )

        # Still raise assertion error to maintain original behavior
        raise AssertionError(
            f"{name} validation failed with {num_mismatches} mismatches"
        )
