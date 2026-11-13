# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host utility helpers shared across NVFP4 and MXFP8 runners."""

from .megamoe_constants import (
    Fp8E4M3FNMax,
    Fp8E4M3RcpLimit,
    Fp8E5M2RcpLimit,
    Mxfp8BlockSize,
    Nvfp4BlockSize,
    Nvfp4E2M1Max,
    Nvfp4E2M1RcpLimit,
    SfPaddingBlock,
    TmaLeadingDimByteAlign,
)
from .utils import ceil_div, round_up


import argparse
import os
import sys
from typing import List, Optional, Tuple
import torch


# ---------------------------------------------------------------------------
# Dtype constants
# ---------------------------------------------------------------------------

_Nvfp4DataDtype: torch.dtype = torch.float4_e2m1fn_x2
_Nvfp4ScaleDtype: torch.dtype = torch.float8_e4m3fn
_Mxfp8DataDtype_e4m3: torch.dtype = torch.float8_e4m3fn
_Mxfp8DataDtype_e5m2: torch.dtype = torch.float8_e5m2
_Mxfp8ScaleDtype: torch.dtype = torch.float8_e8m0fnu
_Fp8KindsE4M3 = ("mxfp8_e4m3", "fp8_e4m3")
_Fp8KindsE5M2 = ("mxfp8_e5m2", "fp8_e5m2")
_Fp8Kinds = _Fp8KindsE4M3 + _Fp8KindsE5M2

# ---------------------------------------------------------------------------
# kind_* helpers
# ---------------------------------------------------------------------------

def get_cutedsl_target_arch() -> str:
    """Return the active cuTeDSL compilation target as an ``sm_XX`` string."""
    from cutlass.cutlass_dsl import CuTeDSL

    arch = CuTeDSL._get_dsl().get_arch_enum()
    return f"sm_{arch.major}{arch.minor}"


def kind_data_dtype(kind: str) -> torch.dtype:
    if kind == "nvfp4":
        return _Nvfp4DataDtype
    if kind in _Fp8KindsE4M3:
        return _Mxfp8DataDtype_e4m3
    if kind in _Fp8KindsE5M2:
        return _Mxfp8DataDtype_e5m2
    if kind == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unknown kind: {kind!r}")


def kind_scale_dtype(kind: str) -> torch.dtype:
    if kind == "nvfp4":
        return _Nvfp4ScaleDtype
    if kind in _Fp8Kinds:
        return _Mxfp8ScaleDtype
    raise ValueError(f"Unknown kind: {kind!r}")


def kind_sf_vec_size(kind: str) -> int:
    if kind == "nvfp4":
        return Nvfp4BlockSize
    if kind in _Fp8Kinds:
        return Mxfp8BlockSize
    raise ValueError(f"Unknown kind: {kind!r}")


# ---------------------------------------------------------------------------
# Mxfp8 quantize function. May move function to mxfp8 folder later
# ---------------------------------------------------------------------------

def _mxfp8_e8m0_scale_raw(
    absmax: torch.Tensor,
    data_dtype: torch.dtype,
) -> torch.Tensor:
    """Return the exact E8M0 scale byte for a non-negative FP32 amax.

    An FP8 limit is ``1.75 * 2**k``.  For ``absmax = m * 2**e`` from
    ``frexp`` (``0.5 <= m < 1``), the rounded-up scale exponent is
    ``e - k - 1 + (m > 0.875)``.  This avoids both an arbitrary lower clamp
    and floating-point ``log2`` rounding at the 1.75 threshold.
    """
    if data_dtype == torch.float8_e4m3fn:
        limit_exponent = 8
    elif data_dtype == torch.float8_e5m2:
        limit_exponent = 15
    else:
        raise ValueError(f"Unsupported MXFP8 data dtype: {data_dtype}.")

    significand, exponent = torch.frexp(absmax)
    scale_exp = (
        exponent.to(torch.int32)
        - (limit_exponent + 1)
        + (significand > 0.875).to(torch.int32)
    )
    raw = torch.clamp(scale_exp + 127, min=0, max=254)
    raw = torch.where(absmax == 0, torch.zeros_like(raw), raw)
    return torch.where(
        torch.isfinite(absmax), raw, torch.full_like(raw, 254)
    )


def mxfp8_quantize_per_block_32_row(
    c_fp32: torch.Tensor,
    data_dtype: torch.dtype,
    *,
    match_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize per-32 blocks using exact scales or kernel-matching arithmetic.

    The default derives the scale exponent exactly and divides by that scale.
    match_kernel uses instruction-rounded scales, approximate reciprocals and
    the epilogue's tiny-scale mask, including its nonfinite behavior.
    """
    M, N = c_fp32.shape
    if N % Mxfp8BlockSize != 0:
        raise ValueError(
            f"Trailing dim ({N}) must be a multiple of sf_vec_size ({Mxfp8BlockSize})."
        )
    n_blocks = N // Mxfp8BlockSize
    blocked = c_fp32.view(M, n_blocks, Mxfp8BlockSize)
    absmax = blocked.abs().amax(dim=-1)
    if match_kernel:
        reciprocal_limit = Fp8E4M3RcpLimit if data_dtype == _Mxfp8DataDtype_e4m3 else Fp8E5M2RcpLimit
        sfc_e8m0, reciprocal = e8m0_scale_and_reciprocal(absmax, reciprocal_limit)
        scaled = c_fp32 * reciprocal.unsqueeze(-1).expand_as(blocked).reshape(M, N)
    else:
        e_uint8 = _mxfp8_e8m0_scale_raw(absmax, data_dtype)
        sfc_e8m0 = e_uint8.to(torch.uint8).view(_Mxfp8ScaleDtype)
        scale_fp32 = torch.pow(2.0, e_uint8.float() - 127.0)
        scale_expanded = scale_fp32.unsqueeze(-1).expand_as(blocked).reshape(M, N)
        scaled = c_fp32 / scale_expanded
    c_fp8 = scaled.to(data_dtype)
    return c_fp8, sfc_e8m0


def mxfp8_quantize_per_block_32_col(
    c_fp32: torch.Tensor,
    data_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MXFP8-quantize along the leading dim with per-32 E8M0 col-block scales.

    Mirrors :func:`mxfp8_quantize_per_block_32_row` but reduces down the M axis:
    one scale covers a ``(Mxfp8BlockSize-row × 1-col)`` block, so the SF
    tensor has shape ``(M // Mxfp8BlockSize, N)``.  The M axis must be a
    multiple of ``Mxfp8BlockSize``.
    """
    M, N = c_fp32.shape
    if M % Mxfp8BlockSize != 0:
        raise ValueError(
            f"Leading dim ({M}) must be a multiple of sf_vec_size ({Mxfp8BlockSize})."
        )
    m_blocks = M // Mxfp8BlockSize
    blocked = c_fp32.view(m_blocks, Mxfp8BlockSize, N)
    absmax = blocked.abs().amax(dim=1)  # (m_blocks, N)
    e_uint8 = _mxfp8_e8m0_scale_raw(absmax, data_dtype)
    sfc_e8m0 = e_uint8.to(torch.uint8).view(_Mxfp8ScaleDtype)
    scale_fp32 = torch.pow(2.0, e_uint8.float() - 127.0)
    scale_expanded = scale_fp32.unsqueeze(1).expand_as(blocked).reshape(M, N)
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
        print(f"\n")

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

    # Compute differences.  Non-finite values are always mismatches: relying
    # only on ``diff > threshold`` would let NaNs (and equal infinities) pass.
    gpu_float = gpu_data.float()
    ref_float = ref_data.float()
    diff = _torch.abs(gpu_float - ref_float)
    threshold = atol + rtol * _torch.abs(ref_float)
    mismatch_mask = (
        ~_torch.isfinite(gpu_float)
        | ~_torch.isfinite(ref_float)
        | (diff > threshold)
    )

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


def rcp_approx_ftz(value: torch.Tensor) -> torch.Tensor:
    """Apply the device FP32 approximate reciprocal on the current stream."""
    if not value.is_cuda or value.dtype != torch.float32:
        raise ValueError("rcp_approx_ftz expects a CUDA float32 tensor")
    from .cute_ref_ops import rcp_approx_ftz_f32

    return rcp_approx_ftz_f32(value.contiguous()).view_as(value)


def physical_axis_order(value: torch.Tensor) -> List[int]:
    """Order tensor axes from largest stride to smallest stride."""
    return sorted(range(value.ndim), key=lambda axis: value.stride(axis), reverse=True)


def pack_f32_to_fp4(value: torch.Tensor, *, physical_layout: bool = False) -> torch.Tensor:
    """Pack E2M1 pairs along the last logical or physically contiguous axis."""
    if not value.is_cuda or value.dtype != torch.float32:
        raise ValueError("pack_f32_to_fp4 expects a CUDA float32 tensor")
    if value.ndim == 0:
        raise ValueError("FP4 packing requires at least one dimension")
    order = physical_axis_order(value) if physical_layout else list(range(value.ndim))
    storage = value.permute(order).contiguous()
    *lead, last = storage.shape
    if last % 2:
        raise ValueError(f"FP4 packing needs an even packed dimension, got {last}")
    from .cute_ref_ops import pack_f32_to_fp4_u8

    packed = pack_f32_to_fp4_u8(storage.reshape(-1))
    packed = packed.view(*lead, last // 2).view(_Nvfp4DataDtype)
    return packed.permute([order.index(axis) for axis in range(value.ndim)])


def nvfp4_quantize_per_block_16(
    c_swiglu: torch.Tensor,
    norm_const: float,
    *,
    physical_layout: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4-quantize the SwiGLU output along its trailing dim."""
    M, N = c_swiglu.shape
    if N % Nvfp4BlockSize != 0:
        raise ValueError(
            f"Trailing dim ({N}) must be a multiple of sf_vec_size ({Nvfp4BlockSize})."
        )
    n_blocks = N // Nvfp4BlockSize

    blocked = c_swiglu.view(M, n_blocks, Nvfp4BlockSize)
    absmax = blocked.abs().amax(dim=-1)
    sfc_fp32 = absmax * Nvfp4E2M1RcpLimit * norm_const
    sfc_fp32_clamped = torch.clamp(sfc_fp32, min=-Fp8E4M3FNMax, max=Fp8E4M3FNMax)
    sfc_fp8 = sfc_fp32_clamped.to(_Nvfp4ScaleDtype)
    sfc_fp32_rt = sfc_fp8.to(torch.float32)

    fp32_max = torch.finfo(torch.float32).max
    acc_scale = float(norm_const) * rcp_approx_ftz(sfc_fp32_rt)
    acc_scale = torch.nan_to_num(acc_scale, nan=fp32_max, posinf=fp32_max, neginf=fp32_max)
    acc_scale = torch.clamp(acc_scale, max=fp32_max)
    acc_scale = torch.where(
        sfc_fp32_rt > 0, acc_scale, torch.zeros_like(acc_scale)
    )

    scaled = c_swiglu * acc_scale.unsqueeze(-1).expand_as(blocked).reshape(M, N)
    c_fp4 = pack_f32_to_fp4(scaled, physical_layout=physical_layout)
    return c_fp4, sfc_fp8


def round_f32_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Round CUDA fp32 through PTX ``cvt.rp.satfinite.ue8m0``."""
    if not x.is_cuda or x.dtype != torch.float32:
        raise ValueError("round_f32_to_e8m0 expects CUDA float32.")
    x_contiguous = x.contiguous()
    flat = x_contiguous.reshape(-1)
    result = torch.empty(flat.numel(), dtype=torch.uint8, device=x.device)
    if flat.numel() > 0:
        from .cute_ref_ops import round_f32_to_e8m0_u8

        result = round_f32_to_e8m0_u8(flat)
    return result.view(_Mxfp8ScaleDtype).reshape(x.shape)


def e8m0_scale_and_reciprocal(absmax: torch.Tensor, data_max_rcp: float) -> Tuple[torch.Tensor, torch.Tensor]:
    block_scale_e8m0 = round_f32_to_e8m0(absmax * data_max_rcp)
    scale = block_scale_e8m0.to(torch.float32)
    fp32_max = torch.finfo(torch.float32).max
    reciprocal = rcp_approx_ftz(scale)
    reciprocal = torch.nan_to_num(reciprocal, nan=fp32_max, posinf=fp32_max, neginf=fp32_max)
    reciprocal = torch.clamp(reciprocal, max=fp32_max)
    mask = torch.clamp(scale * 1e30, max=1.0)
    return block_scale_e8m0, reciprocal * mask


def from_blocked(flat: torch.Tensor, raw_rows: int, raw_cols: int) -> torch.Tensor:
    """Inverse of :func:`to_blocked` for the 32x4x4 FP8 scale layout."""
    if flat.dim() != 1:
        raise ValueError(f"Expected 1D flat tensor, got {flat.dim()}D.")
    if raw_rows == 0 or raw_cols == 0:
        return flat.new_empty((raw_rows, raw_cols))

    row_blocks = ceil_div(raw_rows, SfPaddingBlock)
    col_blocks = ceil_div(raw_cols, 4)
    padded_rows = row_blocks * SfPaddingBlock
    padded_cols = col_blocks * 4
    expected = padded_rows * padded_cols
    if flat.numel() != expected:
        raise ValueError(
            f"from_blocked: flat size {flat.numel()} != expected "
            f"{expected} for raw ({raw_rows}, {raw_cols}) padded to "
            f"({padded_rows}, {padded_cols})."
        )

    rearranged = flat.reshape(-1, 32, 16).reshape(-1, 32, 4, 4)
    blocks = rearranged.transpose(1, 2).reshape(-1, SfPaddingBlock, 4)
    blocks = blocks.reshape(row_blocks, col_blocks, SfPaddingBlock, 4)
    padded = blocks.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return padded[:raw_rows, :raw_cols].contiguous()


def to_blocked(scale_2d: torch.Tensor) -> torch.Tensor:
    """Pad and apply the 32x4x4 FP8 scale swizzle to one raw scale tensor."""
    if scale_2d.dim() != 2:
        raise ValueError(f"Expected 2D scale tensor, got {scale_2d.dim()}D.")
    rows, cols = scale_2d.shape
    if rows == 0 or cols == 0:
        return scale_2d.new_empty((0,))

    row_blocks = ceil_div(rows, SfPaddingBlock)
    col_blocks = ceil_div(cols, 4)
    padded_rows = row_blocks * SfPaddingBlock
    padded_cols = col_blocks * 4

    padded = scale_2d
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols), dtype=scale_2d.dtype, device=scale_2d.device
        )
        padded[:rows, :cols] = scale_2d

    blocks = padded.view(row_blocks, SfPaddingBlock, col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()
