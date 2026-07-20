# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-token-group FP8 / INT8 quantization using the public cuda.tile API,
# with row-major or column-major scale output.

from typing import Optional
from typing import Tuple
from typing import TypeAlias

import cuda.tile as ct
import torch

ConstInt: TypeAlias = ct.Constant[int]


def _next_power_of_2(n):
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@ct.kernel
def _per_token_group_quant_8bit_kernel(
    y_input,
    y_quantized,
    y_scale,
    y_stride,
    n,
    eps,
    bit8_min,
    bit8_max,
    BLOCK: ConstInt,
    y_array_size,
    y_q_array_size,
    y_s_array_size,
):
    """Per-token-group quantization kernel (row-major scales)."""
    g_id = ct.bid(0)

    # Compute base offsets for this group
    y_base = g_id * y_stride
    y_q_base = g_id * y_stride

    # Create column offsets and mask
    cols = ct.arange(BLOCK, dtype=ct.int32)
    mask = cols < n

    # Load input values with gather (element-level access with mask)
    y_indices = y_base + cols
    y = ct.gather(y_input, (y_indices,), check_bounds=True, padding_value=0.0)
    y = ct.astype(y, ct.float32)

    # Zero the padded lanes (BLOCK = next_pow2(group_size) > group_size when
    # group_size is not a power of two). Those lanes gather in-bounds data from
    # the *next* group (gather's padding_value only fires on true OOB), so
    # without this they would corrupt the absmax reduction below.
    y = ct.where(mask, y, 0.0)

    # Compute absmax
    abs_y = ct.abs(y)
    _absmax = ct.max(abs_y, axis=0)
    _absmax = ct.maximum(_absmax, eps)

    # Compute scale and inverse scale
    y_s = _absmax / bit8_max
    y_s_inv = 1.0 / y_s

    # Quantize: clamp(y * y_s_inv, bit8_min, bit8_max)
    y_q = ct.minimum(ct.maximum(y * y_s_inv, bit8_min), bit8_max)
    # ct.astype float->int truncates toward zero, but the INT8 reference rounds
    # to nearest -- truncation would leave a systematic ~0.5-LSB bias. cuda.tile
    # exposes no round op, so round-half-up via floor(x + 0.5) on the (already
    # clamped, so in-range) value. FP8 casts round in hardware, so only the
    # integer path needs this.
    if y_quantized.dtype == ct.int8:
        y_q = ct.floor(y_q + 0.5)
    y_q = ct.astype(y_q, y_quantized.dtype)

    # Store quantized values with mask (use OOB offsets for invalid positions)
    oob_offset = ct.full((BLOCK,), y_q_array_size, dtype=ct.int32)
    y_q_indices = y_q_base + cols
    y_q_indices_masked = ct.where(mask, y_q_indices, oob_offset)
    ct.scatter(y_quantized, (y_q_indices_masked,), y_q, check_bounds=True)

    # Store scale (single scalar per group)
    y_s_idx = g_id
    oob_scalar = ct.full((), y_s_array_size, dtype=ct.int32)
    s_idx_masked = ct.where(y_s_idx < y_s_array_size, y_s_idx, oob_scalar)
    ct.scatter(y_scale, (s_idx_masked,), y_s)


@ct.kernel
def _per_token_group_quant_8bit_colmajor_kernel(
    y_input,
    y_quantized,
    y_scale,
    group_size,
    y_num_columns,
    y_row_stride,
    y_s_col_stride,
    eps,
    bit8_min,
    bit8_max,
    scale_ue8m0,
    BLOCK: ConstInt,
    y_array_size,
    y_q_array_size,
    y_s_array_size,
):
    """Per-token-group quantization kernel (column-major scales)."""
    groups_per_row = y_num_columns // group_size

    g_id = ct.bid(0)
    row = g_id // groups_per_row
    group_id = g_id % groups_per_row

    # Compute base offsets
    y_base = row * y_row_stride + group_id * group_size
    y_q_base = g_id * group_size
    y_s_offset = group_id * y_s_col_stride + row

    # Create column offsets and mask
    cols = ct.arange(BLOCK, dtype=ct.int32)
    mask = cols < group_size

    # Load input values
    y_indices = y_base + cols
    y = ct.gather(y_input, (y_indices,), check_bounds=True, padding_value=0.0)
    y = ct.astype(y, ct.float32)

    # Zero the padded lanes (BLOCK = next_pow2(group_size) > group_size when
    # group_size is not a power of two). Those lanes gather in-bounds data from
    # the *next* group (gather's padding_value only fires on true OOB), so
    # without this they would corrupt the absmax reduction below.
    y = ct.where(mask, y, 0.0)

    # Compute absmax
    abs_y = ct.abs(y)
    _absmax = ct.max(abs_y, axis=0)
    _absmax = ct.maximum(_absmax, eps)

    # Compute scale
    y_s = _absmax / bit8_max

    # Optional: round scale to power of 2 (UE8M0)
    if scale_ue8m0:
        abs_y_s = ct.abs(y_s)
        safe_y_s = ct.maximum(abs_y_s, 1e-10)
        y_s = ct.exp2(ct.ceil(ct.log2(safe_y_s)))

    # Quantize: clamp(y / y_s, bit8_min, bit8_max)
    y_q = ct.minimum(ct.maximum(y / y_s, bit8_min), bit8_max)
    # ct.astype float->int truncates toward zero; round-half-up for INT8 to
    # match the rounding reference (see row-major kernel). FP8 rounds in HW.
    if y_quantized.dtype == ct.int8:
        y_q = ct.floor(y_q + 0.5)
    y_q = ct.astype(y_q, y_quantized.dtype)

    # Store quantized values with mask
    oob_offset = ct.full((BLOCK,), y_q_array_size, dtype=ct.int32)
    y_q_indices = y_q_base + cols
    y_q_indices_masked = ct.where(mask, y_q_indices, oob_offset)
    ct.scatter(y_quantized, (y_q_indices_masked,), y_q, check_bounds=True)

    # Store scale (single scalar)
    oob_scalar = ct.full((), y_s_array_size, dtype=ct.int32)
    s_idx_masked = ct.where(y_s_offset < y_s_array_size, y_s_offset, oob_scalar)
    ct.scatter(y_scale, (s_idx_masked,), y_s)


def _ceil_align(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def per_token_group_quant_8bit_cutile(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dst_dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token group 8-bit quantization (FP8 or INT8) — cuTile backend.

    Implemented with the public cuda.tile API.

    Args:
        x: Input tensor to quantize.
        group_size: Number of elements per quantization group (last dimension).
        eps: Epsilon for numerical stability in scale computation.
        dst_dtype: Output dtype (default: torch.float8_e4m3fn).
        column_major_scales: Store scales in column-major layout (for TMA compatibility).
        scale_tma_aligned: Align scale storage to 4-element (16B) boundaries.
        scale_ue8m0: Round scales to UE8M0 (power-of-2) format.

    Returns:
        (x_q, x_s): Quantized tensor and scale tensor.
    """
    if dst_dtype is None:
        dst_dtype = torch.float8_e4m3fn
    if dst_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8):
        raise ValueError(
            "dst_dtype must be float8_e4m3fn, float8_e5m2, or int8; "
            f"got {dst_dtype}"
        )
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be divisible by `group_size` {group_size}"
    )
    assert x.is_contiguous(), "`x` must be contiguous"
    if scale_tma_aligned or scale_ue8m0:
        assert column_major_scales, (
            "scale_tma_aligned or scale_ue8m0 requires column_major_scales=True"
        )

    if dst_dtype == torch.int8:
        info = torch.iinfo(dst_dtype)
        bit8_min = float(info.min)
        bit8_max = float(info.max)
    else:
        info = torch.finfo(dst_dtype)
        bit8_min = info.min
        bit8_max = info.max

    # These kernels are pure element-wise scatters: every group writes its full
    # [g*group_size, g*group_size+group_size) span of x_q, so the union over all
    # groups covers the whole output — no pre-zeroing needed (unlike the beta=0
    # GEMM epilogue). Using empty_like avoids a redundant full-tensor memset on a
    # bandwidth-bound path.
    x_q = torch.empty_like(x, device=x.device, dtype=dst_dtype)
    M = x.numel() // group_size
    N = group_size

    if column_major_scales:
        if x.dim() != 2:
            raise ValueError(
                "column_major_scales is only supported for 2D inputs; "
                f"got a {x.dim()}D input"
            )
        num_groups = x.shape[-1] // group_size
        num_tokens = x.shape[-2] if x.dim() >= 2 else x.shape[0]
        if scale_tma_aligned:
            # TMA-friendly layout: (num_groups, aligned_num_tokens), align to 4 floats (16B)
            aligned_size = _ceil_align(num_tokens, 4)
            # Every (group, token) slot is written by the kernel; the only
            # uninitialized region is the [num_tokens:aligned_size] alignment
            # pad, which is sliced off below before returning, so empty is safe.
            x_s_raw = torch.empty(
                (num_groups, aligned_size),
                device=x.device,
                dtype=torch.float32,
            )
            x_s_col_stride = aligned_size
        else:
            shape = (num_groups,) + x.shape[:-1]
            x_s_raw = torch.empty(shape, device=x.device, dtype=torch.float32).permute(
                -1, -2
            )
            x_s_col_stride = x_s_raw.stride(1)
        x_s = x_s_raw
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = _next_power_of_2(N)

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)

    # Flatten tensors for gather/scatter access
    x_flat = x.contiguous().view(-1)
    x_q_flat = x_q.view(-1)
    x_s_flat = x_s.contiguous().view(-1) if not column_major_scales else x_s

    if column_major_scales:
        # Use a contiguous flat view of x_s for scatter
        # x_s may be non-contiguous (permuted), so we use as_strided to get the raw storage
        x_s_for_kernel = (
            x_s_raw.view(-1)
            if x_s_raw.is_contiguous()
            else torch.as_strided(
                x_s_raw,
                (x_s_raw.numel(),),
                (1,),
                storage_offset=x_s_raw.storage_offset(),
            )
        )

        ct.launch(
            stream,
            grid,
            _per_token_group_quant_8bit_colmajor_kernel,
            (
                x_flat,
                x_q_flat,
                x_s_for_kernel,
                group_size,
                x.shape[-1],
                x.stride(-2) if x.dim() >= 2 else x.shape[-1],
                x_s_col_stride,
                eps,
                bit8_min,
                bit8_max,
                1 if scale_ue8m0 else 0,
                BLOCK,
                x_flat.numel(),
                x_q_flat.numel(),
                x_s_for_kernel.numel(),
            ),
        )
        if scale_tma_aligned:
            # Return a transposed VIEW, not a contiguous copy: x_s_raw is
            # (num_groups, aligned_size) row-major, so .t() yields a
            # (num_tokens, num_groups) column-major tensor whose token dim is
            # unit-stride and whose group stride keeps the aligned_size (16B)
            # TMA padding. .contiguous() here would repack to a tight row-major
            # layout, destroying both the column-major property and the TMA
            # alignment this branch exists to produce.
            x_s = x_s_raw[:, :num_tokens].t()
    else:
        assert not scale_ue8m0

        ct.launch(
            stream,
            grid,
            _per_token_group_quant_8bit_kernel,
            (
                x_flat,
                x_q_flat,
                x_s_flat,
                group_size,
                N,
                eps,
                bit8_min,
                bit8_max,
                BLOCK,
                x_flat.numel(),
                x_q_flat.numel(),
                x_s_flat.numel(),
            ),
        )

    return x_q, x_s
