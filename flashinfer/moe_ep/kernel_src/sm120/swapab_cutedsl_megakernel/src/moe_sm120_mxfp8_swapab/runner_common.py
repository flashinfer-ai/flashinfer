# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-side MXFP8 helpers shared by the single-rank and MegaMoE runners."""

from __future__ import annotations

import functools
from typing import List, Optional

import torch

from common.megamoe_constants import SfPaddingBlock, TmaLeadingDimByteAlign


Mxfp8DataDtype_e4m3: torch.dtype = torch.float8_e4m3fn
Mxfp8DataDtype_e5m2: torch.dtype = torch.float8_e5m2
Mxfp8ScaleDtype: torch.dtype = torch.float8_e8m0fnu


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def leading_dim_bytes(leading_elems: int, dtype: torch.dtype) -> int:
    """Bytes occupied by one stride-1 row of ``leading_elems`` elements."""
    if dtype in (torch.bfloat16, torch.float16):
        return leading_elems * 2
    if dtype == torch.float32:
        return leading_elems * 4
    if dtype in (
        Mxfp8DataDtype_e4m3,
        Mxfp8DataDtype_e5m2,
        Mxfp8ScaleDtype,
    ):
        return leading_elems
    raise ValueError(f"leading_dim_bytes: unsupported dtype {dtype!r}.")


def check_tma_leading_dim_align(
    tensor_name: str, leading_elems: int, dtype: torch.dtype
) -> None:
    """Reject a tensor whose stride-1 row is not TMA aligned."""
    leading_bytes = leading_dim_bytes(leading_elems, dtype)
    if leading_bytes % TmaLeadingDimByteAlign != 0:
        raise ValueError(
            f"{tensor_name}: leading-dim byte size = {leading_bytes} "
            f"(= {leading_elems} elements of {dtype}) is not a multiple of "
            f"{TmaLeadingDimByteAlign} bytes; TMA descriptor requires "
            f"{TmaLeadingDimByteAlign}-byte alignment for the stride-1 row."
        )


def offs_to_group_sizes(offs: torch.Tensor) -> List[int]:
    """Convert cumulative-end offsets to per-expert valid token counts."""
    offs_cpu = offs.cpu().tolist()
    prev = 0
    sizes: List[int] = []
    for end in offs_cpu:
        sizes.append(int(end) - prev)
        prev = int(end)
    return sizes


def slice_tensor_logical_dim(
    tensor: torch.Tensor, dim: int, start: int, end: int
) -> torch.Tensor:
    """Slice an MXFP8 tensor along a logical dimension."""
    return tensor.narrow(dim, start, end - start)


def dequant_block_scale_to_fp32(
    data: torch.Tensor,
    raw_scale: torch.Tensor,
    blocksize: int,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize one 2D MXFP8 tensor using raw UE8M0 block scales."""
    data_fp32 = data.to(torch.float32)
    if data_fp32.dim() != 2 or raw_scale.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors, got data={data_fp32.dim()}D "
            f"raw_scale={raw_scale.dim()}D."
        )

    expected = (data_fp32.shape[0], ceil_div(data_fp32.shape[1], blocksize))
    if tuple(raw_scale.shape) != expected:
        raise ValueError(
            f"Raw scale shape mismatch: expected {expected}, "
            f"got {tuple(raw_scale.shape)}."
        )

    expanded = raw_scale.to(torch.float32).repeat_interleave(blocksize, dim=-1)[
        :, : data_fp32.shape[1]
    ]
    result = data_fp32 * expanded
    if global_scale is not None:
        result = result * global_scale.to(torch.float32).reshape(1, 1)
    return result


def transpose_rhs_for_block_dequant(data: torch.Tensor) -> torch.Tensor:
    """Convert a K-major ``(K, N)`` RHS into logical ``(N, K)``."""
    if data.dim() != 2:
        raise ValueError(f"Expected 2D RHS tensor, got {data.dim()}D.")
    return data.transpose(0, 1)

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


def _cat_byte_reinterpretable_tensors(
    tensors: List[torch.Tensor], dim: int = 0
) -> torch.Tensor:
    """Concatenate byte-backed float tensors via uint8 view."""
    if not tensors:
        raise ValueError("Expected at least one tensor to concatenate.")
    first = tensors[0]
    if first.is_floating_point() and first.element_size() == 1:
        concatenated = torch.cat([t.view(torch.uint8) for t in tensors], dim=dim)
        return concatenated.view(first.dtype)
    return torch.cat(tensors, dim=dim)


def _stack_byte_reinterpretable_tensors(
    tensors: List[torch.Tensor], dim: int = 0
) -> torch.Tensor:
    """Stack byte-backed float tensors via uint8 view."""
    if not tensors:
        raise ValueError("Expected at least one tensor to stack.")
    first = tensors[0]
    if first.is_floating_point() and first.element_size() == 1:
        stacked = torch.stack([t.view(torch.uint8) for t in tensors], dim=dim)
        return stacked.view(first.dtype)
    return torch.stack(tensors, dim=dim)


def assemble_raw_scales_grouped_token(raw_scales: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-expert raw SF tensors grouped along the token axis."""
    flat_parts = [to_blocked(s) for s in raw_scales]
    all_flat = _cat_byte_reinterpretable_tensors(flat_parts, dim=0)
    total_rows = sum(round_up(s.shape[0], SfPaddingBlock) for s in raw_scales)
    return all_flat.reshape(total_rows, -1)


def assemble_raw_scales_stacked_expert(raw_scales: List[torch.Tensor]) -> torch.Tensor:
    """Stack per-expert raw SF tensors after applying the 32x4x4 swizzle."""
    flat_parts = [to_blocked(s) for s in raw_scales]
    return _stack_byte_reinterpretable_tensors(flat_parts, dim=0)


def _create_raw_scale_tensor(
    non_k_size: int,
    k_size: int,
    blocksize: int,
    scale_dtype: torch.dtype,
    device: str = "cuda",
    strict: bool = False,
) -> torch.Tensor:
    """Create one 2-D raw block-scale tensor with dtype-specific scale values."""
    scale_cols = ceil_div(k_size, blocksize)

    if scale_dtype == torch.float8_e4m3fn:
        scale_values = torch.tensor(
            [0.75, 1.0, 1.25, 1.5] if strict else [1.0, 2.0],
            dtype=torch.float32,
            device=device,
        )
    elif scale_dtype == torch.float8_e8m0fnu:
        scale_values = torch.tensor(
            [0.25, 0.5, 1.0, 2.0] if strict else [1.0, 2.0],
            dtype=torch.float32,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported scale_dtype: {scale_dtype}")

    indices = torch.randint(
        0,
        scale_values.numel(),
        (non_k_size, scale_cols),
        device=device,
    )
    scales = scale_values[indices]
    return scales.to(scale_dtype).reshape(non_k_size, scale_cols)

def swiglu_fold_interleave(
    c_fp32: torch.Tensor,
    gate_up_interleave: int,
    gate_up_clamp: Optional[float] = None,
) -> torch.Tensor:
    """Apply the gate/up SwiGLU fold over a ``gate_up_interleave``-column
    interleaved layout used by the SM120 swap-AB accumulator mapping.

    ``gate_up_clamp`` mirrors DeepSeek-V4's ``config.swiglu_limit``
    (``DeepseekV4Experts.forward``): an asymmetric clamp on the real
    (already-dequanted) gate/up pre-activations, ``gate = clamp(gate, max=limit)``
    and ``up = clamp(up, -limit, +limit)``, applied before SiLU.  ``None``
    disables it.  The caller must pass the post-``fc1_alpha`` tensor so the
    clamp acts on real values, matching the kernel.
    """
    M, intermediate = c_fp32.shape
    if intermediate % (2 * gate_up_interleave) != 0:
        raise ValueError(
            f"intermediate ({intermediate}) must be a multiple of "
            f"{2 * gate_up_interleave} for {gate_up_interleave}-granularity "
            f"gate/up interleave."
        )
    n_pairs = intermediate // (2 * gate_up_interleave)
    reshaped = c_fp32.view(M, n_pairs, 2, gate_up_interleave)
    gate = reshaped[:, :, 0, :]
    up = reshaped[:, :, 1, :]
    if gate_up_clamp is not None:
        limit = float(gate_up_clamp)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    out = _swiglu_pair_hw_match_cuda(gate, up)
    return out.reshape(M, intermediate // 2)



@functools.lru_cache(None)
def _get_swiglu_pair_hw_match_triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_pair_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
        ug = up * gate
        neg_g_l2e = gate * (-1.4426950408889634)
        exp_neg = tl.inline_asm_elementwise(
            "ex2.approx.f32 $0, $1;",
            "=r, r",
            [neg_g_l2e],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        one_plus = exp_neg + 1.0
        sigmoid = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=r, r",
            [one_plus],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        out = ug * sigmoid
        tl.store(out_ptr + offsets, out, mask=mask)

    return triton, _swiglu_pair_kernel


def _swiglu_pair_hw_match_cuda(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute SwiGLU matching the kernel-side PTX op sequence."""
    if not gate.is_cuda or not up.is_cuda:
        return up * (gate * torch.sigmoid(gate))
    if gate.dtype != torch.float32 or up.dtype != torch.float32:
        return _swiglu_pair_hw_match_cuda(
            gate.to(torch.float32), up.to(torch.float32)
        ).to(gate.dtype)
    if gate.shape != up.shape:
        raise ValueError(
            f"_swiglu_pair_hw_match_cuda: gate.shape {tuple(gate.shape)} "
            f"!= up.shape {tuple(up.shape)}."
        )
    if gate.numel() == 0:
        return torch.empty_like(gate)

    gate_c = gate.contiguous()
    up_c = up.contiguous()
    out = torch.empty_like(gate_c)
    n_elements = gate_c.numel()

    triton, kernel = _get_swiglu_pair_hw_match_triton_kernel()
    block = 1024
    grid = (triton.cdiv(n_elements, block),)
    kernel[grid](gate_c, up_c, out, n_elements, BLOCK=block)
    return out.view_as(gate)
