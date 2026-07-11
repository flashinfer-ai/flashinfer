# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-side helpers shared by the lean and distributed MegaMoE runners."""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple

import torch

from common.megamoe_constants import (
    Fp8E5M2Max,
    Fp8E4M3FNMax,
    Nvfp4E2M1Max,
    Nvfp4BlockSize,
    Mxfp8BlockSize,  # noqa: F401
    SfPaddingBlock,
    TmaLeadingDimByteAlign,
    Nvfp4E2M1RcpLimit,
    Fp8E4M3RcpLimit,  # noqa: F401
    Fp8E5M2RcpLimit,  # noqa: F401
)

### TO BE REMOVED
DataDtype: torch.dtype = torch.float4_e2m1fn_x2
ScaleDtype: torch.dtype = torch.float8_e4m3fn

# Backward-compatible aliases used by existing runners.
_DataDtype = DataDtype
_ScaleDtype = ScaleDtype


Nvfp4DataDtype: torch.dtype = torch.float4_e2m1fn_x2
Nvfp4ScaleDtype: torch.dtype = torch.float8_e4m3fn
Mxfp8DataDtype_e4m3: torch.dtype = torch.float8_e4m3fn
Mxfp8DataDtype_e5m2: torch.dtype = torch.float8_e5m2
Mxfp8ScaleDtype: torch.dtype = torch.float8_e8m0fnu


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


from common.host_utils import kind_data_dtype, kind_scale_dtype, kind_sf_vec_size  # noqa: F401


def kind_data_max(kind: str) -> float:
    if kind == "nvfp4":
        return Nvfp4E2M1Max
    if kind == "mxfp8_e4m3":
        return Fp8E4M3FNMax
    if kind == "mxfp8_e5m2":
        return Fp8E5M2Max
    raise ValueError(f"Unknown kind: {kind!r}")


def leading_dim_bytes(leading_elems: int, dtype: torch.dtype) -> int:
    """Bytes occupied by one stride-1 row of ``leading_elems`` elements."""
    if dtype == Nvfp4DataDtype:
        if leading_elems % 2 != 0:
            raise ValueError(
                f"NVFP4 leading-dim element count ({leading_elems}) must be even "
                f"(2 fp4 packed per byte)."
            )
        return leading_elems // 2
    if dtype in (torch.bfloat16, torch.float16):
        return leading_elems * 2
    if dtype == torch.float32:
        return leading_elems * 4
    if dtype in (
        Nvfp4ScaleDtype,
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


_Fp4DecodeTable: torch.Tensor = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def fp4_packed_dim(tensor: torch.Tensor) -> int:
    """Return the packed stride-1 dimension in a ``float4_e2m1fn_x2`` tensor."""
    positive_strides = [
        (abs(stride), idx) for idx, stride in enumerate(tensor.stride()) if stride > 0
    ]
    if not positive_strides:
        return tensor.dim() - 1
    return min(positive_strides)[1]


def unpack_fp4_to_f32(packed: torch.Tensor) -> torch.Tensor:
    """Unpack a ``float4_e2m1fn_x2`` tensor into fp32 along the packed dim."""
    packed_dim = fp4_packed_dim(packed)
    raw = packed.view(torch.uint8)

    if packed_dim != raw.dim() - 1:
        perm = list(range(raw.dim()))
        perm[packed_dim], perm[-1] = perm[-1], perm[packed_dim]
        raw = raw.permute(perm).contiguous()
    else:
        perm = None

    lo = (raw & 0x0F).to(torch.int64)
    hi = (raw >> 4).to(torch.int64)
    lut = _Fp4DecodeTable.to(raw.device)

    unpacked_shape = list(raw.shape)
    unpacked_shape[-1] *= 2
    unpacked = torch.empty(unpacked_shape, dtype=torch.float32, device=raw.device)
    unpacked[..., ::2] = lut[lo]
    unpacked[..., 1::2] = lut[hi]

    if perm is not None:
        unpacked = unpacked.permute(perm)
    return unpacked


def slice_tensor_logical_dim(
    tensor: torch.Tensor, dim: int, start: int, end: int
) -> torch.Tensor:
    """Slice along a logical dimension, compensating for FP4 packing."""
    if tensor.dtype == Nvfp4DataDtype and dim == fp4_packed_dim(tensor):
        if start % 2 != 0 or end % 2 != 0:
            raise ValueError(
                f"FP4 packed slicing requires even indices, got start={start}, end={end}."
            )
        start = start // 2
        end = end // 2
    return tensor.narrow(dim, start, end - start)


def dequant_block_scale_to_fp32(
    data: torch.Tensor,
    raw_scale: torch.Tensor,
    blocksize: int,
    global_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize one 2D NVFP4 tensor using raw block scales."""
    data_fp32 = (
        unpack_fp4_to_f32(data)
        if data.dtype == Nvfp4DataDtype
        else data.to(torch.float32)
    )
    if data_fp32.dim() != 2 or raw_scale.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors, got data={data_fp32.dim()}D raw_scale={raw_scale.dim()}D."
        )

    expected = (data_fp32.shape[0], ceil_div(data_fp32.shape[1], blocksize))
    if tuple(raw_scale.shape) != expected:
        raise ValueError(
            f"Raw scale shape mismatch: expected {expected}, got {tuple(raw_scale.shape)}."
        )

    expanded = raw_scale.to(torch.float32).repeat_interleave(blocksize, dim=-1)[
        :, : data_fp32.shape[1]
    ]
    result = data_fp32 * expanded
    if global_scale is not None:
        result = result * global_scale.to(torch.float32).reshape(1, 1)
    return result


def transpose_rhs_for_block_dequant(data: torch.Tensor) -> torch.Tensor:
    """Convert a ``(K, N)`` RHS slice into ``(N, K)`` for K-block dequant."""
    if data.dim() != 2:
        raise ValueError(f"Expected 2D RHS tensor, got {data.dim()}D.")
    if data.dtype == Nvfp4DataDtype:
        return unpack_fp4_to_f32(data).transpose(0, 1)
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


def make_nvfp4_tensor_from_torch_rng(
    rng: torch.Generator,
    logical_shape: Tuple[int, ...],
    packed_dim: int,
    *,
    perf_run: bool,
    device: str = "cuda",
) -> torch.Tensor:
    """Build a deterministic ``float4_e2m1fn_x2`` tensor from an explicit RNG."""
    ndim = len(logical_shape)
    packed_dim = packed_dim % ndim
    if logical_shape[packed_dim] % 2 != 0:
        raise ValueError(
            f"packed_dim {packed_dim} size ({logical_shape[packed_dim]}) "
            f"must be even (2 fp4 packed per byte)."
        )

    if perf_run:
        total_elements = 1
        for dim_size in logical_shape:
            total_elements *= dim_size
        flat_bytes = torch.randint(
            0,
            256,
            (total_elements // 2,),
            dtype=torch.uint8,
            device=device,
            generator=rng,
        )
    else:
        random_u8 = torch.randint(
            0,
            100,
            logical_shape,
            dtype=torch.uint8,
            device=device,
            generator=rng,
        )
        nibbles = torch.zeros_like(random_u8)
        nibbles[(random_u8 >= 80) & (random_u8 < 90)] = 0x2
        nibbles[random_u8 >= 90] = 0xA

        need_perm = packed_dim != ndim - 1
        if need_perm:
            perm_to_last = list(range(ndim))
            perm_to_last[packed_dim], perm_to_last[-1] = (
                perm_to_last[-1],
                perm_to_last[packed_dim],
            )
            nibbles = nibbles.permute(perm_to_last).contiguous()
        even, odd = nibbles[..., 0::2], nibbles[..., 1::2]
        flat_bytes = ((odd << 4) | even).contiguous().reshape(-1)

    storage_shape = list(logical_shape)
    need_perm = packed_dim != ndim - 1
    if need_perm:
        storage_shape[packed_dim], storage_shape[-1] = (
            storage_shape[-1],
            storage_shape[packed_dim],
        )
    storage_shape[-1] //= 2
    tensor = flat_bytes.view(Nvfp4DataDtype).reshape(storage_shape)
    if need_perm:
        permute_back = list(range(ndim))
        permute_back[packed_dim], permute_back[-1] = (
            permute_back[-1],
            permute_back[packed_dim],
        )
        tensor = tensor.permute(permute_back)
    return tensor


def make_raw_scale_tensor_from_torch_rng(
    rng: torch.Generator,
    non_k_size: int,
    k_size: int,
    blocksize: int,
    *,
    device: str = "cuda",
    strict: bool = False,
) -> torch.Tensor:
    """Create deterministic raw FP8 block scales from an explicit RNG."""
    num_scale_cols = ceil_div(k_size, blocksize)

    if ScaleDtype == torch.float8_e4m3fn:
        scale_values = torch.tensor(
            [0.75, 1.0, 1.25, 1.5] if strict else [1.0, 2.0],
            dtype=torch.float32,
            device=device,
        )
    elif ScaleDtype == torch.float8_e8m0fnu:
        scale_values = torch.tensor(
            [0.25, 0.5, 1.0, 2.0] if strict else [1.0, 2.0],
            dtype=torch.float32,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported ScaleDtype: {ScaleDtype}")

    indices = torch.randint(
        0,
        scale_values.numel(),
        (non_k_size, num_scale_cols),
        device=device,
        generator=rng,
    )
    scales_fp32 = scale_values[indices]
    return scales_fp32.to(ScaleDtype).reshape(non_k_size, num_scale_cols)


def swiglu_fold_interleave(
    c_fp32: torch.Tensor,
    gate_up_interleave: int,
    gate_up_clamp: Optional[float] = None,
) -> torch.Tensor:
    """Apply the gate/up SwiGLU fold over a ``gate_up_interleave``-column
    interleaved layout (16 for NVFP4, 32 for MXFP8).

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


def swiglu_fold_interleave_16(
    c_fp32: torch.Tensor,
    gate_up_clamp: Optional[float] = None,
) -> torch.Tensor:
    """NVFP4 16-column interleave wrapper (kept for existing callers)."""
    return swiglu_fold_interleave(c_fp32, 16, gate_up_clamp=gate_up_clamp)


_Fp4ValuesEvenFirst: torch.Tensor = torch.tensor(
    [
        0.0,
        1.0,
        2.0,
        4.0,
        -0.0,
        -1.0,
        -2.0,
        -4.0,
        0.5,
        1.5,
        3.0,
        6.0,
        -0.5,
        -1.5,
        -3.0,
        -6.0,
    ],
    dtype=torch.float32,
)

_ReorderToNibble: torch.Tensor = torch.tensor(
    [0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF],
    dtype=torch.uint8,
)


def _pack_f32_to_fp4(fp32: torch.Tensor) -> torch.Tensor:
    """Round fp32 to FP4 E2M1 and nibble-pack into ``float4_e2m1fn_x2``."""
    device = fp32.device
    lut = _Fp4ValuesEvenFirst.to(device)
    nibble_map = _ReorderToNibble.to(device)
    flat = fp32.reshape(-1)
    diffs = (flat.unsqueeze(-1) - lut.unsqueeze(0)).abs()
    reordered_idx = diffs.argmin(dim=-1)
    indices = nibble_map[reordered_idx].view(fp32.shape)
    lo = indices[..., 0::2]
    hi = indices[..., 1::2]
    packed = (hi << 4) | lo
    return packed.view(Nvfp4DataDtype)


@functools.lru_cache(None)
def _get_rcp_approx_triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _rcp_approx_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
        y = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            "=r, r",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(y_ptr + offsets, y, mask=mask)

    return triton, _rcp_approx_kernel


def _rcp_approx_ftz_f32_cuda(x: torch.Tensor) -> torch.Tensor:
    """Bit-match kernel-side ``cute.arch.rcp_approx`` for CUDA fp32 tensors."""
    if not x.is_cuda or x.dtype != torch.float32:
        raise ValueError(
            "_rcp_approx_ftz_f32_cuda expects a CUDA float32 tensor; "
            f"got device={x.device}, dtype={x.dtype}."
        )
    x_contig = x.contiguous()
    y = torch.empty_like(x_contig)
    n_elements = x_contig.numel()
    if n_elements == 0:
        return y.view_as(x)

    triton, kernel = _get_rcp_approx_triton_kernel()
    block = 1024
    grid = (triton.cdiv(n_elements, block),)
    kernel[grid](x_contig, y, n_elements, BLOCK=block)
    return y.view_as(x)


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


def nvfp4_quantize_per_block_16(
    c_swiglu: torch.Tensor,
    norm_const: float,
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
    sfc_fp8 = sfc_fp32_clamped.to(Nvfp4ScaleDtype)
    sfc_fp32_rt = sfc_fp8.to(torch.float32)

    fp32_max = torch.finfo(torch.float32).max
    acc_scale = float(norm_const) * _rcp_approx_ftz_f32_cuda(sfc_fp32_rt)
    acc_scale = torch.nan_to_num(
        acc_scale, nan=fp32_max, posinf=fp32_max, neginf=fp32_max
    )
    acc_scale = torch.clamp(acc_scale, max=fp32_max)
    acc_scale = torch.where(sfc_fp32_rt > 0, acc_scale, torch.zeros_like(acc_scale))

    scaled = c_swiglu * acc_scale.unsqueeze(-1).expand_as(blocked).reshape(M, N)
    c_fp4 = _pack_f32_to_fp4(scaled)
    return c_fp4, sfc_fp8
