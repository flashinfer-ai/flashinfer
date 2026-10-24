# Copyright (c) 2025 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Torch quantization and layout reference helpers."""

import torch.nn.functional as F
import torch
from b12x._lib.intrinsics.constants import (
    E8M0_NAN_BYTE,
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    MX_SF_VEC_SIZE,
    SF_VEC_SIZE,
    _FP4_MAG_LUT,
    _INV_FLOAT8_E4M3_MAX,
)
from b12x._lib.intrinsics.math import align_up


def make_swizzle_indices(
    rows_padded: int, cols_padded: int, device: torch.device
) -> torch.Tensor:
    """Pre-compute swizzle gather indices for FP8 block-scale factors."""
    numel = rows_padded * cols_padded
    idx = torch.arange(numel, device=device)
    idx_5d = idx.reshape(rows_padded // 128, 4, 32, cols_padded // 4, 4)
    return idx_5d.permute(0, 3, 2, 1, 4).contiguous().reshape(-1)


def swizzle_block_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        scale = scale.unsqueeze(0)
        squeeze_batch = True
    elif scale.ndim == 3:
        squeeze_batch = False
    else:
        raise ValueError(f"scale must be 2D or 3D, got {tuple(scale.shape)}")

    batch, rows, cols = scale.shape
    rows_padded = align_up(rows, 128)
    cols_padded = align_up(cols, 4)

    padded = torch.zeros(
        (batch, rows_padded, cols_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:, :rows, :cols] = scale
    swizzled = padded.reshape(batch, rows_padded // 128, 4, 32, cols_padded // 4, 4)
    swizzled = swizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    swizzled = swizzled.reshape(batch, rows_padded, cols_padded)
    return swizzled[0] if squeeze_batch else swizzled


def validate_e8m0_scale_grid(
    grid: torch.Tensor,
    *,
    name: str,
    num_experts: int,
    rows: int,
    k: int,
    sf_block: int = 32,
) -> torch.Tensor:
    """Validate a checkpoint-native (unswizzled) UE8M0 block-scale grid.

    The grid must hold ``[num_experts, rows, k // sf_block]`` scale bytes in
    ``uint8`` or ``float8_e8m0fnu`` storage and be contiguous.  Every finite
    UE8M0 byte (``0x00``..``0xFE``) is preserved verbatim; ``0xFF`` (NaN) is
    rejected.  Returns a contiguous ``uint8`` view of the same bytes.
    """
    if not isinstance(grid, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if grid.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError(
            f"{name} must hold UE8M0 bytes (uint8/float8_e8m0fnu), got {grid.dtype}"
        )
    if k % sf_block:
        raise ValueError(
            f"{name}: k={k} must be a multiple of the {sf_block}-element "
            "UE8M0 block scale"
        )
    expected = (int(num_experts), int(rows), int(k) // sf_block)
    if grid.dim() != 3 or tuple(grid.shape) != expected:
        raise ValueError(
            f"{name} must be an unswizzled per-K/{sf_block} grid with shape "
            f"{expected}, got {tuple(grid.shape)}"
        )
    bytes_ = grid.view(torch.uint8)
    if not bytes_.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    invalid = int((bytes_ == E8M0_NAN_BYTE).sum().item())
    if invalid:
        raise ValueError(
            f"{name} holds {invalid} UE8M0 NaN scale byte(s) (0xFF); MX block "
            "scales must be finite (0x00..0xFE) and are never clamped"
        )
    return bytes_


def as_grouped_scale_view(
    scale_storage: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    batch = scale_storage.shape[0]
    rows_padded = align_up(rows, 128)
    cols_padded = align_up(cols // SF_VEC_SIZE, 4)
    sf = scale_storage.view(torch.float8_e4m3fn)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


def as_grouped_scale_view_mx(
    scale_storage: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """Grouped UE8M0 scale view for vec32 (MXFP8 w4a8) block scales."""
    batch = scale_storage.shape[0]
    rows_padded = align_up(rows, 128)
    cols_padded = align_up(cols // 32, 4)
    sf = scale_storage.view(torch.float8_e8m0fnu)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


def _fp4_quantize_values(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x = torch.abs(x.clone())
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def fp4_quantize_values_torch(x: torch.Tensor) -> torch.Tensor:
    """Pure-Torch FP4 E2M1 quantization with kernel-matching tie-breaking."""
    return _fp4_quantize_values(x)


def _fp4_encode_nibbles(values: torch.Tensor) -> torch.Tensor:
    mags = values.abs()
    idx = torch.zeros_like(values, dtype=torch.uint8)
    for code, mag in enumerate(_FP4_MAG_LUT):
        idx = torch.where(mags == mag, torch.full_like(idx, code), idx)
    sign_bit = (values < 0).to(torch.uint8) << 3
    return idx | sign_bit


def pack_grouped_fp4_values(values: torch.Tensor) -> torch.Tensor:
    """Pack grouped FP4 values `[G, R, C]` into uint8 `[R, C/2, G]`."""
    nibbles = _fp4_encode_nibbles(values)
    packed = nibbles.view(values.shape[0], values.shape[1], values.shape[2] // 2, 2)
    packed = packed[..., 0] | (packed[..., 1] << 4)
    return packed.permute(1, 2, 0).contiguous()


def _reciprocal_or_zero_torch(value: torch.Tensor) -> torch.Tensor:
    """Match device quantizers, which map an unrepresentable zero scale to zero."""
    return torch.where(value == 0, torch.zeros_like(value), 1.0 / value)


def quantize_grouped_nvfp4_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch grouped NVFP4 quantization with reciprocal global scales."""
    num_groups, rows, cols = input_tensor.shape
    if cols % SF_VEC_SIZE != 0:
        raise ValueError(f"cols must be divisible by {SF_VEC_SIZE}, got {cols}")
    if global_scale.numel() == 1:
        global_scale = global_scale.expand(num_groups).contiguous()

    quantized = torch.zeros(
        (num_groups, rows, cols), dtype=torch.float32, device=input_tensor.device
    )
    scales = torch.zeros(
        (num_groups, rows, cols // SF_VEC_SIZE),
        dtype=torch.float32,
        device=input_tensor.device,
    )
    for group_idx in range(num_groups):
        valid_rows = int(row_counts[group_idx].item())
        if valid_rows == 0:
            continue
        x = input_tensor[group_idx, :valid_rows].float()
        sliced = x.view(valid_rows, cols // SF_VEC_SIZE, SF_VEC_SIZE)
        block_max = sliced.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scale = (
            (global_scale[group_idx] * (block_max / FLOAT4_E2M1_MAX))
            .to(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        output_scale = _reciprocal_or_zero_torch(
            scale * _reciprocal_or_zero_torch(global_scale[group_idx])
        )
        clipped = torch.clamp(
            sliced * output_scale, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX
        ).view(valid_rows, cols)
        quantized[group_idx, :valid_rows] = _fp4_quantize_values(clipped)
        scales[group_idx, :valid_rows] = scale.squeeze(-1)

    packed = pack_grouped_fp4_values(quantized)
    swizzled = swizzle_block_scale(scales.to(torch.float8_e4m3fn))
    scale_view = as_grouped_scale_view(swizzled.view(torch.uint8), rows, cols)
    return packed, scale_view


def silu_mul_quantize_grouped_nvfp4_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = input_tensor.shape[-1] // 2
    left = input_tensor[..., :cols].float()
    right = input_tensor[..., cols:].float()
    activated = (F.silu(left) * right).to(input_tensor.dtype).to(torch.float32)
    return quantize_grouped_nvfp4_torch(activated, row_counts, global_scale)


def relu2_quantize_grouped_nvfp4_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch grouped NVFP4 quantization for ReLU^2 FC1 outputs."""
    activated = torch.square(F.relu(input_tensor.float()))
    activated = activated.to(input_tensor.dtype).to(torch.float32)
    return quantize_grouped_nvfp4_torch(activated, row_counts, global_scale)


def pow2_ceil_ue8m0_torch(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact Torch replica of the ``pow2_ceil_ue8m0`` device intrinsic.

    Rounds positive fp32 scales UP to a power of two by bumping the exponent
    whenever the mantissa is nonzero.  Returns ``(rounded_fp32, ue8m0_u8)``;
    a zero scale maps to (0.0, byte 0).
    """
    bits = scale.to(torch.float32).contiguous().view(torch.int32)
    mant = bits & 0x007FFFFF
    bumped = torch.where(mant != 0, (bits + 0x00800000) & 0x7F800000, bits)
    rounded = bumped.view(torch.float32)
    byte = ((bumped >> 23) & 0xFF).to(torch.uint8)
    return rounded, byte


def _ue8m0_output_scale_torch(byte: torch.Tensor) -> torch.Tensor:
    """Torch replica of ``ue8m0_to_output_scale``: 2^(127-byte), 0 for byte 0."""
    inv_bits = (254 - byte.to(torch.int32)).clamp(min=0) << 23
    inv = inv_bits.view(torch.float32)
    return torch.where(byte == 0, torch.zeros_like(inv), inv)


def quant_dequant_mxfp8_torch(x: torch.Tensor) -> torch.Tensor:
    """Per-32-block MXFP8 quantize-dequantize roundtrip (oracle helper).

    Matches the in-kernel ``quantize_block_fp8_mx`` numerics bit-for-bit:
    UE8M0 ceil block scale, E4M3 RN-saturating payload.
    """
    orig_shape = x.shape
    cols = orig_shape[-1]
    if cols % MX_SF_VEC_SIZE != 0:
        raise ValueError(f"last dim must be divisible by {MX_SF_VEC_SIZE}, got {cols}")
    blocked = x.to(torch.float32).reshape(-1, cols // MX_SF_VEC_SIZE, MX_SF_VEC_SIZE)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    rounded, byte = pow2_ceil_ue8m0_torch(block_max * _INV_FLOAT8_E4M3_MAX)
    inv = _ue8m0_output_scale_torch(byte)
    payload = (
        (blocked * inv)
        .clamp(-FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    return (payload * rounded).reshape(orig_shape)


def quant_dequant_mxfp8_scaled_torch(
    x: torch.Tensor,
    global_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Per-32-block MXFP8 quantize-dequantize WITH a global scale (oracle).

    Bit-for-bit replica of the fused-MoE route-pack / FC2-requant container
    quantizer ``b12x._lib.fp6.quantize_block_fp8_e4m3_containers``, i.e.
    ``fp6_block_ue8m0_exact`` (IEEE exponent-field + nonzero-mantissa ceil of
    ``(block_amax * global_scale) / 448``) then ``ue8m0_output_scale_exact``
    (``global_scale * 2**(127 - byte)``) then E4M3 round-to-nearest-saturating.
    The dequantized value is ``payload * 2**(byte - 127)``.

    Not interchangeable with ``quantize_mxfp8_rows_torch``: that dense-linear
    helper derives its byte with ``ceil(log2(...))`` and maps an all-zero
    block to byte 127, while the MoE container path uses exact bit extraction
    and byte 0.

    ``global_scale`` is broadcastable against the per-block amax grid: a
    scalar, or one value per row (the per-expert activation global scale a
    routed row was quantized with).
    """
    orig_shape = x.shape
    cols = orig_shape[-1]
    if cols % MX_SF_VEC_SIZE != 0:
        raise ValueError(f"last dim must be divisible by {MX_SF_VEC_SIZE}, got {cols}")
    blocked = x.to(torch.float32).reshape(-1, cols // MX_SF_VEC_SIZE, MX_SF_VEC_SIZE)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    gs = torch.as_tensor(global_scale, dtype=torch.float32, device=x.device)
    gs = gs.reshape(-1, 1, 1)
    if gs.shape[0] not in (1, blocked.shape[0]):
        raise ValueError(
            f"global_scale has {gs.shape[0]} rows, expected 1 or {blocked.shape[0]}"
        )
    rounded, byte = pow2_ceil_ue8m0_torch((block_max * gs) / FLOAT8_E4M3_MAX)
    inv = _ue8m0_output_scale_torch(byte) * gs
    payload = (
        (blocked * inv)
        .clamp(-FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    return (payload * rounded).reshape(orig_shape)


def quantize_grouped_mxfp8_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch grouped MXFP8 quantization (no global scale).

    Mirrors ``quantize_grouped_nvfp4_torch`` for the w4a8 activation path:
    returns ``(payload, scale_view)`` where ``payload`` is E4M3 bytes in
    ``[R, C, G]`` order and ``scale_view`` is the swizzled UE8M0 grouped
    view (vec32 columns).
    """
    num_groups, rows, cols = input_tensor.shape
    if cols % MX_SF_VEC_SIZE != 0:
        raise ValueError(f"cols must be divisible by {MX_SF_VEC_SIZE}, got {cols}")

    payload = torch.zeros(
        (num_groups, rows, cols), dtype=torch.uint8, device=input_tensor.device
    )
    scales = torch.zeros(
        (num_groups, rows, cols // MX_SF_VEC_SIZE),
        dtype=torch.uint8,
        device=input_tensor.device,
    )
    for group_idx in range(num_groups):
        valid_rows = int(row_counts[group_idx].item())
        if valid_rows == 0:
            continue
        x = input_tensor[group_idx, :valid_rows].float()
        blocked = x.view(valid_rows, cols // MX_SF_VEC_SIZE, MX_SF_VEC_SIZE)
        block_max = blocked.abs().amax(dim=-1, keepdim=True)
        rounded, byte = pow2_ceil_ue8m0_torch(block_max * _INV_FLOAT8_E4M3_MAX)
        inv = _ue8m0_output_scale_torch(byte)
        q = (
            (blocked * inv)
            .clamp(-FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .view(valid_rows, cols)
        )
        payload[group_idx, :valid_rows] = q
        scales[group_idx, :valid_rows] = byte.squeeze(-1)

    payload_view = payload.permute(1, 2, 0).contiguous()
    swizzled = swizzle_block_scale(scales)
    scale_view = as_grouped_scale_view_mx(swizzled, rows, cols)
    return payload_view, scale_view


def silu_mul_quantize_grouped_mxfp8_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SiLU(left)*right then grouped MXFP8 quantization (w4a8 FC2 input)."""
    cols = input_tensor.shape[-1] // 2
    left = input_tensor[..., :cols].float()
    right = input_tensor[..., cols:].float()
    activated = (F.silu(left) * right).to(input_tensor.dtype).to(torch.float32)
    return quantize_grouped_mxfp8_torch(activated, row_counts)


def relu2_quantize_grouped_mxfp8_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ReLU^2 then grouped MXFP8 quantization (w4a8 FC2 input, non-gated)."""
    activated = torch.square(F.relu(input_tensor.float()))
    activated = activated.to(input_tensor.dtype).to(torch.float32)
    return quantize_grouped_mxfp8_torch(activated, row_counts)
