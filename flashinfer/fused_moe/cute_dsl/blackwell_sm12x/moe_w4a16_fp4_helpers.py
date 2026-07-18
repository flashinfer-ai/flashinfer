"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Common utilities for FP4 quantization kernels.

This module contains shared PTX intrinsics, helper functions, and reduction
utilities used by the fused frontend kernels.
"""

import functools
import math
import operator
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


# =============================================================================
# Constants
# =============================================================================

FLOAT4_E2M1_MAX = 6.0  # Maximum value representable in FP4 E2M1
FLOAT8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3
SF_VEC_SIZE = 16  # Elements per scale factor block
COPY_BITS = 128  # 128-bit vectorized loads
_FP4_MAG_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


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


def as_grouped_scale_view(
    scale_storage: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    batch = scale_storage.shape[0]
    rows_padded = align_up(rows, 128)
    cols_padded = align_up(cols // SF_VEC_SIZE, 4)
    sf = scale_storage.view(torch.float8_e4m3fn)
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
        output_scale = 1.0 / (scale * (1.0 / global_scale[group_idx]))
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


# =============================================================================
# Architecture Detection
# =============================================================================


@functools.lru_cache(maxsize=16)
def get_sm_version(device: int | torch.device | str | None = None) -> int:
    """Get the SM version of a CUDA device.

    Args:
        device: CUDA device to query. Can be an int (device index), torch.device,
            device string (e.g., 'cuda:0'), or None to use current device.

    Returns:
        SM version as an integer (e.g., 120 on the current target).
    """
    if not torch.cuda.is_available():
        return 80
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


# =============================================================================
# PTX Intrinsics - Cluster Operations
# =============================================================================


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map smem pointer to address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store Float32 value to shared memory on a remote CTA in the cluster."""
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


# =============================================================================
# PTX Intrinsics - 128-bit Vectorized Global Loads/Stores
# =============================================================================


@dsl_user_op
def ld_global_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)

    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


# =============================================================================
# PTX Intrinsics - Non-Coherent Global Loads
# =============================================================================


@dsl_user_op
def ld_global_nc_u32(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load 32 bits from global memory via non-coherent cache (.nc)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def prefetch_global_l2(base_ptr: Int64, *, loc=None, ip=None) -> None:
    """Prefetch a global memory line into L2."""
    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_nc_v2_u32(base_ptr: Int64, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """Load 64 bits (2 x uint32) from global memory via non-coherent cache (.nc)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.u32 {$0, $1}, [$2];",
        "=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def ld_global_nc_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from global memory via non-coherent cache (.nc)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def st_global_u64(base_ptr: Int64, value: Uint64, *, loc=None, ip=None):
    """Store 64 bits to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_global_u8(base_ptr: Int64, value: Uint8, *, loc=None, ip=None):
    """Store 8 bits to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint8(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u8 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_global_f32(base_ptr: Int64, value: Float32, *, loc=None, ip=None):
    """Store 32-bit float to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_global_v2_f32(base_ptr: Int64, v0: Float32, v1: Float32, *, loc=None, ip=None):
    """Store 64 bits (2 x float32) to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_global_v4_f32(
    base_ptr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x float32) to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_global_v4_f32(
    base_ptr: Int64,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x float32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def st_global_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x uint32) to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def smem_ptr_to_addr(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Convert a generic/smem pointer to a shared-memory u32 address for PTX."""
    generic_addr = llvm.ptrtoint(T.i64(), ptr.llvm_ptr, loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [generic_addr],
            "cvta.to.shared.u32 $0, $1;",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def shared_ptr_to_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Convert an address-space-3 shared-memory pointer to a u32 address."""
    return Int32(llvm.ptrtoint(T.i32(), ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def ldmatrix_m8n8x4_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.m8n8.x4.shared.b16` from a shared-memory byte address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1), Uint32(r2), Uint32(r3)


@dsl_user_op
def ldmatrix_m8n8x2_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.m8n8.x2.shared.b16` from a shared-memory byte address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def ldmatrix_m8n8x4_left_half_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.m8n8.x4.shared.b16` and return the left half fragment pair."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, _, $1, _}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def ldmatrix_m8n8x4_right_half_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.m8n8.x4.shared.b16` and return the right half fragment pair."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {_, $0, _, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def ldmatrix_m8n8x4_trans_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16` from a shared-memory byte address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1), Uint32(r2), Uint32(r3)


@dsl_user_op
def ldmatrix_m8n8x4_trans_left_half_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Issue transposed `ldmatrix` and return the left half fragment pair."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {$0, $1, _, _}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def ldmatrix_m8n8x4_trans_right_half_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Issue transposed `ldmatrix` and return the right half fragment pair."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.trans.m8n8.x4.shared.b16 {_, _, $0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def ld_shared_v4_u32(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from shared memory. smem_addr is a u32 shared-memory address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1), Uint32(r2), Uint32(r3)


@dsl_user_op
def ld_shared_v2_u32(smem_addr: Int32, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """Load 64 bits (2 x uint32) from shared memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def st_shared_u64(smem_addr: Int32, value: Uint64, *, loc=None, ip=None):
    """Store 64 bits to shared memory. smem_addr is a u32 shared-memory address."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u64 [$0], $1;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_shared_v4_u32(
    smem_addr: Int32,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x uint32) to shared memory. smem_addr is a u32 shared-memory address."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_shared_u8(smem_addr: Int32, value: Uint8, *, loc=None, ip=None):
    """Store 8 bits to shared memory. smem_addr is a u32 shared-memory address."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint8(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u8 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_async4_shared_global(smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None):
    """16-byte `cp.async.cg.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async4_shared_global_pred(
    smem_addr: Int32, gmem_addr: Int64, pred: Int32, *, loc=None, ip=None
):
    """Predicated 16-byte `cp.async.cg.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p cp.async.cg.shared.global [$1], [$2], 16; }",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async4_ca_shared_global_pred(
    smem_addr: Int32, gmem_addr: Int64, pred: Int32, *, loc=None, ip=None
):
    """Predicated 16-byte `cp.async.ca.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p cp.async.ca.shared.global [$1], [$2], 16; }",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64."""
    elem_ptr = tensor.iterator + offset
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


# =============================================================================
# PTX Intrinsics - Global Atomics
# =============================================================================


@dsl_user_op
def atomic_add_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None) -> Int32:
    """Global memory int32 atomic add. Returns old value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def red_add_global_release_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """No-return global int32 add with a GPU-scope release fence."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "fence.acq_rel.gpu;\nred.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_cas_global_i32(
    addr: Int64, compare: Int32, value: Int32, *, loc=None, ip=None
) -> Int32:
    """Global memory int32 atomic compare-and-swap. Returns old value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(compare).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.cas.b32 $0, [$1], $2, $3;",
            "=r,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def atomic_add_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None) -> Int32:
    """Shared-memory int32 atomic add (CTA-scope). Returns old value.

    Uses ``atom.shared.add.s32`` with a 32-bit shared-memory address
    (the native address width for smem on NVIDIA GPUs).
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared.add.s32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def red_add_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None):
    """No-return shared-memory int32 add.

    Use this when the old value is not needed. NVCC lowers equivalent C++
    histogram increments to cheaper no-return shared atomics.
    """
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_i32(addr: Int32, *, loc=None, ip=None) -> Int32:
    """Volatile-style load int32 from shared memory at a 32-bit byte address."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_i32_relaxed(addr: Int32, *, loc=None, ip=None) -> Int32:
    """Load int32 from shared memory at a 32-bit byte address."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_u32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    """Load uint32 from shared memory at a 32-bit byte address."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None):
    """Store int32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_u32(addr: Int32, val: Uint32, *, loc=None, ip=None):
    """Store uint32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Uint32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    """Load float32 from shared memory at a 32-bit byte address."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_v4_f32(
    addr: Int32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x float32) from shared memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def ld_shared_bf16_to_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    """Load a BF16 from shared memory at a 32-bit byte address and convert to FP32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "{.reg .b16 tmp; ld.shared.b16 tmp, [$1]; cvt.f32.bf16 $0, tmp;}",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Store float32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_v4_f32(
    addr: Int32,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x float32) to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_bf16_from_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Convert float32 to BF16 and store one 16-bit value to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 tmp; cvt.rn.bf16.f32 tmp, $1; st.shared.b16 [$0], tmp; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_f16_from_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Convert float32 to FP16 and store one 16-bit value to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 tmp; cvt.rn.f16.f32 tmp, $1; st.shared.b16 [$0], tmp; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """Store int32 to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# PTX Intrinsics - Global Memory Barriers
# =============================================================================


@dsl_user_op
def ld_global_acquire_i32(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Load int32 from global memory with acquire semantics."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.s32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_release_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """Store int32 to global memory with release semantics."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.release.gpu.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def spin_wait_global_eq_i32(addr: Int64, expected: Int32, *, loc=None, ip=None):
    """Spin-wait until *addr != expected (acquire semantics on load)."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(expected).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred %p0;\n"
        ".reg .s32 %val;\n"
        "spin_loop:\n"
        "  ld.global.acquire.gpu.s32 %val, [$0];\n"
        "  setp.eq.s32 %p0, %val, $1;\n"
        "  @%p0 bra spin_loop;\n"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def spin_wait_global_ge_i32(addr: Int64, target: Int32, *, loc=None, ip=None):
    """Spin-wait until *addr >= target using acquire loads."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(target).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred %p0;\n"
        ".reg .s32 %val;\n"
        "spin_loop_ge:\n"
        "  ld.global.acquire.gpu.s32 %val, [$0];\n"
        "  setp.lt.s32 %p0, %val, $1;\n"
        "  @%p0 bra spin_loop_ge;\n"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def threadfence(*, loc=None, ip=None):
    """Emit membar.gl — threadfence across global memory."""
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# PTX Intrinsics - Scatter Atomics
# =============================================================================


@dsl_user_op
def scatter_add_bf16x2(addr: Int64, val0_f32, val1_f32, *, loc=None, ip=None):
    """BF16x2 atomic reduction add to global memory.

    Packs two f32 values into bf16x2 via cvt.rn.satfinite, then does
    red.relaxed.gpu.global.add.noftz.bf16x2. Workaround for LLVM NVPTX
    bug that decomposes vector<2xbf16> into .v2.bf16x2 which ptxas rejects.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            val0_f32.ir_value(loc=loc, ip=ip),
            val1_f32.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 packed; cvt.rn.satfinite.bf16x2.f32 packed, $2, $1; red.relaxed.gpu.global.add.noftz.bf16x2 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def scatter_add_v4_bf16x2(
    addr: Int64, v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None
):
    """Vectorized BF16x2 atomic reduction: 8 bf16 values (16 bytes) in one go.

    red.global.add.noftz.v4.bf16x2 [addr], {packed0, packed1, packed2, packed3}
    Each packed reg contains 2 bf16 values converted from f32 pairs.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
            v2.ir_value(loc=loc, ip=ip),
            v3.ir_value(loc=loc, ip=ip),
            v4.ir_value(loc=loc, ip=ip),
            v5.ir_value(loc=loc, ip=ip),
            v6.ir_value(loc=loc, ip=ip),
            v7.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 p0,p1,p2,p3;"
        " cvt.rn.satfinite.bf16x2.f32 p0, $2, $1;"
        " cvt.rn.satfinite.bf16x2.f32 p1, $4, $3;"
        " cvt.rn.satfinite.bf16x2.f32 p2, $6, $5;"
        " cvt.rn.satfinite.bf16x2.f32 p3, $8, $7;"
        " red.global.add.noftz.v4.bf16x2 [$0], {p0, p1, p2, p3}; }",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# PTX Intrinsics - Math Operations
# =============================================================================


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """Fast reciprocal using PTX rcp.approx.ftz.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmin_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min of two float32 values using PTX min.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max of two float32 values using PTX max.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fabs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """Compute absolute value of float32 using PTX abs.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Half2 SIMD Intrinsics
# =============================================================================


@dsl_user_op
def half2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two Half2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "mul.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def broadcast_f32_to_half2(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack one float32 value into both lanes of an f16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hadd2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two Half2 values element-wise: (a.x+b.x, a.y+b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 absolute value - clears sign bits of both fp16 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 max - element-wise max of 2 fp16 pairs."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 fp16 values in half2 as float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h0, h1;
                .reg .f32 f0, f1;
                mov.b32 {h0, h1}, $1;
                cvt.f32.f16 f0, h0;
                cvt.f32.f16 f1, h1;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def half2_to_float2_scaled(
    h2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert half2 to float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(h2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1;
            .reg .f32 f0, f1;
            mov.b32 {h0, h1}, $2;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


# =============================================================================
# BFloat2 SIMD Intrinsics
# =============================================================================


@dsl_user_op
def bfloat2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two BFloat2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "mul.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_broadcast_lane(x: Uint32, lane: Int32, *, loc=None, ip=None) -> Uint32:
    """Duplicate one BF16 lane from a packed bf16x2 register into both lanes."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(x).ir_value(loc=loc, ip=ip),
                Int32(lane).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p;
                .reg .b32 lo, hi, val, shifted;
                and.b32 lo, $1, 0x0000ffff;
                shr.u32 hi, $1, 16;
                setp.eq.s32 p, $2, 0;
                @p  mov.b32 val, lo;
                @!p mov.b32 val, hi;
                shl.b32 shifted, val, 16;
                or.b32 $0, val, shifted;
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def broadcast_f32_to_bfloat2(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack one float32 value into both lanes of a bf16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.bf16x2.f32 $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pack_f32x2_to_bfloat2(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack 2 float32 values into one bf16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.satfinite.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_bf16x2_to_e4m3x2(src: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert packed bf16x2 to packed e4m3x2 in the low 16 bits of a u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(src).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out, zero;
                cvt.rn.satfinite.e4m3x2.bf16x2 out, $1;
                mov.u16 zero, 0;
                mov.b32 $0, {out, zero};
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_bf16x2x2_to_e4m3x4(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert two bf16x2 values and pack the e4m3x2 results into one u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out_lo, out_hi;
                cvt.rn.satfinite.e4m3x2.bf16x2 out_lo, $1;
                cvt.rn.satfinite.e4m3x2.bf16x2 out_hi, $2;
                mov.b32 $0, {out_lo, out_hi};
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_add(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two BFloat2 values element-wise: (a.x+b.x, a.y+b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "add.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fp8x4_e4m3_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Widen 4 packed E4M3 bytes into 2 packed bf16x2 registers exactly."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 low;
            .reg .b32 q, out0, out1, sign0, sign1, mant0, mant1, tmp, bias, bias_f32;
            prmt.b32 q, $2, 0, 0x1302;

            and.b32 sign0, q, 0x80008000;
            shr.u32 tmp, q, 4;
            and.b32 mant0, tmp, 0x07F007F0;
            or.b32 out0, mant0, sign0;

            shl.b32 tmp, q, 8;
            and.b32 sign1, tmp, 0x80008000;
            shl.b32 tmp, q, 4;
            and.b32 mant1, tmp, 0x07F007F0;
            or.b32 out1, mant1, sign1;

            mov.b32 bias_f32, 0x7B800000;
            cvt.rn.bf16.f32 low, bias_f32;
            mov.b32 bias, {low, low};

            mul.bf16x2 $0, out0, bias;
            mul.bf16x2 $1, out1, bias;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e2m1x4_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE2M1 -> BF16 register dequant for one packed 4-value fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 out1, $2, 0x80008000;
            and.b32 tmp, $2, 0x70007000;
            shr.u32 tmp, tmp, 6;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 4;
            and.b32 out2, q, 0x80008000;
            and.b32 tmp, q, 0x70007000;
            shr.u32 tmp, tmp, 6;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e2m1x4_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE2M1 -> FP16 register dequant for one packed 4-value fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 out1, $2, 0x80008000;
            and.b32 tmp, $2, 0x70007000;
            shr.u32 tmp, tmp, 3;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 4;
            and.b32 out2, q, 0x80008000;
            and.b32 tmp, q, 0x70007000;
            shr.u32 tmp, tmp, 3;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e4m3x4_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE4M3 scale dequant for one packed 4-value BF16 fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 tmp, $2, 0x80008000;
            shr.u32 out1, tmp, 1;
            and.b32 tmp, $2, 0x7F007F00;
            shr.u32 tmp, tmp, 4;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 8;
            and.b32 tmp, q, 0x80008000;
            shr.u32 out2, tmp, 1;
            and.b32 tmp, q, 0x7F007F00;
            shr.u32 tmp, tmp, 4;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e4m3x4_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE4M3 scale dequant for one packed 4-value FP16 fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2;

            and.b32 out1, $2, 0xFF00FF00;
            shr.u32 out1, out1, 1;

            shl.b32 q, $2, 8;
            and.b32 out2, q, 0xFF00FF00;
            shr.u32 out2, out2, 1;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def bf16_mma_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA helper for `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def f16_mma_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA helper for `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    b0_0: Uint32,
    b1_0: Uint32,
    b0_1: Uint32,
    b1_1: Uint32,
    a0: Uint32,
    a1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """BF16 MMA form used by the routed m-block-size-8 path.

    The dequantized RHS fragments feed the hardware A operand, while the
    routed activation fragment loaded with `ldmatrix.x2` feeds hardware B.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(b0_0).ir_value(loc=loc, ip=ip),
            Uint32(b1_0).ir_value(loc=loc, ip=ip),
            Uint32(b0_1).ir_value(loc=loc, ip=ip),
            Uint32(b1_1).ir_value(loc=loc, ip=ip),
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    b0_0: Uint32,
    b1_0: Uint32,
    b0_1: Uint32,
    b1_1: Uint32,
    a0: Uint32,
    a1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """FP16 MMA form used by the routed m-block-size-8 path."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(b0_0).ir_value(loc=loc, ip=ip),
            Uint32(b1_0).ir_value(loc=loc, ip=ip),
            Uint32(b0_1).ir_value(loc=loc, ip=ip),
            Uint32(b1_1).ir_value(loc=loc, ip=ip),
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def bf16_rowsum_m16k16_f32(
    d0: Float32,
    d1: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """Row-sum helper matching FlashInfer's m16k16 BF16 rowsum MMA."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(1065369472).ir_value(loc=loc, ip=ip),
            Uint32(1065369472).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
            {$0, _, $1, _},
            {$2, $3, $4, $5},
            {$6, $7},
            {$0, 0., $1, 0.};
        }
        """,
        "=f,=f,r,r,r,r,r,r,0,1",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(r0), Float32(r1)


@dsl_user_op
def bf16_mma_m16n16k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    d4: Float32,
    d5: Float32,
    d6: Float32,
    d7: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    b2: Uint32,
    b3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32]:
    """Warp MMA helper matching FlashInfer's `m16n16k16` BF16/BF16->F32 wrapper."""
    result0 = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    result1 = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b2).ir_value(loc=loc, ip=ip),
            Uint32(b3).ir_value(loc=loc, ip=ip),
            Float32(d4).ir_value(loc=loc, ip=ip),
            Float32(d5).ir_value(loc=loc, ip=ip),
            Float32(d6).ir_value(loc=loc, ip=ip),
            Float32(d7).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result0, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result0, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result0, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result0, [3], loc=loc, ip=ip)
    r4 = llvm.extractvalue(T.f32(), result1, [0], loc=loc, ip=ip)
    r5 = llvm.extractvalue(T.f32(), result1, [1], loc=loc, ip=ip)
    r6 = llvm.extractvalue(T.f32(), result1, [2], loc=loc, ip=ip)
    r7 = llvm.extractvalue(T.f32(), result1, [3], loc=loc, ip=ip)
    return (
        Float32(r0),
        Float32(r1),
        Float32(r2),
        Float32(r3),
        Float32(r4),
        Float32(r5),
        Float32(r6),
        Float32(r7),
    )


@dsl_user_op
def mxfp8_mma_m16n8k32_f32_e4m3(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA helper for SM120 MXFP8 block-scaled `m16n8k32` E4M3/E4M3."""
    i16_ty = cutlass._mlir.ir.IntegerType.get_signless(16)
    bid_a_i16 = cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(bid_a))},
    ).result
    tid_a_i16 = cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(tid_a))},
    ).result
    bid_b_i16 = cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(bid_b))},
    ).result
    tid_b_i16 = cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(tid_b))},
    ).result
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Uint32(sfa).ir_value(loc=loc, ip=ip),
            bid_a_i16,
            tid_a_i16,
            Uint32(sfb).ir_value(loc=loc, ip=ip),
            bid_b_i16,
            tid_b_i16,
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$0, $1, $2, $3},
        {$10},
        {$11, $12},
        {$13},
        {$14, $15};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,h,h,r,h,h,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def byte_perm(a: Uint32, b: Uint32, selector: Int32, *, loc=None, ip=None) -> Uint32:
    """PTX byte permutation helper."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Int32(selector).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def frag_layout_swizzle_16b_to_8b(x: Uint32) -> Uint32:
    tmp = cute.arch.shuffle_sync_bfly(x, offset=1)
    x = byte_perm(x, tmp, Int32(0x5410 if cute.arch.lane_idx() & 0x1 == 0 else 0x3276))
    tmp = cute.arch.shuffle_sync_bfly(x, offset=2)
    x = byte_perm(x, tmp, Int32(0x5410 if cute.arch.lane_idx() & 0x2 == 0 else 0x3276))
    return x


@cute.jit
def frag_layout_swizzle_16b_to_8b_trans(x: Uint32) -> Uint32:
    tmp = cute.arch.shuffle_sync_bfly(x, offset=4)
    x = byte_perm(x, tmp, Int32(0x6420 if cute.arch.lane_idx() & 0x4 == 0 else 0x3175))
    tmp = cute.arch.shuffle_sync_bfly(x, offset=8)
    x = byte_perm(x, tmp, Int32(0x5410 if cute.arch.lane_idx() & 0x8 == 0 else 0x3276))
    tmp = cute.arch.shuffle_sync_bfly(x, offset=16)
    x = byte_perm(x, tmp, Int32(0x5410 if cute.arch.lane_idx() & 0x10 == 0 else 0x3276))
    return x


@dsl_user_op
def bfloat2_habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 absolute value - clears sign bits of both bf16 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 max - element-wise max of 2 bf16 pairs."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 bf16 values in bfloat2 as float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_to_float2_scaled(
    bf2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert bfloat16x2 to float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


# =============================================================================
# FP8 E4M3 Intrinsics
# =============================================================================


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float32 to E4M3 using native cvt.rn.satfinite.e4m3x2.f32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def nvfp4_scale_from_amax(
    block_amax: Float32, global_scale: Float32, *, loc=None, ip=None
) -> Float32:
    """Compute block_amax * reciprocal_global_scale / 6 with CUDA tensor-scalar semantics."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(block_amax).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f64 amax_d, gs_d, q_d, six_d;
                cvt.f64.f32 amax_d, $1;
                cvt.f64.f32 gs_d, $2;
                mov.f64 six_d, 0d4018000000000000;
                mul.rn.f64 q_d, amax_d, gs_d;
                div.rn.f64 q_d, q_d, six_d;
                cvt.rn.f32.f64 $0, q_d;
            }
            """,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp8_e4m3_to_f32(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert FP8 E4M3 to float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg;
                .reg .u32 sign_u, exp_u, mant_u;
                .reg .s32 exp_s;
                .reg .f32 exp_f, mant_f, fp8_float, fp8_neg;

                setp.eq.u32 p_zero, $1, 0;
                and.b32 sign_u, $1, 0x80;
                and.b32 mant_u, $1, 7;
                shr.b32 exp_u, $1, 3;
                and.b32 exp_u, exp_u, 15;
                sub.s32 exp_s, exp_u, 7;
                cvt.rn.f32.s32 exp_f, exp_s;
                ex2.approx.f32 exp_f, exp_f;
                cvt.rn.f32.u32 mant_f, mant_u;
                fma.rn.f32 mant_f, mant_f, 0f3E000000, 0f3F800000;
                mul.f32 fp8_float, exp_f, mant_f;
                neg.f32 fp8_neg, fp8_float;
                setp.ne.u32 p_neg, sign_u, 0;
                selp.f32 fp8_float, fp8_neg, fp8_float, p_neg;
                selp.f32 $0, 0f00000000, fp8_float, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fp8_e4m3_to_f32_and_rcp(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert FP8 E4M3 to float32 AND compute reciprocal."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .u32 exp_u, mant_u;
                .reg .s32 exp_s;
                .reg .f32 exp_f, mant_f, fp8_float, result;

                setp.eq.u32 p_zero, $1, 0;
                and.b32 mant_u, $1, 7;
                shr.b32 exp_u, $1, 3;
                and.b32 exp_u, exp_u, 15;
                sub.s32 exp_s, exp_u, 7;
                cvt.rn.f32.s32 exp_f, exp_s;
                ex2.approx.f32 exp_f, exp_f;
                cvt.rn.f32.u32 mant_f, mant_u;
                fma.rn.f32 mant_f, mant_f, 0f3E000000, 0f3F800000;
                mul.f32 fp8_float, exp_f, mant_f;
                rcp.approx.ftz.f32 result, fp8_float;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# E4M3 -> F16/F32 Native Conversion Intrinsics
# =============================================================================


@dsl_user_op
def cvt_e4m3x2_to_f16x2_pair(
    packed_u32: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Decode 4 e4m3 bytes (packed u32) into 2 x f16x2 pairs."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 b16_01, b16_23;
            .reg .b32 hi32;
            cvt.u16.u32 b16_01, $2;
            shr.b32 hi32, $2, 16;
            cvt.u16.u32 b16_23, hi32;
            cvt.rn.f16x2.e4m3x2 $0, b16_01;
            cvt.rn.f16x2.e4m3x2 $1, b16_23;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def f16x2_to_f32x2(packed_h2: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Unpack f16x2 (u32) to two f32 values."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [Uint32(packed_h2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@cute.jit
def cvt_e4m3x4_to_f32x4(
    packed_u32: Uint32,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Decode 4 e4m3 bytes (packed u32) to 4 x f32 via hw-native e4m3->f16->f32."""
    h01, h23 = cvt_e4m3x2_to_f16x2_pair(packed_u32)
    f0, f1 = f16x2_to_f32x2(h01)
    f2, f3 = f16x2_to_f32x2(h23)
    return f0, f1, f2, f3


@dsl_user_op
def cvt_e4m3_to_f32_via_f16(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert single E4M3 byte to f32 via hw-native cvt.rn.f16x2.e4m3x2."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .b32 h2;
                .reg .b16 lo, hi;
                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2, fp8_pair;
                mov.b32 {lo, hi}, h2;
                cvt.f32.f16 $0, lo;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# =============================================================================
# FP4 (E2M1) Decode Intrinsics
# =============================================================================


@dsl_user_op
def fp4_decode_4bytes(
    packed_u32: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Decode 4 FP4 bytes (packed u32) into 4 x f16x2 pairs."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {byte0, byte1, byte2, byte3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, byte0;
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            cvt.rn.f16x2.e2m1x2 $2, byte2;
            cvt.rn.f16x2.e2m1x2 $3, byte3;
        }
        """,
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def fp4_decode_2(byte_val: Uint32, *, loc=None, ip=None) -> Uint32:
    """Decode 1 FP4 byte into 1 f16x2 pair."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(byte_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b8 b0;
                cvt.u8.u32 b0, $1;
                cvt.rn.f16x2.e2m1x2 $0, b0;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_fp32x2_to_e2m1x2(v0: Float32, v1: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert 2 x f32 to 1 x FP4 (E2M1) byte."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 b;
                cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
                cvt.u32.u8 $0, b;
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def quant_dequant_2(
    v0: Float32, v1: Float32, sf_f32: Float32, eff_scale: Float32
) -> Tuple[Float32, Float32]:
    """Quantize-dequantize pair roundtrip through FP4."""
    inv_scale = Float32(1.0) / eff_scale
    fp4_byte = cvt_fp32x2_to_e2m1x2(v0 * inv_scale, v1 * inv_scale)
    h2 = fp4_decode_2(fp4_byte)
    f0, f1 = f16x2_to_f32x2(h2)
    return f0 * sf_f32, f1 * sf_f32


# =============================================================================
# Half2 FMA Dot Product Intrinsics
# =============================================================================


@dsl_user_op
def hfma2_4(
    w0: Uint32,
    x0: Uint32,
    w1: Uint32,
    x1: Uint32,
    w2: Uint32,
    x2: Uint32,
    w3: Uint32,
    x3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """4-element half2 FMA dot product returning (lo, hi) f32."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(x0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(x1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(x2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
            Uint32(x3).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .f16x2 acc;
            .reg .b16 lo, hi;
            mov.b32 acc, 0;
            fma.rn.f16x2 acc, $2, $3, acc;
            fma.rn.f16x2 acc, $4, $5, acc;
            fma.rn.f16x2 acc, $6, $7, acc;
            fma.rn.f16x2 acc, $8, $9, acc;
            mov.b32 {lo, hi}, acc;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def hfma2_4_sum(
    w0: Uint32,
    x0: Uint32,
    w1: Uint32,
    x1: Uint32,
    w2: Uint32,
    x2: Uint32,
    w3: Uint32,
    x3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """4-element half2 FMA dot product returning lane sum."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(w0).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(w1).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(w2).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(w3).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, $1, $2, acc;
                fma.rn.f16x2 acc, $3, $4, acc;
                fma.rn.f16x2 acc, $5, $6, acc;
                fma.rn.f16x2 acc, $7, $8, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def hfma2_8(
    w0: Uint32,
    x0: Uint32,
    w1: Uint32,
    x1: Uint32,
    w2: Uint32,
    x2: Uint32,
    w3: Uint32,
    x3: Uint32,
    w4: Uint32,
    x4: Uint32,
    w5: Uint32,
    x5: Uint32,
    w6: Uint32,
    x6: Uint32,
    w7: Uint32,
    x7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """8-element half2 FMA dot product returning (lo, hi) f32."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(x0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(x1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(x2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
            Uint32(x3).ir_value(loc=loc, ip=ip),
            Uint32(w4).ir_value(loc=loc, ip=ip),
            Uint32(x4).ir_value(loc=loc, ip=ip),
            Uint32(w5).ir_value(loc=loc, ip=ip),
            Uint32(x5).ir_value(loc=loc, ip=ip),
            Uint32(w6).ir_value(loc=loc, ip=ip),
            Uint32(x6).ir_value(loc=loc, ip=ip),
            Uint32(w7).ir_value(loc=loc, ip=ip),
            Uint32(x7).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .f16x2 acc;
            .reg .b16 lo, hi;
            mov.b32 acc, 0;
            fma.rn.f16x2 acc, $2, $3, acc;
            fma.rn.f16x2 acc, $4, $5, acc;
            fma.rn.f16x2 acc, $6, $7, acc;
            fma.rn.f16x2 acc, $8, $9, acc;
            fma.rn.f16x2 acc, $10, $11, acc;
            fma.rn.f16x2 acc, $12, $13, acc;
            fma.rn.f16x2 acc, $14, $15, acc;
            fma.rn.f16x2 acc, $16, $17, acc;
            mov.b32 {lo, hi}, acc;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def hfma2_8_sum(
    w0: Uint32,
    x0: Uint32,
    w1: Uint32,
    x1: Uint32,
    w2: Uint32,
    x2: Uint32,
    w3: Uint32,
    x3: Uint32,
    w4: Uint32,
    x4: Uint32,
    w5: Uint32,
    x5: Uint32,
    w6: Uint32,
    x6: Uint32,
    w7: Uint32,
    x7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """8-element half2 FMA dot product returning lane sum."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(w0).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(w1).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(w2).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(w3).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
                Uint32(w4).ir_value(loc=loc, ip=ip),
                Uint32(x4).ir_value(loc=loc, ip=ip),
                Uint32(w5).ir_value(loc=loc, ip=ip),
                Uint32(x5).ir_value(loc=loc, ip=ip),
                Uint32(w6).ir_value(loc=loc, ip=ip),
                Uint32(x6).ir_value(loc=loc, ip=ip),
                Uint32(w7).ir_value(loc=loc, ip=ip),
                Uint32(x7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, $1, $2, acc;
                fma.rn.f16x2 acc, $3, $4, acc;
                fma.rn.f16x2 acc, $5, $6, acc;
                fma.rn.f16x2 acc, $7, $8, acc;
                fma.rn.f16x2 acc, $9, $10, acc;
                fma.rn.f16x2 acc, $11, $12, acc;
                fma.rn.f16x2 acc, $13, $14, acc;
                fma.rn.f16x2 acc, $15, $16, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# =============================================================================
# FP4 Dot Product Intrinsics
# =============================================================================


@dsl_user_op
def fp4_dot4_sum(
    u_packed: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """FP4 dot product: decode 4 FP4 bytes and dot with 4 x f16x2 inputs."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(u_packed).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 b0, b1, b2, b3;
                .reg .b32 h0, h1, h2, h3;
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 {b0, b1, b2, b3}, $1;
                cvt.rn.f16x2.e2m1x2 h0, b0;
                cvt.rn.f16x2.e2m1x2 h1, b1;
                cvt.rn.f16x2.e2m1x2 h2, b2;
                cvt.rn.f16x2.e2m1x2 h3, b3;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, h0, $2, acc;
                fma.rn.f16x2 acc, h1, $3, acc;
                fma.rn.f16x2 acc, h2, $4, acc;
                fma.rn.f16x2 acc, h3, $5, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp4_dot8_sum(
    u_a: Uint32,
    u_b: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    x4: Uint32,
    x5: Uint32,
    x6: Uint32,
    x7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """FP4 dot product: decode 8 FP4 bytes (2 x u32) and dot with 8 x f16x2 inputs."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(u_a).ir_value(loc=loc, ip=ip),
                Uint32(u_b).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
                Uint32(x4).ir_value(loc=loc, ip=ip),
                Uint32(x5).ir_value(loc=loc, ip=ip),
                Uint32(x6).ir_value(loc=loc, ip=ip),
                Uint32(x7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 a0, a1, a2, a3;
                .reg .b8 b0, b1, b2, b3;
                .reg .b32 h0, h1, h2, h3, h4, h5, h6, h7;
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 {a0, a1, a2, a3}, $1;
                mov.b32 {b0, b1, b2, b3}, $2;
                cvt.rn.f16x2.e2m1x2 h0, a0;
                cvt.rn.f16x2.e2m1x2 h1, a1;
                cvt.rn.f16x2.e2m1x2 h2, a2;
                cvt.rn.f16x2.e2m1x2 h3, a3;
                cvt.rn.f16x2.e2m1x2 h4, b0;
                cvt.rn.f16x2.e2m1x2 h5, b1;
                cvt.rn.f16x2.e2m1x2 h6, b2;
                cvt.rn.f16x2.e2m1x2 h7, b3;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, h0, $3, acc;
                fma.rn.f16x2 acc, h1, $4, acc;
                fma.rn.f16x2 acc, h2, $5, acc;
                fma.rn.f16x2 acc, h3, $6, acc;
                fma.rn.f16x2 acc, h4, $7, acc;
                fma.rn.f16x2 acc, h5, $8, acc;
                fma.rn.f16x2 acc, h6, $9, acc;
                fma.rn.f16x2 acc, h7, $10, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _f16x2_dot_sum_f32acc(a_h2: Uint32, b_h2: Uint32) -> Float32:
    a0, a1 = f16x2_to_f32x2(a_h2)
    b0, b1 = f16x2_to_f32x2(b_h2)
    return a0 * b0 + a1 * b1


@cute.jit
def fp4_dot4_sum_f32acc(
    u_packed: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
) -> Float32:
    """FP4 dot product using f32 accumulation after FP4 decode."""
    h0, h1, h2, h3 = fp4_decode_4bytes(u_packed)
    return (
        _f16x2_dot_sum_f32acc(h0, x0)
        + _f16x2_dot_sum_f32acc(h1, x1)
        + _f16x2_dot_sum_f32acc(h2, x2)
        + _f16x2_dot_sum_f32acc(h3, x3)
    )


@cute.jit
def fp4_dot8_sum_f32acc(
    u_a: Uint32,
    u_b: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    x4: Uint32,
    x5: Uint32,
    x6: Uint32,
    x7: Uint32,
) -> Float32:
    """FP4 dot product using f32 accumulation after FP4 decode."""
    h0, h1, h2, h3 = fp4_decode_4bytes(u_a)
    h4, h5, h6, h7 = fp4_decode_4bytes(u_b)
    return (
        _f16x2_dot_sum_f32acc(h0, x0)
        + _f16x2_dot_sum_f32acc(h1, x1)
        + _f16x2_dot_sum_f32acc(h2, x2)
        + _f16x2_dot_sum_f32acc(h3, x3)
        + _f16x2_dot_sum_f32acc(h4, x4)
        + _f16x2_dot_sum_f32acc(h5, x5)
        + _f16x2_dot_sum_f32acc(h6, x6)
        + _f16x2_dot_sum_f32acc(h7, x7)
    )


# =============================================================================
# Pack Intrinsics
# =============================================================================


@dsl_user_op
def pack_f32x2_to_f16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack two f32 values into one f16x2 u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.f16.f32 lo, $1;
                cvt.rn.f16.f32 hi, $2;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# =============================================================================
# UE8M0 Intrinsics (for MXFP4)
# =============================================================================


@dsl_user_op
def cvt_f32_to_ue8m0(max_val: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Convert float32 max value to UE8M0 scale factor.

    UE8M0 is unsigned 8-bit exponent-only format:
    - value = 2^(ue8m0 - 127)
    - ue8m0 = ceil(log2(max_val)) + 127

    Uses lg2.approx.f32 for fast log2 approximation.
    Uses cvt.rpi (round towards positive infinity, i.e., ceiling).
    Returns value clamped to [0, 255].
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(max_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg, p_ovf;
                .reg .f32 log2_val;
                .reg .s32 exp_int, result;

                // Check for zero/negative
                setp.le.f32 p_zero, $1, 0f00000000;

                // Compute ceil(log2(max_val)) using cvt.rpi (round towards +inf)
                lg2.approx.f32 log2_val, $1;
                cvt.rpi.s32.f32 exp_int, log2_val;

                // Add bias and clamp to [0, 255]
                add.s32 result, exp_int, 127;
                setp.lt.s32 p_neg, result, 0;
                setp.gt.s32 p_ovf, result, 255;
                selp.s32 result, 0, result, p_neg;
                selp.s32 result, 255, result, p_ovf;
                selp.s32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ue8m0_to_output_scale(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """
    Convert UE8M0 to output_scale for MXFP4 quantization.

    UE8M0 value = 2^(ue8m0 - 127)
    Returns 1 / 2^(ue8m0 - 127) = 2^(127 - ue8m0)
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .s32 neg_exp;
                .reg .f32 neg_exp_f, result;

                // Check for zero
                setp.eq.u32 p_zero, $1, 0;

                // Compute 2^(127 - ue8m0) = 1 / 2^(ue8m0 - 127)
                sub.s32 neg_exp, 127, $1;
                cvt.rn.f32.s32 neg_exp_f, neg_exp;
                ex2.approx.f32 result, neg_exp_f;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# E2M1 Conversion
# =============================================================================


@dsl_user_op
def cvt_e2m1x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert eight float32 values to eight E2M1 (4-bit) values packed into uint32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
                Float32(v4).ir_value(loc=loc, ip=ip),
                Float32(v5).ir_value(loc=loc, ip=ip),
                Float32(v6).ir_value(loc=loc, ip=ip),
                Float32(v7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Warp, Block, and Cluster Reduction Utilities
# =============================================================================


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def block_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Block reduction across multiple warps using shared memory."""
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def cluster_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
) -> Float32:
    """Cluster reduction across multiple CTAs using mbarrier."""
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    rows_per_block = reduction_buffer.shape[0]
    warps_per_row = reduction_buffer.shape[1][0]

    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    # Warp 0 sets up mbarrier transaction count
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            expected_bytes = num_warps * cluster_n * 4
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes)

    # Each lane < cluster_n writes to a different CTA's shared memory
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )

    # Wait for all cluster writes
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)

    # Reduce across all values
    num_total = warps_per_row * cluster_n
    num_iter = cute.ceil_div(num_total, 32)

    block_reduce_val = init_val
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * 32
        if idx < num_total:
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])

    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce(
    x: cute.TensorSSA,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
):
    """Row reduction with optional cluster support."""
    local_val = x.reduce(op, init_val=init_val, reduction_profile=0)

    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax,
    }[op]
    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, warp_op, width=warp_width)

    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        if cutlass.const_expr(cluster_n == 1):
            return block_reduce(warp_val, warp_op, reduction_buffer, init_val)
        else:
            return cluster_reduce(
                warp_val, warp_op, reduction_buffer, mbar_ptr, cluster_n, init_val
            )
    else:
        return warp_val


# =============================================================================
# Predicate Utility
# =============================================================================


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][1], limit
            )
    return tXpX


# =============================================================================
# Helper Functions for SF Block Processing (block_size=16)
# =============================================================================


@cute.jit
def load_8_half2(
    mX: cute.Tensor, mW: cute.Tensor, row_offset: Int32, col_offset: Int32, H: int
):
    """Load 16 elements (8 half2 pairs) of X and W from global memory.

    Returns:
        x_h2: rmem_tensor of shape (8,) containing X as half2
        w_h2: rmem_tensor of shape (8,) containing W as half2
    """
    x_h2 = cute.make_rmem_tensor((8,), Uint32)
    w_h2 = cute.make_rmem_tensor((8,), Uint32)

    # Load X (2 x 128-bit loads = 16 elements)
    x_ptr0 = get_ptr_as_int64(mX, row_offset * H + col_offset)
    x_ptr1 = get_ptr_as_int64(mX, row_offset * H + col_offset + Int32(8))
    x_h2[0], x_h2[1], x_h2[2], x_h2[3] = ld_global_v4_u32(x_ptr0)
    x_h2[4], x_h2[5], x_h2[6], x_h2[7] = ld_global_v4_u32(x_ptr1)

    # Load W (2 x 128-bit loads = 16 elements)
    w_ptr0 = get_ptr_as_int64(mW, col_offset)
    w_ptr1 = get_ptr_as_int64(mW, col_offset + Int32(8))
    w_h2[0], w_h2[1], w_h2[2], w_h2[3] = ld_global_v4_u32(w_ptr0)
    w_h2[4], w_h2[5], w_h2[6], w_h2[7] = ld_global_v4_u32(w_ptr1)

    return x_h2, w_h2


@cute.jit
def half2_mul_8(x_h2: cute.Tensor, w_h2: cute.Tensor) -> cute.Tensor:
    """Multiply 8 half2 pairs element-wise."""
    xw_h2 = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        xw_h2[i] = half2_mul(x_h2[i], w_h2[i])
    return xw_h2


@cute.jit
def bfloat2_mul_8(x_h2: cute.Tensor, w_h2: cute.Tensor) -> cute.Tensor:
    """Multiply 8 bfloat2 pairs element-wise."""
    xw_h2 = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        xw_h2[i] = bfloat2_mul(x_h2[i], w_h2[i])
    return xw_h2


@cute.jit
def half2_max_abs_8(xw_h2: cute.Tensor) -> Uint32:
    """Compute max absolute value across 8 half2 values using tree reduction."""
    # Compute abs for all 8 values
    abs_h2 = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        abs_h2[i] = habs2(xw_h2[i])

    # Tree reduction: 8 -> 4 -> 2 -> 1
    max_01 = hmax2(abs_h2[0], abs_h2[1])
    max_23 = hmax2(abs_h2[2], abs_h2[3])
    max_45 = hmax2(abs_h2[4], abs_h2[5])
    max_67 = hmax2(abs_h2[6], abs_h2[7])
    max_0123 = hmax2(max_01, max_23)
    max_4567 = hmax2(max_45, max_67)
    return hmax2(max_0123, max_4567)


@cute.jit
def bfloat2_max_abs_8(xw_h2: cute.Tensor) -> Uint32:
    """Compute max absolute value across 8 bfloat2 values using tree reduction."""
    # Compute abs for all 8 values
    abs_h2 = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        abs_h2[i] = bfloat2_habs2(xw_h2[i])

    # Tree reduction: 8 -> 4 -> 2 -> 1
    max_01 = bfloat2_hmax2(abs_h2[0], abs_h2[1])
    max_23 = bfloat2_hmax2(abs_h2[2], abs_h2[3])
    max_45 = bfloat2_hmax2(abs_h2[4], abs_h2[5])
    max_67 = bfloat2_hmax2(abs_h2[6], abs_h2[7])
    max_0123 = bfloat2_hmax2(max_01, max_23)
    max_4567 = bfloat2_hmax2(max_45, max_67)
    return bfloat2_hmax2(max_0123, max_4567)


@cute.jit
def half2_to_float16(xw_h2: cute.Tensor, scale: Float32) -> cute.Tensor:
    """Convert 8 half2 to 16 float32 with scaling."""
    y_f32 = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(8):
        y_f32[i * 2], y_f32[i * 2 + 1] = half2_to_float2_scaled(xw_h2[i], scale)
    return y_f32


@cute.jit
def bfloat2_to_float16(xw_h2: cute.Tensor, scale: Float32) -> cute.Tensor:
    """Convert 8 bfloat2 to 16 float32 with scaling."""
    y_f32 = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(8):
        y_f32[i * 2], y_f32[i * 2 + 1] = bfloat2_to_float2_scaled(xw_h2[i], scale)
    return y_f32


@cute.jit
def quantize_and_pack_16(y_f32: cute.Tensor, value_scale: Float32) -> Uint64:
    """Quantize 16 float32 values to FP4 and pack into uint64."""
    t025 = value_scale * Float32(0.25)
    t075 = value_scale * Float32(0.75)
    t125 = value_scale * Float32(1.25)
    t175 = value_scale * Float32(1.75)
    t250 = value_scale * Float32(2.5)
    t350 = value_scale * Float32(3.5)
    t500 = value_scale * Float32(5.0)
    packed = Uint64(0)
    for i in cutlass.range_constexpr(16):
        q = y_f32[i]
        mag = fabs_f32(q)
        nibble = Uint8(0)
        if mag > t025 and mag < t075:
            nibble = Uint8(1)
        elif mag >= t075 and mag <= t125:
            nibble = Uint8(2)
        elif mag > t125 and mag < t175:
            nibble = Uint8(3)
        elif mag >= t175 and mag <= t250:
            nibble = Uint8(4)
        elif mag > t250 and mag < t350:
            nibble = Uint8(5)
        elif mag >= t350 and mag <= t500:
            nibble = Uint8(6)
        elif mag > t500:
            nibble = Uint8(7)
        if nibble != Uint8(0) and q < Float32(0.0):
            nibble = nibble | Uint8(0x8)
        packed = packed | (Uint64(nibble) << Uint64(i * 4))
    return packed


@cute.jit
def quantize_and_pack_16_fast(y_f32: cute.Tensor, inv_scale: Float32) -> Uint64:
    """Fast approximate FP4 quantize/pack for 16 float32 values."""
    q = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        q[i] = y_f32[i] * inv_scale

    packed_lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
    packed_hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def quantize_block_fp4(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Quantize 16 float32 values to packed FP4 + e4m3 scale byte.

    Given 16 values and their pre-computed max_abs, derives the NVFP4 block
    scale, quantizes to FP4, and packs into a uint64.  Returns
    (packed_fp4_u64, scale_byte).  The caller handles storage writes.
    """
    scale_float = max_abs * global_scale_val / Float32(FLOAT4_E2M1_MAX)
    scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))
    scale_u32 = cvt_f32_to_e4m3(scale_float)
    scale_byte = cutlass.Uint8(scale_u32 & Uint32(0xFF))
    quantized_scale = fp8_e4m3_to_f32(scale_u32)
    packed64 = Uint64(0)
    if quantized_scale != Float32(0.0) and global_scale_val != Float32(0.0):
        packed64 = quantize_and_pack_16(values, quantized_scale / global_scale_val)
    return packed64, scale_byte


@cute.jit
def quantize_block_fp4_fast(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Fast approximate FP4 block quantization using reciprocal/vector path."""
    scale_u32 = Uint32(0)
    scale_byte = cutlass.Uint8(0)
    packed64 = Uint64(0)
    if global_scale_val != Float32(0.0):
        fp4_max_rcp = rcp_approx_ftz(Float32(FLOAT4_E2M1_MAX))
        scale_float = global_scale_val * (max_abs * fp4_max_rcp)
        scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))
        scale_u32 = cvt_f32_to_e4m3(scale_float)
        scale_byte = cutlass.Uint8(scale_u32 & Uint32(0xFF))
        inv_quantized_scale = fp8_e4m3_to_f32_and_rcp(scale_u32)
        if inv_quantized_scale != Float32(0.0):
            packed64 = quantize_and_pack_16_fast(
                values, inv_quantized_scale * global_scale_val
            )
    return packed64, scale_byte


@cute.jit
def max_abs_16(values: cute.Tensor) -> Float32:
    """Compute the maximum absolute value of 16 float32 values."""
    result = fabs_f32(values[0])
    for i in cutlass.range_constexpr(1, 16):
        result = fmax_f32(result, fabs_f32(values[i]))
    return result


@cute.jit
def silu_mul_16(
    gate: cute.Tensor,
    up: cute.Tensor,
) -> cute.Tensor:
    """Fused SiLU(gate) * up for 16 float32 element pairs.

    Used in the CompactMoEKernel epilogue to fuse the activation function
    between GEMM1 and GEMM2, avoiding the gmem round-trip.
    """
    out = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        g = gate[i]
        sigmoid_g = cute.arch.rcp_approx(
            Float32(1.0) + cute.math.exp(-g, fastmath=False)
        )
        out[i] = g * sigmoid_g * up[i]
    return out


@cute.jit
def silu_mul_quantize_block_fp4(
    gate: cute.Tensor,
    up: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Fused SiLU(gate)*up + FP4 quantize for 16 element pairs.

    Combines silu_mul_16, max_abs_16, and quantize_block_fp4 into a single
    helper for the CompactMoEKernel epilogue.
    """
    activated = silu_mul_16(gate, up)
    block_max = max_abs_16(activated)
    return quantize_block_fp4(activated, block_max, global_scale_val)


@cute.jit
def relu2_16(x: cute.Tensor) -> cute.Tensor:
    """Compute ReLU^2 for 16 float32 values."""
    out = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        v = fmax_f32(x[i], Float32(0.0))
        out[i] = v * v
    return out


@cute.jit
def relu2_quantize_block_fp4(
    x: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Fuse ReLU^2 and FP4 quantization for 16 float32 values."""
    activated = relu2_16(x)
    block_max = max_abs_16(activated)
    return quantize_block_fp4(activated, block_max, global_scale_val)


# =============================================================================
# Helper Functions for Float32 SF Block Processing
# =============================================================================


@cute.jit
def load_f32_16_from_smem(
    sH: cute.Tensor, row_idx: Int32, col_offset: Int32
) -> cute.Tensor:
    """Load 16 Float32 values from shared memory."""
    h_f32 = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        h_f32[i] = Float32(sH[row_idx, col_offset + i])
    return h_f32


@cute.jit
def compute_y_and_max_abs_f32(
    h_f32: cute.Tensor, w_f32: cute.Tensor, rstd: Float32
) -> Tuple[cute.Tensor, Float32]:
    """Compute y = h * rstd * w and max_abs for 16 Float32 values."""
    y_f32 = cute.make_rmem_tensor((16,), Float32)

    # Compute y and track max_abs
    y_f32[0] = h_f32[0] * rstd * w_f32[0]
    max_abs = fabs_f32(y_f32[0])

    for i in cutlass.range_constexpr(1, 16):
        y_f32[i] = h_f32[i] * rstd * w_f32[i]
        max_abs = fmax_f32(max_abs, fabs_f32(y_f32[i]))

    return y_f32, max_abs
