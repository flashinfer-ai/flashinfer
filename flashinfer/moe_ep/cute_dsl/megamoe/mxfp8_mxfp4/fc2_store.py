# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native BF16 transpose and coalesced FC2 stores through the route mapper."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


def fc2_exchange_words(epilogue_warps: int) -> int:
    # FC1 needs 32 FP32 maxima per warp. FC2 aliases that storage only after
    # the tile boundary, staging 16 x 32 BF16 values per warp, reused for the
    # second half of each 32-token group.
    return epilogue_warps * 256


@dsl_user_op
def store_native_bf16(
    smem_ptr,
    byte_offset,
    r0,
    r1,
    r2,
    r3,
    swizzle_mask: cutlass.Constexpr[int],
    *,
    loc=None,
    ip=None,
):
    """Transpose four 8x8 BF16 matrices (16 features x 16 tokens).

    Registers pack adjacent tokens. Matrices advance feature by 8, token
    by 8, then both. Lanes 0..7 supply the first matrix's token-row addresses,
    8..15 the second's, etc. Swizzle logical offsets so the exchange base
    need only be 16-byte aligned; its address must not change the permutation.
    """
    offset = byte_offset ^ ((byte_offset >> 3) & swizzle_mask)
    address = smem_ptr.toint(loc=loc, ip=ip) + offset
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)]
        + [cutlass.Uint32(r).ir_value(loc=loc, ip=ip) for r in (r0, r1, r2, r3)],
        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def stage_native_bf16(
    registers,
    staging,
    warp,
    lane,
    half: cutlass.Constexpr[int],
    chunk: cutlass.Constexpr[int],
):
    """Round 16 token columns once and stage one native 16-feature half.

    A 16-row tile is reused between chunks to retain the N128 operand depth;
    the four warps of a warpgroup share its rows.
    """
    bf16 = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    for i in cutlass.range_constexpr(8):
        bf16[i] = registers[chunk * 8 + i].to(cutlass.BFloat16)
    packed = cute.recast_tensor(bf16, cutlass.Uint32)
    token = lane % 8 + (lane // 16) * 8
    feature = (lane // 8 % 2) * 8 + half * 16
    offset = (warp // 4) * 4096 + token * 256 + (warp % 4) * 64 + feature * 2
    mask = 0x70
    store_native_bf16(staging, offset, packed[0], packed[1], packed[2], packed[3], mask)


@cute.jit
def store_staged_vector(
    staging,
    byte_offset,
    swizzle_mask: cutlass.Constexpr[int],
    row_address,
    feature,
):
    """Move an aligned 16-byte vector without changing peer-store ordering."""
    offset = byte_offset ^ ((byte_offset >> 3) & swizzle_mask)
    source = cute.make_tensor(
        cute.make_ptr(
            cutlass.BFloat16,
            staging.toint() + offset,
            cute.AddressSpace.smem,
            assumed_align=16,
        ),
        cute.make_layout(8),
    )
    destination = cute.make_tensor(
        cute.make_ptr(
            cutlass.BFloat16,
            row_address + cutlass.Int64(feature) * 2,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        cute.make_layout(8),
    )
    registers = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    cute.copy(atom, source, registers)
    cute.copy(atom, registers, destination)
