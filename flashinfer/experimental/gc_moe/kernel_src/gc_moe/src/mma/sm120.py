"""SM120 warp-level block-scaled mma.sync helpers for the swap-AB role.

The target instruction is PTX:

    mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
      .m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0

Swap-AB keeps the e2m1 weights in the register A operand and the e4m3
activations in B, so every TensorMap descriptor stays tile-invariant.
Adapted from the reference SM120 MegaMoE kernel (BSD-3, NVIDIA).
"""

from __future__ import annotations

from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass import Float32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir, llvm
from cutlass.cute.nvgpu.warp.mma import _pack_shape


MMA_M = 16
MMA_N = 8
MMA_K = 32
MXFP8_BLOCK = 32
FP4_SHIFT_BITS = 2
SWAP_AB_INTERLEAVE = 8
CTA_TOKEN_TILE = 64
INTERMEDIATE_ALIGNMENT = 2 * MXFP8_BLOCK


@dsl_user_op
def load_unpack_fp4_quad(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load four FP4 words as a single register-fragment operation."""
    instructions = ['{ .reg .b16 h; .reg .b32 packed, high, expanded, address; '
                    'mov.b32 address, $4; ']
    for register, offset in enumerate((0, 512, 8, 520)):
        instructions.append(
            f'ld.shared.u16 h, [address+{offset}]; cvt.u32.u16 packed, h; '
            'shr.u32 high, packed, 4; '
            'prmt.b32 expanded, packed, high, 0x5140; '
            'and.b32 expanded, expanded, 0x0f0f0f0f; '
            f'shl.b32 ${register}, expanded, 2; ')
    instructions.append('}')
    value = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [pointer.toint(loc=loc, ip=ip).ir_value()], ''.join(instructions),
        '=r,=r,=r,=r,r,~{memory}', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(Uint32(llvm.extractvalue(T.i32(), value, [i], loc=loc, ip=ip))
                 for i in range(4))


@cute.jit
def load_packed_fp4_fragment(fragment: cute.Tensor, source: cute.Tensor,
                             k_inner: cutlass.Constexpr) -> None:
    """Read one M16/K32 warp fragment from a compact (M64,K64-byte) tile."""
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    row = (tid // 32 % 4) * 16 + lane // 4
    column = k_inner * 16 + (lane % 4) * 2
    words = cute.recast_tensor(fragment, Uint32)
    # Raw PTX pointer conversion does not apply CuTe's pointer swizzle.
    # SW64 is Swizzle<2,4,3>: address bits 7:8 xor into bits 4:5.
    offset = row * 64 + column
    offset = offset ^ ((offset >> 3) & 0x30)
    values = load_unpack_fp4_quad(source.iterator + offset)
    for register in cutlass.range_constexpr(4):
        words[register] = values[register]


@cute.jit
def shift_fp4_fragment_for_mxf8f6f4(fragment: cute.Tensor) -> None:
    """Move unpacked E2M1 nibbles into the bit position consumed by QMMA."""

    fragment_i8 = cute.recast_tensor(fragment, cutlass.Int8)
    for i in cutlass.range_constexpr(cute.size(fragment_i8)):
        fragment_i8[i] = cutlass.Int8(fragment_i8[i] << FP4_SHIFT_BITS)


@dsl_user_op
def _arch_mma_m16n8k32_mxfp8(
    acc: cute.Tensor,
    a_reg: cute.Tensor,
    b_reg: cute.Tensor,
    sfa_scalar,
    sfb_scalar,
    *,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    thread_id_a: int = 0,
    thread_id_b: int = 0,
    loc=None,
    ip=None,
) -> None:
    """Issue one QMMA ``m16n8k32`` MXFP8 block-scaled instruction.

    Per lane: A = 4 x b32, B = 2 x b32, C/D = 4 x f32.
    """

    a_i32 = a_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    b_i32 = b_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    sfa_i8 = sfa_scalar.bitcast(cutlass.Int8, loc=loc, ip=ip)
    sfb_i8 = sfb_scalar.bitcast(cutlass.Int8, loc=loc, ip=ip)

    # Byte selectors are for packed 32-bit scale words, not these scalar bytes.
    shape_attr = _pack_shape((MMA_M, MMA_N, MMA_K), loc=loc, ip=ip).type.attribute
    res = _cute_nvgpu_ir.arch_mma_SM120_block_scaled(
        [acc_dtype.mlir_type] * 4,
        shape_attr,
        MXFP8_BLOCK,
        ir.TypeAttr.get(a_dtype.mlir_type),
        ir.TypeAttr.get(b_dtype.mlir_type),
        ir.TypeAttr.get(sf_dtype.mlir_type),
        [
            a_i32[0].ir_value(loc=loc, ip=ip),
            a_i32[1].ir_value(loc=loc, ip=ip),
            a_i32[2].ir_value(loc=loc, ip=ip),
            a_i32[3].ir_value(loc=loc, ip=ip),
        ],
        [
            b_i32[0].ir_value(loc=loc, ip=ip),
            b_i32[1].ir_value(loc=loc, ip=ip),
        ],
        [
            acc[0].ir_value(loc=loc, ip=ip),
            acc[1].ir_value(loc=loc, ip=ip),
            acc[2].ir_value(loc=loc, ip=ip),
            acc[3].ir_value(loc=loc, ip=ip),
        ],
        sfa_i8.ir_value(loc=loc, ip=ip),
        sfb_i8.ir_value(loc=loc, ip=ip),
        thread_id_a=thread_id_a,
        thread_id_b=thread_id_b,
        loc=loc,
        ip=ip,
    )

    acc[0] = acc_dtype(res[0])
    acc[1] = acc_dtype(res[1])
    acc[2] = acc_dtype(res[2])
    acc[3] = acc_dtype(res[3])


@dsl_user_op
def lds_b32_raw_ordered(addr, *, loc=None, ip=None) -> cutlass.Int32:
    """Load one TMA/cp.async-produced shared-memory u32 by raw address.

    Volatile asm plus the memory clobber keeps the load below the stage's
    mbarrier wait instead of being hoisted or CSE'd across stages.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _arch_mma_m16n8k32_mxfp8_packed_sfb(
    acc: cute.Tensor,
    a_reg: cute.Tensor,
    b_reg: cute.Tensor,
    sfa_packed,
    sfb_packed,
    *,
    byte_id_a: int = 0,
    byte_id_b: int,
    loc=None,
    ip=None,
) -> None:
    """Issue mixed W4A8 QMMA using packed SFA and SFB registers.

    PTX selects one byte each from ``sfa_packed``/``sfb_packed`` with the
    byte-id immediates. This keeps the per-lane K128 scale words intact
    across the four K32 issues and avoids explicit SHF/LOP3 byte
    extraction (and the cvt/prmt-heavy SFA/SFB fragment copies) in the
    persistent K2 loop.
    """

    a_i32 = a_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    b_i32 = b_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    bid_a = cutlass.Int16(byte_id_a).ir_value(loc=loc, ip=ip)
    tid_a = cutlass.Int16(0).ir_value(loc=loc, ip=ip)
    bid_b = cutlass.Int16(byte_id_b).ir_value(loc=loc, ip=ip)
    tid_b = cutlass.Int16(0).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a_i32[0]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[1]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[2]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[3]).ir_value(loc=loc, ip=ip),
            Uint32(b_i32[0]).ir_value(loc=loc, ip=ip),
            Uint32(b_i32[1]).ir_value(loc=loc, ip=ip),
            Uint32(sfa_packed).ir_value(loc=loc, ip=ip),
            bid_a,
            tid_a,
            Uint32(sfb_packed).ir_value(loc=loc, ip=ip),
            bid_b,
            tid_b,
            Float32(acc[0]).ir_value(loc=loc, ip=ip),
            Float32(acc[1]).ir_value(loc=loc, ip=ip),
            Float32(acc[2]).ir_value(loc=loc, ip=ip),
            Float32(acc[3]).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0
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
    acc[0] = Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    acc[1] = Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    acc[2] = Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    acc[3] = Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))


@cute.jit
def issue_m64n8k32_mxfp8_packed_sfb(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a_frag,
    b_frag,
    sfa_packed,
    sfb_packed,
    *,
    n_group: int,
    active_n_groups: int,
    k_inner: int,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
) -> None:
    """Issue one QMMA using byte-id selection from packed SFA and SFB."""

    if cutlass.const_expr(n_group >= active_n_groups):
        return
    _arch_mma_m16n8k32_mxfp8_packed_sfb(
        acc,
        a_frag[(None, 0, k_inner)],
        b_frag[(None, n_group, k_inner)],
        sfa_packed,
        sfb_packed,
        byte_id_a=k_inner,
        byte_id_b=k_inner,
    )


def make_swapab_m64n8k128_tiled_mma(
    *,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
) -> cute.TiledMma:
    """Create the warp-MMA tiler used by the SM120 swap A/B mainloop.

    Four compute warps cover the four M16 atoms in one ``M64 x N8`` MMA
    slice.

    The CuTe MMA op is used here only to construct tiling/fragment layouts;
    the routed mainloop issues the instruction with packed scale words.
    """

    if a_dtype == b_dtype:
        op = warp_mma.MmaMXF8Op(a_dtype, acc_dtype, sf_dtype)
    else:
        op = warp_mma.MmaMXF8F6F4Op(
            a_dtype, b_dtype, acc_dtype, sf_dtype
        )
    permutation_mnk = sm120_utils.get_permutation_mnk(
        (64, 64, 128),
        MXFP8_BLOCK,
        True,
    )
    return cute.make_tiled_mma(
        op,
        cute.make_layout((4, 1, 1)),
        permutation_mnk=permutation_mnk,
    )


def make_sm120_ldmatrix_atom(
    operand_dtype: Type[cutlass.Numeric],
    *,
    transpose: bool,
    mixed_mode: bool = False,
) -> cute.CopyAtom:
    """Build the SM120 block-scaled SMEM->RMEM ldmatrix copy atom.

    Mixed FP4 operands use ``ldsm.b4x16_p64``.  That instruction does not
    support transpose, so the packed FP4 weight tensor must be K-major.
    """

    if mixed_mode and operand_dtype.width == 4:
        if transpose:
            raise ValueError("mixed FP4 ldmatrix requires K-major storage")
        return cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x16x8bOp(
                transpose=False,
                num_matrices=4,
                unpack_bits=4,
            ),
            cutlass.Int8,
        )

    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=4,
        ),
        operand_dtype,
    )
