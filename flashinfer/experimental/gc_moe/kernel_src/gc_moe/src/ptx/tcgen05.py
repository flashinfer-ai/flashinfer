"""Mixed FP8/FP4 two-CTA TCGEN05 operations needed by the GC worker.

CuTe's public blockscaled MmaMXF8Op currently fixes A and B to the same
FP8 dtype. The reference GC worker uses the mixed mxf8f6f4 PTX encoding;
these user ops preserve its operand order while CuTe owns the surrounding
SMEM/TMEM tensors and schedule. They are not a stand-alone GEMM path.
"""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32, Uint16, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir


@dsl_user_op
def mma_mxf8f6f4_2cta(desc_a: Uint64, desc_b: Uint64,
                      accum_tmem: Uint32, accumulate: Uint32,
                      instr_desc_hi: Uint32, sfa_tmem: Uint32,
                      sfb_tmem: Uint32, *, loc=None, ip=None) -> None:
    """Issue reference 2CTA FP8×FP4 blockscaled MMA into FP32 TMEM."""
    operands = [accum_tmem, desc_a, desc_b, instr_desc_hi,
                accumulate, sfa_tmem, sfb_tmem]
    llvm.inline_asm(
        None, [x.ir_value(loc=loc, ip=ip) for x in operands],
        '{ .reg .pred p; '
        'setp.ne.b32 p, $4, 0; '
        'tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale '
        '[$0], $1, $2, $3, [$5], [$6], p; }',
        'r,l,l,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def mma_mxf8f6f4_k128_2cta(desc_a: Uint64, desc_b: Uint64,
                          accum_tmem: Uint32, first_accumulate: Uint32,
                          instr0: Uint32, instr1: Uint32,
                          instr2: Uint32, instr3: Uint32,
                          sfa_tmem: Uint32, sfb_tmem: Uint32,
                          *, loc=None, ip=None) -> None:
    """Same four ordered K32 MMAs; reuse two local descriptor registers.

    Each original K128 stage descriptor starts on a 1024B boundary. Its
    encoded low-14-bit start is 64-unit aligned, so +2/+4/+6 cannot carry
    into reserved/layout bits. No asynchronous memory pointer is mutated.
    """
    operands = [accum_tmem, desc_a, desc_b, instr0, instr1,
                instr2, instr3, first_accumulate, sfa_tmem, sfb_tmem]
    llvm.inline_asm(
        None, [x.ir_value(loc=loc, ip=ip) for x in operands],
        '{ .reg .pred p; .reg .b64 mma_desc_a, mma_desc_b; '
        'mov.b64 mma_desc_a, $1; mov.b64 mma_desc_b, $2; '
        'setp.ne.b32 p, $7, 0; '
        'tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale '
        '[$0], mma_desc_a, mma_desc_b, $3, [$8], [$9], p; '
        'add.u64 mma_desc_a, mma_desc_a, 2; '
        'add.u64 mma_desc_b, mma_desc_b, 2; '
        'setp.ne.b32 p, 1, 0; '
        'tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale '
        '[$0], mma_desc_a, mma_desc_b, $4, [$8], [$9], p; '
        'add.u64 mma_desc_a, mma_desc_a, 2; '
        'add.u64 mma_desc_b, mma_desc_b, 2; '
        'tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale '
        '[$0], mma_desc_a, mma_desc_b, $5, [$8], [$9], p; '
        'add.u64 mma_desc_a, mma_desc_a, 2; '
        'add.u64 mma_desc_b, mma_desc_b, 2; '
        'tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale '
        '[$0], mma_desc_a, mma_desc_b, $6, [$8], [$9], p; }',
        'r,l,l,r,r,r,r,r,r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def utccp_sf_4x32_2cta(smem_desc: Uint64, tmem_col: Uint32,
                       *, loc=None, ip=None) -> None:
    """Copy 128 packed-int32 UE8M0 elements (512 bytes) to TMEM."""
    llvm.inline_asm(
        None, [tmem_col.ir_value(loc=loc, ip=ip),
               smem_desc.ir_value(loc=loc, ip=ip)],
        'tcgen05.cp.cta_group::2.32x128b.warpx4 [$0], $1;',
        'r,l', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tcgen05_commit_2cta(mbar_ptr: cute.Pointer, cta_mask: Uint16,
                        *, loc=None, ip=None) -> None:
    """Commit to both CTAs' barriers; the GC worker uses multicast mask 3."""
    llvm.inline_asm(
        None, [mbar_ptr.toint(loc=loc, ip=ip).ir_value(),
               cta_mask.ir_value(loc=loc, ip=ip)],
        'tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.'
        'multicast::cluster.b64 [$0], $1;',
        'r,h', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tcgen05_after_thread_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], 'tcgen05.fence::after_thread_sync;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tcgen05_before_thread_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [], 'tcgen05.fence::before_thread_sync;', '',
                    has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)


@dsl_user_op
def tmem_load_16dp256b4(address: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32, Float32, Float32]:
    """Reference GC epilogue TMEM atom: four FP32 registers per lane."""
    values = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(i32,i32,i32,i32)>'),
        [address.ir_value(loc=loc, ip=ip)],
        'tcgen05.ld.sync.aligned.16x256b.x1.b32 '
        '{$0, $1, $2, $3}, [$4];',
        '=r,=r,=r,=r,r', has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.bitcast(T.f32(),
                                    llvm.extractvalue(T.i32(), values, [i])))
                 for i in range(4))


@dsl_user_op
def tmem_free_2cta(address: Uint32, columns: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(None, [address.ir_value(loc=loc, ip=ip),
                           columns.ir_value(loc=loc, ip=ip)],
                    'tcgen05.dealloc.cta_group::2.sync.aligned.b32 $0, $1;',
                    'r,r', has_side_effects=True,
                    asm_dialect=llvm.AsmDialect.AD_ATT)
