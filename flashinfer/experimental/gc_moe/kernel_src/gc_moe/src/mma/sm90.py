"""Hopper register-A WGMMA, MXFP4 decoding and tile layouts."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


@cute.jit
def canonical_offset(row: Uint32, k: Uint32) -> Uint32:
    return (row * 128 + k) ^ ((row % 8) * 16)


@cute.jit
def smem_descriptor(ptr: cute.Pointer) -> Uint64:
    return ((Uint64(ptr.toint()) >> 4) & Uint64(0x3fff)) | (Uint64(64) << 32) | (Uint64(1) << 62)


@dsl_user_op
def wgmma_rs_k128(frag: tuple, desc_b: Uint64, n: cutlass.Constexpr,
                  *, loc=None, ip=None):
    count = n // 2
    outputs = ', '.join('$' + str(i) for i in range(count))
    body = '{ .reg .b64 b; wgmma.fence.sync.aligned; '
    body += f'mov.b64 b, ${count + 16}; '
    for group in range(4):
        if group:
            body += 'add.u64 b, b, 2; '
        operands = ', '.join('$' + str(count + group * 4 + i) for i in range(4))
        body += (f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 '
                 f'{{{outputs}}}, {{{operands}}}, b, {int(group != 0)}, 1, 1; ')
    body += 'wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0; }'
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(' + ','.join(['f32'] * count) + ')>'),
        [x.ir_value() for x in frag] + [desc_b.ir_value()], body,
        ','.join(['=f'] * count + ['r'] * 16 + ['l', '~{memory}']),
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(count))


@dsl_user_op
def decode_fragment_nib4(words: cute.Pointer, scales: cute.Pointer, *, loc=None, ip=None):
    """Four K32 register-A fragments; one 4-bit packed exponent word.

    Weight exponents are stored two per byte (even index in the low nibble).
    One aligned 32-bit LDS carries all eight exponents, and each group/row
    nibble is extracted with a compile-time ``bfe`` position.
    """
    body = '{ .reg .b32 sf, o, lo, hi, mag, tmp, sign; '
    body += '.reg .b32 p<8>; '
    body += 'ld.shared.u32 sf, [$16]; '
    body += 'ld.shared.v4.u32 {p0,p1,p2,p3}, [$17]; '
    body += 'ld.shared.v4.u32 {p4,p5,p6,p7}, [$17+512]; '
    for g in range(4):
        for r in range(2):
            body += f'bfe.u32 o, sf, {4 * g + 16 * r}, 4; '
            body += 'mad.lo.u32 lo, o, 0x08080800, 0x0c080000; '
            body += 'mad.lo.u32 hi, o, 0x08080808, 0x1c181410; '
            body += f'and.b32 mag, p{g * 2 + r}, 0x77777777; '
            body += f'prmt.b32 tmp, lo, hi, mag; shl.b32 sign, p{g * 2 + r}, 4; '
            body += f'lop3.b32 ${g * 4 + r}, sign, 0x80808080, tmp, 0xea; '
            body += 'shr.u32 mag, mag, 16; prmt.b32 tmp, lo, hi, mag; '
            body += f'lop3.b32 ${g * 4 + r + 2}, p{g * 2 + r}, 0x80808080, tmp, 0xea; '
    body += '}'
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(' + ','.join(['i32'] * 16) + ')>'),
        [scales.toint().ir_value(), words.toint().ir_value()], body,
        ','.join(['=r'] * 16 + ['r','r','~{memory}']),
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Uint32(llvm.extractvalue(T.i32(), result, [i])) for i in range(16))


@dsl_user_op
def wgmma_rs_issue(frag: tuple, desc_b: Uint64, n: cutlass.Constexpr,
                    *, loc=None, ip=None):
    count = n // 2
    outputs = ', '.join('$' + str(i) for i in range(count))
    body = '{ .reg .b64 b; wgmma.fence.sync.aligned; '
    body += f'mov.b64 b, ${count + 16}; '
    for g in range(4):
        if g:
            body += 'add.u64 b, b, 2; '
        inputs = ', '.join('$' + str(count + g * 4 + i) for i in range(4))
        body += (f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 '
                 f'{{{outputs}}}, {{{inputs}}}, b, {int(g != 0)}, 1, 1; ')
    body += 'wgmma.commit_group.sync.aligned; }'
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(' + ','.join(['f32'] * count) + ')>'),
        [x.ir_value() for x in frag] + [desc_b.ir_value()], body,
        ','.join(['=f'] * count + ['r'] * 16 + ['l','~{memory}']),
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(count))


@dsl_user_op
def wgmma_rs_wait(raw: tuple, *, loc=None, ip=None):
    count = len(raw)
    body = 'wgmma.wait_group.sync.aligned 0; '
    for i in range(count):
        body += f'mov.f32 ${i}, ${count + i}; '
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(' + ','.join(['f32'] * count) + ')>'),
        [x.ir_value() for x in raw], body,
        ','.join(['=f'] * count + ['f'] * count + ['~{memory}']),
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(count))


@dsl_user_op
def wgmma_rs_k32_pair(fragment0: tuple, fragment1: tuple, descriptor: Uint64,
                       n: cutlass.Constexpr,
                       *, loc=None, ip=None):
    """Two independent K32 partial banks, one commit/wait; no K accumulation.

    Callers promote bank 0 then bank 1 to retain each output's K32 FMA order.
    The second descriptor addresses the next 32 activation bytes (two units).
    """
    count = n
    out0 = ','.join('$' + str(i) for i in range(n // 2))
    out1 = ','.join('$' + str(n // 2 + i) for i in range(n // 2))
    in0 = ','.join('$' + str(count + i) for i in range(4))
    in1 = ','.join('$' + str(count + 4 + i) for i in range(4))
    body = '{ .reg .b64 b; '
    body += (f'mov.b64 b, ${count + 8}; wgmma.fence.sync.aligned; '
             f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 '
             f'{{{out0}}}, {{{in0}}}, b, 0, 1, 1; '
             'add.u64 b, b, 2; '
             f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 '
             f'{{{out1}}}, {{{in1}}}, b, 0, 1, 1; '
             'wgmma.commit_group.sync.aligned; ')
    body += 'wgmma.wait_group.sync.aligned 0; '
    body += '}'
    types = ['f32'] * count
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(' + ','.join(types) + ')>'),
        [*(x.ir_value() for x in (*fragment0, *fragment1)), descriptor.ir_value()], body,
        ','.join(['=f'] * count + ['r'] * 8 + ['l', '~{memory}']),
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    parts = tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(count))
    return parts


@dsl_user_op
def wgmma_rs_k32_pair_issue(fragment0: tuple, fragment1: tuple, descriptor: Uint64,
                       n: cutlass.Constexpr,
                       *, loc=None, ip=None):
    """Issue independent K32 banks; callers must wait before consuming either.

    Keep A source registers unchanged until the paired wait has completed.
    """
    count = n
    out0 = ','.join('$' + str(i) for i in range(n // 2))
    out1 = ','.join('$' + str(n // 2 + i) for i in range(n // 2))
    in0 = ','.join('$' + str(count + i) for i in range(4))
    in1 = ','.join('$' + str(count + 4 + i) for i in range(4))
    body = '{ .reg .b64 b; '
    body += (f'mov.b64 b, ${count + 8}; wgmma.fence.sync.aligned; '
             f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 '
             f'{{{out0}}}, {{{in0}}}, b, 0, 1, 1; '
             'add.u64 b, b, 2; '
             f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 '
             f'{{{out1}}}, {{{in1}}}, b, 0, 1, 1; '
             'wgmma.commit_group.sync.aligned; ')
    body += '}'
    types = ['f32'] * count
    result = llvm.inline_asm(
        ir.Type.parse('!llvm.struct<(' + ','.join(types) + ')>'),
        [*(x.ir_value() for x in (*fragment0, *fragment1)), descriptor.ir_value()], body,
        ','.join(['=f'] * count + ['r'] * 8 + ['l', '~{memory}']),
        has_side_effects=True, asm_dialect=llvm.AsmDialect.AD_ATT)
    parts = tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(count))
    return parts
