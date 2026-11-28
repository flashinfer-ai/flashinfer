# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Block maxima and exact candidate publication for two-level indexing."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def packed_max(a, b, *, loc=None, ip=None):
    result = llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [
            cutlass.Uint32(a).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(b).ir_value(loc=loc, ip=ip),
        ],
        "max.bf16x2 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def block_pair_store(
    ptr,
    stride,
    row,
    key,
    end0,
    end1,
    score0,
    score1,
    rows,
    dtype=cutlass.BFloat16,
    *,
    loc=None,
    ip=None,
):
    args = [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
    args += [
        cutlass.Int32(x).ir_value(loc=loc, ip=ip)
        for x in (stride, row, key, end0, end1)
    ]
    args += [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (score0, score1)]
    args += [cutlass.Int32(rows).ir_value(loc=loc, ip=ip)]
    asm = """{
    .reg .b32 a,b,v,t,i,last,r,offset,bytes;
    .reg .b16 h0,h1;
    .reg .b64 addr0,addr1;
    .reg .pred p,p0,p1;
    mov.b32 a,$7; mov.b32 b,$8;
    setp.ge.s32 p,$4,$5; selp.b32 a,0xff800000,a,p;
    setp.ge.s32 p,$4,$6; selp.b32 b,0xff800000,b,p;
    prmt.b32 v,a,b,0x7632;
    shfl.sync.bfly.b32 t,v,1,31,0xffffffff; max.bf16x2 v,v,t;
    shfl.sync.bfly.b32 t,v,2,31,0xffffffff; max.bf16x2 v,v,t;
    shfl.sync.bfly.b32 t,v,4,31,0xffffffff; max.bf16x2 v,v,t;
    and.b32 a,v,65535; shr.u32 b,v,16;
    shr.u32 i,$4,3;
    sub.s32 last,$5,1; shr.u32 last,last,3;
    setp.eq.u32 p,i,last; selp.b32 a,32640,a,p;
    sub.s32 last,$6,1; shr.u32 last,last,3;
    setp.eq.u32 p,i,last; selp.b32 b,32640,b,p;
    cvt.u16.u32 h0,a; cvt.u16.u32 h1,b;
    and.b32 t,$4,7; setp.eq.u32 p,t,0;
    setp.lt.and.s32 p0,$3,$9,p;
    add.u32 r,$3,1; setp.lt.and.s32 p1,r,$9,p;
    shl.b32 bytes,$2,1;
    mad.wide.u32 addr0,$3,bytes,$1;
    shl.b32 offset,i,1; cvt.u64.u32 addr1,offset; add.u64 addr0,addr0,addr1;
    cvt.u64.u32 addr1,bytes; add.u64 addr1,addr0,addr1;
    @p0 st.global.b16 [addr0],h0;
    @p1 st.global.b16 [addr1],h1;
    mov.u32 $0,0;
    }"""
    if dtype == cutlass.Float16:
        asm = asm.replace("prmt.b32 v,a,b,0x7632;", "cvt.rn.f16x2.f32 v,b,a;")
        asm = asm.replace("max.bf16x2", "max.f16x2").replace("32640", "31744")
    return cutlass.Int32(
        llvm.inline_asm(
            cutlass.Int32.mlir_type,
            args,
            asm,
            "=r,l,r,r,r,r,r,f,f,r,~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def write_block_pair(
    scores,
    row,
    key,
    end0,
    end1,
    value0,
    value1,
    already_rounded: cutlass.Constexpr = False,
    paired_fp32: cutlass.Constexpr = False,
):
    if cutlass.const_expr(paired_fp32):
        _block_pair_store_fp32(
            scores.iterator,
            scores.stride[0],
            row,
            key,
            end0,
            end1,
            value0,
            value1,
            scores.shape[0],
            False,
        )
    else:
        if cutlass.const_expr(not already_rounded):
            value0 = cutlass.Float32(scores.element_type(value0))
            value1 = cutlass.Float32(scores.element_type(value1))
        if cutlass.const_expr(
            scores.element_type in (cutlass.BFloat16, cutlass.Float16)
        ):
            block_pair_store(
                scores.iterator,
                scores.stride[0],
                row,
                key,
                end0,
                end1,
                value0,
                value1,
                scores.shape[0],
                dtype=scores.element_type,
            )
        else:
            write_block_max(scores, row, key, end0, value0)
            write_block_max(scores, row + 1, key, end1, value1)


@cute.jit
def write_block_max(scores, row, key, end, value):
    value = cutlass.Float32(value)
    if key >= end:
        value = cutlass.Float32(float("-inf"))
    for bit in cutlass.range_constexpr(3):
        value = max(value, cute.arch.shuffle_sync_bfly(value, 1 << bit))
    if key % 8 == 0 and key < end and row < scores.shape[0]:
        if key // 8 == (end - 1) // 8:
            value = cutlass.Float32(float("inf"))
        scores[row, key // 8] = scores.element_type(value)


@dsl_user_op
def block_pair_store_interior(
    ptr, stride, row, key, score0, score1, dtype=cutlass.BFloat16, *, loc=None, ip=None
):
    args = [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
    args += [cutlass.Int32(x).ir_value(loc=loc, ip=ip) for x in (stride, row, key)]
    args += [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (score0, score1)]
    asm = """{
    .reg .b32 a,b,v,t,i,offset,bytes;
    .reg .b16 h0,h1;
    .reg .b64 addr0,addr1;
    .reg .pred p;
    mov.b32 a,$5; mov.b32 b,$6;
    prmt.b32 v,a,b,0x7632;
    shfl.sync.bfly.b32 t,v,1,31,0xffffffff; max.bf16x2 v,v,t;
    shfl.sync.bfly.b32 t,v,2,31,0xffffffff; max.bf16x2 v,v,t;
    shfl.sync.bfly.b32 t,v,4,31,0xffffffff; max.bf16x2 v,v,t;
    mov.b32 {h0,h1},v;
    shr.u32 i,$4,3;
    and.b32 t,$4,7; setp.eq.u32 p,t,0;
    shl.b32 bytes,$2,1;
    mad.wide.u32 addr0,$3,bytes,$1;
    shl.b32 offset,i,1; cvt.u64.u32 addr1,offset; add.u64 addr0,addr0,addr1;
    cvt.u64.u32 addr1,bytes; add.u64 addr1,addr0,addr1;
    @p st.global.b16 [addr0],h0;
    @p st.global.b16 [addr1],h1;
    mov.u32 $0,0;
    }"""
    if dtype == cutlass.Float16:
        asm = asm.replace("prmt.b32 v,a,b,0x7632;", "cvt.rn.f16x2.f32 v,b,a;")
        asm = asm.replace("max.bf16x2", "max.f16x2")
    return cutlass.Int32(
        llvm.inline_asm(
            cutlass.Int32.mlir_type,
            args,
            asm,
            "=r,l,r,r,r,f,f,~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def write_block_pair_interior(
    scores,
    row,
    key,
    value0,
    value1,
    already_rounded: cutlass.Constexpr = False,
    paired_fp32: cutlass.Constexpr = False,
):
    """Store two valid rows; the caller excludes the latest block."""
    if cutlass.const_expr(paired_fp32):
        _block_pair_store_fp32(
            scores.iterator,
            scores.stride[0],
            row,
            key,
            0,
            0,
            value0,
            value1,
            scores.shape[0],
            True,
        )
    else:
        if cutlass.const_expr(not already_rounded):
            value0 = cutlass.Float32(scores.element_type(value0))
            value1 = cutlass.Float32(scores.element_type(value1))
        if cutlass.const_expr(
            scores.element_type in (cutlass.BFloat16, cutlass.Float16)
        ):
            block_pair_store_interior(
                scores.iterator,
                scores.stride[0],
                row,
                key,
                value0,
                value1,
                dtype=scores.element_type,
            )
        else:
            # Interior keys precede the live latest block, hence the capacity tail.
            write_block_max(scores, row, key, scores.shape[1] * 8, value0)
            write_block_max(scores, row + 1, key, scores.shape[1] * 8, value1)


@dsl_user_op
def _block_pair_store_fp32(
    ptr, stride, row, key, end0, end1, a, b, rows, interior, *, loc=None, ip=None
):
    args = [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)]
    args += [
        cutlass.Int32(x).ir_value(loc=loc, ip=ip)
        for x in (stride, row, key, end0, end1)
    ]
    args += [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b)]
    args += [cutlass.Int32(rows).ir_value(loc=loc, ip=ip)]
    code = [
        "{ .reg .f32 a,b,t; .reg .b32 i,last,r,bytes,offset,lane; .reg .b64 p0,p1,tmp; .reg .pred p,p0v,p1v; mov.f32 a,$7; mov.f32 b,$8;"
    ]
    if not interior:
        code += [
            "setp.ge.s32 p,$4,$5; @p mov.b32 a,0xff800000; setp.ge.s32 p,$4,$6; @p mov.b32 b,0xff800000;"
        ]
    for bit in (1, 2, 4):
        code += [
            f"shfl.sync.bfly.b32 t,a,{bit},31,0xffffffff; max.f32 a,a,t;",
            f"shfl.sync.bfly.b32 t,b,{bit},31,0xffffffff; max.f32 b,b,t;",
        ]
    code += ["shr.u32 i,$4,3;"]
    if not interior:
        code += [
            "sub.s32 last,$5,1; shr.u32 last,last,3; setp.eq.u32 p,i,last; @p mov.b32 a,0x7f800000; sub.s32 last,$6,1; shr.u32 last,last,3; setp.eq.u32 p,i,last; @p mov.b32 b,0x7f800000;"
        ]
    code += [
        "and.b32 lane,$4,7; setp.eq.u32 p,lane,0; setp.lt.and.s32 p0v,$3,$9,p; add.u32 r,$3,1; setp.lt.and.s32 p1v,r,$9,p;"
    ]
    if not interior:
        code += ["setp.lt.and.s32 p0v,$4,$5,p0v; setp.lt.and.s32 p1v,$4,$6,p1v;"]
    code += [
        "shl.b32 bytes,$2,2; mad.wide.u32 p0,$3,bytes,$1; shl.b32 offset,i,2; cvt.u64.u32 tmp,offset; add.u64 p0,p0,tmp; cvt.u64.u32 tmp,bytes; add.u64 p1,p0,tmp; @p0v st.global.f32 [p0],a; @p1v st.global.f32 [p1],b; mov.u32 $0,0; }"
    ]
    return cutlass.Int32(
        llvm.inline_asm(
            cutlass.Int32.mlir_type,
            args,
            " ".join(code),
            "=r,l,r,r,r,r,r,f,f,r,~{memory}",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


from .candidate_publication import compile_block_selector as compile_block_selector
