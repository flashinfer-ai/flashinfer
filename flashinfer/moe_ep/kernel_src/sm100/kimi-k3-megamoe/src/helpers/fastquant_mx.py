# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming BF16 -> MXFP8 activation quantize, bit-identical to
``torch.ops.trtllm.mxfp8_quantize(x, is_sf_swizzled_layout=False, alignment=32)``: e4m3 data [m, k] row-major and one
UE8M0 scale byte per 32 elements, LINEAR [m, k / 32].

TRT's ``quantize_with_block_size<FP16_TO_MXFP8>`` has the same structure as its NVFP4 sibling (see fastquant.py): one
16-byte vector per thread per row, a dependent shuffle / SF / convert / store chain, and only one load in flight per
thread. On the heavy rank of a variable-token step (16384 tokens) it takes ~54 us where the HBM floor is ~25 us, and
the whole step waits for the heavy rank's pre-kernel work.

This kernel is fastquant's layout with a 32-element block: ONE lane owns ONE whole UE8M0 block (two 256-bit loads, no
cross-lane shuffle, the SF math once per block, one 32-byte e4m3 store + one SF byte). UNROLL blocks per lane are
loaded before any math. Same arithmetic as TRT ``cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32, 1>``:
    localMax = abs.bf16x2 / max.bf16x2 fold over the 16 words; vecMax = float((lo > hi) ? lo : hi)
    SFValue  = vecMax * rcp.approx.ftz(448.0f)                       mul.rn.f32
    e8m0     = cvt.rp.satfinite.ue8m0x2.f32(SFValue)                 (__nv_cvt_float_to_e8m0(.., SATFINITE, RoundPosInf))
    osc      = vecMax != 0 ? (e8m0 == 0 ? 1.0 : 2^(127 - e8m0)) : 0  (exp2f_rcp; exact power of two, built bitwise)
    q_i      = cvt.rn.satfinite.e4m3x2.f32(f32(e_{2j+1}) * osc, f32(e_{2j}) * osc)
The only operand whose value depends on how TRT's exp2f is compiled is 2^-127 (e8m0 == 254, reachable only from an
inf element); ``SUBNORMAL_RCP`` selects the measured behaviour (see s22 README: verified against TRT on GB300).
"""
from ..helpers import codegen_options as _kimi_options
import os

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op

ELTS_PER_BLOCK = 32  # one UE8M0 scale block = one lane's work item (64 B in, 32 B e4m3 + 1 B SF out)
# 2^(127 - 254) = 2^-127 is subnormal in fp32: exp2f with flush-to-zero returns 0, without it 0x00200000.
SUBNORMAL_RCP = 1  # 1 = keep 2^-127, 0 = flush to 0


def enabled() -> bool:
    return _kimi_options.environ.get("MEGAMOE_FAST_QUANT_MX", "1").strip() != "0"


def config():
    """(UNROLL blocks per lane per step, THREADS per CTA, CTAS per SM)."""
    return (
        int(_kimi_options.environ.get("MEGAMOE_FASTQUANT_MX_UNROLL", "2")),
        int(_kimi_options.environ.get("MEGAMOE_FASTQUANT_MX_THREADS", "256")),
        int(_kimi_options.environ.get("MEGAMOE_FASTQUANT_MX_CTAS_PER_SM", "2")),  # s22 sweep: 2 >= 4 at every size (16k: 33.4 vs 35.5 us)
    )


@dsl_user_op
def ldg_nc_v8_b32(address: Int64, *, loc=None, ip=None):
    """32-byte read-only streaming load (one full sector)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.L2::256B.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=r,=r,=r,=r,=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip)) for index in range(8))


def _quant_asm() -> str:
    """$0..$7: eight e4m3x4 output words (elements 0..31); $8: SF byte; $9..$24: sixteen bf16x2 input words."""
    w0 = 9
    lines = [
        "{",
        ".reg .b32 m, t, ob;",
        ".reg .b16 lo, hi, mx, sfp, eb;",
        ".reg .pred p, nz, ez;",
        ".reg .f32 vmax, sfv, osc, r448;",
        ".reg .f32 a<32>;",
        ".reg .b16 q<16>;",
        f"abs.bf16x2 m, ${w0};",
    ]
    for w in range(1, 16):
        lines.append(f"abs.bf16x2 t, ${w0 + w};")
        lines.append("max.bf16x2 m, m, t;")
    lines += [
        "mov.b32 {lo, hi}, m;",
        "setp.gt.bf16 p, lo, hi;",
        "selp.b16 mx, lo, hi, p;",
        "cvt.f32.bf16 vmax, mx;",
        "rcp.approx.ftz.f32 r448, 0f43E00000;",  # 448.0
        "mul.rn.f32 sfv, vmax, r448;",
        "cvt.rp.satfinite.ue8m0x2.f32 sfp, 0f00000000, sfv;",  # low byte = ue8m0(sfv)
        "and.b16 eb, sfp, 0x00FF;",
        "cvt.u32.u16 $8, eb;",
        # osc = 2^(127 - e) for e in [1, 253] = bits (254 - e) << 23; e == 254 -> 2^-127 (subnormal) or 0 (FTZ);
        # e == 0 -> 1.0 (exp2f_rcp's special case); e == 255 (NaN SF) -> anything finite: every element is NaN then.
        "cvt.u32.u16 t, eb;",
        "sub.u32 ob, 254, t;",
        "shl.b32 ob, ob, 23;",
        "setp.eq.u32 ez, t, 254;",
        f"selp.b32 ob, {0x00200000 if SUBNORMAL_RCP else 0}, ob, ez;",
        "setp.eq.u32 ez, t, 0;",
        "selp.b32 ob, 0x3F800000, ob, ez;",
        "setp.eq.u32 ez, t, 255;",
        "selp.b32 ob, 0x00100000, ob, ez;",  # 2^-128 (only reached by an all-NaN block; value is irrelevant)
        "mov.b32 osc, ob;",
        "setp.neu.f32 nz, vmax, 0f00000000;",
        "selp.f32 osc, osc, 0f00000000, nz;",
    ]
    for w in range(16):
        lines.append(f"mov.b32 {{lo, hi}}, ${w0 + w};")
        lines.append(f"cvt.f32.bf16 a{2 * w}, lo;")
        lines.append(f"cvt.f32.bf16 a{2 * w + 1}, hi;")
    for i in range(32):
        lines.append(f"mul.rn.f32 a{i}, a{i}, osc;")
    for j in range(16):
        lines.append(f"cvt.rn.satfinite.e4m3x2.f32 q{j}, a{2 * j + 1}, a{2 * j};")
    for o in range(8):
        lines.append(f"mov.b32 ${o}, {{q{2 * o}, q{2 * o + 1}}};")
    lines.append("}")
    return "\n".join(lines)


@dsl_user_op
def quantize_block32(words, *, loc=None, ip=None):
    """32 BF16 values (sixteen bf16x2 words, one lane) -> (8 e4m3x4 words, SF byte in bits 0-7)."""
    operands = [Int32(w).ir_value(loc=loc, ip=ip) for w in words]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 9),
        operands,
        _quant_asm(),
        ",".join(["=r"] * 9 + ["r"] * 16),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip)) for index in range(9))


@dsl_user_op
def stg_v8_b32_b8_if(out_address: Int64, words, sf_address: Int64, sf: Int32, predicate: Int32, *, loc=None, ip=None):
    """Predicated 32-byte e4m3 store + 1-byte SF store (one lane's block)."""
    operands = [Int64(out_address).ir_value(loc=loc, ip=ip)]
    operands += [Int32(w).ir_value(loc=loc, ip=ip) for w in words]
    operands += [Int64(sf_address).ir_value(loc=loc, ip=ip), Int32(sf).ir_value(loc=loc, ip=ip),
                 Int32(predicate).ir_value(loc=loc, ip=ip)]
    llvm.inline_asm(
        None,
        operands,
        "{\n.reg .pred p;\n.reg .b16 v;\ncvt.u16.u32 v, $10;\nsetp.ne.s32 p, $11, 0;\n"
        "@p st.global.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};\n@p st.global.b8 [$9], v;\n}",
        "l,r,r,r,r,r,r,r,r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.kernel
def _fastquant_mx_kernel(
    x_address: Int64,
    out_address: Int64,
    sf_address: Int64,
    block_count: Int32,
    vec_unroll: cutlass.Constexpr[int],
    warps_per_cta: cutlass.Constexpr[int],
    pdl_trigger: cutlass.Constexpr[bool] = False,
):
    # Launched with programmatic stream serialization like TRT's quantize: wait for the producer of x first.
    cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(pdl_trigger):
        # [b10 Q] see fastquant._fastquant_kernel
        cute.arch.griddepcontrol_launch_dependents()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    lane = tidx % Int32(32)
    warp = tidx // Int32(32)
    total_warps = gdim * Int32(warps_per_cta)
    blocks_per_step = 32 * vec_unroll
    step_count = (block_count + Int32(blocks_per_step - 1)) // Int32(blocks_per_step)
    last_block = block_count - Int32(1)

    step = bidx * Int32(warps_per_cta) + warp
    while step < step_count:
        first = step * Int32(blocks_per_step) + lane
        words = []
        for k in cutlass.range_constexpr(vec_unroll):
            # Out-of-range lanes reload the last block (in bounds) and store nothing.
            block = cutlass.min(first + Int32(k * 32), last_block)
            base = x_address + Int64(block) * Int64(64)
            words.append(ldg_nc_v8_b32(base) + ldg_nc_v8_b32(base + Int64(32)))
        for k in cutlass.range_constexpr(vec_unroll):
            block = first + Int32(k * 32)
            ok = Int32(0)
            if block < block_count:
                ok = Int32(1)
            res = quantize_block32(words[k])
            stg_v8_b32_b8_if(out_address + Int64(block) * Int64(32), res[:8], sf_address + Int64(block), res[8], ok)
        step = step + total_warps


@cute.jit
def _fastquant_mx_launch(
    x_address: Int64,
    out_address: Int64,
    sf_address: Int64,
    block_count: Int32,
    grid: Int32,
    vec_unroll: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    min_blocks: cutlass.Constexpr[int],
    stream,
    pdl_trigger: cutlass.Constexpr[bool] = False,
):
    _fastquant_mx_kernel(x_address, out_address, sf_address, block_count, vec_unroll, threads // 32, pdl_trigger).launch(
        grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream, min_blocks_per_mp=min_blocks, use_pdl=True
    )
