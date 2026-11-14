# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v35-fastquant: MEGAMOE_FAST_QUANT=1 -- a streaming BF16 -> NVFP4 activation quantize, bit-identical to
``torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)`` (LINEAR SF layout, per-tensor global scale).

Why TRT's kernel is slow (``quantize_with_block_size``, 448 threads x 4 CTAs / SM, grid min(rows, 4 x SMs)): every
thread owns one 16-byte vector per row and loops over rows; the next row's load cannot be hoisted above the current
row's stores, so each thread has exactly ONE 16 B load in flight, then a dependent shuffle / SF / convert / store
chain. That is <= 1792 x 16 B = 28 KB in flight per SM, only while every warp sits in its load (the compute + store
phase is not overlapped within a warp), and 8192 rows / 608 CTAs = 13.5 rounds (last round half empty). On GB300
(~8 TB/s, ~1 us loaded latency -> ~55 KB per SM needed by Little's law) that lands at 2.4 TB/s.

This kernel:
- Flat over 16-element SF blocks (rows x hidden / 16, the FP4 / SF outputs are contiguous in LINEAR layout): ONE lane
  owns ONE whole block: a single 256-bit load (``LDG.256``, a full 32 B sector, L2 256 B prefetch hint, L1
  no-allocate) -> no cross-lane shuffle, the SF math once per block instead of once per lane, one 8-byte FP4 store
  and one SF byte. A warp-instruction moves 1 KB in / 256 B FP4 / 32 B SF, all contiguous (full sectors).
- UNROLL independent blocks per lane per step (blocks lane + 32 k of a contiguous 32 x UNROLL block super-chunk),
  all loads issued before any math: UNROLL x 32 B per lane in flight.
- Persistent full-machine grid: CTAS_PER_SM x SMs CTAs of THREADS threads; warps stride over super-chunks.
  Default 4 x 256 threads x UNROLL 4 = 32 warps / SM x 4 KB = 128 KB per SM in flight at the load point.
- Same arithmetic as TRT (see ``quantize_block16``); bit-identical incl. NaN / inf / denormals / -0.
"""
from ..helpers import codegen_options as _kimi_options
import os

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, dsl_user_op

ELTS_PER_BLOCK = 16  # one NVFP4 scale block = one lane's work item (32 B in, 8 B FP4 + 1 B SF out)


def enabled() -> bool:
    # default ON (MEGAMOE_FAST_QUANT=0 disables); the backend resolves the same switch.
    return _kimi_options.environ.get("MEGAMOE_FAST_QUANT", "").strip() != "0"


def config():
    """(UNROLL blocks per lane per step, THREADS per CTA, CTAS per SM, FMUL2) from the environment."""
    return (
        int(_kimi_options.environ.get("MEGAMOE_FASTQUANT_UNROLL", "4")),
        int(_kimi_options.environ.get("MEGAMOE_FASTQUANT_THREADS", "256")),
        int(_kimi_options.environ.get("MEGAMOE_FASTQUANT_CTAS_PER_SM", "4")),
        int(_kimi_options.environ.get("MEGAMOE_FASTQUANT_FMUL2", "1")),
    )


# ---------------------------------------------------------------------------------------------------------------------
# PTX primitives
# ---------------------------------------------------------------------------------------------------------------------


@dsl_user_op
def rcp_approx_ftz(value: Float32, *, loc=None, ip=None) -> Float32:
    """``rcp.approx.ftz.f32`` (TRT ``reciprocal_approximate_ftz``)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_nc_v8_b32(address: Int64, *, loc=None, ip=None):
    """32-byte read-only streaming load (one full sector; LDG.E.256 on sm_100+)."""
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


def _elementwise_ptx(fmul2: bool) -> str:
    """x_i = f32(bf16 e_i) * osc for the 16 elements (registers a0..a15; words $3..$10)."""
    lines = []
    for w in range(8):
        lines.append(f"mov.b32 {{lo, hi}}, ${3 + w};")
        lines.append(f"cvt.f32.bf16 a{2 * w}, lo;")
        lines.append(f"cvt.f32.bf16 a{2 * w + 1}, hi;")
    if fmul2:
        # mul.rn.f32x2 = two independent IEEE mul.rn (same bits as TRT's scalar x * outputScale; FMUL2 on sm_100+).
        lines.append("mov.b64 s2, {osc, osc};")
        for w in range(8):
            lines.append(f"mov.b64 e, {{a{2 * w}, a{2 * w + 1}}};")
            lines.append("mul.rn.f32x2 e, e, s2;")
            lines.append(f"mov.b64 {{a{2 * w}, a{2 * w + 1}}}, e;")
    else:
        for i in range(16):
            lines.append(f"mul.rn.f32 a{i}, a{i}, osc;")
    for j in range(8):
        lines.append(f"cvt.rn.satfinite.e2m1x2.f32 q{j}, a{2 * j + 1}, a{2 * j};")
    lines.append("mov.b32 $0, {q0, q1, q2, q3};")
    lines.append("mov.b32 $1, {q4, q5, q6, q7};")
    return "\n".join(lines) + "\n"


@dsl_user_op
def quantize_block16(words, global_scale, rcp_six, rcp_global_scale, fmul2: bool = True, *, loc=None, ip=None):
    """Sixteen BF16 values (eight bf16x2 words, one lane) -> (FP4 word lo, FP4 word hi, e4m3 SF byte in bits 0-7).

    TRT ``cvt_warp_fp16_to_fp4<__nv_bfloat16, 16, false>`` splits the block over two lanes (8 values each):
        localMax = __habs2(e0); localMax = __hmax2(localMax, __habs2(e_i))      (per lane, its 4 words)
        localMax = __hmax2(__shfl_xor_sync(mask, localMax, 1), localMax)       (pair)
        vecMax = float((localMax.x > localMax.y) ? localMax.x : localMax.y)
    Here one lane holds all 8 words and folds them with the same abs.bf16x2 / max.bf16x2. max.bf16x2 (no .NaN) is
    IEEE maxNum per half: commutative and associative on the |values| (NaN operands are dropped, an all-NaN half
    gives the canonical NaN either way, and after abs there are no +-0 ties), so the two halves of the bf16x2 result,
    and hence the scalar select, are bit-identical to TRT's tree. The rest is v34's ``quantize_vec8`` verbatim:
        SFValue = SFScaleVal * (vecMax * rcp_approx_ftz(6.0f))                  two mul.rn.f32 (no fast-math)
        fp8 = __nv_fp8_e4m3(SFValue); SFValue = float(fp8)                      cvt.rn.satfinite.e4m3x2 / exact back
        outputScale = vecMax != 0 ? rcp(SFValue * rcp(SFScaleVal)) : 0         mul.rn, rcp.approx.ftz, setp.neu
        x_i = float(e_i) * outputScale; e2m1x2(x_{2j+1}, x_{2j})                 cvt.f32.bf16 exact, mul.rn, cvt.rn.satfinite
    FP4 bytes 0-3 = elements 0-7 (TRT's even lane word), bytes 4-7 = elements 8-15 (odd lane word).
    """
    operands = [Int32(w).ir_value(loc=loc, ip=ip) for w in words]
    operands += [
        Float32(global_scale).ir_value(loc=loc, ip=ip),
        Float32(rcp_six).ir_value(loc=loc, ip=ip),
        Float32(rcp_global_scale).ir_value(loc=loc, ip=ip),
    ]
    asm = (
        "{\n"
        ".reg .b32 m, t, o;\n"
        ".reg .b16 lo, hi, mx, sfp;\n"
        ".reg .pred p, nz;\n"
        ".reg .f32 vmax, sfv, sff, osc;\n"
        ".reg .f32 a<16>;\n"
        ".reg .b64 e, s2;\n"
        ".reg .b8 q<8>;\n"
        "abs.bf16x2 m, $3;\n"
        + "".join(f"abs.bf16x2 t, ${3 + w};\nmax.bf16x2 m, m, t;\n" for w in range(1, 8))
        + "mov.b32 {lo, hi}, m;\n"
        "setp.gt.bf16 p, lo, hi;\n"
        "selp.b16 mx, lo, hi, p;\n"
        "cvt.f32.bf16 vmax, mx;\n"
        "mul.rn.f32 sfv, vmax, $12;\n"
        "mul.rn.f32 sfv, $11, sfv;\n"
        "cvt.rn.satfinite.e4m3x2.f32 sfp, 0f00000000, sfv;\n"
        "cvt.u32.u16 $2, sfp;\n"
        "cvt.rn.f16x2.e4m3x2 o, sfp;\n"
        "mov.b32 {lo, hi}, o;\n"
        "cvt.f32.f16 sff, lo;\n"
        "mul.rn.f32 osc, sff, $13;\n"
        "rcp.approx.ftz.f32 osc, osc;\n"
        "setp.neu.f32 nz, vmax, 0f00000000;\n"
        "selp.f32 osc, osc, 0f00000000, nz;\n"
        + _elementwise_ptx(fmul2)
        + "}"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32()]),
        operands,
        asm,
        "=r,=r,=r,r,r,r,r,r,r,r,r,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip)) for index in range(3))


@dsl_user_op
def stg_v2_b32_b8_if(fp4_address: Int64, word_lo: Int32, word_hi: Int32, sf_address: Int64, sf: Int32,
                     predicate: Int32, *, loc=None, ip=None) -> None:
    """Predicated 8-byte FP4 store + 1-byte SF store (one lane's block)."""
    llvm.inline_asm(
        None,
        [Int64(fp4_address).ir_value(loc=loc, ip=ip), Int32(word_lo).ir_value(loc=loc, ip=ip),
         Int32(word_hi).ir_value(loc=loc, ip=ip), Int64(sf_address).ir_value(loc=loc, ip=ip),
         Int32(sf).ir_value(loc=loc, ip=ip), Int32(predicate).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p;\n.reg .b16 v;\ncvt.u16.u32 v, $4;\nsetp.ne.s32 p, $5, 0;\n"
        "@p st.global.v2.b32 [$0], {$1, $2};\n@p st.global.b8 [$3], v;\n}",
        "l,r,r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------------------------------------


@cute.kernel
def _fastquant_kernel(
    x_address: Int64,
    fp4_address: Int64,
    sf_address: Int64,
    global_scale_address: Int64,
    block_count: Int32,
    vec_unroll: cutlass.Constexpr[int],
    warps_per_cta: cutlass.Constexpr[int],
    fmul2: cutlass.Constexpr[bool],
    pdl_trigger: cutlass.Constexpr[bool] = False,
):
    # ``pdl_trigger`` is always False: it only keeps the kernel symbol (which encodes Constexpr values) and therefore
    # existing AOT / compile caches unchanged after the early-trigger experiment was removed.
    # Launched with programmatic stream serialization (like TRT's fp4_quantize under PDL): the grid may become resident
    # while the producer of x (or of the global scale) is still running; wait for its completion + memory flush before
    # the first read. Without a PDL-triggering predecessor this returns once the previous kernel has finished.
    cute.arch.griddepcontrol_wait()
    if cutlass.const_expr(pdl_trigger):
        # [b10 Q] let the router (PDL dependent) start on free SMs now; it reads only the top-k ids here (complete:
        # the wait above returned) and waits for this grid before it lets any peer pull the rows written below.
        cute.arch.griddepcontrol_launch_dependents()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    lane = tidx % Int32(32)
    warp = tidx // Int32(32)
    total_warps = gdim * Int32(warps_per_cta)
    blocks_per_step = 32 * vec_unroll
    step_count = (block_count + Int32(blocks_per_step - 1)) // Int32(blocks_per_step)

    global_scale = ldg_f32(global_scale_address)
    rcp_six = rcp_approx_ftz(Float32(6.0))
    rcp_global_scale = rcp_approx_ftz(global_scale)
    last_block = block_count - Int32(1)

    step = bidx * Int32(warps_per_cta) + warp
    while step < step_count:
        first = step * Int32(blocks_per_step) + lane
        words = []
        for k in cutlass.range_constexpr(vec_unroll):
            # Out-of-range lanes reload the last block (in bounds) and store nothing.
            block = cutlass.min(first + Int32(k * 32), last_block)
            words.append(ldg_nc_v8_b32(x_address + Int64(block) * Int64(32)))
        for k in cutlass.range_constexpr(vec_unroll):
            block = first + Int32(k * 32)
            ok = Int32(0)
            if block < block_count:
                ok = Int32(1)
            lo, hi, sf = quantize_block16(words[k], global_scale, rcp_six, rcp_global_scale, fmul2)
            stg_v2_b32_b8_if(fp4_address + Int64(block) * Int64(8), lo, hi, sf_address + Int64(block), sf, ok)
        step = step + total_warps


@cute.jit
def _fastquant_launch(
    x_address: Int64,
    fp4_address: Int64,
    sf_address: Int64,
    global_scale_address: Int64,
    block_count: Int32,
    grid: Int32,
    vec_unroll: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    fmul2: cutlass.Constexpr[bool],
    min_blocks: cutlass.Constexpr[int],
    stream,
    pdl_trigger: cutlass.Constexpr[bool] = False,
):
    _fastquant_kernel(
        x_address, fp4_address, sf_address, global_scale_address, block_count, vec_unroll, threads // 32, fmul2,
        pdl_trigger,
    ).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream, min_blocks_per_mp=min_blocks, use_pdl=True)


# ---------------------------------------------------------------------------------------------------------------------
# Host entry
# ---------------------------------------------------------------------------------------------------------------------
