# SPDX-License-Identifier: Apache-2.0
"""PTX / nvvm helpers for the UMMA (tcgen05 / TMEM) ReplaySSM kernels.

Thin ``@dsl_user_op`` wrappers the kernels need and the CuTe DSL does not expose
directly: TMEM allocation and 32x32b loads / stores, kind::f16 ``tcgen05.mma``
with SMEM / TMEM operands and their descriptors, TMA tile and 1-D bulk copies,
predicated vector loads and stores, packed fp16 / bf16 conversions (including
``cvt.rs`` stochastic rounding), Philox4x32 and the LCG dither.
"""

from __future__ import annotations

import cutlass
from cutlass import Boolean, Float32, Int32, Int64, Uint32, Uint64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

# nvvm binding drift in cutlass-dsl 4.6+: Tcgen05GroupKind -> CTAGroupKind,
# tcgen05_commit_arrive -> tcgen05_commit (same members / signature).
_CTA1 = (getattr(nvvm, "Tcgen05GroupKind", None) or nvvm.CTAGroupKind).CTA_1
_tcgen05_commit = getattr(nvvm, "tcgen05_commit_arrive", None) or nvvm.tcgen05_commit
_SHAPE_32X32B = nvvm.Tcgen05LdStShape.SHAPE_32X32B


def _tmem_ptr(addr, *, loc=None, ip=None):
    ptr_ty = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    return llvm.inttoptr(ptr_ty, Int32(addr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


# ----------------------------------------------------------------------------- TMEM
@dsl_user_op
def tmem_alloc(taddr: cute.Pointer, ncols: int, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_alloc(
        taddr.to_llvm_ptr(loc=loc, ip=ip),
        Uint32(ncols).ir_value(loc=loc, ip=ip),
        group=_CTA1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_dealloc(tmem_addr: Int32, ncols: int, *, loc=None, ip=None) -> None:
    nvvm.tcgen05_dealloc(
        _tmem_ptr(tmem_addr, loc=loc, ip=ip),
        Int32(ncols).ir_value(loc=loc, ip=ip),
        group=_CTA1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tmem_ld(row, col, num: int, *, loc=None, ip=None):
    """tcgen05.ld.32x32b.x{num}: ``num`` consecutive fp32 columns of this warp's 32 lanes."""
    ptr = _tmem_ptr((Int32(row) << Int32(16)) | Int32(col), loc=loc, ip=ip)
    if num == 1:
        reg = nvvm.tcgen05_ld(Int32.mlir_type, _SHAPE_32X32B, ptr, loc=loc, ip=ip)
        return [Float32(llvm.bitcast(Float32.mlir_type, reg, loc=loc, ip=ip))]
    vi = ir.VectorType.get([num], Int32.mlir_type, loc=loc)
    vf = ir.VectorType.get([num], Float32.mlir_type, loc=loc)
    regs = nvvm.tcgen05_ld(vi, _SHAPE_32X32B, ptr, loc=loc, ip=ip)
    return cute.TensorSSA(llvm.bitcast(vf, regs, loc=loc, ip=ip), (num,), Float32)


@dsl_user_op
def tmem_st_u32(row, col, words, *, loc=None, ip=None) -> None:
    """tcgen05.st.32x32b.x{len(words)} of raw 32-bit words."""
    ptr = _tmem_ptr((Int32(row) << Int32(16)) | Int32(col), loc=loc, ip=ip)
    n = len(words)
    if n == 1:
        nvvm.tcgen05_st(
            _SHAPE_32X32B,
            ptr,
            Uint32(words[0]).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        return
    vi = ir.VectorType.get([n], Int32.mlir_type, loc=loc)
    vec = vector.from_elements(
        vi, [Uint32(w).ir_value(loc=loc, ip=ip) for w in words], loc=loc, ip=ip
    )
    nvvm.tcgen05_st(_SHAPE_32X32B, ptr, vec, loc=loc, ip=ip)


@dsl_user_op
def tmem_wait_ld(*, loc=None, ip=None):
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD, loc=loc, ip=ip)


@dsl_user_op
def tmem_wait_st(*, loc=None, ip=None):
    nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.STORE, loc=loc, ip=ip)


@dsl_user_op
def fence_after_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def fence_before_sync(*, loc=None, ip=None):
    nvvm.tcgen05_fence(nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


# ----------------------------------------------------------------------------- UMMA
def f16_idesc(m: int, n: int, *, a_mn_major: bool = False, b_mn_major: bool = False):
    """kind::f16 instruction descriptor: fp16 A/B, fp32 D, K-major unless told otherwise."""
    return Uint32(
        (1 << 4)
        | (int(a_mn_major) << 15)
        | (int(b_mn_major) << 16)
        | ((n >> 3) << 17)
        | ((m >> 4) << 24)
    )


def sdesc_sw128(lbo: int):
    """SMEM descriptor template, 128B swizzle, K-major, SBO = 8 rows x 128 B (OR in addr >> 4)."""
    return Uint64(((lbo >> 4) << 16) | ((1024 >> 4) << 32) | (1 << 46) | (2 << 61))


def sdesc_sw32(base16):
    """SMEM descriptor for [N, 16] fp16 K-major tiles, 32B swizzle (SBO = 8 rows x 32 B).

    ``base16`` is the runtime smem address >> 4.  Mode 6 (bit 63 set) is added
    around the runtime operand: the DSL constant-folds pure-constant shifts and
    the folded value overflows a signed int64.
    """
    half_mode = Uint64(3 << 61)
    lo = Uint64((1 << 16) | (16 << 32) | (1 << 46))
    return ((base16 + half_mode) + half_mode) | lo


@dsl_user_op
def umma_ss(d_tmem, a_desc, b_desc, idesc, accumulate, *, loc=None, ip=None) -> None:
    """tcgen05.mma.cta_group::1.kind::f16, A and B from SMEM descriptors."""
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            _CTA1,
            _tmem_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(accumulate).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def umma_ts(d_tmem, a_tmem, b_desc, idesc, accumulate, *, loc=None, ip=None) -> None:
    """tcgen05.mma.cta_group::1.kind::f16, A from TMEM, B from an SMEM descriptor."""
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            _CTA1,
            _tmem_ptr(d_tmem, loc=loc, ip=ip),
            _tmem_ptr(a_tmem, loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(accumulate).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def umma_commit(mbar, *, loc=None, ip=None):
    with cute.arch.elect_one():
        _tcgen05_commit(mbar.to_llvm_ptr(loc=loc, ip=ip), group=_CTA1, loc=loc, ip=ip)


# ----------------------------------------------------------------------------- copies
def simple_tma_copy(atom, src, dst, mbar=None):
    """TMA tile copy of one (grouped) box; call WITHOUT elect_one."""
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        gmem, smem = src, dst
    else:
        smem, gmem = src, dst
    s_part, g_part = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0),
        cute.group_modes(gmem, 0),
    )
    if isinstance(atom.op, cpasync.CopyBulkTensorTileG2SOp):
        cute.copy(atom, g_part, s_part, tma_bar_ptr=mbar)
    else:
        cute.copy(atom, s_part, g_part)


# ----------------------------------------------------------------------------- misc PTX
@dsl_user_op
def exit_if_negative(x: Int32, *, loc=None, ip=None) -> None:
    """CTA-uniform early exit without an MLIR control-flow edge (padded rows)."""
    llvm.inline_asm(
        None,
        [Int32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, $0, 0;\n\t@p exit;\n\t}",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg_v4(addr: Int64, *, loc=None, ip=None):
    """ld.global.v4.b32 (16 B, aligned) -> four u32 words."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def ldg_u16(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """ld.global.u16 zero-extended to 32 bits (raw 16-bit payload)."""
    r = llvm.inline_asm(
        T.i32(),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.u16 $0, [$1];",
        "=r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def stg_u16(addr: Int64, w: Uint32, *, loc=None, ip=None) -> None:
    """st.global.u16 of the low half of ``w``."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u16 [$0], h;\n\t}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_bf16_16816(a0, a1, a2, a3, b0, b1, c, *, loc=None, ip=None):
    """mma.sync m16n8k16 row.col, f32 += bf16 x bf16.  a*: 4 u32, b*: 2 u32, c: 4 f32."""
    st = llvm.StructType.get_literal([T.f32()] * 4)
    ops = [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
    ops += [Float32(x).ir_value(loc=loc, ip=ip) for x in c]
    r = llvm.inline_asm(
        st,
        ops,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def pack_f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """cvt.rn.f16x2.f32: ``lo`` in bits 0..15, ``hi`` in bits 16..31."""
    r = llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def pack_f16x2_rs(
    lo: Float32, hi: Float32, rbits: Uint32, *, loc=None, ip=None
) -> Uint32:
    """cvt.rs.f16x2.f32 (stochastic rounding, sm_100a / sm_103a): ``lo`` in bits 0..15
    rounded with ``rbits[12:0]``, ``hi`` in bits 16..31 with ``rbits[28:16]`` -- the same
    operand mapping as ``conversion::cvt_rs_f16x2_f32`` (include/flashinfer/mamba)."""
    r = llvm.inline_asm(
        T.i32(),
        [
            Float32(hi).ir_value(loc=loc, ip=ip),
            Float32(lo).ir_value(loc=loc, ip=ip),
            Uint32(rbits).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rs.f16x2.f32 $0, $1, $2, $3;",
        "=r,f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


_PHILOX_ROUND_A = 0xD2511F53
_PHILOX_ROUND_B = 0xCD9E8D57
_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85
_PHILOX_MAX_UNROLL = 6


@cute.jit
def philox4x32_streams(
    seed_lo: Uint32,
    seed_hi: Uint32,
    off_lo0: Uint32,
    off_hi: Uint32,
    n_streams: cutlass.Constexpr,
    n_rounds: cutlass.Constexpr,
):
    """Philox4x32 for the ``n_streams`` (2 or 4) counters (off_lo0 + 8 s, off_hi, 0, 0), key
    (seed_lo, seed_hi): ``[s][i]`` uint32 draws. Each stream is bit-identical to
    ``conversion::philox_randint4x`` (include/flashinfer/mamba) / Triton's ``tl.randint4x``
    with an int64 offset; the caller guarantees off_lo0 + 8 (n_streams - 1) does not carry
    into off_hi.

    Up to ``_PHILOX_MAX_UNROLL`` rounds are unrolled, with rounds 1-2 specialised (exact):
    c2 = c3 = 0 and c1 = off_hi is shared by every stream, so round 1 needs one multiply per
    stream and leaves c0 = off_hi ^ k0, c1 = 0 shared, and round 2's c0 product is one
    multiply per call (2 of the 4 per-stream multiplies of rounds 1-2 disappear). More rounds
    run as a plain loop from round 1: unrolled -- or with the peeled rounds repeated at every
    call site -- they push the kernel past the instruction cache (B200, T = 8, B = 512, 100%
    fold, 10 rounds: unrolled 3x the looped cost, peeled +20%)."""
    if cutlass.const_expr(n_rounds > _PHILOX_MAX_UNROLL):
        return _philox_loop(seed_lo, seed_hi, off_lo0, off_hi, n_streams, n_rounds)
    m32 = Uint64(0xFFFFFFFF)
    k0, k1 = Uint32(seed_lo), Uint32(seed_hi)
    # round 1 (per stream: c0 * A only) and the shared half of round 2
    x0 = Uint32(off_hi) ^ k0  # c0 after round 1, every stream
    k0b, k1b = k0 + Uint32(_PHILOX_KEY_A), k1 + Uint32(_PHILOX_KEY_B)
    pa2 = Uint64(x0) * Uint64(_PHILOX_ROUND_A)
    y2 = Uint32(pa2 >> Uint64(32)) ^ k1b  # c2 after round 2 = y2 ^ c3_r1
    z2 = Uint32(pa2 & m32)  # c3 after round 2, every stream
    st = []
    for s_ in cutlass.range_constexpr(n_streams):
        pa = Uint64(Uint32(off_lo0) + Uint32(8 * s_)) * Uint64(_PHILOX_ROUND_A)
        c2 = Uint32(pa >> Uint64(32)) ^ k1  # round 1: c1 = 0, c3 = lo(pa)
        pb = Uint64(c2) * Uint64(_PHILOX_ROUND_B)  # round 2, per stream
        st.append(
            [
                Uint32(pb >> Uint64(32)) ^ k0b,
                Uint32(pb & m32),
                y2 ^ Uint32(pa & m32),
                z2,
            ]
        )
    k0 = k0b + Uint32(_PHILOX_KEY_A)
    k1 = k1b + Uint32(_PHILOX_KEY_B)
    rest = n_rounds - 2
    unroll = rest
    if cutlass.const_expr(n_streams == 4):
        a0, b0, d0, e0 = st[0]
        a1, b1, d1, e1 = st[1]
        a2, b2, d2, e2 = st[2]
        a3, b3, d3, e3 = st[3]
        for _ in cutlass.range(rest, unroll=unroll):
            a0, b0, d0, e0 = _philox_round(a0, b0, d0, e0, k0, k1)
            a1, b1, d1, e1 = _philox_round(a1, b1, d1, e1, k0, k1)
            a2, b2, d2, e2 = _philox_round(a2, b2, d2, e2, k0, k1)
            a3, b3, d3, e3 = _philox_round(a3, b3, d3, e3, k0, k1)
            k0 = k0 + Uint32(_PHILOX_KEY_A)
            k1 = k1 + Uint32(_PHILOX_KEY_B)
        out = [[a0, b0, d0, e0], [a1, b1, d1, e1], [a2, b2, d2, e2], [a3, b3, d3, e3]]
    else:
        a0, b0, d0, e0 = st[0]
        a1, b1, d1, e1 = st[1]
        for _ in cutlass.range(rest, unroll=unroll):
            a0, b0, d0, e0 = _philox_round(a0, b0, d0, e0, k0, k1)
            a1, b1, d1, e1 = _philox_round(a1, b1, d1, e1, k0, k1)
            k0 = k0 + Uint32(_PHILOX_KEY_A)
            k1 = k1 + Uint32(_PHILOX_KEY_B)
        out = [[a0, b0, d0, e0], [a1, b1, d1, e1]]
    return out


@cute.jit
def _philox_loop(
    seed_lo: Uint32,
    seed_hi: Uint32,
    off_lo0: Uint32,
    off_hi: Uint32,
    n_streams: cutlass.Constexpr,
    n_rounds: cutlass.Constexpr,
):
    """Plain Philox4x32 rounds as a runtime loop (see ``philox4x32_streams``)."""
    k0, k1 = Uint32(seed_lo), Uint32(seed_hi)
    z = Uint32(0)
    a0, b0, d0, e0 = Uint32(off_lo0), Uint32(off_hi), z, z
    a1, b1, d1, e1 = Uint32(off_lo0) + Uint32(8), Uint32(off_hi), z, z
    if cutlass.const_expr(n_streams == 4):
        a2, b2, d2, e2 = Uint32(off_lo0) + Uint32(16), Uint32(off_hi), z, z
        a3, b3, d3, e3 = Uint32(off_lo0) + Uint32(24), Uint32(off_hi), z, z
        for _ in cutlass.range(n_rounds, unroll=1):
            a0, b0, d0, e0 = _philox_round(a0, b0, d0, e0, k0, k1)
            a1, b1, d1, e1 = _philox_round(a1, b1, d1, e1, k0, k1)
            a2, b2, d2, e2 = _philox_round(a2, b2, d2, e2, k0, k1)
            a3, b3, d3, e3 = _philox_round(a3, b3, d3, e3, k0, k1)
            k0 = k0 + Uint32(_PHILOX_KEY_A)
            k1 = k1 + Uint32(_PHILOX_KEY_B)
        out = [[a0, b0, d0, e0], [a1, b1, d1, e1], [a2, b2, d2, e2], [a3, b3, d3, e3]]
    else:
        for _ in cutlass.range(n_rounds, unroll=1):
            a0, b0, d0, e0 = _philox_round(a0, b0, d0, e0, k0, k1)
            a1, b1, d1, e1 = _philox_round(a1, b1, d1, e1, k0, k1)
            k0 = k0 + Uint32(_PHILOX_KEY_A)
            k1 = k1 + Uint32(_PHILOX_KEY_B)
        out = [[a0, b0, d0, e0], [a1, b1, d1, e1]]
    return out


def _philox_round(c0, c1, c2, c3, k0, k1):
    pb = Uint64(c2) * Uint64(_PHILOX_ROUND_B)
    pa = Uint64(c0) * Uint64(_PHILOX_ROUND_A)
    return (
        Uint32(pb >> Uint64(32)) ^ c1 ^ k0,
        Uint32(pb & Uint64(0xFFFFFFFF)),
        Uint32(pa >> Uint64(32)) ^ c3 ^ k1,
        Uint32(pa & Uint64(0xFFFFFFFF)),
    )


# ---- LCG stochastic-rounding dither (the cheap alternative to Philox) ----
LCG_A, LCG_C = 1664525, 1013904223  # full-period LCG mod 2^32 (Numerical Recipes)
LCG_A4 = LCG_A**4 % 2**32  # 4 steps at once: x_{n+4} = A4 x_n + C4
LCG_C4 = LCG_C * (LCG_A**3 + LCG_A**2 + LCG_A + 1) % 2**32
_GOLDEN, _MIX1, _MIX2 = 0x9E3779B9, 0x85EBCA6B, 0xC2B2AE35


def fmix32(x):
    """murmur3 32-bit finalizer (bijective avalanche)."""
    x = Uint32(x)
    x = x ^ (x >> Uint32(16))
    x = x * Uint32(_MIX1)
    x = x ^ (x >> Uint32(13))
    x = x * Uint32(_MIX2)
    return x ^ (x >> Uint32(16))


@dsl_user_op
def clock32(*, loc=None, ip=None) -> Uint32:
    """%clock (this SM's cycle counter): the unseeded LCG mode's entropy."""
    r = llvm.inline_asm(
        T.i32(),
        [],
        "mov.u32 $0, %clock;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """cvt.rn.bf16x2.f32: ``lo`` in bits 0..15, ``hi`` in bits 16..31."""
    r = llvm.inline_asm(
        T.i32(),
        [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def u16_as_f32(w: Uint32, is_bf16: bool, *, loc=None, ip=None) -> Float32:
    """Low 16 bits of ``w`` (bf16 or fp16 payload) widened to fp32."""
    if is_bf16:
        asm = "shl.b32 $0, $1, 16;"
    else:
        asm = "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tcvt.f32.f16 $0, h;\n\t}"
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        asm,
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def f32_to_u16(x: Float32, is_bf16: bool, *, loc=None, ip=None) -> Uint32:
    """Round ``x`` to bf16 / fp16 (RN), payload in the low 16 bits."""
    cvt = "cvt.rn.bf16.f32" if is_bf16 else "cvt.rn.f16.f32"
    r = llvm.inline_asm(
        T.i32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\t" + cvt + " h, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def lds_v4(addr: Int32, *, loc=None, ip=None):
    """ld.shared.v4.b32 -> four u32 words."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def sts_v4(addr: Int32, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    """st.shared.v4.b32."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1,$2,$3,$4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_u16(addr: Int32, w: Uint32, *, loc=None, ip=None) -> None:
    """st.shared.u16 of the low half of ``w``."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.shared.u16 [$0], h;\n\t}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def f16x2_lo_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.f16 $0, l;\n\t}",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def f16x2_hi_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.f16 $0, h;\n\t}",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


def make_fake_tensor(dtype, shape, divisibility=1):
    """Fake CuTe tensor with dynamic strides (last dim stride 1) for tensor-free compilation."""
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != len(shape) - 1 else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=stride,
        assumed_align=max(divisibility * dtype.width // 8, 1),
    )


@dsl_user_op
def bulk_g2s(
    dst_smem: Int32,
    src_gmem: Int64,
    nbytes: Int32,
    mbar_smem: Int32,
    *,
    loc=None,
    ip=None,
):
    """cp.async.bulk global -> shared (1-D), completion counted on an mbarrier (complete_tx).
    ``nbytes`` a multiple of 16; addresses 16-B aligned."""
    llvm.inline_asm(
        T.i32(),
        [
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "{\n\tcp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [$1], [$2], $3, [$4];\n\tmov.u32 $0, 0;\n\t}\n",
        "=r,r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_u16(addr: Int32, *, loc=None, ip=None) -> Uint32:
    """ld.shared.u16 zero-extended into a 32-bit register (raw 16-bit payload).  No
    conversion in the asm block: a cvt there would make every load wait for its data
    before the next one issues."""
    r = llvm.inline_asm(
        T.i32(),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.u16 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


def make_fake_tensor_lead_dyn(dtype, shape, divisibility=1):
    """Fake CuTe tensor whose inner dims are dense with STATIC strides and whose leading
    stride is dynamic (paged / block-strided pools), so in-kernel addressing folds the
    inner offsets into immediates."""
    inner = 1
    strides = []
    for d in reversed(shape[1:]):
        strides.append(inner)
        inner *= d
    strides = [cute.sym_int64(divisibility=divisibility)] + list(reversed(strides))
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=tuple(strides),
        assumed_align=max(divisibility * dtype.width // 8, 1),
    )


@dsl_user_op
def bf16x2_lo_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $1, 16;",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def bf16x2_hi_f32(w: Uint32, *, loc=None, ip=None) -> Float32:
    r = llvm.inline_asm(
        T.f32(),
        [Uint32(w).ir_value(loc=loc, ip=ip)],
        "and.b32 $0, $1, 0xffff0000;",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def stg_v4(addr: Int64, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    """st.global.v4.b32 (16 B, aligned)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.b32 [$0], {$1,$2,$3,$4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldg_v4_if(pred: Int32, addr: Int64, *, loc=None, ip=None):
    """Predicated ld.global.v4.b32: threads with pred == 0 skip the load (values undefined)."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "@p ld.global.v4.b32 {$0,$1,$2,$3}, [$5];\n\t}",
        "=r,=r,=r,=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def ldg_v4_if_keep(pred: Int32, addr: Int64, w0, w1, w2, w3, *, loc=None, ip=None):
    """Predicated ld.global.v4.b32 into the four given registers: threads with pred == 0 keep
    their values (tied operands), so one register set can hold either of two loads by role."""
    st = llvm.StructType.get_literal([T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "@p ld.global.v4.b32 {$0,$1,$2,$3}, [$5];\n\t}",
        "=r,=r,=r,=r,r,l,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


def sdesc_mn_sw128(lbo: int, sbo: int):
    """SMEM descriptor template, 128B swizzle, MN-major operand (OR in addr >> 4).  For an
    operand stored row-major [K rows][MN] in 128 B swizzled rows (8-row atoms): SBO = byte
    stride between 8-row K atoms; LBO = stride between 64-element MN atoms (unused for N=64)."""
    return Uint64(((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (1 << 46) | (2 << 61))
