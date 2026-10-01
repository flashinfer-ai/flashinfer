# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Inline PTX shared by the SM12x MXFP8 GEMM kernels.

Fragment coordinates of ``mma.sync.m16n8k32`` follow the PTX ISA tables for
``kind::mxf8f6f4`` and CUTLASS ``include/cute/atom/mma_traits_sm120.hpp``.
Addresses may be CuTe pointers or integers (64-bit global, 32-bit shared).
"""

from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


def _v(x, loc=None, ip=None):
    if hasattr(x, "toint"):
        x = x.toint(loc=loc, ip=ip)
    return x.ir_value(loc=loc, ip=ip) if hasattr(x, "ir_value") else x


def _asm(ret, args, text, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(
        ret,
        [_v(a, loc, ip) for a in args],
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _unpack(res, ty, n, loc=None, ip=None):
    return tuple(
        ty(llvm.extractvalue(ty.mlir_type, res, [i], loc=loc, ip=ip)) for i in range(n)
    )


def _struct(ty, n):
    return llvm.StructType.get_literal([ty.mlir_type] * n)


# --------------------------------------------------------------------- MMA
@dsl_user_op
def mma_mxf8(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    sfa,
    sfb,
    bida=0,
    bidb=0,
    *,
    loc=None,
    ip=None,
):
    """D = A(16x32 e4m3) * B(32x8 e4m3) * 2^(sfa, sfb) + C, FP32 accumulate.

    ``sfa`` / ``sfb`` hold four UE8M0 bytes; ``bida`` / ``bidb`` select the byte
    (one scale per row of A / column of B for this k32 block).
    """
    text = (
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale."
        "scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, "
        f"$14, {{{bida}, 0}}, $15, {{{bidb}, 0}};"
    )
    res = _asm(
        _struct(Float32, 4),
        [a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb],
        text,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, Float32, 4, loc, ip)


# ------------------------------------------------------------ global loads
@dsl_user_op
def ldg_nc_v4_stream(addr, *, loc=None, ip=None):
    """16-byte read-only load that skips L1 allocation (data read once)."""
    res = _asm(
        _struct(Uint32, 4),
        [addr],
        "ld.global.nc.L1::no_allocate.v4.u32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, Uint32, 4, loc, ip)


@dsl_user_op
def ldg_nc_v4(addr, *, loc=None, ip=None):
    res = _asm(
        _struct(Uint32, 4),
        [addr],
        "ld.global.nc.v4.u32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, Uint32, 4, loc, ip)


@dsl_user_op
def ldg_nc_v2(addr, *, loc=None, ip=None):
    res = _asm(
        _struct(Uint32, 2),
        [addr],
        "ld.global.nc.v2.u32 {$0,$1}, [$2];",
        "=r,=r,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, Uint32, 2, loc, ip)


@dsl_user_op
def ldg_nc_u32(addr, *, loc=None, ip=None):
    return Uint32(
        _asm(
            Uint32.mlir_type,
            [addr],
            "ld.global.nc.u32 $0, [$1];",
            "=r,l",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_nc_u16(addr, *, loc=None, ip=None):
    return Uint32(
        _asm(
            Uint32.mlir_type,
            [addr],
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_nc_u8(addr, *, loc=None, ip=None):
    return Uint32(
        _asm(
            Uint32.mlir_type,
            [addr],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ldg_cg_v4_f32(addr, *, loc=None, ip=None):
    """L2-coherent 16-byte load (data written by other CTAs of this launch)."""
    res = _asm(
        _struct(Float32, 4),
        [addr],
        "ld.global.cg.v4.f32 {$0,$1,$2,$3}, [$4];",
        "=f,=f,=f,=f,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, Float32, 4, loc, ip)


# ----------------------------------------------------------- global stores
@dsl_user_op
def stg_v4_f32(addr, x, y, z, w, *, loc=None, ip=None):
    _asm(
        None,
        [addr, x, y, z, w],
        "st.global.v4.f32 [$0], {$1,$2,$3,$4};",
        "l,f,f,f,f",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_v4_b32(addr, x, y, z, w, *, loc=None, ip=None):
    _asm(
        None,
        [addr, x, y, z, w],
        "st.global.v4.b32 [$0], {$1,$2,$3,$4};",
        "l,r,r,r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_b32(addr, v, *, loc=None, ip=None):
    _asm(None, [addr, v], "st.global.b32 [$0], $1;", "l,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def stg_b16(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [addr, v],
        "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.b16 [$0], t; }",
        "l,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_half(addr, v, f16: bool = False, *, loc=None, ip=None):
    """Round an FP32 value to BF16 (or FP16) with RNE and store it."""
    ty = "f16" if f16 else "bf16"
    _asm(
        None,
        [addr, v],
        f"{{ .reg .b16 h; cvt.rn.{ty}.f32 h, $1; st.global.b16 [$0], h; }}",
        "l,f",
        side=True,
        loc=loc,
        ip=ip,
    )


# ------------------------------------------------- atomics and ordering
@dsl_user_op
def atom_add_acq_rel_gpu(addr, v, *, loc=None, ip=None):
    return Int32(
        _asm(
            Int32.mlir_type,
            [addr, v],
            "atom.add.acq_rel.gpu.global.s32 $0, [$1], $2;",
            "=r,l,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_relaxed_gpu_s32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [addr, v],
        "st.relaxed.gpu.global.s32 [$0], $1;",
        "l,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", side=True, loc=loc, ip=ip)


@dsl_user_op
def bar_sync(bar_id, nthreads, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar_id), Int32(nthreads)],
        "bar.sync $0, $1;",
        "r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


# ------------------------------------------------------------ shared memory
@dsl_user_op
def ldsm_x4(addr, *, loc=None, ip=None):
    """ldmatrix x4 (b16 8x8) from a 32-bit shared address."""
    res = _asm(
        _struct(Int32, 4),
        [addr],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        side=True,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, Int32, 4, loc, ip)


@dsl_user_op
def lds_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            Int32.mlir_type,
            [addr],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bulk_g2s(dst_smem, src_gmem, nbytes, bar_smem, *, loc=None, ip=None):
    """1D ``cp.async.bulk`` global -> shared, completing on an mbarrier."""
    _asm(
        None,
        [dst_smem, src_gmem, nbytes, bar_smem],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


# ------------------------------------------------------- register helpers
@dsl_user_op
def shfl_bfly(v, lane_mask: int, *, loc=None, ip=None):
    return Uint32(
        _asm(
            Uint32.mlir_type,
            [v],
            f"shfl.sync.bfly.b32 $0, $1, {lane_mask}, 0x1f, 0xffffffff;",
            "=r,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx(v, src, *, loc=None, ip=None):
    return Int32(
        _asm(
            Int32.mlir_type,
            [v, src],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            side=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sel(p, a, b, *, loc=None, ip=None):
    """``p ? a : b`` on 32-bit words (``p`` is 0 or 1)."""
    return Uint32(
        _asm(
            Uint32.mlir_type,
            [p, a, b],
            "{ .reg .pred q; setp.ne.u32 q, $1, 0; selp.b32 $0, $2, $3, q; }",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def imin(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(Int32.mlir_type, [a, b], "min.s32 $0, $1, $2;", "=r,r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def pack_half2(lo, hi, f16: bool = False, *, loc=None, ip=None):
    """Two FP32 -> BF16x2 (or FP16x2), ``lo`` in the low half, RNE."""
    ty = "f16x2" if f16 else "bf16x2"
    return Int32(
        _asm(
            Int32.mlir_type,
            [lo, hi],
            f"cvt.rn.{ty}.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16_lo(u, *, loc=None, ip=None):
    return Float32(
        _asm(Float32.mlir_type, [u], "shl.b32 $0, $1, 16;", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def bf16_hi(u, *, loc=None, ip=None):
    return Float32(
        _asm(
            Float32.mlir_type,
            [u],
            "and.b32 $0, $1, 0xffff0000;",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3x4_to_f32(u, *, loc=None, ip=None):
    """Four E4M3 bytes (byte 0 first) -> four FP32 values (exact)."""
    res = _asm(
        _struct(Float32, 4),
        [u],
        "{ .reg .b16 lo, hi, a, b, c, d; .reg .b32 h0, h1;\n"
        "mov.b32 {lo, hi}, $4;\n"
        "cvt.rn.f16x2.e4m3x2 h0, lo;\n"
        "cvt.rn.f16x2.e4m3x2 h1, hi;\n"
        "mov.b32 {a, b}, h0;\n"
        "mov.b32 {c, d}, h1;\n"
        "cvt.f32.f16 $0, a; cvt.f32.f16 $1, b;\n"
        "cvt.f32.f16 $2, c; cvt.f32.f16 $3, d; }",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, Float32, 4, loc, ip)


@dsl_user_op
def pow2_e8m0(s, *, loc=None, ip=None):
    """2^(s - 127) as FP32 for a UE8M0 byte ``s`` (``s == 0`` gives 2^-127)."""
    return Float32(
        _asm(
            Float32.mlir_type,
            [s],
            "{ .reg .pred p; .reg .b32 e; shl.b32 e, $1, 23; setp.eq.u32 p, $1, 0;\n"
            "selp.b32 $0, 4194304, e, p; }",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )
