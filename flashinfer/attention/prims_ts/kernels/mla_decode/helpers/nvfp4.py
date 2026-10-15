# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inline-PTX helpers for the token-sparse NVFP4 MLA decode kernel.

They serve the on-chip NVFP4 -> E4M3 dequantizing V producer, the Q NVFP4
quantization and the TMEM scale staging of the native FP4 QK (kind::mxf4nvf4).

The V conversion follows TRT-LLM's ``nvfp4_mla_kv_cache_gather``: every E2M1
value is multiplied by its E4M3 block scale (already folded with the
per-tensor dequantization factor) and rounded once to E4M3.  The multiply
is the PTX ISA 9.4 packed ``mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite`` (one SASS
``QMUL4`` per four values), which needs CUDA 13.4+ / CuTe-DSL 4.8+.
"""

from __future__ import annotations

import cutlass
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as mlir_T

_ASM_KW = dict(
    has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT
)


@cutlass.dsl_user_op
def qmul4_lo(src: Int32, scale4: Int32, *, loc=None, ip=None) -> Int32:
    """``e4m3x4(e2m1x4(src[15:0]) * e4m3x4(scale4))`` with satfinite rounding."""
    return Int32(
        llvm.inline_asm(
            mlir_T.i32(),
            [
                Int32(src).ir_value(loc=loc, ip=ip),
                Int32(scale4).ir_value(loc=loc, ip=ip),
            ],
            "{\n  .reg .b16 a;\n  cvt.u16.u32 a, $1;\n"
            "  mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite $0, a, $2;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )


@cutlass.dsl_user_op
def qmul4_hi(src: Int32, scale4: Int32, *, loc=None, ip=None) -> Int32:
    """``e4m3x4(e2m1x4(src[31:16]) * e4m3x4(scale4))`` with satfinite rounding."""
    return Int32(
        llvm.inline_asm(
            mlir_T.i32(),
            [
                Int32(src).ir_value(loc=loc, ip=ip),
                Int32(scale4).ir_value(loc=loc, ip=ip),
            ],
            "{\n  .reg .b16 a;\n  mov.b32 {_, a}, $1;\n"
            "  mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite $0, a, $2;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )


def f32_to_f16x2(value, *, loc=None, ip=None) -> Int32:
    """Broadcast an fp32 value into both halves of an f16x2 word."""
    return Int32(
        llvm.inline_asm(
            mlir_T.i32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $1, $1;",
            "=r,f",
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )


@cutlass.dsl_user_op
def scaled_e4m3_pair_broadcast(
    s_lo: Int32, s_hi: Int32, pair_sel: Int32, factor_f16x2: Int32, *, loc=None, ip=None
):
    """Fold ``factor`` into two adjacent E4M3 block scales and broadcast each to four bytes.

    ``(s_lo, s_hi)`` hold an 8-byte scale word pair; ``pair_sel`` is the ``prmt``
    selector ``(2c) | ((2c + 1) << 4)`` picking bytes ``2c`` and ``2c + 1``.  Both
    scales are converted and multiplied as one f16x2 and rounded once to E4M3;
    this is exact for a power-of-two ``factor``, which is how the per-tensor
    dequantization scale folds into the block scales without a second rounding.
    Returns ``(sc0, sc1)`` as e4m3x4 words.
    """
    res = llvm.inline_asm(
        cutlass._mlir.ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [
            Int32(s_lo).ir_value(loc=loc, ip=ip),
            Int32(s_hi).ir_value(loc=loc, ip=ip),
            Int32(pair_sel).ir_value(loc=loc, ip=ip),
            Int32(factor_f16x2).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        "  .reg .b32 d, f, p;\n"
        "  .reg .b16 h, e;\n"
        "  prmt.b32 d, $2, $3, $4;\n"
        "  cvt.u16.u32 h, d;\n"
        "  cvt.rn.f16x2.e4m3x2 f, h;\n"
        "  mul.rn.f16x2 f, f, $5;\n"
        "  cvt.rn.satfinite.e4m3x2.f16x2 e, f;\n"
        "  mov.b32 p, {e, e};\n"
        "  prmt.b32 $0, p, 0, 0x0000;\n"
        "  prmt.b32 $1, p, 0, 0x1111;\n"
        "}",
        "=r,=r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(mlir_T.i32(), res, [1], loc=loc, ip=ip)),
    )


@cutlass.dsl_user_op
def ld_shared_v4(addr: Int32, *, loc=None, ip=None):
    """16-byte load from a CTA shared-memory address (returns four Int32)."""
    res = llvm.inline_asm(
        cutlass._mlir.ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(mlir_T.i32(), res, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@cutlass.dsl_user_op
def ld_shared_v2(addr: Int32, *, loc=None, ip=None):
    """8-byte load from a CTA shared-memory address (returns two Int32)."""
    res = llvm.inline_asm(
        cutlass._mlir.ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(mlir_T.i32(), res, [1], loc=loc, ip=ip)),
    )


@cutlass.dsl_user_op
def st_shared_v4(
    addr: Int32, v0: Int32, v1: Int32, v2: Int32, v3: Int32, *, loc=None, ip=None
) -> None:
    """16-byte store to this CTA's shared memory."""
    llvm.inline_asm(
        None,
        [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, v0, v1, v2, v3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def cp_async_8b_zfill(
    smem_addr: Int32, gmem_ptr, src_bytes: Int32, *, loc=None, ip=None
) -> None:
    """``cp.async.ca.shared.global [smem], [gmem], 8, src_bytes`` (zero-fills when ``src_bytes == 0``)."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gmem_ptr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 8, $2;",
        "r,l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def cp_async_commit_group(*, loc=None, ip=None) -> None:
    """``cp.async.commit_group``."""
    llvm.inline_asm(
        None,
        [],
        "cp.async.commit_group;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def cp_async_wait_group(pending: int, *, loc=None, ip=None) -> None:
    """``cp.async.wait_group N``: wait until at most ``pending`` groups are in flight."""
    llvm.inline_asm(
        None,
        [],
        f"cp.async.wait_group {int(pending)};",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def st_shared_u8(smem_addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    """``st.shared.u8 [smem], value`` (low byte of ``value``)."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u8 [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def warp_sync_all(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "bar.warp.sync 0xffffffff;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Native FP4 QK (kind::mxf4nvf4) helpers: Q quantization and TMEM scale staging.
# ---------------------------------------------------------------------------


@cutlass.dsl_user_op
def ld_global_v4(gmem_addr, *, loc=None, ip=None):
    """16-byte global load (returns four Int32)."""
    res = llvm.inline_asm(
        cutlass._mlir.ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [cutlass.Int64(gmem_addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(mlir_T.i32(), res, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@cutlass.dsl_user_op
def st_shared_v2(addr: Int32, v0: Int32, v1: Int32, *, loc=None, ip=None) -> None:
    """8-byte store to this CTA's shared memory."""
    llvm.inline_asm(
        None,
        [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, v0, v1)],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def e4m3x4_to_f32x4(word: Int32, *, loc=None, ip=None):
    """Four packed E4M3 bytes (byte 0 first) -> four Float32 (exact via f16)."""
    res = llvm.inline_asm(
        cutlass._mlir.ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n"
        "  .reg .b16 lo, hi, h0, h1, h2, h3;\n"
        "  .reg .b32 p0, p1;\n"
        "  mov.b32 {lo, hi}, $4;\n"
        "  cvt.rn.f16x2.e4m3x2 p0, lo;\n"
        "  cvt.rn.f16x2.e4m3x2 p1, hi;\n"
        "  mov.b32 {h0, h1}, p0;\n"
        "  mov.b32 {h2, h3}, p1;\n"
        "  cvt.f32.f16 $0, h0;\n"
        "  cvt.f32.f16 $1, h1;\n"
        "  cvt.f32.f16 $2, h2;\n"
        "  cvt.f32.f16 $3, h3;\n"
        "}",
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), res, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@cutlass.dsl_user_op
def f32_to_e4m3_rounded(value, *, loc=None, ip=None):
    """Round ``value`` to E4M3 (RN, satfinite); returns (byte as Int32, rounded Float32)."""
    res = llvm.inline_asm(
        cutlass._mlir.ir.Type.parse("!llvm.struct<(i32, f32)>"),
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "{\n"
        "  .reg .b16 b2, h0, h1;\n"
        "  .reg .b32 p;\n"
        "  .reg .f32 z;\n"
        "  mov.f32 z, 0f00000000;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 b2, z, $2;\n"  # low byte = cvt($2)
        "  cvt.u32.u16 $0, b2;\n"
        "  cvt.rn.f16x2.e4m3x2 p, b2;\n"
        "  mov.b32 {h0, h1}, p;\n"
        "  cvt.f32.f16 $1, h0;\n"
        "}",
        "=r,=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(mlir_T.i32(), res, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(mlir_T.f32(), res, [1], loc=loc, ip=ip)),
    )


@cutlass.dsl_user_op
def f32x8_to_e2m1x8(v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None) -> Int32:
    """Eight Float32 -> one word of eight E2M1 nibbles (RN, satfinite), element 0 in the low nibble."""
    res = llvm.inline_asm(
        mlir_T.i32(),
        [
            cutlass.Float32(v).ir_value(loc=loc, ip=ip)
            for v in (v0, v1, v2, v3, v4, v5, v6, v7)
        ],
        "{\n"
        "  .reg .b8 b0, b1, b2, b3;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n"  # upper nibble = $2 (elem 1), lower = $1 (elem 0)
        "  cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n"
        "  mov.b32 $0, {b0, b1, b2, b3};\n"
        "}",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@cutlass.dsl_user_op
def rcp_approx_f32(value, *, loc=None, ip=None):
    """``rcp.approx.ftz.f32``."""
    res = llvm.inline_asm(
        mlir_T.f32(),
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(res)


@cutlass.dsl_user_op
def mbarrier_try_wait_parity_u32(
    smem_addr: Int32, parity: Int32, *, loc=None, ip=None
) -> Int32:
    """One ``mbarrier.try_wait.parity`` attempt on a 32-bit shared address; returns 1 when complete."""
    res = llvm.inline_asm(
        mlir_T.i32(),
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int32(parity).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        "  .reg .pred p;\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n"
        "  selp.u32 $0, 1, 0, p;\n"
        "}",
        "=r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(res)


@cutlass.dsl_user_op
def tcgen05_fence_after_thread_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "tcgen05.fence::after_thread_sync;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def fence_proxy_async_shared(*, loc=None, ip=None) -> None:
    """``fence.proxy.async.shared::cta`` (generic-proxy SMEM writes -> async proxy / UMMA reads)."""
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cutlass.dsl_user_op
def st_shared_u32(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    """4-byte store to this CTA's shared memory."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
