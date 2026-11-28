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

"""K/V transform intrinsics of the decode kernel.

NVFP4 (E2M1 + E4M3 block scale) and E4M3 to BF16/E4M3 conversions, byte
permutes and shared-memory vector loads used by the transformed-K/V resources.
"""

import cutlass.cute as cute
import cutlass
from cutlass import Int32
from .....device.packing import has_packed_e4m3_e2m1_mul
from .common import Constexpr


@cute.jit
def broadcast_byte(word: Int32, byte_idx: Int32) -> Int32:
    """Replicate byte ``byte_idx`` (runtime, 0..3) of ``word`` into all bytes."""
    return cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r0}, {$r1};",
        write_only_types=[Int32],
        read_only_args=[word, byte_idx * Int32(0x1111)],
    )


@cute.jit
def broadcast_byte_const(word: Int32, byte_idx: Constexpr[int]) -> Int32:
    """``broadcast_byte`` with a compile-time byte, as an immediate selector."""
    imm = ("0x0000", "0x1111", "0x2222", "0x3333")[byte_idx]
    return cute.arch.inline_ptx(
        f"prmt.b32 {{$w0}}, {{$r0}}, {{$r0}}, {imm};",
        write_only_types=[Int32],
        read_only_args=[word],
    )


@cute.jit
def byte_permute(a: Int32, b: Int32, selector: Constexpr[int]) -> Int32:
    """Pick four bytes of the pair (``a`` = bytes 0-3, ``b`` = bytes 4-7).

    Nibble i of ``selector`` (least significant first) names the source byte
    of result byte i, as in PTX ``prmt.b32``. For example ``0x5140`` packs
    a0, b0, a1, b1: it interleaves the even/odd E4M3 products of two words.
    """
    return cute.arch.inline_ptx(
        f"prmt.b32 {{$w0}}, {{$r0}}, {{$r1}}, {selector:#x};",
        write_only_types=[Int32],
        read_only_args=[a, b],
    )


@cute.jit
def byte_permute_runtime(a: Int32, b: Int32, selector: Int32) -> Int32:
    """``byte_permute`` with a runtime ``selector``."""
    return cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r1}, {$r2};",
        write_only_types=[Int32],
        read_only_args=[a, b, selector],
    )


@cute.jit
def e4m3x2_to_bf16x2(packed_pair: Int32) -> Int32:
    """Convert the low two E4M3 bytes of a word to BF16x2 exactly."""
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0}; "
            "cvt.rn.bf16x2.e4m3x2 {$w0}, lo; }",
            write_only_types=[Int32],
            read_only_args=[packed_pair],
        )
    return cute.arch.inline_ptx(
        "{ .reg .b16 lo, hi, h0, h1, b0, b1; .reg .b32 half2; "
        "mov.b32 {lo, hi}, {$r0}; cvt.rn.f16x2.e4m3x2 half2, lo; "
        "mov.b32 {h0, h1}, half2; cvt.rn.bf16.f16 b0, h0; cvt.rn.bf16.f16 b1, h1; "
        "mov.b32 {$w0}, {b0, b1}; }",
        write_only_types=[Int32],
        read_only_args=[packed_pair],
    )


def has_direct_bf16_fp4_fp8_cvt() -> bool:
    """Whether PTX converts E2M1x2 and E4M3x2 directly to BF16x2 (CUDA 13.2+)."""
    return cutlass.target_version(min_version="13.2")


@cute.jit
def ld_shared_b32(addr: Int32) -> Int32:
    """Load one 32-bit word from a 4-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.b32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
        read_only_args=[addr],
    )


@cute.jit
def ld_shared_v2_b32(addr: Int32):
    """Load two 32-bit words from an 8-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.v2.b32 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32, Int32],
        read_only_args=[addr],
    )


# The shared-memory helpers below take 32-bit shared-window byte addresses.
# Transform loops compute per-token row addresses arithmetically (fragment
# slots, token rows, swizzled segments), so raw vector loads and stores avoid
# building a CuTe tensor view per row inside the rolled loops.


@cute.jit
def ld_shared_v4_b32(addr: Int32):
    """Load four 32-bit words from a 16-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[addr],
    )


@cute.jit
def ldsm_m16n16_trans_b8(addr: Int32):
    """Transposing 16x16-byte LDSM: thread t gets byte columns t/4 and t/4+8
    for rows 4*(t%4) .. 4*(t%4)+3 (rows are supplied by threads 0..15)."""
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32, Int32],
        read_only_args=[addr],
    )


@cute.jit
def mul_e2m1x4_e4m3x4(packed_fp4: Int32, packed_sf: Int32) -> Int32:
    """Apply four E4M3 scale factors to four unpacked E2M1 values.

    Each FP4 value occupies the low nibble of one byte in the 32-bit input;
    byte i scales by byte i of ``packed_sf`` and the products round to E4M3.
    Both paths ignore the high nibble of every input byte: the packed
    ``e2m1p4`` operand defines it as padding (checked exhaustively on SM103
    for all byte and scale values), and the f16x2 fallback masks it off.
    Callers may therefore pass packed payload bytes, or the same word shifted
    right by four, without masking.
    """
    if cutlass.const_expr(has_packed_e4m3_e2m1_mul()):
        return cute.arch.inline_ptx(
            "mul.e4m3x4.e2m1p4x4.e4m3x4.satfinite {$w0}, {$r0}, {$r1};",
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )
    else:
        # Keep each byte's low nibble, then OR with a four-bit shift to pack
        # adjacent values into bytes 0 and 2 without scalar extraction.
        return cute.arch.inline_ptx(
            """
            {
                .reg .b8 fp4_01, fp4_23, unused0, unused1;
                .reg .b16 sf_01, sf_23, e4m3_01, e4m3_23;
                .reg .b32 lo, tmp, h_01, h_23, sf_h_01, sf_h_23;
                and.b32 lo, {$r0}, 0x0F0F0F0F;
                shr.u32 tmp, lo, 4;
                or.b32 tmp, tmp, lo;
                mov.b32 {fp4_01, unused0, fp4_23, unused1}, tmp;
                mov.b32 {sf_01, sf_23}, {$r1};
                cvt.rn.f16x2.e2m1x2 h_01, fp4_01;
                cvt.rn.f16x2.e2m1x2 h_23, fp4_23;
                cvt.rn.f16x2.e4m3x2 sf_h_01, sf_01;
                cvt.rn.f16x2.e4m3x2 sf_h_23, sf_23;
                mul.rn.f16x2 h_01, h_01, sf_h_01;
                mul.rn.f16x2 h_23, h_23, sf_h_23;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_01, h_01;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_23, h_23;
                mov.b32 {$w0}, {e4m3_01, e4m3_23};
            }
            """,
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )


@cute.jit
def mul_e2m1x4_e4m3x4_bx(
    packed_fp4: Int32, packed_sf: Int32, sf_byte_idx: Constexpr[int]
) -> Int32:
    """Apply one selected E4M3 scale byte to four unpacked E2M1 values."""
    return mul_e2m1x4_e4m3x4(packed_fp4, broadcast_byte_const(packed_sf, sf_byte_idx))


@cute.jit
def scale_e2m1_token_quad_to_bf16x2x4(word: Int32, sf01: Int32, sf23: Int32):
    """Scale a byte column of four tokens into token-pair BF16x2 words.

    Byte i of ``word`` is token i, holding head-dim rows d (low nibble) and
    d+1 (high nibble). Returns (row d: tokens 0/1, row d: tokens 2/3,
    row d+1: tokens 0/1, row d+1: tokens 2/3); ``sf01``/``sf23`` are the
    per-token BF16 scale pairs. Two byte-selects pair the tokens' nibbles of
    each row so one conversion yields a token pair; products are exact.
    """
    return cute.arch.inline_ptx(
        """{
            .reg .b32 s4, s8, lo, hi;
            .reg .b8 l0, l1, l2, l3, h0, h1, h2, h3;
            shr.u32 s4, {$r0}, 4;
            shr.u32 s8, {$r0}, 8;
            lop3.b32 lo, {$r0}, s4, 0x0F0F0F0F, 0xE4;
            lop3.b32 hi, s4, s8, 0x0F0F0F0F, 0xE4;
            mov.b32 {l0, l1, l2, l3}, lo;
            mov.b32 {h0, h1, h2, h3}, hi;
            cvt.rn.bf16x2.e2m1x2 {$w0}, l0;
            cvt.rn.bf16x2.e2m1x2 {$w1}, l2;
            cvt.rn.bf16x2.e2m1x2 {$w2}, h0;
            cvt.rn.bf16x2.e2m1x2 {$w3}, h2;
            mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
            mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
            mul.rn.bf16x2 {$w2}, {$w2}, {$r1};
            mul.rn.bf16x2 {$w3}, {$w3}, {$r2};
        }""",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[word, sf01, sf23],
    )


@cute.jit
def scale_e2m1x4_unpacked_to_bf16x2x2(
    unpacked_fp4: Int32, sf_lo: Int32, sf_hi: Int32
) -> tuple[Int32, Int32]:
    """Scale four unpacked E2M1 values into two exact BF16x2 words.

    ``unpacked_fp4`` holds one E2M1 value in the low nibble of each byte (the
    ``B4X16_P64`` LDSM output). Values 0/1 use ``sf_lo`` and values 2/3 use
    ``sf_hi``. Every E2M1 x E4M3 product has at most five significant bits,
    so the BF16 multiply is exact; nothing is rounded through E4M3.
    """
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            """{
                .reg .b32 x, y;
                .reg .b8 b0, b1, b2, b3;
                and.b32 x, {$r0}, 0x0F0F0F0F;
                shr.u32 y, x, 4;
                or.b32 x, x, y;
                mov.b32 {b0, b1, b2, b3}, x;
                cvt.rn.bf16x2.e2m1x2 {$w0}, b0;
                cvt.rn.bf16x2.e2m1x2 {$w1}, b2;
                mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
                mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
            }""",
            write_only_types=[Int32, Int32],
            read_only_args=[unpacked_fp4, sf_lo, sf_hi],
        )
    return cute.arch.inline_ptx(
        """{
            .reg .b32 x, y, h01, h23;
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 a0, a1, a2, a3, c0, c1, c2, c3;
            and.b32 x, {$r0}, 0x0F0F0F0F;
            shr.u32 y, x, 4;
            or.b32 x, x, y;
            mov.b32 {b0, b1, b2, b3}, x;
            cvt.rn.f16x2.e2m1x2 h01, b0;
            cvt.rn.f16x2.e2m1x2 h23, b2;
            mov.b32 {a0, a1}, h01;
            mov.b32 {a2, a3}, h23;
            cvt.rn.bf16.f16 c0, a0;
            cvt.rn.bf16.f16 c1, a1;
            cvt.rn.bf16.f16 c2, a2;
            cvt.rn.bf16.f16 c3, a3;
            mov.b32 {$w0}, {c0, c1};
            mov.b32 {$w1}, {c2, c3};
            mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
            mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
        }""",
        write_only_types=[Int32, Int32],
        read_only_args=[unpacked_fp4, sf_lo, sf_hi],
    )


@cute.jit
def scale_e2m1x8_packed_to_bf16x2x4(packed: Int32, sf_pair: Int32):
    """Scale eight packed E2M1 values (two per byte) into four exact BF16x2."""
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            """{
                .reg .b8 b0, b1, b2, b3;
                mov.b32 {b0, b1, b2, b3}, {$r0};
                cvt.rn.bf16x2.e2m1x2 {$w0}, b0;
                cvt.rn.bf16x2.e2m1x2 {$w1}, b1;
                cvt.rn.bf16x2.e2m1x2 {$w2}, b2;
                cvt.rn.bf16x2.e2m1x2 {$w3}, b3;
                mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
                mul.rn.bf16x2 {$w1}, {$w1}, {$r1};
                mul.rn.bf16x2 {$w2}, {$w2}, {$r1};
                mul.rn.bf16x2 {$w3}, {$w3}, {$r1};
            }""",
            write_only_types=[Int32, Int32, Int32, Int32],
            read_only_args=[packed, sf_pair],
        )
    lo = scale_e2m1x4_unpacked_to_bf16x2x2(packed & Int32(0x0F0F0F0F), sf_pair, sf_pair)
    hi = scale_e2m1x4_unpacked_to_bf16x2x2(
        (packed >> Int32(4)) & Int32(0x0F0F0F0F), sf_pair, sf_pair
    )
    # Unpacked order is (b0lo, b1lo), (b2lo, b3lo); restore (lo, hi) per byte.
    return (
        byte_permute(lo[0], hi[0], 0x5410),
        byte_permute(lo[0], hi[0], 0x7632),
        byte_permute(lo[1], hi[1], 0x5410),
        byte_permute(lo[1], hi[1], 0x7632),
    )


@cute.jit
def scale_e2m1x8_to_e4m3x8(packed_fp4: Int32, packed_sf: Int32):
    """Scale eight dense-packed E2M1 values; return two E4M3 words in order.

    ``packed_fp4`` holds values 0-7 in nibble order (byte k: value 2k low,
    2k+1 high); every byte of ``packed_sf`` is the block's E4M3 scale. With the
    packed multiply, the dense ``e2m1x4`` operand takes four consecutive
    nibbles per 16-bit half, so each half is one multiply and the products are
    already in order. Otherwise the padded form scales even and odd nibbles
    separately and two byte permutes interleave them back.
    """
    if cutlass.const_expr(has_packed_e4m3_e2m1_mul()):
        return cute.arch.inline_ptx(
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0}; "
            "mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite {$w0}, lo, {$r1}; "
            "mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite {$w1}, hi, {$r1}; }",
            write_only_types=[Int32, Int32],
            read_only_args=[packed_fp4, packed_sf],
        )
    even = mul_e2m1x4_e4m3x4(packed_fp4, packed_sf)
    odd = mul_e2m1x4_e4m3x4(packed_fp4 >> Int32(4), packed_sf)
    return byte_permute(even, odd, 0x5140), byte_permute(even, odd, 0x7362)


@cute.jit
def st_shared_v4_b32(addr: Int32, w0: Int32, w1: Int32, w2: Int32, w3: Int32) -> None:
    """Store four 32-bit words to a 16-byte-aligned shared address."""
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, w0, w1, w2, w3],
    )
