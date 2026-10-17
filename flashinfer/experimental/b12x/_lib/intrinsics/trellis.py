# Copyright (c) 2025 by the b12x authors.
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

"""Trellis decoding intrinsics."""

from cutlass import Int32
from cutlass import Int64
from cutlass.cutlass_dsl import T
from cutlass import Uint32
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from b12x._lib.intrinsics.constants import (
    _TRELLIS_MASK,
    _TRELLIS_MCG,
    _TRELLIS_MUL1,
    _TRELLIS_OR,
)


def _trellis_mcg_asm_constants(asm: str) -> str:
    return (
        asm.replace("__MCG__", f"{_TRELLIS_MCG:#010x}")
        .replace("__MASK__", f"{_TRELLIS_MASK:#010x}")
        .replace("__OR__", f"{_TRELLIS_OR:#010x}")
    )


@dsl_user_op
def trellis_align_stream_u32x2(z0, z1, z2, shift, word_delta, *, loc=None, ip=None):
    """Align two words from a two- or three-word circular Trellis span.

    This is the 32-bit SHF equivalent of shifting a temporary 64/96-bit integer.
    Keeping the operation on the native integer pipe avoids compiler-generated
    wide-integer carry chains in the K6 dequantization hot loop.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(z0).ir_value(loc=loc, ip=ip),
            Uint32(z1).ir_value(loc=loc, ip=ip),
            Uint32(z2).ir_value(loc=loc, ip=ip),
            Uint32(shift).ir_value(loc=loc, ip=ip),
            Uint32(word_delta).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred two_words;
            .reg .b32 mid, upper;
            setp.eq.u32 two_words, $6, 1;
            selp.b32 mid, $2, $3, two_words;
            selp.b32 upper, 0, $2, two_words;
            // PTX shf.r forms (b:a), so z0/mid and mid/upper are low/high.
            shf.r.wrap.b32 $0, $4, mid, $5;
            shf.r.wrap.b32 $1, mid, upper, $5;
        }
        """,
        "=r,=r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_decode_trellis_mul1_e4m3_to_e4m3x8(
    win_a, win_b, bits: int = 3, *, loc=None, ip=None
):
    """Decode eight MUL1-E4M3 trellis windows to two packed FP8 words.

    The output is the native B-register form consumed by SM120's
    E4M3/E4M3 block-scaled MMA: ``lo`` contains decoded windows w0..w3 and
    ``hi`` contains w4..w7, one E4M3 value per byte.  Unlike the W4A16
    decoder, this primitive never widens the reconstructed values after the
    encoder-defined E4M3 rounding step.

    ``bits`` is a trace-time specialization. ``lut_e4m3`` uses only K2/K3/K4,
    whose four-window spans fit in each supplied 32-bit funnel window.
    """
    bits = int(bits)
    if bits not in (2, 3, 4):
        raise ValueError(
            f"unsupported MUL1-E4M3 trellis bitrate {bits}; expected 2, 3, or 4"
        )
    asm = """
        {
            .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7;
            .reg .b32 h01,h23,h45,h67, M, ones, acc;
            .reg .b32 kinv, kbias;
            .reg .b16 e01,e23,e45,e67;
            mov.b32 M, __MUL1__;
            mov.b32 ones, 0x01010101;
            mov.b32 acc, 0x00006400;
            mov.b32 kinv, 0x1eee1eee;
            mov.b32 kbias, 0xc931c931;
            and.b32 w7, $2, 0xffff;
            shr.u32 w6, $2, __B1__;  and.b32 w6, w6, 0xffff;
            shr.u32 w5, $2, __B2__;  and.b32 w5, w5, 0xffff;
            shr.u32 w4, $2, __B3__;  and.b32 w4, w4, 0xffff;
            and.b32 w3, $3, 0xffff;
            shr.u32 w2, $3, __B1__;  and.b32 w2, w2, 0xffff;
            shr.u32 w1, $3, __B2__;  and.b32 w1, w1, 0xffff;
            shr.u32 w0, $3, __B3__;  and.b32 w0, w0, 0xffff;
            mul.lo.u32 w0, w0, M;  dp4a.u32.u32 w0, w0, ones, acc;
            mul.lo.u32 w1, w1, M;  dp4a.u32.u32 w1, w1, ones, acc;
            mul.lo.u32 w2, w2, M;  dp4a.u32.u32 w2, w2, ones, acc;
            mul.lo.u32 w3, w3, M;  dp4a.u32.u32 w3, w3, ones, acc;
            mul.lo.u32 w4, w4, M;  dp4a.u32.u32 w4, w4, ones, acc;
            mul.lo.u32 w5, w5, M;  dp4a.u32.u32 w5, w5, ones, acc;
            mul.lo.u32 w6, w6, M;  dp4a.u32.u32 w6, w6, ones, acc;
            mul.lo.u32 w7, w7, M;  dp4a.u32.u32 w7, w7, ones, acc;
            prmt.b32 h01, w0, w1, 0x5410;
            prmt.b32 h23, w2, w3, 0x5410;
            prmt.b32 h45, w4, w5, 0x5410;
            prmt.b32 h67, w6, w7, 0x5410;
            fma.rn.f16x2 h01, h01, kinv, kbias;
            fma.rn.f16x2 h23, h23, kinv, kbias;
            fma.rn.f16x2 h45, h45, kinv, kbias;
            fma.rn.f16x2 h67, h67, kinv, kbias;
            cvt.rn.satfinite.e4m3x2.f16x2 e01, h01;
            cvt.rn.satfinite.e4m3x2.f16x2 e23, h23;
            cvt.rn.satfinite.e4m3x2.f16x2 e45, h45;
            cvt.rn.satfinite.e4m3x2.f16x2 e67, h67;
            mov.b32 $0, {e01, e23};
            mov.b32 $1, {e45, e67};
        }
        """
    asm = (
        asm.replace("__MUL1__", f"{_TRELLIS_MUL1:#010x}")
        .replace("__B1__", str(bits))
        .replace("__B2__", str(2 * bits))
        .replace("__B3__", str(3 * bits))
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(win_a).ir_value(loc=loc, ip=ip),
            Uint32(win_b).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_decode_lut_e4m3_to_e4m3x8(
    win_a,
    win_b,
    value_table_addr,
    bits: int = 3,
    value_table_in_shared: bool = False,
    *,
    loc=None,
    ip=None,
):
    """Decode eight ``lut_e4m3`` windows to E4M3 bytes.

    A two-xorshift, one-IMAD permutation of each window's history bits gives
    its rank: the product's high bits, combined with the bit-reversed branch,
    select the upper rank bits and its low bits give the lower rank bits. The
    4 KiB value table maps ``rank >> 4`` to an E4M3 byte.
    """

    bits = int(bits)
    value_table_in_shared = bool(value_table_in_shared)
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported lut_e4m3 bit width {bits}; expected 2, 3, or 4")
    width = 16 - bits
    low_index_mask = (1 << (width - 4)) - 1
    decode_blocks: list[str] = []
    for pair in range(4):
        indices = (2 * pair, 2 * pair + 1)
        extract_lines: list[str] = []
        product_lines: list[str] = []
        rank_lines: list[str] = []
        load_lines: list[str] = []
        pack_lines: list[str] = []
        for slot, index in enumerate(indices):
            source = "$3" if index < 4 else "$2"
            shift = (3 - (index & 3)) * bits
            target = "out0" if index < 4 else "out1"
            byte_shift = 8 * (index & 3)
            extract_lines.append(
                f"""
                    bfe.u32 w{slot}, {source}, {shift}, 16;
                """
            )
            product_lines.append(
                f"""
                    shr.u32 p{slot}, w{slot}, {bits};
                    shr.u32 t{slot}, p{slot}, 11;
                    bfi.b32 p{slot}, p{slot}, p{slot}, 11, {width - 11};
                    xor.b32 p{slot}, p{slot}, t{slot};
                    mad.lo.u32 p{slot}, p{slot}, 0x3fa7d929, 0xc928fd8e;
                """
            )
            reverse = f"""
                brev.b32 t{slot}, w{slot};
                xor.b32 t{slot}, t{slot}, p{slot};
                shr.u32 t{slot}, t{slot}, {32 - bits};
            """
            rank_lines.append(
                f"""
                    {reverse}
                    shr.u32 w{slot}, p{slot}, 4;
                    and.b32 w{slot}, w{slot}, {low_index_mask:#x};
                    shl.b32 t{slot}, t{slot}, {width - 4};
                    or.b32 t{slot}, t{slot}, w{slot};
                """
            )
            if value_table_in_shared:
                load_lines.append(
                    f"""
                        add.u32 addr{slot}, t{slot}, $4;
                        ld.shared.u8 e{slot}, [addr{slot}];
                    """
                )
            else:
                load_lines.append(
                    f"""
                        cvt.u64.u32 addr{slot}, t{slot};
                        add.u64 addr{slot}, addr{slot}, $4;
                        ld.global.u8 e{slot}, [addr{slot}];
                    """
                )
            pack = f"shl.b32 e{slot}, e{slot}, {byte_shift};" if byte_shift else ""
            pack_lines.append(
                f"""
                    {pack}
                    or.b32 {target}, {target}, e{slot};
                """
            )
        decode_blocks.append(
            "\n".join(
                extract_lines + product_lines + rank_lines + load_lines + pack_lines
            )
        )

    address_reg = (
        ".reg .b32 addr0,addr1;" if value_table_in_shared else ".reg .b64 addr0,addr1;"
    )
    asm = (
        """
        {
            .reg .b32 out0,out1;
            .reg .b32 w0,w1,p0,p1,t0,t1,e0,e1;
        """
        + address_reg
        + """
            mov.b32 out0, 0;
            mov.b32 out1, 0;
        """
        + "\n".join(decode_blocks)
        + """
            mov.b32 $0, out0;
            mov.b32 $1, out1;
        """
        + "\n}"
    )
    address_value = (
        Int32(value_table_addr).ir_value(loc=loc, ip=ip)
        if value_table_in_shared
        else Int64(value_table_addr).ir_value(loc=loc, ip=ip)
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(win_a).ir_value(loc=loc, ip=ip),
            Uint32(win_b).ir_value(loc=loc, ip=ip),
            address_value,
        ],
        asm,
        "=r,=r,r,r," + ("r" if value_table_in_shared else "l"),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_decode_lut_fp16_to_half2x4(
    win_a,
    win_b,
    descriptor_addr,
    bits: int,
    *,
    loc=None,
    ip=None,
):
    """Decode eight 5- or 6-bit ``lut_fp16`` windows to FP16.

    The permutation produces a 16-bit rank. Its sign-symmetric magnitude
    selects one of 104 segments, followed by one native FP16 FMA using the
    interleaved ``(base, slope)`` segment pair.
    """

    bits = int(bits)
    if bits not in (5, 6):
        raise ValueError(f"lut_fp16 supports only K5/K6, got K{bits}")
    width = 16 - bits
    history_mask = (1 << width) - 1
    decode_blocks: list[str] = []
    for pair in range(4):
        extract_lines: list[str] = []
        graph_lines: list[str] = []
        coordinate_lines: list[str] = []
        decode_lines: list[str] = []
        for slot, index in enumerate((2 * pair, 2 * pair + 1)):
            if bits == 6:
                stream_shift = (7 - index) * bits
                if stream_shift < 32:
                    extract_lines.append(
                        f"shf.r.wrap.b32 w{slot}, $4, $5, {stream_shift}; "
                        f"and.b32 w{slot}, w{slot}, 0xffff;"
                    )
                else:
                    extract_lines.append(
                        f"shr.u32 w{slot}, $5, {stream_shift - 32}; "
                        f"and.b32 w{slot}, w{slot}, 0xffff;"
                    )
            else:
                source = "$5" if index < 4 else "$4"
                bit_shift = (3 - (index & 3)) * bits
                extract_lines.append(f"bfe.u32 w{slot}, {source}, {bit_shift}, 16;")
            graph_lines.append(
                f"""
                shr.u32 p{slot}, w{slot}, {bits};
                mad.lo.u32 p{slot}, p{slot}, 0x3fa7d929, 0xc928fd8e;
                brev.b32 t{slot}, w{slot};
                xor.b32 t{slot}, t{slot}, p{slot};
                shr.u32 t{slot}, t{slot}, {32 - bits};
                and.b32 rank{slot}, p{slot}, {history_mask:#x};
                shl.b32 t{slot}, t{slot}, {width};
                or.b32 rank{slot}, rank{slot}, t{slot};
                """
            )
            coordinate_lines.append(
                f"""
                and.b32 m{slot}, rank{slot}, 0x7fff;
                xor.b32 t{slot}, m{slot}, 0x7fff;
                setp.lt.u32 neg{slot}, rank{slot}, 0x8000;
                selp.u32 m{slot}, t{slot}, m{slot}, neg{slot};
                shl.b32 t{slot}, m{slot}, 1;
                sub.u32 v{slot}, 65535, t{slot};
                bfind.u32 e{slot}, v{slot};
                shl.b32 pow{slot}, 1, e{slot};
                sub.u32 delta{slot}, v{slot}, pow{slot};

                sub.s32 shift{slot}, e{slot}, 3;
                max.s32 shift{slot}, shift{slot}, 0;
                shr.u32 sub{slot}, delta{slot}, shift{slot};
                shl.b32 idxf{slot}, shift{slot}, 3;
                add.u32 idxf{slot}, idxf{slot}, sub{slot};
                shl.b32 max{slot}, sub{slot}, shift{slot};
                add.u32 max{slot}, max{slot}, pow{slot};
                shl.b32 t{slot}, 1, shift{slot};
                add.u32 max{slot}, max{slot}, t{slot};
                sub.u32 max{slot}, max{slot}, 1;
                sub.u32 xf{slot}, max{slot}, v{slot};
                shr.u32 xf{slot}, xf{slot}, 1;

                sub.s32 em1{slot}, e{slot}, 1;
                max.s32 em1{slot}, em1{slot}, 0;
                shl.b32 idxe{slot}, 1, em1{slot};
                sub.s32 t{slot}, delta{slot}, 1;
                shr.s32 t{slot}, t{slot}, 1;
                add.s32 idxe{slot}, idxe{slot}, t{slot};
                setp.le.u32 exact{slot}, e{slot}, 3;
                selp.u32 idx{slot}, idxe{slot}, idxf{slot}, exact{slot};
                selp.u32 x{slot}, 0, xf{slot}, exact{slot};
                """
            )
            decode_lines.append(
                f"""
                mul.wide.u32 addr{slot}, idx{slot}, 4;
                add.u64 addr{slot}, addr{slot}, $6;
                ld.global.nc.u32 d{slot}, [addr{slot}];
                mov.b32 {{base{slot}, slope{slot}}}, d{slot};
                cvt.rn.f16.u32 hx{slot}, x{slot};
                fma.rn.f16 hval{slot}, slope{slot}, hx{slot}, base{slot};
                xor.b32 sign{slot}, rank{slot}, 0x8000;
                and.b32 sign{slot}, sign{slot}, 0x8000;
                cvt.u16.u32 hsign{slot}, sign{slot};
                xor.b16 hval{slot}, hval{slot}, hsign{slot};
                """
            )
        decode_blocks.append(
            "\n".join(
                extract_lines
                + graph_lines
                + coordinate_lines
                + decode_lines
                + [f"mov.b32 ${pair}, {{hval0, hval1}};"]
            )
        )

    asm = (
        """
        {
            .reg .pred neg0,neg1,exact0,exact1;
            .reg .b16 base0,base1,slope0,slope1,hx0,hx1;
            .reg .b16 hval0,hval1,hsign0,hsign1;
            .reg .b32 w0,w1,p0,p1,t0,t1,rank0,rank1;
            .reg .b32 m0,m1,v0,v1,e0,e1,pow0,pow1,delta0,delta1;
            .reg .b32 shift0,shift1,sub0,sub1,idxf0,idxf1,max0,max1;
            .reg .b32 xf0,xf1,em10,em11,idxe0,idxe1,idx0,idx1,x0,x1;
            .reg .b32 d0,d1,sign0,sign1;
            .reg .b64 addr0,addr1;
        """
        + "\n".join(decode_blocks)
        + "\n}"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(win_a).ir_value(loc=loc, ip=ip),
            Uint32(win_b).ir_value(loc=loc, ip=ip),
            Int64(descriptor_addr).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def packed_decode_lut_e4m3_direct_to_e4m3x8(
    win_a,
    win_b,
    direct_lut_addr,
    bits: int = 3,
    rate_indexed: bool = True,
    *,
    in_shared: bool = False,
    loc=None,
    ip=None,
):
    """Decode eight 16-bit windows through a rate-specific direct E4M3 table.

    This is the lookup lower-bound primitive: one byte load per reconstructed
    weight. Every slot owns its own address/code registers so all eight
    gathers are in flight together. With ``in_shared``, the address names a
    caller-staged shared-memory table instead of a global-memory table.
    """

    bits = int(bits)
    if bits not in (2, 3, 4):
        raise ValueError(f"unsupported lut_e4m3 bit width {bits}; expected 2, 3, or 4")
    table_offset = ((bits - 2) << 16) if rate_indexed else 0
    extract_lines: list[str] = []
    load_lines: list[str] = []
    for index in range(8):
        source = "$3" if index < 4 else "$2"
        shift = (3 - (index & 3)) * bits
        # Accumulate into declared scratch registers, never directly into
        # the "=r" outputs: without early-clobber the outputs may alias the
        # window inputs that later iterations still read.
        if shift:
            extract_lines.append(f"bfe.u32 w{index}, {source}, {shift}, 16;")
        else:
            extract_lines.append(f"and.b32 w{index}, {source}, 0xffff;")
        if in_shared:
            load_lines.append(
                f"add.u32 w{index}, w{index}, $4; ld.shared.u8 w{index}, [w{index}];"
            )
        else:
            load_lines.append(
                f"""
                cvt.u64.u32 addr{index}, w{index};
                add.u64 addr{index}, addr{index}, $4;
                ld.global.u8 w{index}, [addr{index}];
                """
            )
    # Each load returns one zero-extended byte. Byte permutations pack four
    # independent results without a serial shift/or accumulator.
    pack_lines = [
        "prmt.b32 w0, w0, w1, 0x4040;",
        "prmt.b32 w2, w2, w3, 0x4040;",
        "prmt.b32 out0, w0, w2, 0x5410;",
        "prmt.b32 w4, w4, w5, 0x4040;",
        "prmt.b32 w6, w6, w7, 0x4040;",
        "prmt.b32 out1, w4, w6, 0x5410;",
    ]
    asm = (
        """
        {
            .reg .b32 out0,out1;
            .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7;
            .reg .b64 addr0,addr1,addr2,addr3,addr4,addr5,addr6,addr7;
    """
        + "\n".join(extract_lines)
        + "\n".join(load_lines)
        + "\n".join(pack_lines)
        + """
            mov.b32 $0, out0;
            mov.b32 $1, out1;
        }"""
    )
    table_base = (
        Int32(direct_lut_addr) + Int32(table_offset)
        if in_shared
        else Int64(direct_lut_addr) + Int64(table_offset)
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(win_a).ir_value(loc=loc, ip=ip),
            Uint32(win_b).ir_value(loc=loc, ip=ip),
            table_base.ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,r,r,r" if in_shared else "=r,=r,r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_trellis_to_half2x4(
    win_a, win_b, bits: int = 3, *, loc=None, ip=None
):
    """8 trellis windows -> 4 fp16x2 fragments for one compile-time bitrate.

    win_a/win_b: two 32-bit funnel windows extracted from the staged unit by the
    caller (_trellis_funnel32). Element/lane order:
      o0=(dec w0, dec w1) o1=(dec w2, dec w3) o2=(dec w4, dec w5) o3=(dec w6, dec w7)
    with w7=win_a, w6=win_a>>bits, ... w4=win_a>>(3*bits), and
         w3=win_b, w2=win_b>>bits, ... w0=win_b>>(3*bits).

    ``bits`` is a trace-time specialization in {2,3,4,5,6}, so the generated PTX
    keeps the same immediate shifts as exl3's templated dq4/dq8 implementations.
    The caller supplies independently valid four-weight windows for 5/6 bpw,
    whose full eight-weight span can cross three ring words.
    """
    bits = int(bits)
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError(
            f"unsupported trellis bitrate {bits}; expected 2, 3, 4, 5, or 6"
        )
    asm = """
        {
            .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7, lo, hi, M;
            mov.b32 M, __MCG__;
            and.b32 w7, $4, 0xffff;
            shr.u32 w6, $4, __B1__;  and.b32 w6, w6, 0xffff;
            shr.u32 w5, $4, __B2__;  and.b32 w5, w5, 0xffff;
            shr.u32 w4, $4, __B3__;  and.b32 w4, w4, 0xffff;
            and.b32 w3, $5, 0xffff;
            shr.u32 w2, $5, __B1__;  and.b32 w2, w2, 0xffff;
            shr.u32 w1, $5, __B2__;  and.b32 w1, w1, 0xffff;
            shr.u32 w0, $5, __B3__;  and.b32 w0, w0, 0xffff;
            mul.lo.u32 w0, w0, M;  lop3.b32 w0, w0, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w1, w1, M;  lop3.b32 w1, w1, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w2, w2, M;  lop3.b32 w2, w2, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w3, w3, M;  lop3.b32 w3, w3, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w4, w4, M;  lop3.b32 w4, w4, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w5, w5, M;  lop3.b32 w5, w5, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w6, w6, M;  lop3.b32 w6, w6, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w7, w7, M;  lop3.b32 w7, w7, __MASK__, __OR__, 0x6a;
            prmt.b32 lo, w0, w1, 0x5410;  prmt.b32 hi, w0, w1, 0x7632;  add.rn.f16x2 $0, lo, hi;
            prmt.b32 lo, w2, w3, 0x5410;  prmt.b32 hi, w2, w3, 0x7632;  add.rn.f16x2 $1, lo, hi;
            prmt.b32 lo, w4, w5, 0x5410;  prmt.b32 hi, w4, w5, 0x7632;  add.rn.f16x2 $2, lo, hi;
            prmt.b32 lo, w6, w7, 0x5410;  prmt.b32 hi, w6, w7, 0x7632;  add.rn.f16x2 $3, lo, hi;
        }
        """
    asm = _trellis_mcg_asm_constants(
        asm.replace("__B1__", str(bits))
        .replace("__B2__", str(2 * bits))
        .replace("__B3__", str(3 * bits))
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(win_a).ir_value(loc=loc, ip=ip),
            Uint32(win_b).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    o0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    o1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    o2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    o3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(o0), Uint32(o1), Uint32(o2), Uint32(o3)


@dsl_user_op
def packed_dequant_trellis_stream_to_half2x4(
    stream_lo, stream_hi, bits: int, *, loc=None, ip=None
):
    """Decode eight windows from a bitrate-strided 64-bit aligned stream.

    Bit zero of ``{stream_hi,stream_lo}`` begins w7; w6..w0 begin at successive
    ``bits`` offsets. This is required at 6 bpw, where four 16-bit windows span
    34 bits and cannot be represented by one 32-bit base window.
    """
    bits = int(bits)
    if bits not in (2, 3, 4, 5, 6):
        raise ValueError(
            f"unsupported trellis bitrate {bits}; expected 2, 3, 4, 5, or 6"
        )
    asm = """
        {
            .reg .b32 w0,w1,w2,w3,w4,w5,w6,w7, lo, hi, M;
            mov.b32 M, __MCG__;
            and.b32 w7, $4, 0xffff;
            __I6__
            __I5__
            __I4__
            __I3__
            __I2__
            __I1__
            __I0__
            mul.lo.u32 w0, w0, M;  lop3.b32 w0, w0, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w1, w1, M;  lop3.b32 w1, w1, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w2, w2, M;  lop3.b32 w2, w2, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w3, w3, M;  lop3.b32 w3, w3, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w4, w4, M;  lop3.b32 w4, w4, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w5, w5, M;  lop3.b32 w5, w5, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w6, w6, M;  lop3.b32 w6, w6, __MASK__, __OR__, 0x6a;
            mul.lo.u32 w7, w7, M;  lop3.b32 w7, w7, __MASK__, __OR__, 0x6a;
            prmt.b32 lo, w0, w1, 0x5410;  prmt.b32 hi, w0, w1, 0x7632;  add.rn.f16x2 $0, lo, hi;
            prmt.b32 lo, w2, w3, 0x5410;  prmt.b32 hi, w2, w3, 0x7632;  add.rn.f16x2 $1, lo, hi;
            prmt.b32 lo, w4, w5, 0x5410;  prmt.b32 hi, w4, w5, 0x7632;  add.rn.f16x2 $2, lo, hi;
            prmt.b32 lo, w6, w7, 0x5410;  prmt.b32 hi, w6, w7, 0x7632;  add.rn.f16x2 $3, lo, hi;
        }
        """
    for name, register, multiple in (
        ("__I6__", "w6", 1),
        ("__I5__", "w5", 2),
        ("__I4__", "w4", 3),
        ("__I3__", "w3", 4),
        ("__I2__", "w2", 5),
        ("__I1__", "w1", 6),
        ("__I0__", "w0", 7),
    ):
        shift = multiple * bits
        if shift < 32:
            instruction = (
                f"shf.r.wrap.b32 {register}, $4, $5, {shift}; "
                f"and.b32 {register}, {register}, 0xffff;"
            )
        else:
            instruction = (
                f"shr.u32 {register}, $5, {shift - 32}; "
                f"and.b32 {register}, {register}, 0xffff;"
            )
        asm = asm.replace(name, instruction)
    asm = _trellis_mcg_asm_constants(asm)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(stream_lo).ir_value(loc=loc, ip=ip),
            Uint32(stream_hi).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    o0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    o1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    o2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    o3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(o0), Uint32(o1), Uint32(o2), Uint32(o3)


@dsl_user_op
def packed_dequant_trellis3_to_half2x4(win_a, win_b, *, loc=None, ip=None):
    """Compatibility wrapper for the original 3-bpw primitive."""
    return packed_dequant_trellis_to_half2x4(win_a, win_b, 3, loc=loc, ip=ip)


@dsl_user_op
def _f16x2_to_bf16x2(p, *, loc=None, ip=None):
    """{hi_f16, lo_f16} -> {hi_bf16, lo_bf16}, RNE, lane order preserved."""
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()]),
        [Uint32(p).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 hlo, hhi;
            .reg .f32 flo, fhi;
            mov.b32 {hlo, hhi}, $1;
            cvt.f32.f16 flo, hlo;
            cvt.f32.f16 fhi, hhi;
            cvt.rn.bf16x2.f32 $0, fhi, flo;
        }
        """,
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(llvm.extractvalue(T.i32(), r, [0], loc=loc, ip=ip))


@cute.jit
def iq2_xxs_descriptor_pair(grids: Uint32, signs_scale: Uint32, half: Int32):
    """Normalize one K16 half in registers, retaining native K32 storage."""
    grids = grids >> (half * Int32(16))
    signs = signs_scale >> (half * Int32(14))
    lo = (grids & Uint32(255)) | ((signs & Uint32(127)) << 9)
    hi = ((grids >> 8) & Uint32(255)) | (((signs >> 7) & Uint32(127)) << 9)
    return lo | (hi << 16)


@dsl_user_op
def q8_0_pair_to_bf16x2(packed, base, *, loc=None, ip=None):
    """Multiply signed INT8 values by the FP16 block scale, rounding once."""
    result = llvm.inline_asm(
        T.i32(),
        [
            Uint32(packed).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .b16 dh;
            .reg .s32 q0, q1;
            .reg .f32 d, lo, hi;
            cvt.u16.u32 dh, $2;
            cvt.f32.f16 d, dh;
            bfe.s32 q0, $1, 0, 8;
            bfe.s32 q1, $1, 8, 8;
            cvt.rn.f32.s32 lo, q0;
            cvt.rn.f32.s32 hi, q1;
            mul.f32 lo, lo, d;
            mul.f32 hi, hi, d;
            cvt.rn.bf16x2.f32 $0, hi, lo;
        }""",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def iq2_xs_pair_to_bf16x2(
    descriptor,
    base,
    subscale,
    lut_addr,
    pair_offset,
    *,
    shared_lut=False,
    loc=None,
    ip=None,
):
    """Decode adjacent IQ2_XS weights with one BF16 rounding step."""
    load = (
        "cvt.u32.u64 shared_address, address; ld.shared.u32 word, [shared_address];"
        if shared_lut
        else "ld.global.nc.u32 word, [address];"
    )
    result = llvm.inline_asm(
        T.i32(),
        [
            Uint32(descriptor).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
            Uint32(subscale).ir_value(loc=loc, ip=ip),
            Int64(lut_addr).ir_value(loc=loc, ip=ip),
            Int32(pair_offset).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b16 dh;
            .reg .u32 grid, signs, parity, word, sign_pair, offset, shared_address;
            .reg .u64 address, byte_offset;
            .reg .f32 scale, nibble, lo, hi;
            cvt.u16.u32 dh, $2;
            cvt.f32.f16 scale, dh;
            cvt.rn.f32.u32 nibble, $3;
            add.f32 nibble, nibble, 0f3f000000;
            mul.f32 scale, scale, nibble;
            mul.f32 scale, scale, 0f3e800000;
            shr.u32 signs, $1, 9;
            popc.b32 parity, signs;
            mad.lo.u32 signs, parity, 128, signs;
            shr.u32 signs, signs, $5;
            and.b32 grid, $1, 0x1ff;
            mad.wide.u32 address, grid, 16, $4;
            shl.b32 offset, $5, 1;
            cvt.u64.u32 byte_offset, offset;
            add.u64 address, address, byte_offset;
            """
        + load
        + """
            shl.b32 lo, word, 16;
            and.b32 hi, word, 0xffff0000;
            mul.f32 lo, lo, scale;
            mul.f32 hi, hi, scale;
            cvt.rn.satfinite.bf16x2.f32 $0, hi, lo;
            mul.lo.u32 sign_pair, signs, 0x40008000;
            lop3.b32 $0, $0, sign_pair, 0x80008000, 0x78;
        }
        """,
        "=r,r,r,r,l,r",
        has_side_effects=bool(shared_lut),
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def iq2_xs_descriptor_pair_to_bf16x2x2(
    descriptors, base, subscale, lut_addr, pair_offset, *, loc=None, ip=None
):
    """Decode matching pairs from two adjacent descriptors using one FP32 scale."""
    decode = []
    for i in range(2):
        decode.append(f"""
            bfe.u32 descriptor, descriptor_word, {i * 16}, 16;
            shr.u32 signs, descriptor, 9;
            popc.b32 parity, signs;
            mad.lo.u32 signs, parity, 128, signs;
            shr.u32 signs, signs, pair_start;
            and.b32 grid, descriptor, 0x1ff;
            mad.lo.u32 address, grid, 16, table;
            add.u32 address, address, offset;
            ld.shared.u32 word, [address];
            shl.b32 lo, word, 16;
            and.b32 hi, word, 0xffff0000;
            mul.f32 lo, lo, scale;
            mul.f32 hi, hi, scale;
            cvt.rn.satfinite.bf16x2.f32 ${i}, hi, lo;
            mul.lo.u32 sign_pair, signs, 0x40008000;
            lop3.b32 ${i}, ${i}, sign_pair, 0x80008000, 0x78;
        """)
    asm = (
        """{
        .reg .b16 dh;
        .reg .u32 descriptor, descriptor_word, grid, signs, parity, word, sign_pair, offset, address, table, pair_start;
        .reg .f32 scale, nibble, lo, hi;
        mov.u32 descriptor_word, $2;
        mov.u32 pair_start, $6;
        cvt.u16.u32 dh, $3;
        cvt.f32.f16 scale, dh;
        cvt.rn.f32.u32 nibble, $4;
        add.f32 nibble, nibble, 0f3f000000;
        mul.f32 scale, scale, nibble;
        mul.f32 scale, scale, 0f3e800000;
        cvt.u32.u64 table, $5;
        shl.b32 offset, $6, 1;
    """
        + "".join(decode)
        + "\n}"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(descriptors).ir_value(loc=loc, ip=ip),
            Uint32(base).ir_value(loc=loc, ip=ip),
            Uint32(subscale).ir_value(loc=loc, ip=ip),
            Int64(lut_addr).ir_value(loc=loc, ip=ip),
            Int32(pair_offset).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,r,r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )


def _packed_decode_iq2_xs_to_bfloat2x4(
    q_row0,
    q_row1,
    base_pair,
    subscale_pair,
    execution_lut_addr,
    pair_byte_offset,
    *,
    shared_lut=False,
    selector_lut=False,
    loc=None,
    ip=None,
):
    """Decode one IQ2_XS MMA RHS fragment with one-round BF16 conversion."""

    extracts = (
        "and.b32 d0, $4, 0xffff;",
        "shr.u32 d1, $4, 16;",
        "and.b32 d2, $5, 0xffff;",
        "shr.u32 d3, $5, 16;",
    )
    rows = (0, 0, 1, 1)
    table_stride = 8 if selector_lut else 16
    load_type = "u16" if selector_lut else "u32"
    decode = []
    for index, (extract, row) in enumerate(zip(extracts, rows, strict=True)):
        load = (
            f"""
            cvt.u32.u64 sa{index}, $8;
            mad.lo.u32 sa{index}, d{index}, {table_stride}, sa{index};
            add.u32 sa{index}, sa{index}, so;
            ld.shared.{load_type} u{index}, [sa{index}];
            """
            if shared_lut
            else f"""
            mad.wide.u32 a{index}, d{index}, {table_stride}, $8;
            add.u64 a{index}, a{index}, po;
            ld.global.nc.{load_type} u{index}, [a{index}];
            """
        )
        conversion = (
            f"prmt.b32 ${index}, t{row}0, t{row}1, u{index};"
            if selector_lut
            else f"""
            shl.b32 fl{index}, u{index}, 16;
            and.b32 fh{index}, u{index}, 0xffff0000;
            mul.f32 fl{index}, fl{index}, s{row};
            mul.f32 fh{index}, fh{index}, s{row};
            cvt.rn.satfinite.bf16x2.f32 ${index}, fh{index}, fl{index};
            """
        )
        decode.append(
            f"""
            {extract}
            shr.u32 signs{index}, d{index}, 9;
            popc.b32 parity{index}, signs{index};
            mad.lo.u32 signs{index}, parity{index}, 128, signs{index};
            shr.u32 signs{index}, signs{index}, $9;
            and.b32 d{index}, d{index}, 0x1ff;
            {load}
            {conversion}
            mul.lo.u32 sign_pair{index}, signs{index}, 0x40008000;
            lop3.b32 ${index}, ${index}, sign_pair{index}, 0x80008000, 0x78;
            """
        )
    selector_setup = ""
    if selector_lut:
        selector_setup = """
            .reg .f32 m00, m01, m02, m10, m11, m12;
            .reg .b32 t00, t01, t10, t11;
        """
        for row in range(2):
            selector_setup += f"""
            mul.f32 m{row}0, s{row}, 0f41000000;
            mul.f32 m{row}1, s{row}, 0f41c80000;
            mul.f32 m{row}2, s{row}, 0f422c0000;
            cvt.rn.satfinite.bf16x2.f32 t{row}0, m{row}1, m{row}0;
            cvt.rn.satfinite.bf16x2.f32 t{row}1, m{row}2, m{row}2;
            """
    pair_shift = 0 if selector_lut else 1
    asm = (
        """
        {
            .reg .b16 dh0, dh1;
            .reg .b32 d0, d1, d2, d3, n0, n1;
            .reg .u32 u0, u1, u2, u3, xl0, xl1, xl2, xl3,
                      xh0, xh1, xh2, xh3;
            .reg .u32 signs0, signs1, signs2, signs3,
                      parity0, parity1, parity2, parity3,
                      sign_pair0, sign_pair1, sign_pair2, sign_pair3;
            .reg .u64 po, a0, a1, a2, a3;
            .reg .u32 sa0, sa1, sa2, sa3, so;
            .reg .f32 s0, s1, fl0, fl1, fl2, fl3,
                      fh0, fh1, fh2, fh3;
            mov.b32 {dh0, dh1}, $6;
            cvt.f32.f16 s0, dh0;
            cvt.f32.f16 s1, dh1;
            bfe.u32 n0, $7, 0, 4;
            bfe.u32 n1, $7, 8, 4;
            cvt.rn.f32.u32 fl0, n0;
            cvt.rn.f32.u32 fl1, n1;
            add.f32 fl0, fl0, 0f3f000000;
            add.f32 fl1, fl1, 0f3f000000;
            mul.f32 s0, s0, fl0;
            mul.f32 s1, s1, fl1;
            mul.f32 s0, s0, 0f3e800000;
            mul.f32 s1, s1, 0f3e800000;
        """
        + f"shl.b32 so, $9, {pair_shift};\n            cvt.u64.u32 po, so;\n"
        + selector_setup
        + "".join(decode)
        + "\n}"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Uint32(q_row0).ir_value(loc=loc, ip=ip),
            Uint32(q_row1).ir_value(loc=loc, ip=ip),
            Uint32(base_pair).ir_value(loc=loc, ip=ip),
            Uint32(subscale_pair).ir_value(loc=loc, ip=ip),
            Int64(execution_lut_addr).ir_value(loc=loc, ip=ip),
            Int32(pair_byte_offset).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r,r,r,l,r",
        has_side_effects=bool(shared_lut),
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def packed_decode_iq2_xs_to_bfloat2x4(
    q_row0,
    q_row1,
    base_pair,
    subscale_pair,
    execution_lut_addr,
    pair_byte_offset,
    *,
    shared_lut=False,
    selector_lut=False,
    loc=None,
    ip=None,
):
    return _packed_decode_iq2_xs_to_bfloat2x4(
        q_row0,
        q_row1,
        base_pair,
        subscale_pair,
        execution_lut_addr,
        pair_byte_offset,
        shared_lut=shared_lut,
        selector_lut=selector_lut,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def packed_decode_lut_fp16_to_bfloat2x4(
    win_a, win_b, descriptor_addr, bits: int, *, loc=None, ip=None
):
    """Decode 5- or 6-bit ``lut_fp16`` values and convert them to BF16 pairs."""

    h0, h1, h2, h3 = packed_decode_lut_fp16_to_half2x4(
        win_a,
        win_b,
        descriptor_addr,
        bits,
        loc=loc,
        ip=ip,
    )
    return (
        _f16x2_to_bf16x2(h0, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h1, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h2, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h3, loc=loc, ip=ip),
    )


@dsl_user_op
def packed_dequant_trellis_to_bfloat2x4(
    win_a, win_b, bits: int = 3, *, loc=None, ip=None
):
    """Diagnostic bf16 conversion of the generalized fp16 trellis decode.

    Production trellis kernels should use the fp16 primitive and fp16 MMA arm:
    that is bit-faithful to exl3 and saves 16 conversion instructions per
    fragment.  This wrapper remains available for the D4 bf16 A/B gate.
    """
    h0, h1, h2, h3 = packed_dequant_trellis_to_half2x4(
        win_a, win_b, bits, loc=loc, ip=ip
    )
    return (
        _f16x2_to_bf16x2(h0, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h1, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h2, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h3, loc=loc, ip=ip),
    )


@dsl_user_op
def packed_dequant_trellis_stream_to_bfloat2x4(
    stream_lo, stream_hi, bits: int, *, loc=None, ip=None
):
    """Diagnostic bf16 conversion of the 64-bit-stream trellis decode."""
    h0, h1, h2, h3 = packed_dequant_trellis_stream_to_half2x4(
        stream_lo, stream_hi, bits, loc=loc, ip=ip
    )
    return (
        _f16x2_to_bf16x2(h0, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h1, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h2, loc=loc, ip=ip),
        _f16x2_to_bf16x2(h3, loc=loc, ip=ip),
    )


@dsl_user_op
def packed_dequant_trellis3_to_bfloat2x4(win_a, win_b, *, loc=None, ip=None):
    """Compatibility wrapper for the former 3-bpw bf16 primitive."""
    return packed_dequant_trellis_to_bfloat2x4(win_a, win_b, 3, loc=loc, ip=ip)
