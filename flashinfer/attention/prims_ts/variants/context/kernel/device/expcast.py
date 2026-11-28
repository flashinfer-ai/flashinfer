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

"""ExpCast and packed f16x2 / FP8 intrinsics of the context softmax."""

import cutlass.cute as cute
import cutlass
from cutlass import Float16, Float32, Int32, Uint32
from cutlass.cutlass_dsl import Constexpr

# ExpCast-FP8 (VC-Attention). An E4M3 byte read as an integer equals
# ``8*log2(v) + 56 + eps``. With ``u`` the log2-domain probability and the
# paper's 2^8 P scaling, ``code = round(8*u + VC_EXPCAST_CODE_BIAS)`` replaces
# exp2 and the E4M3 cast by one FMA. beta is the paper's centering of
# Mitchell's error.
E4M3_EXP_BIAS = 7
E4M3_CODES_PER_OCTAVE = 8
VC_EXPCAST_BETA = -0.35
VC_EXPCAST_P_SCALE_LOG2 = 8
VC_EXPCAST_CODE_BIAS = (
    E4M3_CODES_PER_OCTAVE * (E4M3_EXP_BIAS + VC_EXPCAST_P_SCALE_LOG2) + VC_EXPCAST_BETA
)


@cute.jit
def add_f16x2_to_f32x2(
    acc: tuple[Float32, Float32], word: Int32
) -> tuple[Float32, Float32]:
    """Add the two f16 halves of a word to two fp32 accumulators with one
    add.f32x2.f16x2.f32x2 instruction."""
    acc_i64 = cutlass.Vector.from_elements((acc[0], acc[1]), Float32).bitcast(
        cutlass.Int64
    )[0]
    res_i64 = cute.arch.inline_ptx(
        "add.f32x2.f16x2.f32x2 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Int64],
        read_only_args=[word, acc_i64],
    )
    res = cutlass.Vector.from_elements((res_i64,), cutlass.Int64).bitcast(Float32)
    return (res[0], res[1])


@cute.jit
def exp2_f16x2_word(x0: Float32, x1: Float32) -> Int32:
    """exp2 of two fp32 values as one f16x2 word, x0 in the low half."""
    x_f16x2 = cutlass.Vector.from_elements((x0, x1), Float32).to(cutlass.Float16)
    return cute.arch.exp2_packed_f16x2(x_f16x2.bitcast(Int32)[0])


@cute.jit
def expcast_e4m3_quad_relu(
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    acc01: Int32,
    acc23: Int32,
    init: Constexpr[bool],
) -> tuple[Int32, Int32, Int32]:
    """ExpCast four unclamped codes into one packed E4M3 word and accumulate
    their values into two f16x2 row-sum registers.

    With ``init`` the quad starts a chain and the accumulators are ignored. The
    returned sums are then the quantized values themselves.

    ``c = 8 * (s * scale - m) + VC_EXPCAST_CODE_BIAS``

    ``cvt.rn.relu.f16x2.f32`` packs a pair and clamps underflow to code 0, an
    f16 FMA adds 1024 so the low byte of each half is the rounded code, one
    ``prmt`` gathers the four bytes, and the E4M3-pair to f16x2 conversion
    with one f16 FMA per pair accumulates the quantized values.
    """
    sum_pair = (
        "mov.b32 {$w1}, v01;\n            mov.b32 {$w2}, v23;"
        if init
        else "fma.rn.f16x2 {$w1}, v01, one2, {$r4};\n"
        "            fma.rn.f16x2 {$w2}, v23, one2, {$r5};"
    )
    return cute.arch.inline_ptx(
        """
        {
            .reg .b32 h01, h23, y01, y23, v01, v23, one2, k1024;
            .reg .b16 c01, c23;
            mov.b32 one2, 0x3C003C00;
            mov.b32 k1024, 0x64006400;
            cvt.rn.relu.f16x2.f32 h01, {$r1}, {$r0};
            cvt.rn.relu.f16x2.f32 h23, {$r3}, {$r2};
            fma.rn.f16x2 y01, h01, one2, k1024;
            fma.rn.f16x2 y23, h23, one2, k1024;
            prmt.b32 {$w0}, y01, y23, 0x6420;
            mov.b32 {c01, c23}, {$w0};
            cvt.rn.f16x2.e4m3x2 v01, c01;
            cvt.rn.f16x2.e4m3x2 v23, c23;
            """
        + sum_pair
        + """
        }""",
        write_only_types=[Int32, Int32, Int32],
        read_only_args=[c0, c1, c2, c3, acc01, acc23],
    )


@cute.jit
def f16x2_sum4(a0: Int32, a1: Int32, a2: Int32, a3: Int32) -> Float32:
    """fp32 sum of the eight fp16 halves of four f16x2 registers."""
    f = []
    for word in (a0, a1, a2, a3):
        halves = cutlass.Vector.from_elements((word,), Int32).bitcast(Float16)
        f.append(halves[0].to(Float32))
        f.append(halves[1].to(Float32))
    return ((f[0] + f[1]) + (f[2] + f[3])) + ((f[4] + f[5]) + (f[6] + f[7]))


@cute.jit
def f16x2_to_f32x2(word: Int32) -> tuple[Float32, Float32]:
    """The two f16 halves of a word as fp32."""
    halves = cutlass.Vector.from_elements((word,), Int32).bitcast(cutlass.Float16)
    halves_f32 = halves.to(Float32)
    return (halves_f32[0], halves_f32[1])


@cute.jit
def pack_f16x2x2_to_fp8_e4m3(w0: Int32, w1: Int32) -> Int32:
    """Pack two f16x2 words into one e4m3x4 word, w0 in the low half."""
    return cute.arch.inline_ptx(
        """{
            .reg .b16 lo, hi;
            cvt.rn.satfinite.e4m3x2.f16x2 lo, {$r0};
            cvt.rn.satfinite.e4m3x2.f16x2 hi, {$r1};
            mov.b32 {$w0}, {lo, hi};
        }""",
        write_only_types=[Int32],
        read_only_args=[w0, w1],
    )


@cute.jit
def split_bf16_hi_lo_word(value: Float32) -> Int32:
    """Pack ``value`` as two bf16 halves: low 16 bits = truncated high part,
    high 16 bits = truncated remainder ``value - hi``."""
    bits = cutlass.Vector.from_elements((value,), Float32).bitcast(Uint32)[0]
    hi_bits = bits >> Uint32(16)
    hi = cutlass.Vector.from_elements((hi_bits << Uint32(16),), Uint32).bitcast(
        Float32
    )[0]
    lo_bits = cutlass.Vector.from_elements((value - hi,), Float32).bitcast(Uint32)[0]
    word = hi_bits | (lo_bits & Uint32(0xFFFF0000))
    return cutlass.Vector.from_elements((word,), Uint32).bitcast(Int32)[0]
