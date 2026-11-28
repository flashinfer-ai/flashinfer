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

"""Register packing and descriptor helpers shared by the PrimTS kernels."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float32, Int32, Int64
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
from ....cute_dsl.availability import is_rubin_cute_dsl_available


@cute.jit
def cvt_e2m1_word_to_e4m3_words(
    packed_word: Int32, sf_byte: Int32
) -> tuple[Int32, Int32]:
    """Scale eight dense-packed E2M1 values and return two E4M3 words."""
    sf_pair = (sf_byte & Int32(0xFF)) * Int32(0x101)
    return cute.arch.inline_ptx(
        """
        {
            .reg .b16 sf;
            .reg .b32 sfFp16x2;
            .reg .b8 b0, b1, b2, b3;
            .reg .b32 h0, h1, h2, h3;
            .reg .b16 e0, e1, e2, e3;
            mov.b32 {sf, _}, {$r1};
            cvt.rn.f16x2.e4m3x2 sfFp16x2, sf;
            mov.b32 {b0, b1, b2, b3}, {$r0};
            cvt.rn.f16x2.e2m1x2 h0, b0;
            cvt.rn.f16x2.e2m1x2 h1, b1;
            cvt.rn.f16x2.e2m1x2 h2, b2;
            cvt.rn.f16x2.e2m1x2 h3, b3;
            mul.rn.f16x2 h0, h0, sfFp16x2;
            mul.rn.f16x2 h1, h1, sfFp16x2;
            mul.rn.f16x2 h2, h2, sfFp16x2;
            mul.rn.f16x2 h3, h3, sfFp16x2;
            cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;
            cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;
            cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;
            cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;
            mov.b32 {$w0}, {e0, e1};
            mov.b32 {$w1}, {e2, e3};
        }
        """,
        write_only_types=[Int32, Int32],
        read_only_args=[packed_word, sf_pair],
    )


@cute.jit
def freeze_smem_descriptor(desc):
    """Copy an SMEM descriptor through a register to prevent rematerialization."""
    return cute.arch.inline_ptx(
        "mov.b64 {$w0}, {$r0};",
        write_only_types=[Int64],
        read_only_args=[desc],
    )


def has_packed_e4m3_e2m1_mul() -> bool:
    """Whether the compile target has the packed E4M3 x E2M1 multiply.

    PTX 9.4 (CUDA 13.4) adds it for SM100 and SM103; ptxas rejects it for
    SM107, which takes the f16x2 path instead.
    """
    if not cutlass.target_version(min_version="13.4"):
        return False
    if not is_rubin_cute_dsl_available():
        return True
    arch = BaseDSL._get_dsl().get_arch_enum()
    return not (Arch.sm_107 <= arch <= Arch.sm_107f)


@cute.jit
def pack_float2_to_bf16(v0: Float32, v1: Float32) -> Int32:
    """Pack two FP32 values into one BF16x2 register."""
    return (
        cutlass.Vector.from_elements((v0, v1), Float32).to(BFloat16).bitcast(Int32)[0]
    )


@cute.jit
def pack_float2_to_fp16(v0: Float32, v1: Float32) -> Int32:
    """Pack two FP32 values into one FP16x2 register."""
    return cutlass.Vector.from_elements((v0, v1), Float32).to(Float16).bitcast(Int32)[0]


@cute.jit
def pack_float4_to_fp8_e4m3(
    v0: Float32, v1: Float32, v2: Float32, v3: Float32
) -> Int32:
    """Pack four FP32 values into one FP8 E4M3x4 register.

    Two b16 conversions are joined by mov.b32, which ptxas emits as F2FP pairs
    with no PRMT.
    """
    return cute.arch.inline_ptx(
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        "  cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};\n"
        "  cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};\n"
        "  mov.b32 {$w0}, {lo, hi};\n"
        "}",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
    )
