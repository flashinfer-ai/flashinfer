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

"""Transformed K/V SMEM resource for mixed-dtype FMHA decode."""

import cutlass.cute as cute
from cutlass import Int32


@cute.jit
def _cvt_e2m1_word_to_e4m3_words(
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
