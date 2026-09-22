# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Final joint normalization of native sparse MLA partial outputs."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64


class FinishSparseMla:
    def __init__(self, independent_sources: bool):
        self.independent_sources = independent_sources

    @cute.jit
    def __call__(
        self, part_s, part_c, lse_s, lse_c, count_s, count_c, sinks, out, lse, stream
    ):
        self.finish(
            part_s, part_c, lse_s, lse_c, count_s, count_c, sinks, out, lse
        ).launch(
            grid=((out.shape[0] * out.shape[1] + 3) // 4, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def finish(self, part_s, part_c, lse_s, lse_c, count_s, count_c, sinks, out, lse):
        block, _, _ = cute.arch.block_idx()
        tid = cute.arch.thread_idx()[0]
        lane = tid % 32
        # One warp handles one output head. Group four heads per CTA to
        # amortize launch overhead and avoid repeating normalization 128 times.
        row = Int64(block) * 4 + Int64(tid // 32)
        q = row // out.shape[1]
        h = row % out.shape[1]
        if row < out.shape[0] * out.shape[1]:
            have_s = Int32(count_s[q]) > 0
            have_c = (Int32(count_c[q]) > 0) if self.independent_sources else False
            l0 = Float32(lse_s[q, h]) if have_s else Float32(-Float32.inf)
            l1 = Float32(lse_c[q, h]) if have_c else Float32(-Float32.inf)
            ws = Float32(0)
            wc = Float32(0)
            public_lse = Float32(-Float32.inf)
            if have_s or have_c:
                maximum = cute.math.max(l0, l1)
                total = cute.math.exp2(l0 - maximum, approx=True) + cute.math.exp2(
                    l1 - maximum, approx=True
                )
                public_lse = (maximum + cute.math.log2(total, approx=True)) * Float32(
                    0.6931471805599453
                )
                sink = Float32(sinks[h]) * Float32(1.4426950408889634)
                if sink != Float32(Float32.inf):
                    normalizer_max = cute.math.max(maximum, sink)
                    a = cute.math.exp2(l0 - normalizer_max, approx=True)
                    b = cute.math.exp2(l1 - normalizer_max, approx=True)
                    denominator = (
                        a + b + cute.math.exp2(sink - normalizer_max, approx=True)
                    )
                    ws = a / denominator
                    wc = b / denominator
            for chunk in cutlass.range_constexpr(8):
                column = lane * 2 + chunk * 64
                values = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
                values[0], values[1] = Float32(0), Float32(0)
                if have_s:
                    offset = q * part_s.stride[0] + h * part_s.stride[1] + column
                    pair = cutlass.Pointer(
                        part_s.iterator + offset, dtype=part_s.element_type
                    ).load(count=2, alignment=4)
                    for j in cutlass.range_constexpr(2):
                        values[j] = Float32(pair[j]) * ws
                if have_c:
                    offset = q * part_c.stride[0] + h * part_c.stride[1] + column
                    pair = cutlass.Pointer(
                        part_c.iterator + offset, dtype=part_c.element_type
                    ).load(count=2, alignment=4)
                    for j in cutlass.range_constexpr(2):
                        values[j] += Float32(pair[j]) * wc
                packed = cutlass.Array(
                    out.element_type, 2, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(2):
                    packed[j] = values[j].to(out.element_type)
                offset = q * out.stride[0] + h * out.stride[1] + column
                cutlass.Pointer(out.iterator + offset, dtype=out.element_type).store(
                    packed.load(0, 2), alignment=4
                )
            if lane == 0:
                lse[q, h] = public_lse
