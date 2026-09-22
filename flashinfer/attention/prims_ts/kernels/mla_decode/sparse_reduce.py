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

"""Finish split-KV or independently scaled source outputs, including sinks/LSE."""

from .helpers.constants import LN_2, LOG2_E

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.utils import SmemAllocator
from cutlass.experimental import primitives as prims


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
                    LN_2
                )
                sink = Float32(sinks[h]) * Float32(LOG2_E)
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


class FinishSparseMlaSplit:
    def __init__(self, splits: int, direct: bool = False):
        self.splits = splits
        self.direct = direct

    @cute.jit
    def __call__(self, partial, partial_lse, counts, sinks, out, lse, stream):
        self.reduce(partial, partial_lse, counts, sinks, out, lse).launch(
            grid=(out.shape[0], out.shape[1], 1),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=self.direct,
        )

    @cute.kernel
    def reduce(self, partial, partial_lse, counts, sinks, out, lse):
        row, head, _ = cute.arch.block_idx()
        row = Int64(row)
        tid = cute.arch.thread_idx()[0]
        shared = SmemAllocator().allocate_tensor(
            Float32, cute.make_layout(self.splits + 1), byte_alignment=16
        )
        if cutlass.const_expr(self.direct):
            # The producer may still be retiring when this setup begins.
            # Wait for the whole prerequisite grid and its global stores
            # before reading any partial, including neutral skipped splits.
            prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
        # One warp loads contiguous split statistics and computes weights.
        # Sharing them avoids repeating normalization in every output thread.
        if tid < 32:
            values = cutlass.Array(
                Float32, (self.splits + 31) // 32, space=cutlass.AddressSpace.rmem
            )
            have_values = (
                cutlass.Boolean(True) if self.direct else Int32(counts[row]) > 0
            )
            max_val = Float32(-Float32.inf)
            for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                split = tid + chunk * 32
                values[chunk] = Float32(-Float32.inf)
                if have_values and split < self.splits:
                    values[chunk] = Float32(partial_lse[row, head, split])
                max_val = cute.math.max(max_val, values[chunk])
            for shift in cutlass.range_constexpr(5):
                other = Float32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=max_val,
                        offset=1 << shift,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                )
                max_val = cute.math.max(max_val, other)
            have_values = have_values & (max_val != Float32(-Float32.inf))
            public_lse = Float32(-Float32.inf)
            if have_values:
                total = Float32(0)
                for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                    total += cute.math.exp2(values[chunk] - max_val, approx=True)
                for shift in cutlass.range_constexpr(5):
                    total += Float32(
                        prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=total,
                            offset=1 << shift,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.BFLY,
                        )
                    )
                public_lse = (max_val + cute.math.log2(total, approx=True)) * Float32(
                    LN_2
                )
                sink = Float32(sinks[head]) * Float32(LOG2_E)
                if sink == Float32(Float32.inf):
                    for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                        values[chunk] = Float32(0)
                else:
                    norm_max = cute.math.max(max_val, sink)
                    denominator = total * cute.math.exp2(
                        max_val - norm_max, approx=True
                    ) + cute.math.exp2(sink - norm_max, approx=True)
                    for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                        values[chunk] = (
                            cute.math.exp2(values[chunk] - norm_max, approx=True)
                            / denominator
                        )
            else:
                for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                    values[chunk] = Float32(0)
            for chunk in cutlass.range_constexpr((self.splits + 31) // 32):
                split = tid + chunk * 32
                if split < self.splits:
                    shared[split] = values[chunk]
            if tid == 0:
                shared[self.splits] = Float32(have_values)
                lse[row, head] = public_lse
        cute.arch.sync_threads()
        have_values = Float32(shared[self.splits]) > Float32(0)
        # Four independent accumulators expose load/FMA overlap. Pair loads
        # keep the existing four-byte alignment contract for BF16 partials.
        column = tid * 4
        acc = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(4):
            acc[j] = Float32(0)
        if have_values:
            for split in cutlass.range_constexpr(self.splits):
                weight = Float32(shared[split])
                if weight > Float32(0):
                    offset = (
                        row * partial.stride[0]
                        + head * partial.stride[1]
                        + split * partial.stride[2]
                        + column
                    )
                    # Pair loads preserve the existing four-byte pointer ABI.
                    first = cutlass.Pointer(
                        partial.iterator + offset, dtype=partial.element_type
                    ).load(count=2, alignment=4)
                    second = cutlass.Pointer(
                        partial.iterator + offset + 2, dtype=partial.element_type
                    ).load(count=2, alignment=4)
                    for j in cutlass.range_constexpr(2):
                        acc[j] += Float32(first[j]) * weight
                        acc[j + 2] += Float32(second[j]) * weight
        for j in cutlass.range_constexpr(4):
            out[row, head, column + j] = acc[j].to(out.element_type)
