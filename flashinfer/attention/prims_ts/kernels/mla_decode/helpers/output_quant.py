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

"""Final sparse MLA output: conjugate RoPE and block E4M3 quantization."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.experimental import primitives as prims

from .ops import pack_float4_to_fp8_e4m3


@dataclass(frozen=True)
class MlaOutputQuant:
    block_size: int
    ue8m0: bool
    num_heads: int
    num_groups: int

    @cute.jit
    def rotate(
        self, args, values, row, column, value_stride: cutlass.Constexpr[int] = 1
    ):
        """Apply conjugate RoPE to an FP32 fragment in its current lane layout."""
        _, positions, cos_sin, _ = args
        token = Int64(row) // self.num_heads
        count = cutlass.const_expr(values.shape[0])
        fragment = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(count):
            fragment[i] = values[i]

        # The cache contains forward sin; conjugation changes its sign here.
        if column + (count - 1) * value_stride >= 448:
            position = positions[token]
            if cutlass.const_expr(value_stride == 1):
                for i in cutlass.range_constexpr(0, count, 2):
                    d = column + i
                    if d >= 448:
                        c = cos_sin[position, (d - 448) // 2]
                        s = cos_sin[position, (d - 448) // 2 + 32]
                        x, y = fragment[i], fragment[i + 1]
                        fragment[i] = x * c + y * s
                        fragment[i + 1] = y * c - x * s
            else:
                for i in cutlass.range_constexpr(count):
                    d = column + i * value_stride
                    if d >= 448:
                        peer = prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=fragment[i],
                            offset=1,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.BFLY,
                        )
                        c = cos_sin[position, (d - 448) // 2]
                        s = cos_sin[position, (d - 448) // 2 + 32]
                        if d % 2 == 0:
                            fragment[i] = fragment[i] * c + peer * s
                        else:
                            fragment[i] = fragment[i] * c - peer * s

        return fragment

    @cute.jit
    def store(self, args, values, row, column, lane_stride: cutlass.Constexpr[int] = 1):
        """Finalize contiguous lane fragments after all attention reductions."""
        rotated = self.rotate(args, values, row, column)
        self.store_rotated(args, rotated, row, column, lane_stride=lane_stride)

    @cute.jit
    def store_rotated(
        self,
        args,
        fragment,
        row,
        column,
        value_stride: cutlass.Constexpr[int] = 1,
        lane_stride: cutlass.Constexpr[int] = 1,
        amax_override=None,
        scale_owner=None,
    ):
        """Store rotated values; an optional CTA reduction supplies block amax."""
        out, _, _, scales = args
        token = Int64(row) // self.num_heads
        head = Int64(row) % self.num_heads
        lane = cute.arch.thread_idx()[0] % 32
        count = cutlass.const_expr(fragment.shape[0])
        per_block = cutlass.const_expr(
            min(count, self.block_size)
            if value_stride == 1
            else min(count, self.block_size // 32)
        )
        lanes = cutlass.const_expr(min(32, self.block_size // per_block))
        for start in cutlass.range_constexpr(0, count, per_block):
            # Match the model/FlashMLA quantizer, including zero-valued blocks.
            amax = Float32(1e-4)
            for i in cutlass.range_constexpr(per_block):
                amax = cute.math.max(amax, cute.math.abs(fragment[start + i]))
            if cutlass.const_expr(amax_override is not None):
                amax = amax_override
            else:
                for step in cutlass.range_constexpr(lanes.bit_length() - 1):
                    peer = prims.shfl_sync(
                        thread_mask=cute.arch.activemask(),
                        val=amax,
                        offset=(1 << step) * lane_stride,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                    amax = cute.math.max(amax, peer)
            scale = amax * Float32(1.0 / 448.0)
            if cutlass.const_expr(self.ue8m0):
                exponent = (scale.bitcast(Uint32) + Uint32(0x007FFFFF)) >> 23
                scale = (exponent << 23).bitcast(Float32)
            inv_scale = cute.math.rcp(scale, approx=True)
            block = (column + start * value_stride) // self.block_size
            group = head // (self.num_heads // self.num_groups)
            block_in_group = (
                head % (self.num_heads // self.num_groups) * (512 // self.block_size)
                + block
            )
            owns_scale = (
                lane // lane_stride % lanes == 0
                if cutlass.const_expr(scale_owner is None)
                else scale_owner
            )
            if owns_scale:
                if cutlass.const_expr(self.ue8m0):
                    word = block_in_group // 4
                    byte = block_in_group % 4
                    offset = token + group * scales.stride[1] + word * scales.stride[2]
                    byte_ptr = cutlass.inttoptr(
                        (scales.iterator.raw_ptr() + offset).toint(Int64) + byte,
                        mem_space=1,
                        dtype=cutlass.Uint8,
                    )
                    byte_ptr.store(exponent.to(cutlass.Uint8))
                else:
                    scales[token, group, block_in_group] = scale

            if cutlass.const_expr(value_stride == 1):
                for i in cutlass.range_constexpr(0, per_block, 4):
                    v = start + i
                    packed = pack_float4_to_fp8_e4m3(
                        fragment[v] * inv_scale,
                        fragment[v + 1] * inv_scale,
                        fragment[v + 2] * inv_scale,
                        fragment[v + 3] * inv_scale,
                    )
                    ptr = cutlass.inttoptr(
                        (out.iterator.raw_ptr() + Int64(row) * 512 + column + v).toint(
                            Int64
                        ),
                        mem_space=1,
                        dtype=Int32,
                    )
                    ptr.store(packed, alignment=4)
            else:
                for i in cutlass.range_constexpr(per_block):
                    d = column + (start + i) * value_stride
                    out[Int64(row) * 512 + d] = (fragment[start + i] * inv_scale).to(
                        out.element_type
                    )
