# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SM80 grouped projection using CUTLASS warp MMA and cp.async.

Adapted from the CUTLASS Ampere tensorop_gemm.py pipeline and the qualified
Frost grouped route/epilogue contract. Original up/gate weights, no TMA.
"""

import cutlass
from cutlass import cute
import cutlass.utils as utils
from .route_members import route_members
from ..fma.activations import apply_activation


class AmpereGroupedProjection:
    def __init__(
        self,
        fc1,
        group_size,
        stages,
        tile_k,
        top_k,
        activation="swiglu",
        gated=True,
        parameters=None,
    ):
        if (
            group_size not in (2, 4, 8)
            or stages not in (2, 4, 8)
            or tile_k not in (64, 128)
        ):
            raise ValueError("Unsupported Ampere projection tactic")
        self.fc1 = fc1
        self.split_gate = fc1 and gated
        self.group_size = group_size
        self.stages = stages
        self.tile_k = tile_k
        self.input_route_divisor = top_k if fc1 else 1
        self.activation = activation
        self.parameters = parameters
        self.rows_per_tile = 32 if self.split_gate else 64

    @cute.jit
    def __call__(self, weights, inputs, ids, output, stream):
        a = cute.make_tensor(
            weights.iterator,
            cute.make_layout(
                (
                    cute.size(weights, mode=[1]),
                    cute.size(weights, mode=[2]),
                    cute.size(weights, mode=[0]),
                ),
                stride=(weights.stride[1], weights.stride[2], weights.stride[0]),
            ),
        )
        route_ids = cute.make_tensor(ids.iterator, cute.make_layout(cute.size(ids)))
        dtype = weights.element_type
        op = cute.nvgpu.warp.MmaF16BF16Op(dtype, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout((4, 1, 1)), permutation_mnk=(64, 8, 16)
        )
        # Same 128-bit contiguous copy / ldmatrix-compatible swizzle as CUTLASS.
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
        )
        a_layout = cute.tile_to_shape(atom, (64, self.tile_k, self.stages), (0, 1, 2))
        b_layout = cute.tile_to_shape(atom, (8, self.tile_k, self.stages), (0, 1, 2))
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            dtype,
            num_bits_per_copy=128,
        )
        threads_k = self.tile_k // 8
        tiled_copy = cute.make_tiled_copy_tv(
            copy_atom,
            cute.make_layout((128 // threads_k, threads_k), stride=(threads_k, 1)),
            cute.make_layout((1, 8), stride=(8, 1)),
        )

        @cute.struct
        class Storage:
            sA: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(a_layout)], 128
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(b_layout)], 128
            ]

        self.storage = Storage
        self.kernel(
            a, inputs, route_ids, output, tiled_mma, tiled_copy, a_layout, b_layout
        ).launch(
            grid=(cute.size(output, mode=[1]) // self.rows_per_tile, 1, cute.size(ids)),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def load_b(self, inputs, ids, members, sB, tidx, tile, slot):
        if cutlass.const_expr(self.tile_k == 128):
            # Guard the vectorized copy before constructing aligned copy pointers.
            # Public inputs are 16-byte aligned; retain the scalar device fallback.
            # Row stride and K-tile offsets are multiples of 128 elements.
            if inputs.iterator.toint() % 16 == 0:
                atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(
                        cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
                    ),
                    inputs.element_type,
                    num_bits_per_copy=128,
                )
                column = (tidx % 16) * 8
                for member in cutlass.range_constexpr(8):
                    if tidx // 16 == member:
                        valid = cutlass.Boolean(False)
                        if cutlass.const_expr(member < self.group_size):
                            valid = members[member] >= 0
                        if valid:
                            offset = (
                                (members[member] // self.input_route_divisor)
                                * inputs.stride[0]
                                + tile * self.tile_k
                                + column
                            )
                            src = cute.make_tensor(
                                cute.make_ptr(
                                    inputs.element_type,
                                    (inputs.iterator + offset).toint(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((8,)),
                            )
                            dst = cute.make_tensor(
                                cute.make_ptr(
                                    inputs.element_type,
                                    (
                                        sB.iterator + sB.layout((member, column, slot))
                                    ).toint(),
                                    cute.AddressSpace.smem,
                                    assumed_align=16,
                                ),
                                cute.make_layout((8,)),
                            )
                            cute.copy(atom, src, dst)
                        else:
                            for element in cutlass.range_constexpr(8):
                                sB[member, column + element, slot] = (
                                    inputs.element_type(0)
                                )
            else:
                self.load_b_scalar(inputs, ids, members, sB, tidx, tile, slot)
        else:
            self.load_b_scalar(inputs, ids, members, sB, tidx, tile, slot)

    @cute.jit
    def load_b_scalar(self, inputs, ids, members, sB, tidx, tile, slot):
        if cutlass.const_expr(self.tile_k == 128):
            # Each warp owns disjoint rows. Member indices stay compile-time constants
            # so register-held route membership never needs dynamic indexing.
            warp = tidx // 32
            lane = tidx % 32
            for member in cutlass.range_constexpr(8):
                if warp == member % 4:
                    for segment in cutlass.range_constexpr(self.tile_k // 32):
                        column = lane + segment * 32
                        value = inputs.element_type(0)
                        if cutlass.const_expr(member < self.group_size):
                            if members[member] >= 0:
                                value = inputs[
                                    members[member] // self.input_route_divisor,
                                    tile * self.tile_k + column,
                                ]
                        sB[member, column, slot] = value
        else:
            if tidx < 32:
                for member in cutlass.range_constexpr(8):
                    for segment in cutlass.range_constexpr(self.tile_k // 32):
                        column = tidx + segment * 32
                        value = inputs.element_type(0)
                        if cutlass.const_expr(member < self.group_size):
                            if members[member] >= 0:
                                value = inputs[
                                    members[member] // self.input_route_divisor,
                                    tile * self.tile_k + column,
                                ]
                        sB[member, column, slot] = value

    @cute.kernel
    def kernel(
        self,
        a,
        inputs,
        ids,
        output,
        tiled_mma: cute.TiledMma,
        tiled_copy: cute.TiledCopy,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
    ):
        bx, _, route = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        expert = ids[route].to(cutlass.Int64)
        if expert < 0 or expert >= cute.size(a, mode=[2]):
            if tidx < self.rows_per_tile:
                output[route, bx * self.rows_per_tile + tidx] = output.element_type(0)
        else:
            members = route_members(ids, route, self.group_size)
            if members[0] >= 0:
                storage = utils.SmemAllocator().allocate(self.storage)
                # This Ampere layout swizzle is in element units (PISL). Keeping
                # it in the layout avoids interpreting it as byte-pointer swizzle.
                sA = cute.make_tensor(storage.sA.data_ptr(), a_layout)
                sB = cute.make_tensor(storage.sB.data_ptr(), b_layout)
                # Reuse A only after all async loads and readers complete.
                sC = cute.make_tensor(
                    cute.recast_ptr(storage.sA.data_ptr(), dtype=cutlass.Float32),
                    cute.make_layout((64, 8), stride=(1, 64)),
                )
                gA = cute.local_tile(
                    a,
                    (self.rows_per_tile, 8, self.tile_k),
                    (bx, 0, None, expert),
                    proj=(1, None, 1),
                )
                copy_thread = tiled_copy.get_slice(tidx)
                if cutlass.const_expr(self.split_gate):
                    up_s = cute.local_tile(sA, (32, self.tile_k), (0, 0, None))
                    gate_s = cute.local_tile(sA, (32, self.tile_k), (1, 0, None))
                    gGate = cute.local_tile(
                        a,
                        (32, 8, self.tile_k),
                        (bx + cute.size(a, mode=[0]) // 64, 0, None, expert),
                        proj=(1, None, 1),
                    )
                    copy_gate_g = copy_thread.partition_S(gGate)
                    copy_gate_s = copy_thread.partition_D(gate_s)
                    copy_a_s = copy_thread.partition_D(up_s)
                else:
                    copy_a_s = copy_thread.partition_D(sA)
                copy_a_g = copy_thread.partition_S(gA)
                mma_thread = tiled_mma.get_slice(tidx)
                frag_a = tiled_mma.make_fragment_A(
                    mma_thread.partition_A(sA)[None, None, None, 0]
                )
                frag_b = tiled_mma.make_fragment_B(
                    mma_thread.partition_B(sB)[None, None, None, 0]
                )
                copy_a = cute.make_tiled_copy_A(
                    cute.make_copy_atom(
                        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), a.element_type
                    ),
                    tiled_mma,
                )
                copy_b = cute.make_tiled_copy_B(
                    cute.make_copy_atom(
                        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 2), a.element_type
                    ),
                    tiled_mma,
                )
                ca = copy_a.get_slice(tidx)
                cb = copy_b.get_slice(tidx)
                sa = ca.partition_S(sA)
                sb = cb.partition_S(sB)
                ra = ca.retile(frag_a)
                rb = cb.retile(frag_b)
                acc = cute.make_rmem_tensor(
                    mma_thread.partition_C(cute.make_identity_tensor((64, 8))).shape,
                    cutlass.Float32,
                )
                acc.fill(0)
                count = cute.size(gA, mode=[2])
                # Fill S-1 slots, keeping one free for overlap with compute.
                for tile in cutlass.range_constexpr(self.stages - 1):
                    if tile < count:
                        cute.copy(
                            tiled_copy,
                            copy_a_g[None, None, None, tile],
                            copy_a_s[None, None, None, tile],
                        )
                        if cutlass.const_expr(self.split_gate):
                            cute.copy(
                                tiled_copy,
                                copy_gate_g[None, None, None, tile],
                                copy_gate_s[None, None, None, tile],
                            )
                        self.load_b(inputs, ids, members, sB, tidx, tile, tile)
                    cute.arch.cp_async_commit_group()
                for tile in cutlass.range(count, unroll=1):
                    slot = tile % self.stages
                    if tile + self.stages - 2 < count:
                        cute.arch.cp_async_wait_group(self.stages - 2)
                    else:
                        cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    cute.copy(copy_a, sa[None, None, None, slot], ra)
                    cute.copy(copy_b, sb[None, None, None, slot], rb)
                    # The write slot is distinct from this iteration's read slot.
                    next_tile = tile + self.stages - 1
                    next_slot = next_tile % self.stages
                    if next_tile < count:
                        cute.copy(
                            tiled_copy,
                            copy_a_g[None, None, None, next_tile],
                            copy_a_s[None, None, None, next_slot],
                        )
                        if cutlass.const_expr(self.split_gate):
                            cute.copy(
                                tiled_copy,
                                copy_gate_g[None, None, None, next_tile],
                                copy_gate_s[None, None, None, next_slot],
                            )
                        self.load_b(
                            inputs, ids, members, sB, tidx, next_tile, next_slot
                        )
                    cute.arch.cp_async_commit_group()
                    for kb in cutlass.range(
                        cute.size(frag_a, mode=[2]), unroll_full=True
                    ):
                        cute.gemm(
                            tiled_mma,
                            acc,
                            frag_a[None, None, kb],
                            frag_b[None, None, kb],
                            acc,
                        )
                    cute.arch.sync_threads()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                thread_c = mma_thread.partition_C(sC)
                for elem in cutlass.range_constexpr(cute.size(acc)):
                    thread_c[elem] = acc[elem]
                cute.arch.sync_threads()
                warp = cute.arch.warp_idx()
                lane = cute.arch.lane_idx()
                for member in cutlass.range_constexpr(self.group_size):
                    if cutlass.const_expr(self.split_gate):
                        if warp == member % 4 and members[member] >= 0:
                            output[members[member], bx * 32 + lane] = apply_activation(
                                sC[lane, member],
                                sC[lane + 32, member],
                                self.activation,
                                False,
                                self.parameters,
                            ).to(output.element_type)
                    else:
                        if warp // 2 == member % 2 and members[member] >= 0:
                            row = (warp % 2) * 32 + lane
                            value = sC[row, member]
                            if cutlass.const_expr(self.fc1):
                                value = apply_activation(
                                    value,
                                    cutlass.Float32(0),
                                    self.activation,
                                    False,
                                    self.parameters,
                                )
                            output[members[member], bx * 64 + row] = value.to(
                                output.element_type
                            )
