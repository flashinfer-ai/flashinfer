# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Packed NVFP4 GEMM with double-buffered K stages and masked M/K tails.

MMA register coordinates follow NVIDIA CUTLASS mma_traits_sm120.hpp at
b46b16d003484063bca4ed365e44095c4c6ed633. Inputs use physical 128x4 scales.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator


@dsl_user_op
def nvfp4_mma(a0, a1, a2, a3, b0, b1, sfa, sfb, c0, c1, c2, c3, *, loc=None, ip=None):
    """One PTX m16n8k64 atom with four consecutive vec16 E4M3 scales."""
    operands = [
        x.ir_value(loc=loc, ip=ip)
        for x in (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb)
    ]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        operands,
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, "
        "$14, {0,0}, $15, {0,0};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def copy16(dst, src, count, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst.toint().ir_value(loc=loc, ip=ip),
            src.toint().ir_value(loc=loc, ip=ip),
            count.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_a4(src, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [src.toint().ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(Int32.mlir_type, result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def copy4(dst, src, count, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst.toint().ir_value(loc=loc, ip=ip),
            src.toint().ir_value(loc=loc, ip=ip),
            count.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 4, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_b2(src, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 2),
        [src.toint().ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {$0,$1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(Int32.mlir_type, result, [i], loc=loc, ip=ip))
        for i in range(2)
    )


class IndependentSmall:
    def __init__(self, m, n, k):
        self.m, self.n, self.k = m, n, k
        self.tile_m = min(m, 64)
        self.rows = ((self.tile_m + 7) // 8) * 8
        self.tile_k = 256 if 16 < m <= 32 else 512
        self.words = self.tile_k // 8
        self.groups = self.tile_k // 64
        self.vectors = self.tile_k // 32
        self.stages = (k + self.tile_k - 1) // self.tile_k
        self.smem_bytes = 2 * (64 + self.rows) * (self.words + self.groups) * 4

    @cute.jit
    def stage_weights(self, b, shared, tx, n0, stage, buffer):
        for rep in cutlass.range_constexpr(self.groups):
            chunk = tx + rep * 128
            row = chunk // self.vectors
            word = (chunk % self.vectors) * 4
            swizzled = word ^ ((row & 7) << 2)
            source = (n0 + row) * (self.k // 8) + stage * self.words + word
            target = buffer * 64 * self.words + row * self.words + swizzled
            count = Int32(16)
            if cutlass.const_expr(self.k % self.tile_k != 0):  # noqa: SIM102
                if stage * self.words + word >= self.k // 8:
                    source = Int32(0)
                    count = Int32(0)
            copy16(shared.iterator + target, b.iterator + source, count)

    @cute.jit
    def stage_inputs(
        self, a, b, sfa, sfb, weights, acts, sw, sx, tx, m0, n0, stage, buffer
    ):
        self.stage_weights(b, weights, tx, n0, stage, buffer)
        for rep in cutlass.range_constexpr((self.rows * self.vectors + 127) // 128):
            chunk = tx + rep * 128
            if chunk < self.rows * self.vectors:
                row = chunk // self.vectors
                word = (chunk % self.vectors) * 4
                swizzled = word ^ ((row & 7) << 2)
                src = Int32(0)
                count = Int32(0)
                if m0 + row < self.m:
                    if cutlass.const_expr(self.k % self.tile_k == 0):
                        src = (m0 + row) * (self.k // 8) + stage * self.words + word
                        count = Int32(16)
                    else:
                        if stage * self.words + word < self.k // 8:
                            src = (m0 + row) * (self.k // 8) + stage * self.words + word
                            count = Int32(16)
                dst = buffer * self.rows * self.words + row * self.words + swizzled
                copy16(acts.iterator + dst, a.iterator + src, count)
        for rep in cutlass.range_constexpr(self.tile_k // 128):
            row = tx // self.groups + rep * (128 // self.groups)
            half = tx % self.groups
            kg = stage * self.groups + half
            global_row = n0 + row
            sfidx = (
                (global_row // 128) * (self.k // 64) * 128
                + kg * 128
                + (global_row % 32) * 4
                + (global_row // 32) % 4
            )
            if cutlass.const_expr(self.tile_k == 512):
                dst = buffer * 64 * self.groups + half * 64 + row
            else:
                dst = buffer * 64 * self.groups + row * self.groups + half
            count = Int32(4)
            if cutlass.const_expr(self.k % self.tile_k != 0):  # noqa: SIM102
                if kg >= self.k // 64:
                    sfidx = Int32(0)
                    count = Int32(0)
            copy4(sw.iterator + dst, sfb.iterator + sfidx, count)
            if row < self.rows:
                global_a_row = m0 + row
                sfidx = (
                    (global_a_row // 128) * (self.k // 64) * 128
                    + kg * 128
                    + (global_a_row % 32) * 4
                    + (global_a_row // 32) % 4
                )
                if cutlass.const_expr(self.tile_k == 512):
                    dst = buffer * self.rows * self.groups + half * self.rows + row
                else:
                    dst = buffer * self.rows * self.groups + row * self.groups + half
                count = Int32(4)
                if cutlass.const_expr(self.k % self.tile_k != 0):  # noqa: SIM102
                    if kg >= self.k // 64:
                        sfidx = Int32(0)
                        count = Int32(0)
                copy4(sx.iterator + dst, sfa.iterator + sfidx, count)

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        alpha: cute.Tensor,
        out: cute.Tensor,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        lane = tx % 32
        warp = tx // 32
        group = lane // 4
        quarter = lane % 4
        m0 = by * self.tile_m
        n0 = bx * 64
        weight0 = n0 + warp * 16 + group
        weight1 = weight0 + 8
        smem = SmemAllocator()
        weights = smem.allocate_tensor(
            Int32,
            cute.make_layout(
                (2, 64, self.words), stride=(64 * self.words, self.words, 1)
            ),
            byte_alignment=16,
        )
        acts = smem.allocate_tensor(
            Int32,
            cute.make_layout(
                (2, self.rows, self.words),
                stride=(self.rows * self.words, self.words, 1),
            ),
            byte_alignment=16,
        )
        if cutlass.const_expr(self.tile_k == 512):
            sw_layout = cute.make_layout(
                (2, self.groups, 64), stride=(64 * self.groups, 64, 1)
            )
            sx_layout = cute.make_layout(
                (2, self.groups, self.rows),
                stride=(self.rows * self.groups, self.rows, 1),
            )
        else:
            sw_layout = cute.make_layout(
                (2, 64, self.groups), stride=(64 * self.groups, self.groups, 1)
            )
            sx_layout = cute.make_layout(
                (2, self.rows, self.groups),
                stride=(self.rows * self.groups, self.groups, 1),
            )
        sw = smem.allocate_tensor(Int32, sw_layout, byte_alignment=16)
        sx = smem.allocate_tensor(Int32, sx_layout, byte_alignment=16)
        acc = cute.make_rmem_tensor((self.rows // 8, 4), cutlass.Float32)
        acc.fill(0.0)
        self.stage_inputs(a, b, sfa, sfb, weights, acts, sw, sx, tx, m0, n0, 0, 0)
        cute.arch.cp_async_commit_group()
        for stage in range(self.stages):
            buffer = stage % 2
            if stage + 1 < self.stages:
                self.stage_inputs(
                    a,
                    b,
                    sfa,
                    sfb,
                    weights,
                    acts,
                    sw,
                    sx,
                    tx,
                    m0,
                    n0,
                    stage + 1,
                    1 - buffer,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            for half in cutlass.range_constexpr(self.groups):
                ar = warp * 16 + lane % 16
                aw = half * 8 + (lane // 16) * 4
                aw = aw ^ ((ar & 7) << 2)
                a0, a1, a2, a3 = load_a4(
                    weights.iterator + buffer * 64 * self.words + ar * self.words + aw
                )
                for mi in cutlass.range_constexpr(self.rows // 8):
                    br = mi * 8 + lane % 8
                    bw = half * 8 + ((lane // 8) % 2) * 4
                    bw = bw ^ ((br & 7) << 2)
                    b0, b1 = load_b2(
                        acts.iterator
                        + buffer * self.rows * self.words
                        + br * self.words
                        + bw
                    )
                    if cutlass.const_expr(self.tile_k == 512):
                        sfw = cutlass.Uint32(
                            sw[buffer, half, warp * 16 + group + (lane % 2) * 8]
                        )
                        sfx = cutlass.Uint32(sx[buffer, half, mi * 8 + group])
                    else:
                        sfw = cutlass.Uint32(
                            sw[buffer, warp * 16 + group + (lane % 2) * 8, half]
                        )
                        sfx = cutlass.Uint32(sx[buffer, mi * 8 + group, half])
                    c0, c1, c2, c3 = nvfp4_mma(
                        a0,
                        a1,
                        a2,
                        a3,
                        b0,
                        b1,
                        sfw,
                        sfx,
                        acc[mi, 0],
                        acc[mi, 1],
                        acc[mi, 2],
                        acc[mi, 3],
                    )
                    acc[mi, 0] = c0
                    acc[mi, 1] = c1
                    acc[mi, 2] = c2
                    acc[mi, 3] = c3
            cute.arch.barrier()
        scale = alpha[0]
        for mi in cutlass.range_constexpr(self.rows // 8):
            activation0 = m0 + mi * 8 + quarter * 2
            if activation0 < self.m:
                out[activation0, weight0] = (acc[mi, 0] * scale).to(cutlass.BFloat16)
                out[activation0, weight1] = (acc[mi, 2] * scale).to(cutlass.BFloat16)
            if activation0 + 1 < self.m:
                out[activation0 + 1, weight0] = (acc[mi, 1] * scale).to(
                    cutlass.BFloat16
                )
                out[activation0 + 1, weight1] = (acc[mi, 3] * scale).to(
                    cutlass.BFloat16
                )

    @cute.jit
    def launch(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        alpha: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(a, b, sfa, sfb, alpha, out).launch(
            grid=(self.n // 64, (self.m + self.tile_m - 1) // self.tile_m, 1),
            block=(128, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
        )
