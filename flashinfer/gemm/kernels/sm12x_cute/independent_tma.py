# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Independent TMA-staged NVFP4 GEMM with masked output tiles.

MMA register coordinates follow NVIDIA CUTLASS mma_traits_sm120.hpp at
b46b16d003484063bca4ed365e44095c4c6ed633. Inputs use physical 128x4 scales.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Uint16
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils import SmemAllocator


@dsl_user_op
def mma_nvfp4(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sa, sb, *, loc=None, ip=None):
    """One warp's native 16x8x64 FP4 MMA, vec16 E4M3 scale factors."""
    args = [
        x.ir_value(loc=loc, ip=ip)
        for x in (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sa, sb)
    ]
    args.append(Uint16(0).ir_value(loc=loc, ip=ip))
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Float32.mlir_type] * 4),
        args,
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13}, "
        "{$14}, {$16,$16}, {$15}, {$16,$16};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(Float32.mlir_type, result, [i], loc=loc, ip=ip))
        for i in range(4)
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


@dsl_user_op
def copy_bulk(dst, src, count, bar, *, loc=None, ip=None):
    # Public PTX cp.async.bulk shared::cta global complete_tx instruction.
    llvm.inline_asm(
        None,
        [
            dst.toint().ir_value(loc=loc, ip=ip),
            src.toint().ir_value(loc=loc, ip=ip),
            count.ir_value(loc=loc, ip=ip),
            bar.toint().ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_bf16_pair(low, high, *, loc=None, ip=None):
    result = llvm.inline_asm(
        Int32.mlir_type,
        [low.ir_value(loc=loc, ip=ip), high.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def transpose4(x0, x1, x2, x3, lane, *, loc=None, ip=None):
    """Transpose four packed words across each converged four-lane subgroup."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Int32.mlir_type] * 4),
        [x.ir_value(loc=loc, ip=ip) for x in (x0, x1, x2, x3, lane)],
        "{ .reg .b32 bit, tmp0, tmp1, r0, r1, r2, r3; .reg .pred p; "
        "and.b32 bit, $8, 1; setp.ne.u32 p, bit, 0; "
        "selp.b32 tmp0, $4, $5, p; selp.b32 tmp1, $6, $7, p; "
        "shfl.sync.bfly.b32 tmp0, tmp0, 1, 31, 0xffffffff; "
        "shfl.sync.bfly.b32 tmp1, tmp1, 1, 31, 0xffffffff; "
        "selp.b32 r0, tmp0, $4, p; selp.b32 r1, $5, tmp0, p; "
        "selp.b32 r2, tmp1, $6, p; selp.b32 r3, $7, tmp1, p; "
        "and.b32 bit, $8, 2; setp.ne.u32 p, bit, 0; "
        "selp.b32 tmp0, r0, r2, p; selp.b32 tmp1, r1, r3, p; "
        "shfl.sync.bfly.b32 tmp0, tmp0, 2, 31, 0xffffffff; "
        "shfl.sync.bfly.b32 tmp1, tmp1, 2, 31, 0xffffffff; "
        "selp.b32 $0, tmp0, r0, p; selp.b32 $2, r2, tmp0, p; "
        "selp.b32 $1, tmp1, r1, p; selp.b32 $3, r3, tmp1, p; }",
        "=r,=r,=r,=r,r,r,r,r,r",
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
def store_bf16_eight(dst, p0, p1, p2, p3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [dst.toint().ir_value(loc=loc, ip=ip)]
        + [x.ir_value(loc=loc, ip=ip) for x in (p0, p1, p2, p3)],
        "st.global.v4.u32 [$0], {$1,$2,$3,$4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class IndependentMedium:
    def __init__(self, m, n, k):
        self.m, self.n, self.k = m, n, k
        self.tile_m = 64 if m <= 64 and k % 256 == 0 else 128
        self.tile_n = self.tile_m
        self.mi = self.tile_m // 32
        self.threads = 64 * (self.tile_n // 32)
        self.tile_k = 256 if m <= 128 and k % 256 == 0 else 128
        self.words = self.tile_k // 8
        self.groups = self.tile_k // 64
        self.swizzle_bits = 2 if self.tile_k == 128 else 3
        self.row_shift = 1 if self.tile_k == 128 else 0
        self.swizzle_mask = (1 << self.swizzle_bits) - 1
        self.smem_alignment = 128 if self.tile_k == 128 else 1024
        self.smem_bytes = (self.tile_m + self.tile_n + 32) * self.tile_k + 16
        self.stages = (k + self.tile_k - 1) // self.tile_k

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        alpha: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        la = cute.make_composed_layout(
            cute.make_swizzle(self.swizzle_bits, 4, 3),
            0,
            cute.make_layout(
                (self.tile_m, self.tile_k // 2, 2),
                stride=(self.tile_k // 2, 1, self.tile_m * self.tile_k // 2),
            ),
        )
        lb = cute.make_composed_layout(
            cute.make_swizzle(self.swizzle_bits, 4, 3),
            0,
            cute.make_layout(
                (self.tile_n, self.tile_k // 2, 2),
                stride=(self.tile_k // 2, 1, self.tile_n * self.tile_k // 2),
            ),
        )
        atom_a, ta = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a, la, (self.tile_m, self.tile_k // 2)
        )
        atom_b, tb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), b, lb, (self.tile_n, self.tile_k // 2)
        )
        self.kernel(atom_a, ta, atom_b, tb, sfa, sfb, alpha, out, la, lb).launch(
            grid=(
                (self.m + self.tile_m - 1) // self.tile_m,
                (self.n + self.tile_n - 1) // self.tile_n,
                1,
            ),
            block=(self.threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
        )

    @cute.jit
    def tma_stage(
        self,
        atom_a,
        ag,
        ass,
        atom_b,
        bg,
        bs,
        sfa,
        sfb,
        ssa,
        ssb,
        bars,
        tx,
        bx,
        by,
        stage,
        buffer,
    ):
        sf_bytes = Int32(self.groups * 512)
        if cutlass.const_expr(self.k % 128 != 0):  # noqa: SIM102
            if stage == self.stages - 1:
                sf_bytes = Int32(512)
                if tx < 128:
                    ssa[buffer, 1, tx] = Int32(0)
                    ssb[buffer, 1, tx] = Int32(0)
        if tx < 32:
            bar = bars + buffer
            if tx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar, (self.tile_m + self.tile_n) * self.tile_k // 2 + 2 * sf_bytes
                )
                ai = (bx // (128 // self.tile_m)) * (
                    self.k // 64
                ) * 128 + stage * self.groups * 128
                bi = (by // (128 // self.tile_n)) * (
                    self.k // 64
                ) * 128 + stage * self.groups * 128
                copy_bulk(
                    ssa.iterator + buffer * self.groups * 128,
                    sfa.iterator + ai,
                    sf_bytes,
                    bar,
                )
                copy_bulk(
                    ssb.iterator + buffer * self.groups * 128,
                    sfb.iterator + bi,
                    sf_bytes,
                    bar,
                )
            cute.copy(atom_a, ag[None, bx, stage], ass[None, buffer], tma_bar_ptr=bar)
            cute.copy(atom_b, bg[None, by, stage], bs[None, buffer], tma_bar_ptr=bar)

    @cute.kernel
    def kernel(
        self,
        atom_a: cute.CopyAtom,
        ta: cute.Tensor,
        atom_b: cute.CopyAtom,
        tb: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        alpha: cute.Tensor,
        out: cute.Tensor,
        la: cute.ComposedLayout,
        lb: cute.ComposedLayout,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        lane = tx % 32
        warp = tx // 32
        group = lane // 4
        t = lane % 4
        wm = (warp // (self.tile_n // 32)) * (self.tile_m // 2)
        wn = (warp % (self.tile_n // 32)) * 32
        m0 = bx * self.tile_m
        n0 = by * self.tile_n
        smem = SmemAllocator()
        sa = smem.allocate_tensor(
            Int32,
            cute.make_layout(
                (2, self.tile_m, self.words),
                stride=(self.tile_m * self.words, self.words, 1),
            ),
            byte_alignment=self.smem_alignment,
        )
        sb = smem.allocate_tensor(
            Int32,
            cute.make_layout(
                (2, self.tile_n, self.words),
                stride=(self.tile_n * self.words, self.words, 1),
            ),
            byte_alignment=self.smem_alignment,
        )
        ssa = smem.allocate_tensor(
            Int32,
            cute.make_layout((2, self.groups, 128), stride=(self.groups * 128, 128, 1)),
            byte_alignment=16,
        )
        ssb = smem.allocate_tensor(
            Int32,
            cute.make_layout((2, self.groups, 128), stride=(self.groups * 128, 128, 1)),
            byte_alignment=16,
        )
        bars = smem.allocate_array(cutlass.Int64, 2, byte_alignment=8)
        asmem = cute.make_tensor(
            cute.recast_ptr(sa.iterator, la.inner, dtype=cutlass.Uint8), la.outer
        )
        bsmem = cute.make_tensor(
            cute.recast_ptr(sb.iterator, lb.inner, dtype=cutlass.Uint8), lb.outer
        )
        agmem = cute.local_tile(ta, (self.tile_m, self.tile_k // 2), (None, None))
        bgmem = cute.local_tile(tb, (self.tile_n, self.tile_k // 2), (None, None))
        ass, ag = cpasync.tma_partition(
            atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(asmem, 0, 2),
            cute.group_modes(agmem, 0, 2),
        )
        bs, bg = cpasync.tma_partition(
            atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(bsmem, 0, 2),
            cute.group_modes(bgmem, 0, 2),
        )
        if tx == 0:
            cute.arch.mbarrier_init(bars, 1)
            cute.arch.mbarrier_init(bars + 1, 1)
            cute.arch.mbarrier_init_fence()
            cpasync.prefetch_descriptor(atom_a)
            cpasync.prefetch_descriptor(atom_b)
        cute.arch.barrier()
        acc = cute.make_rmem_tensor((self.mi * 16,), Float32)
        acc.fill(0.0)
        self.tma_stage(
            atom_a, ag, ass, atom_b, bg, bs, sfa, sfb, ssa, ssb, bars, tx, bx, by, 0, 0
        )
        for stage in range(self.stages):
            buffer = stage % 2
            if stage + 1 < self.stages:
                self.tma_stage(
                    atom_a,
                    ag,
                    ass,
                    atom_b,
                    bg,
                    bs,
                    sfa,
                    sfb,
                    ssa,
                    ssb,
                    bars,
                    tx,
                    bx,
                    by,
                    stage + 1,
                    (stage + 1) % 2,
                )
            cute.arch.mbarrier_wait(bars + buffer, (stage // 2) % 2)
            if cutlass.const_expr(self.k % 128 != 0):
                cute.arch.barrier()
            for half_k in cutlass.range_constexpr(self.groups):
                for mi in cutlass.range_constexpr(self.mi):
                    ar = wm + mi * 16 + lane % 16
                    aw = half_k * 8 + (lane // 16) * 4
                    aw = aw ^ (((ar >> self.row_shift) & self.swizzle_mask) << 2)
                    a0, a1, a2, a3 = load_a4(
                        sa.iterator
                        + buffer * self.tile_m * self.words
                        + ar * self.words
                        + aw
                    )
                    sf_ar = (
                        (bx % (128 // self.tile_m)) * self.tile_m
                        + wm
                        + mi * 16
                        + group
                        + (lane % 2) * 8
                    )
                    ascale = ssa[buffer, half_k, (sf_ar % 32) * 4 + sf_ar // 32]
                    for ni in cutlass.range_constexpr(4):
                        br = wn + ni * 8 + lane % 8
                        bw = half_k * 8 + ((lane // 8) % 2) * 4
                        bw = bw ^ (((br >> self.row_shift) & self.swizzle_mask) << 2)
                        b0, b1 = load_b2(
                            sb.iterator
                            + buffer * self.tile_n * self.words
                            + br * self.words
                            + bw
                        )
                        bn = (
                            (by % (128 // self.tile_n)) * self.tile_n
                            + wn
                            + ni * 8
                            + group
                        )
                        bscale = ssb[buffer, half_k, (bn % 32) * 4 + bn // 32]
                        ai = (mi * 4 + ni) * 4
                        d0, d1, d2, d3 = mma_nvfp4(
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            acc[ai],
                            acc[ai + 1],
                            acc[ai + 2],
                            acc[ai + 3],
                            ascale,
                            bscale,
                        )
                        acc[ai] = d0
                        acc[ai + 1] = d1
                        acc[ai + 2] = d2
                        acc[ai + 3] = d3
            cute.arch.barrier()
        scale = alpha[0]
        for mi in cutlass.range_constexpr(self.mi):
            for pair in cutlass.range_constexpr(2):
                ai = mi * 16 + pair * 2
                p0 = pack_bf16_pair(acc[ai] * scale, acc[ai + 1] * scale)
                p1 = pack_bf16_pair(acc[ai + 4] * scale, acc[ai + 5] * scale)
                p2 = pack_bf16_pair(acc[ai + 8] * scale, acc[ai + 9] * scale)
                p3 = pack_bf16_pair(acc[ai + 12] * scale, acc[ai + 13] * scale)
                p0, p1, p2, p3 = transpose4(p0, p1, p2, p3, lane)
                row = m0 + wm + mi * 16 + group + pair * 8
                col = n0 + wn + t * 8
                if row < self.m and col + 7 < self.n:
                    store_bf16_eight(out.iterator + row * self.n + col, p0, p1, p2, p3)
