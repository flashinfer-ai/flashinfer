# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Asynchronous canonical weight staging for small-M W4A16."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ....cute_dsl.fp4_common import (
    get_ptr_as_int64,
    get_smem_ptr_as_int32,
)
from .kernel import _DIRECT_BF16_CVT, mma_bf16, reduce_split_k, scaled_bf16_pair


@dsl_user_op
def copy_async_16(dst: Int32, src: Int64, valid: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(dst).ir_value(loc=loc, ip=ip),
            Int64(src).ir_value(loc=loc, ip=ip),
            Int32(valid).ir_value(loc=loc, ip=ip),
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
def scaled_pair_e4m3(packed: Uint32, scale: Uint32, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(packed).ir_value(loc=loc, ip=ip),
                Uint32(scale).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b8 q; .reg .b16 s; .reg .b32 packed_s, values, scales; "
            "cvt.u8.u32 q, $1; mul.lo.u32 packed_s, $2, 257; "
            "cvt.u16.u32 s, packed_s; cvt.rn.bf16x2.e4m3x2 scales, s; "
            "cvt.rn.bf16x2.e2m1x2 values, q; mul.rn.bf16x2 $0, values, scales; }",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_v2_u32(addr: Int32, *, loc=None, ip=None):
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@dsl_user_op
def load_matrix_a(addr: Int32, *, loc=None, ip=None):
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
        for i in range(4)
    )


class NativeBf16Fp4StagedKernel:
    def __init__(self, splits, warps, enable_pdl, tile_k=128, stages=2):
        self.splits = splits
        self.warps = warps
        self.enable_pdl = enable_pdl
        self.tile_k = tile_k
        self.stages = stages
        self.n_tiles = 128 // (warps * 8)

    @cute.jit
    def __call__(self, a, b, sf, alpha, out, partial, stream: cuda.CUstream):
        m, n = out.shape
        self.kernel(a, b, sf, alpha, out, partial).launch(
            grid=(cute.ceil_div(n, 128), self.splits, 1),
            block=(32 * self.warps, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        if cutlass.const_expr(self.splits > 1):
            reduce_split_k(partial, alpha, out, self.enable_pdl).launch(
                grid=(cute.ceil_div(m * n, 256), 1, 1),
                block=(256, 1, 1),
                stream=stream,
                use_pdl=self.enable_pdl,
            )

    @cute.jit
    def load_stage(self, a, b, sf, sa, sb, ssf, bn, tile, stage, stop):
        tid, _, _ = cute.arch.thread_idx()
        k = b.shape[1] * 8
        n = b.shape[0]
        m = a.shape[0]
        threads = self.warps * 32
        a_cols, b_cols = self.tile_k // 2, self.tile_k // 8
        for i in cutlass.range_constexpr(cute.ceil_div(16 * a_cols, threads * 4)):
            idx = tid * 4 + i * threads * 4
            if idx < 16 * a_cols:
                row, col = idx // a_cols, idx % a_cols
                valid = Int32(0)
                if row < m and tile * self.tile_k < k and tile < stop:
                    valid = Int32(16)
                src = get_ptr_as_int64(a, row * (k // 2) + tile * a_cols + col)
                dst = get_smem_ptr_as_int32(sa, sa.layout((row, col, stage)))
                copy_async_16(dst, src, valid)
        for i in cutlass.range_constexpr(cute.ceil_div(128 * b_cols, threads * 4)):
            idx = tid * 4 + i * threads * 4
            if idx < 128 * b_cols:
                row, col = idx // b_cols, idx % b_cols
                valid = Int32(0)
                if bn * 128 + row < n and tile * self.tile_k < k and tile < stop:
                    valid = Int32(16)
                src = get_ptr_as_int64(
                    b, (bn * 128 + row) * (k // 8) + tile * b_cols + col
                )
                dst = get_smem_ptr_as_int32(sb, sb.layout((row, col, stage)))
                copy_async_16(dst, src, valid)
        sf_bytes = (self.tile_k // 64) * 512
        for i in cutlass.range_constexpr(cute.ceil_div(sf_bytes, threads * 16)):
            idx = tid * 16 + i * threads * 16
            if idx < sf_bytes:
                valid = Int32(0)
                if tile * self.tile_k < k and tile < stop:
                    valid = Int32(16)
                src = get_ptr_as_int64(sf, bn * (k // 64) * 512 + tile * sf_bytes + idx)
                dst = get_smem_ptr_as_int32(ssf, ssf.layout((idx, stage)))
                copy_async_16(dst, src, valid)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sf: cute.Tensor,
        alpha: cute.Tensor | None,
        out: cute.Tensor,
        partial: cute.Tensor | None,
    ):
        lane, warp = cute.arch.lane_idx(), cute.arch.warp_idx()
        bn, split, _ = cute.arch.block_idx()
        group, pair = lane // 4, lane % 4
        m, n = out.shape
        k = b.shape[1] * 8
        k_tiles = k // self.tile_k
        per_split = cute.ceil_div(k_tiles, self.splits)
        first = split * per_split
        smem = cutlass.utils.SmemAllocator()
        a_cols, b_cols = self.tile_k // 2, self.tile_k // 8
        sa = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout(
                (16, a_cols + 4, self.stages), stride=(a_cols + 4, 1, 16 * (a_cols + 4))
            ),
            byte_alignment=16,
        )
        sb = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout(
                (128, b_cols + 4, self.stages),
                stride=(b_cols + 4, 1, 128 * (b_cols + 4)),
            ),
            byte_alignment=16,
        )
        ssf = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_ordered_layout(
                ((self.tile_k // 64) * 512, self.stages), order=(0, 1)
            ),
            byte_alignment=16,
        )
        acc = cute.make_rmem_tensor((self.n_tiles, 4), Float32)
        acc.fill(0)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        for stage in cutlass.range_constexpr(self.stages):
            self.load_stage(
                a, b, sf, sa, sb, ssf, bn, first + stage, stage, first + per_split
            )

        for offset in cutlass.range(per_split):
            stage = offset % self.stages
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            for fragment in cutlass.range_constexpr(self.tile_k // 16):
                addr = get_smem_ptr_as_int32(
                    sa,
                    sa.layout((lane % 16, fragment * 8 + (lane // 16) * 4, stage)),
                )
                a0, a1, a2, a3 = load_matrix_a(addr)
                for nt in cutlass.range_constexpr(self.n_tiles):
                    ni = warp * self.n_tiles * 8 + nt * 8 + group
                    addr = get_smem_ptr_as_int32(
                        sb, sb.layout((ni, fragment * 2, stage))
                    )
                    lo, hi = ld_shared_v2_u32(addr)
                    si = (ni % 32) * 16 + (ni // 32) * 4
                    si += (fragment // 4) * 512 + fragment % 4
                    b0, b1 = Uint32(0), Uint32(0)
                    if cutlass.const_expr(_DIRECT_BF16_CVT):
                        scale = Uint32(ssf[si, stage])
                        b0 = scaled_pair_e4m3(lo >> (pair * 8), scale)
                        b1 = scaled_pair_e4m3(hi >> (pair * 8), scale)
                    else:
                        scale = Float32(
                            cute.recast_tensor(ssf, cutlass.Float8E4M3FN)[si, stage]
                        )
                        b0 = scaled_bf16_pair(lo >> (pair * 8), scale)
                        b1 = scaled_bf16_pair(hi >> (pair * 8), scale)
                    c0, c1, c2, c3 = mma_bf16(
                        a0,
                        a1,
                        a2,
                        a3,
                        b0,
                        b1,
                        acc[nt, 0],
                        acc[nt, 1],
                        acc[nt, 2],
                        acc[nt, 3],
                    )
                    acc[nt, 0], acc[nt, 1], acc[nt, 2], acc[nt, 3] = c0, c1, c2, c3
            cute.arch.sync_threads()
            self.load_stage(
                a,
                b,
                sf,
                sa,
                sb,
                ssf,
                bn,
                first + offset + self.stages,
                stage,
                first + per_split,
            )
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        for nt in cutlass.range_constexpr(self.n_tiles):
            for row in cutlass.range_constexpr(2):
                for col in cutlass.range_constexpr(2):
                    mi = group + row * 8
                    ni = bn * 128 + warp * self.n_tiles * 8 + nt * 8 + pair * 2 + col
                    value = acc[nt, row * 2 + col]
                    if mi < m and ni < n:
                        if cutlass.const_expr(self.splits > 1):
                            partial[split, mi, ni] = value
                        else:
                            if cutlass.const_expr(alpha is not None):
                                value = value * alpha[0]
                            out[mi, ni] = out.element_type(value)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
