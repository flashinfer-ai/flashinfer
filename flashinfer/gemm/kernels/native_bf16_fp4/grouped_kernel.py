# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Reuse decoded weights across M tiles and group CTAs for cache locality."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None):
    elem_ptr = tensor.iterator + Int32(offset)
    return Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def get_smem_ptr_as_int32(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None):
    elem_ptr = tensor.iterator + Int32(offset)
    return elem_ptr.toint(loc=loc, ip=ip)


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
def scaled_two_pairs_e4m3(
    packed0: Uint32, packed1: Uint32, scale: Uint32, *, loc=None, ip=None
):
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [
            Uint32(packed0).ir_value(loc=loc, ip=ip),
            Uint32(packed1).ir_value(loc=loc, ip=ip),
            Uint32(scale).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b8 q0, q1; .reg .b16 s; .reg .b32 packed_s, v0, v1, scales; cvt.u8.u32 q0, $2; cvt.u8.u32 q1, $3; mul.lo.u32 packed_s, $4, 257; cvt.u16.u32 s, packed_s; cvt.rn.bf16x2.e4m3x2 scales, s; cvt.rn.bf16x2.e2m1x2 v0, q0; cvt.rn.bf16x2.e2m1x2 v1, q1; mul.rn.bf16x2 $0, v0, scales; mul.rn.bf16x2 $1, v1, scales; }",
        "=r,=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(2)
        )
    )


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
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
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(2)
        )
    )


@dsl_user_op
def ld_shared_u32(addr: Int32, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def scaled_two_pairs_e4m3_word(
    packed0: Uint32,
    packed1: Uint32,
    scale_word: Uint32,
    scale_byte: int,
    *,
    loc=None,
    ip=None,
):
    values = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [
            Uint32(packed0).ir_value(loc=loc, ip=ip),
            Uint32(packed1).ir_value(loc=loc, ip=ip),
            Uint32(scale_word).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b8 q0, q1; .reg .b16 s; .reg .b32 packed_s, v0, v1, scales; cvt.u8.u32 q0, $2; cvt.u8.u32 q1, $3; prmt.b32.rc8 packed_s, $4, $4, "
        + str(scale_byte)
        + "; cvt.u16.u32 s, packed_s; cvt.rn.bf16x2.e4m3x2 scales, s; cvt.rn.bf16x2.e2m1x2 v0, q0; cvt.rn.bf16x2.e2m1x2 v1, q1; mul.rn.bf16x2 $0, v0, scales; mul.rn.bf16x2 $1, v1, scales; }",
        "=r,=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(2)
        )
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
        (
            Uint32(llvm.extractvalue(T.i32(), values, [i], loc=loc, ip=ip))
            for i in range(4)
        )
    )


@dsl_user_op
def mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        (
            Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
            for i in range(4)
        )
    )


class NativeBf16Fp4GroupedKernel:
    def __init__(
        self,
        m_tiles: int,
        warps: int = 8,
        tile_k: int = 128,
        stages: int = 3,
        alpha_one: bool = False,
        loop_unroll: int = 4,
        group_m: int = 1,
        force_compact_a: bool = False,
        word_scales: bool = False,
    ):
        self.word_scales = word_scales
        self.m_tiles = m_tiles
        self.tile_m = 16 * m_tiles
        self.warps = warps
        self.tile_k = tile_k
        self.stages = stages
        self.alpha_one = alpha_one
        self.loop_unroll = loop_unroll
        self.group_m = group_m
        self.n_tiles = 128 // (warps * 8)
        self.compact_a = m_tiles >= 20 or force_compact_a
        self.compact_b = tile_k == 64

    @cute.jit
    def __call__(self, a, b_u8, sf, alpha: cute.Tensor, out, stream: cuda.CUstream):
        a = cute.recast_tensor(a, Uint32)
        b = cute.recast_tensor(b_u8, Uint32)
        out_u32 = cute.recast_tensor(out, Uint32)
        m = out_u32.shape[0]
        n = out_u32.shape[1] * 2
        if cutlass.const_expr(self.group_m > 1):
            self.kernel(a, b, sf, alpha, out_u32).launch(
                grid=(cute.ceil_div(n, 128) * cute.ceil_div(m, self.tile_m), 1, 1),
                block=(32 * self.warps, 1, 1),
                min_blocks_per_mp=1,
                stream=stream,
            )
        else:
            self.kernel(a, b, sf, alpha, out_u32).launch(
                grid=(cute.ceil_div(n, 128), cute.ceil_div(m, self.tile_m), 1),
                block=(32 * self.warps, 1, 1),
                min_blocks_per_mp=1,
                stream=stream,
            )

    @cute.jit
    def load_stage(self, a, b, sf, sa, sb, ssf, bn, bm, tile, stage, stop):
        tid, _, _ = cute.arch.thread_idx()
        k = b.shape[1] * 8
        n = b.shape[0]
        m = a.shape[0]
        threads = self.warps * 32
        a_cols = self.tile_k // 2
        b_cols = self.tile_k // 8
        for i in cutlass.range_constexpr(
            cute.ceil_div(self.tile_m * a_cols, threads * 4)
        ):
            idx = tid * 4 + i * threads * 4
            if idx < self.tile_m * a_cols:
                row, col = (idx // a_cols, idx % a_cols)
                g_row = bm * self.tile_m + row
                valid = Int32(0)
                if g_row < m and tile < stop:
                    valid = Int32(16)
                src = get_ptr_as_int64(a, g_row * (k // 2) + tile * a_cols + col)
                smem_col = col
                if cutlass.const_expr(self.compact_a):
                    smem_col = (col >> 2 ^ row & 7) << 2
                dst = get_smem_ptr_as_int32(sa, sa.layout((row, smem_col, stage)))
                copy_async_16(dst, src, valid)
        for i in cutlass.range_constexpr(cute.ceil_div(128 * b_cols, threads * 4)):
            idx = tid * 4 + i * threads * 4
            if idx < 128 * b_cols:
                row, col = (idx // b_cols, idx % b_cols)
                g_row = bn * 128 + row
                valid = Int32(0)
                if g_row < n and tile < stop:
                    valid = Int32(16)
                src = get_ptr_as_int64(b, g_row * (k // 8) + tile * b_cols + col)
                smem_col = col
                if cutlass.const_expr(self.compact_b):
                    smem_col = ((col >> 2 ^ row >> 2 & 1) << 2) + (col & 3)
                dst = get_smem_ptr_as_int32(sb, sb.layout((row, smem_col, stage)))
                copy_async_16(dst, src, valid)
        sf_bytes = self.tile_k // 64 * 512
        for i in cutlass.range_constexpr(cute.ceil_div(sf_bytes, threads * 16)):
            idx = tid * 16 + i * threads * 16
            if idx < sf_bytes:
                valid = Int32(0)
                if tile < stop:
                    valid = Int32(16)
                src = get_ptr_as_int64(sf, bn * (k // 64) * 512 + tile * sf_bytes + idx)
                dst = get_smem_ptr_as_int32(ssf, ssf.layout((idx, stage)))
                copy_async_16(dst, src, valid)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def accumulate_stage(self, sa, sb, ssf, acc, stage):
        # Bound tensor-core accumulation depth before rounded FP32 addition.
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        group, pair = (lane // 4, lane % 4)
        bfrag0 = cute.make_rmem_tensor((self.n_tiles, self.tile_k // 16), Uint32)
        bfrag1 = cute.make_rmem_tensor((self.n_tiles, self.tile_k // 16), Uint32)
        if cutlass.const_expr(self.word_scales):
            scale_words = cute.make_rmem_tensor((self.n_tiles,), Uint32)
            for scale_group in cutlass.range_constexpr(self.tile_k // 64):
                for nt in cutlass.range_constexpr(self.n_tiles):
                    ni = warp * self.n_tiles * 8 + nt * 8 + group
                    si = ni % 32 * 16 + ni // 32 * 4 + scale_group * 512
                    scale_addr = get_smem_ptr_as_int32(ssf, ssf.layout((si, stage)))
                    scale_words[nt] = ld_shared_u32(scale_addr)
                for scale_byte in cutlass.range_constexpr(4):
                    fragment = scale_group * 4 + scale_byte
                    for nt in cutlass.range_constexpr(self.n_tiles):
                        ni = warp * self.n_tiles * 8 + nt * 8 + group
                        b_col = fragment * 2
                        if cutlass.const_expr(self.compact_b):
                            b_col = ((b_col >> 2 ^ ni >> 2 & 1) << 2) + (b_col & 3)
                        addr = get_smem_ptr_as_int32(sb, sb.layout((ni, b_col, stage)))
                        lo, hi = ld_shared_v2_u32(addr)
                        bfrag0[nt, fragment], bfrag1[nt, fragment] = (
                            scaled_two_pairs_e4m3_word(
                                lo >> pair * 8,
                                hi >> pair * 8,
                                scale_words[nt],
                                scale_byte,
                            )
                        )
        else:
            for fragment in cutlass.range_constexpr(self.tile_k // 16):
                for nt in cutlass.range_constexpr(self.n_tiles):
                    ni = warp * self.n_tiles * 8 + nt * 8 + group
                    b_col = fragment * 2
                    if cutlass.const_expr(self.compact_b):
                        b_col = ((b_col >> 2 ^ ni >> 2 & 1) << 2) + (b_col & 3)
                    addr = get_smem_ptr_as_int32(sb, sb.layout((ni, b_col, stage)))
                    lo, hi = ld_shared_v2_u32(addr)
                    si = ni % 32 * 16 + ni // 32 * 4
                    si = si + fragment // 4 * 512 + fragment % 4
                    scale = Uint32(ssf[si, stage])
                    bfrag0[nt, fragment], bfrag1[nt, fragment] = scaled_two_pairs_e4m3(
                        lo >> pair * 8, hi >> pair * 8, scale
                    )
        for mt in cutlass.range_constexpr(self.m_tiles):
            partial = cute.make_rmem_tensor((self.n_tiles, 4), Float32)
            partial.fill(0.0)
            for fragment in cutlass.range_constexpr(self.tile_k // 16):
                a_row = mt * 16 + lane % 16
                a_col = fragment * 8 + lane // 16 * 4
                if cutlass.const_expr(self.compact_a):
                    a_col = (a_col >> 2 ^ a_row & 7) << 2
                addr = get_smem_ptr_as_int32(sa, sa.layout((a_row, a_col, stage)))
                a0, a1, a2, a3 = load_matrix_a(addr)
                for nt in cutlass.range_constexpr(self.n_tiles):
                    c0, c1, c2, c3 = mma_bf16(
                        a0,
                        a1,
                        a2,
                        a3,
                        bfrag0[nt, fragment],
                        bfrag1[nt, fragment],
                        partial[nt, 0],
                        partial[nt, 1],
                        partial[nt, 2],
                        partial[nt, 3],
                    )
                    partial[nt, 0] = c0
                    partial[nt, 1] = c1
                    partial[nt, 2] = c2
                    partial[nt, 3] = c3
            for nt in cutlass.range_constexpr(self.n_tiles):
                for i in cutlass.range_constexpr(4):
                    acc[mt, nt, i] += partial[nt, i]

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sf: cute.Tensor,
        alpha: cute.Tensor,
        out: cute.Tensor,
    ):
        alpha = alpha[0]
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        bx, by, _ = cute.arch.block_idx()
        bn, bm = (bx, by)
        group, pair = (lane // 4, lane % 4)
        m = out.shape[0]
        n = out.shape[1] * 2
        if cutlass.const_expr(self.group_m > 1):
            n_blocks = cute.ceil_div(n, 128)
            m_blocks = cute.ceil_div(m, self.tile_m)
            group_span = self.group_m * n_blocks
            group_id = bx // group_span
            first_m = group_id * self.group_m
            group_size = Int32(self.group_m)
            if first_m + group_size > m_blocks:
                group_size = m_blocks - first_m
            within_group = bx % group_span
            bm = first_m + within_group % group_size
            bn = within_group // group_size
        k = b.shape[1] * 8
        k_tiles = k // self.tile_k
        smem = cutlass.utils.SmemAllocator()
        a_cols = self.tile_k // 2
        b_cols = self.tile_k // 8
        if cutlass.const_expr(self.compact_a):
            sa = smem.allocate_tensor(
                cutlass.Int32,
                cute.make_layout(
                    (self.tile_m, a_cols, self.stages),
                    stride=(a_cols, 1, self.tile_m * a_cols),
                ),
                byte_alignment=128,
            )
        else:
            sa = smem.allocate_tensor(
                cutlass.Int32,
                cute.make_layout(
                    (self.tile_m, a_cols + 4, self.stages),
                    stride=(a_cols + 4, 1, self.tile_m * (a_cols + 4)),
                ),
                byte_alignment=16,
            )
        if cutlass.const_expr(self.compact_b):
            sb = smem.allocate_tensor(
                cutlass.Int32,
                cute.make_layout(
                    (128, b_cols, self.stages), stride=(b_cols, 1, 128 * b_cols)
                ),
                byte_alignment=128,
            )
        else:
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
                (self.tile_k // 64 * 512, self.stages), order=(0, 1)
            ),
            byte_alignment=16,
        )
        acc = cute.make_rmem_tensor((self.m_tiles, self.n_tiles, 4), Float32)
        bfrag0 = cute.make_rmem_tensor((self.n_tiles,), Uint32)
        bfrag1 = cute.make_rmem_tensor((self.n_tiles,), Uint32)
        if cutlass.const_expr(self.word_scales):
            scale_words = cute.make_rmem_tensor((self.n_tiles,), Uint32)
        acc.fill(0.0)
        for stage in cutlass.range_constexpr(self.stages):
            self.load_stage(a, b, sf, sa, sb, ssf, bn, bm, stage, stage, k_tiles)
        for offset in cutlass.range(k_tiles, unroll=self.loop_unroll):
            stage = offset % self.stages
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            if cutlass.const_expr(k == 17408):
                self.accumulate_stage(sa, sb, ssf, acc, stage)
            elif cutlass.const_expr(self.word_scales):
                for scale_group in cutlass.range_constexpr(self.tile_k // 64):
                    for nt in cutlass.range_constexpr(self.n_tiles):
                        ni = warp * self.n_tiles * 8 + nt * 8 + group
                        si = ni % 32 * 16 + ni // 32 * 4
                        si = si + scale_group * 512
                        scale_addr = get_smem_ptr_as_int32(ssf, ssf.layout((si, stage)))
                        scale_words[nt] = ld_shared_u32(scale_addr)
                    for scale_byte in cutlass.range_constexpr(4):
                        fragment = scale_group * 4 + scale_byte
                        for nt in cutlass.range_constexpr(self.n_tiles):
                            ni = warp * self.n_tiles * 8 + nt * 8 + group
                            b_col = fragment * 2
                            if cutlass.const_expr(self.compact_b):
                                b_col = ((b_col >> 2 ^ ni >> 2 & 1) << 2) + (b_col & 3)
                            addr = get_smem_ptr_as_int32(
                                sb, sb.layout((ni, b_col, stage))
                            )
                            lo, hi = ld_shared_v2_u32(addr)
                            bfrag0[nt], bfrag1[nt] = scaled_two_pairs_e4m3_word(
                                lo >> pair * 8,
                                hi >> pair * 8,
                                scale_words[nt],
                                scale_byte,
                            )
                        for mt in cutlass.range_constexpr(self.m_tiles):
                            a_row = mt * 16 + lane % 16
                            a_col = fragment * 8 + lane // 16 * 4
                            if cutlass.const_expr(self.compact_a):
                                a_col = (a_col >> 2 ^ a_row & 7) << 2
                            addr = get_smem_ptr_as_int32(
                                sa, sa.layout((a_row, a_col, stage))
                            )
                            a0, a1, a2, a3 = load_matrix_a(addr)
                            for nt in cutlass.range_constexpr(self.n_tiles):
                                c0, c1, c2, c3 = mma_bf16(
                                    a0,
                                    a1,
                                    a2,
                                    a3,
                                    bfrag0[nt],
                                    bfrag1[nt],
                                    acc[mt, nt, 0],
                                    acc[mt, nt, 1],
                                    acc[mt, nt, 2],
                                    acc[mt, nt, 3],
                                )
                                acc[mt, nt, 0] = c0
                                acc[mt, nt, 1] = c1
                                acc[mt, nt, 2] = c2
                                acc[mt, nt, 3] = c3
            else:
                for fragment in cutlass.range_constexpr(self.tile_k // 16):
                    for nt in cutlass.range_constexpr(self.n_tiles):
                        ni = warp * self.n_tiles * 8 + nt * 8 + group
                        b_col = fragment * 2
                        if cutlass.const_expr(self.compact_b):
                            b_col = ((b_col >> 2 ^ ni >> 2 & 1) << 2) + (b_col & 3)
                        addr = get_smem_ptr_as_int32(sb, sb.layout((ni, b_col, stage)))
                        lo, hi = ld_shared_v2_u32(addr)
                        si = ni % 32 * 16 + ni // 32 * 4
                        si = si + fragment // 4 * 512 + fragment % 4
                        scale = Uint32(ssf[si, stage])
                        bfrag0[nt], bfrag1[nt] = scaled_two_pairs_e4m3(
                            lo >> pair * 8, hi >> pair * 8, scale
                        )
                    for mt in cutlass.range_constexpr(self.m_tiles):
                        a_row = mt * 16 + lane % 16
                        a_col = fragment * 8 + lane // 16 * 4
                        if cutlass.const_expr(self.compact_a):
                            a_col = (a_col >> 2 ^ a_row & 7) << 2
                        addr = get_smem_ptr_as_int32(
                            sa, sa.layout((a_row, a_col, stage))
                        )
                        a0, a1, a2, a3 = load_matrix_a(addr)
                        for nt in cutlass.range_constexpr(self.n_tiles):
                            c0, c1, c2, c3 = mma_bf16(
                                a0,
                                a1,
                                a2,
                                a3,
                                bfrag0[nt],
                                bfrag1[nt],
                                acc[mt, nt, 0],
                                acc[mt, nt, 1],
                                acc[mt, nt, 2],
                                acc[mt, nt, 3],
                            )
                            acc[mt, nt, 0] = c0
                            acc[mt, nt, 1] = c1
                            acc[mt, nt, 2] = c2
                            acc[mt, nt, 3] = c3
            cute.arch.sync_threads()
            next_tile = offset + self.stages
            if next_tile < k_tiles:
                self.load_stage(
                    a, b, sf, sa, sb, ssf, bn, bm, next_tile, stage, k_tiles
                )
            else:
                cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        m_base = bm * self.tile_m
        n_base = bn * 128 + warp * self.n_tiles * 8
        for mt in cutlass.range_constexpr(self.m_tiles):
            for nt in cutlass.range_constexpr(self.n_tiles):
                for row in cutlass.range_constexpr(2):
                    mi = m_base + mt * 16 + group + row * 8
                    ni = n_base + nt * 8 + pair * 2
                    if mi < m and ni < n:
                        lo = acc[mt, nt, row * 2]
                        hi = acc[mt, nt, row * 2 + 1]
                        if cutlass.const_expr(not self.alpha_one):
                            lo = lo * alpha
                            hi = hi * alpha
                        out[mi, ni // 2] = pack_bf16x2(lo, hi)
