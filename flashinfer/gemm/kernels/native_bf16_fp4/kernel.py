# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Small-M W4A16 with packed E2M1 weights and 128x4 E4M3 scales."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ....cute_dsl.fp4_common import (
    f16x2_to_f32x2,
    fp4_decode_2,
    fp4_decode_4bytes,
    pack_16bit_to_u32,
)


_DIRECT_BF16_CVT = cutlass.CUDA_VERSION.major > 13 or (
    cutlass.CUDA_VERSION.major == 13 and cutlass.CUDA_VERSION.minor >= 2
)


@dsl_user_op
def direct_scaled_bf16_pair(packed: Uint32, scale: Float32, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(packed).ir_value(loc=loc, ip=ip),
                Float32(scale).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b8 q; .reg .b32 values, scales; "
            "cvt.u8.u32 q, $1; cvt.rn.bf16x2.e2m1x2 values, q; "
            "cvt.rn.bf16x2.f32 scales, $2, $2; "
            "mul.rn.bf16x2 $0, values, scales; }",
            "=r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def scaled_bf16_pair(packed: Uint32, scale: Float32):
    result = Uint32(0)
    if cutlass.const_expr(_DIRECT_BF16_CVT):
        result = direct_scaled_bf16_pair(packed, scale)
    else:
        lo, hi = f16x2_to_f32x2(fp4_decode_2(packed))
        result = cute.arch.inline_ptx(
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            write_only_types=[Uint32],
            read_only_args=[lo * scale, hi * scale],
        )
    return result


@dsl_user_op
def mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    # Register order follows SM80_16x8x16_F32BF16BF16F32_TN in CUTLASS.
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [Uint32(x).ir_value(loc=loc, ip=ip) for x in (a0, a1, a2, a3, b0, b1)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
        "{$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


class NativeBf16Fp4MmaKernel:
    def __init__(self, splits: int, warps: int, enable_pdl: bool, n_tiles: int):
        self.splits = splits
        self.warps = warps
        self.enable_pdl = enable_pdl
        self.n_tiles = n_tiles

    @cute.jit
    def __call__(self, a, b, sf, alpha, out, partial, stream: cuda.CUstream):
        m, n = out.shape
        self.kernel(a, b, sf, alpha, out, partial).launch(
            grid=(cute.ceil_div(n, 8 * self.warps * self.n_tiles), self.splits, 1),
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
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        bn, split, _ = cute.arch.block_idx()
        group, pair = lane // 4, lane % 4
        n_base = (bn * self.warps + warp) * 8 * self.n_tiles
        m, k = a.shape
        n = out.shape[1]
        k_tiles = cute.ceil_div(k, 64)
        tiles_per_split = cute.ceil_div(k_tiles, self.splits)
        acc = cute.make_rmem_tensor((self.n_tiles, 4), Float32)
        words0 = cute.make_rmem_tensor((self.n_tiles,), Uint32)
        words1 = cute.make_rmem_tensor((self.n_tiles,), Uint32)
        scales = cute.make_rmem_tensor((self.n_tiles,), Float32)
        acc.fill(0.0)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        for tile_offset in cutlass.range(tiles_per_split):
            tile = split * tiles_per_split + tile_offset
            if tile < k_tiles:
                # Four lanes load one complete 32-byte row segment, then
                # exchange words to form four consecutive MMA K fragments.
                for nt in cutlass.range_constexpr(self.n_tiles):
                    n_idx = n_base + nt * 8 + group
                    words0[nt], words1[nt], scales[nt] = (
                        Uint32(0),
                        Uint32(0),
                        Float32(0),
                    )
                    if n_idx < n:
                        if tile * 8 + pair < k // 8:
                            words0[nt] = Uint32(b[n_idx, tile * 8 + pair])
                        if tile * 8 + pair + 4 < k // 8:
                            words1[nt] = Uint32(b[n_idx, tile * 8 + pair + 4])
                        sf_row = (n_idx // 128) * k_tiles * 512
                        sf_row += (n_idx % 32) * 16 + ((n_idx % 128) // 32) * 4
                        scales[nt] = Float32(sf[sf_row + tile * 512 + pair])
                for fragment in cutlass.range_constexpr(4):
                    k_base = tile * 64 + fragment * 16
                    if k_base < k:
                        a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        ak = k_base + pair * 2
                        if group < m:
                            a0 = pack_16bit_to_u32(a[group, ak], a[group, ak + 1])
                            a2 = pack_16bit_to_u32(a[group, ak + 8], a[group, ak + 9])
                        if group + 8 < m:
                            a1 = pack_16bit_to_u32(
                                a[group + 8, ak], a[group + 8, ak + 1]
                            )
                            a3 = pack_16bit_to_u32(
                                a[group + 8, ak + 8], a[group + 8, ak + 9]
                            )
                        for nt in cutlass.range_constexpr(self.n_tiles):
                            word = words0[nt]
                            if cutlass.const_expr(fragment >= 2):
                                word = words1[nt]
                            source = group * 4 + (fragment % 2) * 2
                            b_lo = cute.arch.shuffle_sync(word, source)
                            b_hi = cute.arch.shuffle_sync(word, source + 1)
                            weight_scale = cute.arch.shuffle_sync(
                                scales[nt], group * 4 + fragment
                            )
                            b0 = scaled_bf16_pair(b_lo >> (pair * 8), weight_scale)
                            b1 = scaled_bf16_pair(b_hi >> (pair * 8), weight_scale)
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
                            acc[nt, 0], acc[nt, 1], acc[nt, 2], acc[nt, 3] = (
                                c0,
                                c1,
                                c2,
                                c3,
                            )

        for nt in cutlass.range_constexpr(self.n_tiles):
            for row in cutlass.range_constexpr(2):
                for col in cutlass.range_constexpr(2):
                    m_idx, out_n = group + row * 8, n_base + nt * 8 + pair * 2 + col
                    value = acc[nt, row * 2 + col]
                    if m_idx < m and out_n < n:
                        if cutlass.const_expr(self.splits > 1):
                            partial[split, m_idx, out_n] = value
                        else:
                            if cutlass.const_expr(alpha is not None):
                                value = value * alpha[0]
                            out[m_idx, out_n] = out.element_type(value)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


@cute.kernel
def reduce_split_k(
    partial: cute.Tensor,
    alpha: cute.Tensor | None,
    out: cute.Tensor,
    enable_pdl: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    m, n = out.shape
    idx = bid * 256 + tid
    if cutlass.const_expr(enable_pdl):
        cute.arch.griddepcontrol_wait()
    if idx < m * n:
        row, col = idx // n, idx % n
        value = Float32(0)
        for split in cutlass.range_constexpr(partial.shape[0]):
            value = value + partial[split, row, col]
        if cutlass.const_expr(alpha is not None):
            value = value * alpha[0]
        out[row, col] = out.element_type(value)
    if cutlass.const_expr(enable_pdl):
        cute.arch.griddepcontrol_launch_dependents()


class NativeBf16Fp4Kernel:
    def __init__(self, rows_per_warp: int, warps: int, enable_pdl: bool, splits: int):
        self.rows_per_warp = rows_per_warp
        self.warps = warps
        self.enable_pdl = enable_pdl
        self.splits = splits

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sf: cute.Tensor,
        alpha: cute.Tensor | None,
        out: cute.Tensor,
        partial: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        m, n = out.shape
        self.kernel(a, b, sf, alpha, out, partial).launch(
            grid=(
                cute.ceil_div(n, self.warps),
                cute.ceil_div(m, self.rows_per_warp),
                self.splits,
            ),
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
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        bn, bm, split = cute.arch.block_idx()
        n_idx = bn * self.warps + warp
        m_base = bm * self.rows_per_warp
        m, k = a.shape
        n = b.shape[0]

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        # A warp streams one contiguous weight row. Each lane decodes eight
        # consecutive K values, reusing them across a small activation batch.
        if n_idx < n:
            acc = cute.make_rmem_tensor((self.rows_per_warp,), Float32)
            acc.fill(0.0)
            sf_blocks = cute.ceil_div(k // 16, 4)
            sf_row = (n_idx // 128) * sf_blocks * 512
            sf_row += (n_idx % 32) * 16 + ((n_idx % 128) // 32) * 4

            tiles_per_split = cute.ceil_div(cute.ceil_div(k, 256), self.splits)
            for tile_offset in cutlass.range(tiles_per_split):
                tile = split * tiles_per_split + tile_offset
                k_base = tile * 256 + lane * 8
                if k_base < k:
                    packed = Uint32(b[n_idx, k_base // 8])
                    scale_k = k_base // 16
                    scale = Float32(sf[sf_row + (scale_k // 4) * 512 + scale_k % 4])
                    fragments = fp4_decode_4bytes(packed)
                    dot = cute.make_rmem_tensor((self.rows_per_warp,), Float32)
                    dot.fill(0.0)
                    for pair, fragment in enumerate(fragments):
                        lo, hi = f16x2_to_f32x2(fragment)
                        for row in cutlass.range_constexpr(self.rows_per_warp):
                            m_idx = m_base + row
                            if m_idx < m:
                                x_lo = Float32(a[m_idx, k_base + 2 * pair])
                                x_hi = Float32(a[m_idx, k_base + 2 * pair + 1])
                                dot[row] = dot[row] + x_lo * lo + x_hi * hi
                    for row in cutlass.range_constexpr(self.rows_per_warp):
                        acc[row] = acc[row] + dot[row] * scale

            for row in cutlass.range_constexpr(self.rows_per_warp):
                value = acc[row]
                for step in cutlass.range_constexpr(5):
                    value = value + cute.arch.shuffle_sync_bfly(value, 1 << step)
                if lane == 0 and m_base + row < m:
                    if cutlass.const_expr(self.splits > 1):
                        partial[split, m_base + row, n_idx] = value
                    else:
                        if cutlass.const_expr(alpha is not None):
                            value = value * alpha[0]
                        out[m_base + row, n_idx] = out.element_type(value)

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
