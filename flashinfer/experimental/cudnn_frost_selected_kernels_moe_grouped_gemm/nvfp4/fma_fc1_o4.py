# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""FMA FC1 with four outputs per warp, fused activation and NVFP4 quantization."""

import cutlass
from cutlass.cute.typing import Float16
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import vector
from cutlass import cute, utils
from flashinfer.cute_dsl.fp4_common import (
    cvt_e2m1x8_f32,
    cvt_f32_to_e4m3,
    cvt_e4m3_to_f32_via_f16,
    rcp_approx_ftz,
)

from flashinfer.experimental.cudnn_frost_selected_kernels_moe_grouped_gemm.fma_activations import (
    apply_activation,
)

THREADS = 256
WARPS = THREADS // 32
ROWS_PER_WARP = 4


# Overridden by the generated host specialization before compilation.
_ACTIVATION = "swiglu"
_GATED = True


@dsl_user_op
def _fp4_quad(packed, *, loc=None, ip=None):
    decoded = cute.arch.cvt_f4e2m1x4_to_f16x4(
        packed.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    return cutlass.Vector.from_elements(
        (
            Float16(vector.extract(decoded, [], [0], loc=loc, ip=ip)),
            Float16(vector.extract(decoded, [], [1], loc=loc, ip=ip)),
            Float16(vector.extract(decoded, [], [2], loc=loc, ip=ip)),
            Float16(vector.extract(decoded, [], [3], loc=loc, ip=ip)),
        ),
        cutlass.Float16,
    )


@cute.kernel
def _kernel(
    x,
    x_scale,
    weights,
    weight_scale,
    alpha,
    ids,
    quant_out,
    quant_sf,
    quant_global,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    SCALE_COLS = cute.size(x, mode=[1]) // 8
    block, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    I = cute.size(weights, mode=[1]) // (2 if gated else 1)
    top_k = cute.size(ids, mode=[1])
    experts = cute.size(weights, mode=[0])
    rows_per_cta = WARPS * ROWS_PER_WARP
    row_tiles = I // rows_per_cta
    route = block // row_tiles
    row = (block - route * row_tiles) * rows_per_cta + warp * ROWS_PER_WARP
    token = route // top_k
    slot = route - token * top_k
    expert = ids[token, slot]
    intermediate_values = utils.SmemAllocator().allocate_tensor(
        cutlass.Float32, cute.make_layout(32), byte_alignment=16
    )

    x64 = cute.recast_ptr(x.iterator, dtype=cutlass.Uint64)
    w64 = cute.recast_ptr(weights.iterator, dtype=cutlass.Uint64)
    ws8 = cute.recast_ptr(weight_scale.iterator, dtype=cutlass.Uint8)

    if expert >= 0:
        if expert < experts:
            u0 = cutlass.Float32(0.0)
            g0 = cutlass.Float32(0.0)
            u1 = cutlass.Float32(0.0)
            g1 = cutlass.Float32(0.0)
            u2 = cutlass.Float32(0.0)
            g2 = cutlass.Float32(0.0)
            u3 = cutlass.Float32(0.0)
            g3 = cutlass.Float32(0.0)

            row_scale_base = (
                (row // 128) * (128 * SCALE_COLS)
                + (row % 32) * 16
                + ((row % 128) // 32) * 4
            )
            lane_scale_base = (lane // 4) * 512 + lane % 4

            for lane_group in range(SCALE_COLS // 32):
                sc = lane + lane_group * 32
                if cutlass.const_expr(
                    cute.size(x_scale, mode=[0]) == cute.size(ids, mode=[0])
                    and cute.size(x_scale, mode=[1]) == SCALE_COLS
                ):
                    xs_bits = x_scale[token, sc]
                else:
                    xs_bits = x_scale[0, (sc // 4) * 512 + token * 16 + sc % 4]
                xs = xs_bits.bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                sf = row_scale_base + lane_scale_base + lane_group * 4096
                plane = expert * I * SCALE_COLS + sf
                gate_plane = experts * I * SCALE_COLS + plane
                su0 = ws8[plane].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                if cutlass.const_expr(gated):
                    sg0 = (
                        ws8[gate_plane]
                        .bitcast(cutlass.Float8E4M3FN)
                        .to(cutlass.Float32)
                    )
                su1 = ws8[plane + 16].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                if cutlass.const_expr(gated):
                    sg1 = (
                        ws8[gate_plane + 16]
                        .bitcast(cutlass.Float8E4M3FN)
                        .to(cutlass.Float32)
                    )
                su2 = ws8[plane + 32].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                if cutlass.const_expr(gated):
                    sg2 = (
                        ws8[gate_plane + 32]
                        .bitcast(cutlass.Float8E4M3FN)
                        .to(cutlass.Float32)
                    )
                su3 = ws8[plane + 48].bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)
                if cutlass.const_expr(gated):
                    sg3 = (
                        ws8[gate_plane + 48]
                        .bitcast(cutlass.Float8E4M3FN)
                        .to(cutlass.Float32)
                    )

                gu0 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                gg0 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                gu1 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                gg1 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                gu2 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                gg2 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                gu3 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                gg3 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)

                xword = x64[token * SCALE_COLS + sc]
                erow = expert * ((2 if gated else 1) * I) * SCALE_COLS
                up0 = w64[erow + row * SCALE_COLS + sc]
                if cutlass.const_expr(gated):
                    gt0 = w64[erow + (I + row) * SCALE_COLS + sc]
                up1 = w64[erow + (row + 1) * SCALE_COLS + sc]
                if cutlass.const_expr(gated):
                    gt1 = w64[erow + (I + row + 1) * SCALE_COLS + sc]
                up2 = w64[erow + (row + 2) * SCALE_COLS + sc]
                if cutlass.const_expr(gated):
                    gt2 = w64[erow + (I + row + 2) * SCALE_COLS + sc]
                up3 = w64[erow + (row + 3) * SCALE_COLS + sc]
                if cutlass.const_expr(gated):
                    gt3 = w64[erow + (I + row + 3) * SCALE_COLS + sc]

                for j in range(4):
                    shift = j * 16
                    xv = _fp4_quad((xword >> shift).to(cutlass.Uint16))
                    vu0 = _fp4_quad((up0 >> shift).to(cutlass.Uint16))
                    if cutlass.const_expr(gated):
                        vg0 = _fp4_quad((gt0 >> shift).to(cutlass.Uint16))
                    vu1 = _fp4_quad((up1 >> shift).to(cutlass.Uint16))
                    if cutlass.const_expr(gated):
                        vg1 = _fp4_quad((gt1 >> shift).to(cutlass.Uint16))
                    vu2 = _fp4_quad((up2 >> shift).to(cutlass.Uint16))
                    if cutlass.const_expr(gated):
                        vg2 = _fp4_quad((gt2 >> shift).to(cutlass.Uint16))
                    vu3 = _fp4_quad((up3 >> shift).to(cutlass.Uint16))
                    if cutlass.const_expr(gated):
                        vg3 = _fp4_quad((gt3 >> shift).to(cutlass.Uint16))
                    gu0 = gu0 + xv * vu0
                    if cutlass.const_expr(gated):
                        gg0 = gg0 + xv * vg0
                    gu1 = gu1 + xv * vu1
                    if cutlass.const_expr(gated):
                        gg1 = gg1 + xv * vg1
                    gu2 = gu2 + xv * vu2
                    if cutlass.const_expr(gated):
                        gg2 = gg2 + xv * vg2
                    gu3 = gu3 + xv * vu3
                    if cutlass.const_expr(gated):
                        gg3 = gg3 + xv * vg3

                u0 += gu0.to(cutlass.Float32).reduce("add") * (xs * su0)
                if cutlass.const_expr(gated):
                    g0 += gg0.to(cutlass.Float32).reduce("add") * (xs * sg0)
                u1 += gu1.to(cutlass.Float32).reduce("add") * (xs * su1)
                if cutlass.const_expr(gated):
                    g1 += gg1.to(cutlass.Float32).reduce("add") * (xs * sg1)
                u2 += gu2.to(cutlass.Float32).reduce("add") * (xs * su2)
                if cutlass.const_expr(gated):
                    g2 += gg2.to(cutlass.Float32).reduce("add") * (xs * sg2)
                u3 += gu3.to(cutlass.Float32).reduce("add") * (xs * su3)
                if cutlass.const_expr(gated):
                    g3 += gg3.to(cutlass.Float32).reduce("add") * (xs * sg3)

            u0 = cute.arch.warp_reduction_sum(u0)
            if cutlass.const_expr(gated):
                g0 = cute.arch.warp_reduction_sum(g0)
            u1 = cute.arch.warp_reduction_sum(u1)
            if cutlass.const_expr(gated):
                g1 = cute.arch.warp_reduction_sum(g1)
            u2 = cute.arch.warp_reduction_sum(u2)
            if cutlass.const_expr(gated):
                g2 = cute.arch.warp_reduction_sum(g2)
            u3 = cute.arch.warp_reduction_sum(u3)
            if cutlass.const_expr(gated):
                g3 = cute.arch.warp_reduction_sum(g3)
            if lane == 0:
                a = alpha[expert]
                for r, u, g in ((0, u0, g0), (1, u1, g1), (2, u2, g2), (3, u3, g3)):
                    u = u * a
                    g = g * a
                    value = apply_activation(u, g, activation, True)
                    intermediate_values[warp * 4 + r] = value.to(cutlass.BFloat16).to(
                        cutlass.Float32
                    )
        else:
            if lane == 0:
                for r in range(4):
                    intermediate_values[warp * 4 + r] = cutlass.Float32(0.0)
    else:
        if lane == 0:
            for r in range(4):
                intermediate_values[warp * 4 + r] = cutlass.Float32(0.0)
    cute.arch.sync_threads()
    intermediate = I
    if tid < 2:
        values = cute.make_rmem_tensor((16,), cutlass.Float32)
        maximum = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(16):
            values[j] = intermediate_values[tid * 16 + j]
            maximum = cutlass.max(maximum, cute.math.abs(values[j]))
        gs = quant_global[0]
        sf_bits = cvt_f32_to_e4m3(maximum * cutlass.Float32(1.0 / 6.0) * gs)
        sf_float = cvt_e4m3_to_f32_via_f16(sf_bits)
        inverse = cutlass.Float32(0.0)
        if maximum != 0.0:
            inverse = gs * rcp_approx_ftz(sf_float)
        for j in cutlass.range_constexpr(16):
            values[j] = values[j] * inverse
        lo = cvt_e2m1x8_f32(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
        )
        hi = cvt_e2m1x8_f32(
            values[8],
            values[9],
            values[10],
            values[11],
            values[12],
            values[13],
            values[14],
            values[15],
        )
        qcol = (block - route * row_tiles) * 2 + tid
        packed = cutlass.Uint64(lo) | (cutlass.Uint64(hi) << 32)
        cute.recast_ptr(quant_out.iterator, dtype=cutlass.Uint64)[
            route * (intermediate // 16) + qcol
        ] = packed
        sf_index = route * 128 * (intermediate // 16) + (qcol // 4) * 512 + qcol % 4
        quant_sf[sf_index] = sf_bits.to(cutlass.Uint8)


@cute.jit
def _launch(
    x,
    x_scale,
    weights,
    weight_scale,
    alpha,
    ids,
    quant_out,
    quant_sf,
    quant_global,
    stream,
):
    routes = cute.size(ids)
    I = cute.size(weights, mode=[1]) // (2 if _GATED else 1)
    blocks = routes * (I // (WARPS * ROWS_PER_WARP))
    _kernel(
        x,
        x_scale,
        weights,
        weight_scale,
        alpha,
        ids,
        quant_out,
        quant_sf,
        quant_global,
        _ACTIVATION,
        _GATED,
    ).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)


_kernel.set_name_prefix("flashinfer_nvfp4_fma", remove_cutlass_symbol=True)
