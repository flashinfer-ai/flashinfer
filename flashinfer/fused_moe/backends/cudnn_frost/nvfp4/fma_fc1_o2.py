# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""FMA FC1 with two outputs per warp, fused activation and NVFP4 quantization."""

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

from flashinfer.fused_moe.backends.cudnn_frost.fma_activations import (
    apply_activation,
)

THREADS = 256
WARPS = THREADS // 32


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
def _routed_fc1_kernel(
    x: cute.Tensor,
    x_scale: cute.Tensor,
    weights: cute.Tensor,
    weight_scale: cute.Tensor,
    alpha: cute.Tensor,
    ids: cute.Tensor,
    quant_out: cute.Tensor,
    quant_sf: cute.Tensor,
    quant_global: cute.Tensor,
    activation: cutlass.Constexpr,
    gated: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    SCALE_COLS = cute.size(x, mode=[1]) // 8
    block, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()

    intermediate = cute.size(weights, mode=[1]) // (2 if gated else 1)
    top_k = cute.size(ids, mode=[1])
    experts = cute.size(weights, mode=[0])
    rows_per_cta = WARPS * 2
    row_tiles = intermediate // rows_per_cta

    route = block // row_tiles
    row = (block - route * row_tiles) * rows_per_cta + warp * 2
    row2 = row + 1
    token = route // top_k
    slot = route - token * top_k
    expert = ids[token, slot]
    smem = utils.SmemAllocator()
    intermediate_values = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout(16), byte_alignment=16
    )

    # Reinterpret the packed-byte operands as aligned 64-bit scale blocks.
    # Adjacent lanes then issue adjacent 8-byte loads while retaining one
    # scale block per lane (no scale shuffles or repacking).
    x64 = cute.recast_ptr(x.iterator, dtype=cutlass.Uint64)
    w64 = cute.recast_ptr(weights.iterator, dtype=cutlass.Uint64)

    if expert >= 0:
        if expert < experts:
            acc_up = cutlass.Float32(0.0)
            acc_gate = cutlass.Float32(0.0)
            acc_up2 = cutlass.Float32(0.0)
            acc_gate2 = cutlass.Float32(0.0)

            # Lanes traverse scale blocks spaced one warp apart.
            # Each scale block covers 16 scalar FP4 values (8 packed bytes).
            for lane_group in range(SCALE_COLS // 32):
                scale_col = lane + lane_group * 32
                if cutlass.const_expr(
                    cute.size(x_scale, mode=[0]) == cute.size(ids, mode=[0])
                    and cute.size(x_scale, mode=[1]) == SCALE_COLS
                ):
                    xs_bits = x_scale[token, scale_col]
                else:
                    xs_bits = x_scale[
                        0, (scale_col // 4) * 512 + token * 16 + scale_col % 4
                    ]
                xs = xs_bits.bitcast(cutlass.Float8E4M3FN).to(cutlass.Float32)

                # F8_128x4 physical address for logical (row, scale_col).
                sf_index = (
                    (row // 128) * (128 * SCALE_COLS)
                    + (scale_col // 4) * 512
                    + (row % 32) * 16
                    + ((row % 128) // 32) * 4
                    + scale_col % 4
                )
                sf_phys_row = sf_index // SCALE_COLS
                sf_phys_col = sf_index - sf_phys_row * SCALE_COLS
                ws_up = (
                    weight_scale[0, expert, sf_phys_row, sf_phys_col]
                    .bitcast(cutlass.Float8E4M3FN)
                    .to(cutlass.Float32)
                )
                if cutlass.const_expr(gated):
                    ws_gate = (
                        weight_scale[1, expert, sf_phys_row, sf_phys_col]
                        .bitcast(cutlass.Float8E4M3FN)
                        .to(cutlass.Float32)
                    )

                sf_index2 = (
                    (row2 // 128) * (128 * SCALE_COLS)
                    + (scale_col // 4) * 512
                    + (row2 % 32) * 16
                    + ((row2 % 128) // 32) * 4
                    + scale_col % 4
                )
                sf_phys_row2 = sf_index2 // SCALE_COLS
                sf_phys_col2 = sf_index2 - sf_phys_row2 * SCALE_COLS
                ws_up2 = (
                    weight_scale[0, expert, sf_phys_row2, sf_phys_col2]
                    .bitcast(cutlass.Float8E4M3FN)
                    .to(cutlass.Float32)
                )
                if cutlass.const_expr(gated):
                    ws_gate2 = (
                        weight_scale[1, expert, sf_phys_row2, sf_phys_col2]
                        .bitcast(cutlass.Float8E4M3FN)
                        .to(cutlass.Float32)
                    )

                group_up = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                group_gate = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                group_up2 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                group_gate2 = cutlass.vector.full((4,), 0.0, dtype=cutlass.Float16)
                xword = x64[token * SCALE_COLS + scale_col]
                expert_row_base = (
                    expert * (intermediate * (2 if gated else 1)) * SCALE_COLS
                )
                upword = w64[expert_row_base + row * SCALE_COLS + scale_col]
                if cutlass.const_expr(gated):
                    gateword = w64[
                        expert_row_base + (intermediate + row) * SCALE_COLS + scale_col
                    ]
                upword2 = w64[expert_row_base + row2 * SCALE_COLS + scale_col]
                if cutlass.const_expr(gated):
                    gateword2 = w64[
                        expert_row_base + (intermediate + row2) * SCALE_COLS + scale_col
                    ]
                for byte_in_group in range(4):
                    byte_shift = byte_in_group * 16
                    xb = (xword >> byte_shift).to(cutlass.Uint16)
                    wub = (upword >> byte_shift).to(cutlass.Uint16)
                    if cutlass.const_expr(gated):
                        wgb = (gateword >> byte_shift).to(cutlass.Uint16)
                    wub2 = (upword2 >> byte_shift).to(cutlass.Uint16)
                    if cutlass.const_expr(gated):
                        wgb2 = (gateword2 >> byte_shift).to(cutlass.Uint16)

                    xv = _fp4_quad(xb)
                    wuv = _fp4_quad(wub)
                    if cutlass.const_expr(gated):
                        wgv = _fp4_quad(wgb)
                    wuv2 = _fp4_quad(wub2)
                    if cutlass.const_expr(gated):
                        wgv2 = _fp4_quad(wgb2)
                    group_up = group_up + xv * wuv
                    if cutlass.const_expr(gated):
                        group_gate = group_gate + xv * wgv
                    group_up2 = group_up2 + xv * wuv2
                    if cutlass.const_expr(gated):
                        group_gate2 = group_gate2 + xv * wgv2

                acc_up = acc_up + group_up.to(cutlass.Float32).reduce("add") * (
                    xs * ws_up
                )
                if cutlass.const_expr(gated):
                    acc_gate = acc_gate + group_gate.to(cutlass.Float32).reduce(
                        "add"
                    ) * (xs * ws_gate)
                acc_up2 = acc_up2 + group_up2.to(cutlass.Float32).reduce("add") * (
                    xs * ws_up2
                )
                if cutlass.const_expr(gated):
                    acc_gate2 = acc_gate2 + group_gate2.to(cutlass.Float32).reduce(
                        "add"
                    ) * (xs * ws_gate2)

            up = cute.arch.warp_reduction_sum(acc_up)
            gate = cutlass.Float32(0.0)
            if cutlass.const_expr(gated):
                gate = cute.arch.warp_reduction_sum(acc_gate)
            up2 = cute.arch.warp_reduction_sum(acc_up2)
            gate2 = cutlass.Float32(0.0)
            if cutlass.const_expr(gated):
                gate2 = cute.arch.warp_reduction_sum(acc_gate2)
            if lane == 0:
                a = alpha[expert]
                up = up * a
                gate = gate * a
                value = apply_activation(up, gate, activation, True)
                intermediate_values[warp * 2] = value.to(cutlass.BFloat16).to(
                    cutlass.Float32
                )
                up2 = up2 * a
                gate2 = gate2 * a
                value2 = apply_activation(up2, gate2, activation, True)
                intermediate_values[warp * 2 + 1] = value2.to(cutlass.BFloat16).to(
                    cutlass.Float32
                )
        else:
            if lane == 0:
                intermediate_values[warp * 2] = cutlass.Float32(0.0)
                intermediate_values[warp * 2 + 1] = cutlass.Float32(0.0)
    else:
        if lane == 0:
            intermediate_values[warp * 2] = cutlass.Float32(0.0)
            intermediate_values[warp * 2 + 1] = cutlass.Float32(0.0)

    cute.arch.sync_threads()
    if tid == 0:
        values = cute.make_rmem_tensor((16,), cutlass.Float32)
        maximum = cutlass.Float32(0.0)
        for j in cutlass.range_constexpr(16):
            values[j] = intermediate_values[j]
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
        qcol = block - route * row_tiles
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
    intermediate = cute.size(weights, mode=[1]) // (2 if _GATED else 1)
    blocks = routes * (intermediate // (WARPS * 2))
    _routed_fc1_kernel(
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


_routed_fc1_kernel.set_name_prefix("flashinfer_nvfp4_fma", remove_cutlass_symbol=True)
