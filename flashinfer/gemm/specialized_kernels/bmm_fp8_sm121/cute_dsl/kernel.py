# mypy: disable-error-code="valid-type"
from __future__ import annotations

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

CUTE_COMPILE = cute.compile

WARP_THREADS = 32
SIMT_VEC = 8
SIMT_COLS = 8
SIMT_VEC16 = 16

_MMA_T_COMPILED = None
_MMA_T8_COMPILED = None
_MMA_S_COMPILED = None
_MMA_L_COMPILED = None
_SIMT8_COMPILED = None
_SIMT1_V16_COMPILED = None
_SIMT2_COMPILED = None

BKW = 8
BKW_PAD = 9


@dsl_user_op
def _mma_m16n8k32_e4m3(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    *,
    loc=None,
    ip=None,
):
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            a0.ir_value(loc=loc, ip=ip),
            a1.ir_value(loc=loc, ip=ip),
            a2.ir_value(loc=loc, ip=ip),
            a3.ir_value(loc=loc, ip=ip),
            b0.ir_value(loc=loc, ip=ip),
            b1.ir_value(loc=loc, ip=ip),
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{$0, $1, $2, $3}, "
        "{$4, $5, $6, $7}, "
        "{$8, $9}, "
        "{$10, $11, $12, $13};\n\t"
        "}",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [0])),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [1])),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [2])),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [3])),
    )


# ---- Tiny MMA: 1-warp, 16x32 register tile, 4 MMAs per K-iter ----
T_M_TILE = 16
T_N_TILE = 32
T8_N_TILE = 8
T_K_TILE = 32

# ---- Med MMA: 1-warp, 32x32 register tile, 8 MMAs per K-iter (A+B reuse) ----
M_M_TILE = 32
M_N_TILE = 32
M_K_TILE = 32

_MMA_M_COMPILED = None


@cute.kernel
def _bmm_fp8_mma_kernel_tiny(
    a: cute.Tensor,
    b: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    c: cute.Tensor,
    m_size: cutlass.Int32,
    k_tiles: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    tile_m, tile_n, _ = cute.arch.block_idx()

    group = lane >> 2
    thread_in_group = lane & 3
    row0 = tile_m * T_M_TILE + group
    row1 = row0 + 8
    out_col_base = tile_n * T_N_TILE

    d00 = cutlass.Float32(0.0)
    d01 = cutlass.Float32(0.0)
    d02 = cutlass.Float32(0.0)
    d03 = cutlass.Float32(0.0)
    d10 = cutlass.Float32(0.0)
    d11 = cutlass.Float32(0.0)
    d12 = cutlass.Float32(0.0)
    d13 = cutlass.Float32(0.0)
    d20 = cutlass.Float32(0.0)
    d21 = cutlass.Float32(0.0)
    d22 = cutlass.Float32(0.0)
    d23 = cutlass.Float32(0.0)
    d30 = cutlass.Float32(0.0)
    d31 = cutlass.Float32(0.0)
    d32 = cutlass.Float32(0.0)
    d33 = cutlass.Float32(0.0)
    zero = cutlass.Int32(0)

    bcol0 = out_col_base + group
    bcol1 = out_col_base + 8 + group
    bcol2 = out_col_base + 16 + group
    bcol3 = out_col_base + 24 + group

    for kt in cutlass.range(0, k_tiles, 1, unroll=4):
        k_word = kt * 8 + thread_in_group

        a0 = zero
        a1 = zero
        a2 = zero
        a3 = zero
        if cutlass.dynamic_expr(row0 < m_size):
            a0 = a[row0, k_word]
            a2 = a[row0, k_word + 4]
        if cutlass.dynamic_expr(row1 < m_size):
            a1 = a[row1, k_word]
            a3 = a[row1, k_word + 4]

        bb00 = b[bcol0, k_word]
        bb01 = b[bcol0, k_word + 4]
        bb10 = b[bcol1, k_word]
        bb11 = b[bcol1, k_word + 4]
        bb20 = b[bcol2, k_word]
        bb21 = b[bcol2, k_word + 4]
        bb30 = b[bcol3, k_word]
        bb31 = b[bcol3, k_word + 4]

        d00, d01, d02, d03 = _mma_m16n8k32_e4m3(
            a0, a1, a2, a3, bb00, bb01, d00, d01, d02, d03
        )
        d10, d11, d12, d13 = _mma_m16n8k32_e4m3(
            a0, a1, a2, a3, bb10, bb11, d10, d11, d12, d13
        )
        d20, d21, d22, d23 = _mma_m16n8k32_e4m3(
            a0, a1, a2, a3, bb20, bb21, d20, d21, d22, d23
        )
        d30, d31, d32, d33 = _mma_m16n8k32_e4m3(
            a0, a1, a2, a3, bb30, bb31, d30, d31, d32, d33
        )

    scale = sa[0] * sb[0]
    oc0 = out_col_base + thread_in_group * 2
    oc1 = oc0 + 1
    oc2 = out_col_base + 8 + thread_in_group * 2
    oc3 = oc2 + 1
    oc4 = out_col_base + 16 + thread_in_group * 2
    oc5 = oc4 + 1
    oc6 = out_col_base + 24 + thread_in_group * 2
    oc7 = oc6 + 1
    if cutlass.dynamic_expr(row0 < m_size):
        c[row0, oc0] = (d00 * scale).to(cutlass.BFloat16)
        c[row0, oc1] = (d01 * scale).to(cutlass.BFloat16)
        c[row0, oc2] = (d10 * scale).to(cutlass.BFloat16)
        c[row0, oc3] = (d11 * scale).to(cutlass.BFloat16)
        c[row0, oc4] = (d20 * scale).to(cutlass.BFloat16)
        c[row0, oc5] = (d21 * scale).to(cutlass.BFloat16)
        c[row0, oc6] = (d30 * scale).to(cutlass.BFloat16)
        c[row0, oc7] = (d31 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(row1 < m_size):
        c[row1, oc0] = (d02 * scale).to(cutlass.BFloat16)
        c[row1, oc1] = (d03 * scale).to(cutlass.BFloat16)
        c[row1, oc2] = (d12 * scale).to(cutlass.BFloat16)
        c[row1, oc3] = (d13 * scale).to(cutlass.BFloat16)
        c[row1, oc4] = (d22 * scale).to(cutlass.BFloat16)
        c[row1, oc5] = (d23 * scale).to(cutlass.BFloat16)
        c[row1, oc6] = (d32 * scale).to(cutlass.BFloat16)
        c[row1, oc7] = (d33 * scale).to(cutlass.BFloat16)


@cute.kernel
def _bmm_fp8_mma_kernel_tiny8(
    a: cute.Tensor,
    b: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    c: cute.Tensor,
    m_size: cutlass.Int32,
    k_tiles: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    tile_m, tile_n, _ = cute.arch.block_idx()

    group = lane >> 2
    thread_in_group = lane & 3
    row0 = tile_m * T_M_TILE + group
    row1 = row0 + 8
    col = tile_n * T8_N_TILE + group
    out_col0 = tile_n * T8_N_TILE + thread_in_group * 2
    out_col1 = out_col0 + 1

    d0 = cutlass.Float32(0.0)
    d1 = cutlass.Float32(0.0)
    d2 = cutlass.Float32(0.0)
    d3 = cutlass.Float32(0.0)
    zero = cutlass.Int32(0)

    for kt in cutlass.range(0, k_tiles, 1, unroll=8):
        k_word = kt * 8 + thread_in_group

        a0 = zero
        a1 = zero
        a2 = zero
        a3 = zero
        if cutlass.dynamic_expr(row0 < m_size):
            a0 = a[row0, k_word]
            a2 = a[row0, k_word + 4]
        if cutlass.dynamic_expr(row1 < m_size):
            a1 = a[row1, k_word]
            a3 = a[row1, k_word + 4]

        b0 = b[col, k_word]
        b1 = b[col, k_word + 4]
        d0, d1, d2, d3 = _mma_m16n8k32_e4m3(a0, a1, a2, a3, b0, b1, d0, d1, d2, d3)

    scale = sa[0] * sb[0]
    if cutlass.dynamic_expr(row0 < m_size):
        c[row0, out_col0] = (d0 * scale).to(cutlass.BFloat16)
        c[row0, out_col1] = (d1 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(row1 < m_size):
        c[row1, out_col0] = (d2 * scale).to(cutlass.BFloat16)
        c[row1, out_col1] = (d3 * scale).to(cutlass.BFloat16)


# ---- Med MMA: 1-warp, 32x32 register tile, 8 MMAs per K-iter ----
@cute.kernel
def _bmm_fp8_mma_kernel_med(
    a: cute.Tensor,
    b: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    c: cute.Tensor,
    m_size: cutlass.Int32,
    k_tiles: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    tile_m, tile_n, _ = cute.arch.block_idx()

    group = lane >> 2
    thread_in_group = lane & 3
    base_m = tile_m * M_M_TILE
    out_col_base = tile_n * M_N_TILE

    # A row groups: 0..15 (rg0) and 16..31 (rg1)
    rg0_row0 = base_m + group  # row in 0..7
    rg0_row1 = rg0_row0 + 8  # row in 8..15
    rg1_row0 = base_m + 16 + group  # row in 16..23
    rg1_row1 = rg1_row0 + 8  # row in 24..31

    # 8 outputs per row-group (4 col groups × 2 cols-per-thread)
    z = cutlass.Float32(0.0)
    # row group 0
    d000 = z
    d001 = z
    d002 = z
    d003 = z
    d010 = z
    d011 = z
    d012 = z
    d013 = z
    d020 = z
    d021 = z
    d022 = z
    d023 = z
    d030 = z
    d031 = z
    d032 = z
    d033 = z
    # row group 1
    d100 = z
    d101 = z
    d102 = z
    d103 = z
    d110 = z
    d111 = z
    d112 = z
    d113 = z
    d120 = z
    d121 = z
    d122 = z
    d123 = z
    d130 = z
    d131 = z
    d132 = z
    d133 = z
    zero = cutlass.Int32(0)

    bcol0 = out_col_base + group
    bcol1 = out_col_base + 8 + group
    bcol2 = out_col_base + 16 + group
    bcol3 = out_col_base + 24 + group

    for kt in cutlass.range(0, k_tiles, 1, unroll=4):
        k_word = kt * 8 + thread_in_group

        # Row group 0
        a0 = zero
        a1 = zero
        a2 = zero
        a3 = zero
        if cutlass.dynamic_expr(rg0_row0 < m_size):
            a0 = a[rg0_row0, k_word]
            a2 = a[rg0_row0, k_word + 4]
        if cutlass.dynamic_expr(rg0_row1 < m_size):
            a1 = a[rg0_row1, k_word]
            a3 = a[rg0_row1, k_word + 4]
        # Row group 1
        a4 = zero
        a5 = zero
        a6 = zero
        a7 = zero
        if cutlass.dynamic_expr(rg1_row0 < m_size):
            a4 = a[rg1_row0, k_word]
            a6 = a[rg1_row0, k_word + 4]
        if cutlass.dynamic_expr(rg1_row1 < m_size):
            a5 = a[rg1_row1, k_word]
            a7 = a[rg1_row1, k_word + 4]

        bb00 = b[bcol0, k_word]
        bb01 = b[bcol0, k_word + 4]
        bb10 = b[bcol1, k_word]
        bb11 = b[bcol1, k_word + 4]
        bb20 = b[bcol2, k_word]
        bb21 = b[bcol2, k_word + 4]
        bb30 = b[bcol3, k_word]
        bb31 = b[bcol3, k_word + 4]

        # Row group 0 × 4 col groups
        d000, d001, d002, d003 = _mma_m16n8k32_e4m3(
            a0, a1, a2, a3, bb00, bb01, d000, d001, d002, d003
        )
        d010, d011, d012, d013 = _mma_m16n8k32_e4m3(
            a0, a1, a2, a3, bb10, bb11, d010, d011, d012, d013
        )
        d020, d021, d022, d023 = _mma_m16n8k32_e4m3(
            a0, a1, a2, a3, bb20, bb21, d020, d021, d022, d023
        )
        d030, d031, d032, d033 = _mma_m16n8k32_e4m3(
            a0, a1, a2, a3, bb30, bb31, d030, d031, d032, d033
        )
        # Row group 1 × 4 col groups (B reuse)
        d100, d101, d102, d103 = _mma_m16n8k32_e4m3(
            a4, a5, a6, a7, bb00, bb01, d100, d101, d102, d103
        )
        d110, d111, d112, d113 = _mma_m16n8k32_e4m3(
            a4, a5, a6, a7, bb10, bb11, d110, d111, d112, d113
        )
        d120, d121, d122, d123 = _mma_m16n8k32_e4m3(
            a4, a5, a6, a7, bb20, bb21, d120, d121, d122, d123
        )
        d130, d131, d132, d133 = _mma_m16n8k32_e4m3(
            a4, a5, a6, a7, bb30, bb31, d130, d131, d132, d133
        )

    scale = sa[0] * sb[0]
    oc0 = out_col_base + thread_in_group * 2
    oc1 = oc0 + 1
    oc2 = out_col_base + 8 + thread_in_group * 2
    oc3 = oc2 + 1
    oc4 = out_col_base + 16 + thread_in_group * 2
    oc5 = oc4 + 1
    oc6 = out_col_base + 24 + thread_in_group * 2
    oc7 = oc6 + 1
    # Row group 0
    if cutlass.dynamic_expr(rg0_row0 < m_size):
        c[rg0_row0, oc0] = (d000 * scale).to(cutlass.BFloat16)
        c[rg0_row0, oc1] = (d001 * scale).to(cutlass.BFloat16)
        c[rg0_row0, oc2] = (d010 * scale).to(cutlass.BFloat16)
        c[rg0_row0, oc3] = (d011 * scale).to(cutlass.BFloat16)
        c[rg0_row0, oc4] = (d020 * scale).to(cutlass.BFloat16)
        c[rg0_row0, oc5] = (d021 * scale).to(cutlass.BFloat16)
        c[rg0_row0, oc6] = (d030 * scale).to(cutlass.BFloat16)
        c[rg0_row0, oc7] = (d031 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(rg0_row1 < m_size):
        c[rg0_row1, oc0] = (d002 * scale).to(cutlass.BFloat16)
        c[rg0_row1, oc1] = (d003 * scale).to(cutlass.BFloat16)
        c[rg0_row1, oc2] = (d012 * scale).to(cutlass.BFloat16)
        c[rg0_row1, oc3] = (d013 * scale).to(cutlass.BFloat16)
        c[rg0_row1, oc4] = (d022 * scale).to(cutlass.BFloat16)
        c[rg0_row1, oc5] = (d023 * scale).to(cutlass.BFloat16)
        c[rg0_row1, oc6] = (d032 * scale).to(cutlass.BFloat16)
        c[rg0_row1, oc7] = (d033 * scale).to(cutlass.BFloat16)
    # Row group 1
    if cutlass.dynamic_expr(rg1_row0 < m_size):
        c[rg1_row0, oc0] = (d100 * scale).to(cutlass.BFloat16)
        c[rg1_row0, oc1] = (d101 * scale).to(cutlass.BFloat16)
        c[rg1_row0, oc2] = (d110 * scale).to(cutlass.BFloat16)
        c[rg1_row0, oc3] = (d111 * scale).to(cutlass.BFloat16)
        c[rg1_row0, oc4] = (d120 * scale).to(cutlass.BFloat16)
        c[rg1_row0, oc5] = (d121 * scale).to(cutlass.BFloat16)
        c[rg1_row0, oc6] = (d130 * scale).to(cutlass.BFloat16)
        c[rg1_row0, oc7] = (d131 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(rg1_row1 < m_size):
        c[rg1_row1, oc0] = (d102 * scale).to(cutlass.BFloat16)
        c[rg1_row1, oc1] = (d103 * scale).to(cutlass.BFloat16)
        c[rg1_row1, oc2] = (d112 * scale).to(cutlass.BFloat16)
        c[rg1_row1, oc3] = (d113 * scale).to(cutlass.BFloat16)
        c[rg1_row1, oc4] = (d122 * scale).to(cutlass.BFloat16)
        c[rg1_row1, oc5] = (d123 * scale).to(cutlass.BFloat16)
        c[rg1_row1, oc6] = (d132 * scale).to(cutlass.BFloat16)
        c[rg1_row1, oc7] = (d133 * scale).to(cutlass.BFloat16)


# ---- Small MMA: 4 warps 2x2, BM=64 BN=64, SMEM tiled, 2K-depth ----
BM_S = 64
BN_S = 64


@cute.kernel
def _bmm_fp8_mma_kernel_small(
    a: cute.Tensor,
    b: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    c: cute.Tensor,
    m_size: cutlass.Int32,
    k_tiles: cutlass.Int32,
):
    tid_x, _, _ = cute.arch.thread_idx()
    bid_m, bid_n, _ = cute.arch.block_idx()
    tid = tid_x
    warp = tid >> 5
    lane = tid & 31
    warp_m = warp >> 1
    warp_n = warp & 1
    ldiv4 = lane >> 2
    lmod4 = lane & 3

    smem = cutlass.utils.SmemAllocator()
    sA0_ptr = smem.allocate_array(cutlass.Int32, BM_S * BKW_PAD)
    sB0_ptr = smem.allocate_array(cutlass.Int32, BN_S * BKW_PAD)
    sA1_ptr = smem.allocate_array(cutlass.Int32, BM_S * BKW_PAD)
    sB1_ptr = smem.allocate_array(cutlass.Int32, BN_S * BKW_PAD)
    sA0 = cute.make_tensor(sA0_ptr, cute.make_layout((BM_S, BKW), stride=(BKW_PAD, 1)))
    sB0 = cute.make_tensor(sB0_ptr, cute.make_layout((BN_S, BKW), stride=(BKW_PAD, 1)))
    sA1 = cute.make_tensor(sA1_ptr, cute.make_layout((BM_S, BKW), stride=(BKW_PAD, 1)))
    sB1 = cute.make_tensor(sB1_ptr, cute.make_layout((BN_S, BKW), stride=(BKW_PAD, 1)))

    d000 = cutlass.Float32(0.0)
    d001 = cutlass.Float32(0.0)
    d002 = cutlass.Float32(0.0)
    d003 = cutlass.Float32(0.0)
    d010 = cutlass.Float32(0.0)
    d011 = cutlass.Float32(0.0)
    d012 = cutlass.Float32(0.0)
    d013 = cutlass.Float32(0.0)
    d020 = cutlass.Float32(0.0)
    d021 = cutlass.Float32(0.0)
    d022 = cutlass.Float32(0.0)
    d023 = cutlass.Float32(0.0)
    d030 = cutlass.Float32(0.0)
    d031 = cutlass.Float32(0.0)
    d032 = cutlass.Float32(0.0)
    d033 = cutlass.Float32(0.0)
    d100 = cutlass.Float32(0.0)
    d101 = cutlass.Float32(0.0)
    d102 = cutlass.Float32(0.0)
    d103 = cutlass.Float32(0.0)
    d110 = cutlass.Float32(0.0)
    d111 = cutlass.Float32(0.0)
    d112 = cutlass.Float32(0.0)
    d113 = cutlass.Float32(0.0)
    d120 = cutlass.Float32(0.0)
    d121 = cutlass.Float32(0.0)
    d122 = cutlass.Float32(0.0)
    d123 = cutlass.Float32(0.0)
    d130 = cutlass.Float32(0.0)
    d131 = cutlass.Float32(0.0)
    d132 = cutlass.Float32(0.0)
    d133 = cutlass.Float32(0.0)

    row_load = tid >> 1
    col_load_start = (tid & 1) << 2
    base_m = bid_m * BM_S
    base_n = bid_n * BN_S

    for kt in cutlass.range(0, k_tiles, 2, unroll=2):
        # Load K-block 0 into sA0/sB0
        ka_base = kt * BKW + col_load_start
        if cutlass.dynamic_expr(base_m + row_load < m_size):
            sA0[row_load, col_load_start + 0] = a[base_m + row_load, ka_base + 0]
            sA0[row_load, col_load_start + 1] = a[base_m + row_load, ka_base + 1]
            sA0[row_load, col_load_start + 2] = a[base_m + row_load, ka_base + 2]
            sA0[row_load, col_load_start + 3] = a[base_m + row_load, ka_base + 3]
        else:
            zz = cutlass.Int32(0)
            sA0[row_load, col_load_start + 0] = zz
            sA0[row_load, col_load_start + 1] = zz
            sA0[row_load, col_load_start + 2] = zz
            sA0[row_load, col_load_start + 3] = zz
        sB0[row_load, col_load_start + 0] = b[base_n + row_load, ka_base + 0]
        sB0[row_load, col_load_start + 1] = b[base_n + row_load, ka_base + 1]
        sB0[row_load, col_load_start + 2] = b[base_n + row_load, ka_base + 2]
        sB0[row_load, col_load_start + 3] = b[base_n + row_load, ka_base + 3]

        # Load K-block 1 into sA1/sB1
        ka_base2 = (kt + 1) * BKW + col_load_start
        if cutlass.dynamic_expr(base_m + row_load < m_size):
            sA1[row_load, col_load_start + 0] = a[base_m + row_load, ka_base2 + 0]
            sA1[row_load, col_load_start + 1] = a[base_m + row_load, ka_base2 + 1]
            sA1[row_load, col_load_start + 2] = a[base_m + row_load, ka_base2 + 2]
            sA1[row_load, col_load_start + 3] = a[base_m + row_load, ka_base2 + 3]
        else:
            zz = cutlass.Int32(0)
            sA1[row_load, col_load_start + 0] = zz
            sA1[row_load, col_load_start + 1] = zz
            sA1[row_load, col_load_start + 2] = zz
            sA1[row_load, col_load_start + 3] = zz
        sB1[row_load, col_load_start + 0] = b[base_n + row_load, ka_base2 + 0]
        sB1[row_load, col_load_start + 1] = b[base_n + row_load, ka_base2 + 1]
        sB1[row_load, col_load_start + 2] = b[base_n + row_load, ka_base2 + 2]
        sB1[row_load, col_load_start + 3] = b[base_n + row_load, ka_base2 + 3]

        cute.arch.barrier()

        # Compute from K-block 0
        rb0 = warp_m * 32
        rb1 = warp_m * 32 + 16
        a000 = sA0[rb0 + ldiv4, lmod4]
        a001 = sA0[rb0 + ldiv4 + 8, lmod4]
        a002 = sA0[rb0 + ldiv4, lmod4 + 4]
        a003 = sA0[rb0 + ldiv4 + 8, lmod4 + 4]
        a100 = sA0[rb1 + ldiv4, lmod4]
        a101 = sA0[rb1 + ldiv4 + 8, lmod4]
        a102 = sA0[rb1 + ldiv4, lmod4 + 4]
        a103 = sA0[rb1 + ldiv4 + 8, lmod4 + 4]
        cb0 = warp_n * 32
        cb1 = warp_n * 32 + 8
        cb2 = warp_n * 32 + 16
        cb3 = warp_n * 32 + 24
        b00 = sB0[cb0 + ldiv4, lmod4]
        b01 = sB0[cb0 + ldiv4, lmod4 + 4]
        b10 = sB0[cb1 + ldiv4, lmod4]
        b11 = sB0[cb1 + ldiv4, lmod4 + 4]
        b20 = sB0[cb2 + ldiv4, lmod4]
        b21 = sB0[cb2 + ldiv4, lmod4 + 4]
        b30 = sB0[cb3 + ldiv4, lmod4]
        b31 = sB0[cb3 + ldiv4, lmod4 + 4]

        d000, d001, d002, d003 = _mma_m16n8k32_e4m3(
            a000, a001, a002, a003, b00, b01, d000, d001, d002, d003
        )
        d010, d011, d012, d013 = _mma_m16n8k32_e4m3(
            a000, a001, a002, a003, b10, b11, d010, d011, d012, d013
        )
        d020, d021, d022, d023 = _mma_m16n8k32_e4m3(
            a000, a001, a002, a003, b20, b21, d020, d021, d022, d023
        )
        d030, d031, d032, d033 = _mma_m16n8k32_e4m3(
            a000, a001, a002, a003, b30, b31, d030, d031, d032, d033
        )
        d100, d101, d102, d103 = _mma_m16n8k32_e4m3(
            a100, a101, a102, a103, b00, b01, d100, d101, d102, d103
        )
        d110, d111, d112, d113 = _mma_m16n8k32_e4m3(
            a100, a101, a102, a103, b10, b11, d110, d111, d112, d113
        )
        d120, d121, d122, d123 = _mma_m16n8k32_e4m3(
            a100, a101, a102, a103, b20, b21, d120, d121, d122, d123
        )
        d130, d131, d132, d133 = _mma_m16n8k32_e4m3(
            a100, a101, a102, a103, b30, b31, d130, d131, d132, d133
        )

        # Compute from K-block 1
        a000 = sA1[rb0 + ldiv4, lmod4]
        a001 = sA1[rb0 + ldiv4 + 8, lmod4]
        a002 = sA1[rb0 + ldiv4, lmod4 + 4]
        a003 = sA1[rb0 + ldiv4 + 8, lmod4 + 4]
        a100 = sA1[rb1 + ldiv4, lmod4]
        a101 = sA1[rb1 + ldiv4 + 8, lmod4]
        a102 = sA1[rb1 + ldiv4, lmod4 + 4]
        a103 = sA1[rb1 + ldiv4 + 8, lmod4 + 4]
        b00 = sB1[cb0 + ldiv4, lmod4]
        b01 = sB1[cb0 + ldiv4, lmod4 + 4]
        b10 = sB1[cb1 + ldiv4, lmod4]
        b11 = sB1[cb1 + ldiv4, lmod4 + 4]
        b20 = sB1[cb2 + ldiv4, lmod4]
        b21 = sB1[cb2 + ldiv4, lmod4 + 4]
        b30 = sB1[cb3 + ldiv4, lmod4]
        b31 = sB1[cb3 + ldiv4, lmod4 + 4]

        d000, d001, d002, d003 = _mma_m16n8k32_e4m3(
            a000, a001, a002, a003, b00, b01, d000, d001, d002, d003
        )
        d010, d011, d012, d013 = _mma_m16n8k32_e4m3(
            a000, a001, a002, a003, b10, b11, d010, d011, d012, d013
        )
        d020, d021, d022, d023 = _mma_m16n8k32_e4m3(
            a000, a001, a002, a003, b20, b21, d020, d021, d022, d023
        )
        d030, d031, d032, d033 = _mma_m16n8k32_e4m3(
            a000, a001, a002, a003, b30, b31, d030, d031, d032, d033
        )
        d100, d101, d102, d103 = _mma_m16n8k32_e4m3(
            a100, a101, a102, a103, b00, b01, d100, d101, d102, d103
        )
        d110, d111, d112, d113 = _mma_m16n8k32_e4m3(
            a100, a101, a102, a103, b10, b11, d110, d111, d112, d113
        )
        d120, d121, d122, d123 = _mma_m16n8k32_e4m3(
            a100, a101, a102, a103, b20, b21, d120, d121, d122, d123
        )
        d130, d131, d132, d133 = _mma_m16n8k32_e4m3(
            a100, a101, a102, a103, b30, b31, d130, d131, d132, d133
        )

        cute.arch.barrier()

    scale = sa[0] * sb[0]
    cta_m = bid_m * BM_S + warp_m * 32
    cta_n = bid_n * BN_S + warp_n * 32
    row0 = cta_m + ldiv4
    row8 = cta_m + ldiv4 + 8
    cn0 = cta_n + lmod4 * 2
    cn1 = cta_n + 8 + lmod4 * 2
    cn2 = cta_n + 16 + lmod4 * 2
    cn3 = cta_n + 24 + lmod4 * 2
    if cutlass.dynamic_expr(row0 < m_size):
        c[row0, cn0 + 0] = (d000 * scale).to(cutlass.BFloat16)
        c[row0, cn0 + 1] = (d001 * scale).to(cutlass.BFloat16)
        c[row0, cn1 + 0] = (d010 * scale).to(cutlass.BFloat16)
        c[row0, cn1 + 1] = (d011 * scale).to(cutlass.BFloat16)
        c[row0, cn2 + 0] = (d020 * scale).to(cutlass.BFloat16)
        c[row0, cn2 + 1] = (d021 * scale).to(cutlass.BFloat16)
        c[row0, cn3 + 0] = (d030 * scale).to(cutlass.BFloat16)
        c[row0, cn3 + 1] = (d031 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(row8 < m_size):
        c[row8, cn0 + 0] = (d002 * scale).to(cutlass.BFloat16)
        c[row8, cn0 + 1] = (d003 * scale).to(cutlass.BFloat16)
        c[row8, cn1 + 0] = (d012 * scale).to(cutlass.BFloat16)
        c[row8, cn1 + 1] = (d013 * scale).to(cutlass.BFloat16)
        c[row8, cn2 + 0] = (d022 * scale).to(cutlass.BFloat16)
        c[row8, cn2 + 1] = (d023 * scale).to(cutlass.BFloat16)
        c[row8, cn3 + 0] = (d032 * scale).to(cutlass.BFloat16)
        c[row8, cn3 + 1] = (d033 * scale).to(cutlass.BFloat16)
    row16 = cta_m + 16 + ldiv4
    row24 = cta_m + 16 + ldiv4 + 8
    if cutlass.dynamic_expr(row16 < m_size):
        c[row16, cn0 + 0] = (d100 * scale).to(cutlass.BFloat16)
        c[row16, cn0 + 1] = (d101 * scale).to(cutlass.BFloat16)
        c[row16, cn1 + 0] = (d110 * scale).to(cutlass.BFloat16)
        c[row16, cn1 + 1] = (d111 * scale).to(cutlass.BFloat16)
        c[row16, cn2 + 0] = (d120 * scale).to(cutlass.BFloat16)
        c[row16, cn2 + 1] = (d121 * scale).to(cutlass.BFloat16)
        c[row16, cn3 + 0] = (d130 * scale).to(cutlass.BFloat16)
        c[row16, cn3 + 1] = (d131 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(row24 < m_size):
        c[row24, cn0 + 0] = (d102 * scale).to(cutlass.BFloat16)
        c[row24, cn0 + 1] = (d103 * scale).to(cutlass.BFloat16)
        c[row24, cn1 + 0] = (d112 * scale).to(cutlass.BFloat16)
        c[row24, cn1 + 1] = (d113 * scale).to(cutlass.BFloat16)
        c[row24, cn2 + 0] = (d122 * scale).to(cutlass.BFloat16)
        c[row24, cn2 + 1] = (d123 * scale).to(cutlass.BFloat16)
        c[row24, cn3 + 0] = (d132 * scale).to(cutlass.BFloat16)
        c[row24, cn3 + 1] = (d133 * scale).to(cutlass.BFloat16)


# ---- Large MMA: 8 warps 2x4, BM=128 BN=128, SMEM tiled, 2K-depth ----
BM_L = 128
BN_L = 128


@cute.kernel
def _bmm_fp8_mma_kernel_large(
    a: cute.Tensor,
    b: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    c: cute.Tensor,
    m_size: cutlass.Int32,
    k_tiles: cutlass.Int32,
):
    tid_x, _, _ = cute.arch.thread_idx()
    bid_m, bid_n, _ = cute.arch.block_idx()
    tid = tid_x
    warp = tid >> 5
    lane = tid & 31
    warp_m = warp >> 2
    warp_n = warp & 3
    ldiv4 = lane >> 2
    lmod4 = lane & 3

    smem = cutlass.utils.SmemAllocator()
    sA0_ptr = smem.allocate_array(cutlass.Int32, BM_L * BKW_PAD)
    sB0_ptr = smem.allocate_array(cutlass.Int32, BN_L * BKW_PAD)
    sA1_ptr = smem.allocate_array(cutlass.Int32, BM_L * BKW_PAD)
    sB1_ptr = smem.allocate_array(cutlass.Int32, BN_L * BKW_PAD)
    sA0 = cute.make_tensor(sA0_ptr, cute.make_layout((BM_L, BKW), stride=(BKW_PAD, 1)))
    sB0 = cute.make_tensor(sB0_ptr, cute.make_layout((BN_L, BKW), stride=(BKW_PAD, 1)))
    sA1 = cute.make_tensor(sA1_ptr, cute.make_layout((BM_L, BKW), stride=(BKW_PAD, 1)))
    sB1 = cute.make_tensor(sB1_ptr, cute.make_layout((BN_L, BKW), stride=(BKW_PAD, 1)))

    z = cutlass.Float32(0.0)
    d000 = z
    d001 = z
    d002 = z
    d003 = z
    d010 = z
    d011 = z
    d012 = z
    d013 = z
    d020 = z
    d021 = z
    d022 = z
    d023 = z
    d030 = z
    d031 = z
    d032 = z
    d033 = z
    d100 = z
    d101 = z
    d102 = z
    d103 = z
    d110 = z
    d111 = z
    d112 = z
    d113 = z
    d120 = z
    d121 = z
    d122 = z
    d123 = z
    d130 = z
    d131 = z
    d132 = z
    d133 = z
    d200 = z
    d201 = z
    d202 = z
    d203 = z
    d210 = z
    d211 = z
    d212 = z
    d213 = z
    d220 = z
    d221 = z
    d222 = z
    d223 = z
    d230 = z
    d231 = z
    d232 = z
    d233 = z
    d300 = z
    d301 = z
    d302 = z
    d303 = z
    d310 = z
    d311 = z
    d312 = z
    d313 = z
    d320 = z
    d321 = z
    d322 = z
    d323 = z
    d330 = z
    d331 = z
    d332 = z
    d333 = z

    row_load = tid >> 1
    col_load_start = (tid & 1) << 2
    base_m = bid_m * BM_L
    base_n = bid_n * BN_L

    for kt in cutlass.range(0, k_tiles, 2, unroll=1):
        # Load K-block 0
        ka_base = kt * BKW + col_load_start
        if cutlass.dynamic_expr(base_m + row_load < m_size):
            sA0[row_load, col_load_start + 0] = a[base_m + row_load, ka_base + 0]
            sA0[row_load, col_load_start + 1] = a[base_m + row_load, ka_base + 1]
            sA0[row_load, col_load_start + 2] = a[base_m + row_load, ka_base + 2]
            sA0[row_load, col_load_start + 3] = a[base_m + row_load, ka_base + 3]
        else:
            zz = cutlass.Int32(0)
            sA0[row_load, col_load_start + 0] = zz
            sA0[row_load, col_load_start + 1] = zz
            sA0[row_load, col_load_start + 2] = zz
            sA0[row_load, col_load_start + 3] = zz
        sB0[row_load, col_load_start + 0] = b[base_n + row_load, ka_base + 0]
        sB0[row_load, col_load_start + 1] = b[base_n + row_load, ka_base + 1]
        sB0[row_load, col_load_start + 2] = b[base_n + row_load, ka_base + 2]
        sB0[row_load, col_load_start + 3] = b[base_n + row_load, ka_base + 3]

        # Load K-block 1
        ka_base2 = (kt + 1) * BKW + col_load_start
        if cutlass.dynamic_expr(base_m + row_load < m_size):
            sA1[row_load, col_load_start + 0] = a[base_m + row_load, ka_base2 + 0]
            sA1[row_load, col_load_start + 1] = a[base_m + row_load, ka_base2 + 1]
            sA1[row_load, col_load_start + 2] = a[base_m + row_load, ka_base2 + 2]
            sA1[row_load, col_load_start + 3] = a[base_m + row_load, ka_base2 + 3]
        else:
            zz = cutlass.Int32(0)
            sA1[row_load, col_load_start + 0] = zz
            sA1[row_load, col_load_start + 1] = zz
            sA1[row_load, col_load_start + 2] = zz
            sA1[row_load, col_load_start + 3] = zz
        sB1[row_load, col_load_start + 0] = b[base_n + row_load, ka_base2 + 0]
        sB1[row_load, col_load_start + 1] = b[base_n + row_load, ka_base2 + 1]
        sB1[row_load, col_load_start + 2] = b[base_n + row_load, ka_base2 + 2]
        sB1[row_load, col_load_start + 3] = b[base_n + row_load, ka_base2 + 3]

        cute.arch.barrier()

        # ---- Compute K-block 0 ----
        rb0 = warp_m * 64
        rb1 = warp_m * 64 + 16
        rb2 = warp_m * 64 + 32
        rb3 = warp_m * 64 + 48
        a00 = sA0[rb0 + ldiv4, lmod4]
        a01 = sA0[rb0 + ldiv4 + 8, lmod4]
        a02 = sA0[rb0 + ldiv4, lmod4 + 4]
        a03 = sA0[rb0 + ldiv4 + 8, lmod4 + 4]
        a10 = sA0[rb1 + ldiv4, lmod4]
        a11 = sA0[rb1 + ldiv4 + 8, lmod4]
        a12 = sA0[rb1 + ldiv4, lmod4 + 4]
        a13 = sA0[rb1 + ldiv4 + 8, lmod4 + 4]
        a20 = sA0[rb2 + ldiv4, lmod4]
        a21 = sA0[rb2 + ldiv4 + 8, lmod4]
        a22 = sA0[rb2 + ldiv4, lmod4 + 4]
        a23 = sA0[rb2 + ldiv4 + 8, lmod4 + 4]
        a30 = sA0[rb3 + ldiv4, lmod4]
        a31 = sA0[rb3 + ldiv4 + 8, lmod4]
        a32 = sA0[rb3 + ldiv4, lmod4 + 4]
        a33 = sA0[rb3 + ldiv4 + 8, lmod4 + 4]
        cb0 = warp_n * 32
        cb1 = warp_n * 32 + 8
        cb2 = warp_n * 32 + 16
        cb3 = warp_n * 32 + 24
        b00 = sB0[cb0 + ldiv4, lmod4]
        b01 = sB0[cb0 + ldiv4, lmod4 + 4]
        b10 = sB0[cb1 + ldiv4, lmod4]
        b11 = sB0[cb1 + ldiv4, lmod4 + 4]
        b20 = sB0[cb2 + ldiv4, lmod4]
        b21 = sB0[cb2 + ldiv4, lmod4 + 4]
        b30 = sB0[cb3 + ldiv4, lmod4]
        b31 = sB0[cb3 + ldiv4, lmod4 + 4]

        d000, d001, d002, d003 = _mma_m16n8k32_e4m3(
            a00, a01, a02, a03, b00, b01, d000, d001, d002, d003
        )
        d010, d011, d012, d013 = _mma_m16n8k32_e4m3(
            a00, a01, a02, a03, b10, b11, d010, d011, d012, d013
        )
        d020, d021, d022, d023 = _mma_m16n8k32_e4m3(
            a00, a01, a02, a03, b20, b21, d020, d021, d022, d023
        )
        d030, d031, d032, d033 = _mma_m16n8k32_e4m3(
            a00, a01, a02, a03, b30, b31, d030, d031, d032, d033
        )
        d100, d101, d102, d103 = _mma_m16n8k32_e4m3(
            a10, a11, a12, a13, b00, b01, d100, d101, d102, d103
        )
        d110, d111, d112, d113 = _mma_m16n8k32_e4m3(
            a10, a11, a12, a13, b10, b11, d110, d111, d112, d113
        )
        d120, d121, d122, d123 = _mma_m16n8k32_e4m3(
            a10, a11, a12, a13, b20, b21, d120, d121, d122, d123
        )
        d130, d131, d132, d133 = _mma_m16n8k32_e4m3(
            a10, a11, a12, a13, b30, b31, d130, d131, d132, d133
        )
        d200, d201, d202, d203 = _mma_m16n8k32_e4m3(
            a20, a21, a22, a23, b00, b01, d200, d201, d202, d203
        )
        d210, d211, d212, d213 = _mma_m16n8k32_e4m3(
            a20, a21, a22, a23, b10, b11, d210, d211, d212, d213
        )
        d220, d221, d222, d223 = _mma_m16n8k32_e4m3(
            a20, a21, a22, a23, b20, b21, d220, d221, d222, d223
        )
        d230, d231, d232, d233 = _mma_m16n8k32_e4m3(
            a20, a21, a22, a23, b30, b31, d230, d231, d232, d233
        )
        d300, d301, d302, d303 = _mma_m16n8k32_e4m3(
            a30, a31, a32, a33, b00, b01, d300, d301, d302, d303
        )
        d310, d311, d312, d313 = _mma_m16n8k32_e4m3(
            a30, a31, a32, a33, b10, b11, d310, d311, d312, d313
        )
        d320, d321, d322, d323 = _mma_m16n8k32_e4m3(
            a30, a31, a32, a33, b20, b21, d320, d321, d322, d323
        )
        d330, d331, d332, d333 = _mma_m16n8k32_e4m3(
            a30, a31, a32, a33, b30, b31, d330, d331, d332, d333
        )

        # ---- Compute K-block 1 ----
        a00 = sA1[rb0 + ldiv4, lmod4]
        a01 = sA1[rb0 + ldiv4 + 8, lmod4]
        a02 = sA1[rb0 + ldiv4, lmod4 + 4]
        a03 = sA1[rb0 + ldiv4 + 8, lmod4 + 4]
        a10 = sA1[rb1 + ldiv4, lmod4]
        a11 = sA1[rb1 + ldiv4 + 8, lmod4]
        a12 = sA1[rb1 + ldiv4, lmod4 + 4]
        a13 = sA1[rb1 + ldiv4 + 8, lmod4 + 4]
        a20 = sA1[rb2 + ldiv4, lmod4]
        a21 = sA1[rb2 + ldiv4 + 8, lmod4]
        a22 = sA1[rb2 + ldiv4, lmod4 + 4]
        a23 = sA1[rb2 + ldiv4 + 8, lmod4 + 4]
        a30 = sA1[rb3 + ldiv4, lmod4]
        a31 = sA1[rb3 + ldiv4 + 8, lmod4]
        a32 = sA1[rb3 + ldiv4, lmod4 + 4]
        a33 = sA1[rb3 + ldiv4 + 8, lmod4 + 4]
        b00 = sB1[cb0 + ldiv4, lmod4]
        b01 = sB1[cb0 + ldiv4, lmod4 + 4]
        b10 = sB1[cb1 + ldiv4, lmod4]
        b11 = sB1[cb1 + ldiv4, lmod4 + 4]
        b20 = sB1[cb2 + ldiv4, lmod4]
        b21 = sB1[cb2 + ldiv4, lmod4 + 4]
        b30 = sB1[cb3 + ldiv4, lmod4]
        b31 = sB1[cb3 + ldiv4, lmod4 + 4]

        d000, d001, d002, d003 = _mma_m16n8k32_e4m3(
            a00, a01, a02, a03, b00, b01, d000, d001, d002, d003
        )
        d010, d011, d012, d013 = _mma_m16n8k32_e4m3(
            a00, a01, a02, a03, b10, b11, d010, d011, d012, d013
        )
        d020, d021, d022, d023 = _mma_m16n8k32_e4m3(
            a00, a01, a02, a03, b20, b21, d020, d021, d022, d023
        )
        d030, d031, d032, d033 = _mma_m16n8k32_e4m3(
            a00, a01, a02, a03, b30, b31, d030, d031, d032, d033
        )
        d100, d101, d102, d103 = _mma_m16n8k32_e4m3(
            a10, a11, a12, a13, b00, b01, d100, d101, d102, d103
        )
        d110, d111, d112, d113 = _mma_m16n8k32_e4m3(
            a10, a11, a12, a13, b10, b11, d110, d111, d112, d113
        )
        d120, d121, d122, d123 = _mma_m16n8k32_e4m3(
            a10, a11, a12, a13, b20, b21, d120, d121, d122, d123
        )
        d130, d131, d132, d133 = _mma_m16n8k32_e4m3(
            a10, a11, a12, a13, b30, b31, d130, d131, d132, d133
        )
        d200, d201, d202, d203 = _mma_m16n8k32_e4m3(
            a20, a21, a22, a23, b00, b01, d200, d201, d202, d203
        )
        d210, d211, d212, d213 = _mma_m16n8k32_e4m3(
            a20, a21, a22, a23, b10, b11, d210, d211, d212, d213
        )
        d220, d221, d222, d223 = _mma_m16n8k32_e4m3(
            a20, a21, a22, a23, b20, b21, d220, d221, d222, d223
        )
        d230, d231, d232, d233 = _mma_m16n8k32_e4m3(
            a20, a21, a22, a23, b30, b31, d230, d231, d232, d233
        )
        d300, d301, d302, d303 = _mma_m16n8k32_e4m3(
            a30, a31, a32, a33, b00, b01, d300, d301, d302, d303
        )
        d310, d311, d312, d313 = _mma_m16n8k32_e4m3(
            a30, a31, a32, a33, b10, b11, d310, d311, d312, d313
        )
        d320, d321, d322, d323 = _mma_m16n8k32_e4m3(
            a30, a31, a32, a33, b20, b21, d320, d321, d322, d323
        )
        d330, d331, d332, d333 = _mma_m16n8k32_e4m3(
            a30, a31, a32, a33, b30, b31, d330, d331, d332, d333
        )

        cute.arch.barrier()

    scale = sa[0] * sb[0]
    cta_m = bid_m * BM_L + warp_m * 64
    cta_n = bid_n * BN_L + warp_n * 32
    cn0 = cta_n + lmod4 * 2
    cn1 = cta_n + 8 + lmod4 * 2
    cn2 = cta_n + 16 + lmod4 * 2
    cn3 = cta_n + 24 + lmod4 * 2

    rA = cta_m + ldiv4
    rB = cta_m + ldiv4 + 8
    if cutlass.dynamic_expr(rA < m_size):
        c[rA, cn0 + 0] = (d000 * scale).to(cutlass.BFloat16)
        c[rA, cn0 + 1] = (d001 * scale).to(cutlass.BFloat16)
        c[rA, cn1 + 0] = (d010 * scale).to(cutlass.BFloat16)
        c[rA, cn1 + 1] = (d011 * scale).to(cutlass.BFloat16)
        c[rA, cn2 + 0] = (d020 * scale).to(cutlass.BFloat16)
        c[rA, cn2 + 1] = (d021 * scale).to(cutlass.BFloat16)
        c[rA, cn3 + 0] = (d030 * scale).to(cutlass.BFloat16)
        c[rA, cn3 + 1] = (d031 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(rB < m_size):
        c[rB, cn0 + 0] = (d002 * scale).to(cutlass.BFloat16)
        c[rB, cn0 + 1] = (d003 * scale).to(cutlass.BFloat16)
        c[rB, cn1 + 0] = (d012 * scale).to(cutlass.BFloat16)
        c[rB, cn1 + 1] = (d013 * scale).to(cutlass.BFloat16)
        c[rB, cn2 + 0] = (d022 * scale).to(cutlass.BFloat16)
        c[rB, cn2 + 1] = (d023 * scale).to(cutlass.BFloat16)
        c[rB, cn3 + 0] = (d032 * scale).to(cutlass.BFloat16)
        c[rB, cn3 + 1] = (d033 * scale).to(cutlass.BFloat16)
    rA = cta_m + 16 + ldiv4
    rB = cta_m + 16 + ldiv4 + 8
    if cutlass.dynamic_expr(rA < m_size):
        c[rA, cn0 + 0] = (d100 * scale).to(cutlass.BFloat16)
        c[rA, cn0 + 1] = (d101 * scale).to(cutlass.BFloat16)
        c[rA, cn1 + 0] = (d110 * scale).to(cutlass.BFloat16)
        c[rA, cn1 + 1] = (d111 * scale).to(cutlass.BFloat16)
        c[rA, cn2 + 0] = (d120 * scale).to(cutlass.BFloat16)
        c[rA, cn2 + 1] = (d121 * scale).to(cutlass.BFloat16)
        c[rA, cn3 + 0] = (d130 * scale).to(cutlass.BFloat16)
        c[rA, cn3 + 1] = (d131 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(rB < m_size):
        c[rB, cn0 + 0] = (d102 * scale).to(cutlass.BFloat16)
        c[rB, cn0 + 1] = (d103 * scale).to(cutlass.BFloat16)
        c[rB, cn1 + 0] = (d112 * scale).to(cutlass.BFloat16)
        c[rB, cn1 + 1] = (d113 * scale).to(cutlass.BFloat16)
        c[rB, cn2 + 0] = (d122 * scale).to(cutlass.BFloat16)
        c[rB, cn2 + 1] = (d123 * scale).to(cutlass.BFloat16)
        c[rB, cn3 + 0] = (d132 * scale).to(cutlass.BFloat16)
        c[rB, cn3 + 1] = (d133 * scale).to(cutlass.BFloat16)
    rA = cta_m + 32 + ldiv4
    rB = cta_m + 32 + ldiv4 + 8
    if cutlass.dynamic_expr(rA < m_size):
        c[rA, cn0 + 0] = (d200 * scale).to(cutlass.BFloat16)
        c[rA, cn0 + 1] = (d201 * scale).to(cutlass.BFloat16)
        c[rA, cn1 + 0] = (d210 * scale).to(cutlass.BFloat16)
        c[rA, cn1 + 1] = (d211 * scale).to(cutlass.BFloat16)
        c[rA, cn2 + 0] = (d220 * scale).to(cutlass.BFloat16)
        c[rA, cn2 + 1] = (d221 * scale).to(cutlass.BFloat16)
        c[rA, cn3 + 0] = (d230 * scale).to(cutlass.BFloat16)
        c[rA, cn3 + 1] = (d231 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(rB < m_size):
        c[rB, cn0 + 0] = (d202 * scale).to(cutlass.BFloat16)
        c[rB, cn0 + 1] = (d203 * scale).to(cutlass.BFloat16)
        c[rB, cn1 + 0] = (d212 * scale).to(cutlass.BFloat16)
        c[rB, cn1 + 1] = (d213 * scale).to(cutlass.BFloat16)
        c[rB, cn2 + 0] = (d222 * scale).to(cutlass.BFloat16)
        c[rB, cn2 + 1] = (d223 * scale).to(cutlass.BFloat16)
        c[rB, cn3 + 0] = (d232 * scale).to(cutlass.BFloat16)
        c[rB, cn3 + 1] = (d233 * scale).to(cutlass.BFloat16)
    rA = cta_m + 48 + ldiv4
    rB = cta_m + 48 + ldiv4 + 8
    if cutlass.dynamic_expr(rA < m_size):
        c[rA, cn0 + 0] = (d300 * scale).to(cutlass.BFloat16)
        c[rA, cn0 + 1] = (d301 * scale).to(cutlass.BFloat16)
        c[rA, cn1 + 0] = (d310 * scale).to(cutlass.BFloat16)
        c[rA, cn1 + 1] = (d311 * scale).to(cutlass.BFloat16)
        c[rA, cn2 + 0] = (d320 * scale).to(cutlass.BFloat16)
        c[rA, cn2 + 1] = (d321 * scale).to(cutlass.BFloat16)
        c[rA, cn3 + 0] = (d330 * scale).to(cutlass.BFloat16)
        c[rA, cn3 + 1] = (d331 * scale).to(cutlass.BFloat16)
    if cutlass.dynamic_expr(rB < m_size):
        c[rB, cn0 + 0] = (d302 * scale).to(cutlass.BFloat16)
        c[rB, cn0 + 1] = (d303 * scale).to(cutlass.BFloat16)
        c[rB, cn1 + 0] = (d312 * scale).to(cutlass.BFloat16)
        c[rB, cn1 + 1] = (d313 * scale).to(cutlass.BFloat16)
        c[rB, cn2 + 0] = (d322 * scale).to(cutlass.BFloat16)
        c[rB, cn2 + 1] = (d323 * scale).to(cutlass.BFloat16)
        c[rB, cn3 + 0] = (d332 * scale).to(cutlass.BFloat16)
        c[rB, cn3 + 1] = (d333 * scale).to(cutlass.BFloat16)


# ---- SIMT vec8 kernel ----
@cute.kernel
def _simt_vec8_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    c: cute.Tensor,
    total_groups: cutlass.Int32,
    k_blocks: cutlass.Int32,
    n_groups: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    group_idx, _, _ = cute.arch.block_idx()
    if cutlass.dynamic_expr(group_idx < total_groups):
        row = group_idx // n_groups
        col0 = (group_idx - row * n_groups) * SIMT_COLS
        acc0 = cutlass.Float32(0.0)
        acc1 = cutlass.Float32(0.0)
        acc2 = cutlass.Float32(0.0)
        acc3 = cutlass.Float32(0.0)
        acc4 = cutlass.Float32(0.0)
        acc5 = cutlass.Float32(0.0)
        acc6 = cutlass.Float32(0.0)
        acc7 = cutlass.Float32(0.0)
        for kb in cutlass.range(lane, k_blocks, WARP_THREADS, unroll=2):
            av = a[row, kb, None].load().to(cutlass.Float32)
            bv0 = b[col0 + 0, kb, None].load().to(cutlass.Float32)
            bv1 = b[col0 + 1, kb, None].load().to(cutlass.Float32)
            bv2 = b[col0 + 2, kb, None].load().to(cutlass.Float32)
            bv3 = b[col0 + 3, kb, None].load().to(cutlass.Float32)
            bv4 = b[col0 + 4, kb, None].load().to(cutlass.Float32)
            bv5 = b[col0 + 5, kb, None].load().to(cutlass.Float32)
            bv6 = b[col0 + 6, kb, None].load().to(cutlass.Float32)
            bv7 = b[col0 + 7, kb, None].load().to(cutlass.Float32)
            p0 = av * bv0
            p1 = av * bv1
            p2 = av * bv2
            p3 = av * bv3
            p4 = av * bv4
            p5 = av * bv5
            p6 = av * bv6
            p7 = av * bv7
            acc0 += p0[0] + p0[1] + p0[2] + p0[3] + p0[4] + p0[5] + p0[6] + p0[7]
            acc1 += p1[0] + p1[1] + p1[2] + p1[3] + p1[4] + p1[5] + p1[6] + p1[7]
            acc2 += p2[0] + p2[1] + p2[2] + p2[3] + p2[4] + p2[5] + p2[6] + p2[7]
            acc3 += p3[0] + p3[1] + p3[2] + p3[3] + p3[4] + p3[5] + p3[6] + p3[7]
            acc4 += p4[0] + p4[1] + p4[2] + p4[3] + p4[4] + p4[5] + p4[6] + p4[7]
            acc5 += p5[0] + p5[1] + p5[2] + p5[3] + p5[4] + p5[5] + p5[6] + p5[7]
            acc6 += p6[0] + p6[1] + p6[2] + p6[3] + p6[4] + p6[5] + p6[6] + p6[7]
            acc7 += p7[0] + p7[1] + p7[2] + p7[3] + p7[4] + p7[5] + p7[6] + p7[7]
        acc0 = cute.arch.warp_reduction_sum(acc0, threads_in_group=WARP_THREADS)
        acc1 = cute.arch.warp_reduction_sum(acc1, threads_in_group=WARP_THREADS)
        acc2 = cute.arch.warp_reduction_sum(acc2, threads_in_group=WARP_THREADS)
        acc3 = cute.arch.warp_reduction_sum(acc3, threads_in_group=WARP_THREADS)
        acc4 = cute.arch.warp_reduction_sum(acc4, threads_in_group=WARP_THREADS)
        acc5 = cute.arch.warp_reduction_sum(acc5, threads_in_group=WARP_THREADS)
        acc6 = cute.arch.warp_reduction_sum(acc6, threads_in_group=WARP_THREADS)
        acc7 = cute.arch.warp_reduction_sum(acc7, threads_in_group=WARP_THREADS)
        if lane == 0:
            scale = sa[0] * sb[0]
            c[row, col0 + 0] = (acc0 * scale).to(cutlass.BFloat16)
            c[row, col0 + 1] = (acc1 * scale).to(cutlass.BFloat16)
            c[row, col0 + 2] = (acc2 * scale).to(cutlass.BFloat16)
            c[row, col0 + 3] = (acc3 * scale).to(cutlass.BFloat16)
            c[row, col0 + 4] = (acc4 * scale).to(cutlass.BFloat16)
            c[row, col0 + 5] = (acc5 * scale).to(cutlass.BFloat16)
            c[row, col0 + 6] = (acc6 * scale).to(cutlass.BFloat16)
            c[row, col0 + 7] = (acc7 * scale).to(cutlass.BFloat16)


# ---- SIMT vec1 with VEC=16 (b128 loads) — best for M<5, small N ----
@cute.kernel
def _simt_vec1_v16_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    c: cute.Tensor,
    total: cutlass.Int32,
    k_blocks: cutlass.Int32,
    n_size: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    out_idx, _, _ = cute.arch.block_idx()
    if cutlass.dynamic_expr(out_idx < total):
        row = out_idx // n_size
        col = out_idx - row * n_size
        acc = cutlass.Float32(0.0)
        for kb in cutlass.range(lane, k_blocks, WARP_THREADS, unroll=1):
            av = a[row, kb, None].load().to(cutlass.Float32)
            bv = b[col, kb, None].load().to(cutlass.Float32)
            prod = av * bv
            acc += (
                prod[0]
                + prod[1]
                + prod[2]
                + prod[3]
                + prod[4]
                + prod[5]
                + prod[6]
                + prod[7]
                + prod[8]
                + prod[9]
                + prod[10]
                + prod[11]
                + prod[12]
                + prod[13]
                + prod[14]
                + prod[15]
            )
        acc = cute.arch.warp_reduction_sum(acc, threads_in_group=WARP_THREADS)
        if lane == 0:
            c[row, col] = (acc * sa[0] * sb[0]).to(cutlass.BFloat16)


# ---- SIMT 2-row vec8 — B-reuse across 2 A rows for M=5-15 ----
@cute.kernel
def _simt_2row_vec8_kernel(
    a: cute.Tensor,
    b: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    c: cute.Tensor,
    total_groups: cutlass.Int32,
    k_blocks: cutlass.Int32,
    n_groups: cutlass.Int32,
    m_size: cutlass.Int32,
    m_pairs: cutlass.Int32,
):
    lane, _, _ = cute.arch.thread_idx()
    group_idx, _, _ = cute.arch.block_idx()
    if cutlass.dynamic_expr(group_idx < total_groups):
        row_pair = group_idx // n_groups
        col0 = (group_idx - row_pair * n_groups) * SIMT_COLS
        row0 = row_pair * 2
        row1 = row0 + 1
        acc00 = cutlass.Float32(0.0)
        acc01 = cutlass.Float32(0.0)
        acc02 = cutlass.Float32(0.0)
        acc03 = cutlass.Float32(0.0)
        acc04 = cutlass.Float32(0.0)
        acc05 = cutlass.Float32(0.0)
        acc06 = cutlass.Float32(0.0)
        acc07 = cutlass.Float32(0.0)
        acc10 = cutlass.Float32(0.0)
        acc11 = cutlass.Float32(0.0)
        acc12 = cutlass.Float32(0.0)
        acc13 = cutlass.Float32(0.0)
        acc14 = cutlass.Float32(0.0)
        acc15 = cutlass.Float32(0.0)
        acc16 = cutlass.Float32(0.0)
        acc17 = cutlass.Float32(0.0)
        for kb in cutlass.range(lane, k_blocks, WARP_THREADS, unroll=2):
            av0 = a[row0, kb, None].load().to(cutlass.Float32)
            av1 = a[row1, kb, None].load().to(cutlass.Float32)
            bv0 = b[col0 + 0, kb, None].load().to(cutlass.Float32)
            bv1 = b[col0 + 1, kb, None].load().to(cutlass.Float32)
            bv2 = b[col0 + 2, kb, None].load().to(cutlass.Float32)
            bv3 = b[col0 + 3, kb, None].load().to(cutlass.Float32)
            bv4 = b[col0 + 4, kb, None].load().to(cutlass.Float32)
            bv5 = b[col0 + 5, kb, None].load().to(cutlass.Float32)
            bv6 = b[col0 + 6, kb, None].load().to(cutlass.Float32)
            bv7 = b[col0 + 7, kb, None].load().to(cutlass.Float32)
            p00 = av0 * bv0
            p01 = av0 * bv1
            p02 = av0 * bv2
            p03 = av0 * bv3
            p04 = av0 * bv4
            p05 = av0 * bv5
            p06 = av0 * bv6
            p07 = av0 * bv7
            p10 = av1 * bv0
            p11 = av1 * bv1
            p12 = av1 * bv2
            p13 = av1 * bv3
            p14 = av1 * bv4
            p15 = av1 * bv5
            p16 = av1 * bv6
            p17 = av1 * bv7
            acc00 += (
                p00[0] + p00[1] + p00[2] + p00[3] + p00[4] + p00[5] + p00[6] + p00[7]
            )
            acc01 += (
                p01[0] + p01[1] + p01[2] + p01[3] + p01[4] + p01[5] + p01[6] + p01[7]
            )
            acc02 += (
                p02[0] + p02[1] + p02[2] + p02[3] + p02[4] + p02[5] + p02[6] + p02[7]
            )
            acc03 += (
                p03[0] + p03[1] + p03[2] + p03[3] + p03[4] + p03[5] + p03[6] + p03[7]
            )
            acc04 += (
                p04[0] + p04[1] + p04[2] + p04[3] + p04[4] + p04[5] + p04[6] + p04[7]
            )
            acc05 += (
                p05[0] + p05[1] + p05[2] + p05[3] + p05[4] + p05[5] + p05[6] + p05[7]
            )
            acc06 += (
                p06[0] + p06[1] + p06[2] + p06[3] + p06[4] + p06[5] + p06[6] + p06[7]
            )
            acc07 += (
                p07[0] + p07[1] + p07[2] + p07[3] + p07[4] + p07[5] + p07[6] + p07[7]
            )
            acc10 += (
                p10[0] + p10[1] + p10[2] + p10[3] + p10[4] + p10[5] + p10[6] + p10[7]
            )
            acc11 += (
                p11[0] + p11[1] + p11[2] + p11[3] + p11[4] + p11[5] + p11[6] + p11[7]
            )
            acc12 += (
                p12[0] + p12[1] + p12[2] + p12[3] + p12[4] + p12[5] + p12[6] + p12[7]
            )
            acc13 += (
                p13[0] + p13[1] + p13[2] + p13[3] + p13[4] + p13[5] + p13[6] + p13[7]
            )
            acc14 += (
                p14[0] + p14[1] + p14[2] + p14[3] + p14[4] + p14[5] + p14[6] + p14[7]
            )
            acc15 += (
                p15[0] + p15[1] + p15[2] + p15[3] + p15[4] + p15[5] + p15[6] + p15[7]
            )
            acc16 += (
                p16[0] + p16[1] + p16[2] + p16[3] + p16[4] + p16[5] + p16[6] + p16[7]
            )
            acc17 += (
                p17[0] + p17[1] + p17[2] + p17[3] + p17[4] + p17[5] + p17[6] + p17[7]
            )
        acc00 = cute.arch.warp_reduction_sum(acc00, threads_in_group=WARP_THREADS)
        acc01 = cute.arch.warp_reduction_sum(acc01, threads_in_group=WARP_THREADS)
        acc02 = cute.arch.warp_reduction_sum(acc02, threads_in_group=WARP_THREADS)
        acc03 = cute.arch.warp_reduction_sum(acc03, threads_in_group=WARP_THREADS)
        acc04 = cute.arch.warp_reduction_sum(acc04, threads_in_group=WARP_THREADS)
        acc05 = cute.arch.warp_reduction_sum(acc05, threads_in_group=WARP_THREADS)
        acc06 = cute.arch.warp_reduction_sum(acc06, threads_in_group=WARP_THREADS)
        acc07 = cute.arch.warp_reduction_sum(acc07, threads_in_group=WARP_THREADS)
        acc10 = cute.arch.warp_reduction_sum(acc10, threads_in_group=WARP_THREADS)
        acc11 = cute.arch.warp_reduction_sum(acc11, threads_in_group=WARP_THREADS)
        acc12 = cute.arch.warp_reduction_sum(acc12, threads_in_group=WARP_THREADS)
        acc13 = cute.arch.warp_reduction_sum(acc13, threads_in_group=WARP_THREADS)
        acc14 = cute.arch.warp_reduction_sum(acc14, threads_in_group=WARP_THREADS)
        acc15 = cute.arch.warp_reduction_sum(acc15, threads_in_group=WARP_THREADS)
        acc16 = cute.arch.warp_reduction_sum(acc16, threads_in_group=WARP_THREADS)
        acc17 = cute.arch.warp_reduction_sum(acc17, threads_in_group=WARP_THREADS)
        if lane == 0:
            scale = sa[0] * sb[0]
            if cutlass.dynamic_expr(row0 < m_size):
                c[row0, col0 + 0] = (acc00 * scale).to(cutlass.BFloat16)
                c[row0, col0 + 1] = (acc01 * scale).to(cutlass.BFloat16)
                c[row0, col0 + 2] = (acc02 * scale).to(cutlass.BFloat16)
                c[row0, col0 + 3] = (acc03 * scale).to(cutlass.BFloat16)
                c[row0, col0 + 4] = (acc04 * scale).to(cutlass.BFloat16)
                c[row0, col0 + 5] = (acc05 * scale).to(cutlass.BFloat16)
                c[row0, col0 + 6] = (acc06 * scale).to(cutlass.BFloat16)
                c[row0, col0 + 7] = (acc07 * scale).to(cutlass.BFloat16)
            if cutlass.dynamic_expr(row1 < m_size):
                c[row1, col0 + 0] = (acc10 * scale).to(cutlass.BFloat16)
                c[row1, col0 + 1] = (acc11 * scale).to(cutlass.BFloat16)
                c[row1, col0 + 2] = (acc12 * scale).to(cutlass.BFloat16)
                c[row1, col0 + 3] = (acc13 * scale).to(cutlass.BFloat16)
                c[row1, col0 + 4] = (acc14 * scale).to(cutlass.BFloat16)
                c[row1, col0 + 5] = (acc15 * scale).to(cutlass.BFloat16)
                c[row1, col0 + 6] = (acc16 * scale).to(cutlass.BFloat16)
                c[row1, col0 + 7] = (acc17 * scale).to(cutlass.BFloat16)


# ---- Launchers ----
@cute.jit
def _launch_mma_tiny(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    m: cute.sym_int32(),
    n: cute.sym_int32(divisibility=32),
    k: cute.sym_int32(divisibility=256),
    stream: cuda.CUstream,
):
    k_words = k // 4
    k_tiles = k // T_K_TILE
    a = cute.make_tensor(a_ptr, cute.make_layout((m, k_words), stride=(k_words, 1)))
    b = cute.make_tensor(b_ptr, cute.make_layout((n, k_words), stride=(k_words, 1)))
    sa = cute.make_tensor(sa_ptr, cute.make_layout((1,), stride=(1,)))
    sb = cute.make_tensor(sb_ptr, cute.make_layout((1,), stride=(1,)))
    c = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
    _bmm_fp8_mma_kernel_tiny(a, b, sa, sb, c, m, k_tiles).launch(
        grid=(cute.ceil_div(m, T_M_TILE), n // T_N_TILE, 1),
        block=(WARP_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_mma_tiny8(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    m: cute.sym_int32(),
    n: cute.sym_int32(divisibility=8),
    k: cute.sym_int32(divisibility=256),
    stream: cuda.CUstream,
):
    k_words = k // 4
    k_tiles = k // T_K_TILE
    a = cute.make_tensor(a_ptr, cute.make_layout((m, k_words), stride=(k_words, 1)))
    b = cute.make_tensor(b_ptr, cute.make_layout((n, k_words), stride=(k_words, 1)))
    sa = cute.make_tensor(sa_ptr, cute.make_layout((1,), stride=(1,)))
    sb = cute.make_tensor(sb_ptr, cute.make_layout((1,), stride=(1,)))
    c = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
    _bmm_fp8_mma_kernel_tiny8(a, b, sa, sb, c, m, k_tiles).launch(
        grid=(cute.ceil_div(m, T_M_TILE), n // T8_N_TILE, 1),
        block=(WARP_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_mma_med(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    m: cute.sym_int32(),
    n: cute.sym_int32(divisibility=32),
    k: cute.sym_int32(divisibility=256),
    stream: cuda.CUstream,
):
    k_words = k // 4
    k_tiles = k // M_K_TILE
    a = cute.make_tensor(a_ptr, cute.make_layout((m, k_words), stride=(k_words, 1)))
    b = cute.make_tensor(b_ptr, cute.make_layout((n, k_words), stride=(k_words, 1)))
    sa = cute.make_tensor(sa_ptr, cute.make_layout((1,), stride=(1,)))
    sb = cute.make_tensor(sb_ptr, cute.make_layout((1,), stride=(1,)))
    c = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
    _bmm_fp8_mma_kernel_med(a, b, sa, sb, c, m, k_tiles).launch(
        grid=(cute.ceil_div(m, M_M_TILE), n // M_N_TILE, 1),
        block=(WARP_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_mma_small(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    m: cute.sym_int32(),
    n: cute.sym_int32(divisibility=64),
    k: cute.sym_int32(divisibility=256),
    stream: cuda.CUstream,
):
    k_words = k // 4
    k_tiles = k // 32
    a = cute.make_tensor(a_ptr, cute.make_layout((m, k_words), stride=(k_words, 1)))
    b = cute.make_tensor(b_ptr, cute.make_layout((n, k_words), stride=(k_words, 1)))
    sa = cute.make_tensor(sa_ptr, cute.make_layout((1,), stride=(1,)))
    sb = cute.make_tensor(sb_ptr, cute.make_layout((1,), stride=(1,)))
    c = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
    _bmm_fp8_mma_kernel_small(a, b, sa, sb, c, m, k_tiles).launch(
        grid=(cute.ceil_div(m, BM_S), n // BN_S, 1),
        block=(128, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_mma_large(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    m: cute.sym_int32(),
    n: cute.sym_int32(divisibility=128),
    k: cute.sym_int32(divisibility=256),
    stream: cuda.CUstream,
):
    k_words = k // 4
    k_tiles = k // 32
    a = cute.make_tensor(a_ptr, cute.make_layout((m, k_words), stride=(k_words, 1)))
    b = cute.make_tensor(b_ptr, cute.make_layout((n, k_words), stride=(k_words, 1)))
    sa = cute.make_tensor(sa_ptr, cute.make_layout((1,), stride=(1,)))
    sb = cute.make_tensor(sb_ptr, cute.make_layout((1,), stride=(1,)))
    c = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
    _bmm_fp8_mma_kernel_large(a, b, sa, sb, c, m, k_tiles).launch(
        grid=(cute.ceil_div(m, BM_L), n // BN_L, 1),
        block=(256, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_simt8(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    m: cute.sym_int32(),
    n: cute.sym_int32(divisibility=64),
    k: cute.sym_int32(divisibility=512),
    stream: cuda.CUstream,
):
    k_blocks = k // SIMT_VEC
    n_groups = n // SIMT_COLS
    a = cute.make_tensor(
        a_ptr, cute.make_layout((m, k_blocks, SIMT_VEC), stride=(k, SIMT_VEC, 1))
    )
    b = cute.make_tensor(
        b_ptr, cute.make_layout((n, k_blocks, SIMT_VEC), stride=(k, SIMT_VEC, 1))
    )
    sa = cute.make_tensor(sa_ptr, cute.make_layout((1,), stride=(1,)))
    sb = cute.make_tensor(sb_ptr, cute.make_layout((1,), stride=(1,)))
    c = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
    _simt_vec8_kernel(a, b, sa, sb, c, m * n_groups, k_blocks, n_groups).launch(
        grid=(m * n_groups, 1, 1),
        block=(WARP_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_simt1_v16(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    m: cute.sym_int32(),
    n: cute.sym_int32(divisibility=1),
    k: cute.sym_int32(divisibility=512),
    stream: cuda.CUstream,
):
    k_blocks = k // SIMT_VEC16
    a = cute.make_tensor(
        a_ptr, cute.make_layout((m, k_blocks, SIMT_VEC16), stride=(k, SIMT_VEC16, 1))
    )
    b = cute.make_tensor(
        b_ptr, cute.make_layout((n, k_blocks, SIMT_VEC16), stride=(k, SIMT_VEC16, 1))
    )
    sa = cute.make_tensor(sa_ptr, cute.make_layout((1,), stride=(1,)))
    sb = cute.make_tensor(sb_ptr, cute.make_layout((1,), stride=(1,)))
    c = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
    _simt_vec1_v16_kernel(a, b, sa, sb, c, m * n, k_blocks, n).launch(
        grid=(m * n, 1, 1),
        block=(WARP_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_simt2(
    a_ptr,
    b_ptr,
    sa_ptr,
    sb_ptr,
    c_ptr,
    m: cute.sym_int32(),
    n: cute.sym_int32(divisibility=64),
    k: cute.sym_int32(divisibility=512),
    stream: cuda.CUstream,
):
    k_blocks = k // SIMT_VEC
    n_groups = n // SIMT_COLS
    m_pairs = (m + 1) // 2
    a = cute.make_tensor(
        a_ptr, cute.make_layout((m, k_blocks, SIMT_VEC), stride=(k, SIMT_VEC, 1))
    )
    b = cute.make_tensor(
        b_ptr, cute.make_layout((n, k_blocks, SIMT_VEC), stride=(k, SIMT_VEC, 1))
    )
    sa = cute.make_tensor(sa_ptr, cute.make_layout((1,), stride=(1,)))
    sb = cute.make_tensor(sb_ptr, cute.make_layout((1,), stride=(1,)))
    c = cute.make_tensor(c_ptr, cute.make_layout((m, n), stride=(n, 1)))
    _simt_2row_vec8_kernel(
        a, b, sa, sb, c, m_pairs * n_groups, k_blocks, n_groups, m, m_pairs
    ).launch(
        grid=(m_pairs * n_groups, 1, 1),
        block=(WARP_THREADS, 1, 1),
        stream=stream,
    )


# ---- Compile functions ----
def _compile_mma_tiny():
    fake_stream = cuda.CUstream(0)
    return CUTE_COMPILE(
        _launch_mma_tiny,
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        T_M_TILE,
        1024,
        4096,
        fake_stream,
    )


def _compile_mma_tiny8():
    fake_stream = cuda.CUstream(0)
    return CUTE_COMPILE(
        _launch_mma_tiny8,
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        T_M_TILE,
        1024,
        4096,
        fake_stream,
    )


def _compile_mma_med():
    fake_stream = cuda.CUstream(0)
    return CUTE_COMPILE(
        _launch_mma_med,
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        M_M_TILE,
        1024,
        4096,
        fake_stream,
    )


def _compile_mma_small():
    fake_stream = cuda.CUstream(0)
    return CUTE_COMPILE(
        _launch_mma_small,
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        BM_S,
        1024,
        4096,
        fake_stream,
    )


def _compile_mma_large():
    fake_stream = cuda.CUstream(0)
    return CUTE_COMPILE(
        _launch_mma_large,
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        BM_L,
        1024,
        4096,
        fake_stream,
    )


def _compile_simt8():
    fake_stream = cuda.CUstream(0)
    return CUTE_COMPILE(
        _launch_simt8,
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        24,
        1024,
        4096,
        fake_stream,
    )


def _compile_simt1_v16():
    fake_stream = cuda.CUstream(0)
    return CUTE_COMPILE(
        _launch_simt1_v16,
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        4,
        1024,
        4096,
        fake_stream,
    )


def _compile_simt2():
    fake_stream = cuda.CUstream(0)
    return CUTE_COMPILE(
        _launch_simt2,
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float8E4M3FN, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(cutlass.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        10,
        1024,
        4096,
        fake_stream,
    )


def _u32_ptr(t):
    return make_ptr(
        cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )


def _fp8_ptr(t):
    return make_ptr(
        cutlass.Float8E4M3FN, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )


def _f32_ptr(t):
    return make_ptr(
        cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )


def _bf16_ptr(t):
    return make_ptr(
        cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )


def run(A, B, A_scale, B_scale, out) -> None:
    global \
        _MMA_T_COMPILED, \
        _MMA_T8_COMPILED, \
        _MMA_M_COMPILED, \
        _MMA_S_COMPILED, \
        _MMA_L_COMPILED, \
        _SIMT8_COMPILED
    global _SIMT1_V16_COMPILED, _SIMT2_COMPILED
    _, m, k = A.shape
    _, _, n = B.shape
    stream = cuda.CUstream(torch.cuda.current_stream(device=A.device).cuda_stream)
    M, N, K = int(m), int(n), int(k)

    use_tiny8 = False
    if M >= 5 and M < 16:
        # Extended dispatch: tiny8 for ALL N>=5120 (was N>=9216)
        use_tiny8 = N >= 5120 or (M <= 12 and N == 5120)
    elif 16 <= M <= 32:
        use_tiny8 = N >= 4096
    elif M >= 40 and M <= 64:
        # Extended: tiny8 for all M=40-64 (was N<8192)
        use_tiny8 = True
    elif M > 64 and M <= 128:
        use_tiny8 = N == 1024 or (((M <= 96) or (M >= 120)) and K == 1024 and N == 4096)
    if use_tiny8 and (N & 7) == 0:
        if _MMA_T8_COMPILED is None:
            _MMA_T8_COMPILED = _compile_mma_tiny8()
        _MMA_T8_COMPILED(
            _u32_ptr(A),
            _u32_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # SIMT vec1+VEC=16 for M=1 (any N) or M<5 with M*N<=8192
    if (K & 15) == 0 and (M == 1 or (M < 5 and M * N <= 8192)):
        if _SIMT1_V16_COMPILED is None:
            _SIMT1_V16_COMPILED = _compile_simt1_v16()
        _SIMT1_V16_COMPILED(
            _fp8_ptr(A),
            _fp8_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # 2-row SIMT for M=5-15 with N>=1024 N<5120 (B reuse across 2 A rows)
    if M >= 5 and M < 16 and N >= 1024 and N < 5120 and (N & 7) == 0:
        if _SIMT2_COMPILED is None:
            _SIMT2_COMPILED = _compile_simt2()
        _SIMT2_COMPILED(
            _fp8_ptr(A),
            _fp8_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # 2-row SIMT also for M=16-56 with low N (TINY MMA underfills SMs at low N)
    if M >= 16 and M <= 56 and N <= 2048 and (N & 7) == 0:
        if _SIMT2_COMPILED is None:
            _SIMT2_COMPILED = _compile_simt2()
        _SIMT2_COMPILED(
            _fp8_ptr(A),
            _fp8_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # SIMT for M<5 (large N), or M=5-15 with N<5120 (fallback)
    if M < 5 or (M < 16 and N < 5120):
        if _SIMT8_COMPILED is None:
            _SIMT8_COMPILED = _compile_simt8()
        _SIMT8_COMPILED(
            _fp8_ptr(A),
            _fp8_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # Small-M MMA path.
    if M >= 256 and M < 1024 and N <= 2048 and (N & (BN_S - 1)) == 0:
        if _MMA_S_COMPILED is None:
            _MMA_S_COMPILED = _compile_mma_small()
        _MMA_S_COMPILED(
            _u32_ptr(A),
            _u32_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # Large MMA for M>=256 with N divisible by 128
    if M >= 256 and (N & (BN_L - 1)) == 0:
        if _MMA_L_COMPILED is None:
            _MMA_L_COMPILED = _compile_mma_large()
        _MMA_L_COMPILED(
            _u32_ptr(A),
            _u32_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # Med MMA for M=64-255 N>=16384 - 32x32 register tile
    if M >= 64 and M < 256 and N >= 16384 and (N & 31) == 0:
        if _MMA_M_COMPILED is None:
            _MMA_M_COMPILED = _compile_mma_med()
        _MMA_M_COMPILED(
            _u32_ptr(A),
            _u32_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # Tiny MMA for M=5..255 - register-only, no barriers
    if M >= 5 and M < 256:
        if _MMA_T_COMPILED is None:
            _MMA_T_COMPILED = _compile_mma_tiny()
        _MMA_T_COMPILED(
            _u32_ptr(A),
            _u32_ptr(B),
            _f32_ptr(A_scale),
            _f32_ptr(B_scale),
            _bf16_ptr(out),
            M,
            N,
            K,
            stream,
        )
        return

    # Small MMA (64x64, 4 warps, SMEM 2K-depth) for M>=256 with N not divisible by 128
    if _MMA_S_COMPILED is None:
        _MMA_S_COMPILED = _compile_mma_small()
    _MMA_S_COMPILED(
        _u32_ptr(A),
        _u32_ptr(B),
        _f32_ptr(A_scale),
        _f32_ptr(B_scale),
        _bf16_ptr(out),
        M,
        N,
        K,
        stream,
    )
