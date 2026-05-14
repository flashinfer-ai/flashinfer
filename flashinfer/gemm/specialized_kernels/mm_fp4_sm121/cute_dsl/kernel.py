"""CUTE DSL NVFP4 GEMM backend for SM121 specialized routing."""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

K_MMA = 8
K_STEPS = 4
K_TILE = K_MMA * K_STEPS
SMEM_K_STRIDE = K_TILE + 4
KT_PER_ROW = K_TILE // 4

NSTAGES_S = 3
NSTAGES_L = 3

TILE_M_L = 64
TILE_N_L = 128
WM_L = 2
WN_L = 8

TILE_M_S = 16
TILE_N_S = 64
WM_S = 1
WN_S = 2

TILE_M_N64 = 64
TILE_N_N64 = 64
WM_N64 = 2
WN_N64 = 4

TILE_M_SN64 = 16
TILE_N_SN64 = 64
WM_SN64 = 1
WN_SN64 = 2

# Micro N64 path: 1-warp CTAs with TILE_N=16.
TILE_M_XS = 16
TILE_N_XS = 16
WM_XS = 1
WN_XS = 2

# XL path: TILE_M=128, TILE_N=128, eight warps.
TILE_M_XL = 128
TILE_N_XL = 128
WM_XL = 2
WN_XL = 8
WARPS_XL = 8
TPB_XL = WARPS_XL * 32

WARPS = 4
TPB = WARPS * 32

WARPS_XS = 1
TPB_XS = WARPS_XS * 32

A_SMEM_L = TILE_M_L * SMEM_K_STRIDE
B_SMEM_L = TILE_N_L * SMEM_K_STRIDE
A_TOTAL_L = TILE_M_L * K_TILE
B_TOTAL_L = TILE_N_L * K_TILE
A_VEC4_L = A_TOTAL_L // 4
B_VEC4_L = B_TOTAL_L // 4
A_LOADS_L = (A_VEC4_L + TPB - 1) // TPB
B_LOADS_L = (B_VEC4_L + TPB - 1) // TPB

A_SMEM_S = TILE_M_S * SMEM_K_STRIDE
B_SMEM_S = TILE_N_S * SMEM_K_STRIDE
A_TOTAL_S = TILE_M_S * K_TILE
B_TOTAL_S = TILE_N_S * K_TILE
A_VEC4_S = A_TOTAL_S // 4
B_VEC4_S = B_TOTAL_S // 4
A_LOADS_S = (A_VEC4_S + TPB - 1) // TPB
B_LOADS_S = (B_VEC4_S + TPB - 1) // TPB

A_SMEM_N64 = TILE_M_N64 * SMEM_K_STRIDE
B_SMEM_N64 = TILE_N_N64 * SMEM_K_STRIDE
A_TOTAL_N64 = TILE_M_N64 * K_TILE
B_TOTAL_N64 = TILE_N_N64 * K_TILE
A_VEC4_N64 = A_TOTAL_N64 // 4
B_VEC4_N64 = B_TOTAL_N64 // 4
A_LOADS_N64 = (A_VEC4_N64 + TPB - 1) // TPB
B_LOADS_N64 = (B_VEC4_N64 + TPB - 1) // TPB

A_SMEM_SN64 = TILE_M_SN64 * SMEM_K_STRIDE
B_SMEM_SN64 = TILE_N_SN64 * SMEM_K_STRIDE
A_TOTAL_SN64 = TILE_M_SN64 * K_TILE
B_TOTAL_SN64 = TILE_N_SN64 * K_TILE
A_VEC4_SN64 = A_TOTAL_SN64 // 4
B_VEC4_SN64 = B_TOTAL_SN64 // 4
A_LOADS_SN64 = (A_VEC4_SN64 + TPB - 1) // TPB
B_LOADS_SN64 = (B_VEC4_SN64 + TPB - 1) // TPB

# 3-stage SMEM totals for gemm_large
A_SMEM_L_TOTAL = NSTAGES_L * A_SMEM_L
B_SMEM_L_TOTAL = NSTAGES_L * B_SMEM_L
# 2-stage SMEM totals (legacy)
A_SMEM_L_DB = 2 * A_SMEM_L
B_SMEM_L_DB = 2 * B_SMEM_L
A_SMEM_S_TOTAL = NSTAGES_S * A_SMEM_S
B_SMEM_S_TOTAL = NSTAGES_S * B_SMEM_S
A_SMEM_N64_DB = 2 * A_SMEM_N64
B_SMEM_N64_DB = 2 * B_SMEM_N64

# 3-stage SMEM totals for small_n64
A_SMEM_SN64_3S = NSTAGES_S * A_SMEM_SN64
B_SMEM_SN64_3S = NSTAGES_S * B_SMEM_SN64

# Micro N64 kernel SMEM
A_SMEM_XS = TILE_M_XS * SMEM_K_STRIDE
B_SMEM_XS = TILE_N_XS * SMEM_K_STRIDE
A_TOTAL_XS = TILE_M_XS * K_TILE
B_TOTAL_XS = TILE_N_XS * K_TILE
A_VEC4_XS = A_TOTAL_XS // 4
B_VEC4_XS = B_TOTAL_XS // 4
A_LOADS_XS = (A_VEC4_XS + TPB_XS - 1) // TPB_XS
B_LOADS_XS = (B_VEC4_XS + TPB_XS - 1) // TPB_XS
A_SMEM_XS_3S = NSTAGES_S * A_SMEM_XS
B_SMEM_XS_3S = NSTAGES_S * B_SMEM_XS

# XL kernel SMEM (TILE_M=128 TILE_N=128, 8 warps, 2-stage)
A_SMEM_XL = TILE_M_XL * SMEM_K_STRIDE
B_SMEM_XL = TILE_N_XL * SMEM_K_STRIDE
A_TOTAL_XL = TILE_M_XL * K_TILE
B_TOTAL_XL = TILE_N_XL * K_TILE
A_VEC4_XL = A_TOTAL_XL // 4
B_VEC4_XL = B_TOTAL_XL // 4
A_LOADS_XL = (A_VEC4_XL + TPB_XL - 1) // TPB_XL
B_LOADS_XL = (B_VEC4_XL + TPB_XL - 1) // TPB_XL
A_SMEM_XL_DB = 2 * A_SMEM_XL
B_SMEM_XL_DB = 2 * B_SMEM_XL


@cute.jit
def _mma(
    a0: cutlass.Int32,
    a1: cutlass.Int32,
    a2: cutlass.Int32,
    a3: cutlass.Int32,
    b0: cutlass.Int32,
    b1: cutlass.Int32,
    c0: cutlass.Float32,
    c1: cutlass.Float32,
    c2: cutlass.Float32,
    c3: cutlass.Float32,
    sf_a: cutlass.Int32,
    sf_b: cutlass.Int32,
):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            a0.ir_value(),
            a1.ir_value(),
            a2.ir_value(),
            a3.ir_value(),
            b0.ir_value(),
            b1.ir_value(),
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
            sf_a.ir_value(),
            sf_b.ir_value(),
        ],
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13},$14,{0,0},$15,{0,0};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [0])),
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [1])),
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [2])),
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [3])),
    )


@cute.jit
def _sync():
    llvm.inline_asm(
        T.i32(),
        [],
        "bar.sync 0;\nmov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _clamp(val: cutlass.Int32, limit: cutlass.Int32) -> cutlass.Int32:
    result = val
    if val >= limit:
        result = cutlass.Int32(0)
    return result


@cute.jit
def _cvta_to_shared_u32(generic_ptr_int64: cutlass.Int64) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [generic_ptr_int64.ir_value()],
            "{ .reg .b64 t; cvta.to.shared.u64 t, $1; cvt.u32.u64 $0, t; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _cp_async_16(smem_addr_u32: cutlass.Int32, gmem_addr: cutlass.Int64):
    llvm.inline_asm(
        T.i32(),
        [smem_addr_u32.ir_value(), gmem_addr.ir_value()],
        "cp.async.ca.shared.global.L2::128B [$1], [$2], 16; mov.u32 $0, 0;",
        "=r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _cp_async_commit():
    llvm.inline_asm(
        T.i32(),
        [],
        "cp.async.commit_group; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _cp_async_wait_1():
    llvm.inline_asm(
        T.i32(),
        [],
        "cp.async.wait_group 1; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _cp_async_wait_0():
    llvm.inline_asm(
        T.i32(),
        [],
        "cp.async.wait_group 0; mov.u32 $0, 0;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _pack_bf16x2(v0: cutlass.Float32, v1: cutlass.Float32) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [v0.ir_value(), v1.ir_value()],
            "{\n.reg .b16 lo, hi;\ncvt.rn.bf16.f32 lo, $1;\ncvt.rn.bf16.f32 hi, $2;\nmov.b32 $0, {lo, hi};\n}",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# gemm_large path for the larger-M dispatch cases.
@cute.kernel
def _gemm_large(
    a_p_int64: cutlass.Int64,
    b_p_int64: cutlass.Int64,
    sfa_i32: cute.Tensor,
    sfb_i32: cute.Tensor,
    out_i32: cute.Tensor,
    alpha_vec: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    K_u32: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    N_half: cutlass.Int32,
    gm: cutlass.Int32,
    gn: cutlass.Int32,
    group_m: cutlass.Int32,
):
    bx_raw, by_raw, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    warp_id = tid >> 5
    lane_id = tid & cutlass.Int32(31)
    g = lane_id >> 2
    t = lane_id & cutlass.Int32(3)
    warp_m = warp_id >> 1
    warp_n = warp_id & cutlass.Int32(1)
    # Triton-style M-major tile swizzle for L2 cache reuse (group_m param)
    linear = bx_raw + by_raw * gm
    in_group = group_m * gn
    group_idx = linear // in_group
    first_m = group_idx * group_m
    rem = linear - group_idx * in_group
    # min(group_m, gm - first_m) using arithmetic (avoid dynamic if-stmt)
    diff = (gm - first_m) - group_m
    g_size = (gm - first_m) - (diff & ~(diff >> 31))
    inner_m = rem - (rem // g_size) * g_size
    inner_n = rem // g_size
    bx = first_m + inner_m
    by = inner_n
    m_block = bx * cutlass.Int32(TILE_M_L)
    n_block = by * cutlass.Int32(TILE_N_L)
    m_warp = m_block + warp_m * cutlass.Int32(WM_L * 16)
    n_warp = n_block + warp_n * cutlass.Int32(WN_L * 8)

    sA_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, A_SMEM_L_TOTAL, alignment=128)
    sB_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, B_SMEM_L_TOTAL, alignment=128)
    sA = cute.make_tensor(
        sA_ptr, cute.make_layout((A_SMEM_L_TOTAL,), stride=(cutlass.Int32(1),))
    )
    sB = cute.make_tensor(
        sB_ptr, cute.make_layout((B_SMEM_L_TOTAL,), stride=(cutlass.Int32(1),))
    )
    sA_base_u32 = _cvta_to_shared_u32(sA_ptr.toint())
    sB_base_u32 = _cvta_to_shared_u32(sB_ptr.toint())

    acc = cute.make_rmem_tensor(cute.make_layout((WM_L, WN_L, 4)), cutlass.Float32)
    for mi in cutlass.range_constexpr(WM_L):
        for nj in cutlass.range_constexpr(WN_L):
            for ci in cutlass.range_constexpr(4):
                acc[mi, nj, ci] = cutlass.Float32(0.0)

    K_u32_i64 = K_u32.to(cutlass.Int64)

    # Prologue: stage 0
    for li in cutlass.range_constexpr(A_LOADS_L):
        v4 = tid + cutlass.Int32(li * TPB)
        if v4 < cutlass.Int32(A_VEC4_L):
            ar = v4 // cutlass.Int32(KT_PER_ROW)
            ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
            ac = ac4 * cutlass.Int32(4)
            gm_load = _clamp(m_block + ar, M)
            smem_off = ar * cutlass.Int32(SMEM_K_STRIDE) + ac
            smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = a_p_int64 + (
                gm_load.to(cutlass.Int64) * K_u32_i64 + ac.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    for li in cutlass.range_constexpr(B_LOADS_L):
        v4 = tid + cutlass.Int32(li * TPB)
        if v4 < cutlass.Int32(B_VEC4_L):
            br = v4 // cutlass.Int32(KT_PER_ROW)
            bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
            bc = bc4 * cutlass.Int32(4)
            gn_load = _clamp(n_block + br, N)
            smem_off = br * cutlass.Int32(SMEM_K_STRIDE) + bc
            smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = b_p_int64 + (
                gn_load.to(cutlass.Int64) * K_u32_i64 + bc.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    _cp_async_commit()

    # Prologue: stage 1
    if cutlass.Int32(1) < n_k_tiles:
        k_off_1 = cutlass.Int32(K_TILE)
        for li in cutlass.range_constexpr(A_LOADS_L):
            v4 = tid + cutlass.Int32(li * TPB)
            if v4 < cutlass.Int32(A_VEC4_L):
                ar = v4 // cutlass.Int32(KT_PER_ROW)
                ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                ac = ac4 * cutlass.Int32(4)
                gm_load = _clamp(m_block + ar, M)
                smem_off = (
                    cutlass.Int32(A_SMEM_L) + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                )
                smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                gmem_addr = a_p_int64 + (
                    gm_load.to(cutlass.Int64) * K_u32_i64
                    + (k_off_1 + ac).to(cutlass.Int64)
                ) * cutlass.Int64(4)
                _cp_async_16(smem_addr, gmem_addr)
        for li in cutlass.range_constexpr(B_LOADS_L):
            v4 = tid + cutlass.Int32(li * TPB)
            if v4 < cutlass.Int32(B_VEC4_L):
                br = v4 // cutlass.Int32(KT_PER_ROW)
                bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                bc = bc4 * cutlass.Int32(4)
                gn_load = _clamp(n_block + br, N)
                smem_off = (
                    cutlass.Int32(B_SMEM_L) + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                )
                smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                gmem_addr = b_p_int64 + (
                    gn_load.to(cutlass.Int64) * K_u32_i64
                    + (k_off_1 + bc).to(cutlass.Int64)
                ) * cutlass.Int64(4)
                _cp_async_16(smem_addr, gmem_addr)
        _cp_async_commit()

    for kt in cutlass.range(0, n_k_tiles, 1, unroll=1):
        _cp_async_wait_1()
        _sync()

        stage_kt = kt - (kt // cutlass.Int32(NSTAGES_L)) * cutlass.Int32(NSTAGES_L)
        cur_a_off = stage_kt * cutlass.Int32(A_SMEM_L)
        cur_b_off = stage_kt * cutlass.Int32(B_SMEM_L)

        kt_next = kt + cutlass.Int32(2)
        if kt_next < n_k_tiles:
            stage_next = kt_next - (
                kt_next // cutlass.Int32(NSTAGES_L)
            ) * cutlass.Int32(NSTAGES_L)
            nxt_a_smem_off = stage_next * cutlass.Int32(A_SMEM_L)
            nxt_b_smem_off = stage_next * cutlass.Int32(B_SMEM_L)
            k_off_next = kt_next * cutlass.Int32(K_TILE)
            for li in cutlass.range_constexpr(A_LOADS_L):
                v4 = tid + cutlass.Int32(li * TPB)
                if v4 < cutlass.Int32(A_VEC4_L):
                    ar = v4 // cutlass.Int32(KT_PER_ROW)
                    ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                    ac = ac4 * cutlass.Int32(4)
                    gm_load = _clamp(m_block + ar, M)
                    smem_off = nxt_a_smem_off + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                    smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = a_p_int64 + (
                        gm_load.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + ac).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            for li in cutlass.range_constexpr(B_LOADS_L):
                v4 = tid + cutlass.Int32(li * TPB)
                if v4 < cutlass.Int32(B_VEC4_L):
                    br = v4 // cutlass.Int32(KT_PER_ROW)
                    bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                    bc = bc4 * cutlass.Int32(4)
                    gn_load = _clamp(n_block + br, N)
                    smem_off = nxt_b_smem_off + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                    smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = b_p_int64 + (
                        gn_load.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + bc).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            _cp_async_commit()

        for ks in cutlass.range_constexpr(K_STEPS):
            ki = cutlass.Int32(ks * K_MMA)
            ct = ki + t
            ct4 = ct + cutlass.Int32(4)
            a_f = cute.make_rmem_tensor(cute.make_layout((WM_L, 4)), cutlass.Int32)
            sf_a = cute.make_rmem_tensor(cute.make_layout((WM_L,)), cutlass.Int32)
            sf_b_arr_l = cute.make_rmem_tensor(cute.make_layout((WN_L,)), cutlass.Int32)
            rb_arr_l = cute.make_rmem_tensor(cute.make_layout((WN_L, 2)), cutlass.Int32)
            kg = kt * cutlass.Int32(K_STEPS) + cutlass.Int32(ks)
            for mi in cutlass.range_constexpr(WM_L):
                mt = warp_m * cutlass.Int32(WM_L) + cutlass.Int32(mi)
                r0 = mt * cutlass.Int32(16) + g
                r1 = r0 + cutlass.Int32(8)
                a_f[mi, 0] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 1] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 2] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                a_f[mi, 3] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                mb = m_warp + cutlass.Int32(mi * 16)
                sr = mb + g + (t & cutlass.Int32(1)) * cutlass.Int32(8)
                sf_a[mi] = sfa_i32[
                    (sr >> 7) * sfa_stride
                    + kg * cutlass.Int32(128)
                    + (sr & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sr >> 5 & cutlass.Int32(3))
                ]
            for nj in cutlass.range_constexpr(WN_L):
                nc = n_warp + cutlass.Int32(nj * 8)
                sc = nc + g
                sf_b_arr_l[nj] = sfb_i32[
                    (sc >> 7) * sfb_stride
                    + kg * cutlass.Int32(128)
                    + (sc & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sc >> 5 & cutlass.Int32(3))
                ]
            for nj in cutlass.range_constexpr(WN_L):
                nt = warp_n * cutlass.Int32(WN_L) + cutlass.Int32(nj)
                br_idx = nt * cutlass.Int32(8) + g
                rb_arr_l[nj, 0] = sB[
                    cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct
                ]
                rb_arr_l[nj, 1] = sB[
                    cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct4
                ]
            for nj in cutlass.range_constexpr(WN_L):
                for mi in cutlass.range_constexpr(WM_L):
                    d0, d1, d2, d3 = _mma(
                        a_f[mi, 0],
                        a_f[mi, 1],
                        a_f[mi, 2],
                        a_f[mi, 3],
                        rb_arr_l[nj, 0],
                        rb_arr_l[nj, 1],
                        acc[mi, nj, 0],
                        acc[mi, nj, 1],
                        acc[mi, nj, 2],
                        acc[mi, nj, 3],
                        sf_a[mi],
                        sf_b_arr_l[nj],
                    )
                    acc[mi, nj, 0] = d0
                    acc[mi, nj, 1] = d1
                    acc[mi, nj, 2] = d2
                    acc[mi, nj, 3] = d3

    alpha = alpha_vec[0]
    for mi in cutlass.range_constexpr(WM_L):
        mb = m_warp + cutlass.Int32(mi * 16)
        for nj in cutlass.range_constexpr(WN_L):
            nc = n_warp + cutlass.Int32(nj * 8)
            or0 = mb + g
            or1 = or0 + cutlass.Int32(8)
            oc0 = nc + t * cutlass.Int32(2)
            oc_half = (nc >> 1) + t
            if or0 < M:
                if oc0 < N:
                    out_i32[or0 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 0] * alpha, acc[mi, nj, 1] * alpha
                    )
            if or1 < M:
                if oc0 < N:
                    out_i32[or1 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 2] * alpha, acc[mi, nj, 3] * alpha
                    )


# gemm_xlarge path with TILE_M=128, TILE_N=128, and eight warps.
@cute.kernel
def _gemm_xlarge(
    a_p_int64: cutlass.Int64,
    b_p_int64: cutlass.Int64,
    sfa_i32: cute.Tensor,
    sfb_i32: cute.Tensor,
    out_i32: cute.Tensor,
    alpha_vec: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    K_u32: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    N_half: cutlass.Int32,
    gm: cutlass.Int32,
    gn: cutlass.Int32,
    group_m: cutlass.Int32,
):
    bx_raw, by_raw, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    warp_id = tid >> 5
    lane_id = tid & cutlass.Int32(31)
    g = lane_id >> 2
    t = lane_id & cutlass.Int32(3)
    warp_m = warp_id >> 1
    warp_n = warp_id & cutlass.Int32(1)
    # Triton-style M-major tile swizzle for L2 cache reuse (group_m param)
    linear = bx_raw + by_raw * gm
    in_group = group_m * gn
    group_idx = linear // in_group
    first_m = group_idx * group_m
    rem = linear - group_idx * in_group
    # min(group_m, gm - first_m) using arithmetic (avoid dynamic if-stmt)
    diff = (gm - first_m) - group_m
    g_size = (gm - first_m) - (diff & ~(diff >> 31))
    inner_m = rem - (rem // g_size) * g_size
    inner_n = rem // g_size
    bx = first_m + inner_m
    by = inner_n
    m_block = bx * cutlass.Int32(TILE_M_XL)
    n_block = by * cutlass.Int32(TILE_N_XL)
    m_warp = m_block + warp_m * cutlass.Int32(WM_XL * 16)
    n_warp = n_block + warp_n * cutlass.Int32(WN_XL * 8)

    sA_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, A_SMEM_XL_DB, alignment=128)
    sB_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, B_SMEM_XL_DB, alignment=128)
    sA = cute.make_tensor(
        sA_ptr, cute.make_layout((A_SMEM_XL_DB,), stride=(cutlass.Int32(1),))
    )
    sB = cute.make_tensor(
        sB_ptr, cute.make_layout((B_SMEM_XL_DB,), stride=(cutlass.Int32(1),))
    )
    sA_base_u32 = _cvta_to_shared_u32(sA_ptr.toint())
    sB_base_u32 = _cvta_to_shared_u32(sB_ptr.toint())

    acc = cute.make_rmem_tensor(cute.make_layout((WM_XL, WN_XL, 4)), cutlass.Float32)
    for mi in cutlass.range_constexpr(WM_XL):
        for nj in cutlass.range_constexpr(WN_XL):
            for ci in cutlass.range_constexpr(4):
                acc[mi, nj, ci] = cutlass.Float32(0.0)

    K_u32_i64 = K_u32.to(cutlass.Int64)

    for li in cutlass.range_constexpr(A_LOADS_XL):
        v4 = tid + cutlass.Int32(li * TPB_XL)
        if v4 < cutlass.Int32(A_VEC4_XL):
            ar = v4 // cutlass.Int32(KT_PER_ROW)
            ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
            ac = ac4 * cutlass.Int32(4)
            gm = _clamp(m_block + ar, M)
            smem_off = ar * cutlass.Int32(SMEM_K_STRIDE) + ac
            smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = a_p_int64 + (
                gm.to(cutlass.Int64) * K_u32_i64 + ac.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    for li in cutlass.range_constexpr(B_LOADS_XL):
        v4 = tid + cutlass.Int32(li * TPB_XL)
        if v4 < cutlass.Int32(B_VEC4_XL):
            br = v4 // cutlass.Int32(KT_PER_ROW)
            bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
            bc = bc4 * cutlass.Int32(4)
            gn = _clamp(n_block + br, N)
            smem_off = br * cutlass.Int32(SMEM_K_STRIDE) + bc
            smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = b_p_int64 + (
                gn.to(cutlass.Int64) * K_u32_i64 + bc.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    _cp_async_commit()

    for kt in cutlass.range(0, n_k_tiles, 1, unroll=1):
        _cp_async_wait_0()
        _sync()

        cur_stage = kt & cutlass.Int32(1)
        cur_a_off = cur_stage * cutlass.Int32(A_SMEM_XL)
        cur_b_off = cur_stage * cutlass.Int32(B_SMEM_XL)

        if kt + cutlass.Int32(1) < n_k_tiles:
            nxt_stage = cur_stage ^ cutlass.Int32(1)
            nxt_a_smem_off = nxt_stage * cutlass.Int32(A_SMEM_XL)
            nxt_b_smem_off = nxt_stage * cutlass.Int32(B_SMEM_XL)
            k_off_next = (kt + cutlass.Int32(1)) * cutlass.Int32(K_TILE)
            for li in cutlass.range_constexpr(A_LOADS_XL):
                v4 = tid + cutlass.Int32(li * TPB_XL)
                if v4 < cutlass.Int32(A_VEC4_XL):
                    ar = v4 // cutlass.Int32(KT_PER_ROW)
                    ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                    ac = ac4 * cutlass.Int32(4)
                    gm = _clamp(m_block + ar, M)
                    smem_off = nxt_a_smem_off + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                    smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = a_p_int64 + (
                        gm.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + ac).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            for li in cutlass.range_constexpr(B_LOADS_XL):
                v4 = tid + cutlass.Int32(li * TPB_XL)
                if v4 < cutlass.Int32(B_VEC4_XL):
                    br = v4 // cutlass.Int32(KT_PER_ROW)
                    bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                    bc = bc4 * cutlass.Int32(4)
                    gn = _clamp(n_block + br, N)
                    smem_off = nxt_b_smem_off + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                    smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = b_p_int64 + (
                        gn.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + bc).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            _cp_async_commit()

        for ks in cutlass.range_constexpr(K_STEPS):
            ki = cutlass.Int32(ks * K_MMA)
            ct = ki + t
            ct4 = ct + cutlass.Int32(4)
            a_f = cute.make_rmem_tensor(cute.make_layout((WM_XL, 4)), cutlass.Int32)
            sf_a = cute.make_rmem_tensor(cute.make_layout((WM_XL,)), cutlass.Int32)
            sf_b_arr = cute.make_rmem_tensor(cute.make_layout((WN_XL,)), cutlass.Int32)
            rb_arr = cute.make_rmem_tensor(cute.make_layout((WN_XL, 2)), cutlass.Int32)
            kg = kt * cutlass.Int32(K_STEPS) + cutlass.Int32(ks)
            for mi in cutlass.range_constexpr(WM_XL):
                mt = warp_m * cutlass.Int32(WM_XL) + cutlass.Int32(mi)
                r0 = mt * cutlass.Int32(16) + g
                r1 = r0 + cutlass.Int32(8)
                a_f[mi, 0] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 1] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 2] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                a_f[mi, 3] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                mb = m_warp + cutlass.Int32(mi * 16)
                sr = mb + g + (t & cutlass.Int32(1)) * cutlass.Int32(8)
                sf_a[mi] = sfa_i32[
                    (sr >> 7) * sfa_stride
                    + kg * cutlass.Int32(128)
                    + (sr & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sr >> 5 & cutlass.Int32(3))
                ]
            for nj in cutlass.range_constexpr(WN_XL):
                nc = n_warp + cutlass.Int32(nj * 8)
                sc = nc + g
                sf_b_arr[nj] = sfb_i32[
                    (sc >> 7) * sfb_stride
                    + kg * cutlass.Int32(128)
                    + (sc & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sc >> 5 & cutlass.Int32(3))
                ]
            for nj in cutlass.range_constexpr(WN_XL):
                nt = warp_n * cutlass.Int32(WN_XL) + cutlass.Int32(nj)
                br_idx = nt * cutlass.Int32(8) + g
                rb_arr[nj, 0] = sB[
                    cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct
                ]
                rb_arr[nj, 1] = sB[
                    cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct4
                ]
            for nj in cutlass.range_constexpr(WN_XL):
                for mi in cutlass.range_constexpr(WM_XL):
                    d0, d1, d2, d3 = _mma(
                        a_f[mi, 0],
                        a_f[mi, 1],
                        a_f[mi, 2],
                        a_f[mi, 3],
                        rb_arr[nj, 0],
                        rb_arr[nj, 1],
                        acc[mi, nj, 0],
                        acc[mi, nj, 1],
                        acc[mi, nj, 2],
                        acc[mi, nj, 3],
                        sf_a[mi],
                        sf_b_arr[nj],
                    )
                    acc[mi, nj, 0] = d0
                    acc[mi, nj, 1] = d1
                    acc[mi, nj, 2] = d2
                    acc[mi, nj, 3] = d3

    alpha = alpha_vec[0]
    for mi in cutlass.range_constexpr(WM_XL):
        mb = m_warp + cutlass.Int32(mi * 16)
        for nj in cutlass.range_constexpr(WN_XL):
            nc = n_warp + cutlass.Int32(nj * 8)
            or0 = mb + g
            or1 = or0 + cutlass.Int32(8)
            oc0 = nc + t * cutlass.Int32(2)
            oc_half = (nc >> 1) + t
            if or0 < M:
                if oc0 < N:
                    out_i32[or0 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 0] * alpha, acc[mi, nj, 1] * alpha
                    )
            if or1 < M:
                if oc0 < N:
                    out_i32[or1 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 2] * alpha, acc[mi, nj, 3] * alpha
                    )


# 2-stage gemm_n64 (for M >= 256 N=64)
@cute.kernel
def _gemm_n64(
    a_p_int64: cutlass.Int64,
    b_p_int64: cutlass.Int64,
    sfa_i32: cute.Tensor,
    sfb_i32: cute.Tensor,
    out_i32: cute.Tensor,
    alpha_vec: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    K_u32: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    N_half: cutlass.Int32,
):
    bx, by, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    warp_id = tid >> 5
    lane_id = tid & cutlass.Int32(31)
    g = lane_id >> 2
    t = lane_id & cutlass.Int32(3)
    warp_m = warp_id >> 1
    warp_n = warp_id & cutlass.Int32(1)
    m_block = bx * cutlass.Int32(TILE_M_N64)
    n_block = by * cutlass.Int32(TILE_N_N64)
    m_warp = m_block + warp_m * cutlass.Int32(WM_N64 * 16)
    n_warp = n_block + warp_n * cutlass.Int32(WN_N64 * 8)

    sA_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, A_SMEM_N64_DB, alignment=128)
    sB_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, B_SMEM_N64_DB, alignment=128)
    sA = cute.make_tensor(
        sA_ptr, cute.make_layout((A_SMEM_N64_DB,), stride=(cutlass.Int32(1),))
    )
    sB = cute.make_tensor(
        sB_ptr, cute.make_layout((B_SMEM_N64_DB,), stride=(cutlass.Int32(1),))
    )
    sA_base_u32 = _cvta_to_shared_u32(sA_ptr.toint())
    sB_base_u32 = _cvta_to_shared_u32(sB_ptr.toint())

    acc = cute.make_rmem_tensor(cute.make_layout((WM_N64, WN_N64, 4)), cutlass.Float32)
    for mi in cutlass.range_constexpr(WM_N64):
        for nj in cutlass.range_constexpr(WN_N64):
            for ci in cutlass.range_constexpr(4):
                acc[mi, nj, ci] = cutlass.Float32(0.0)

    K_u32_i64 = K_u32.to(cutlass.Int64)

    for li in cutlass.range_constexpr(A_LOADS_N64):
        v4 = tid + cutlass.Int32(li * TPB)
        if v4 < cutlass.Int32(A_VEC4_N64):
            ar = v4 // cutlass.Int32(KT_PER_ROW)
            ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
            ac = ac4 * cutlass.Int32(4)
            gm = _clamp(m_block + ar, M)
            smem_off = ar * cutlass.Int32(SMEM_K_STRIDE) + ac
            smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = a_p_int64 + (
                gm.to(cutlass.Int64) * K_u32_i64 + ac.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    for li in cutlass.range_constexpr(B_LOADS_N64):
        v4 = tid + cutlass.Int32(li * TPB)
        if v4 < cutlass.Int32(B_VEC4_N64):
            br = v4 // cutlass.Int32(KT_PER_ROW)
            bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
            bc = bc4 * cutlass.Int32(4)
            gn = _clamp(n_block + br, N)
            smem_off = br * cutlass.Int32(SMEM_K_STRIDE) + bc
            smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = b_p_int64 + (
                gn.to(cutlass.Int64) * K_u32_i64 + bc.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    _cp_async_commit()

    for kt in cutlass.range(0, n_k_tiles, 1, unroll=1):
        _cp_async_wait_0()
        _sync()

        cur_stage = kt & cutlass.Int32(1)
        cur_a_off = cur_stage * cutlass.Int32(A_SMEM_N64)
        cur_b_off = cur_stage * cutlass.Int32(B_SMEM_N64)

        if kt + cutlass.Int32(1) < n_k_tiles:
            nxt_stage = cur_stage ^ cutlass.Int32(1)
            nxt_a_smem_off = nxt_stage * cutlass.Int32(A_SMEM_N64)
            nxt_b_smem_off = nxt_stage * cutlass.Int32(B_SMEM_N64)
            k_off_next = (kt + cutlass.Int32(1)) * cutlass.Int32(K_TILE)
            for li in cutlass.range_constexpr(A_LOADS_N64):
                v4 = tid + cutlass.Int32(li * TPB)
                if v4 < cutlass.Int32(A_VEC4_N64):
                    ar = v4 // cutlass.Int32(KT_PER_ROW)
                    ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                    ac = ac4 * cutlass.Int32(4)
                    gm = _clamp(m_block + ar, M)
                    smem_off = nxt_a_smem_off + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                    smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = a_p_int64 + (
                        gm.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + ac).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            for li in cutlass.range_constexpr(B_LOADS_N64):
                v4 = tid + cutlass.Int32(li * TPB)
                if v4 < cutlass.Int32(B_VEC4_N64):
                    br = v4 // cutlass.Int32(KT_PER_ROW)
                    bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                    bc = bc4 * cutlass.Int32(4)
                    gn = _clamp(n_block + br, N)
                    smem_off = nxt_b_smem_off + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                    smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = b_p_int64 + (
                        gn.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + bc).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            _cp_async_commit()

        for ks in cutlass.range_constexpr(K_STEPS):
            ki = cutlass.Int32(ks * K_MMA)
            ct = ki + t
            ct4 = ct + cutlass.Int32(4)
            a_f = cute.make_rmem_tensor(cute.make_layout((WM_N64, 4)), cutlass.Int32)
            sf_a = cute.make_rmem_tensor(cute.make_layout((WM_N64,)), cutlass.Int32)
            for mi in cutlass.range_constexpr(WM_N64):
                mt = warp_m * cutlass.Int32(WM_N64) + cutlass.Int32(mi)
                r0 = mt * cutlass.Int32(16) + g
                r1 = r0 + cutlass.Int32(8)
                a_f[mi, 0] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 1] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 2] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                a_f[mi, 3] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                mb = m_warp + cutlass.Int32(mi * 16)
                sr = mb + g + (t & cutlass.Int32(1)) * cutlass.Int32(8)
                kg = kt * cutlass.Int32(K_STEPS) + cutlass.Int32(ks)
                sf_a[mi] = sfa_i32[
                    (sr >> 7) * sfa_stride
                    + kg * cutlass.Int32(128)
                    + (sr & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sr >> 5 & cutlass.Int32(3))
                ]
            for nj in cutlass.range_constexpr(WN_N64):
                nt = warp_n * cutlass.Int32(WN_N64) + cutlass.Int32(nj)
                br_idx = nt * cutlass.Int32(8) + g
                rb0 = sB[cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct]
                rb1 = sB[cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                nc = n_warp + cutlass.Int32(nj * 8)
                sc = nc + g
                sf_b = sfb_i32[
                    (sc >> 7) * sfb_stride
                    + kg * cutlass.Int32(128)
                    + (sc & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sc >> 5 & cutlass.Int32(3))
                ]
                for mi in cutlass.range_constexpr(WM_N64):
                    d0, d1, d2, d3 = _mma(
                        a_f[mi, 0],
                        a_f[mi, 1],
                        a_f[mi, 2],
                        a_f[mi, 3],
                        rb0,
                        rb1,
                        acc[mi, nj, 0],
                        acc[mi, nj, 1],
                        acc[mi, nj, 2],
                        acc[mi, nj, 3],
                        sf_a[mi],
                        sf_b,
                    )
                    acc[mi, nj, 0] = d0
                    acc[mi, nj, 1] = d1
                    acc[mi, nj, 2] = d2
                    acc[mi, nj, 3] = d3

    alpha = alpha_vec[0]
    for mi in cutlass.range_constexpr(WM_N64):
        mb = m_warp + cutlass.Int32(mi * 16)
        for nj in cutlass.range_constexpr(WN_N64):
            nc = n_warp + cutlass.Int32(nj * 8)
            or0 = mb + g
            or1 = or0 + cutlass.Int32(8)
            oc0 = nc + t * cutlass.Int32(2)
            oc_half = (nc >> 1) + t
            if or0 < M:
                if oc0 < N:
                    out_i32[or0 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 0] * alpha, acc[mi, nj, 1] * alpha
                    )
            if or1 < M:
                if oc0 < N:
                    out_i32[or1 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 2] * alpha, acc[mi, nj, 3] * alpha
                    )


# gemm_small_n64 path for the small-M, N=64 dispatch case.
@cute.kernel
def _gemm_small_n64(
    a_p_int64: cutlass.Int64,
    b_p_int64: cutlass.Int64,
    sfa_i32: cute.Tensor,
    sfb_i32: cute.Tensor,
    out_i32: cute.Tensor,
    alpha_vec: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    K_u32: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    N_half: cutlass.Int32,
):
    bx, by, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    warp_id = tid >> 5
    lane_id = tid & cutlass.Int32(31)
    g = lane_id >> 2
    t = lane_id & cutlass.Int32(3)
    warp_n = warp_id
    m_block = bx * cutlass.Int32(TILE_M_SN64)
    n_block = by * cutlass.Int32(TILE_N_SN64)
    n_warp = n_block + warp_n * cutlass.Int32(WN_SN64 * 8)

    sA_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, A_SMEM_SN64_3S, alignment=128)
    sB_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, B_SMEM_SN64_3S, alignment=128)
    sA = cute.make_tensor(
        sA_ptr, cute.make_layout((A_SMEM_SN64_3S,), stride=(cutlass.Int32(1),))
    )
    sB = cute.make_tensor(
        sB_ptr, cute.make_layout((B_SMEM_SN64_3S,), stride=(cutlass.Int32(1),))
    )
    sA_base_u32 = _cvta_to_shared_u32(sA_ptr.toint())
    sB_base_u32 = _cvta_to_shared_u32(sB_ptr.toint())

    acc = cute.make_rmem_tensor(
        cute.make_layout((WM_SN64, WN_SN64, 4)), cutlass.Float32
    )
    for mi in cutlass.range_constexpr(WM_SN64):
        for nj in cutlass.range_constexpr(WN_SN64):
            for ci in cutlass.range_constexpr(4):
                acc[mi, nj, ci] = cutlass.Float32(0.0)

    K_u32_i64 = K_u32.to(cutlass.Int64)

    # Prologue: stage 0
    for li in cutlass.range_constexpr(A_LOADS_SN64):
        v4 = tid + cutlass.Int32(li * TPB)
        if v4 < cutlass.Int32(A_VEC4_SN64):
            ar = v4 // cutlass.Int32(KT_PER_ROW)
            ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
            ac = ac4 * cutlass.Int32(4)
            gm = _clamp(m_block + ar, M)
            smem_off = ar * cutlass.Int32(SMEM_K_STRIDE) + ac
            smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = a_p_int64 + (
                gm.to(cutlass.Int64) * K_u32_i64 + ac.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    for li in cutlass.range_constexpr(B_LOADS_SN64):
        v4 = tid + cutlass.Int32(li * TPB)
        if v4 < cutlass.Int32(B_VEC4_SN64):
            br = v4 // cutlass.Int32(KT_PER_ROW)
            bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
            bc = bc4 * cutlass.Int32(4)
            gn = _clamp(n_block + br, N)
            smem_off = br * cutlass.Int32(SMEM_K_STRIDE) + bc
            smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = b_p_int64 + (
                gn.to(cutlass.Int64) * K_u32_i64 + bc.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    _cp_async_commit()

    # Prologue: stage 1
    if cutlass.Int32(1) < n_k_tiles:
        k_off_1 = cutlass.Int32(K_TILE)
        for li in cutlass.range_constexpr(A_LOADS_SN64):
            v4 = tid + cutlass.Int32(li * TPB)
            if v4 < cutlass.Int32(A_VEC4_SN64):
                ar = v4 // cutlass.Int32(KT_PER_ROW)
                ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                ac = ac4 * cutlass.Int32(4)
                gm = _clamp(m_block + ar, M)
                smem_off = (
                    cutlass.Int32(A_SMEM_SN64) + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                )
                smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                gmem_addr = a_p_int64 + (
                    gm.to(cutlass.Int64) * K_u32_i64 + (k_off_1 + ac).to(cutlass.Int64)
                ) * cutlass.Int64(4)
                _cp_async_16(smem_addr, gmem_addr)
        for li in cutlass.range_constexpr(B_LOADS_SN64):
            v4 = tid + cutlass.Int32(li * TPB)
            if v4 < cutlass.Int32(B_VEC4_SN64):
                br = v4 // cutlass.Int32(KT_PER_ROW)
                bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                bc = bc4 * cutlass.Int32(4)
                gn = _clamp(n_block + br, N)
                smem_off = (
                    cutlass.Int32(B_SMEM_SN64) + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                )
                smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                gmem_addr = b_p_int64 + (
                    gn.to(cutlass.Int64) * K_u32_i64 + (k_off_1 + bc).to(cutlass.Int64)
                ) * cutlass.Int64(4)
                _cp_async_16(smem_addr, gmem_addr)
        _cp_async_commit()

    for kt in cutlass.range(0, n_k_tiles, 1, unroll=1):
        _cp_async_wait_1()
        _sync()

        stage_kt = kt - (kt // cutlass.Int32(NSTAGES_S)) * cutlass.Int32(NSTAGES_S)
        cur_a_off = stage_kt * cutlass.Int32(A_SMEM_SN64)
        cur_b_off = stage_kt * cutlass.Int32(B_SMEM_SN64)

        kt_next = kt + cutlass.Int32(2)
        if kt_next < n_k_tiles:
            stage_next = kt_next - (
                kt_next // cutlass.Int32(NSTAGES_S)
            ) * cutlass.Int32(NSTAGES_S)
            nxt_a_smem_off = stage_next * cutlass.Int32(A_SMEM_SN64)
            nxt_b_smem_off = stage_next * cutlass.Int32(B_SMEM_SN64)
            k_off_next = kt_next * cutlass.Int32(K_TILE)
            for li in cutlass.range_constexpr(A_LOADS_SN64):
                v4 = tid + cutlass.Int32(li * TPB)
                if v4 < cutlass.Int32(A_VEC4_SN64):
                    ar = v4 // cutlass.Int32(KT_PER_ROW)
                    ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                    ac = ac4 * cutlass.Int32(4)
                    gm = _clamp(m_block + ar, M)
                    smem_off = nxt_a_smem_off + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                    smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = a_p_int64 + (
                        gm.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + ac).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            for li in cutlass.range_constexpr(B_LOADS_SN64):
                v4 = tid + cutlass.Int32(li * TPB)
                if v4 < cutlass.Int32(B_VEC4_SN64):
                    br = v4 // cutlass.Int32(KT_PER_ROW)
                    bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                    bc = bc4 * cutlass.Int32(4)
                    gn = _clamp(n_block + br, N)
                    smem_off = nxt_b_smem_off + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                    smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = b_p_int64 + (
                        gn.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + bc).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            _cp_async_commit()

        for ks in cutlass.range_constexpr(K_STEPS):
            ki = cutlass.Int32(ks * K_MMA)
            ct = ki + t
            ct4 = ct + cutlass.Int32(4)
            a_f = cute.make_rmem_tensor(cute.make_layout((WM_SN64, 4)), cutlass.Int32)
            sf_a = cute.make_rmem_tensor(cute.make_layout((WM_SN64,)), cutlass.Int32)
            for mi in cutlass.range_constexpr(WM_SN64):
                r0 = cutlass.Int32(mi) * cutlass.Int32(16) + g
                r1 = r0 + cutlass.Int32(8)
                a_f[mi, 0] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 1] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 2] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                a_f[mi, 3] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                mb = m_block + cutlass.Int32(mi * 16)
                sr = mb + g + (t & cutlass.Int32(1)) * cutlass.Int32(8)
                kg = kt * cutlass.Int32(K_STEPS) + cutlass.Int32(ks)
                sf_a[mi] = sfa_i32[
                    (sr >> 7) * sfa_stride
                    + kg * cutlass.Int32(128)
                    + (sr & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sr >> 5 & cutlass.Int32(3))
                ]
            for nj in cutlass.range_constexpr(WN_SN64):
                nt = warp_n * cutlass.Int32(WN_SN64) + cutlass.Int32(nj)
                br_idx = nt * cutlass.Int32(8) + g
                rb0 = sB[cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct]
                rb1 = sB[cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                nc = n_warp + cutlass.Int32(nj * 8)
                sc = nc + g
                sf_b = sfb_i32[
                    (sc >> 7) * sfb_stride
                    + kg * cutlass.Int32(128)
                    + (sc & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sc >> 5 & cutlass.Int32(3))
                ]
                for mi in cutlass.range_constexpr(WM_SN64):
                    d0, d1, d2, d3 = _mma(
                        a_f[mi, 0],
                        a_f[mi, 1],
                        a_f[mi, 2],
                        a_f[mi, 3],
                        rb0,
                        rb1,
                        acc[mi, nj, 0],
                        acc[mi, nj, 1],
                        acc[mi, nj, 2],
                        acc[mi, nj, 3],
                        sf_a[mi],
                        sf_b,
                    )
                    acc[mi, nj, 0] = d0
                    acc[mi, nj, 1] = d1
                    acc[mi, nj, 2] = d2
                    acc[mi, nj, 3] = d3

    alpha = alpha_vec[0]
    for mi in cutlass.range_constexpr(WM_SN64):
        mb = m_block + cutlass.Int32(mi * 16)
        for nj in cutlass.range_constexpr(WN_SN64):
            nc = n_warp + cutlass.Int32(nj * 8)
            or0 = mb + g
            or1 = or0 + cutlass.Int32(8)
            oc0 = nc + t * cutlass.Int32(2)
            oc_half = (nc >> 1) + t
            if or0 < M:
                if oc0 < N:
                    out_i32[or0 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 0] * alpha, acc[mi, nj, 1] * alpha
                    )
            if or1 < M:
                if oc0 < N:
                    out_i32[or1 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 2] * alpha, acc[mi, nj, 3] * alpha
                    )


# 3-stage gemm_small for M ≤ 128 N != 64 (already 3-stage in baseline)
@cute.kernel
def _gemm_small(
    a_p_int64: cutlass.Int64,
    b_p_int64: cutlass.Int64,
    sfa_i32: cute.Tensor,
    sfb_i32: cute.Tensor,
    out_i32: cute.Tensor,
    alpha_vec: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    K_u32: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    N_half: cutlass.Int32,
):
    bx, by, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    warp_id = tid >> 5
    lane_id = tid & cutlass.Int32(31)
    g = lane_id >> 2
    t = lane_id & cutlass.Int32(3)
    warp_n = warp_id
    m_block = bx * cutlass.Int32(TILE_M_S)
    n_block = by * cutlass.Int32(TILE_N_S)
    n_warp = n_block + warp_n * cutlass.Int32(WN_S * 8)

    sA_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, A_SMEM_S_TOTAL, alignment=128)
    sB_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, B_SMEM_S_TOTAL, alignment=128)
    sA = cute.make_tensor(
        sA_ptr, cute.make_layout((A_SMEM_S_TOTAL,), stride=(cutlass.Int32(1),))
    )
    sB = cute.make_tensor(
        sB_ptr, cute.make_layout((B_SMEM_S_TOTAL,), stride=(cutlass.Int32(1),))
    )
    sA_base_u32 = _cvta_to_shared_u32(sA_ptr.toint())
    sB_base_u32 = _cvta_to_shared_u32(sB_ptr.toint())

    acc = cute.make_rmem_tensor(cute.make_layout((WM_S, WN_S, 4)), cutlass.Float32)
    for mi in cutlass.range_constexpr(WM_S):
        for nj in cutlass.range_constexpr(WN_S):
            for ci in cutlass.range_constexpr(4):
                acc[mi, nj, ci] = cutlass.Float32(0.0)

    K_u32_i64 = K_u32.to(cutlass.Int64)

    # Prologue: stage 0
    for li in cutlass.range_constexpr(A_LOADS_S):
        v4 = tid + cutlass.Int32(li * TPB)
        if v4 < cutlass.Int32(A_VEC4_S):
            ar = v4 // cutlass.Int32(KT_PER_ROW)
            ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
            ac = ac4 * cutlass.Int32(4)
            gm = _clamp(m_block + ar, M)
            smem_off = ar * cutlass.Int32(SMEM_K_STRIDE) + ac
            smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = a_p_int64 + (
                gm.to(cutlass.Int64) * K_u32_i64 + ac.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    for li in cutlass.range_constexpr(B_LOADS_S):
        v4 = tid + cutlass.Int32(li * TPB)
        if v4 < cutlass.Int32(B_VEC4_S):
            br = v4 // cutlass.Int32(KT_PER_ROW)
            bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
            bc = bc4 * cutlass.Int32(4)
            gn = _clamp(n_block + br, N)
            smem_off = br * cutlass.Int32(SMEM_K_STRIDE) + bc
            smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = b_p_int64 + (
                gn.to(cutlass.Int64) * K_u32_i64 + bc.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    _cp_async_commit()

    # Prologue: stage 1
    if cutlass.Int32(1) < n_k_tiles:
        k_off_1 = cutlass.Int32(K_TILE)
        for li in cutlass.range_constexpr(A_LOADS_S):
            v4 = tid + cutlass.Int32(li * TPB)
            if v4 < cutlass.Int32(A_VEC4_S):
                ar = v4 // cutlass.Int32(KT_PER_ROW)
                ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                ac = ac4 * cutlass.Int32(4)
                gm = _clamp(m_block + ar, M)
                smem_off = (
                    cutlass.Int32(A_SMEM_S) + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                )
                smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                gmem_addr = a_p_int64 + (
                    gm.to(cutlass.Int64) * K_u32_i64 + (k_off_1 + ac).to(cutlass.Int64)
                ) * cutlass.Int64(4)
                _cp_async_16(smem_addr, gmem_addr)
        for li in cutlass.range_constexpr(B_LOADS_S):
            v4 = tid + cutlass.Int32(li * TPB)
            if v4 < cutlass.Int32(B_VEC4_S):
                br = v4 // cutlass.Int32(KT_PER_ROW)
                bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                bc = bc4 * cutlass.Int32(4)
                gn = _clamp(n_block + br, N)
                smem_off = (
                    cutlass.Int32(B_SMEM_S) + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                )
                smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                gmem_addr = b_p_int64 + (
                    gn.to(cutlass.Int64) * K_u32_i64 + (k_off_1 + bc).to(cutlass.Int64)
                ) * cutlass.Int64(4)
                _cp_async_16(smem_addr, gmem_addr)
        _cp_async_commit()

    for kt in cutlass.range(0, n_k_tiles, 1, unroll=1):
        _cp_async_wait_1()
        _sync()

        stage_kt = kt - (kt // cutlass.Int32(NSTAGES_S)) * cutlass.Int32(NSTAGES_S)
        cur_a_off = stage_kt * cutlass.Int32(A_SMEM_S)
        cur_b_off = stage_kt * cutlass.Int32(B_SMEM_S)

        kt_next = kt + cutlass.Int32(2)
        if kt_next < n_k_tiles:
            stage_next = kt_next - (
                kt_next // cutlass.Int32(NSTAGES_S)
            ) * cutlass.Int32(NSTAGES_S)
            nxt_a_smem_off = stage_next * cutlass.Int32(A_SMEM_S)
            nxt_b_smem_off = stage_next * cutlass.Int32(B_SMEM_S)
            k_off_next = kt_next * cutlass.Int32(K_TILE)
            for li in cutlass.range_constexpr(A_LOADS_S):
                v4 = tid + cutlass.Int32(li * TPB)
                if v4 < cutlass.Int32(A_VEC4_S):
                    ar = v4 // cutlass.Int32(KT_PER_ROW)
                    ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                    ac = ac4 * cutlass.Int32(4)
                    gm = _clamp(m_block + ar, M)
                    smem_off = nxt_a_smem_off + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                    smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = a_p_int64 + (
                        gm.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + ac).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            for li in cutlass.range_constexpr(B_LOADS_S):
                v4 = tid + cutlass.Int32(li * TPB)
                if v4 < cutlass.Int32(B_VEC4_S):
                    br = v4 // cutlass.Int32(KT_PER_ROW)
                    bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                    bc = bc4 * cutlass.Int32(4)
                    gn = _clamp(n_block + br, N)
                    smem_off = nxt_b_smem_off + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                    smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = b_p_int64 + (
                        gn.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + bc).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            _cp_async_commit()

        for ks in cutlass.range_constexpr(K_STEPS):
            ki = cutlass.Int32(ks * K_MMA)
            ct = ki + t
            ct4 = ct + cutlass.Int32(4)
            a_f = cute.make_rmem_tensor(cute.make_layout((WM_S, 4)), cutlass.Int32)
            sf_a = cute.make_rmem_tensor(cute.make_layout((WM_S,)), cutlass.Int32)
            for mi in cutlass.range_constexpr(WM_S):
                r0 = cutlass.Int32(mi) * cutlass.Int32(16) + g
                r1 = r0 + cutlass.Int32(8)
                a_f[mi, 0] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 1] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 2] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                a_f[mi, 3] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                mb = m_block + cutlass.Int32(mi * 16)
                sr = mb + g + (t & cutlass.Int32(1)) * cutlass.Int32(8)
                kg = kt * cutlass.Int32(K_STEPS) + cutlass.Int32(ks)
                sf_a[mi] = sfa_i32[
                    (sr >> 7) * sfa_stride
                    + kg * cutlass.Int32(128)
                    + (sr & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sr >> 5 & cutlass.Int32(3))
                ]
            for nj in cutlass.range_constexpr(WN_S):
                nt = warp_n * cutlass.Int32(WN_S) + cutlass.Int32(nj)
                br_idx = nt * cutlass.Int32(8) + g
                rb0 = sB[cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct]
                rb1 = sB[cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                nc = n_warp + cutlass.Int32(nj * 8)
                sc = nc + g
                sf_b = sfb_i32[
                    (sc >> 7) * sfb_stride
                    + kg * cutlass.Int32(128)
                    + (sc & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sc >> 5 & cutlass.Int32(3))
                ]
                for mi in cutlass.range_constexpr(WM_S):
                    d0, d1, d2, d3 = _mma(
                        a_f[mi, 0],
                        a_f[mi, 1],
                        a_f[mi, 2],
                        a_f[mi, 3],
                        rb0,
                        rb1,
                        acc[mi, nj, 0],
                        acc[mi, nj, 1],
                        acc[mi, nj, 2],
                        acc[mi, nj, 3],
                        sf_a[mi],
                        sf_b,
                    )
                    acc[mi, nj, 0] = d0
                    acc[mi, nj, 1] = d1
                    acc[mi, nj, 2] = d2
                    acc[mi, nj, 3] = d3

    alpha = alpha_vec[0]
    for mi in cutlass.range_constexpr(WM_S):
        mb = m_block + cutlass.Int32(mi * 16)
        for nj in cutlass.range_constexpr(WN_S):
            nc = n_warp + cutlass.Int32(nj * 8)
            or0 = mb + g
            or1 = or0 + cutlass.Int32(8)
            oc0 = nc + t * cutlass.Int32(2)
            oc_half = (nc >> 1) + t
            if or0 < M:
                if oc0 < N:
                    out_i32[or0 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 0] * alpha, acc[mi, nj, 1] * alpha
                    )
            if or1 < M:
                if oc0 < N:
                    out_i32[or1 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 2] * alpha, acc[mi, nj, 3] * alpha
                    )


# Micro N64 path with TILE_N=16 and TILE_M=16.
@cute.kernel
def _gemm_micro_n64(
    a_p_int64: cutlass.Int64,
    b_p_int64: cutlass.Int64,
    sfa_i32: cute.Tensor,
    sfb_i32: cute.Tensor,
    out_i32: cute.Tensor,
    alpha_vec: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    K_u32: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    N_half: cutlass.Int32,
):
    bx, by, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    lane_id = tid  # only 1 warp, tid is 0..31
    g = lane_id >> 2
    t = lane_id & cutlass.Int32(3)
    m_block = bx * cutlass.Int32(TILE_M_XS)
    n_block = by * cutlass.Int32(TILE_N_XS)
    n_warp = n_block

    sA_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, A_SMEM_XS_3S, alignment=128)
    sB_ptr = cute.arch.smem.alloc_smem(cutlass.Int32, B_SMEM_XS_3S, alignment=128)
    sA = cute.make_tensor(
        sA_ptr, cute.make_layout((A_SMEM_XS_3S,), stride=(cutlass.Int32(1),))
    )
    sB = cute.make_tensor(
        sB_ptr, cute.make_layout((B_SMEM_XS_3S,), stride=(cutlass.Int32(1),))
    )
    sA_base_u32 = _cvta_to_shared_u32(sA_ptr.toint())
    sB_base_u32 = _cvta_to_shared_u32(sB_ptr.toint())

    acc = cute.make_rmem_tensor(cute.make_layout((WM_XS, WN_XS, 4)), cutlass.Float32)
    for mi in cutlass.range_constexpr(WM_XS):
        for nj in cutlass.range_constexpr(WN_XS):
            for ci in cutlass.range_constexpr(4):
                acc[mi, nj, ci] = cutlass.Float32(0.0)

    K_u32_i64 = K_u32.to(cutlass.Int64)

    # Prologue: stage 0
    for li in cutlass.range_constexpr(A_LOADS_XS):
        v4 = tid + cutlass.Int32(li * TPB_XS)
        if v4 < cutlass.Int32(A_VEC4_XS):
            ar = v4 // cutlass.Int32(KT_PER_ROW)
            ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
            ac = ac4 * cutlass.Int32(4)
            gm = _clamp(m_block + ar, M)
            smem_off = ar * cutlass.Int32(SMEM_K_STRIDE) + ac
            smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = a_p_int64 + (
                gm.to(cutlass.Int64) * K_u32_i64 + ac.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    for li in cutlass.range_constexpr(B_LOADS_XS):
        v4 = tid + cutlass.Int32(li * TPB_XS)
        if v4 < cutlass.Int32(B_VEC4_XS):
            br = v4 // cutlass.Int32(KT_PER_ROW)
            bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
            bc = bc4 * cutlass.Int32(4)
            gn = _clamp(n_block + br, N)
            smem_off = br * cutlass.Int32(SMEM_K_STRIDE) + bc
            smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
            gmem_addr = b_p_int64 + (
                gn.to(cutlass.Int64) * K_u32_i64 + bc.to(cutlass.Int64)
            ) * cutlass.Int64(4)
            _cp_async_16(smem_addr, gmem_addr)
    _cp_async_commit()

    # Prologue: stage 1
    if cutlass.Int32(1) < n_k_tiles:
        k_off_1 = cutlass.Int32(K_TILE)
        for li in cutlass.range_constexpr(A_LOADS_XS):
            v4 = tid + cutlass.Int32(li * TPB_XS)
            if v4 < cutlass.Int32(A_VEC4_XS):
                ar = v4 // cutlass.Int32(KT_PER_ROW)
                ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                ac = ac4 * cutlass.Int32(4)
                gm = _clamp(m_block + ar, M)
                smem_off = (
                    cutlass.Int32(A_SMEM_XS) + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                )
                smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                gmem_addr = a_p_int64 + (
                    gm.to(cutlass.Int64) * K_u32_i64 + (k_off_1 + ac).to(cutlass.Int64)
                ) * cutlass.Int64(4)
                _cp_async_16(smem_addr, gmem_addr)
        for li in cutlass.range_constexpr(B_LOADS_XS):
            v4 = tid + cutlass.Int32(li * TPB_XS)
            if v4 < cutlass.Int32(B_VEC4_XS):
                br = v4 // cutlass.Int32(KT_PER_ROW)
                bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                bc = bc4 * cutlass.Int32(4)
                gn = _clamp(n_block + br, N)
                smem_off = (
                    cutlass.Int32(B_SMEM_XS) + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                )
                smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                gmem_addr = b_p_int64 + (
                    gn.to(cutlass.Int64) * K_u32_i64 + (k_off_1 + bc).to(cutlass.Int64)
                ) * cutlass.Int64(4)
                _cp_async_16(smem_addr, gmem_addr)
        _cp_async_commit()

    for kt in cutlass.range(0, n_k_tiles, 1, unroll=1):
        _cp_async_wait_1()
        # No bar.sync needed: only 1 warp per CTA. cp.async ordering within warp is enough.

        stage_kt = kt - (kt // cutlass.Int32(NSTAGES_S)) * cutlass.Int32(NSTAGES_S)
        cur_a_off = stage_kt * cutlass.Int32(A_SMEM_XS)
        cur_b_off = stage_kt * cutlass.Int32(B_SMEM_XS)

        kt_next = kt + cutlass.Int32(2)
        if kt_next < n_k_tiles:
            stage_next = kt_next - (
                kt_next // cutlass.Int32(NSTAGES_S)
            ) * cutlass.Int32(NSTAGES_S)
            nxt_a_smem_off = stage_next * cutlass.Int32(A_SMEM_XS)
            nxt_b_smem_off = stage_next * cutlass.Int32(B_SMEM_XS)
            k_off_next = kt_next * cutlass.Int32(K_TILE)
            for li in cutlass.range_constexpr(A_LOADS_XS):
                v4 = tid + cutlass.Int32(li * TPB_XS)
                if v4 < cutlass.Int32(A_VEC4_XS):
                    ar = v4 // cutlass.Int32(KT_PER_ROW)
                    ac4 = v4 - ar * cutlass.Int32(KT_PER_ROW)
                    ac = ac4 * cutlass.Int32(4)
                    gm = _clamp(m_block + ar, M)
                    smem_off = nxt_a_smem_off + ar * cutlass.Int32(SMEM_K_STRIDE) + ac
                    smem_addr = sA_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = a_p_int64 + (
                        gm.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + ac).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            for li in cutlass.range_constexpr(B_LOADS_XS):
                v4 = tid + cutlass.Int32(li * TPB_XS)
                if v4 < cutlass.Int32(B_VEC4_XS):
                    br = v4 // cutlass.Int32(KT_PER_ROW)
                    bc4 = v4 - br * cutlass.Int32(KT_PER_ROW)
                    bc = bc4 * cutlass.Int32(4)
                    gn = _clamp(n_block + br, N)
                    smem_off = nxt_b_smem_off + br * cutlass.Int32(SMEM_K_STRIDE) + bc
                    smem_addr = sB_base_u32 + smem_off * cutlass.Int32(4)
                    gmem_addr = b_p_int64 + (
                        gn.to(cutlass.Int64) * K_u32_i64
                        + (k_off_next + bc).to(cutlass.Int64)
                    ) * cutlass.Int64(4)
                    _cp_async_16(smem_addr, gmem_addr)
            _cp_async_commit()

        for ks in cutlass.range_constexpr(K_STEPS):
            ki = cutlass.Int32(ks * K_MMA)
            ct = ki + t
            ct4 = ct + cutlass.Int32(4)
            a_f = cute.make_rmem_tensor(cute.make_layout((WM_XS, 4)), cutlass.Int32)
            sf_a = cute.make_rmem_tensor(cute.make_layout((WM_XS,)), cutlass.Int32)
            for mi in cutlass.range_constexpr(WM_XS):
                r0 = cutlass.Int32(mi) * cutlass.Int32(16) + g
                r1 = r0 + cutlass.Int32(8)
                a_f[mi, 0] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 1] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct]
                a_f[mi, 2] = sA[cur_a_off + r0 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                a_f[mi, 3] = sA[cur_a_off + r1 * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                mb = m_block + cutlass.Int32(mi * 16)
                sr = mb + g + (t & cutlass.Int32(1)) * cutlass.Int32(8)
                kg = kt * cutlass.Int32(K_STEPS) + cutlass.Int32(ks)
                sf_a[mi] = sfa_i32[
                    (sr >> 7) * sfa_stride
                    + kg * cutlass.Int32(128)
                    + (sr & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sr >> 5 & cutlass.Int32(3))
                ]
            for nj in cutlass.range_constexpr(WN_XS):
                br_idx = cutlass.Int32(nj) * cutlass.Int32(8) + g
                rb0 = sB[cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct]
                rb1 = sB[cur_b_off + br_idx * cutlass.Int32(SMEM_K_STRIDE) + ct4]
                nc = n_warp + cutlass.Int32(nj * 8)
                sc = nc + g
                sf_b = sfb_i32[
                    (sc >> 7) * sfb_stride
                    + kg * cutlass.Int32(128)
                    + (sc & cutlass.Int32(31)) * cutlass.Int32(4)
                    + (sc >> 5 & cutlass.Int32(3))
                ]
                for mi in cutlass.range_constexpr(WM_XS):
                    d0, d1, d2, d3 = _mma(
                        a_f[mi, 0],
                        a_f[mi, 1],
                        a_f[mi, 2],
                        a_f[mi, 3],
                        rb0,
                        rb1,
                        acc[mi, nj, 0],
                        acc[mi, nj, 1],
                        acc[mi, nj, 2],
                        acc[mi, nj, 3],
                        sf_a[mi],
                        sf_b,
                    )
                    acc[mi, nj, 0] = d0
                    acc[mi, nj, 1] = d1
                    acc[mi, nj, 2] = d2
                    acc[mi, nj, 3] = d3

    alpha = alpha_vec[0]
    for mi in cutlass.range_constexpr(WM_XS):
        mb = m_block + cutlass.Int32(mi * 16)
        for nj in cutlass.range_constexpr(WN_XS):
            nc = n_warp + cutlass.Int32(nj * 8)
            or0 = mb + g
            or1 = or0 + cutlass.Int32(8)
            oc0 = nc + t * cutlass.Int32(2)
            oc_half = (nc >> 1) + t
            if or0 < M:
                if oc0 < N:
                    out_i32[or0 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 0] * alpha, acc[mi, nj, 1] * alpha
                    )
            if or1 < M:
                if oc0 < N:
                    out_i32[or1 * N_half + oc_half] = _pack_bf16x2(
                        acc[mi, nj, 2] * alpha, acc[mi, nj, 3] * alpha
                    )


@cute.jit
def _entry_micro_n64(
    a_p: cute.Pointer,
    b_p: cute.Pointer,
    sfa_p: cute.Pointer,
    sfb_p: cute.Pointer,
    out_p: cute.Pointer,
    alpha_p: cute.Pointer,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K_u32: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    sfa_elems: cutlass.Int32,
    sfb_elems: cutlass.Int32,
    N_half: cutlass.Int32,
    stream: cuda.CUstream,
):
    sfa = cute.make_tensor(
        sfa_p, cute.make_layout((sfa_elems,), stride=(cutlass.Int32(1),))
    )
    sfb = cute.make_tensor(
        sfb_p, cute.make_layout((sfb_elems,), stride=(cutlass.Int32(1),))
    )
    out = cute.make_tensor(
        out_p, cute.make_layout((M * N_half,), stride=(cutlass.Int32(1),))
    )
    av = cute.make_tensor(alpha_p, cute.make_layout((1,), stride=(cutlass.Int32(1),)))
    a_p_int64 = a_p.toint()
    b_p_int64 = b_p.toint()
    gm = (M + cutlass.Int32(TILE_M_XS - 1)) // cutlass.Int32(TILE_M_XS)
    gn = (N + cutlass.Int32(TILE_N_XS - 1)) // cutlass.Int32(TILE_N_XS)
    _gemm_micro_n64(
        a_p_int64,
        b_p_int64,
        sfa,
        sfb,
        out,
        av,
        M,
        N,
        n_k_tiles,
        K_u32,
        sfa_stride,
        sfb_stride,
        N_half,
    ).launch(grid=(gm, gn, 1), block=(TPB_XS, 1, 1), min_blocks_per_mp=4, stream=stream)


@cute.jit
def _entry_large(
    a_p: cute.Pointer,
    b_p: cute.Pointer,
    sfa_p: cute.Pointer,
    sfb_p: cute.Pointer,
    out_p: cute.Pointer,
    alpha_p: cute.Pointer,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K_u32: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    sfa_elems: cutlass.Int32,
    sfb_elems: cutlass.Int32,
    N_half: cutlass.Int32,
    group_m: cutlass.Int32,
    stream: cuda.CUstream,
):
    sfa = cute.make_tensor(
        sfa_p, cute.make_layout((sfa_elems,), stride=(cutlass.Int32(1),))
    )
    sfb = cute.make_tensor(
        sfb_p, cute.make_layout((sfb_elems,), stride=(cutlass.Int32(1),))
    )
    out = cute.make_tensor(
        out_p, cute.make_layout((M * N_half,), stride=(cutlass.Int32(1),))
    )
    av = cute.make_tensor(alpha_p, cute.make_layout((1,), stride=(cutlass.Int32(1),)))
    a_p_int64 = a_p.toint()
    b_p_int64 = b_p.toint()
    gm = (M + cutlass.Int32(TILE_M_L - 1)) // cutlass.Int32(TILE_M_L)
    gn = (N + cutlass.Int32(TILE_N_L - 1)) // cutlass.Int32(TILE_N_L)
    _gemm_large(
        a_p_int64,
        b_p_int64,
        sfa,
        sfb,
        out,
        av,
        M,
        N,
        n_k_tiles,
        K_u32,
        sfa_stride,
        sfb_stride,
        N_half,
        gm,
        gn,
        group_m,
    ).launch(grid=(gm, gn, 1), block=(TPB, 1, 1), stream=stream)


@cute.jit
def _entry_xlarge(
    a_p: cute.Pointer,
    b_p: cute.Pointer,
    sfa_p: cute.Pointer,
    sfb_p: cute.Pointer,
    out_p: cute.Pointer,
    alpha_p: cute.Pointer,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K_u32: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    sfa_elems: cutlass.Int32,
    sfb_elems: cutlass.Int32,
    N_half: cutlass.Int32,
    group_m: cutlass.Int32,
    stream: cuda.CUstream,
):
    sfa = cute.make_tensor(
        sfa_p, cute.make_layout((sfa_elems,), stride=(cutlass.Int32(1),))
    )
    sfb = cute.make_tensor(
        sfb_p, cute.make_layout((sfb_elems,), stride=(cutlass.Int32(1),))
    )
    out = cute.make_tensor(
        out_p, cute.make_layout((M * N_half,), stride=(cutlass.Int32(1),))
    )
    av = cute.make_tensor(alpha_p, cute.make_layout((1,), stride=(cutlass.Int32(1),)))
    a_p_int64 = a_p.toint()
    b_p_int64 = b_p.toint()
    gm = (M + cutlass.Int32(TILE_M_XL - 1)) // cutlass.Int32(TILE_M_XL)
    gn = (N + cutlass.Int32(TILE_N_XL - 1)) // cutlass.Int32(TILE_N_XL)
    _gemm_xlarge(
        a_p_int64,
        b_p_int64,
        sfa,
        sfb,
        out,
        av,
        M,
        N,
        n_k_tiles,
        K_u32,
        sfa_stride,
        sfb_stride,
        N_half,
        gm,
        gn,
        group_m,
    ).launch(grid=(gm, gn, 1), block=(TPB_XL, 1, 1), stream=stream)


@cute.jit
def _entry_n64(
    a_p: cute.Pointer,
    b_p: cute.Pointer,
    sfa_p: cute.Pointer,
    sfb_p: cute.Pointer,
    out_p: cute.Pointer,
    alpha_p: cute.Pointer,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K_u32: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    sfa_elems: cutlass.Int32,
    sfb_elems: cutlass.Int32,
    N_half: cutlass.Int32,
    stream: cuda.CUstream,
):
    sfa = cute.make_tensor(
        sfa_p, cute.make_layout((sfa_elems,), stride=(cutlass.Int32(1),))
    )
    sfb = cute.make_tensor(
        sfb_p, cute.make_layout((sfb_elems,), stride=(cutlass.Int32(1),))
    )
    out = cute.make_tensor(
        out_p, cute.make_layout((M * N_half,), stride=(cutlass.Int32(1),))
    )
    av = cute.make_tensor(alpha_p, cute.make_layout((1,), stride=(cutlass.Int32(1),)))
    a_p_int64 = a_p.toint()
    b_p_int64 = b_p.toint()
    gm = (M + cutlass.Int32(TILE_M_N64 - 1)) // cutlass.Int32(TILE_M_N64)
    gn = (N + cutlass.Int32(TILE_N_N64 - 1)) // cutlass.Int32(TILE_N_N64)
    _gemm_n64(
        a_p_int64,
        b_p_int64,
        sfa,
        sfb,
        out,
        av,
        M,
        N,
        n_k_tiles,
        K_u32,
        sfa_stride,
        sfb_stride,
        N_half,
    ).launch(grid=(gm, gn, 1), block=(TPB, 1, 1), min_blocks_per_mp=2, stream=stream)


@cute.jit
def _entry_small_n64(
    a_p: cute.Pointer,
    b_p: cute.Pointer,
    sfa_p: cute.Pointer,
    sfb_p: cute.Pointer,
    out_p: cute.Pointer,
    alpha_p: cute.Pointer,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K_u32: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    sfa_elems: cutlass.Int32,
    sfb_elems: cutlass.Int32,
    N_half: cutlass.Int32,
    stream: cuda.CUstream,
):
    sfa = cute.make_tensor(
        sfa_p, cute.make_layout((sfa_elems,), stride=(cutlass.Int32(1),))
    )
    sfb = cute.make_tensor(
        sfb_p, cute.make_layout((sfb_elems,), stride=(cutlass.Int32(1),))
    )
    out = cute.make_tensor(
        out_p, cute.make_layout((M * N_half,), stride=(cutlass.Int32(1),))
    )
    av = cute.make_tensor(alpha_p, cute.make_layout((1,), stride=(cutlass.Int32(1),)))
    a_p_int64 = a_p.toint()
    b_p_int64 = b_p.toint()
    gm = (M + cutlass.Int32(TILE_M_SN64 - 1)) // cutlass.Int32(TILE_M_SN64)
    gn = (N + cutlass.Int32(TILE_N_SN64 - 1)) // cutlass.Int32(TILE_N_SN64)
    _gemm_small_n64(
        a_p_int64,
        b_p_int64,
        sfa,
        sfb,
        out,
        av,
        M,
        N,
        n_k_tiles,
        K_u32,
        sfa_stride,
        sfb_stride,
        N_half,
    ).launch(grid=(gm, gn, 1), block=(TPB, 1, 1), min_blocks_per_mp=2, stream=stream)


@cute.jit
def _entry_small(
    a_p: cute.Pointer,
    b_p: cute.Pointer,
    sfa_p: cute.Pointer,
    sfb_p: cute.Pointer,
    out_p: cute.Pointer,
    alpha_p: cute.Pointer,
    M: cutlass.Int32,
    N: cutlass.Int32,
    K_u32: cutlass.Int32,
    n_k_tiles: cutlass.Int32,
    sfa_stride: cutlass.Int32,
    sfb_stride: cutlass.Int32,
    sfa_elems: cutlass.Int32,
    sfb_elems: cutlass.Int32,
    N_half: cutlass.Int32,
    stream: cuda.CUstream,
):
    sfa = cute.make_tensor(
        sfa_p, cute.make_layout((sfa_elems,), stride=(cutlass.Int32(1),))
    )
    sfb = cute.make_tensor(
        sfb_p, cute.make_layout((sfb_elems,), stride=(cutlass.Int32(1),))
    )
    out = cute.make_tensor(
        out_p, cute.make_layout((M * N_half,), stride=(cutlass.Int32(1),))
    )
    av = cute.make_tensor(alpha_p, cute.make_layout((1,), stride=(cutlass.Int32(1),)))
    a_p_int64 = a_p.toint()
    b_p_int64 = b_p.toint()
    gm = (M + cutlass.Int32(TILE_M_S - 1)) // cutlass.Int32(TILE_M_S)
    gn = (N + cutlass.Int32(TILE_N_S - 1)) // cutlass.Int32(TILE_N_S)
    _gemm_small(
        a_p_int64,
        b_p_int64,
        sfa,
        sfb,
        out,
        av,
        M,
        N,
        n_k_tiles,
        K_u32,
        sfa_stride,
        sfb_stride,
        N_half,
    ).launch(grid=(gm, gn, 1), block=(TPB, 1, 1), stream=stream)


_compiled_large = None
_compiled_small = None
_compiled_n64 = None
_compiled_small_n64 = None
_compiled_micro_n64 = None
_compiled_xlarge = None


@torch.no_grad()
def run(a, b, a_descale, b_descale, alpha, out):
    global \
        _compiled_large, \
        _compiled_small, \
        _compiled_n64, \
        _compiled_small_n64, \
        _compiled_micro_n64, \
        _compiled_xlarge
    M, K_packed = a.shape
    K = K_packed * 2
    N = out.shape[1]
    stream = cuda.CUstream(torch.cuda.current_stream(device=a.device).cuda_stream)
    fake_stream = cuda.CUstream(0)
    K_u32 = K_packed // 4
    K_blocks = K // 16
    K_blocks_pad = ((K_blocks + 3) // 4) * 4
    n_k_tiles = K // 256
    N_half = N // 2

    M_pad = a_descale.shape[0]
    N_pad = b_descale.shape[1]
    sfa_stride = 128 * K_blocks_pad // 4
    sfb_stride = 128 * K_blocks_pad // 4
    sfa_elems = M_pad * K_blocks_pad // 4
    sfb_elems = N_pad * K_blocks_pad // 4

    a_ptr = make_ptr(
        cutlass.Int32, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        cutlass.Int32, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        cutlass.Int32, a_descale.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )
    sfb_ptr = make_ptr(
        cutlass.Int32, b_descale.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )
    out_ptr = make_ptr(
        cutlass.Int32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )
    alpha_ptr = make_ptr(
        cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
    )

    use_n64 = N == 64 and M >= 256
    use_micro_n64 = M <= 16 and N <= 1024
    use_small_n64 = N == 64 and M <= 128 and not use_micro_n64
    use_xlarge = (
        not use_n64
        and N >= 256
        and (
            M >= 4096
            or (M >= 1024 and K >= 4096)
            or (M == 1024 and N >= 8192)
            or (M == 128 and K >= 4096 and N >= 4096)
        )
    )
    use_small = (M <= 128) and (not use_xlarge)
    if use_n64:
        if _compiled_n64 is None:
            _compiled_n64 = cute.compile(
                _entry_n64,
                a_ptr,
                b_ptr,
                sfa_ptr,
                sfb_ptr,
                out_ptr,
                alpha_ptr,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                fake_stream,
            )
        _compiled_n64(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            out_ptr,
            alpha_ptr,
            cutlass.Int32(M),
            cutlass.Int32(N),
            cutlass.Int32(K_u32),
            cutlass.Int32(n_k_tiles),
            cutlass.Int32(sfa_stride),
            cutlass.Int32(sfb_stride),
            cutlass.Int32(sfa_elems),
            cutlass.Int32(sfb_elems),
            cutlass.Int32(N_half),
            stream,
        )
    elif use_micro_n64:
        if _compiled_micro_n64 is None:
            _compiled_micro_n64 = cute.compile(
                _entry_micro_n64,
                a_ptr,
                b_ptr,
                sfa_ptr,
                sfb_ptr,
                out_ptr,
                alpha_ptr,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                fake_stream,
            )
        _compiled_micro_n64(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            out_ptr,
            alpha_ptr,
            cutlass.Int32(M),
            cutlass.Int32(N),
            cutlass.Int32(K_u32),
            cutlass.Int32(n_k_tiles),
            cutlass.Int32(sfa_stride),
            cutlass.Int32(sfb_stride),
            cutlass.Int32(sfa_elems),
            cutlass.Int32(sfb_elems),
            cutlass.Int32(N_half),
            stream,
        )
    elif use_small_n64:
        if _compiled_small_n64 is None:
            _compiled_small_n64 = cute.compile(
                _entry_small_n64,
                a_ptr,
                b_ptr,
                sfa_ptr,
                sfb_ptr,
                out_ptr,
                alpha_ptr,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                fake_stream,
            )
        _compiled_small_n64(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            out_ptr,
            alpha_ptr,
            cutlass.Int32(M),
            cutlass.Int32(N),
            cutlass.Int32(K_u32),
            cutlass.Int32(n_k_tiles),
            cutlass.Int32(sfa_stride),
            cutlass.Int32(sfb_stride),
            cutlass.Int32(sfa_elems),
            cutlass.Int32(sfb_elems),
            cutlass.Int32(N_half),
            stream,
        )
    elif use_small:
        if _compiled_small is None:
            _compiled_small = cute.compile(
                _entry_small,
                a_ptr,
                b_ptr,
                sfa_ptr,
                sfb_ptr,
                out_ptr,
                alpha_ptr,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                fake_stream,
            )
        _compiled_small(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            out_ptr,
            alpha_ptr,
            cutlass.Int32(M),
            cutlass.Int32(N),
            cutlass.Int32(K_u32),
            cutlass.Int32(n_k_tiles),
            cutlass.Int32(sfa_stride),
            cutlass.Int32(sfb_stride),
            cutlass.Int32(sfa_elems),
            cutlass.Int32(sfb_elems),
            cutlass.Int32(N_half),
            stream,
        )
    elif use_xlarge:
        if _compiled_xlarge is None:
            _compiled_xlarge = cute.compile(
                _entry_xlarge,
                a_ptr,
                b_ptr,
                sfa_ptr,
                sfb_ptr,
                out_ptr,
                alpha_ptr,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                fake_stream,
            )
        # Adaptive GROUP_M: use gm value when gn > gm (effectively no swizzle), else default 4
        gm_val = (M + 127) // 128
        gn_val = (N + 127) // 128
        group_m_val = gm_val if gn_val > gm_val else 4
        _compiled_xlarge(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            out_ptr,
            alpha_ptr,
            cutlass.Int32(M),
            cutlass.Int32(N),
            cutlass.Int32(K_u32),
            cutlass.Int32(n_k_tiles),
            cutlass.Int32(sfa_stride),
            cutlass.Int32(sfb_stride),
            cutlass.Int32(sfa_elems),
            cutlass.Int32(sfb_elems),
            cutlass.Int32(N_half),
            cutlass.Int32(group_m_val),
            stream,
        )
    else:
        if _compiled_large is None:
            _compiled_large = cute.compile(
                _entry_large,
                a_ptr,
                b_ptr,
                sfa_ptr,
                sfb_ptr,
                out_ptr,
                alpha_ptr,
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                fake_stream,
            )
        # Adaptive GROUP_M for large kernel
        gm_l_val = (M + 63) // 64
        gn_l_val = (N + 127) // 128
        group_m_l_val = gm_l_val if gn_l_val > gm_l_val else 4
        _compiled_large(
            a_ptr,
            b_ptr,
            sfa_ptr,
            sfb_ptr,
            out_ptr,
            alpha_ptr,
            cutlass.Int32(M),
            cutlass.Int32(N),
            cutlass.Int32(K_u32),
            cutlass.Int32(n_k_tiles),
            cutlass.Int32(sfa_stride),
            cutlass.Int32(sfb_stride),
            cutlass.Int32(sfa_elems),
            cutlass.Int32(sfb_elems),
            cutlass.Int32(N_half),
            cutlass.Int32(group_m_l_val),
            stream,
        )
