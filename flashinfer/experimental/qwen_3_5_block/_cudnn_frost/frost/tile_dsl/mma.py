# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from typing import Type

from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute

from .swizzle import swizzle_xor_128b, swizzle_lin_128b


@cute.jit
def mma_m16n8k16_f32(
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
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    """``mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32``."""
    if cutlass.const_expr(ab_dtype != cutlass.Float16 and ab_dtype != cutlass.BFloat16):
        raise TypeError(f"Invalid A/B dtype: {ab_dtype}")
    ab_tag = "f16" if cutlass.const_expr(ab_dtype == cutlass.Float16) else "bf16"
    return cute.arch.inline_ptx(
        f"mma.sync.aligned.m16n8k16.row.col.f32.{ab_tag}.{ab_tag}.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};",
        write_only_types=[
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
        ],
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


@cute.jit
def mma_m16n8k32_f32(
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
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    """``mma.sync.aligned.m16n8k32.row.col.f32.{e4m3|e5m2}.{e4m3|e5m2}.f32``.

    The C operands travel as ``Int32`` bit patterns and are ``mov.b32``'d into
    ``.f32`` temps inside the asm block: cutlass-dsl 4.7.0's ``inline_ptx``
    fails in libNVVM when a compile-time-constant ``Float32`` reaches
    ``read_only_args``, and an accumulator's zero-init can fold to a constant.
    """
    if cutlass.const_expr(ab_dtype != cutlass.Float8E4M3FN and ab_dtype != cutlass.Float8E5M2):
        raise TypeError(f"Invalid A/B dtype: {ab_dtype}")
    ab_tag = "e4m3" if cutlass.const_expr(ab_dtype == cutlass.Float8E4M3FN) else "e5m2"
    return cute.arch.inline_ptx(
        "{ .reg .f32 fc<4>; "
        "mov.b32 fc0, {$r6}; mov.b32 fc1, {$r7}; mov.b32 fc2, {$r8}; mov.b32 fc3, {$r9}; "
        f"mma.sync.aligned.m16n8k32.row.col.f32.{ab_tag}.{ab_tag}.f32 "
        "{{$w0},{$w1},{$w2},{$w3}}, {{$r0},{$r1},{$r2},{$r3}}, {{$r4},{$r5}}, {fc0,fc1,fc2,fc3}; }",
        write_only_types=[
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
        ],
        read_only_args=[
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
            c0.bitcast(cutlass.Int32),
            c1.bitcast(cutlass.Int32),
            c2.bitcast(cutlass.Int32),
            c3.bitcast(cutlass.Int32),
        ],
    )


@cute.jit
def mma_m16n8k32_mxf8_f32(
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
    sf_a_byte: cutlass.Constexpr[int],
    sf_b_byte: cutlass.Constexpr[int],
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    """Block-scaled ``mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`` (UE8M0 scales).

    One E8M0 scale per row of A and per column of B covers the whole k32 step.
    ``sf_a`` / ``sf_b`` are 32-bit registers holding up to four scale bytes;
    ``sf_a_byte`` / ``sf_b_byte`` select one (immediate byte selectors, thread
    selector 0). Lane ``t`` supplies the A scale of row ``t // 4 + 8 * (t % 2)``
    and the B scale of column ``t // 4`` (the ``m16n8`` C-fragment rows and the
    B-fragment column).
    """
    if cutlass.const_expr(ab_dtype != cutlass.Float8E4M3FN and ab_dtype != cutlass.Float8E5M2):
        raise TypeError(f"Invalid A/B dtype: {ab_dtype}")
    if cutlass.const_expr(sf_a_byte not in (0, 1, 2, 3) or sf_b_byte not in (0, 1, 2, 3)):
        raise ValueError(f"scale byte selectors must be in [0, 4); got {sf_a_byte}, {sf_b_byte}")
    ab_tag = "e4m3" if cutlass.const_expr(ab_dtype == cutlass.Float8E4M3FN) else "e5m2"
    return cute.arch.inline_ptx(
        "{ .reg .f32 fc<4>; .reg .b32 sfa_data, sfb_data; "
        "mov.b32 fc0, {$r6}; mov.b32 fc1, {$r7}; mov.b32 fc2, {$r8}; mov.b32 fc3, {$r9}; "
        "mov.b32 sfa_data, {$r10}; mov.b32 sfb_data, {$r11}; "
        f"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.{ab_tag}.{ab_tag}.f32.ue8m0 "
        "{{$w0},{$w1},{$w2},{$w3}}, {{$r0},{$r1},{$r2},{$r3}}, {{$r4},{$r5}}, {fc0,fc1,fc2,fc3}, "
        f"sfa_data, {{{sf_a_byte}, 0}}, sfb_data, {{{sf_b_byte}, 0}}; }}",
        write_only_types=[
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
            cutlass.Float32,
        ],
        read_only_args=[
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
            c0.bitcast(cutlass.Int32),
            c1.bitcast(cutlass.Int32),
            c2.bitcast(cutlass.Int32),
            c3.bitcast(cutlass.Int32),
            sf_a,
            sf_b,
        ],
    )


@cute.jit
def desc_opaque(value, anchor=None):
    """Identity mov.b64 so LLVM cannot reassociate a descriptor base into per-k-step 64-bit literals.

    ``anchor`` (optional): a loop-VARIANT ``Int32`` (the kv-block or q-tile counter) passed as an extra, unreferenced
    asm operand.  Without it, a base whose SMEM address is static and whose ring index folds to a constant (a 1-stage
    ring: ``idx % 1 == 0``) is KERNEL-invariant, so LLVM's LICM hoists this mov AND every ``base + k_step`` add to the
    persistent loop's preheader; the sm107 d256 backward's 40-register MMA warp then held 32 descriptor pairs (K front
    8 + V 8 + K back-half UTCCP 16 = 64 URs, over the 63-UR file) live across the WHOLE kernel and spilled 38 STL /
    45 LDL (2026-09-23, sm_107a).  A loop-variant operand pins the mov -- and the adds that depend on it -- inside the
    loop body next to the MMA / UTCCP that consumes them, so their live range is one iteration.  The default path
    (no anchor) emits exactly the asm it always did.
    """
    args = [value] if cutlass.const_expr(anchor is None) else [value, anchor]
    return inline_ptx(
        "mov.b64 $0, $1;",
        write_only_types=[type(value)],
        read_only_args=args,
    )


@cute.jit
def mma_ss(
    desc,
    desc_a_base,
    desc_b_base,
    tmem_c,
    tmem_sf_a=None,
    tmem_sf_b=None,
    accumulate: bool = False,
    k_start: int = 0,
    k_count=None,
    elect_once: bool = False,
):
    if cutlass.const_expr(desc.cta_group == 1):
        cta_group_kind = nvvm.CTAGroup.CTA_1
    else:
        cta_group_kind = nvvm.CTAGroup.CTA_2
    intra_a = desc.smem_advance_A_intra
    intra_b = desc.smem_advance_B_intra
    subtile_a = desc.smem_subtile_A
    subtile_b = desc.smem_subtile_B
    sps_a = desc.sps_A
    sps_b = desc.sps_B
    num_k_steps = desc.num_k_steps
    k_count = num_k_steps if cutlass.const_expr(k_count is None) else k_count
    enable_input_d = accumulate
    SF_CYCLE = 4
    SF_REGISTERS_PER_BLOCK = 4
    if cutlass.const_expr(elect_once):
        elected = nvvm.elect_sync()
    for kk in cutlass.range_constexpr(k_count):
        k = k_start + kk
        ki_a = k % sps_a
        s_a = k // sps_a
        ki_b = k % sps_b
        s_b = k // sps_b
        inc_a = (intra_a * ki_a + subtile_a * s_a) >> 4
        inc_b = (intra_b * ki_b + subtile_b * s_b) >> 4
        da = desc_a_base + inc_a
        db = desc_b_base + inc_b
        if cutlass.const_expr(elect_once):
            issue_mma = elected
        else:
            issue_mma = nvvm.elect_sync()
        if issue_mma:
            if cutlass.const_expr(desc.is_block_scale):
                sf_id = (k * desc.sf_blocks_per_step) % SF_CYCLE
                sf_group = (k * desc.sf_blocks_per_step) // SF_CYCLE
                step_idesc = desc.idesc.set_sf_ids(a_sf_id=sf_id, b_sf_id=sf_id)
                sf_off = sf_group * SF_REGISTERS_PER_BLOCK
                sf_a = tmem_sf_a if sf_off == 0 else tmem_sf_a.subview(sf_off)
                sf_b = tmem_sf_b if sf_off == 0 else tmem_sf_b.subview(sf_off)
                nvvm.tcgen05_mma_block_scale(
                    desc.kind,
                    cta_group_kind,
                    tmem_c,
                    da,
                    db,
                    step_idesc,
                    enable_input_d,
                    scale_a=sf_a,
                    scale_b=sf_b,
                    scale_vec_size=desc.scale_vec_size,
                )
            else:
                nvvm.tcgen05_mma(desc.kind, cta_group_kind, tmem_c, da, db, desc.idesc, enable_input_d)
        enable_input_d = True


@cute.jit
def mma_ts(desc, tmem_a_base, desc_b_base, tmem_c, tmem_sf_a=None, tmem_sf_b=None, accumulate: bool = False):
    if cutlass.const_expr(desc.cta_group == 1):
        cta_group_kind = nvvm.CTAGroup.CTA_1
    else:
        cta_group_kind = nvvm.CTAGroup.CTA_2
    num_k_steps = desc.num_k_steps
    intra_b = desc.smem_advance_B_intra
    subtile_b = desc.smem_subtile_B
    sps_b = desc.sps_B
    enable_input_d = accumulate
    SF_CYCLE = 4
    SF_REGISTERS_PER_BLOCK = 4
    for k in cutlass.range_constexpr(num_k_steps):
        dp = tmem_a_base.subview(k * desc.tmem_advance_A)
        ki_b = k % sps_b
        s_b = k // sps_b
        inc_b = (intra_b * ki_b + subtile_b * s_b) >> 4
        db = desc_b_base + inc_b
        if nvvm.elect_sync():
            if cutlass.const_expr(desc.is_block_scale):
                sf_id = (k * desc.sf_blocks_per_step) % SF_CYCLE
                sf_group = (k * desc.sf_blocks_per_step) // SF_CYCLE
                step_idesc = desc.idesc.set_sf_ids(a_sf_id=sf_id, b_sf_id=sf_id)
                sf_off = sf_group * SF_REGISTERS_PER_BLOCK
                sf_a = tmem_sf_a if sf_off == 0 else tmem_sf_a.subview(sf_off)
                sf_b = tmem_sf_b if sf_off == 0 else tmem_sf_b.subview(sf_off)
                nvvm.tcgen05_mma_block_scale(
                    desc.kind,
                    cta_group_kind,
                    tmem_c,
                    dp,
                    db,
                    step_idesc,
                    enable_input_d,
                    scale_a=sf_a,
                    scale_b=sf_b,
                    scale_vec_size=desc.scale_vec_size,
                )
            else:
                nvvm.tcgen05_mma(desc.kind, cta_group_kind, tmem_c, dp, db, desc.idesc, enable_input_d)
        enable_input_d = True


@cute.jit
def mma_ts_step(
    desc,
    tmem_a_base,
    desc_b_base,
    tmem_c,
    k_idx: int,
    accumulate,
    tmem_sf_a=None,
    tmem_sf_b=None,
    issue_mma=None,
):
    if cutlass.const_expr(desc.cta_group == 1):
        cta_group_kind = nvvm.CTAGroup.CTA_1
    else:
        cta_group_kind = nvvm.CTAGroup.CTA_2
    dp = tmem_a_base.subview(k_idx * desc.tmem_advance_A)
    increment = (desc.smem_advance_B_intra * k_idx) >> 4
    db = desc_b_base + increment
    SF_CYCLE = 4
    SF_REGISTERS_PER_BLOCK = 4
    if cutlass.const_expr(issue_mma is None):
        issue_mma = nvvm.elect_sync()
    if issue_mma:
        if cutlass.const_expr(desc.is_block_scale):
            sf_id = (k_idx * desc.sf_blocks_per_step) % SF_CYCLE
            sf_group = (k_idx * desc.sf_blocks_per_step) // SF_CYCLE
            step_idesc = desc.idesc.set_sf_ids(a_sf_id=sf_id, b_sf_id=sf_id)
            sf_off = sf_group * SF_REGISTERS_PER_BLOCK
            sf_a = tmem_sf_a if sf_off == 0 else tmem_sf_a.subview(sf_off)
            sf_b = tmem_sf_b if sf_off == 0 else tmem_sf_b.subview(sf_off)
            nvvm.tcgen05_mma_block_scale(
                desc.kind,
                cta_group_kind,
                tmem_c,
                dp,
                db,
                step_idesc,
                accumulate,
                scale_a=sf_a,
                scale_b=sf_b,
                scale_vec_size=desc.scale_vec_size,
            )
        else:
            nvvm.tcgen05_mma(desc.kind, cta_group_kind, tmem_c, dp, db, desc.idesc, accumulate)


@cute.jit
def load_b_smem(
    sB_base,
    *,
    k_step: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    sB_elems_per_row,
    b_trans: cutlass.Constexpr[bool],
    lane,
    swizzle: cutlass.Constexpr[bool] = False,
    elem_bytes: cutlass.Constexpr[int] = 2,
):
    N_FRAGS = N // 8
    if cutlass.const_expr(b_trans):
        b_row = lane % 16
        layout_flag = nvvm.MMALayout.COL
    else:
        b_row = lane % 8
        b_col_subchunk = (lane % 16) // 8
        layout_flag = nvvm.MMALayout.ROW

    b_frag = [None] * (N_FRAGS * 2)
    for n_frag in cutlass.range_constexpr(N_FRAGS):
        if cutlass.const_expr(b_trans):
            row = k_step * 16 + b_row
            col = n_frag * 8
        else:
            row = n_frag * 8 + b_row
            col = k_step * 16 + b_col_subchunk * 8
        if cutlass.const_expr(swizzle):
            smem_col = swizzle_xor_128b(row, col, elem_bytes=elem_bytes)
        else:
            smem_col = col
        b_ptr = sB_base.subview(row * sB_elems_per_row + smem_col)
        b_v = nvvm.ldmatrix(b_ptr.data_ptr(), 2, layout_flag)
        b_frag[n_frag * 2 + 0] = b_v[0]
        b_frag[n_frag * 2 + 1] = b_v[1]
    return b_frag


@cute.jit
def load_b_smem_x4(
    sB_base,
    *,
    k_step: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    sB_elems_per_row,
    b_trans: cutlass.Constexpr[bool],
    lane,
    swizzle: cutlass.Constexpr[bool] = False,
    elem_bytes: cutlass.Constexpr[int] = 2,
    col_base=0,
    row_stride_log2: cutlass.Constexpr = None,
):
    N_FRAGS = N // 8
    if cutlass.const_expr(N_FRAGS % 2 != 0):
        raise ValueError(f"load_b_smem_x4: N//8 must be even (got N={N}, N//8={N_FRAGS})")
    PAIRS = N_FRAGS // 2

    if cutlass.const_expr(b_trans):
        b_row = lane % 16
        n_offset = lane // 16
        layout_flag = nvvm.MMALayout.COL
    else:
        b_row = lane % 8
        b_col_subchunk = (lane // 8) % 2
        n_offset = lane // 16
        layout_flag = nvvm.MMALayout.ROW

    b_frag = [None] * (N_FRAGS * 2)
    for pair in cutlass.range_constexpr(PAIRS):
        n_frag = pair * 2
        if cutlass.const_expr(b_trans):
            row = k_step * 16 + b_row
            col = (n_frag + n_offset) * 8
        else:
            row = (n_frag + n_offset) * 8 + b_row
            col = k_step * 16 + b_col_subchunk * 8
        col = col + col_base
        if cutlass.const_expr(swizzle and row_stride_log2 is not None):
            lin = row * (1 << row_stride_log2) + col
            b_ptr = sB_base.subview(swizzle_lin_128b(lin, row_stride_log2=row_stride_log2, elem_bytes=elem_bytes))
        elif cutlass.const_expr(swizzle):
            smem_col = swizzle_xor_128b(row, col, elem_bytes=elem_bytes)
            b_ptr = sB_base.subview(row * sB_elems_per_row + smem_col)
        else:
            b_ptr = sB_base.subview(row * sB_elems_per_row + col)
        b_v = nvvm.ldmatrix(b_ptr.data_ptr(), 4, layout_flag)
        b_frag[n_frag * 2 + 0] = b_v[0]
        b_frag[n_frag * 2 + 1] = b_v[1]
        b_frag[(n_frag + 1) * 2 + 0] = b_v[2]
        b_frag[(n_frag + 1) * 2 + 1] = b_v[3]
    return b_frag


@cute.jit
def b_smem_x4_prepare(
    sB_base,
    *,
    N: cutlass.Constexpr[int],
    sB_elems_per_row,
    b_trans: cutlass.Constexpr[bool],
    lane,
    swizzle: cutlass.Constexpr[bool] = False,
    elem_bytes: cutlass.Constexpr[int] = 2,
    col_base=0,
):
    N_FRAGS = N // 8
    if cutlass.const_expr(N_FRAGS % 2 != 0):
        raise ValueError(f"b_smem_x4_prepare: N//8 must be even (got N={N}, N//8={N_FRAGS})")
    PAIRS = N_FRAGS // 2

    if cutlass.const_expr(b_trans):
        b_row = lane % 16
        n_offset = lane // 16
    else:
        b_row = lane % 8
        b_col_subchunk = (lane // 8) % 2
        n_offset = lane // 16

    prep = [None] * PAIRS
    for pair in cutlass.range_constexpr(PAIRS):
        n_frag = pair * 2
        if cutlass.const_expr(b_trans):
            col = (n_frag + n_offset) * 8 + col_base
            if cutlass.const_expr(swizzle):
                swz_col = swizzle_xor_128b(b_row, col, elem_bytes=elem_bytes)
            else:
                swz_col = col
            base_ptr = sB_base.subview(b_row * sB_elems_per_row + swz_col)
            stride16 = 16 * sB_elems_per_row
            prep[pair] = (base_ptr, stride16, cutlass.Int32(0))
        else:
            row = (n_frag + n_offset) * 8 + b_row
            base_ptr = sB_base.subview(row * sB_elems_per_row)
            prep[pair] = (base_ptr, row & 7, b_col_subchunk)
    return prep


@cute.jit
def load_b_smem_x4_step(
    prep,
    *,
    k_step: cutlass.Constexpr[int],
    b_trans: cutlass.Constexpr[bool],
    swizzle: cutlass.Constexpr[bool] = False,
    elem_bytes: cutlass.Constexpr[int] = 2,
    col_base=0,
):
    PAIRS = len(prep)
    N_FRAGS = PAIRS * 2
    chunk_elems = cutlass.const_expr(16 // elem_bytes)
    layout_flag = nvvm.MMALayout.COL if cutlass.const_expr(b_trans) else nvvm.MMALayout.ROW

    b_frag = [None] * (N_FRAGS * 2)
    for pair in cutlass.range_constexpr(PAIRS):
        n_frag = pair * 2
        base_ptr, key2, key3 = prep[pair]
        if cutlass.const_expr(b_trans):
            b_ptr = base_ptr.subview(k_step * key2)
        else:
            r7 = key2
            sub = key3
            col = k_step * 16 + sub * 8 + col_base
            if cutlass.const_expr(swizzle):
                swz_chunk = (col // chunk_elems) ^ r7
                b_ptr = base_ptr.subview(swz_chunk * chunk_elems + (col % chunk_elems))
            else:
                b_ptr = base_ptr.subview(col)
        b_v = nvvm.ldmatrix(b_ptr.data_ptr(), 4, layout_flag)
        b_frag[n_frag * 2 + 0] = b_v[0]
        b_frag[n_frag * 2 + 1] = b_v[1]
        b_frag[(n_frag + 1) * 2 + 0] = b_v[2]
        b_frag[(n_frag + 1) * 2 + 1] = b_v[3]
    return b_frag


@cute.jit
def mma_step(
    acc,
    a_frag,
    b_frag,
    *,
    k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]] = cutlass.Float16,
):
    if cutlass.const_expr(M % 16 != 0):
        raise ValueError(f"mma_step: M must be a multiple of 16, got M={M}")
    if cutlass.const_expr(ab_dtype != cutlass.Float16 and ab_dtype != cutlass.BFloat16):
        raise TypeError(f"mma_step: ab_dtype must be Float16 or BFloat16, got {ab_dtype}")
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    a_stride = len(a_frag) // M_BLOCKS

    ab_tag = "f16" if cutlass.const_expr(ab_dtype == cutlass.Float16) else "bf16"
    mma_ptx = f"mma.sync.aligned.m16n8k16.row.col.f32.{ab_tag}.{ab_tag}.f32" " {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"

    for m_block in cutlass.range_constexpr(M_BLOCKS):
        a_off = m_block * a_stride + k_step * 4
        a0 = a_frag[a_off + 0]
        a1 = a_frag[a_off + 1]
        a2 = a_frag[a_off + 2]
        a3 = a_frag[a_off + 3]
        acc_base = m_block * N_FRAGS * 4
        for n_frag in cutlass.range_constexpr(N_FRAGS):
            b0 = b_frag[n_frag * 2 + 0]
            b1 = b_frag[n_frag * 2 + 1]
            s_off = acc_base + n_frag * 4
            c0, c1, c2, c3 = inline_ptx(
                mma_ptx,
                write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
                read_only_args=[
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    acc[s_off + 0],
                    acc[s_off + 1],
                    acc[s_off + 2],
                    acc[s_off + 3],
                ],
            )
            acc[s_off + 0] = c0
            acc[s_off + 1] = c1
            acc[s_off + 2] = c2
            acc[s_off + 3] = c3


@cute.jit
def mma_step_k8(
    acc,
    a_frag,
    b_frag,
    *,
    k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]] = cutlass.Float16,
):
    if cutlass.const_expr(M % 16 != 0):
        raise ValueError(f"mma_step_k8: M must be a multiple of 16, got M={M}")
    if cutlass.const_expr(ab_dtype != cutlass.Float16 and ab_dtype != cutlass.BFloat16):
        raise TypeError(f"mma_step_k8: ab_dtype must be Float16 or BFloat16, got {ab_dtype}")
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    a_stride = len(a_frag) // M_BLOCKS

    ab_tag = "f16" if cutlass.const_expr(ab_dtype == cutlass.Float16) else "bf16"
    mma_ptx = f"mma.sync.aligned.m16n8k8.row.col.f32.{ab_tag}.{ab_tag}.f32" " {$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};"

    for m_block in cutlass.range_constexpr(M_BLOCKS):
        a_off = m_block * a_stride + k_step * 2
        a0 = a_frag[a_off + 0]
        a1 = a_frag[a_off + 1]
        acc_base = m_block * N_FRAGS * 4
        for n_frag in cutlass.range_constexpr(N_FRAGS):
            b0 = b_frag[n_frag]
            s_off = acc_base + n_frag * 4
            c0, c1, c2, c3 = inline_ptx(
                mma_ptx,
                write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
                read_only_args=[
                    a0,
                    a1,
                    b0,
                    acc[s_off + 0],
                    acc[s_off + 1],
                    acc[s_off + 2],
                    acc[s_off + 3],
                ],
            )
            acc[s_off + 0] = c0
            acc[s_off + 1] = c1
            acc[s_off + 2] = c2
            acc[s_off + 3] = c3


@cute.jit
def mma(
    acc,
    a_frag,
    sB_base,
    *,
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    sB_elems_per_row,
    b_trans: cutlass.Constexpr[bool],
    lane,
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]] = cutlass.Float16,
):
    K_STEPS = K // 16
    for k in cutlass.range_constexpr(K_STEPS):
        b_frag = load_b_smem(sB_base, k_step=k, N=N, sB_elems_per_row=sB_elems_per_row, b_trans=b_trans, lane=lane)
        mma_step(acc, a_frag, b_frag, k_step=k, M=M, N=N, ab_dtype=ab_dtype)
