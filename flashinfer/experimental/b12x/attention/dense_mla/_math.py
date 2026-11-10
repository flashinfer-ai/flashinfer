"""K3-specific tensor-core math for the standalone dense MLA kernel.

The score fragment is deliberately transposed: a warp places sixteen cache
records in the hardware-M dimension and the eight K3 query heads in hardware-N.
Four warps therefore cover one 64-record window without padding away half of an
MMA tile.  The output fragment is converted back to ordinary head-row ownership
before the PV epilogue.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics import (
    atomic_max_shared_f32,
    byte_perm,
    cp_async4_shared_global,
    cvt_f32_to_e4m3,
    fabs_f32,
    fmax_f32,
    fmin_f32,
    fp8_e4m3_to_f32,
    get_ptr_as_int64,
    ld_shared_f32,
    ld_shared_u32,
    ldmatrix_m8n8x2_b16,
    ldmatrix_m8n8x4_b16,
    mma_m16n8k16_f32_bf16,
    mma_m16n8k32_f32_e4m3,
    st_shared_bf16_from_f32,
    st_shared_f32,
    st_shared_u8,
)

from ._layout import (
    CANDIDATES_PER_CHUNK,
    HEADS_PER_TILE,
    MATH_WARPS_PER_QUERY,
)

_FP8_MAX = 448.0
_MASK = -1.0e30


@dsl_user_op
def exp2_approx(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def log2_approx(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _load_shared_u16(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u16 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def stage_absorbed_query(
    q_bytes: cute.Tensor,
    q_smem_addr: Int32,
    q_start: Int32,
    head_base: Int32,
    total_q: Int32,
    math_tid: Int32,
    q_stride_row_bytes: Int64,
    q_stride_head_bytes: Int64,
    *,
    num_heads: cutlass.Constexpr,
    query_tile: cutlass.Constexpr,
    record_bytes: cutlass.Constexpr,
    record_stride_bytes: cutlass.Constexpr,
    math_threads: cutlass.Constexpr,
):
    """Asynchronously stage every absorbed query record exactly once."""
    segments_per_record = record_bytes // 16
    total_segments = query_tile * HEADS_PER_TILE * segments_per_record
    segment = math_tid
    while segment < Int32(total_segments):
        record = segment // Int32(segments_per_record)
        byte_in_record = (segment - record * Int32(segments_per_record)) * Int32(16)
        query_slot = record // Int32(HEADS_PER_TILE)
        local_head = record - query_slot * Int32(HEADS_PER_TILE)
        query_row = q_start + query_slot
        safe_row = query_row
        if query_row >= total_q:
            safe_row = Int32(0)
        safe_head = head_base + local_head
        if safe_head >= Int32(num_heads):
            safe_head = head_base
        source_offset = (
            safe_row.to(Int64) * q_stride_row_bytes
            + safe_head.to(Int64) * q_stride_head_bytes
            + byte_in_record.to(Int64)
        )
        destination = q_smem_addr + record * Int32(record_stride_bytes) + byte_in_record
        cp_async4_shared_global(
            destination,
            get_ptr_as_int64(q_bytes, source_offset),
        )
        segment += Int32(math_threads)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier(barrier_id=1, number_of_threads=math_threads)


@cute.jit
def qk_fp8(
    qk,
    q_smem_addr: Int32,
    kv_smem_addr: Int32,
    local_warp: Int32,
    lane: Int32,
    *,
    record_stride_bytes: cutlass.Constexpr,
    qk_dim: cutlass.Constexpr,
):
    """Raw E4M3 K-by-Q MMA for one 16-candidate warp tile."""
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(16)
    b_row = lane & Int32(7)
    b_col = ((lane >> Int32(3)) & Int32(1)) * Int32(16)
    first_candidate = local_warp * Int32(16)

    for ks in cutlass.range_constexpr(qk_dim // 32):
        ko = Int32(ks * 32)
        a_addr = (
            kv_smem_addr
            + (first_candidate + a_row) * Int32(record_stride_bytes)
            + ko
            + a_col
        )
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
        b_addr = q_smem_addr + b_row * Int32(record_stride_bytes) + ko + b_col
        b0, b1 = ldmatrix_m8n8x2_b16(b_addr)
        qk[0], qk[1], qk[2], qk[3] = mma_m16n8k32_f32_e4m3(
            qk[0],
            qk[1],
            qk[2],
            qk[3],
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
        )
    return qk


@cute.jit
def qk_bf16(
    qk,
    q_smem_addr: Int32,
    kv_smem_addr: Int32,
    local_warp: Int32,
    lane: Int32,
    *,
    record_stride_bytes: cutlass.Constexpr,
    qk_dim: cutlass.Constexpr,
):
    """Raw BF16 K-by-Q MMA for one 16-candidate warp tile."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    first_candidate = local_warp * Int32(16)

    for ks in cutlass.range_constexpr(qk_dim // 16):
        ko = Int32(ks * 16)
        a_addr = (
            kv_smem_addr
            + (first_candidate + a_row) * Int32(record_stride_bytes)
            + (ko + a_col) * Int32(2)
        )
        a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
        q_row_addr = q_smem_addr + gid * Int32(record_stride_bytes) + ko * Int32(2)
        b0 = ld_shared_u32(q_row_addr + tid * Int32(4))
        b1 = ld_shared_u32(q_row_addr + (tid * Int32(2) + Int32(8)) * Int32(2))
        qk[0], qk[1], qk[2], qk[3] = mma_m16n8k16_f32_bf16(
            qk[0],
            qk[1],
            qk[2],
            qk[3],
            a0,
            a1,
            a2,
            a3,
            b0,
            b1,
        )
    return qk


@cute.jit
def mask_and_scale(
    qk,
    chunk_begin: Int32,
    visible_begin: Int32,
    visible_end: Int32,
    scale_log2: Float32,
    local_warp: Int32,
    lane: Int32,
):
    """Apply K3 causal bounds and convert scores to the base-2 domain."""
    gid = lane >> Int32(2)
    candidate0 = chunk_begin + local_warp * Int32(16) + gid
    candidate1 = candidate0 + Int32(8)
    if candidate0 < visible_begin or candidate0 >= visible_end:
        qk[0] = Float32(_MASK)
        qk[1] = Float32(_MASK)
    if candidate1 < visible_begin or candidate1 >= visible_end:
        qk[2] = Float32(_MASK)
        qk[3] = Float32(_MASK)
    for idx in cutlass.range_constexpr(4):
        qk[idx] = qk[idx] * scale_log2
    return qk


@cute.jit
def online_softmax(
    qk,
    p,
    acc,
    global_max,
    global_sum,
    reduce_max_addr: Int32,
    reduce_sum_addr: Int32,
    local_warp: Int32,
    lane: Int32,
    local_tid: Int32,
    *,
    barrier_id: cutlass.Constexpr,
    value_dim: cutlass.Constexpr,
):
    """Update a base-2 online softmax and rescale persistent PV registers."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    head0 = tid * Int32(2)
    head1 = head0 + Int32(1)

    local_max0 = fmax_f32(qk[0], qk[2])
    local_max1 = fmax_f32(qk[1], qk[3])
    for offset in (4, 8, 16):
        local_max0 = fmax_f32(
            local_max0,
            cute.arch.shuffle_sync_bfly(local_max0, offset=offset),
        )
        local_max1 = fmax_f32(
            local_max1,
            cute.arch.shuffle_sync_bfly(local_max1, offset=offset),
        )

    p[0] = exp2_approx(qk[0] - local_max0)
    p[1] = exp2_approx(qk[1] - local_max1)
    p[2] = exp2_approx(qk[2] - local_max0)
    p[3] = exp2_approx(qk[3] - local_max1)
    local_sum0 = p[0] + p[2]
    local_sum1 = p[1] + p[3]
    for offset in (4, 8, 16):
        local_sum0 = local_sum0 + cute.arch.shuffle_sync_bfly(local_sum0, offset=offset)
        local_sum1 = local_sum1 + cute.arch.shuffle_sync_bfly(local_sum1, offset=offset)

    if gid == Int32(0):
        st_shared_f32(
            reduce_max_addr + (local_warp * Int32(HEADS_PER_TILE) + head0) * Int32(4),
            local_max0,
        )
        st_shared_f32(
            reduce_max_addr + (local_warp * Int32(HEADS_PER_TILE) + head1) * Int32(4),
            local_max1,
        )
        st_shared_f32(
            reduce_sum_addr + (local_warp * Int32(HEADS_PER_TILE) + head0) * Int32(4),
            local_sum0,
        )
        st_shared_f32(
            reduce_sum_addr + (local_warp * Int32(HEADS_PER_TILE) + head1) * Int32(4),
            local_sum1,
        )
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

    if local_tid < Int32(HEADS_PER_TILE):
        head = local_tid
        block_max = Float32(_MASK)
        for warp in cutlass.range_constexpr(MATH_WARPS_PER_QUERY):
            block_max = fmax_f32(
                block_max,
                ld_shared_f32(
                    reduce_max_addr + (Int32(warp * HEADS_PER_TILE) + head) * Int32(4)
                ),
            )
        block_sum = Float32(0.0)
        for warp in cutlass.range_constexpr(MATH_WARPS_PER_QUERY):
            warp_max = ld_shared_f32(
                reduce_max_addr + (Int32(warp * HEADS_PER_TILE) + head) * Int32(4)
            )
            warp_sum = ld_shared_f32(
                reduce_sum_addr + (Int32(warp * HEADS_PER_TILE) + head) * Int32(4)
            )
            block_sum = block_sum + warp_sum * exp2_approx(warp_max - block_max)
        st_shared_f32(reduce_max_addr + head * Int32(4), block_max)
        st_shared_f32(reduce_sum_addr + head * Int32(4), block_sum)
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

    block_max0 = ld_shared_f32(reduce_max_addr + head0 * Int32(4))
    block_max1 = ld_shared_f32(reduce_max_addr + head1 * Int32(4))
    block_sum0 = ld_shared_f32(reduce_sum_addr + head0 * Int32(4))
    block_sum1 = ld_shared_f32(reduce_sum_addr + head1 * Int32(4))
    new_max0 = fmax_f32(global_max[0], block_max0)
    new_max1 = fmax_f32(global_max[1], block_max1)
    alpha0 = Float32(0.0)
    alpha1 = Float32(0.0)
    if global_sum[0] > Float32(0.0):
        alpha0 = exp2_approx(global_max[0] - new_max0)
    if global_sum[1] > Float32(0.0):
        alpha1 = exp2_approx(global_max[1] - new_max1)
    block_scale0 = exp2_approx(block_max0 - new_max0)
    block_scale1 = exp2_approx(block_max1 - new_max1)
    warp_scale0 = exp2_approx(local_max0 - new_max0)
    warp_scale1 = exp2_approx(local_max1 - new_max1)

    pair_lane = gid >> Int32(1)
    row_alpha0 = cute.arch.shuffle_sync(alpha0, pair_lane)
    row_alpha1 = cute.arch.shuffle_sync(alpha1, pair_lane)
    row_alpha = row_alpha0
    if (gid & Int32(1)) != Int32(0):
        row_alpha = row_alpha1
    for tile in cutlass.range_constexpr(value_dim // 32):
        acc[tile][0] = acc[tile][0] * row_alpha
        acc[tile][1] = acc[tile][1] * row_alpha

    global_sum[0] = global_sum[0] * alpha0 + block_sum0 * block_scale0
    global_sum[1] = global_sum[1] * alpha1 + block_sum1 * block_scale1
    global_max[0] = new_max0
    global_max[1] = new_max1
    return p, warp_scale0, warp_scale1


@cute.jit
def _quantized_probability(value: Float32):
    bounded = fmax_f32(Float32(-_FP8_MAX), fmin_f32(Float32(_FP8_MAX), value))
    high = cvt_f32_to_e4m3(bounded) & Uint32(0xFF)
    residual = bounded - fp8_e4m3_to_f32(high)
    residual = fmax_f32(
        Float32(-_FP8_MAX),
        fmin_f32(Float32(_FP8_MAX), residual),
    )
    low = cvt_f32_to_e4m3(residual) & Uint32(0xFF)
    return high, low


@cute.jit
def _direct_b_fp8(
    kv_smem_addr: Int32,
    entry_base: Int32,
    dim: Int32,
    lane: Int32,
    *,
    record_stride_bytes: cutlass.Constexpr,
):
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    dimension = dim + gid
    dimension_base = dimension & ~Int32(3)
    dimension_select = dimension & Int32(3)
    selector = ((Int32(4) + dimension_select) << Int32(4)) | dimension_select

    r0 = ld_shared_u32(
        kv_smem_addr
        + (entry_base + tid * Int32(4)) * Int32(record_stride_bytes)
        + dimension_base
    )
    r1 = ld_shared_u32(
        kv_smem_addr
        + (entry_base + tid * Int32(4) + Int32(1)) * Int32(record_stride_bytes)
        + dimension_base
    )
    r2 = ld_shared_u32(
        kv_smem_addr
        + (entry_base + tid * Int32(4) + Int32(2)) * Int32(record_stride_bytes)
        + dimension_base
    )
    r3 = ld_shared_u32(
        kv_smem_addr
        + (entry_base + tid * Int32(4) + Int32(3)) * Int32(record_stride_bytes)
        + dimension_base
    )
    t01 = byte_perm(r0, r1, selector)
    t23 = byte_perm(r2, r3, selector)
    b0 = byte_perm(t01, t23, Int32(0x5410))

    r0 = ld_shared_u32(
        kv_smem_addr
        + (entry_base + Int32(16) + tid * Int32(4)) * Int32(record_stride_bytes)
        + dimension_base
    )
    r1 = ld_shared_u32(
        kv_smem_addr
        + (entry_base + Int32(16) + tid * Int32(4) + Int32(1))
        * Int32(record_stride_bytes)
        + dimension_base
    )
    r2 = ld_shared_u32(
        kv_smem_addr
        + (entry_base + Int32(16) + tid * Int32(4) + Int32(2))
        * Int32(record_stride_bytes)
        + dimension_base
    )
    r3 = ld_shared_u32(
        kv_smem_addr
        + (entry_base + Int32(16) + tid * Int32(4) + Int32(3))
        * Int32(record_stride_bytes)
        + dimension_base
    )
    t01 = byte_perm(r0, r1, selector)
    t23 = byte_perm(r2, r3, selector)
    b1 = byte_perm(t01, t23, Int32(0x5410))
    return b0, b1


@cute.jit
def pv_fp8(
    weights,
    acc,
    kv_smem_addr: Int32,
    weight_scale_addr: Int32,
    weight_addr: Int32,
    local_warp: Int32,
    lane: Int32,
    local_tid: Int32,
    *,
    record_stride_bytes: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    value_dim: cutlass.Constexpr,
):
    """HIGH/LOW E4M3 probability MMA against the raw K3 value prefix."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    head0 = tid * Int32(2)
    head1 = head0 + Int32(1)
    candidate0 = local_warp * Int32(16) + gid
    candidate1 = candidate0 + Int32(8)
    weight_stride = CANDIDATES_PER_CHUNK + 16

    if local_tid < Int32(16):
        st_shared_f32(
            weight_scale_addr + local_tid * Int32(4),
            Float32(0.0),
        )
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

    atomic_max_shared_f32(
        weight_scale_addr + head0 * Int32(4),
        fmax_f32(fabs_f32(weights[0]), fabs_f32(weights[2])),
    )
    atomic_max_shared_f32(
        weight_scale_addr + head1 * Int32(4),
        fmax_f32(fabs_f32(weights[1]), fabs_f32(weights[3])),
    )
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

    if local_tid < Int32(HEADS_PER_TILE):
        maximum = fmax_f32(
            ld_shared_f32(weight_scale_addr + local_tid * Int32(4)),
            Float32(1.0e-10),
        )
        scale = maximum * Float32(1.0 / _FP8_MAX)
        st_shared_f32(weight_scale_addr + local_tid * Int32(4), scale)
        st_shared_f32(
            weight_scale_addr + (local_tid + Int32(HEADS_PER_TILE)) * Int32(4),
            Float32(1.0) / scale,
        )
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

    inv0 = ld_shared_f32(weight_scale_addr + (head0 + Int32(HEADS_PER_TILE)) * Int32(4))
    inv1 = ld_shared_f32(weight_scale_addr + (head1 + Int32(HEADS_PER_TILE)) * Int32(4))
    h00, l00 = _quantized_probability(weights[0] * inv0)
    h01, l01 = _quantized_probability(weights[1] * inv1)
    h10, l10 = _quantized_probability(weights[2] * inv0)
    h11, l11 = _quantized_probability(weights[3] * inv1)
    st_shared_u8(
        weight_addr + head0 * Int32(weight_stride) + candidate0,
        h00.to(cutlass.Uint8),
    )
    st_shared_u8(
        weight_addr + head1 * Int32(weight_stride) + candidate0,
        h01.to(cutlass.Uint8),
    )
    st_shared_u8(
        weight_addr + head0 * Int32(weight_stride) + candidate1,
        h10.to(cutlass.Uint8),
    )
    st_shared_u8(
        weight_addr + head1 * Int32(weight_stride) + candidate1,
        h11.to(cutlass.Uint8),
    )
    st_shared_u8(
        weight_addr
        + (head0 + Int32(HEADS_PER_TILE)) * Int32(weight_stride)
        + candidate0,
        l00.to(cutlass.Uint8),
    )
    st_shared_u8(
        weight_addr
        + (head1 + Int32(HEADS_PER_TILE)) * Int32(weight_stride)
        + candidate0,
        l01.to(cutlass.Uint8),
    )
    st_shared_u8(
        weight_addr
        + (head0 + Int32(HEADS_PER_TILE)) * Int32(weight_stride)
        + candidate1,
        l10.to(cutlass.Uint8),
    )
    st_shared_u8(
        weight_addr
        + (head1 + Int32(HEADS_PER_TILE)) * Int32(weight_stride)
        + candidate1,
        l11.to(cutlass.Uint8),
    )
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(16)
    row_scale = ld_shared_f32(weight_scale_addr + gid * Int32(4))
    for value_chunk in cutlass.range_constexpr(value_dim // 64):
        for n_tile in cutlass.range_constexpr(2):
            dimension = Int32(value_chunk * 64) + (
                Int32(n_tile * MATH_WARPS_PER_QUERY) + local_warp
            ) * Int32(8)
            x0 = Float32(0.0)
            x1 = Float32(0.0)
            x2 = Float32(0.0)
            x3 = Float32(0.0)
            for k_step in cutlass.range_constexpr(2):
                ko = Int32(k_step * 32)
                a_addr = weight_addr + a_row * Int32(weight_stride) + ko + a_col
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
                b0, b1 = _direct_b_fp8(
                    kv_smem_addr,
                    ko,
                    dimension,
                    lane,
                    record_stride_bytes=record_stride_bytes,
                )
                x0, x1, x2, x3 = mma_m16n8k32_f32_e4m3(
                    x0,
                    x1,
                    x2,
                    x3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                )
            tile = value_chunk * 2 + n_tile
            acc[tile][0] = acc[tile][0] + (x0 + x2) * row_scale
            acc[tile][1] = acc[tile][1] + (x1 + x3) * row_scale
    return acc


@cute.jit
def pv_bf16(
    weights,
    acc,
    kv_smem_addr: Int32,
    weight_addr: Int32,
    local_warp: Int32,
    lane: Int32,
    *,
    record_stride_bytes: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    value_dim: cutlass.Constexpr,
):
    """BF16 probability/value MMA for the K3 value prefix."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    head0 = tid * Int32(2)
    head1 = head0 + Int32(1)
    candidate0 = local_warp * Int32(16) + gid
    candidate1 = candidate0 + Int32(8)
    weight_stride = CANDIDATES_PER_CHUNK

    for head, candidate, value in (
        (head0, candidate0, weights[0]),
        (head1, candidate0, weights[1]),
        (head0, candidate1, weights[2]),
        (head1, candidate1, weights[3]),
    ):
        st_shared_bf16_from_f32(
            weight_addr + (head * Int32(weight_stride) + candidate) * Int32(2),
            value,
        )
        # The upper rows are not part of K3 H8, but zeroing them prevents
        # uninitialized values from entering the otherwise-unused MMA fragment.
        st_shared_bf16_from_f32(
            weight_addr
            + ((head + Int32(HEADS_PER_TILE)) * Int32(weight_stride) + candidate)
            * Int32(2),
            Float32(0.0),
        )
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_col = (lane >> Int32(4)) * Int32(8)
    for value_chunk in cutlass.range_constexpr(value_dim // 64):
        for n_tile in cutlass.range_constexpr(2):
            column = (
                Int32(value_chunk * 64)
                + (Int32(n_tile * MATH_WARPS_PER_QUERY) + local_warp) * Int32(8)
                + gid
            )
            x0 = Float32(0.0)
            x1 = Float32(0.0)
            x2 = Float32(0.0)
            x3 = Float32(0.0)
            for k_step in cutlass.range_constexpr(4):
                k_base = Int32(k_step * 16)
                a_addr = weight_addr + (
                    a_row * Int32(weight_stride) + k_base + a_col
                ) * Int32(2)
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_addr)
                entry = k_base + tid * Int32(2)
                v0 = _load_shared_u16(
                    kv_smem_addr
                    + entry * Int32(record_stride_bytes)
                    + column * Int32(2)
                )
                v1 = _load_shared_u16(
                    kv_smem_addr
                    + (entry + Int32(1)) * Int32(record_stride_bytes)
                    + column * Int32(2)
                )
                v8 = _load_shared_u16(
                    kv_smem_addr
                    + (entry + Int32(8)) * Int32(record_stride_bytes)
                    + column * Int32(2)
                )
                v9 = _load_shared_u16(
                    kv_smem_addr
                    + (entry + Int32(9)) * Int32(record_stride_bytes)
                    + column * Int32(2)
                )
                b0 = v0 | (v1 << Uint32(16))
                b1 = v8 | (v9 << Uint32(16))
                x0, x1, x2, x3 = mma_m16n8k16_f32_bf16(
                    x0,
                    x1,
                    x2,
                    x3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                )
            tile = value_chunk * 2 + n_tile
            acc[tile][0] = acc[tile][0] + x0
            acc[tile][1] = acc[tile][1] + x1
    return acc


@cute.jit
def write_partial_or_final(
    acc,
    global_max,
    global_sum,
    output: cute.Tensor,
    output_lse: cute.Tensor,
    query_row: Int32,
    head_base: Int32,
    split: Int32,
    local_warp: Int32,
    lane: Int32,
    value_scale: Float32,
    query_valid: Int32,
    *,
    num_heads: cutlass.Constexpr,
    has_splits: cutlass.Constexpr,
    fp8: cutlass.Constexpr,
    ln2: cutlass.Constexpr,
    value_dim: cutlass.Constexpr,
):
    """Write a normalized split partial, or the final one-split result."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    pair_lane = gid >> Int32(1)
    row_max0 = cute.arch.shuffle_sync(global_max[0], pair_lane)
    row_max1 = cute.arch.shuffle_sync(global_max[1], pair_lane)
    row_sum0 = cute.arch.shuffle_sync(global_sum[0], pair_lane)
    row_sum1 = cute.arch.shuffle_sync(global_sum[1], pair_lane)
    row_max = row_max0
    row_sum = row_sum0
    if (gid & Int32(1)) != Int32(0):
        row_max = row_max1
        row_sum = row_sum1

    inverse = Float32(0.0)
    if row_sum > Float32(0.0):
        inverse = Float32(1.0) / row_sum
    scale = inverse
    if cutlass.const_expr(fp8):
        scale = scale * value_scale

    if query_valid != Int32(0) and head_base + gid < Int32(num_heads):
        for value_chunk in cutlass.range_constexpr(value_dim // 64):
            for n_tile in cutlass.range_constexpr(2):
                tile = value_chunk * 2 + n_tile
                dimension = (
                    Int32(value_chunk * 64)
                    + (Int32(n_tile * MATH_WARPS_PER_QUERY) + local_warp) * Int32(8)
                    + tid * Int32(2)
                )
                if cutlass.const_expr(has_splits):
                    output[
                        query_row,
                        head_base + gid,
                        split,
                        dimension,
                    ] = (acc[tile][0] * scale).to(output.element_type)
                    output[
                        query_row,
                        head_base + gid,
                        split,
                        dimension + Int32(1),
                    ] = (acc[tile][1] * scale).to(output.element_type)
                else:
                    output[
                        query_row,
                        head_base + gid,
                        dimension,
                    ] = (acc[tile][0] * scale).to(output.element_type)
                    output[
                        query_row,
                        head_base + gid,
                        dimension + Int32(1),
                    ] = (acc[tile][1] * scale).to(output.element_type)

        if local_warp == Int32(0) and tid == Int32(0):
            lse = Float32(-Float32.inf)
            if row_sum > Float32(0.0):
                lse = log2_approx(row_sum) + row_max
                if cutlass.const_expr(not has_splits):
                    lse = lse * Float32(ln2)
            if cutlass.const_expr(has_splits):
                output_lse[
                    query_row,
                    head_base + gid,
                    split,
                ] = lse
            else:
                output_lse[query_row, head_base + gid] = lse


__all__ = [
    "exp2_approx",
    "log2_approx",
    "mask_and_scale",
    "online_softmax",
    "pv_bf16",
    "pv_fp8",
    "qk_bf16",
    "qk_fp8",
    "stage_absorbed_query",
    "write_partial_or_final",
]
