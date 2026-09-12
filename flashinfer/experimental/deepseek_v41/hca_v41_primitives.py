# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit SM100 hardware primitives for the V4.1 HCA derivation.

The standard schedule derives from Mengyu Guo's hca_fp8.py (FlashInfer
#3943/#4368). The WS QK/PV layouts, partial-score exchange and shared/TMA
output path are informed by DeepSeek-AI/FlashMLA's SM100 head64 kernel (MIT).
Both schedules express hardware operations through public CUTLASS primitives.
WS MMA uses public inline_ptx because CUTLASS DSL 4.7's typed WS wrapper
forwards an unsupported zero_col_mask operand to its dialect operation.

Standard M64 TMEM uses interleaved datapaths. WS uses all 128 datapaths:
O occupies columns 0:256, Q 256:384, and two score stages 384:512. WS QK
computes two K256 partial dot products, joined before softmax; WS PV computes
both output halves. K/V share SW128 BF16 tiles, and WS P uses INTER layout.
"""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims


@cute.jit
def fp4x8_to_bf16_words(word: cutlass.Int32):
    """Native PTX 9.2 conversion; CUTLASS 4.7 lacks this typed wrapper."""
    return prims.inline_ptx(
        "{ .reg .b8 b0, b1, b2, b3; "
        "mov.b32 {b0, b1, b2, b3}, {$r0}; "
        "cvt.rn.bf16x2.e2m1x2 {$w0}, b0; "
        "cvt.rn.bf16x2.e2m1x2 {$w1}, b1; "
        "cvt.rn.bf16x2.e2m1x2 {$w2}, b2; "
        "cvt.rn.bf16x2.e2m1x2 {$w3}, b3; }",
        write_only_types=[cutlass.Int32] * 4,
        read_only_args=[word],
    )


@cute.jit
def fp8x2_to_bf16_word(word: cutlass.Int32):
    return prims.inline_ptx(
        "cvt.rn.bf16x2.e4m3x2 {$w0}, {$r0};",
        write_only_types=[cutlass.Int32],
        read_only_args=[word.to(cutlass.Uint16)],
    )


@cute.jit
def qk_mma(sq, sk, dst, kv_stage: cutlass.Int32):
    """M64 N64 K512, Q/K both SW128 K-major; overwrite the score tile."""
    # The primitive consumes a native addrspace-6 pointer. Rebuild it from
    # the packed 32-bit TMEM address rather than passing a CuTe pointer.
    dst = prims.make_tmem_ptr(cutlass.Int32(dst.toint()), cutlass.Float32)
    desc = prims.Tcgen05InstrDesc.build(
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        c_dtype=cutlass.Float32,
        m_dim=64,
        n_dim=64,
    )
    q_base = sq.iterator.toint()
    k_base = sk.iterator.toint() + kv_stage * (64 * 512 * 2)
    a_base = prims.Tcgen05SmemDesc.build(
        q_base,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    b_base = prims.Tcgen05SmemDesc.build(
        k_base,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    # SW128 holds 64 BF16 columns per swizzle atom. The K512 axis is
    # eight consecutive 64x64 tiles, each occupying 8192 bytes.
    if prims.elect_sync():
        for k in cutlass.range_constexpr(32):
            offset = (k // 4) * 8192 + (k % 4) * 32
            a = a_base.advance_start_address(offset)
            b = b_base.advance_start_address(offset)
            prims.tcgen05_mma("f16", "cta_1", dst, a, b, desc, k != 0)


@cute.jit
def pv_mma(
    sp,
    sv,
    dst,
    p_stage: cutlass.Int32,
    kv_stage: cutlass.Int32,
    n_tile: cutlass.Int32,
    accumulate: cutlass.Boolean,
):
    """M64 N128 K64; V reuses the K tile through its MN-major view."""
    # The primitive consumes a native addrspace-6 pointer. Rebuild it from
    # the packed 32-bit TMEM address rather than passing a CuTe pointer.
    dst = prims.make_tmem_ptr(cutlass.Int32(dst.toint()), cutlass.Float32)
    desc = prims.Tcgen05InstrDesc.build(
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        c_dtype=cutlass.Float32,
        m_dim=64,
        n_dim=128,
        b_major=1,
    )
    p_base = sp.iterator.toint() + p_stage * (64 * 64 * 2)
    v_base = sv.iterator.toint() + kv_stage * (64 * 512 * 2) + n_tile * (64 * 128 * 2)
    a_base = prims.Tcgen05SmemDesc.build(
        p_base,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    b_base = prims.Tcgen05SmemDesc.build(
        v_base,
        leading_byte_offset=8192,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    if prims.elect_sync():
        for k in cutlass.range_constexpr(4):
            a = a_base.advance_start_address(k * 32)
            b = b_base.advance_start_address(k * 2048)
            prims.tcgen05_mma(
                "f16", "cta_1", dst, a, b, desc, accumulate if k == 0 else True
            )


@cute.jit
def load_tmem_m64(src, fragment: cute.Tensor):
    """Load the interleaved M64 accumulator with the existing x32 grouping."""
    addr = cutlass.Int32(src.toint())
    chunks = []
    for block in cutlass.range_constexpr(cute.size(fragment) // 32):
        ptr = prims.make_tmem_ptr(addr + block * 64, cutlass.Float32)
        chunks.append(prims.tcgen05_ld("16x32bx2", ptr, num=32, offset=32))
    # All loads may be in flight together; wait before consuming any result.
    prims.tcgen05_wait("load")
    for block in cutlass.range_constexpr(cute.size(fragment) // 32):
        for i in cutlass.range_constexpr(32):
            fragment[block * 32 + i] = chunks[block][i]


@cute.jit
def store_tmem_m64(fragment: cute.Tensor, dst):
    """Write the same M64 fragment layout after output rescaling."""
    addr = cutlass.Int32(dst.toint())
    for block in cutlass.range_constexpr(cute.size(fragment) // 32):
        ptr = prims.make_tmem_ptr(addr + block * 64, cutlass.Float32)
        values = cutlass.Vector.from_elements(
            [fragment[block * 32 + i] for i in range(32)], cutlass.Float32
        )
        prims.tcgen05_st("16x32bx2", ptr, values, offset=32)


@cute.jit
def load_cache_vector(ptr, bypass_l1: cutlass.Constexpr = False):
    """Read one aligned cache chunk, optionally reserving L1 for scales."""
    addr = cutlass.Array(ptr.toint(), dtype=cutlass.Int32, shape=(4,), addrspace=1)
    if cutlass.const_expr(bypass_l1):
        values = prims.load_ext(
            addr, count=4, cache_modifier=prims.LoadCacheModifier.CG
        )
    else:
        values = prims.load_ext(addr, count=4)
    return cute.TensorSSA(values, (4,), cutlass.Int32)


@cute.jit
def load_cache_scale(ptr, dtype: cutlass.Constexpr):
    addr = cutlass.Array(ptr.toint(), dtype=dtype, shape=(1,), addrspace=1)
    return prims.load_ext(addr).to(cutlass.Int32)


@cute.jit
def store_bf16_fragment(fragment: cute.Tensor, dst):
    """Store eight BF16 values as four 32-bit words to aligned CTA SMEM."""
    bits = cute.make_tensor(
        cute.recast_ptr(fragment.iterator, dtype=cutlass.Int32), cute.make_layout(4)
    )
    values = cutlass.Vector.from_elements([bits[i] for i in range(4)], cutlass.Int32)
    addr = cutlass.Array(
        cutlass.Int32(dst.toint()), dtype=cutlass.Int32, shape=(4,), addrspace=3
    )
    prims.store_ext(values, addr)


@cute.jit
def qk_mma_ws(sk, kv_stage: cutlass.Int32):
    # Two K256 dot products: Q's even/odd 64-column chunks occupy the
    # two WS halves. K is viewed as 128 rows of 256 BF16 values.
    desc = prims.Tcgen05InstrDesc.build(
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        c_dtype=cutlass.Float32,
        m_dim=64,
        n_dim=128,
    )
    b_base = prims.Tcgen05SmemDesc.build(
        sk.iterator.toint() + kv_stage * 65536,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    if prims.elect_sync():
        for k in cutlass.range_constexpr(16):
            b = b_base.advance_start_address((k // 4) * 16384 + (k % 4) * 32)
            prims.inline_ptx(
                "{ .reg .pred p; .reg .b32 aa, dd, ii; mov.b32 aa, {$r1}; mov.b32 dd, {$r0}; mov.b32 ii, {$r3}; setp.ne.u32 p, {$r4}, 0; "
                "tcgen05.mma.ws.cta_group::1.kind::f16 [dd], [aa], {$r2}, ii, p, 0; }",
                read_only_args=[
                    cutlass.Int32(384) + kv_stage * 64,
                    cutlass.Int32(256 + k * 8),
                    b,
                    desc,
                    cutlass.Int32(k != 0),
                ],
            )


@cute.jit
def pv_mma_ws(
    sp,
    sv,
    dst,
    p_stage: cutlass.Int32,
    kv_stage: cutlass.Int32,
    n_tile: cutlass.Int32,
    accumulate: cutlass.Boolean,
):
    desc = prims.Tcgen05InstrDesc.build(
        a_dtype=cutlass.BFloat16,
        b_dtype=cutlass.BFloat16,
        c_dtype=cutlass.Float32,
        m_dim=64,
        n_dim=256,
        b_major=1,
    )
    a_base = prims.Tcgen05SmemDesc.build(
        sp.iterator.toint() + p_stage * 8192,
        leading_byte_offset=1024,
        stride_byte_offset=128,
        layout=prims.Tcgen05SmemSwizzle.NONE,
    )
    b_base = prims.Tcgen05SmemDesc.build(
        sv.iterator.toint() + kv_stage * 65536 + n_tile * 16384,
        leading_byte_offset=8192,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    if prims.elect_sync():
        for k in cutlass.range_constexpr(4):
            a = a_base.advance_start_address(k * 2048)
            b = b_base.advance_start_address(k * 2048)
            use_d = accumulate if k == 0 else cutlass.Boolean(True)
            prims.inline_ptx(
                "{ .reg .pred p; .reg .b32 dd, ii; mov.b32 dd, {$r0}; mov.b32 ii, {$r3}; setp.ne.u32 p, {$r4}, 0; "
                "tcgen05.mma.ws.cta_group::1.kind::f16 [dd], {$r1}, {$r2}, ii, p, 0; }",
                read_only_args=[(n_tile // 2) * 128, a, b, desc, cutlass.Int32(use_d)],
            )


@cute.jit
def load_tmem_ws(src, fragment):
    for block in cutlass.range_constexpr(2):
        ptr = prims.make_tmem_ptr(
            cutlass.Int32(src.toint()) + block * 32, cutlass.Float32
        )
        values = prims.tcgen05_ld("32x32b", ptr, num=32)
        prims.tcgen05_wait("load")
        for i in cutlass.range_constexpr(32):
            fragment[block * 32 + i] = values[i]


@cute.jit
def store_tmem_ws(fragment, dst):
    for block in cutlass.range_constexpr(2):
        ptr = prims.make_tmem_ptr(
            cutlass.Int32(dst.toint()) + block * 32, cutlass.Float32
        )
        values = cutlass.Vector.from_elements(
            [fragment[block * 32 + i] for i in range(32)], cutlass.Float32
        )
        prims.tcgen05_st("32x32b", ptr, values)


@cute.jit
def stage_q_tmem(sq, tmem):
    tid, _, _ = cute.arch.thread_idx()
    head = tid % 64
    half = (tid % 128) // 64
    words = cute.make_tensor(
        cute.recast_ptr(sq.iterator, dtype=cutlass.Int32), cute.make_layout(16384)
    )
    for block in cutlass.range_constexpr(4):
        values = cutlass.Vector.from_elements(
            [
                words[(block * 2 + half) * 2048 + head * 32 + (i ^ ((head % 8) * 4))]
                for i in range(32)
            ],
            cutlass.Int32,
        )
        ptr = prims.make_tmem_ptr(cutlass.Int32(256 + block * 32), cutlass.Int32)
        prims.tcgen05_st("32x32b", ptr, values)
    prims.tcgen05_wait("store")
    prims.tcgen05_fence("before_thread_sync")


@cute.jit
def load_ws_scores(fragment, exchange, stage, barrier_id: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 128
    half = lane // 64
    own = prims.make_tmem_ptr(
        cutlass.Int32(384) + stage * 64 + half * 32, cutlass.Float32
    )
    peer = prims.make_tmem_ptr(
        cutlass.Int32(384) + stage * 64 + (1 - half) * 32, cutlass.Float32
    )
    kept = prims.tcgen05_ld("32x32b", own, num=32)
    sent = prims.tcgen05_ld("32x32b", peer, num=32)
    prims.tcgen05_wait("load")
    base = cutlass.Int32(exchange.toint()) + stage * 16384
    for block in cutlass.range_constexpr(8):
        addr = cutlass.Array(
            cutlass.Int32(base + (block * 512 + lane * 4) * 4),
            dtype=cutlass.Float32,
            shape=(4,),
            addrspace=3,
        )
        values = cutlass.Vector.from_elements(
            [sent[block * 4 + i] for i in range(4)], cutlass.Float32
        )
        prims.store_ext(values, addr)
    prims.barrier_cta_sync(barrier_id, thread_count=128)
    for block in cutlass.range_constexpr(8):
        addr = cutlass.Array(
            cutlass.Int32(base + (block * 512 + (lane ^ 64) * 4) * 4),
            dtype=cutlass.Float32,
            shape=(4,),
            addrspace=3,
        )
        other = prims.load_ext(addr, count=4)
        for pair in cutlass.range_constexpr(2):
            summed = cute.arch.add_packed_f32x2(
                (kept[block * 4 + pair * 2], kept[block * 4 + pair * 2 + 1]),
                (other[pair * 2], other[pair * 2 + 1]),
            )
            fragment[block * 4 + pair * 2] = summed[0]
            fragment[block * 4 + pair * 2 + 1] = summed[1]


@cute.jit
def store_p_inter(fragment, sp, stage):
    tid, _, _ = cute.arch.thread_idx()
    head = tid % 64
    key_start = ((tid % 128) // 64) * 32
    for block in cutlass.range_constexpr(4):
        key = key_start + block * 8
        byte_offset = stage * 8192 + (key // 8) * 1024 + head * 16
        dst = cute.make_ptr(
            cutlass.BFloat16,
            sp.toint() + byte_offset,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        frag = cute.make_tensor(fragment.iterator + block * 8, cute.make_layout(8))
        store_bf16_fragment(frag, dst)


@cute.jit
def store_ws_output(
    fragment, exchange, output, accum, coord, heads, iter_n: cutlass.Constexpr
):
    # WS TMEM assigns each lane a head. Exchange to contiguous global vectors
    # after the final softmax metadata hand-off makes Q scratch available.
    dtype = fragment.element_type
    width = 128 // dtype.width
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 128
    base = cutlass.Int32(exchange.toint())
    bits = cute.make_tensor(
        cute.recast_ptr(fragment.iterator, dtype=cutlass.Int32),
        cute.make_layout(64 * dtype.width // 32),
    )
    for block in cutlass.range_constexpr(64 // width):
        values = cutlass.Vector.from_elements(
            [bits[block * 4 + i] for i in range(4)], cutlass.Int32
        )
        addr = cutlass.Array(
            base + (block * 128 + lane) * width * (dtype.width // 8),
            dtype=cutlass.Int32,
            shape=(4,),
            addrspace=3,
        )
        prims.store_ext(values, addr)
    prims.barrier_cta_sync(11, thread_count=128)
    for block in cutlass.range_constexpr(64 // width):
        linear = (block * 128 + lane) * width
        head = linear // 128
        col = linear % 128
        source_lane = head + (col // 64) * 64
        source_block = (col % 64) // width
        addr = cutlass.Array(
            base + (source_block * 128 + source_lane) * width * (dtype.width // 8),
            dtype=cutlass.Int32,
            shape=(4,),
            addrspace=3,
        )
        values = prims.load_ext(addr, count=4)
        global_col = (
            (col // 64) * 128 + (iter_n // 2) * 256 + (iter_n % 2) * 64 + col % 64
        )
        if cutlass.const_expr(accum is None):
            row = output[head, None, coord[1], coord[2]]
        else:
            row = accum[head, coord[3], None, coord[1], coord[2]]
        if head < heads:
            dst = cutlass.Array(
                (row.iterator + global_col).toint(),
                dtype=cutlass.Int32,
                shape=(4,),
                addrspace=1,
            )
            prims.store_ext(values, dst)
    prims.barrier_cta_sync(11, thread_count=128)


@cute.jit
def stage_ws_output_sw128(fragment, scratch, iter_n: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    head = tid % 64
    col = ((tid % 128) // 64) * 128 + (iter_n // 2) * 256 + (iter_n % 2) * 64
    for block in cutlass.range_constexpr(8):
        offset = (col // 64) * 4096 + head * 64 + ((block * 8) ^ ((head % 8) * 8))
        dst = cute.make_ptr(
            cutlass.BFloat16,
            scratch.toint() + offset * 2,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        frag = cute.make_tensor(fragment.iterator + block * 8, cute.make_layout(8))
        store_bf16_fragment(frag, dst)


@cute.jit
def issue_ws_output(desc, scratch, row, iter_n: cutlass.Constexpr):
    prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
    prims.barrier_cta_sync(11, thread_count=128)
    tid, _, _ = cute.arch.thread_idx()
    if tid < 64:
        if prims.elect_sync():
            col = (tid // 32) * 128 + (iter_n // 2) * 256 + (iter_n % 2) * 64
            src = cutlass.Array(
                cutlass.Int32(scratch.toint()) + (col // 64) * 8192,
                dtype=cutlass.Int32,
                shape=(2048,),
                addrspace=3,
            )
            prims.cp_async_bulk_tensor_global_shared_cta(
                desc.get_ptr(), src, [cutlass.Int32(col), cutlass.Int32(row)]
            )


@cute.jit
def flush_ws_output(desc, scratch, row):
    tid, _, _ = cute.arch.thread_idx()
    if tid < 64:
        if prims.elect_sync():
            prims.cp_async_bulk_commit_group()
            prims.cp_async_bulk_wait_group(0)
    prims.barrier_cta_sync(11, thread_count=128)
