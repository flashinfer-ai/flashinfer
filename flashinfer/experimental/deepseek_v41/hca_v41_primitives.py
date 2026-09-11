# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit SM100 hardware primitives for the V4.1 HCA derivation.

The layouts and QK/PV scheduling come from Mengyu Guo's hca_fp8.py
(FlashInfer #3943/#4368), adapted by the V4.1 HCA work. These helpers preserve
that scheduling and fragment layout while expressing MMA, TMEM access, and
raw-cache memory operations through cutlass.experimental.primitives.
All SMEM byte offsets refer to the existing SW128 BF16 tiles.
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
def load_cache_vector(ptr):
    """Read one aligned 16-byte cache chunk through the public load primitive."""
    addr = cutlass.Array(ptr.toint(), dtype=cutlass.Int32, shape=(4,), addrspace=1)
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
