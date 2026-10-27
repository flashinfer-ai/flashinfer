# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2M1 and signed INT4 decoding for the shared BF16 FMA bodies."""

import cutlass
from cutlass import cute
from .nvfp4.fma_fc2 import _fp4_quad


@cute.jit
def _software_fp4_quad(word):
    # E2M1 magnitudes have FP16 high bytes [00,38,3c,3e,40,42,44,46].
    # A byte permutation looks up all four magnitudes at once. Two further
    # permutations insert zero low bytes; independent sign bits preserve -0.
    h02, h13 = cute.arch.inline_ptx(
        """{
      .reg .b32 selectors, bytes, spread, sign02, sign13, half02, half13;
      and.b32 selectors, {$r0}, 0x00007777;
      prmt.b32 bytes, 0x3e3c3800, 0x46444240, selectors;
      prmt.b32 spread, {$r0}, 0, 0x4140;
      and.b32 sign02, spread, 0x00080008;
      shl.b32 sign02, sign02, 12;
      and.b32 sign13, spread, 0x00800080;
      shl.b32 sign13, sign13, 8;
      prmt.b32 half02, bytes, 0, 0x2404;
      prmt.b32 half13, bytes, 0, 0x3414;
      or.b32 {$w0}, half02, sign02;
      or.b32 {$w1}, half13, sign13;
    }""",
        write_only_types=[cutlass.Uint32, cutlass.Uint32],
        read_only_args=[word.to(cutlass.Uint32)],
    )
    a = cutlass.Vector.from_elements((h02,), cutlass.Uint32).bitcast(cutlass.Float16)
    b = cutlass.Vector.from_elements((h13,), cutlass.Uint32).bitcast(cutlass.Float16)
    return cutlass.Vector.from_elements((a[0], b[0], a[1], b[1]), cutlass.Float16)


@cute.jit
def scaled_quad(
    word,
    scale,
    native: cutlass.Constexpr,
    scale_block_size: cutlass.Constexpr = 16,
    integer_weights: cutlass.Constexpr = False,
):
    if cutlass.const_expr(integer_weights):
        values = _int4_quad(word)
    elif cutlass.const_expr(native):
        values = _fp4_quad(word)
    else:
        values = _software_fp4_quad(word)
    # Every finite E2M1 * E4M3 product fits FP16 exactly. Dot accumulation
    # remains FP32; Hopper substitutes bit decoding for the FP4 instruction.
    if cutlass.const_expr(scale_block_size == 32):
        # E8M0 spans the FP32 exponent range; an FP16 conversion can overflow
        # or underflow even when the subsequent BF16 dot is finite.
        return values.to(cutlass.Float32) * cutlass.vector.full(
            (4,), scale, dtype=cutlass.Float32
        )
    return values * cutlass.vector.full((4,), scale, dtype=cutlass.Float16)


@cute.jit
def weight_scale(
    byte,
    scale_block_size: cutlass.Constexpr,
    integer_weights: cutlass.Constexpr = False,
):
    if cutlass.const_expr(integer_weights):
        return byte.to(cutlass.Float32)
    elif cutlass.const_expr(scale_block_size == 32):
        bits = byte.to(cutlass.Uint32) << 23
        if byte == 0:
            bits = cutlass.Uint32(0x00400000)
        if byte == 255:
            bits = cutlass.Uint32(0x7FC00000)
        return bits.bitcast(cutlass.Float32)
    return byte.bitcast(cutlass.Float8E4M3FN).to(cutlass.Float16)


@cute.jit
def _int4_quad(word):
    # Adapted from TRT-LLM/CUTLASS FastInterleavedAndBiasedNumericArrayConverter
    # in cutlass_extensions/interleaved_numeric_conversion.h. Spread four
    # consecutive signed nibbles into two half2 pairs, undo the sign bias,
    # then widen before applying arbitrary BF16 scales in FP32.
    h02, h13 = cute.arch.inline_ptx(
        """{
        .reg .b32 spread, biased, low, high, magic, one16, neg72;
        mov.b32 magic, 0x64086408;
        mov.b32 one16, 0x2c002c00;
        mov.b32 neg72, 0xd480d480;
        prmt.b32 spread, {$r0}, 0, 0x4140;
        xor.b32 biased, spread, 0x00880088;
        lop3.b32 low, biased, 0x000f000f, 0x64006400, 0xea;
        lop3.b32 high, biased, 0x00f000f0, 0x64006400, 0xea;
        sub.f16x2 {$w0}, low, magic;
        fma.rn.f16x2 {$w1}, high, one16, neg72;
        }""",
        write_only_types=[cutlass.Uint32, cutlass.Uint32],
        read_only_args=[word.to(cutlass.Uint32)],
    )
    a = cutlass.Vector.from_elements((h02,), cutlass.Uint32).bitcast(cutlass.Float16)
    b = cutlass.Vector.from_elements((h13,), cutlass.Uint32).bitcast(cutlass.Float16)
    values = cutlass.Vector.from_elements(
        (
            a[0].to(cutlass.Float32),
            b[0].to(cutlass.Float32),
            a[1].to(cutlass.Float32),
            b[1].to(cutlass.Float32),
        ),
        cutlass.Float32,
    )
    return values


@cute.jit
def scaled_values(
    word,
    scale,
    native: cutlass.Constexpr,
    scale_block_size: cutlass.Constexpr,
    integer_weights: cutlass.Constexpr,
    values_per_lane: cutlass.Constexpr,
):
    if cutlass.const_expr(values_per_lane == 4):
        return scaled_quad(word, scale, native, scale_block_size, integer_weights)
    # Native FP4 conversion consumes a 16-bit packed word even when the
    # coalesced load fetched eight nibbles in a uint32.
    low = scaled_quad(
        word.to(cutlass.Uint16), scale, native, scale_block_size, integer_weights
    )
    high = scaled_quad(
        (word >> 16).to(cutlass.Uint16),
        scale,
        native,
        scale_block_size,
        integer_weights,
    )
    return cutlass.Vector.from_elements(
        (low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3]),
        cutlass.Float32,
    )
