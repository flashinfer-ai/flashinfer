# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import triton as tr
import triton.language as tl


@tr.jit
def _fp4(data, channel):
    nibble = (data.to(tl.int32) >> ((channel % 2) * 4)) & 15
    magnitude = nibble & 7
    value = tl.where(
        magnitude < 2,
        magnitude * 0.5,
        tl.exp2(((magnitude >> 1) - 1).to(tl.float32)) * (1.0 + (magnitude & 1) * 0.5),
    )
    return tl.where((nibble & 8) != 0, -value, value)


@tr.jit
def _scale(byte):
    exponent = byte.to(tl.int32)
    bits = tl.where(
        exponent == 0, 0x00400000, tl.where(exponent == 255, 0x7FC00000, exponent << 23)
    )
    return bits.to(tl.float32, bitcast=True)


@tr.jit
def _fp4_bits(data, channel):
    nibble = (data.to(tl.int32) >> ((channel % 2) * 4)) & 15
    magnitude = nibble & 7
    bits = tl.where(magnitude < 2, magnitude * 0x3F000000, (magnitude + 252) << 22) | (
        (nibble & 8) << 28
    )
    return bits.to(tl.float32, bitcast=True)
