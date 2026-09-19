"""Device primitives matching FlashInfer's software stochastic FP16 path."""

import triton
import triton.language as tl


@triton.jit
def philox4x32(seed, offset, rounds: tl.constexpr):
    # Triton 3.2 selects Philox4x64 for a uint64 randint4x offset, whereas
    # 3.6 splits that offset into two Philox4x32 words. Select the 32-bit
    # algorithm explicitly and retain both counter words and the full seed.
    counter = offset.to(tl.uint64)
    low = counter.to(tl.uint32)
    high = (counter >> 32).to(tl.uint32)
    zero = tl.full(low.shape, 0, tl.uint32)
    return tl.philox(seed.to(tl.uint64), low, high, zero, zero, rounds)


@triton.jit
def cvt_rs_f16(value, random_word):
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 16) & 0x8000
    magnitude = (bits & 0x7FFFFFFF) + (random_word & 0x1FFF)
    exponent = (magnitude >> 23) & 0xFF
    mantissa = magnitude & 0x7FFFFF
    half_bits = ((exponent - 112) << 10) | (mantissa >> 13)
    half_bits = tl.where(exponent < 113, 0, half_bits)
    half_bits = tl.where(exponent > 142, 0x7C00, half_bits)
    half_bits = tl.where(
        exponent == 255, tl.where(mantissa != 0, 0x7E00, 0x7C00), half_bits
    )
    return (half_bits | sign).to(tl.uint16).to(tl.float16, bitcast=True)
