"""Independent scalar oracles for future CUDA/MUSA stochastic kernel checks.

The half conversion follows conversion.cuh:cvt_rs_f16_sw, including its
software-path treatment of noise on non-finite bit patterns. This is not a
replacement for comparing the device primitives against these vectors.
"""

import pytest


def philox4x32_words(seed: int, offset: int, rounds: int):
    mask = (1 << 32) - 1
    c0, c1, c2, c3 = offset & mask, (offset >> 32) & mask, 0, 0
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(rounds):
        p0, p1 = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & mask,
            p1 & mask,
            ((p0 >> 32) ^ c3 ^ k1) & mask,
            p0 & mask,
        )
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return c0, c1, c2, c3


def cvt_rs_f16_bits(bits: int, random_word: int):
    sign = (bits >> 16) & 0x8000
    magnitude = ((bits & 0x7FFFFFFF) + (random_word & 0x1FFF)) & 0xFFFFFFFF
    exponent, mantissa = (magnitude >> 23) & 0xFF, magnitude & 0x7FFFFF
    if exponent == 255:
        value = 0x7E00 if mantissa else 0x7C00
    elif exponent > 142:
        value = 0x7C00
    elif exponent < 113:
        value = 0
    else:
        value = ((exponent - 112) << 10) | (mantissa >> 13)
    return sign | value


@pytest.mark.parametrize(
    "seed,offset,rounds,expected",
    [
        (0, 0, 5, (0xF10446C0, 0xC841128B, 0x23F43D26, 0x6CB9F044)),
        (0, 0, 10, (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (42, 4294967297, 5, (0xA8F74EE6, 0x79FDAC93, 0x112F03D5, 0x5643E4A2)),
        (42, 4294967297, 10, (0x220383A8, 0x6880FFBE, 0x93EF3466, 0xDA94BB0D)),
    ],
)
def test_known_philox_words(seed, offset, rounds, expected):
    assert philox4x32_words(seed, offset, rounds) == expected


@pytest.mark.parametrize(
    "bits,noise,expected",
    [
        (0x00000000, 8191, 0x0000),
        (0x80000000, 8191, 0x8000),
        (0x00000001, 8191, 0x0000),
        (0x80000001, 8191, 0x8000),
        (0x38800000, 0, 0x0400),
        (0x387FFFFF, 0, 0x0000),
        (0x387FFFFF, 1, 0x0400),
        (0xB87FFFFF, 1, 0x8400),
        (0x3F800000, 8191, 0x3C00),
        (0x3F801000, 4095, 0x3C00),
        (0x3F801000, 4096, 0x3C01),
        (0xBF801000, 4096, 0xBC01),
        (0x477FF000, 4095, 0x7BFF),
        (0x477FF000, 4096, 0x7C00),
        (0x7F800000, 0, 0x7C00),
        (0xFF800000, 0, 0xFC00),
        (0x7F800000, 1, 0x7E00),
        (0x7FC00000, 0, 0x7E00),
    ],
)
def test_conversion_boundaries(bits, noise, expected):
    assert cvt_rs_f16_bits(bits, noise) == expected
