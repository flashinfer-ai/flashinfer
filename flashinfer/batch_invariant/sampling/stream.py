# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The request-keyed random stream of the batch-invariant samplers (``stream.philox4x32-10.splitmix-key.v1``), as an
exact integer specification.

* **key**: a row's ``seed`` and ``offset`` (64-bit; ``offset`` is the sequence position or a request-local counter) are
  mixed with SplitMix64's finalizer, ``K = mix(seed ^ mix(offset))``; the Philox key is ``(K mod 2^32, K >> 32)``;
* **counter**: Philox4x32-10 (Random123 / CUDA ``curand``: multipliers ``0xD2511F53``, ``0xCD9E8D57``, Weyl constants
  ``0x9E3779B9``, ``0xBB67AE85``, ten rounds) on the counter ``(n, 0, 0, 0)`` gives the four 32-bit words of block
  ``n``: ``word(i) = block(i >> 2)[i & 3]`` is what ``curand_init(K, 0, 0)`` followed by ``i`` calls of ``curand``
  returns. **No batch term anywhere**: a row's words depend on its key alone, never on its slot in the batch (the
  operator's own samplers seed ``curand_init(seed, slot, offset)``: the slot is the Philox subsequence there, so a row's
  draw changes with its position; under ``batch_invariant=True`` it cannot);
* **draws**: a class consumes whole blocks: draw ``d`` of a call reads block ``d``; the fp32 lane is word 0 of the
  block, the fp64 lane words 1 and 2, so enabling one lane never shifts the other (``words(key, d)``);
* **uniforms**: ``uniform32(w) = add(mul(cvt.rn.f32.u32(w), 2^-32), 2^-33)`` (``curand_uniform``'s conversion: one
  rounding in the cvt, one in the add; the value lies in (0, 1], 1.0 at the all-ones word as ``curand_uniform``
  documents); ``uniform64(w1, w2)`` takes the top 52 bits ``z`` of the 64-bit word ``(w2 << 32) | w1`` and returns
  ``(2 z + 1) * 2^-53`` (exact in binary64; in (0, 1), symmetric about 1/2, never 0 or 1).

Everything here is integer arithmetic plus the two declared conversions; the CPU model below is the specification, the
device helpers (``stream_kernel.py``) are checked against it word for word, and ``lean/BIKernels/Stream.lean`` states
the round function and the key layout and decides the published vectors.
"""

from __future__ import annotations

import hashlib
import struct
from typing import List, Tuple

from ..fp_interface import exact as fp32

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
PHILOX_ROUNDS = 10
SPLITMIX_GOLDEN = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
TWO_POW_M32 = 2.0**-32
TWO_POW_M33 = 2.0**-33
TWO_POW_M53 = 2.0**-53

STREAM_NAME = "stream.philox4x32-10.splitmix-key.v1"


def mix64(value: int) -> int:
    """SplitMix64's finalizer: a bijection of the 64-bit integers."""
    value = (value + SPLITMIX_GOLDEN) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return value ^ (value >> 31)


def request_key(seed: int, offset: int) -> int:
    """The 64-bit Philox key of a request at a position: ``mix(seed ^ mix(offset))``."""
    return mix64((seed & MASK64) ^ mix64(offset & MASK64))


def philox4x32_10(
    counter: Tuple[int, int, int, int], key: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """One Philox4x32-10 block: the four output words of ``counter`` under ``key`` (Random123's round function)."""
    c0, c1, c2, c3 = (c & MASK32 for c in counter)
    k0, k1 = key[0] & MASK32, key[1] & MASK32
    for r in range(PHILOX_ROUNDS):
        p0 = PHILOX_M0 * c0
        p1 = PHILOX_M1 * c2
        hi0, lo0 = p0 >> 32, p0 & MASK32
        hi1, lo1 = p1 >> 32, p1 & MASK32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        if r != PHILOX_ROUNDS - 1:
            k0 = (k0 + PHILOX_W0) & MASK32
            k1 = (k1 + PHILOX_W1) & MASK32
    return c0, c1, c2, c3


def key_words(key64: int) -> Tuple[int, int]:
    """The two 32-bit Philox key words of a 64-bit key: ``(low, high)``."""
    return key64 & MASK32, (key64 >> 32) & MASK32


def block(seed: int, offset: int, n: int) -> Tuple[int, int, int, int]:
    """The four words of block ``n`` of the request's stream (``curand_init(K, 0, 0)`` + skipahead ``4 n``)."""
    assert 0 <= n < 1 << 64
    return philox4x32_10(
        (n & MASK32, (n >> 32) & MASK32, 0, 0), key_words(request_key(seed, offset))
    )


def word(seed: int, offset: int, i: int) -> int:
    """The ``i``-th 32-bit word of the stream, as ``curand`` would return it after ``i`` draws."""
    return block(seed, offset, i >> 2)[i & 3]


def words(seed: int, offset: int, draw: int) -> Tuple[int, int, int, int]:
    """The block a class reads for draw ``draw`` of a call: word 0 is the fp32 lane, words 1 and 2 the fp64 lane."""
    return block(seed, offset, draw)


# ---------------------------------------------------------------------------------------------- the conversions
def cvt_f32_u32(w: int) -> float:
    """``cvt.rn.f32.u32``: the binary32 nearest to the integer (ties to even)."""
    return fp32.to_f32(float(w & MASK32))


def uniform32(w: int) -> float:
    """``curand_uniform``'s conversion of one word: ``add(mul(cvt(w), 2^-32), 2^-33)`` in binary32; in (0, 1] (the
    all-ones word converts to 2^32 and gives exactly 1.0, as ``curand_uniform`` documents)."""
    return fp32.add(fp32.mul(cvt_f32_u32(w), TWO_POW_M32), TWO_POW_M33)


def uniform64(w1: int, w2: int) -> float:
    """The fp64 lane: ``z`` = the top 52 bits of ``(w2 << 32) | w1``, ``(2 z + 1) * 2^-53`` (exact; in (0, 1))."""
    z = (((w2 & MASK32) << 32) | (w1 & MASK32)) >> 12
    return (
        float(2 * z + 1) * TWO_POW_M53
    )  # exact: 2z+1 < 2^53 and the scale is a power of two


def uniforms(seed: int, offset: int, draw: int) -> Tuple[float, float]:
    """``(uniform32, uniform64)`` of draw ``draw``: the fp32 and the fp64 lane of one block."""
    w = words(seed, offset, draw)
    return uniform32(w[0]), uniform64(w[1], w[2])


def f32_bits(x: float) -> int:
    """The IEEE binary32 bit pattern of ``x`` as an unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f64_bits(x: float) -> int:
    """The IEEE binary64 bit pattern of ``x`` as an unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def stream_fingerprint() -> str:
    """The receipt of the stream class: the constants and the layout, hashed."""
    parts = [
        STREAM_NAME,
        "philox4x32",
        str(PHILOX_ROUNDS),
        hex(PHILOX_M0),
        hex(PHILOX_M1),
        hex(PHILOX_W0),
        hex(PHILOX_W1),
        "splitmix64",
        hex(SPLITMIX_GOLDEN),
        hex(SPLITMIX_M1),
        hex(SPLITMIX_M2),
        "key=mix(seed^mix(offset))",
        "counter=(block,0,0,0)",
        "u32=add(mul(cvt(w),2^-32),2^-33)",
        "u64=(2*top52+1)*2^-53",
    ]
    return "stream-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def reference_words(
    seeds: List[int], offsets: List[int], draws: int
) -> List[List[Tuple[int, int, int, int]]]:
    """``[row][draw] -> block``: the CPU model for a batch (the GPU conformance test's oracle)."""
    return [
        [block(s, o, d) for d in range(draws)]
        for s, o in zip(seeds, offsets, strict=True)
    ]
