# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw FP8 boundary/noise corpus; no host float conversions, Torch or CUDA."""


def reverse16(value):
    value = ((value & 0x5555) << 1) | ((value >> 1) & 0x5555)
    value = ((value & 0x3333) << 2) | ((value >> 2) & 0x3333)
    value = ((value & 0x0F0F) << 4) | ((value >> 4) & 0x0F0F)
    return ((value & 255) << 8) | ((value >> 8) & 255)


def base_make():
    import numpy as np

    exponents = (0, 1, 120, 121, 122, 134, 135, 136, 254, 255)
    mantissas = (
        0,
        1,
        0xFFFFF,
        0x100000,
        0x5FFFFF,
        0x600000,
        0x6FFFFF,
        0x700000,
        0x7FFFFE,
        0x7FFFFF,
    )
    companions = (
        0,
        0x80000000,
        0x3F800001,
        0xBF800001,
        0x7F800000,
        0x7FC00001,
        0x3C7FFFFF,
        0x43E00000,
    )
    low = np.arange(65536, dtype=np.uint32)
    high = (low * 4051 + 73) & 65535
    av = []
    bv = []
    rv = []
    for e in exponents:
        for m in mantissas:
            for sign in (0, 0x80000000):
                av.append(np.full(65536, sign | (e << 23) | m, dtype=np.uint32))
                bv.append(
                    np.full(
                        65536,
                        companions[(len(av) - 1) % len(companions)],
                        dtype=np.uint32,
                    )
                )
                rv.append(low | (high << 16))
    a0, b0, r0 = (np.concatenate(x) for x in (av, bv, rv))
    one = len(a0)
    reversed_bits = reverse16(r0 & 65535) | (reverse16(r0 >> 16) << 16)
    a = np.concatenate((a0, b0))
    b = np.concatenate((b0, a0))
    r = np.concatenate((r0, reversed_bits))
    base = len(a)
    random = np.random.default_rng(3074198).integers(
        0, 2**32, size=(3, 65536), dtype=np.uint32
    )
    a = np.concatenate((a, random[0]))
    b = np.concatenate((b, random[1]))
    r = np.concatenate((r, random[2]))
    return (a, b, r), dict(
        pairs=len(a),
        one_way_boundary_pairs=one,
        both_orderings_pairs=base,
        random_raw_pairs=65536,
        CPU_seed=3074198,
        original_f32_exponents=list(exponents),
        mantissas=list(mantissas),
        all65536_low_and_high_noise_marginals_per_boundary=True,
        not_exhaustive_2pow32_joint_random_patterns=True,
        four_values="a,b,signflip(a),signflip(b)",
        swap_property="swap adjacent bytes with both random16 halves bitreversed",
        host_float_NaN_conversion=False,
    )


def make():
    import numpy as np

    arrays, coverage = base_make()
    low = np.arange(65536, dtype=np.uint32)
    high = (low * 4051 + 73) & 65535
    av = []
    bv = []
    rv = []
    for e in (121, 122, 135, 253):
        for block in (0, 1, 6, 7):
            tie = ((-(low << 4)) & np.uint32(0xFFFFF)) + np.uint32(block << 20)
            for delta in (-1, 0, 1):
                m = np.clip(tie.astype(np.int64) + delta, 0, 0x7FFFFF).astype(np.uint32)
                for sign in (0, 0x80000000):
                    av.append(np.uint32(sign | (e << 23)) | m)
                    bv.append(np.full(65536, 0x3F800001, dtype=np.uint32))
                    rv.append(low | (high << 16))
    extra = tuple(np.concatenate(v) for v in (av, bv, rv))
    result = tuple(np.concatenate((a, b)) for a, b in zip(arrays, extra, strict=True))
    coverage.update(
        pairs=len(result[0]),
        eligible_packed_tie_pairs=len(extra[0]),
        eligible_packed_tie_exponents=[121, 122, 135, 253],
        eligible_packed_tie_all_noise=True,
        eligible_packed_tie_offsets=[-1, 0, 1],
        eligible_companion_original_exponent=127,
        actual_new_packed_x4_path_tested=True,
    )
    return result, coverage
