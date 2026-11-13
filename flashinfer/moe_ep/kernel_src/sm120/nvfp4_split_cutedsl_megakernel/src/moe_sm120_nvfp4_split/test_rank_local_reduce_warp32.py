# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-only coverage model for the whole-warp rank-local reducer.

GPU arithmetic, transport and replay equivalence require the multi-rank tests;
these tests only cover the channel mapping and preserved winner-pair boundary.
"""

import random


def _channels(lanes, chunk_hidden):
    channels = []
    for offset in range(0, chunk_hidden, lanes * 32):
        for lane in range(lanes):
            for fragment in range(2):
                for part in range(2):
                    first = offset + fragment * lanes * 16 + lane * 8 + part * lanes * 8
                    assert first * 2 % 16 == 0
                    channels.extend(range(first, first + 8))
    return channels


def test_whole_warp_channels_match_half_warp():
    for chunk_hidden in (1024, 2048, 3072, 4096):
        expected = list(range(chunk_hidden))
        assert sorted(_channels(16, chunk_hidden)) == expected
        assert sorted(_channels(32, chunk_hidden)) == expected
        # Each original pair still owns exactly two complete token-groups.
        pair = [(token, channel) for token in range(2)
                for channel in _channels(32, chunk_hidden)]
        assert len(pair) == len(set(pair)) == 2 * chunk_hidden


def test_whole_warp_preserves_winner_pairs_and_odd_tail():
    rng = random.Random(34)
    masks = [0, 1, 1 << 31, 0xffffffff, 0xaaaaaaaa, 0x55555555]
    masks += [(1 << i) | (1 << j) for i in range(32) for j in range(32)]
    masks += [rng.getrandbits(32) for _ in range(10000)]
    for mask in masks:
        expected = [lane for lane in range(32) if mask & (1 << lane)]
        emitted = []
        while mask:
            first = (mask & -mask).bit_length() - 1
            remaining = mask & (mask - 1)
            second = (remaining & -remaining).bit_length() - 1
            pair = []
            for token_item in range(2):
                if token_item == 0 or remaining:
                    pair.append(first if token_item == 0 else second)
            assert pair == expected[len(emitted):len(emitted) + 2]
            emitted.extend(pair)
            mask = remaining & (remaining - 1) if remaining else 0
        assert emitted == expected
