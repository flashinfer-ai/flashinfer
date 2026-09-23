# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent integer models for MXFP4 local-copy layout transformations.

These prove coordinate/bit-preservation properties, not generated GPU code or
memory-ordering correctness. Full-output and native GPU gates are mandatory.
"""

import unittest


def scalar_cells(valid_tokens, tile_base=0):
    # Unique 16-bit test patterns within one physical 64-token x M256 tile.
    return {
        (token, tile_base + hidden): token * 256 + hidden
        for token in range(valid_tokens)
        for hidden in range(256)
    }


def peer32_cells(valid_tokens, tile_base=0):
    cells = {}
    stores = 0
    for wg in range(2):
        for warp in range(4):
            for token_group in range(8):
                for m_sub in range(2):
                    for hpair in range(2):
                        lane_words = []
                        for lane in range(32):
                            group, lane_mod = divmod(lane, 4)
                            token0 = token_group * 8 + lane_mod * 2
                            hidden = (
                                wg * 128 + m_sub * 64 + warp * 16 + group + hpair * 8
                            )
                            lo = token0 * 256 + hidden
                            hi = (token0 + 1) * 256 + hidden
                            lane_words.append(lo | (hi << 16))
                        for lane in range(32):
                            group, lane_mod = divmod(lane, 4)
                            odd = group % 2
                            own, peer = lane_words[lane], lane_words[lane ^ 4]
                            packed = (own & 0xFFFF) | ((peer << 16) & 0xFFFFFFFF)
                            if odd:
                                packed = (peer >> 16) | (own & 0xFFFF0000)
                            token = token_group * 8 + lane_mod * 2 + odd
                            hidden = (
                                tile_base
                                + wg * 128
                                + m_sub * 64
                                + warp * 16
                                + group
                                - odd
                                + hpair * 8
                            )
                            if token < valid_tokens:
                                assert (hidden * 2) % 4 == 0
                                for delta, bits in (
                                    (0, packed & 0xFFFF),
                                    (1, packed >> 16),
                                ):
                                    key = (token, hidden + delta)
                                    assert key not in cells, ("duplicate write", key)
                                    cells[key] = bits
                                stores += 1
    return cells, stores


def scalar_offset_map(
    tile_m, logical_k, output_channels, expert, m_tile, k_tile, stage
):
    blocks = tile_m // 64
    source = {}
    for lane in range(32):
        for item in range(blocks):
            vector = lane + item * 32
            m64, vector_in_m64 = divmod(vector, 32)
            k128, row = divmod(vector_in_m64, 16)
            global_m64 = m_tile * blocks + m64
            global_k128 = k_tile * 2 + k128
            for byte in range(16):
                gmem = (
                    (
                        (expert * (output_channels // 64) + global_m64)
                        * (logical_k // 128)
                        + global_k128
                    )
                    * 16
                    + row
                ) * 16 + byte
                smem = byte + row * 16 + m64 * 512 + k128 * 256 + stage * blocks * 512
                assert smem not in source
                source[smem] = gmem
    return source


def bulk_offset_map(tile_m, logical_k, output_channels, expert, m_tile, k_tile, stage):
    blocks = tile_m // 64
    source = {}
    for block in range(blocks):
        global_m64 = m_tile * blocks + block
        global_k128 = k_tile * 2
        gmem_base = (
            (expert * (output_channels // 64) + global_m64) * (logical_k // 128)
            + global_k128
        ) * 256
        smem_base = block * 512 + stage * blocks * 512
        for byte in range(512):
            assert smem_base + byte not in source
            source[smem_base + byte] = gmem_base + byte
    return source


class TestMxfp4LocalLayout(unittest.TestCase):
    def test_peer32_all_valid_row_counts_including_odd_tails(self):
        for valid in range(65):
            for base in (0, 7168 - 256):
                with self.subTest(valid=valid, hidden_base=base):
                    actual, stores = peer32_cells(valid, base)
                    self.assertEqual(actual, scalar_cells(valid, base))
                    self.assertEqual(stores * 2, valid * 256)

    def test_bulk_offset_map_preserves_every_source_byte(self):
        for m in (128, 256):
            for k, channels in ((7168, 6144), (3072, 7168)):
                for expert in (0, 95):
                    for m_tile in (0, channels // m - 1):
                        for k_tile in (0, k // 256 - 1):
                            for stage in (0, 2):
                                args = (m, k, channels, expert, m_tile, k_tile, stage)
                                actual = bulk_offset_map(*args)
                                self.assertEqual(actual, scalar_offset_map(*args))
                                self.assertEqual(len(actual), m // 64 * 512)


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    print("LAYOUT_CPU_PASS" if result.wasSuccessful() else "LAYOUT_CPU_FAIL")
    raise SystemExit(0 if result.wasSuccessful() else 1)
