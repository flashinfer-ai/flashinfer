# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""FP8 copy of the fused fc1+fc2 scheduler's tail-split pair contract.

Pure Python (no CuTe DSL, no package import): the scheduler mirror
(``phase_tiles`` / ``decode``) is copied verbatim from
``moe_hopper_bf16/test_tail_split_sched.py`` because the scheduler is shared,
and the same invariants are checked for the FP8 CTA token tiles: the swap-AB
tiles N=8/16/32/64/128 plus the non-swap M=64 tile.
"""

from __future__ import annotations

import itertools
import unittest


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def phase_tiles(tokens: int, num_weight_blocks: int, cluster_tile_m: int,
                cta_tile_m: int, tail_split: bool) -> int:
    """Mirror of ``fc12_expert_phase_tiles``."""
    tiles = ceil_div(tokens, cluster_tile_m) * num_weight_blocks
    if tail_split and (ceil_div(tokens, cta_tile_m) & 1) == 1:
        tiles = tiles - num_weight_blocks + (num_weight_blocks + 1) // 2
    return tiles


def decode(local_id: int, tokens: int, num_weight_blocks: int, cta_tile_m: int,
           cta_rank: int, tail_split: bool, is_fc1: bool):
    """Mirror of the (internal M = tokens) part of ``_decode_inside_expert``.

    Returns (cta_token_tile, cta_weight_tile, valid_tokens, is_pair).
    """
    cm = 2
    cluster_tile_m = cm * cta_tile_m
    blocks = ceil_div(tokens, cluster_tile_m)
    cluster_tb, wb = divmod(local_id, num_weight_blocks)
    cta_tb = cluster_tb * cm + cta_rank
    cta_wb = wb
    is_pair = False
    force_zero = False
    if tail_split:
        is_split = (ceil_div(tokens, cta_tile_m) & 1) == 1
        normal_tiles = (blocks - 1) * num_weight_blocks
        if is_split and local_id >= normal_tiles:
            j = local_id - normal_tiles
            wb0 = 2 * j
            tail_tb = blocks - 1
            if wb0 + 1 < num_weight_blocks:
                cta_tb = tail_tb * cm
                cta_wb = wb0 + cta_rank
                is_pair = True
            else:
                cta_wb = wb0
                cta_tb = tail_tb * cm + cta_rank
                if not is_fc1 and cta_rank != 0:
                    cta_tb = tail_tb * cm
                    force_zero = True
    valid = min(max(tokens - cta_tb * cta_tile_m, 0), cta_tile_m)
    if force_zero:
        valid = 0
    return cta_tb, cta_wb, valid, is_pair


TOKEN_CASES = (1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 260, 384, 385, 513, 2048, 2049)
WEIGHT_CASES = (1, 2, 7, 9, 12, 14, 17, 24, 28, 48, 49, 56)
# FP8 CTA token tiles: swap-AB N in SwapABTokenTileNChoices (8/16/32/64/128)
# and the non-swap CTA M=64 tile.  The mirror is tile-agnostic.
TILE_CASES = (8, 16, 32, 64, 128)


def _valid(tokens: int, tb: int, tm: int) -> int:
    return min(max(tokens - tb * tm, 0), tm)


class Fp8TailSplitSchedulerContract(unittest.TestCase):
    def test_flag_off_matches_plain_formula(self) -> None:
        for tokens, w, tm in itertools.product(TOKEN_CASES, WEIGHT_CASES, TILE_CASES):
            n_tasks = phase_tiles(tokens, w, 2 * tm, tm, False)
            self.assertEqual(n_tasks, ceil_div(tokens, 2 * tm) * w)
            for local_id in range(n_tasks):
                for rank in (0, 1):
                    tb, wb, valid, pair = decode(local_id, tokens, w, tm, rank, False, True)
                    self.assertFalse(pair)
                    self.assertEqual((tb, wb), ((local_id // w) * 2 + rank, local_id % w))
                    self.assertEqual(valid, _valid(tokens, tb, tm))

    def test_coverage_and_cluster_uniformity(self) -> None:
        for tokens, w, tm in itertools.product(TOKEN_CASES, WEIGHT_CASES, TILE_CASES):
            for is_fc1 in (True, False):
                with self.subTest(tokens=tokens, w=w, tm=tm, fc1=is_fc1):
                    n_tasks = phase_tiles(tokens, w, 2 * tm, tm, True)
                    seen = {}
                    for local_id in range(n_tasks):
                        d0 = decode(local_id, tokens, w, tm, 0, True, is_fc1)
                        d1 = decode(local_id, tokens, w, tm, 1, True, is_fc1)
                        self.assertEqual(d0[3], d1[3], "split flag differs between CTAs")
                        if d0[3]:
                            # Pair task: valid tokens on both CTAs, same token
                            # tile, adjacent in-range weight tiles.
                            self.assertGreater(min(d0[2], d1[2]), 0)
                            self.assertEqual(d0[0], d1[0])
                            self.assertEqual(d1[1], d0[1] + 1)
                            self.assertLess(d1[1], w)
                        for tb, wb, valid, _ in (d0, d1):
                            self.assertLess(wb, w)
                            if valid > 0:
                                self.assertNotIn((tb, wb), seen, "tile computed twice")
                                seen[(tb, wb)] = valid
                    expected = {
                        (tb, wb): _valid(tokens, tb, tm)
                        for tb in range(ceil_div(tokens, tm))
                        for wb in range(w)
                    }
                    self.assertEqual(seen, expected)
                    # An odd CTA-tile count never wastes a task on two padding tiles.
                    if (ceil_div(tokens, tm) & 1) == 1:
                        self.assertEqual(
                            n_tasks, (ceil_div(tokens, 2 * tm) - 1) * w + (w + 1) // 2
                        )

    def test_fc1_done_slots_and_fc2_done_expectation(self) -> None:
        for tokens, tm in itertools.product(TOKEN_CASES, TILE_CASES):
            # fc1 publishes are unconditional: one per FC1 (task, CTA) per slot.
            fc1_publishes = {}
            for w1 in WEIGHT_CASES:
                counts = {}
                for local_id in range(phase_tiles(tokens, w1, 2 * tm, tm, True)):
                    for rank in (0, 1):
                        tb = decode(local_id, tokens, w1, tm, rank, True, True)[0]
                        counts[tb] = counts.get(tb, 0) + 1
                fc1_publishes[w1] = counts
            blocks = ceil_div(tokens, 2 * tm)
            odd_cta_tiles = (ceil_div(tokens, tm) & 1) == 1
            for w2 in WEIGHT_CASES:
                waited = set()
                fc2_publishes = 0
                for local_id in range(phase_tiles(tokens, w2, 2 * tm, tm, True)):
                    for rank in (0, 1):
                        waited.add(decode(local_id, tokens, w2, tm, rank, True, False)[0])
                        fc2_publishes += 1
                for w1 in WEIGHT_CASES:
                    with self.subTest(tokens=tokens, w1=w1, w2=w2, tm=tm):
                        for tb in waited:
                            self.assertEqual(
                                fc1_publishes[w1].get(tb, 0), w1, f"slot {tb} publishes"
                            )
                # Mirror of the token_comm fc2_done expectation
                # (megamoe_kernel_fp8: token_cluster * ceil(W2 / 2) on a split tail).
                if odd_cta_tiles:
                    expected = 2 * w2 * (blocks - 1) + 2 * ((w2 + 1) // 2)
                else:
                    expected = 2 * w2 * blocks
                with self.subTest(tokens=tokens, w2=w2, tm=tm):
                    self.assertEqual(fc2_publishes, expected)

    def test_zero_token_expert_has_no_tasks(self) -> None:
        for w, tm in itertools.product(WEIGHT_CASES, TILE_CASES):
            self.assertEqual(phase_tiles(0, w, 2 * tm, tm, True), 0)
            self.assertEqual(phase_tiles(0, w, 2 * tm, tm, False), 0)


if __name__ == "__main__":
    unittest.main()
