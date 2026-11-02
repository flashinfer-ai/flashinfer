# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import unittest
from unittest.mock import patch

from moe_hopper_bf16.heuristic_config import (
    ALL_EXPERTS_GROUP,
    HEURISTIC_CONFIGS,
    TAIL_SPLIT_ENV,
    TOKEN_BUCKETS,
    resolve_hopper_bf16_config,
    select_heuristic_config,
    tail_split_env_enabled,
    token_bucket,
)


def _token_cluster_is_two(config) -> bool:
    """Mirror of the geometry gate: swap-AB cga (1,2,1), non-swap cga (2,1,1)."""
    if config.swap_ab:
        return config.cluster_shape_mnk == (1, 2, 1)
    return config.cluster_shape_mnk == (2, 1, 1)


class HopperBf16HeuristicConfigTest(unittest.TestCase):
    def test_all_token_entries_exist(self) -> None:
        self.assertEqual(tuple(HEURISTIC_CONFIGS), TOKEN_BUCKETS)
        for config in HEURISTIC_CONFIGS.values():
            self.assertEqual(config.mma_tiler_mnk[2], 64)
            self.assertEqual(config.cluster_shape_mnk[2], 1)
            self.assertIn(
                config.token_back_mode,
                ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
            )

    def test_token_back_is_epi_warps_everywhere(self) -> None:
        # epi_warps won every bucket in the 2026-09-17 token-back sweep.
        for bucket, config in HEURISTIC_CONFIGS.items():
            with self.subTest(bucket=bucket):
                self.assertEqual(config.token_back_mode, "epi_warps")

    def test_group_hint_only_on_small_expert_buckets(self) -> None:
        # Several experts per group only help small experts (<= 8192 tokens).
        expected = {32: ALL_EXPERTS_GROUP, 64: ALL_EXPERTS_GROUP, 512: ALL_EXPERTS_GROUP}
        for bucket, config in HEURISTIC_CONFIGS.items():
            with self.subTest(bucket=bucket):
                if bucket >= 16384:
                    self.assertIsNone(config.group_hint)
                else:
                    self.assertEqual(config.group_hint, expected.get(bucket, 264))

    def test_token_bucket_uses_clamped_ceil_power_of_two(self) -> None:
        expected = {
            1: 8,
            8: 8,
            9: 16,
            31: 32,
            32: 32,
            33: 64,
            32768: 32768,
            32769: 32768,
        }
        for tokens, bucket in expected.items():
            with self.subTest(tokens=tokens):
                self.assertEqual(token_bucket(tokens), bucket)
        with self.assertRaises(ValueError):
            token_bucket(0)

    def test_representative_configs(self) -> None:
        selection = select_heuristic_config(32768)
        self.assertEqual(selection.token_bucket, 32768)
        # Tile-K=64 twins on a 4x H200 SXM node (2026-09-11): swap ping-pong
        # M128N128 K64 CGA1x2 is the measured best for the 32768 bucket.
        self.assertTrue(selection.config.swap_ab)
        self.assertTrue(selection.config.pingpong)
        self.assertEqual(selection.config.mma_tiler_mnk, (128, 128, 64))
        self.assertEqual(selection.config.cluster_shape_mnk, (1, 2, 1))
        self.assertEqual(selection.config.token_back_mode, "epi_warps")
        self.assertIsNone(selection.config.group_hint)

    def test_manual_geometry_disables_heuristic(self) -> None:
        selection = resolve_hopper_bf16_config(
            32768,
            mma_tiler_mnk=(64, 256, 128),
            cluster_shape_mnk=(1, 1, 1),
        )
        self.assertEqual(selection.source, "manual")
        self.assertIsNone(selection.token_bucket)
        self.assertFalse(selection.config.swap_ab)
        self.assertFalse(selection.config.pingpong)
        self.assertEqual(selection.config.mma_tiler_mnk, (64, 256, 128))

    def test_manual_swap_preserves_legacy_default_tile(self) -> None:
        legacy = resolve_hopper_bf16_config(128, swap_ab=True)
        self.assertEqual(legacy.config.mma_tiler_mnk, (256, 32, 64))
        pingpong = resolve_hopper_bf16_config(128, swap_ab=True, pingpong=True)
        self.assertEqual(pingpong.config.mma_tiler_mnk, (128, 32, 64))

    def test_default_selection_is_the_heuristic_table_entry(self) -> None:
        selection = resolve_hopper_bf16_config(64)
        self.assertEqual(selection.source, "heuristic")
        self.assertEqual(selection.token_bucket, 64)
        self.assertEqual(selection.config, HEURISTIC_CONFIGS[64])

    def test_tail_split_pairs_on_large_buckets_only(self) -> None:
        # Tail-split is on for the 1024-32768 swap CGA1x2 entries, always on a
        # geometry the kernel accepts; with the env off the entry is returned as-is.
        for bucket, config in HEURISTIC_CONFIGS.items():
            with self.subTest(bucket=bucket):
                self.assertEqual(config.tail_split_pairs, bucket >= 1024)
                if config.tail_split_pairs:
                    self.assertTrue(_token_cluster_is_two(config))
        with patch.dict(os.environ, {TAIL_SPLIT_ENV: "0"}):
            self.assertFalse(tail_split_env_enabled())
            for bucket in TOKEN_BUCKETS:
                with self.subTest(bucket=bucket):
                    selection = select_heuristic_config(bucket)
                    self.assertEqual(selection.config, HEURISTIC_CONFIGS[bucket])
            manual = resolve_hopper_bf16_config(
                4096, swap_ab=True, cluster_shape_mnk=(1, 2, 1)
            )
            self.assertFalse(manual.config.tail_split_pairs)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TAIL_SPLIT_ENV, None)
            self.assertFalse(tail_split_env_enabled())
            self.assertTrue(select_heuristic_config(8192).config.tail_split_pairs)
            self.assertFalse(select_heuristic_config(512).config.tail_split_pairs)

    def test_tail_split_env_override_only_on_token_cluster_two(self) -> None:
        # BF16_TAIL_SPLIT=1 flips the flag only on 2-CTA token clusters; other
        # buckets keep the table value so a sweep across cga(1,1,1) does not raise.
        with patch.dict(os.environ, {TAIL_SPLIT_ENV: "1"}):
            self.assertTrue(tail_split_env_enabled())
            outcomes = set()
            for bucket, table_config in HEURISTIC_CONFIGS.items():
                with self.subTest(bucket=bucket):
                    selection = select_heuristic_config(bucket)
                    expected = _token_cluster_is_two(table_config)
                    self.assertEqual(selection.config.tail_split_pairs, expected)
                    self.assertEqual(selection.config.tail_split_geometry, expected)
                    if table_config.tail_split_pairs:
                        self.assertTrue(expected)
                    # Only the flag differs from the table entry.
                    self.assertEqual(
                        selection.config,
                        HEURISTIC_CONFIGS[bucket].__class__(
                            **{
                                **HEURISTIC_CONFIGS[bucket].__dict__,
                                "tail_split_pairs": expected,
                            }
                        ),
                    )
                    # The override never leaks back into the table.
                    self.assertEqual(
                        HEURISTIC_CONFIGS[bucket].tail_split_pairs, bucket >= 1024
                    )
                    outcomes.add(selection.config.tail_split_pairs)
            # The current table exercises both branches of the gate.
            self.assertEqual(outcomes, {False, True})
            # Spot checks: swap cga(1,2,1) and non-swap cga(2,1,1) qualify,
            # swap cga(2,1,1) (weight-side pair) and cga(1,1,1) do not.
            self.assertTrue(select_heuristic_config(8192).config.tail_split_pairs)
            self.assertTrue(select_heuristic_config(2048).config.tail_split_pairs)
            self.assertFalse(select_heuristic_config(8).config.tail_split_pairs)
            self.assertFalse(select_heuristic_config(32).config.tail_split_pairs)

    def test_tail_split_env_override_manual_geometry(self) -> None:
        cases = {
            (True, (1, 2, 1)): True,
            (True, (2, 1, 1)): False,
            (True, (1, 1, 1)): False,
            (True, (2, 2, 1)): False,
            (False, (2, 1, 1)): True,
            (False, (1, 2, 1)): False,
            (False, (1, 1, 1)): False,
            (False, (2, 2, 1)): False,
        }
        with patch.dict(os.environ, {TAIL_SPLIT_ENV: "1"}):
            for (swap_ab, cga), flag in cases.items():
                with self.subTest(swap_ab=swap_ab, cga=cga):
                    selection = resolve_hopper_bf16_config(
                        4096, swap_ab=swap_ab, cluster_shape_mnk=cga
                    )
                    self.assertEqual(selection.source, "manual")
                    self.assertEqual(selection.config.cluster_shape_mnk, cga)
                    self.assertEqual(selection.config.tail_split_pairs, flag)

    def test_tail_split_env_rejects_unknown_values(self) -> None:
        with patch.dict(os.environ, {TAIL_SPLIT_ENV: "yes"}):
            with self.assertRaises(ValueError):
                tail_split_env_enabled()
            with self.assertRaises(ValueError):
                select_heuristic_config(8192)


if __name__ == "__main__":
    unittest.main()
