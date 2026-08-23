# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

from moe_hopper_fp8.heuristic_config import (
    HEURISTIC_CONFIGS,
    TOKEN_BUCKETS,
    resolve_hopper_fp8_config,
    select_heuristic_config,
    token_bucket,
)


class HopperFp8HeuristicConfigTest(unittest.TestCase):
    def test_all_scale_token_entries_exist(self) -> None:
        self.assertEqual(set(HEURISTIC_CONFIGS), {"per_tensor", "blockwise"})
        for configs in HEURISTIC_CONFIGS.values():
            self.assertEqual(tuple(configs), TOKEN_BUCKETS)
            for config in configs.values():
                self.assertEqual(config.accum_mode, "1xacc")
                self.assertEqual(config.mma_tiler_mnk[2], 128)
                self.assertEqual(config.cluster_shape_mnk[2], 1)

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

    def test_representative_scale_configs(self) -> None:
        per_tensor = select_heuristic_config("per_tensor", 32768)
        self.assertEqual(per_tensor.token_bucket, 32768)
        self.assertFalse(per_tensor.config.swap_ab)
        self.assertTrue(per_tensor.config.pingpong)
        self.assertEqual(per_tensor.config.mma_tiler_mnk, (64, 128, 128))
        self.assertEqual(per_tensor.config.cluster_shape_mnk, (2, 2, 1))

        blockwise = select_heuristic_config("blockwise", 256)
        self.assertEqual(blockwise.token_bucket, 256)
        self.assertTrue(blockwise.config.swap_ab)
        self.assertTrue(blockwise.config.pingpong)
        self.assertEqual(blockwise.config.mma_tiler_mnk, (128, 32, 128))
        self.assertEqual(blockwise.config.cluster_shape_mnk, (1, 2, 1))

    def test_manual_geometry_disables_heuristic(self) -> None:
        selection = resolve_hopper_fp8_config(
            "per_tensor",
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
        legacy = resolve_hopper_fp8_config("per_tensor", 128, swap_ab=True)
        self.assertEqual(legacy.config.mma_tiler_mnk, (256, 32, 128))
        pingpong = resolve_hopper_fp8_config(
            "per_tensor", 128, swap_ab=True, pingpong=True
        )
        self.assertEqual(pingpong.config.mma_tiler_mnk, (128, 32, 128))

    def test_accum_mode_overrides_heuristic_without_disabling_it(self) -> None:
        selection = resolve_hopper_fp8_config("per_tensor", 64, accum_mode="2xacc")
        self.assertEqual(selection.source, "heuristic")
        self.assertEqual(selection.token_bucket, 64)
        self.assertEqual(selection.config.accum_mode, "2xacc")

    def test_invalid_scale_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_heuristic_config("invalid", 128)


if __name__ == "__main__":
    unittest.main()
