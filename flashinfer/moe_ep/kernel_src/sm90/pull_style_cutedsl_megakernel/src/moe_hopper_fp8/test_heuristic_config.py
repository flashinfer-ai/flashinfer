# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import unittest
from unittest.mock import patch

from moe_hopper_fp8.heuristic_config import (
    ALL_EXPERTS_GROUP,
    HEURISTIC_CONFIGS,
    HEURISTIC_GENERATE_C_OVERRIDES,
    TAIL_SPLIT_ENV,
    TOKEN_BUCKETS,
    resolve_hopper_fp8_config,
    select_heuristic_config,
    tail_split_env_enabled,
    token_bucket,
)

SCALE_MODES = ("per_tensor", "blockwise")
# (group_hint, tail_split_pairs) per bucket where the table sets them.
KNOB_TABLE = {
    "per_tensor": {
        512: (264, False), 1024: (264, True),
        2048: (264, True), 4096: (264, True), 8192: (264, True),
        16384: (None, True), 32768: (None, True),
    },
    "blockwise": {
        32: (None, True), 512: (264, False), 1024: (264, False),
        2048: (264, False), 8192: (264, True),
    },
}
BLOCKWISE_REUSE_BUCKETS = {16384, 32768}



def _token_cluster_is_two(config) -> bool:
    """Mirror of the geometry gate: swap-AB cga (1,2,1), non-swap cga (2,1,1)."""
    if config.swap_ab:
        return config.cluster_shape_mnk == (1, 2, 1)
    return config.cluster_shape_mnk == (2, 1, 1)


class HopperFp8HeuristicConfigTest(unittest.TestCase):
    def test_all_scale_token_entries_exist(self) -> None:
        self.assertEqual(set(HEURISTIC_CONFIGS), {"per_tensor", "blockwise"})
        for configs in HEURISTIC_CONFIGS.values():
            self.assertEqual(tuple(configs), TOKEN_BUCKETS)
            for config in configs.values():
                self.assertEqual(config.accum_mode, "1xacc")
                self.assertEqual(config.mma_tiler_mnk[2], 128)
                self.assertEqual(config.cluster_shape_mnk[2], 1)
                self.assertIn(
                    config.token_back_mode,
                    ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
                )

    def test_token_back_modes(self) -> None:
        # per_tensor: epi_warps everywhere (2026-09-18 four-rank H200 A/B: the
        # swap-AB N128 tail-split entries tie reuse_dispatch_warps at 16384+).
        # blockwise: reuse_dispatch_warps only where it measured ahead.
        reuse_buckets = {"per_tensor": set(), "blockwise": BLOCKWISE_REUSE_BUCKETS}
        for scale_mode in SCALE_MODES:
            for bucket, config in HEURISTIC_CONFIGS[scale_mode].items():
                expected = (
                    "reuse_dispatch_warps"
                    if bucket in reuse_buckets[scale_mode]
                    else "epi_warps"
                )
                with self.subTest(scale_mode=scale_mode, bucket=bucket):
                    self.assertEqual(config.token_back_mode, expected)

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
        self.assertTrue(per_tensor.config.swap_ab)
        self.assertTrue(per_tensor.config.pingpong)
        self.assertEqual(per_tensor.config.mma_tiler_mnk, (128, 128, 128))
        self.assertEqual(per_tensor.config.cluster_shape_mnk, (1, 2, 1))
        self.assertTrue(per_tensor.config.tail_split_pairs)
        self.assertIsNone(per_tensor.config.group_hint)
        self.assertEqual(
            select_heuristic_config("per_tensor", 4096).config.group_hint, 264
        )

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

    def test_group_hint_and_tail_split_table_values(self) -> None:
        # Only the buckets with a measured win carry the knobs (2026-09-18
        # four-rank H200 A/B); every other entry keeps the defaults.
        self.assertEqual(ALL_EXPERTS_GROUP, 1 << 20)
        for scale_mode in SCALE_MODES:
            for bucket, config in HEURISTIC_CONFIGS[scale_mode].items():
                group_hint, tail_split = KNOB_TABLE[scale_mode].get(
                    bucket, (None, False)
                )
                with self.subTest(scale_mode=scale_mode, bucket=bucket):
                    self.assertEqual(config.group_hint, group_hint)
                    self.assertEqual(config.tail_split_pairs, tail_split)
                    if tail_split:
                        self.assertTrue(config.tail_split_geometry)

    def test_generate_c_override_only_where_measured(self) -> None:
        # Training forward: only per_tensor 16384 swaps to its previous entry.
        for scale_mode in SCALE_MODES:
            overrides = HEURISTIC_GENERATE_C_OVERRIDES[scale_mode]
            for bucket in TOKEN_BUCKETS:
                table_config = HEURISTIC_CONFIGS[scale_mode][bucket]
                with self.subTest(scale_mode=scale_mode, bucket=bucket):
                    inference = select_heuristic_config(scale_mode, bucket)
                    training = select_heuristic_config(
                        scale_mode, bucket, generate_c=True
                    )
                    self.assertEqual(inference.config, table_config)
                    self.assertEqual(
                        training.config, overrides.get(bucket, table_config)
                    )
                    self.assertEqual(training.source, "heuristic")
        self.assertEqual(set(HEURISTIC_GENERATE_C_OVERRIDES["per_tensor"]), {16384})
        self.assertEqual(HEURISTIC_GENERATE_C_OVERRIDES["blockwise"], {})
        override = select_heuristic_config("per_tensor", 16384, generate_c=True).config
        self.assertFalse(override.swap_ab)
        self.assertEqual(override.mma_tiler_mnk, (64, 256, 128))
        self.assertEqual(override.cluster_shape_mnk, (2, 1, 1))
        self.assertEqual(override.token_back_mode, "reuse_dispatch_warps")
        self.assertFalse(override.tail_split_pairs)
        resolved = resolve_hopper_fp8_config("per_tensor", 16384, generate_c=True)
        self.assertEqual(resolved.config, override)
        manual = resolve_hopper_fp8_config(
            "per_tensor", 16384, swap_ab=True, cluster_shape_mnk=(1, 2, 1),
            generate_c=True,
        )
        self.assertEqual(manual.source, "manual")

    def test_tail_split_env_off_returns_table_entry(self) -> None:
        with patch.dict(os.environ, {TAIL_SPLIT_ENV: "0"}):
            self.assertFalse(tail_split_env_enabled())
            for scale_mode in SCALE_MODES:
                for bucket in TOKEN_BUCKETS:
                    with self.subTest(scale_mode=scale_mode, bucket=bucket):
                        selection = select_heuristic_config(scale_mode, bucket)
                        self.assertEqual(
                            selection.config, HEURISTIC_CONFIGS[scale_mode][bucket]
                        )
                manual = resolve_hopper_fp8_config(
                    scale_mode, 4096, swap_ab=True, cluster_shape_mnk=(1, 2, 1)
                )
                self.assertFalse(manual.config.tail_split_pairs)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(TAIL_SPLIT_ENV, None)
            self.assertFalse(tail_split_env_enabled())
            for scale_mode in SCALE_MODES:
                for bucket in (16, 8192):
                    self.assertEqual(
                        select_heuristic_config(scale_mode, bucket).config,
                        HEURISTIC_CONFIGS[scale_mode][bucket],
                    )

    def test_tail_split_env_override_only_on_token_cluster_two(self) -> None:
        # FP8_TAIL_SPLIT=1 flips the flag only on 2-CTA token clusters; other
        # buckets keep the table value so a sweep across cga(1,1,1) does not raise.
        spot_checks = {
            "per_tensor": {16: True, 2048: True, 4096: True, 16384: True,
                           8: False, 512: False, 256: False},
            "blockwise": {32: True, 256: True, 8192: True, 32768: True,
                          16: False, 1024: False, 16384: False, 4096: False},
        }
        with patch.dict(os.environ, {TAIL_SPLIT_ENV: "1"}):
            self.assertTrue(tail_split_env_enabled())
            for scale_mode in SCALE_MODES:
                outcomes = set()
                table = HEURISTIC_CONFIGS[scale_mode]
                for bucket, table_config in table.items():
                    with self.subTest(scale_mode=scale_mode, bucket=bucket):
                        selection = select_heuristic_config(scale_mode, bucket)
                        expected = _token_cluster_is_two(table_config)
                        self.assertEqual(selection.config.tail_split_pairs, expected)
                        self.assertEqual(
                            selection.config.tail_split_geometry, expected
                        )
                        # Only the flag differs from the table entry.
                        self.assertEqual(
                            selection.config,
                            table_config.__class__(
                                **{
                                    **table_config.__dict__,
                                    "tail_split_pairs": expected,
                                }
                            ),
                        )
                        # The override never leaks back into the table.
                        self.assertIs(
                            HEURISTIC_CONFIGS[scale_mode][bucket], table_config
                        )
                        outcomes.add(selection.config.tail_split_pairs)
                # The current table exercises both branches of the gate.
                self.assertEqual(outcomes, {False, True})
                for bucket, flag in spot_checks[scale_mode].items():
                    with self.subTest(scale_mode=scale_mode, spot=bucket):
                        self.assertEqual(
                            select_heuristic_config(
                                scale_mode, bucket
                            ).config.tail_split_pairs,
                            flag,
                        )

    def test_tail_split_env_override_survives_accum_mode_replace(self) -> None:
        with patch.dict(os.environ, {TAIL_SPLIT_ENV: "1"}):
            selection = resolve_hopper_fp8_config(
                "per_tensor", 16, accum_mode="2xacc"
            )
            self.assertEqual(selection.source, "heuristic")
            self.assertEqual(selection.config.accum_mode, "2xacc")
            self.assertTrue(selection.config.tail_split_pairs)

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
            for scale_mode in SCALE_MODES:
                for (swap_ab, cga), flag in cases.items():
                    with self.subTest(scale_mode=scale_mode, swap_ab=swap_ab, cga=cga):
                        selection = resolve_hopper_fp8_config(
                            scale_mode, 4096, swap_ab=swap_ab, cluster_shape_mnk=cga
                        )
                        self.assertEqual(selection.source, "manual")
                        self.assertEqual(selection.config.cluster_shape_mnk, cga)
                        self.assertEqual(selection.config.tail_split_pairs, flag)

    def test_tail_split_env_rejects_unknown_values(self) -> None:
        with patch.dict(os.environ, {TAIL_SPLIT_ENV: "yes"}):
            with self.assertRaises(ValueError):
                tail_split_env_enabled()
            with self.assertRaises(ValueError):
                select_heuristic_config("per_tensor", 8192)


if __name__ == "__main__":
    unittest.main()
