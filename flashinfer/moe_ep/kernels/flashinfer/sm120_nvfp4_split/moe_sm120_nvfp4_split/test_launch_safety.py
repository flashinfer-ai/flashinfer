"""CPU tests for the first-launch hang guard (heuristic.check_launch_safety)."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from . import tuned_table
from .heuristic import (
    MegaMoEHeuristicInput,
    check_launch_safety,
    select_megamoe_config,
    select_transport_overlay,
)

SIG = [5120, 4608, 6, 384]


def shape(tokens=16, ep=8, same=7, cross=0):
    return MegaMoEHeuristicInput(
        tokens_per_rank=tokens, hidden=5120, intermediate=4608, num_topk=6, num_total_experts=384,
        data_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=ep,
        ep_same_numa_peer_count=same, ep_cross_numa_peer_count=cross, num_sms=110,
        sm_min_partition=8, sm_partition_alignment=8)


def with_k2(cfg, n, lf, tail):
    return replace(cfg, k2_tile=(64, n, 128), k2_warps=8, k2_launch_factor=lf, k2_tail_reclaim=tail,
                   k2_natural_regs=True, k2_min_blocks_per_sm=2)


class LaunchSafetyTests(unittest.TestCase):
    def test_k2_n16_rejected_everywhere(self):
        for shp in (shape(ep=4, same=3), shape(), shape(ep=16, same=15)):
            cfg = with_k2(select_megamoe_config(shp), 16, 2, False)
            with self.assertRaisesRegex(ValueError, "k2_tail_reclaim"):
                check_launch_safety(shp, cfg)

    def test_k2_n32_stays_valid_on_p2p_split(self):
        # NVFP4 p2p_split EP8 ran K2 N32 with launch factor 2 and no tail reclaim.
        for ep, same in ((8, 7), (16, 15)):
            cfg = with_k2(select_megamoe_config(shape(ep=ep, same=same)), 32, 2, False)
            check_launch_safety(shape(ep=ep, same=same), cfg)

    def test_neighbours_stay_valid(self):
        for shp in (shape(ep=4, same=3), shape()):
            base = select_megamoe_config(shp)
            for lf, tail in ((2, True), (1, False), (1, True)):
                check_launch_safety(shp, with_k2(base, 16, lf, tail))
        # K2 N32 without tail reclaim stays allowed on one NUMA node.
        ep4 = shape(ep=4, same=3)
        check_launch_safety(ep4, with_k2(select_megamoe_config(ep4), 32, 2, False))

    def test_overlay_runs_the_guard(self):
        shp = shape(ep=4, same=3)
        cfg = with_k2(select_megamoe_config(shp), 16, 2, False)
        with self.assertRaisesRegex(ValueError, "k2_tail_reclaim"):
            select_transport_overlay(shp, cfg)

    def test_defaults_pass(self):
        for ep, same in ((4, 3), (8, 7), (16, 15)):
            for tokens in (1, 16, 32, 64, 128, 256, 512, 2048, 8192):
                with self.subTest(ep=ep, tokens=tokens):
                    shp = shape(tokens, ep=ep, same=same)
                    select_transport_overlay(shp, select_megamoe_config(shp))

    def test_table_entry_cannot_select_it(self):
        bad = dict(backend="p2p_split", k2_n=16, k2_warps=8, k2_launch_factor=2, k2_tail_reclaim="off")
        data = dict(schema=1, tables=[dict(model="dsv41", signature=SIG, topology="ep8_2numa",
                                           entries=[dict(tokens=16, knobs=bad)])])
        with patch.object(tuned_table, "_load", return_value=data):
            from .heuristic import _select_with_profiles
            _select_with_profiles.cache_clear()
            try:
                with self.assertRaises(ValueError):
                    shp = shape()
                    select_transport_overlay(shp, select_megamoe_config(shp))
            finally:
                _select_with_profiles.cache_clear()


if __name__ == "__main__":
    unittest.main()
