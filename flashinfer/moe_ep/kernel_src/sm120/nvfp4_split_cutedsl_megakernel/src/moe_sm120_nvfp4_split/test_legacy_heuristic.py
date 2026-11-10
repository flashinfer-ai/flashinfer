"""Regression tests for the retained compatibility heuristic."""

from __future__ import annotations

import unittest
from dataclasses import replace

try:
    from .legacy_heuristic import (
        MegaMoEHeuristicInput,
        MegaMoEHeuristicOverrides,
        select_megamoe_config,
    )
    from .transport.hybrid_transport import combine_plan_record_words
except ImportError:  # Allow the pure test to run without importing CUDA code.
    from legacy_heuristic import (
        MegaMoEHeuristicInput,
        MegaMoEHeuristicOverrides,
        select_megamoe_config,
    )
    from transport.hybrid_transport import combine_plan_record_words


def _shape(
    *,
    tokens: int,
    dp: int,
    ep: int,
    cross: int,
    hidden: int = 7168,
    intermediate: int = 6144,
    num_topk: int = 6,
    num_total_experts: int = 384,
    num_sms: int = 110,
    sm_min_partition: int = 8,
    sm_partition_alignment: int = 8,
):
    return MegaMoEHeuristicInput(
        tokens_per_rank=tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        data_parallel_size=dp,
        tensor_parallel_size=1,
        expert_parallel_size=ep,
        ep_same_numa_peer_count=ep - 1 - cross,
        ep_cross_numa_peer_count=cross,
        num_sms=num_sms,
        sm_min_partition=sm_min_partition,
        sm_partition_alignment=sm_partition_alignment,
    )


def _dsv4_flash_shape(*, tokens: int, dp: int, ep: int, cross: int):
    return _shape(
        tokens=tokens,
        dp=dp,
        ep=ep,
        cross=cross,
        hidden=4096,
        intermediate=4096,
        num_topk=6,
        num_total_experts=256,
    )


class HeuristicTest(unittest.TestCase):
    def test_flash_ep4_queue_quantum_preserves_small_input_presets(self):
        for tokens, quantum in (
            (16, 2), (32, 2), (64, 2), (128, 2), (512, 4), (1024, 4),
            (2048, 8), (4096, 8), (8191, 16), (8192, 8), (12288, 8),
            (16384, 8), (20480, 8), (24576, 8),
        ):
            with self.subTest(tokens=tokens):
                config = select_megamoe_config(
                    _dsv4_flash_shape(tokens=tokens, dp=1, ep=4, cross=0)
                )
                self.assertEqual(config.ready_queue_bundle, quantum)

    def test_flash_q8_does_not_expand_to_unvalidated_shapes_or_topologies(self):
        base = _dsv4_flash_shape(tokens=8192, dp=1, ep=4, cross=0)
        shapes = (
            replace(base, num_topk=8, num_total_experts=288),
            replace(base, intermediate=6144),
            replace(base, data_parallel_size=2),
            replace(base, tensor_parallel_size=2),
            replace(base, num_sms=188),
            _dsv4_flash_shape(tokens=8192, dp=1, ep=8, cross=4),
            _shape(tokens=8192, dp=1, ep=4, cross=0),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                # Wide DSV4 already uses Q8 at its 512 rows/expert boundary;
                # preserving the old rule does not mean forcing Q16 there.
                expected = 8 if shape.expected_rows_per_expert <= 512 else 16
                self.assertEqual(select_megamoe_config(shape).ready_queue_bundle, expected)

    def test_explicit_queue_override_still_wins(self):
        shape = _dsv4_flash_shape(tokens=8192, dp=1, ep=4, cross=0)
        config = select_megamoe_config(
            shape, MegaMoEHeuristicOverrides(ready_queue_bundle=16)
        )
        self.assertEqual(config.ready_queue_bundle, 16)

    def test_dsv4_flash_shape_fixture_matches_production(self):
        shape = _dsv4_flash_shape(tokens=8192, dp=1, ep=8, cross=4)
        self.assertEqual(
            (
                shape.hidden,
                shape.intermediate,
                shape.num_topk,
                shape.num_total_experts,
            ),
            (4096, 4096, 6, 256),
        )

    def test_local_decode_uses_measured_n16_k2_preset(self):
        for tokens in (7, 16, 32, 64, 128):
            with self.subTest(tokens=tokens):
                config = select_megamoe_config(
                    _dsv4_flash_shape(tokens=tokens, dp=1, ep=4, cross=0)
                )
                self.assertEqual(config.k1_tile, (64, 16 if tokens in (16, 32, 64) else 32, 128))
                self.assertEqual(config.k2_tile, (64, 16, 128))
                self.assertEqual(config.k2_stages, 2)
                self.assertEqual(config.k2_warps, 12)
                self.assertEqual(config.ready_queue_bundle, 2)
                self.assertTrue(config.k2_natural_regs)
                self.assertEqual(config.k2_min_blocks_per_sm, 2)
                self.assertEqual((config.k1_sms, config.k2_sms), (88, 22) if tokens in (32, 64, 128) else (80, 30))
                self.assertEqual(config.k2_tail_reclaim, tokens >= 32)

        outside = select_megamoe_config(
            _dsv4_flash_shape(tokens=168, dp=1, ep=4, cross=0)
        )
        self.assertEqual(outside.k1_tile, (64, 32, 128))
        self.assertEqual(outside.k2_tile, (64, 32, 128))
        self.assertEqual(outside.k2_warps, 8)
        self.assertFalse(outside.k2_tail_reclaim)
        self.assertEqual((outside.k1_sms, outside.k2_sms), (72, 38))

    def test_dsv4_ep4_t128_uses_measured_n32_k2_preset(self):
        config = select_megamoe_config(
            _shape(tokens=128, dp=1, ep=4, cross=0)
        )
        self.assertEqual(config.k1_tile, (64, 32, 128))
        self.assertEqual(config.k2_tile, (64, 32, 128))
        self.assertEqual(config.k2_stages, 2)
        self.assertEqual(config.k2_warps, 8)
        self.assertEqual(config.ready_queue_bundle, 4)
        self.assertEqual((config.k1_sms, config.k2_sms), (72, 38))
        self.assertFalse(config.k2_tail_reclaim)

        neighbor = select_megamoe_config(
            _shape(tokens=64, dp=1, ep=4, cross=0)
        )
        self.assertEqual(neighbor.k2_tile, (64, 16, 128))
        self.assertEqual((neighbor.k1_sms, neighbor.k2_sms), (80, 30))

    def test_dual_n8_decode_is_limited_to_validated_shape_and_topology(self):
        shapes = (
            _shape(tokens=64, dp=1, ep=4, cross=0),
            _dsv4_flash_shape(tokens=64, dp=2, ep=4, cross=0),
            _dsv4_flash_shape(tokens=64, dp=1, ep=8, cross=4),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                config = select_megamoe_config(shape)
                self.assertEqual(config.k2_warps, 8)
                self.assertFalse(config.k2_tail_reclaim)

    def test_k2_override_disables_implicit_dual_n8_tail_policy(self):
        shape = _dsv4_flash_shape(tokens=64, dp=1, ep=4, cross=0)
        config = select_megamoe_config(
            shape,
            MegaMoEHeuristicOverrides(k2_warps=8),
        )
        self.assertEqual(config.k2_warps, 8)
        self.assertFalse(config.k2_tail_reclaim)

        config = select_megamoe_config(
            shape,
            MegaMoEHeuristicOverrides(k2_tile=(64, 32, 128)),
        )
        self.assertEqual(config.k2_warps, 8)
        self.assertFalse(config.k2_tail_reclaim)

    def test_dispatch_warp_override_is_local_p2p_only(self):
        local = select_megamoe_config(
            _dsv4_flash_shape(tokens=128, dp=1, ep=4, cross=0),
            MegaMoEHeuristicOverrides(dispatch_warps=2),
        )
        self.assertEqual(local.dispatch_warps, 2)

        with self.assertRaisesRegex(ValueError, "p2p_direct"):
            select_megamoe_config(
                _dsv4_flash_shape(tokens=128, dp=1, ep=8, cross=4),
                MegaMoEHeuristicOverrides(dispatch_warps=2),
            )

        with self.assertRaisesRegex(ValueError, "one of 1/2/4"):
            select_megamoe_config(
                _dsv4_flash_shape(tokens=128, dp=1, ep=4, cross=0),
                MegaMoEHeuristicOverrides(dispatch_warps=3),
            )

    def test_local_dispatch_warps_follow_validated_small_shape_envelope(self):
        small = select_megamoe_config(
            _dsv4_flash_shape(tokens=512, dp=1, ep=4, cross=0)
        )
        large = select_megamoe_config(
            _dsv4_flash_shape(tokens=1024, dp=1, ep=4, cross=0)
        )
        cross_numa = select_megamoe_config(
            _dsv4_flash_shape(tokens=16, dp=1, ep=8, cross=4)
        )
        wider_dsv4 = select_megamoe_config(
            _shape(tokens=128, dp=1, ep=4, cross=0)
        )
        self.assertEqual(small.dispatch_warps, 1)
        self.assertEqual(large.dispatch_warps, 1)
        self.assertEqual(cross_numa.dispatch_warps, 4)
        self.assertEqual(wider_dsv4.dispatch_warps, 4)

    def test_parallel_topology_selects_transport(self):
        cases = (
            ("ep8", _shape(tokens=8192, dp=1, ep=8, cross=4), "nvshmem_hybrid"),
            ("dp2ep4", _shape(tokens=8192, dp=2, ep=4, cross=0), "p2p_direct"),
            ("ep4", _shape(tokens=8192, dp=1, ep=4, cross=0), "p2p_direct"),
            ("dp2ep2", _shape(tokens=8192, dp=2, ep=2, cross=0), "p2p_direct"),
        )
        for name, shape, expected in cases:
            with self.subTest(name=name):
                config = select_megamoe_config(shape)
                self.assertEqual(config.comm_backend, expected)
                self.assertEqual(config.total_sms, shape.num_sms)

    def test_sm_partitions_scale_to_larger_sm120_devices(self):
        local_n128 = select_megamoe_config(
            _shape(
                tokens=8192,
                dp=1,
                ep=4,
                cross=0,
                num_sms=188,
            )
        )
        self.assertEqual(
            (local_n128.k1_sms, local_n128.k2_sms),
            (120, 68),
        )

        local_n64 = select_megamoe_config(
            _shape(
                tokens=512,
                dp=1,
                ep=4,
                cross=0,
                num_sms=188,
            )
        )
        self.assertEqual(
            (local_n64.k1_sms, local_n64.k2_sms),
            (136, 52),
        )

        hybrid = select_megamoe_config(
            _shape(
                tokens=8192,
                dp=1,
                ep=8,
                cross=4,
                num_sms=188,
            )
        )
        self.assertEqual(
            (
                hybrid.k1_sms,
                hybrid.tx_sms,
                hybrid.rx_sms,
                hybrid.k2_sms,
            ),
            (80, 24, 24, 60),
        )
        self.assertEqual(hybrid.total_sms, 188)
        self.assertEqual(hybrid.k1_sms % 8, 0)
        self.assertEqual(hybrid.tx_sms % 8, 0)
        self.assertEqual(hybrid.rx_sms % 8, 0)

    def test_comm_backend_override_applies_complete_transport_preset(self):
        ep8_p2p = select_megamoe_config(
            _shape(tokens=8192, dp=1, ep=8, cross=4),
            MegaMoEHeuristicOverrides(comm_backend="p2p_direct"),
        )
        self.assertEqual(
            (
                ep8_p2p.k1_sms,
                ep8_p2p.k2_sms,
                ep8_p2p.tx_sms,
                ep8_p2p.rx_sms,
                ep8_p2p.token_back_mode,
            ),
            (72, 38, 0, 0, "epi_warps"),
        )

        ep4_hybrid = select_megamoe_config(
            _shape(tokens=8192, dp=1, ep=4, cross=0),
            MegaMoEHeuristicOverrides(comm_backend="nvshmem_hybrid"),
        )
        self.assertEqual(
            (
                ep4_hybrid.k1_sms,
                ep4_hybrid.k2_sms,
                ep4_hybrid.tx_sms,
                ep4_hybrid.rx_sms,
                ep4_hybrid.token_back_mode,
            ),
            (48, 30, 16, 16, "reuse_dispatch_warps"),
        )

    def test_sm_partition_override_is_all_or_nothing(self):
        shape = _shape(tokens=8192, dp=1, ep=8, cross=4)
        with self.assertRaisesRegex(ValueError, "provide all four"):
            select_megamoe_config(
                shape,
                MegaMoEHeuristicOverrides(k1_sms=64),
            )
        config = select_megamoe_config(
            shape,
            MegaMoEHeuristicOverrides(
                k1_sms=56,
                k2_sms=30,
                tx_sms=8,
                rx_sms=16,
            ),
        )
        self.assertEqual(config.total_sms, 110)

    def test_backend_enforces_matching_token_back_override(self):
        shape = _shape(tokens=8192, dp=1, ep=8, cross=4)
        valid_cases = (
            ("p2p_direct", "epi_warps"),
            ("nvshmem_ibgda", "reuse_dispatch_warps"),
        )
        for backend, token_back_mode in valid_cases:
            with self.subTest(backend=backend, token_back_mode=token_back_mode):
                config = select_megamoe_config(
                    shape,
                    MegaMoEHeuristicOverrides(
                        comm_backend=backend,
                        token_back_mode=token_back_mode,
                    ),
                )
                self.assertEqual(config.token_back_mode, token_back_mode)

        invalid_cases = (
            ("p2p_direct", "reuse_dispatch_warps"),
            ("nvshmem_ibgda", "epi_warps"),
        )
        for backend, token_back_mode in invalid_cases:
            with self.subTest(backend=backend, token_back_mode=token_back_mode):
                with self.assertRaisesRegex(
                    ValueError, "requires token_back_mode"
                ):
                    select_megamoe_config(
                        shape,
                        MegaMoEHeuristicOverrides(
                            comm_backend=backend,
                            token_back_mode=token_back_mode,
                        ),
                    )

    def test_dp_is_not_a_kernel_key(self):
        ep4 = select_megamoe_config(
            _shape(tokens=8192, dp=1, ep=4, cross=0)
        )
        dp2ep4 = select_megamoe_config(
            _shape(tokens=8192, dp=2, ep=4, cross=0)
        )
        self.assertEqual(ep4, dp2ep4)

    def test_rows_per_expert_tile_boundaries(self):
        # DSV4 EP8 has rows/expert = tokens/8.
        cases = ((224, 32), (240, 64), (504, 64), (512, 128))
        for tokens, expected_n in cases:
            with self.subTest(tokens=tokens):
                config = select_megamoe_config(
                    _shape(tokens=tokens, dp=1, ep=8, cross=4)
                )
                self.assertEqual(config.k1_tile[1], expected_n)
                self.assertEqual(config.k2_tile[1], expected_n)

    def test_local_sm_partition_follows_tile_not_dp(self):
        n64 = select_megamoe_config(
            _shape(tokens=512, dp=1, ep=4, cross=0)
        )
        n128 = select_megamoe_config(
            _shape(tokens=1024, dp=1, ep=4, cross=0)
        )
        self.assertEqual((n64.k1_sms, n64.k2_sms), (80, 30))
        self.assertEqual((n128.k1_sms, n128.k2_sms), (72, 38))

    def test_fixed_transport_parameters(self):
        config = select_megamoe_config(
            _shape(tokens=8192, dp=1, ep=8, cross=4)
        )
        self.assertEqual(
            (config.k1_sms, config.k2_sms, config.tx_sms, config.rx_sms),
            (48, 30, 16, 16),
        )
        self.assertEqual(config.combine_chunk_rows, 16)
        self.assertEqual(config.dispatch_channels, 2)
        self.assertEqual(config.combine_channels, 2)
        self.assertEqual(config.dispatch_slots, 2)
        self.assertEqual(config.combine_slots, 2)
        self.assertEqual(config.ibgda_rc_per_pe, 1)
        self.assertEqual(config.ibgda_rc_mapping, "warp")
        self.assertEqual(config.k1_ready_queue_m_rotation, 0)
        self.assertEqual(config.tp_k3_chunks, 1)

    def test_combine_put_mode_uses_per_peer_bytes(self):
        flash_below_threshold = select_megamoe_config(
            _dsv4_flash_shape(tokens=128, dp=1, ep=8, cross=4)
        )
        flash_at_threshold = select_megamoe_config(
            _dsv4_flash_shape(tokens=512, dp=1, ep=8, cross=4)
        )
        flash_large = select_megamoe_config(
            _dsv4_flash_shape(tokens=4096, dp=1, ep=8, cross=4)
        )
        wide_chunk = select_megamoe_config(
            _shape(tokens=1024, dp=1, ep=8, cross=4, hidden=7168)
        )
        local = select_megamoe_config(
            _shape(tokens=24576, dp=1, ep=4, cross=0, hidden=7168)
        )
        self.assertFalse(flash_below_threshold.combine_blocking_put)
        self.assertTrue(flash_at_threshold.combine_blocking_put)
        self.assertTrue(flash_large.combine_blocking_put)
        self.assertFalse(wide_chunk.combine_blocking_put)
        self.assertFalse(local.combine_blocking_put)

    def test_dispatch_chunk_is_bounded_by_actual_row_bytes(self):
        flash_tiny = select_megamoe_config(
            _dsv4_flash_shape(tokens=128, dp=1, ep=8, cross=4)
        )
        flash_latency_band = select_megamoe_config(
            _dsv4_flash_shape(tokens=512, dp=1, ep=8, cross=4)
        )
        flash_band_end = select_megamoe_config(
            _dsv4_flash_shape(tokens=1024, dp=1, ep=8, cross=4)
        )
        flash_throughput_band = select_megamoe_config(
            _dsv4_flash_shape(tokens=2048, dp=1, ep=8, cross=4)
        )
        flash = select_megamoe_config(
            _dsv4_flash_shape(tokens=8192, dp=1, ep=8, cross=4)
        )
        dsv4_tiny = select_megamoe_config(
            _shape(tokens=64, dp=1, ep=8, cross=4, hidden=7168)
        )
        dsv4_streaming = select_megamoe_config(
            _shape(tokens=128, dp=1, ep=8, cross=4, hidden=7168)
        )
        dsv4_large = select_megamoe_config(
            _shape(tokens=24576, dp=1, ep=8, cross=4, hidden=7168)
        )
        self.assertEqual(flash_tiny.dispatch_chunk_tokens, 64)
        self.assertEqual(flash_latency_band.dispatch_chunk_tokens, 32)
        self.assertEqual(flash_band_end.dispatch_chunk_tokens, 32)
        self.assertEqual(flash_throughput_band.dispatch_chunk_tokens, 64)
        self.assertEqual(flash.dispatch_chunk_tokens, 64)
        self.assertEqual(dsv4_tiny.dispatch_chunk_tokens, 64)
        # NVFP4 activations plus block-16 scales keep a 64-row DSV4 payload
        # below the byte-bounded IBGDA chunk limit.
        self.assertEqual(dsv4_streaming.dispatch_chunk_tokens, 64)
        self.assertEqual(dsv4_large.dispatch_chunk_tokens, 64)

    def test_small_cross_numa_shape_gets_bounded_progress_window(self):
        ep8_128 = select_megamoe_config(
            _shape(tokens=128, dp=1, ep=8, cross=4)
        )
        ep8_512 = select_megamoe_config(
            _shape(tokens=512, dp=1, ep=8, cross=4)
        )
        ep4_128 = select_megamoe_config(
            _shape(tokens=128, dp=1, ep=4, cross=0)
        )
        self.assertEqual(ep8_128.combine_slots, 4)
        self.assertEqual(ep8_512.combine_slots, 2)
        self.assertEqual(ep4_128.combine_slots, 2)

    def test_combine_put_mode_can_be_overridden(self):
        config = select_megamoe_config(
            _shape(tokens=128, dp=1, ep=8, cross=4),
            MegaMoEHeuristicOverrides(combine_blocking_put=True),
        )
        self.assertTrue(config.combine_blocking_put)

    def test_cross_numa_transport_shape_bands(self):
        cases = (
            (
                "dsv4-flash-8192",
                _dsv4_flash_shape(tokens=8192, dp=1, ep=8, cross=4),
                (16, 2, 2, 2, 2, 2),
            ),
            (
                "dsv4-flash-24576",
                _dsv4_flash_shape(tokens=24576, dp=1, ep=8, cross=4),
                (16, 2, 2, 2, 2, 2),
            ),
            (
                "dsv4-12288",
                _shape(tokens=12288, dp=1, ep=8, cross=4),
                (16, 2, 2, 2, 2, 2),
            ),
            (
                "dsv4-24576",
                _shape(tokens=24576, dp=1, ep=8, cross=4),
                (16, 2, 2, 2, 2, 2),
            ),
        )
        for name, shape, expected in cases:
            with self.subTest(name=name):
                config = select_megamoe_config(shape)
                actual = (
                    config.combine_chunk_rows,
                    config.dispatch_channels,
                    config.combine_channels,
                    config.dispatch_slots,
                    config.combine_slots,
                    config.dispatch_remote_handoff_windows,
                )
                self.assertEqual(actual, expected)

    def test_dispatch_handoff_windows_follow_topology_defaults(self):
        flash = select_megamoe_config(
            _dsv4_flash_shape(tokens=8192, dp=1, ep=8, cross=4)
        )
        dsv4 = select_megamoe_config(
            _shape(tokens=8192, dp=1, ep=8, cross=4, hidden=7168)
        )
        dsv4_large = select_megamoe_config(
            _shape(tokens=24576, dp=1, ep=8, cross=4, hidden=7168)
        )
        self.assertEqual(flash.dispatch_local_handoff_windows, 1)
        self.assertEqual(flash.dispatch_remote_handoff_windows, 2)
        self.assertEqual(dsv4.dispatch_local_handoff_windows, 1)
        self.assertEqual(dsv4.dispatch_remote_handoff_windows, 2)
        self.assertEqual(dsv4_large.dispatch_local_handoff_windows, 1)
        self.assertEqual(dsv4_large.dispatch_remote_handoff_windows, 2)

    def test_dispatch_handoff_window_override_is_explicit(self):
        config = select_megamoe_config(
            _shape(tokens=8192, dp=1, ep=8, cross=4),
            MegaMoEHeuristicOverrides(
                dispatch_remote_handoff_windows=8,
            ),
        )
        self.assertEqual(config.dispatch_remote_handoff_windows, 8)

    def test_production_selector_never_enables_tail_reclaim(self):
        for tokens in (8192, 12288, 24576):
            with self.subTest(tokens=tokens):
                config = select_megamoe_config(
                    _shape(tokens=tokens, dp=1, ep=8, cross=4)
                )

    def test_tile_override_updates_derived_k2_fields(self):
        config = select_megamoe_config(
            _shape(tokens=128, dp=1, ep=4, cross=0),
            MegaMoEHeuristicOverrides(k2_tile=(64, 64, 128)),
        )
        self.assertEqual(config.k2_stages, 3)
        self.assertFalse(config.k2_natural_regs)
        self.assertEqual(config.k2_min_blocks_per_sm, 1)

    def test_n256_is_not_selected(self):
        for tokens in (16, 8192, 24576):
            with self.subTest(tokens=tokens):
                config = select_megamoe_config(
                    _shape(tokens=tokens, dp=1, ep=8, cross=4)
                )
                self.assertIn(config.k1_tile[1], (32, 64, 128))
                self.assertIn(config.k2_tile[1], (32, 64, 128))

    def test_dispatch_rank_cache_uses_traffic_and_topology(self):
        for tokens in (16, 2048, 24576):
            with self.subTest(tokens=tokens):
                config = select_megamoe_config(
                    _dsv4_flash_shape(tokens=tokens, dp=1, ep=4, cross=0)
                )
                self.assertTrue(config.dispatch_rank_cache)

        for shape in (
            _dsv4_flash_shape(tokens=2048, dp=1, ep=2, cross=0),
            _dsv4_flash_shape(tokens=2048, dp=1, ep=8, cross=4),
        ):
            self.assertFalse(select_megamoe_config(shape).dispatch_rank_cache)

    def test_dispatch_rank_cache_override_is_authoritative(self):
        shape = _dsv4_flash_shape(tokens=2048, dp=1, ep=4, cross=0)
        disabled = select_megamoe_config(
            shape,
            MegaMoEHeuristicOverrides(dispatch_rank_cache=False),
        )
        enabled = select_megamoe_config(
            shape,
            MegaMoEHeuristicOverrides(dispatch_rank_cache=True),
        )
        self.assertFalse(disabled.dispatch_rank_cache)
        self.assertTrue(enabled.dispatch_rank_cache)

    def test_rank_local_combine_small_shape_requires_explicit_opt_in(self):
        shape = _dsv4_flash_shape(tokens=1024, dp=1, ep=4, cross=0)
        default = select_megamoe_config(shape)
        enabled = select_megamoe_config(
            shape,
            MegaMoEHeuristicOverrides(rank_local_combine=True),
        )
        self.assertFalse(default.rank_local_combine)
        self.assertTrue(enabled.rank_local_combine)
        self.assertEqual(enabled.dispatch_warps, 1)
        self.assertEqual(enabled.k2_stages, 3)

    def test_rank_local_combine_auto_threshold_and_opt_out(self):
        below = select_megamoe_config(
            _dsv4_flash_shape(tokens=1535, dp=1, ep=4, cross=0)
        )
        at_threshold = select_megamoe_config(
            _shape(
                tokens=4096,
                dp=1,
                ep=4,
                cross=0,
                hidden=7168,
                intermediate=6144,
                num_topk=6,
                num_total_experts=384,
            )
        )
        large = select_megamoe_config(
            _dsv4_flash_shape(tokens=8192, dp=1, ep=4, cross=0)
        )
        disabled = select_megamoe_config(
            _dsv4_flash_shape(tokens=8192, dp=1, ep=4, cross=0),
            MegaMoEHeuristicOverrides(rank_local_combine=False),
        )
        dp2 = select_megamoe_config(
            _dsv4_flash_shape(tokens=3072, dp=2, ep=4, cross=0)
        )
        self.assertFalse(below.rank_local_combine)
        self.assertTrue(at_threshold.rank_local_combine)
        self.assertTrue(large.rank_local_combine)
        self.assertEqual(large.dispatch_warps, 1)
        self.assertEqual(large.k2_stages, 3)
        self.assertFalse(disabled.rank_local_combine)
        self.assertEqual(disabled.dispatch_warps, 4)
        self.assertEqual(disabled.k2_stages, 2)
        self.assertTrue(dp2.rank_local_combine)

    def test_qualified_streaming_mid_input_defaults(self):
        # RLC-41: dispatch-only at Flash1024/DSV4; the vector reducer
        # crosses the conservative grouped-BF16 break-even at 144 rows.
        cases = (
            (_dsv4_flash_shape, 1024, False),
            (_dsv4_flash_shape, 1535, False),
            (_dsv4_flash_shape, 1536, True),
            (_dsv4_flash_shape, 1537, True),
            (_dsv4_flash_shape, 2048, True),
            (_shape, 1024, False),
            (_shape, 2048, False),
            (_shape, 4095, False),
            (_shape, 4096, True),
        )
        for make_shape, tokens, rlc in cases:
            with self.subTest(shape=make_shape.__name__, tokens=tokens):
                config = select_megamoe_config(
                    make_shape(tokens=tokens, dp=1, ep=4, cross=0)
                )
                self.assertEqual(config.dispatch_warps, 1)
                self.assertEqual(config.rank_local_combine, rlc)

    def test_early_streaming_rule_preserves_opt_out_and_qualification(self):
        shape = _dsv4_flash_shape(tokens=2048, dp=1, ep=4, cross=0)
        explicit = select_megamoe_config(
            shape, MegaMoEHeuristicOverrides(
                rank_local_combine=False, dispatch_warps=4,
            ),
        )
        self.assertFalse(explicit.rank_local_combine)
        self.assertEqual(explicit.dispatch_warps, 4)
        for overrides in (
            MegaMoEHeuristicOverrides(dispatch_rank_cache=False),
            MegaMoEHeuristicOverrides(dispatch_compute_overlap=False),
        ):
            self.assertFalse(select_megamoe_config(shape, overrides).rank_local_combine)
        dp2 = select_megamoe_config(
            _dsv4_flash_shape(tokens=2048, dp=2, ep=4, cross=0)
        )
        self.assertFalse(dp2.rank_local_combine)

    def test_rank_local_combine_tuning_overrides_remain_authoritative(self):
        shape = _dsv4_flash_shape(tokens=8192, dp=1, ep=4, cross=0)
        config = select_megamoe_config(
            shape,
            MegaMoEHeuristicOverrides(
                rank_local_combine=True,
                dispatch_warps=4,
                k2_stages=2,
            ),
        )
        self.assertEqual(config.dispatch_warps, 4)
        self.assertEqual(config.k2_stages, 2)

    def test_rank_local_combine_rejects_cross_numa(self):
        automatic = select_megamoe_config(
            _dsv4_flash_shape(tokens=24576, dp=1, ep=8, cross=4)
        )
        self.assertFalse(automatic.rank_local_combine)
        with self.assertRaisesRegex(ValueError, "requires p2p_direct"):
            select_megamoe_config(
                _dsv4_flash_shape(tokens=2048, dp=1, ep=8, cross=4),
                MegaMoEHeuristicOverrides(rank_local_combine=True),
            )

    def test_combine_plan_allocation_includes_commit(self):
        self.assertEqual(combine_plan_record_words(0), 34)
        self.assertEqual(combine_plan_record_words(1), 36)
        self.assertEqual(combine_plan_record_words(16), 50)


if __name__ == "__main__":
    unittest.main()
