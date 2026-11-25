"""CPU tests for the transport overlay (push dispatch, t=0 FC1 weight
prefetch) and the fused graph-reset plan."""

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from . import tuned_table
from .heuristic import (
    K1_T0_PREFETCH_DEFAULT_K_TILES,
    PUSH_DISPATCH_MAX_TOKENS,
    MegaMoEHeuristicInput,
    TransportOverlay,
    policy_asdict,
    select_megamoe_config,
    select_transport_overlay,
)
from .runtime import fused_fill
from .token_comm import PushDispatchConfig


def shape(tokens, ep=4, same=3, cross=0, model="dsv41"):
    h, i, k, e = dict(dsv41=(5120, 4608, 6, 384), glm53=(4096, 4096, 8, 288))[model]
    return MegaMoEHeuristicInput(
        tokens_per_rank=tokens, hidden=h, intermediate=i, num_topk=k, num_total_experts=e,
        data_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=ep,
        ep_same_numa_peer_count=same, ep_cross_numa_peer_count=cross, num_sms=110,
        sm_min_partition=8, sm_partition_alignment=8)


def resolve(shp, overlay=None):
    return select_transport_overlay(shp, select_megamoe_config(shp), overlay)


class TransportOverlayTests(unittest.TestCase):
    def test_single_host_decode_defaults(self):
        for tokens in (1, 16, 32, 64):
            for shp in (shape(tokens), shape(tokens, ep=8, same=7)):
                cfg = resolve(shp)
                self.assertTrue(cfg.push_dispatch, (shp.expert_parallel_size, tokens))
                if cfg.k1_ready_queue:
                    self.assertEqual(cfg.k1_t0_prefetch, K1_T0_PREFETCH_DEFAULT_K_TILES)

    def test_defaults_off_above_threshold_and_ep16(self):
        cfg = resolve(shape(PUSH_DISPATCH_MAX_TOKENS * 2))
        self.assertFalse(cfg.push_dispatch)
        self.assertEqual(cfg.k1_t0_prefetch, 0)
        ep16 = resolve(shape(16, ep=16, same=15))
        self.assertEqual((ep16.push_dispatch, ep16.k1_t0_prefetch), (False, 0))
        ep16 = resolve(shape(16, ep=16, same=15), TransportOverlay(push_dispatch=True))
        self.assertEqual((ep16.push_dispatch, ep16.k1_t0_prefetch), (True, 0))

    def test_t0_prefetch_follows_push(self):
        cfg = resolve(shape(16), TransportOverlay(push_dispatch=False))
        self.assertEqual(cfg.k1_t0_prefetch, 0)
        cfg = resolve(shape(16, ep=8, same=7), TransportOverlay(push_dispatch=False))
        self.assertEqual(cfg.k1_t0_prefetch, 0)

    def test_explicit_overrides_win(self):
        cfg = resolve(shape(16), TransportOverlay(push_dispatch=False, k1_t0_prefetch=0))
        self.assertEqual((cfg.push_dispatch, cfg.k1_t0_prefetch), (False, 0))
        cfg = resolve(shape(512), TransportOverlay(push_dispatch=True, k1_t0_prefetch=8))
        self.assertTrue(cfg.push_dispatch)

    def test_table_knobs(self):
        knobs = dict(push_dispatch="off", k1_t0_prefetch=16)
        sig = [5120, 4608, 6, 384]
        data = dict(schema=1, tables=[dict(model="dsv41", signature=sig, topology="ep4_1numa",
                                           entries=[dict(tokens=16, knobs=knobs)])])
        with patch.object(tuned_table, "_load", return_value=data):
            cfg = select_transport_overlay(shape(16), select_megamoe_config(shape(16)))
            self.assertFalse(cfg.push_dispatch)
            if cfg.k1_ready_queue:
                self.assertEqual(cfg.k1_t0_prefetch, 16)
            # Explicit values still win over the table.
            cfg = select_transport_overlay(shape(16), cfg, TransportOverlay(push_dispatch=True))
            self.assertTrue(cfg.push_dispatch)
        # The knobs never become selection overrides.
        self.assertEqual(tuned_table.overrides_for(knobs), {})

    def test_overlay_invisible_to_policy_records(self):
        cfg = resolve(shape(16))
        base = replace(cfg, push_dispatch=False, k1_t0_prefetch=0)
        self.assertEqual(policy_asdict(cfg), policy_asdict(base))

    def test_invalid_values(self):
        cfg = resolve(shape(16))
        for bad in (-1, 65):
            with self.assertRaises(ValueError):
                replace(cfg, k1_t0_prefetch=bad)

    def test_config_is_hashable_constant(self):
        cfg = PushDispatchConfig(stride=94208, peers=(1, 2, 3), peer_mask=0b1110)
        self.assertEqual(hash(cfg), hash(PushDispatchConfig(stride=94208, peers=(1, 2, 3), peer_mask=0b1110)))


class FusedFillPlanTests(unittest.TestCase):
    def test_patterns_and_fallbacks(self):
        zeros = torch.zeros(64, dtype=torch.int32)
        ones = torch.zeros(16, dtype=torch.int64)
        minus = torch.zeros(10, dtype=torch.int32)
        odd = torch.zeros(3, dtype=torch.uint8)
        epoch = torch.zeros(4, dtype=torch.int64)
        strided = torch.zeros(8, 8, dtype=torch.int32)[:, 0]
        regions, rest = fused_fill.plan_regions(
            [(zeros, 0), (ones, -1), (minus, -1), (odd, 0), (epoch, 7), (strided, 0)])
        self.assertEqual([r[1:] for r in regions], [(256, 0), (128, 0xFFFFFFFF), (40, 0xFFFFFFFF)])
        self.assertEqual([t.data_ptr() for t, _ in rest], [odd.data_ptr(), epoch.data_ptr(), strided.data_ptr()])

    def test_front_path_switches_default_on(self):
        from . import token_comm
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEGA_SPLIT_FUSED_RESET", None)
            self.assertTrue(fused_fill.enabled())
            os.environ["MEGA_SPLIT_FUSED_RESET"] = "0"
            self.assertFalse(fused_fill.enabled())
        if os.environ.get("MEGA_SPLIT_FENCED_PUBLISH") is None:
            self.assertEqual(token_comm._PUBLISH_SEM, "relaxed")


class XnumaPackTests(unittest.TestCase):
    def test_rule(self):
        from .heuristic import XNUMA_PACK_MIN_TOKENS, select_xnuma_pack

        lo = XNUMA_PACK_MIN_TOKENS[16]
        self.assertFalse(select_xnuma_pack(lo - 1, 16))
        self.assertTrue(select_xnuma_pack(lo, 16))
        self.assertTrue(select_xnuma_pack(4096, 16))
        for ep in (2, 4, 8):
            if ep not in XNUMA_PACK_MIN_TOKENS:
                self.assertFalse(select_xnuma_pack(4096, ep))

    def test_overlay_default_and_override(self):
        lo = __import__("moe_sm120_nvfp4_split.heuristic", fromlist=["x"]).XNUMA_PACK_MIN_TOKENS[16]
        self.assertTrue(resolve(shape(lo, ep=16, same=15)).xnuma_pack)
        self.assertFalse(resolve(shape(lo, ep=8, same=7)).xnuma_pack)
        self.assertFalse(resolve(shape(lo, ep=16, same=15), TransportOverlay(xnuma_pack=False)).xnuma_pack)
        self.assertTrue(resolve(shape(16, ep=8, same=7), TransportOverlay(xnuma_pack=True)).xnuma_pack)

    def test_layout_pack_regions(self):
        from .transport.xnuma_layout import XnumaLayout

        base = dict(world=16, n_cross=12, experts_per_rank=24, tokens_per_rank=32, topk=6, hidden=5120)
        plain, packed = XnumaLayout(**base), XnumaLayout(**dict(base, pack=True))
        o, q = plain.offsets, packed.offsets
        for k in ("inbox", "txrec", "dsig", "done", "txd", "txf", "k2cnt", "k2tot", "k2meta", "fscr", "sscr"):
            self.assertEqual(o[k], q[k])
        self.assertEqual(o["end"], o["pstage"])
        self.assertEqual(q["end"] - q["pstage"], 12 * packed.pack_stride)
        self.assertEqual(packed.act_row_bytes, 2560)
        self.assertEqual(packed.sf_row_bytes, 320)
        self.assertEqual(XnumaLayout(**dict(base, hidden=4096, topk=8)).sf_row_bytes, 256)
        self.assertEqual(packed.pack_small_bytes, 32 * 4 + 32 * 6 * 4 + 32 * 320)
        self.assertEqual(packed.pack_stride % 4096, 0)
        self.assertGreaterEqual(packed.pack_stride, 32 * 2560 + packed.pack_small_bytes)


if __name__ == "__main__":
    unittest.main()
