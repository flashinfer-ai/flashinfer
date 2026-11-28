"""CPU tests for the cross-NUMA backend rule, the hybrid capability check and
the generic decode preset."""

import os
import unittest
from unittest.mock import patch

from . import tuned_table
from .heuristic import (
    GENERIC_DECODE_KNOBS,
    MegaMoEHeuristicInput,
    generic_decode_knobs,
    select_megamoe_config,
)
from .runtime import driver_caps

MODELS = dict(dsv41=(5120, 4608, 6, 384), glm53=(4096, 4096, 8, 288), flash=(4096, 4096, 6, 256))


def shape(tokens, ep=8, same=3, cross=4, model="flash", sms=110):
    h, i, k, e = MODELS[model]
    return MegaMoEHeuristicInput(
        tokens_per_rank=tokens, hidden=h, intermediate=i, num_topk=k, num_total_experts=e,
        data_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=ep,
        ep_same_numa_peer_count=same, ep_cross_numa_peer_count=cross, num_sms=sms,
        sm_min_partition=8, sm_partition_alignment=8)


def hybrid(flag):
    driver_caps.hybrid_sm_split_available.cache_clear()
    return patch.dict(os.environ, {driver_caps.HYBRID_SM_SPLIT_ENV: "1" if flag else "0"})


class BackendRuleTests(unittest.TestCase):
    def tearDown(self):
        driver_caps.hybrid_sm_split_available.cache_clear()

    def test_rule_threshold(self):
        x = tuned_table.SPLIT_RULE_MAX_TOKENS
        with hybrid(True):
            self.assertEqual(tuned_table.preferred_ep8_backend(shape(x)), "p2p_split")
            self.assertEqual(tuned_table.preferred_ep8_backend(shape(x + 1)), "gin_hybrid")
        with hybrid(False):
            self.assertEqual(tuned_table.preferred_ep8_backend(shape(x + 1)), "p2p_split")

    def test_rule_independent_of_sm_count(self):
        x = tuned_table.SPLIT_RULE_MAX_TOKENS
        with hybrid(True):
            for sms in (84, 110, 156, 188):
                self.assertEqual(tuned_table.preferred_ep8_backend(shape(16, sms=sms)), "p2p_split")
                self.assertEqual(tuned_table.preferred_ep8_backend(shape(x * 2, sms=sms)), "gin_hybrid")

    def test_ep16_always_p2p_split(self):
        with hybrid(True):
            for tokens in (16, 4096, 24576):
                self.assertEqual(tuned_table.preferred_ep8_backend(shape(tokens, ep=16, same=3, cross=12)),
                                 "p2p_split")

    def test_folded_or_single_numa_has_no_rule(self):
        self.assertIsNone(tuned_table.preferred_ep8_backend(shape(16, same=7, cross=0)))
        self.assertIsNone(tuned_table.preferred_ep8_backend(shape(16, ep=4, same=3, cross=0)))

    def test_table_anchor_wins_and_hybrid_anchor_falls_back(self):
        data = dict(schema=1, tables=[dict(model="flash", signature=list(MODELS["flash"]), topology="ep8_2numa",
                                           entries=[dict(tokens=16, knobs=dict(backend="gin_hybrid"))])])
        with patch.object(tuned_table, "_load", return_value=data):
            with hybrid(True):
                self.assertEqual(tuned_table.preferred_ep8_backend(shape(16)), "gin_hybrid")
            with hybrid(False):
                self.assertEqual(tuned_table.preferred_ep8_backend(shape(16)), "p2p_split")
            # The table only applies on its device; other devices use the rule.
            with hybrid(True):
                self.assertEqual(tuned_table.preferred_ep8_backend(shape(16, sms=156)), "p2p_split")

    def test_too_many_experts_per_rank(self):
        many = MegaMoEHeuristicInput(
            tokens_per_rank=16, hidden=4096, intermediate=4096, num_topk=8, num_total_experts=1024,
            data_parallel_size=1, tensor_parallel_size=1, expert_parallel_size=8,
            ep_same_numa_peer_count=3, ep_cross_numa_peer_count=4, num_sms=110,
            sm_min_partition=8, sm_partition_alignment=8)
        with hybrid(True):
            self.assertEqual(tuned_table.preferred_ep8_backend(many), "gin_hybrid")
        with hybrid(False):
            self.assertIsNone(tuned_table.preferred_ep8_backend(many))


class GenericDecodePresetTests(unittest.TestCase):
    def test_applies_to_untabulated_ep8_split_decode(self):
        knobs = generic_decode_knobs(shape(16, same=7, cross=0))
        self.assertEqual({k: v for k, v in knobs.items() if k in GENERIC_DECODE_KNOBS}, GENERIC_DECODE_KNOBS)
        self.assertEqual(knobs["backend"], "p2p_split")
        self.assertNotIn("k1_stages", knobs)
        self.assertEqual(generic_decode_knobs(shape(128, same=7, cross=0))["k1_stages"], 3)
        # Other devices: no table there, the preset applies.
        self.assertIsNotNone(generic_decode_knobs(shape(16, same=7, cross=0, model="glm53", sms=156)))

    def test_out_of_scope(self):
        self.assertIsNone(generic_decode_knobs(shape(256, same=7, cross=0)))           # tokens
        data = dict(schema=1, tables=[dict(model="glm53", signature=list(MODELS["glm53"]), topology="ep8_2numa",
                                           entries=[dict(tokens=16, knobs=dict(backend="p2p_split"))])])
        with patch.object(tuned_table, "_load", return_value=data):
            self.assertIsNone(generic_decode_knobs(shape(16, same=7, cross=0, model="glm53")))  # tabulated
            self.assertIsNotNone(generic_decode_knobs(shape(16, same=7, cross=0)))  # other signature
        self.assertIsNone(generic_decode_knobs(shape(16, ep=4, same=3, cross=0)))      # EP4: candidate selector
        self.assertIsNone(generic_decode_knobs(shape(16)))                              # unfolded (hybrid)
        self.assertIsNone(generic_decode_knobs(shape(16, ep=16, same=15, cross=0)))    # EP16

    def test_switch_and_utility(self):
        s = shape(16, same=7, cross=0)
        on = select_megamoe_config(s)
        off = select_megamoe_config(s, generic_decode_preset=False)
        self.assertEqual(on.k2_tile[1], 64)
        self.assertEqual(on.weight_prefetch_ahead, 4)
        self.assertNotEqual(off.k2_tile[1], 64)
        self.assertTrue(tuned_table.utility_for(s, True, on))
        self.assertIsNone(tuned_table.utility_for(s, False, on))
        self.assertIsNone(tuned_table.utility_for(s, True, None))


if __name__ == "__main__":
    unittest.main()
