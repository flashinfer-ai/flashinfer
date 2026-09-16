"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../flashinfer/production_debt.py",
)
spec = importlib.util.spec_from_file_location("flashinfer_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["flashinfer_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtAttentionGate = production_debt_mod.ProductionDebtAttentionGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtAttentionGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtAttentionGate(
            never_equate_intent_to_approval=True,
            max_acceptable_fdi=12.0,
        )

    def test_clean_attention_kernel_passes_readiness(self) -> None:
        report = self.gate.evaluate_attention_kernel(
            kernel_id="flashinfer_page_attention_gqa_h100",
            allocated_kv_page_bytes=16000000000,
            utilized_kv_page_bytes=16800000000,
            attention_latency_us=11.2,
            ragged_warp_divergences=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.fdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_attention_kernel_fails_debt(self) -> None:
        report = self.gate.evaluate_attention_kernel(
            kernel_id="uncalibrated_ragged_prefill_kernel",
            allocated_kv_page_bytes=16000000000,
            utilized_kv_page_bytes=45000000000,  # 2.81x page fragmentation sprawl
            attention_latency_us=85.0,  # High kernel latency
            ragged_warp_divergences=3,  # 3 ragged warp divergence stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.fdi_score, 50.0)
        self.assertIn("HIGH_PAGED_KV_FRAGMENTATION_SPRAWL_2.81X", report.critical_smells)
        self.assertIn("HIGH_ATTENTION_KERNEL_LATENCY_85.0US", report.critical_smells)
        self.assertIn("DETECTED_3_RAGGED_WARP_DIVERGENCE_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_ATTENTION_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_attention_kernel("kernel-1")
        self.gate.evaluate_attention_kernel("kernel-2")
        self.gate.evaluate_attention_kernel("kernel-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
