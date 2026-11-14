"""Evidence budgeting is independent of GitHub, model access, and GPU hardware."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/pr_checks"))
from scope_evidence import EvidenceContext


class EvidenceContextTests(unittest.TestCase):
    def build(
        self, changed=(), inventory=(), candidates=(), ci_files=None, limit=20_000
    ):
        return EvidenceContext(
            {"body": "PR description"},
            {"head_sha": "head"},
            changed,
            inventory,
            candidates,
            ci_files or {},
            limit,
        )

    def test_complete_changed_metadata_and_nearby_tests_take_priority(self):
        tests = [f"tests/aaa/test_unrelated_{i}.py" for i in range(1000)]
        tests += ["tests/zebra/test_changed.py", "tests/zebra/test_neighbour.py"]
        changed = [
            {
                "filename": "tests/zebra/test_changed.py",
                "status": "renamed",
                "previous_filename": "tests/old/test_changed.py",
                "patch": "x" * 30_000,
            }
        ]
        budget = self.build(changed, tests, tests)
        context = budget.context
        self.assertEqual(tests[-2:], context["candidate_test_files"][:2])
        record = context["changed_files"][0]
        self.assertEqual("tests/old/test_changed.py", record["previous_filename"])
        self.assertIsNone(record["patch"])
        self.assertTrue(record["patch_omitted"])
        self.assertEqual(1, context["omitted_evidence"]["patches"])
        self.assertEqual(0, context["omitted_evidence"]["changed_files"])

    def test_full_source_bodies_and_omission_metadata_stay_bounded(self):
        budget = self.build()
        body = 'def test_unicode():\n    assert "☃" != "\\"\n'
        budget.add_source("tests/test_complete.py", body)
        for index in range(2000):
            budget.add_source(
                "tests/" + "long_name_" * 15 + str(index) + ".py", "x" * 20_000
            )
        self.assertEqual(body, budget.context["source_files"]["tests/test_complete.py"])
        self.assertGreater(budget.context["omitted_evidence"]["unavailable_paths"], 0)
        self.assertLessEqual(len(json.dumps(budget.context)), budget.limit)
        self.assertEqual(len(json.dumps(budget.context)), budget.chars)

    def test_requested_files_can_use_reserved_space_without_stale_omissions(self):
        budget = self.build()
        body = "# inspected body\n" * 700
        budget.add_source("tests/test_requested.py", body, reserve=8000)
        self.assertNotIn("tests/test_requested.py", budget.context["source_files"])
        budget.add_source("tests/test_requested.py", body)
        self.assertEqual(
            body, budget.context["source_files"]["tests/test_requested.py"]
        )
        self.assertNotIn(
            "tests/test_requested.py", budget.context["unavailable_evidence"]
        )
        self.assertEqual(len(json.dumps(budget.context)), budget.chars)

    def test_omission_records_do_not_spend_reserved_source_space(self):
        budget = self.build(limit=60_000)
        budget.add_source("tests/test_seed.py", "x" * 30_000, reserve=20_000)
        for index in range(200):
            budget.unavailable(
                "tests/" + "long_name_" * 15 + str(index) + ".py", "context budget"
            )
        body = "x" * 20_000
        budget.add_source("tests/test_requested.py", body)
        self.assertEqual(
            body, budget.context["source_files"]["tests/test_requested.py"]
        )
        self.assertGreater(budget.context["omitted_evidence"]["unavailable_paths"], 0)
        self.assertLessEqual(len(json.dumps(budget.context)), budget.limit)
        self.assertEqual(len(json.dumps(budget.context)), budget.chars)

    def test_trusted_ci_bodies_are_never_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "Mandatory CI evidence"):
            self.build(ci_files={"ci.yml": "x" * 30_000})


if __name__ == "__main__":
    unittest.main()
