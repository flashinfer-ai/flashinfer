"""Source-to-diff anchoring contracts migrated from the Node suite."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr_checks.api_anchors import select_anchor


class AnchorTest(unittest.TestCase):
    def test_multiline_changed_argument(self):
        finding = {
            "anchors": [["api.py", "RIGHT", 200, 220], ["api.py", "LEFT", 200, 221]]
        }
        files = [
            {
                "filename": "api.py",
                "patch": "@@ -208,3 +208,3 @@\n x,\n-old=0,\n+new=0,\n ):",
            }
        ]
        self.assertEqual(
            select_anchor(finding, files),
            {"path": "api.py", "line": 209, "side": "RIGHT"},
        )

    def test_removed_parameter_prefers_changed_left_to_right_context(self):
        finding = {"anchors": [["api.py", "RIGHT", 1, 6], ["api.py", "LEFT", 1, 7]]}
        self.assertEqual(
            select_anchor(
                finding,
                [
                    {
                        "filename": "api.py",
                        "patch": "@@ -3,3 +3,2 @@\n x,\n-removed=0,\n ):",
                    }
                ],
            ),
            {"path": "api.py", "line": 4, "side": "LEFT"},
        )

    def test_deleted_api_uses_left(self):
        self.assertEqual(
            select_anchor(
                {"anchors": [["api.py", "LEFT", 10, 12]]},
                [
                    {
                        "filename": "api.py",
                        "patch": "@@ -10,3 +9,0 @@\n-def foo(\n- x,\n-): pass",
                    }
                ],
            ),
            {"path": "api.py", "line": 10, "side": "LEFT"},
        )

    def test_rename_maps_old_path(self):
        self.assertEqual(
            select_anchor(
                {"anchors": [["old.py", "LEFT", 10, 12]]},
                [
                    {
                        "filename": "new.py",
                        "previous_filename": "old.py",
                        "patch": "@@ -10,2 +10,2 @@\n-old\n+new\n context",
                    }
                ],
            ),
            {"path": "new.py", "line": 10, "side": "LEFT"},
        )

    def test_unrelated_line_uses_explicit_file_comment(self):
        self.assertEqual(
            select_anchor(
                {"anchors": [["api.py", "RIGHT", 2, 8]]},
                [{"filename": "api.py", "patch": "@@ -100 +100 @@\n-old\n+new"}],
            ),
            {"path": "api.py", "subject_type": "file"},
        )

    def test_missing_patch_uses_file_comment(self):
        self.assertEqual(
            select_anchor({"path": "api.py", "line": 12}, [{"filename": "api.py"}]),
            {"path": "api.py", "subject_type": "file"},
        )

    def test_unchanged_file_has_no_comment(self):
        self.assertIsNone(
            select_anchor({"path": "api.py", "line": 12}, [{"filename": "other.py"}])
        )

    def test_real_missed_inline_regressions(self):
        fixtures = json.loads(
            Path(__file__).with_name("api_inline_real_prs.json").read_text()
        )
        self.assertEqual(len(fixtures), 3)
        for case in fixtures:
            with self.subTest(pr=case["upstream_pr"]):
                anchor = select_anchor(case["finding"], [case["file"]])
                self.assertTrue(anchor.get("line"))
                self.assertNotIn("subject_type", anchor)
