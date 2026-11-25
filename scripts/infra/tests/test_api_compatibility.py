"""Policy regressions for the merge-blocking API classification."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_check_pr_api_diff import api_function, compare_snapshots
import check_pr_api_diff as checker


class ClassificationTest(unittest.TestCase):
    def test_signature_policy(self):
        cases = [
            ("x", "x, y", "BREAKING"),
            ("x, y=0", "x, y", "BREAKING"),
            ("x, y", "x, *, y", "BREAKING"),
            ("x, y", "y, x", "BREAKING"),
            ("x, y=0", "x", "BREAKING"),
            ("x, y=0", "x, z=0", "BREAKING"),
            ("x", "x, /", "BREAKING"),
            ("x, /", "renamed, /", "NON_BREAKING"),
            ("x, /", "x", "NON_BREAKING"),
            ("x, y=0", "x, *, y=0", "LIKELY_BREAKING"),
            ("x, y=0", "x, z=0, y=0", "LIKELY_BREAKING"),
            ("x, y", "x, z=0, y=0", "BREAKING"),
            ("x, y=1", "x, y=None", "LIKELY_BREAKING"),
            ("x, *args", "x, mode=None, *args", "LIKELY_BREAKING"),
            ("x, **kwargs", "x, *, mode=None, **kwargs", "LIKELY_BREAKING"),
            ("x, y=0", "x, **kwargs", "LIKELY_BREAKING"),
            ("x, *, y=0", "x, *, z=0, y=0", "NON_BREAKING"),
            ("x", "x, y=0", "NON_BREAKING"),
            ("x: Tensor", "x: Tensor | None", "NON_BREAKING"),
            ("x: Tensor | None", "x: Tensor", "LIKELY_BREAKING"),
        ]
        for old, new, expected in cases:
            with self.subTest(old=old, new=new):
                self.assertEqual(
                    checker.classify_signature(
                        api_function(f"def api({old})"),
                        api_function(f"def api({new})"),
                    ).classification,
                    expected,
                )

    def test_return_widening_is_non_breaking(self):
        for result in (
            "Union[Tensor, Tuple[Tensor, Tensor]]",
            "Tensor | tuple[Tensor, Tensor]",
        ):
            finding = checker.classify_signature(
                api_function("def api(x) -> Tensor"),
                api_function(f"def api(x, *, return_logprob=False) -> {result}"),
            )
            self.assertEqual(finding.classification, "NON_BREAKING")
        finding = checker.classify_signature(
            api_function("def api(x) -> Tensor | None"),
            api_function("def api(x) -> Tensor"),
        )
        self.assertEqual(finding.classification, "LIKELY_BREAKING")

    def test_decorator_only_removal_is_warning_but_deletion_is_error(self):
        old = {"flashinfer/example.py": "@flashinfer_api\ndef api(x): pass\n"}
        for source, classification in [
            ("def api(x): pass\n", "LIKELY_BREAKING"),
            ("", "BREAKING"),
        ]:
            finding = compare_snapshots(old, {"flashinfer/example.py": source})[0]
            self.assertEqual(finding.classification, classification)

    def test_malformed_source_is_not_a_clean_scan(self):
        with self.assertRaises(SyntaxError):
            compare_snapshots(
                {"flashinfer/example.py": "def broken(:"},
                {"flashinfer/example.py": "x = 1"},
            )

    def test_failure_report_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(checker.__file__)),
                    "--base",
                    "missing-ref",
                    "--head",
                    "HEAD",
                    "--fail-on",
                    "error",
                    "--report-json",
                    str(report),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertEqual(
                json.loads(report.read_text())["execution_status"], "FAILED"
            )


class AliasAndSurfaceTest(unittest.TestCase):
    def test_same_alias_spelling_does_not_hide_narrowing(self):
        before = {
            "flashinfer/example.py": "Kind = int | str\n@flashinfer_api\ndef api(x: Kind): pass\n"
        }
        after = {
            "flashinfer/example.py": "Kind = int\n@flashinfer_api\ndef api(x: Kind): pass\n"
        }
        self.assertEqual(
            compare_snapshots(before, after)[0].classification, "LIKELY_BREAKING"
        )

    def test_ambiguous_reexport_is_not_a_proven_deletion(self):
        before = {"flashinfer/example.py": "@flashinfer_api\ndef api(x): pass\n"}
        after = {
            "flashinfer/example.py": "if use_first:\n from .first import api\nelse:\n from .second import api\n"
        }
        self.assertEqual(
            compare_snapshots(before, after)[0].classification, "LIKELY_BREAKING"
        )

    def test_dynamic_namespace_needs_review(self):
        before = {"flashinfer/example.py": "@flashinfer_api\ndef api(x): pass\n"}
        after = {
            "flashinfer/example.py": "def __getattr__(name): return resolve(name)\n"
        }
        self.assertEqual(
            compare_snapshots(before, after)[0].classification, "LIKELY_BREAKING"
        )

    def test_decorator_removal_cannot_hide_a_required_parameter(self):
        before = {"flashinfer/example.py": "@flashinfer_api\ndef api(x): pass\n"}
        after = {"flashinfer/example.py": "def api(x, workspace): pass\n"}
        findings = compare_snapshots(before, after)
        self.assertIn("BREAKING", [finding.classification for finding in findings])

    def test_dynamic_top_level_export_is_not_a_proven_removal(self):
        before = {"flashinfer/__init__.py": "from .example import api\n"}
        after = {
            "flashinfer/__init__.py": "def __getattr__(name): return resolve(name)\n"
        }
        self.assertEqual(
            compare_snapshots(before, after)[0].classification, "LIKELY_BREAKING"
        )
