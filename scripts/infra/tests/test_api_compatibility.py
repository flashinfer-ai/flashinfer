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
            ("x, y=0", "x, **kwargs", "BREAKING"),
            ("x, /, **kwargs", "x, **kwargs", "BREAKING"),
            ("x, /", "**kwargs", "BREAKING"),
            ("x, /", "*args", "LIKELY_BREAKING"),
            ("x, *, y=0", "x, **kwargs", "LIKELY_BREAKING"),
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


class InlineSourceRangeTest(unittest.TestCase):
    def test_multiline_signature_carries_both_source_ranges(self):
        old = {
            "flashinfer/example.py": "@flashinfer_api\ndef api(\n x,\n y=0,\n): pass\n"
        }
        new = {"flashinfer/example.py": "@flashinfer_api\ndef api(\n x,\n): pass\n"}
        finding = compare_snapshots(old, new)[0]
        self.assertIn(("flashinfer/example.py", "RIGHT", 2, 4), finding.anchors)
        self.assertIn(("flashinfer/example.py", "LEFT", 2, 5), finding.anchors)

    def test_deleted_api_carries_old_side_signature(self):
        old = {"flashinfer/example.py": "@flashinfer_api\ndef api(\n x,\n): pass\n"}
        finding = compare_snapshots(old, {"flashinfer/example.py": ""})[0]
        self.assertIn(("flashinfer/example.py", "LEFT", 1, 4), finding.anchors)


class ReviewRegressionTest(unittest.TestCase):
    def test_recursive_alias_ignores_unrelated_constants(self):
        aliases = {"Tree": "tuple[Tree] | None", "OTHER": "1"}
        self.assertTrue(
            checker.is_compatible_annotation(
                "Tree", "Tree", aliases, {**aliases, "OTHER": "2"}
            )
        )

    def test_recursive_alias_changes_terminate_and_detect_narrowing(self):
        old = {"Tree": "tuple[Tree, int | str] | None"}
        new = {"Tree": "tuple[Tree, int] | None"}
        self.assertFalse(checker.is_compatible_annotation("Tree", "Tree", old, new))
        self.assertTrue(checker.is_compatible_annotation("Tree", "Tree", new, old))
        self.assertTrue(
            checker.is_compatible_annotation(
                "Tree",
                "Tree",
                {"Tree": "Tree | int"},
                {"Tree": "Tree | int", "OTHER": "2"},
            )
        )

    def test_nested_function_cannot_preserve_a_removed_module_api(self):
        old = {"flashinfer/example.py": "@flashinfer_api\ndef api(x): pass\n"}
        for source in (
            "def helper():\n def api(x): pass\n",
            "class Helper:\n def __getattr__(self, name): return resolve(name)\n",
        ):
            with self.subTest(source=source):
                findings = compare_snapshots(old, {"flashinfer/example.py": source})
                self.assertEqual(findings[0].classification, "BREAKING")

    def test_experimental_syntax_does_not_block_stable_gate(self):
        self.assertEqual(
            compare_snapshots(
                {"flashinfer/experimental/example.py": "x=1\n"},
                {"flashinfer/experimental/example.py": "def broken(:\n"},
            ),
            [],
        )

    def test_known_lazy_hook_does_not_hide_removed_export(self):
        prefix = "from .activation import gelu_and_mul\n"
        # Exercise the repository's real lazy-export hook and computed table.
        init = (
            Path(checker.__file__).resolve().parents[2] / "flashinfer/__init__.py"
        ).read_text()
        removed = init.replace(
            "from .activation import gelu_and_mul as gelu_and_mul\n", "", 1
        )
        full_findings = compare_snapshots(
            {"flashinfer/__init__.py": init}, {"flashinfer/__init__.py": removed}
        )
        self.assertEqual(
            next(
                f for f in full_findings if f.api == "flashinfer.gelu_and_mul"
            ).classification,
            "BREAKING",
        )
        lazy = init[
            init.index("_PRIMS_TS_LAZY_EXPORTS =") : init.index("def __dir__()")
        ]
        findings = compare_snapshots(
            {"flashinfer/__init__.py": prefix + lazy}, {"flashinfer/__init__.py": lazy}
        )
        self.assertEqual(findings[0].classification, "BREAKING")
        # A runtime update makes table membership uncertain again.
        after = lazy + "\n_LAZY_EXPORTS.update(runtime_exports)\n"
        findings = compare_snapshots(
            {"flashinfer/__init__.py": prefix + lazy}, {"flashinfer/__init__.py": after}
        )
        self.assertEqual(findings[0].classification, "LIKELY_BREAKING")

    def test_module_package_conversion_preserves_import_and_checks_signature(self):
        for reverse in (False, True):
            for broken in (False, True):
                with self.subTest(reverse=reverse, broken=broken):
                    paths = ["flashinfer/example.py", "flashinfer/example/__init__.py"]
                    if reverse:
                        paths.reverse()
                    old = {paths[0]: "@flashinfer_api\ndef api(x): pass\n"}
                    new = {
                        paths[1]: "@flashinfer_api\ndef api("
                        + ("x, y" if broken else "x")
                        + "): pass\n"
                    }
                    findings = compare_snapshots(old, new)
                    self.assertFalse(
                        any(
                            f.check
                            in (
                                "public_module_deleted",
                                "public_module_moved",
                                "public_api_removed",
                            )
                            for f in findings
                        )
                    )
                    self.assertEqual(
                        any(f.classification == "BREAKING" for f in findings), broken
                    )
