from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from pr_checks.check_api_docs import (  # noqa: E402
    collect_flashinfer_api_functions,
    run_check,
)


class ImportedAliasDocumentationTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.pkg = Path(directory.name) / "flashinfer"
        self.write("__init__.py", "")
        self.write("sampling.py", "@flashinfer_api\ndef softmax(logits): pass\n")
        self.write("logits_processor/__init__.py", "")
        self.write(
            "logits_processor/operators.py",
            "from flashinfer.sampling import softmax as sampling_softmax\n",
        )

    def write(self, path: str, source: str) -> None:
        target = self.pkg / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(source))

    def check(self, documented: dict[str, set[str]]) -> dict[str, dict]:
        apis = collect_flashinfer_api_functions(self.pkg)
        return {row["module"]: row for row in run_check(apis, documented)}

    def test_internal_sampling_import_does_not_require_its_own_page(self) -> None:
        rows = self.check({"flashinfer.sampling": {"softmax"}})
        self.assertNotIn("flashinfer.logits_processor.operators", rows)
        self.assertFalse(any(row["missing"] for row in rows.values()))

    def test_undocumented_definition_is_still_reported(self) -> None:
        rows = self.check({})
        self.assertEqual(rows["flashinfer.sampling"]["missing"], ["softmax"])
        self.assertNotIn("flashinfer.logits_processor.operators", rows)

    def test_relative_internal_alias_is_not_an_export(self) -> None:
        self.write("consumer.py", "from .sampling import softmax as internal\n")
        apis = collect_flashinfer_api_functions(self.pkg)
        self.assertNotIn("flashinfer.consumer", apis)

    def test_literal_module_exports_require_documentation(self) -> None:
        declarations = (
            "__all__ = ['sampling_softmax']",
            "__all__: tuple[str, ...] = ('sampling_softmax',)",
            "__all__ = []\n__all__ += ['sampling_softmax']",
        )
        for declaration in declarations:
            with self.subTest(declaration=declaration):
                self.write(
                    "logits_processor/operators.py",
                    "from flashinfer.sampling import softmax as sampling_softmax\n"
                    + declaration
                    + "\n",
                )
                rows = self.check({"flashinfer.sampling": {"softmax"}})
                self.assertEqual(
                    rows["flashinfer.logits_processor.operators"]["missing"],
                    ["sampling_softmax"],
                )
                rows = self.check(
                    {
                        "flashinfer.sampling": {"softmax"},
                        "flashinfer.logits_processor.operators": {"sampling_softmax"},
                    }
                )
                self.assertFalse(any(row["missing"] for row in rows.values()))

    def test_explicit_exports_exclude_other_imported_dependencies(self) -> None:
        self.write(
            "logits_processor/operators.py",
            """
            from flashinfer.sampling import softmax as sampling_softmax
            from flashinfer.sampling import softmax as public_softmax
            __all__ = ('public_softmax',)
            """,
        )
        apis = collect_flashinfer_api_functions(self.pkg)
        self.assertEqual(
            apis["flashinfer.logits_processor.operators"], {"public_softmax"}
        )

    def test_dynamic_exports_are_retained_conservatively(self) -> None:
        self.write(
            "logits_processor/operators.py",
            "from flashinfer.sampling import softmax as sampling_softmax\n"
            "__all__ = exported_names()\n",
        )
        rows = self.check({"flashinfer.sampling": {"softmax"}})
        self.assertEqual(
            rows["flashinfer.logits_processor.operators"]["missing"],
            ["sampling_softmax"],
        )

    def test_guarded_public_reexport_through_internal_alias_is_preserved(self) -> None:
        self.write(
            "logits_processor/__init__.py",
            """
            try:
                from .operators import sampling_softmax as probability
            except ImportError:
                pass
            """,
        )
        rows = self.check({"flashinfer.sampling": {"softmax"}})
        self.assertEqual(
            rows["flashinfer.logits_processor"]["missing"], ["probability"]
        )
        self.assertNotIn("flashinfer.logits_processor.operators", rows)
        rows = self.check(
            {
                "flashinfer.sampling": {"softmax"},
                "flashinfer.logits_processor": {"probability"},
            }
        )
        self.assertFalse(any(row["missing"] for row in rows.values()))

    def test_nested_imports_are_not_module_exports(self) -> None:
        self.write(
            "logits_processor/operators.py",
            """
            __all__ = ('guarded_alias', 'local_alias', 'async_alias', 'class_alias')
            if enabled:
                from flashinfer.sampling import softmax as guarded_alias

            def helper():
                from flashinfer.sampling import softmax as local_alias

            async def async_helper():
                from flashinfer.sampling import softmax as async_alias

            class Helper:
                from flashinfer.sampling import softmax as class_alias
            """,
        )
        apis = collect_flashinfer_api_functions(self.pkg)
        self.assertEqual(
            apis["flashinfer.logits_processor.operators"], {"guarded_alias"}
        )


if __name__ == "__main__":
    unittest.main()
