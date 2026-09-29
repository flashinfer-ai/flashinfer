from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.pr_checks.inspect_sources import collect_module_alias_exports


class ModuleAliasExportsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.package = Path(self.tempdir.name) / "flashinfer"
        self.package.mkdir()
        (self.package / "implementation.py").write_text(
            "@flashinfer_api\ndef original():\n    pass\n", encoding="utf-8"
        )

    def test_ordinary_module_import_is_internal(self) -> None:
        (self.package / "consumer.py").write_text(
            "from .implementation import original as internal_alias\n",
            encoding="utf-8",
        )

        self.assertEqual(collect_module_alias_exports(self.package), {})

    def test_package_init_import_is_public(self) -> None:
        (self.package / "__init__.py").write_text(
            "from .implementation import original as public_alias\n",
            encoding="utf-8",
        )

        self.assertEqual(
            collect_module_alias_exports(self.package),
            {"flashinfer": {"public_alias"}},
        )

    def test_explicit_module_all_makes_import_public(self) -> None:
        (self.package / "consumer.py").write_text(
            "from .implementation import original as public_alias\n"
            "__all__ = ('public_alias',)\n",
            encoding="utf-8",
        )

        self.assertEqual(
            collect_module_alias_exports(self.package),
            {"flashinfer.consumer": {"public_alias"}},
        )


if __name__ == "__main__":
    unittest.main()
