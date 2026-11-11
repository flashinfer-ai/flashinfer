"""Regression coverage for canonical agent-guide selection across revisions."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pr_checks.configure_checks import find_agent_guide
from pr_checks import check_cross_sources as checker


class TestAgentGuideMigration(unittest.TestCase):
    def test_old_and_new_layouts_keep_checks_active(self):
        guide = (
            "FlashInfer supports NVIDIA SM90.\n"
            "FLASHINFER_GUIDE_TEST\n"
            "| Guide | `docs/missing.rst` |\n"
        )
        shim = (
            "This repository’s agent instructions live in [CLAUDE.md](./CLAUDE.md).\n"
        )
        layouts = (
            ({"CLAUDE.md": guide}, "CLAUDE.md"),
            ({"AGENTS.md": shim, "CLAUDE.md": guide}, "CLAUDE.md"),
            ({"AGENTS.md": guide}, "AGENTS.md"),
            ({"AGENTS.md": guide, "CLAUDE.md": "stale"}, "AGENTS.md"),
        )
        for contents, expected in layouts:
            with (
                self.subTest(expected=expected, files=list(contents)),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                pkg = root / "flashinfer"
                pkg.mkdir()
                (pkg / "utils.py").write_text(
                    'import os\nos.getenv("FLASHINFER_GUIDE_TEST")\n'
                    'os.getenv("FLASHINFER_GUIDE_MISSING")\n'
                    "def is_sm90_supported(): pass\n"
                    "def is_sm100_supported(): pass\n"
                )
                for name, text in contents.items():
                    (root / name).write_text(text)
                selected = find_agent_guide(root)
                self.assertEqual(selected.name, expected)
                with (
                    patch.object(checker, "AGENT_GUIDE_MD", selected),
                    patch.object(checker, "FLASHINFER_ROOT", root),
                    patch.object(checker, "FLASHINFER_PKG", pkg),
                    patch.object(checker, "CSRC_DIR", root / "csrc"),
                ):
                    env = checker.check_env_vars_consistency()
                    self.assertEqual(len(env), 1)
                    self.assertIn("FLASHINFER_GUIDE_MISSING", env[0].message)
                    arch = checker.check_supported_arch()
                    self.assertEqual(len(arch), 1)
                    self.assertIn("SM100", arch[0].message)
                    self.assertIn(expected, arch[0].message)
                    paths = checker.check_quickref_paths_exist()
                    self.assertEqual(len(paths), 1)
                    self.assertTrue(paths[0].location.startswith(expected))


if __name__ == "__main__":
    unittest.main()
