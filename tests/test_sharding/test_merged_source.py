"""Real Git regression tests for isolated, current-base source preparation."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(ROOT / "src" if (ROOT / "src").is_dir() else ROOT / "scripts/pr_checks")
)
from merged_source import MergedSource


class MergedSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.git("init", "-b", "main")
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.invalid")
        (self.repo / "value.py").write_text("original\n")
        self.git("add", ".")
        self.git("commit", "-m", "original base")
        self.original = self.git("rev-parse", "HEAD")
        self.git("checkout", "-b", "pr")
        (self.repo / "pr.py").write_text("PR content\n")
        self.git("add", ".")
        self.git("commit", "-m", "PR")
        self.head = self.git("rev-parse", "HEAD")
        self.git("checkout", "main")
        (self.repo / "new_base.py").write_text("base content\n")
        self.git("add", ".")
        self.git("commit", "-m", "base advanced")
        self.base = self.git("rev-parse", "HEAD")

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()

    def test_merged_tree_contains_current_base_and_pr_without_changing_source(self):
        source = MergedSource("owner/repo", self.base, self.head, self.repo)
        self.addCleanup(source.close)
        entries = {item["path"]: item for item in source.tree(source.tree_sha)["tree"]}
        self.assertIn("new_base.py", entries)
        self.assertEqual("PR content\n", source.blob(entries["pr.py"]["sha"], 100))
        self.assertEqual(
            [{"sha": self.base}, {"sha": self.head}], source.commit["parents"]
        )
        self.assertEqual(self.base, self.git("rev-parse", "HEAD"))
        self.assertFalse((self.repo / "pr.py").exists())
        with self.assertRaises(ValueError):
            source.blob(entries["pr.py"]["sha"], 1)

    def test_shallow_reference_is_not_mistaken_for_complete_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            shallow = Path(directory) / "shallow"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--branch=main",
                    self.repo.as_uri(),
                    str(shallow),
                ],
                check=True,
                capture_output=True,
            )
            original_git = MergedSource.git
            fetches = []

            def local_transport(source, *args, **kwargs):
                if args[0] == "fetch":
                    fetches.append(args[-1])
                    args = tuple(
                        str(self.repo) if arg == "origin" else arg for arg in args
                    )
                return original_git(source, *args, **kwargs)

            with patch.object(MergedSource, "git", local_transport):
                source = MergedSource("owner/repo", self.base, self.head, shallow)
            self.addCleanup(source.close)
            self.assertEqual([self.base, self.head], fetches)
            self.assertFalse((source.root / ".git/objects/info/alternates").exists())
            paths = {item["path"] for item in source.tree(source.tree_sha)["tree"]}
            self.assertTrue({"new_base.py", "pr.py"}.issubset(paths))

    def test_conflicts_do_not_run_configured_merge_drivers_or_hooks(self):
        marker = self.repo / "executed"
        global_config = self.repo / "global_config"
        global_config.write_text(
            f'[merge "custom"]\n driver = touch {marker}\n[core]\n hooksPath = {self.repo}/hooks\n'
        )
        (self.repo / "hooks").mkdir()
        hook = self.repo / "hooks/post-checkout"
        hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
        hook.chmod(0o755)
        for branch, value in (("pr", "PR"), ("main", "base")):
            self.git("checkout", branch)
            (self.repo / "value.py").write_text(value + "\n")
            (self.repo / ".gitattributes").write_text("value.py merge=custom\n")
            self.git("add", "value.py", ".gitattributes")
            self.git("commit", "-m", value)
        head = self.git("rev-parse", "pr")
        base = self.git("rev-parse", "main")
        with (
            patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(global_config)}),
            self.assertRaisesRegex(ValueError, "merge conflict"),
        ):
            MergedSource("owner/repo", base, head, self.repo)
        self.assertFalse(marker.exists())

    def test_invalid_repository_or_commit_is_rejected_before_git(self):
        for repository, base in (("../bad", self.base), ("owner/repo", "main")):
            with (
                self.subTest(repository=repository, base=base),
                self.assertRaises(ValueError),
            ):
                MergedSource(repository, base, self.head, self.repo)
