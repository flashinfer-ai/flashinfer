"""Tests for scripts/pr_checks/select_tests.py on a small synthetic repository."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pr_checks"))

import select_tests  # noqa: E402

FILES = {
    "requirements.txt": "torch\n",
    "docs/index.md": "docs\n",
    "include/flashinfer/norm.cuh": "#pragma once\n",
    "include/flashinfer/vec.cuh": "#pragma once\n",
    "csrc/norm.cu": '#include "flashinfer/norm.cuh"\n',
    "csrc/gemm/mm.cu": '#include "flashinfer/vec.cuh"\n',
    "csrc/orphan.cu": "// referenced by nothing\n",
    "flashinfer/__init__.py": "from .version import __version__\n",
    "flashinfer/version.py": '__version__ = "0"\n',
    # The kernel generator is imported inside a function, as in the real APIs.
    "flashinfer/norm.py": """
        def rmsnorm():
            from .jit.norm import gen_norm_module
            return gen_norm_module()
    """,
    "flashinfer/jit/__init__.py": "from .norm import gen_norm_module\n",
    "flashinfer/jit/norm.py": """
        SOURCES = ["norm.cu"]
        def gen_norm_module():
            return SOURCES
    """,
    "flashinfer/jit/norm_manifest.json": "{}\n",
    "flashinfer/gemm/__init__.py": "from .core import mm\n",
    "flashinfer/gemm/core.py": """
        KERNEL_DIR = "gemm"
        def mm():
            return KERNEL_DIR
    """,
    # A lazy package, and a second one that re-exports from it.
    "flashinfer/lazy/__init__.py": """
        import importlib
        _LAZY_EXPORTS = {"fast_op": ".impl"}
        def __getattr__(name):
            return getattr(importlib.import_module(_LAZY_EXPORTS[name], __name__), name)
    """,
    "flashinfer/lazy/impl.py": "def fast_op():\n    return 1\n",
    "flashinfer/lazy/other.py": "def other():\n    return 2\n",
    "flashinfer/lazy2/__init__.py": """
        import importlib
        _LAZY_EXPORTS = {"fast_op": "flashinfer.lazy"}
        def __getattr__(name):
            return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    """,
    "flashinfer/dyn.py": """
        import importlib
        def load():
            return importlib.import_module("flashinfer.lazy.other")
    """,
    # An eager package __init__ couples its submodules.
    "flashinfer/hub/__init__.py": "from .a import fa\nfrom .b import fb\n",
    "flashinfer/hub/a.py": "def fa():\n    return 1\n",
    "flashinfer/hub/b.py": "def fb():\n    return 2\n",
    # Generated sources that a runtime loads by path from a manifest.
    "flashinfer/gemm/generated/manifest.json": '{"sources": ["kernel_a.py"]}\n',
    "flashinfer/gemm/generated/kernel_a.py": "from flashinfer.gemm.tile import TILE\n",
    "flashinfer/gemm/tile.py": "TILE = 128\n",
    # b12x is installed from _compat; its shim loads _api.py by path and serves
    # b12x.<name> from flashinfer/experimental/b12x.
    "flashinfer/experimental/__init__.py": "",
    "flashinfer/experimental/b12x/__init__.py": """
        import importlib
        importlib.import_module("b12x")
    """,
    "flashinfer/experimental/b12x/_api.py": "def api():\n    return 0\n",
    "flashinfer/experimental/b12x/attention.py": "def attend():\n    return 1\n",
    "flashinfer/experimental/b12x/norm.py": "def norm():\n    return 2\n",
    "flashinfer/experimental/b12x/_compat/b12x/__init__.py": """
        from pathlib import Path
        API = Path(__file__).parents[2] / "_api.py"
    """,
    # benchmarks/ is not a package: tests reach its modules through sys.path.
    "benchmarks/bench_tool.py": "def copy_selected():\n    return 0\n",
    # A script loaded by path, which imports a helper.
    "scripts/__init__.py": "",
    "scripts/tool.py": "from scripts.tool_lib import run\n",
    "scripts/tool_lib.py": "def run():\n    return 0\n",
    "tests/conftest.py": 'pytest_plugins = ["tests.helpers.plugin"]\n',
    "tests/helpers/plugin.py": "",
    "tests/helpers/util.py": "def helper():\n    return 0\n",
    "tests/utils/test_norm.py": """
        from flashinfer.norm import rmsnorm
        from tests.helpers.util import helper
        def test_norm():
            rmsnorm()
            helper()
    """,
    "tests/gemm/conftest.py": "",
    "tests/gemm/data/expected.json": "{}\n",
    "tests/gemm/test_mm.py": """
        import flashinfer
        def test_mm():
            flashinfer.gemm.mm()
    """,
    "tests/lazy/test_lazy.py": """
        import flashinfer.lazy2
        def test_lazy():
            flashinfer.lazy2.fast_op()
    """,
    "tests/dyn/test_dyn.py": "from flashinfer.dyn import load\n",
    "tests/hub/test_a.py": "from flashinfer.hub.a import fa\n",
    "tests/b12x/test_attention.py": "from b12x.attention import attend\n",
    "tests/b12x/test_norm.py": "import b12x.norm\n",
    "tests/moe/test_bench_tool.py": """
        def test_copy(monkeypatch):
            monkeypatch.syspath_prepend("benchmarks")
            from bench_tool import copy_selected
    """,
    # A script nothing imports; the conftest beside it must not make it global.
    "tests/repro_issue.py": "print('repro')\n",
    # A bare "__init__.py" names no particular package, so it adds no dependency.
    "tests/test_tool.py": """
        from pathlib import Path
        TOOL = Path(__file__).parents[1] / "scripts" / "tool.py"
        INIT = TOOL.parent / "__init__.py"
    """,
}
ALL_TESTS = sorted(p for p in FILES if select_tests.TEST_FILE_RE.match(p))


def git(repo: Path, *args: str) -> str:
    # Ignore the developer's git config: commit signing or hooks would stall commits.
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t",
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")  # fmt: skip
    return subprocess.check_output(["git", *args], cwd=repo, env=env, text=True).strip()


class SelectTestsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.repo = Path(cls._tmp.name)
        for path, text in FILES.items():
            target = cls.repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(textwrap.dedent(text).lstrip("\n"))
        git(cls.repo, "init", "-q")
        git(cls.repo, "add", "-A")
        git(cls.repo, "commit", "-q", "-m", "base")
        cls.selector = select_tests.Selector(select_tests.build_graph(cls.repo))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def select(self, *changed: str) -> select_tests.Selection:
        return self.selector.select(list(changed))

    def test_header_reaches_tests_through_function_level_import(self) -> None:
        self.assertEqual(
            self.select("include/flashinfer/norm.cuh").tests,
            ["tests/utils/test_norm.py"],
        )
        self.assertEqual(
            self.select("flashinfer/jit/norm.py").tests, ["tests/utils/test_norm.py"]
        )

    def test_directory_literal_includes_headers_of_its_sources(self) -> None:
        self.assertEqual(
            self.select("include/flashinfer/vec.cuh").tests, ["tests/gemm/test_mm.py"]
        )

    def test_data_files_belong_to_their_directory(self) -> None:
        self.assertEqual(self.select("flashinfer/jit/norm_manifest.json").tests,
                         ["tests/utils/test_norm.py"])  # fmt: skip
        self.assertEqual(
            self.select("tests/gemm/data/expected.json").tests,
            ["tests/gemm/test_mm.py"],
        )

    def test_lazy_exports_resolve_through_reexport_chains(self) -> None:
        self.assertEqual(
            self.select("flashinfer/lazy/impl.py").tests, ["tests/lazy/test_lazy.py"]
        )
        # other.py is not exported, so only the string import in dyn.py reaches it.
        self.assertEqual(
            self.select("flashinfer/lazy/other.py").tests, ["tests/dyn/test_dyn.py"]
        )

    def test_eager_package_init_couples_submodules(self) -> None:
        self.assertEqual(
            self.select("flashinfer/hub/b.py").tests, ["tests/hub/test_a.py"]
        )

    def test_aliased_package_resolves_each_submodule(self) -> None:
        self.assertEqual(
            self.select("flashinfer/experimental/b12x/attention.py").tests,
            ["tests/b12x/test_attention.py"],
        )
        # The shim loads _api.py by path for every b12x import.
        self.assertEqual(
            self.select("flashinfer/experimental/b12x/_api.py").tests,
            ["tests/b12x/test_attention.py", "tests/b12x/test_norm.py"],
        )

    def test_modules_outside_packages_resolve_through_sys_path(self) -> None:
        self.assertEqual(
            self.select("benchmarks/bench_tool.py").tests,
            ["tests/moe/test_bench_tool.py"],
        )

    def test_python_file_named_by_path_is_loaded_with_its_imports(self) -> None:
        self.assertEqual(
            self.select("scripts/tool_lib.py").tests, ["tests/test_tool.py"]
        )

    def test_sources_loaded_by_path_belong_to_their_directory(self) -> None:
        for path in (
            "flashinfer/gemm/generated/kernel_a.py",
            "flashinfer/gemm/generated/manifest.json",
            "flashinfer/gemm/tile.py",
        ):
            self.assertEqual(self.select(path).tests, ["tests/gemm/test_mm.py"], path)

    def test_stray_test_scripts_belong_to_tests_not_conftest(self) -> None:
        self.assertEqual(
            self.select("tests/repro_issue.py").tests, ["tests/test_tool.py"]
        )

    def test_global_inputs_select_everything(self) -> None:
        for path in (
            "requirements.txt",
            "tests/conftest.py",
            "tests/helpers/plugin.py",
            "scripts/task_jit_run_tests_part1.sh",
            "scripts/run_with_test_sccache.sh",
        ):
            selection = self.select(path, "docs/index.md")
            self.assertEqual(selection.tests, ALL_TESTS, path)
            self.assertEqual(selection.global_inputs, [path])

    def test_directory_conftest_selects_its_tests(self) -> None:
        self.assertEqual(
            self.select("tests/gemm/conftest.py").tests, ["tests/gemm/test_mm.py"]
        )

    def test_changed_test_file_selects_itself(self) -> None:
        self.assertEqual(
            self.select("tests/hub/test_a.py").tests, ["tests/hub/test_a.py"]
        )

    def test_unrelated_changes_select_nothing(self) -> None:
        selection = self.select("docs/index.md", "csrc/orphan.cu")
        self.assertEqual(selection.tests, [])
        self.assertEqual(selection.unreached, ["csrc/orphan.cu"])

    def test_deleted_files_fall_back_to_their_directory(self) -> None:
        self.assertEqual(
            self.select("flashinfer/gemm/removed.py").tests, ["tests/gemm/test_mm.py"]
        )
        # No surviving directory below the top level: everything.
        self.assertEqual(self.select("flashinfer/removed.py").tests, ALL_TESTS)

    def test_explain_returns_the_chain_to_the_change(self) -> None:
        chain = self.selector.explain(
            "tests/utils/test_norm.py", ["include/flashinfer/norm.cuh"]
        )
        assert chain is not None
        self.assertEqual(chain[0], "tests/utils/test_norm.py")
        self.assertEqual(
            chain[-2:], ["flashinfer/jit/norm.py", "include/flashinfer/norm.cuh"]
        )
        self.assertIsNone(
            self.selector.explain(
                "tests/hub/test_a.py", ["include/flashinfer/norm.cuh"]
            )
        )

    def test_markdown_reports_scope_gaps(self) -> None:
        selection = self.select(
            "include/flashinfer/norm.cuh", "include/flashinfer/vec.cuh"
        )
        report = select_tests.render_markdown(selection, ["tests/gemm"])
        self.assertIn("Would run **2 of", report)
        self.assertIn(
            "leaves out 1 selected test files: `tests/utils/test_norm.py`", report
        )

    def run_cli(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = select_tests.main([*args, "--repo", str(self.repo)])
        return code, out.getvalue()

    def run_main(self, *args: str) -> dict:
        code, out = self.run_cli(*args, "--format", "json")
        self.assertEqual(code, 0)
        return json.loads(out)

    def test_explain_reports_full_suite_and_unknown_paths(self) -> None:
        head = git(self.repo, "rev-parse", "HEAD")
        code, out = self.run_cli(
            "--base", head, "--explain", "tests/utils/test_nrom.py"
        )
        self.assertEqual((code, out), (2, ""))
        code, out = self.run_cli(
            "--base", "no-such-revision", "--explain", "tests/hub/test_a.py"
        )
        self.assertEqual(code, 0)
        self.assertIn("is selected because the selector failed", out)
        requirements = self.repo / "requirements.txt"
        requirements.write_text("torch\nnumpy\n")
        try:
            code, out = self.run_cli(
                "--base", head, "--explain", "./tests/hub/test_a.py"
            )
        finally:
            git(self.repo, "checkout", "--", "requirements.txt")
        self.assertEqual(
            out.strip(),
            "tests/hub/test_a.py is selected because global inputs changed: "
            "requirements.txt",
        )

    def test_main_diffs_from_the_merge_base(self) -> None:
        base = git(self.repo, "rev-parse", "HEAD")
        git(self.repo, "checkout", "-q", "-b", "change")
        (self.repo / "flashinfer/hub/b.py").write_text("def fb():\n    return 3\n")
        git(self.repo, "commit", "-q", "-am", "change")
        head = git(self.repo, "rev-parse", "HEAD")
        try:
            result = self.run_main("--base", base, "--head", head)
        finally:
            git(self.repo, "checkout", "-q", base)
        self.assertEqual(result["changed"], ["flashinfer/hub/b.py"])
        self.assertEqual(result["tests"], ["tests/hub/test_a.py"])
        self.assertFalse(result["full_suite"])

    def test_errors_select_everything(self) -> None:
        result = self.run_main("--base", "no-such-revision")
        self.assertTrue(result["full_suite"])
        self.assertEqual(result["tests"], ALL_TESTS)
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
