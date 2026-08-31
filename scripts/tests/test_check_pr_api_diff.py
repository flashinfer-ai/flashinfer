from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import check_pr_api_diff as api_diff  # noqa: E402


def api_function(signature: str) -> api_diff.ApiFunction:
    source = textwrap.dedent(
        f"""
        @flashinfer_api
        {signature}:
            pass
        """
    )
    return api_diff.extract_public_apis("flashinfer/example.py", source)["api"]


def class_source(name: str, signature: str) -> str:
    return textwrap.dedent(
        f"""
        class {name}:
            @flashinfer_api
            {signature}:
                pass
        """
    )


def run_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def commit_snapshot(repo: Path, files: dict[str, str], message: str) -> str:
    for child in repo.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for relative_path, source in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", message)
    return run_git(repo, "rev-parse", "HEAD")


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def compare_snapshots(
    base_files: dict[str, str], head_files: dict[str, str]
) -> list[api_diff.PrFinding]:
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        run_git(repo, "init", "-q")
        run_git(repo, "config", "user.name", "API checker test")
        run_git(repo, "config", "user.email", "api-checker@example.com")
        base = commit_snapshot(repo, base_files, "base")
        head = commit_snapshot(repo, head_files, "head")
        with working_directory(repo):
            return api_diff.check(base, head)


class SignatureCompatibilityTest(unittest.TestCase):
    def test_compatible_signature_changes(self) -> None:
        cases = (
            (
                "keyword-only insertion",
                "def api(value: int, *, first: int = 1, last: int = 2) -> None",
                "def api(value: int, *, first: int = 1, inserted: str = 'x', last: int = 2) -> None",
            ),
            (
                "keyword-only reorder and append",
                "def api(value: int, *, first: int = 1, last: int = 2) -> None",
                "def api(value: int, *, last: int = 2, first: int = 1, appended: bool = False) -> None",
            ),
            (
                "optional positional append",
                "def api(value: int) -> None",
                "def api(value: int, added: str = 'x') -> None",
            ),
            (
                "required parameter gains default",
                "def api(first: int, second: str) -> None",
                "def api(first: int = 1, second: str = 'x') -> None",
            ),
            (
                "positional-only parameter widens",
                "def api(value: Tensor, /) -> None",
                "def api(value: Optional[Tensor] = None, /) -> None",
            ),
            (
                "Optional annotation",
                "def api(value: Tensor) -> None",
                "def api(value: Optional[Tensor] = None) -> None",
            ),
            (
                "qualified Optional annotation",
                "def api(value: Tensor) -> None",
                "def api(value: typing.Optional[Tensor] = None) -> None",
            ),
            (
                "Union with None annotation",
                "def api(value: Tensor) -> None",
                "def api(value: Union[Tensor, None] = None) -> None",
            ),
            (
                "pipe union with None annotation",
                "def api(value: Tensor) -> None",
                "def api(value: Tensor | None = None) -> None",
            ),
        )

        for name, before, after in cases:
            with self.subTest(name):
                self.assertTrue(
                    api_diff.is_compatible_api(
                        api_function(before),
                        api_function(after),
                    )
                )

    def test_breaking_signature_changes(self) -> None:
        cases = (
            (
                "positional-only kind changed",
                "def api(value: int, /) -> None",
                "def api(value: int) -> None",
            ),
            (
                "positional parameter removed",
                "def api(first: int, second: str) -> None",
                "def api(first: int) -> None",
            ),
            (
                "positional parameter renamed",
                "def api(first: int, second: str) -> None",
                "def api(renamed: int, second: str) -> None",
            ),
            (
                "positional parameters reordered",
                "def api(first: int, second: str) -> None",
                "def api(second: str, first: int) -> None",
            ),
            (
                "positional parameter inserted",
                "def api(first: int, second: str) -> None",
                "def api(first: int, inserted: bool = False, second: str = 'x') -> None",
            ),
            (
                "existing default changed",
                "def api(value: int = 1) -> None",
                "def api(value: int = 2) -> None",
            ),
            (
                "annotation narrowed",
                "def api(value: Optional[int] = None) -> None",
                "def api(value: int = 0) -> None",
            ),
            (
                "general union widening",
                "def api(value: int) -> None",
                "def api(value: int | str) -> None",
            ),
            (
                "unrelated Optional attribute",
                "def api(value: Tensor) -> None",
                "def api(value: custom.Optional[Tensor] = None) -> None",
            ),
            (
                "new required keyword-only parameter",
                "def api(value: int) -> None",
                "def api(value: int, *, required: str) -> None",
            ),
            (
                "new named parameter before varargs",
                "def api(value: int, *args: object) -> None",
                "def api(value: int, added: str = 'x', *args: object) -> None",
            ),
            (
                "new named parameter before kwargs",
                "def api(value: int, **kwargs: object) -> None",
                "def api(value: int, added: str = 'x', **kwargs: object) -> None",
            ),
            (
                "return annotation changed",
                "def api(value: int) -> int",
                "def api(value: int) -> str",
            ),
            (
                "function became async",
                "def api(value: int) -> None",
                "async def api(value: int) -> None",
            ),
        )

        for name, before, after in cases:
            with self.subTest(name):
                self.assertFalse(
                    api_diff.is_compatible_api(
                        api_function(before),
                        api_function(after),
                    )
                )


class ClassReexportIntegrationTest(unittest.TestCase):
    def test_class_member_reexport_outcomes(self) -> None:
        base = {
            "flashinfer/public.py": class_source(
                "Exported",
                "def member(self, value: Tensor, *, mode: str = 'fast') -> None",
            )
        }
        compatible_target = class_source(
            "Target",
            "def member(self, value: Optional[Tensor] = None, *, mode: str = 'fast', metadata: object = None) -> None",
        )
        cases: tuple[tuple[str, dict[str, str], tuple[str, ...]], ...] = (
            (
                "direct compatible re-export",
                {
                    "flashinfer/public.py": "from flashinfer.impl import Target as Exported\n",
                    "flashinfer/impl.py": compatible_target,
                },
                (),
            ),
            (
                "duplicate identical re-export",
                {
                    "flashinfer/public.py": (
                        "from flashinfer.impl import Target as Exported\n"
                        "from flashinfer.impl import Target as Exported\n"
                    ),
                    "flashinfer/impl.py": compatible_target,
                },
                (),
            ),
            (
                "runtime-guarded re-export",
                {
                    "flashinfer/public.py": (
                        "if runtime_condition:\n"
                        "    from flashinfer.impl import Target as Exported\n"
                    ),
                    "flashinfer/impl.py": compatible_target,
                },
                (),
            ),
            (
                "TYPE_CHECKING-only re-export",
                {
                    "flashinfer/public.py": (
                        "from typing import TYPE_CHECKING\n\n"
                        "if TYPE_CHECKING:\n"
                        "    from flashinfer.impl import Target as Exported\n"
                    ),
                    "flashinfer/impl.py": compatible_target,
                },
                ("public_api_removed",),
            ),
            (
                "typing.TYPE_CHECKING-only re-export",
                {
                    "flashinfer/public.py": (
                        "import typing\n\n"
                        "if typing.TYPE_CHECKING:\n"
                        "    from flashinfer.impl import Target as Exported\n"
                    ),
                    "flashinfer/impl.py": compatible_target,
                },
                ("public_api_removed",),
            ),
            (
                "__main__-guarded re-export",
                {
                    "flashinfer/public.py": (
                        'if __name__ == "__main__":\n'
                        "    from flashinfer.impl import Target as Exported\n"
                    ),
                    "flashinfer/impl.py": compatible_target,
                },
                ("public_api_removed",),
            ),
            (
                "conflicting re-export",
                {
                    "flashinfer/public.py": (
                        "from flashinfer.first import Target as Exported\n"
                        "from flashinfer.second import Target as Exported\n"
                    ),
                    "flashinfer/first.py": compatible_target,
                    "flashinfer/second.py": compatible_target,
                },
                ("public_api_removed",),
            ),
            (
                "breaking target signature",
                {
                    "flashinfer/public.py": "from flashinfer.impl import Target as Exported\n",
                    "flashinfer/impl.py": class_source(
                        "Target",
                        "def member(self, renamed: Tensor, *, mode: str = 'fast') -> None",
                    ),
                },
                ("public_api_signature_changed",),
            ),
            (
                "missing target member",
                {
                    "flashinfer/public.py": "from flashinfer.impl import Target as Exported\n",
                    "flashinfer/impl.py": "class Target:\n    pass\n",
                },
                ("public_api_removed",),
            ),
        )

        for name, head, expected_checks in cases:
            with self.subTest(name):
                self.assertEqual(
                    tuple(finding.check for finding in compare_snapshots(base, head)),
                    expected_checks,
                )

    def test_exact_function_reexports_keep_working(self) -> None:
        base = {
            "flashinfer/public.py": textwrap.dedent(
                """
                @flashinfer_api
                def exported(value: int) -> None:
                    pass
                """
            )
        }
        head = {
            "flashinfer/public.py": "from flashinfer.impl import target as exported\n",
            "flashinfer/impl.py": textwrap.dedent(
                """
                @flashinfer_api
                def target(value: int) -> None:
                    pass
                """
            ),
        }

        self.assertEqual(compare_snapshots(base, head), [])


if __name__ == "__main__":
    unittest.main()
