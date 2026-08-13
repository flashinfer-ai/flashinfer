# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

import check_pr_api_diff as api_diff  # noqa: E402
from check_pr_api_diff import (  # noqa: E402
    ChangedFile,
    extract_public_apis,
    is_compatible_signature_extension,
    module_reexports,
)
from pr_checks.check_cross_sources import (
    _parse_toctree_entries,
    _env_var_reads,
    iter_markdown_paths,
)  # noqa: E402
from pr_checks.git_compare import resolve_merge_base  # noqa: E402
from pr_checks.inspect_sources import iter_decorated_functions  # noqa: E402


def test_api_diff_visits_class_body_once() -> None:
    source = """
class PublicClass:
    @flashinfer_api
    def method(self):
        pass
"""

    apis = extract_public_apis("flashinfer/example.py", source)

    assert set(apis) == {"PublicClass.method"}


def test_module_reexports_include_plain_relative_imports() -> None:
    assert module_reexports("flashinfer/comm.py", "from . import all_reduce\n") == {
        "all_reduce": ("flashinfer", "all_reduce")
    }


def _api(source: str, name: str = "example") -> api_diff.ApiFunction:
    return extract_public_apis("flashinfer/example.py", source)[name]


def test_signature_change_is_not_hidden_by_updated_docstring() -> None:
    before = _api(
        '''
@flashinfer_api
def example(value):
    """Old documentation."""
'''
    )
    after = _api(
        '''
@flashinfer_api
def example(value, required):
    """Updated documentation for both arguments."""
'''
    )

    assert not is_compatible_signature_extension(before, after)


def test_check_reports_signature_change_despite_updated_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "flashinfer/example.py"
    contents = {
        (
            "base",
            path,
        ): '''
@flashinfer_api
def example(value):
    """Old documentation."""
''',
        (
            "head",
            path,
        ): '''
@flashinfer_api
def example(value, required):
    """Updated documentation for both arguments."""
''',
    }
    monkeypatch.setattr(api_diff, "resolve_merge_base", lambda base, head: "base")
    monkeypatch.setattr(
        api_diff,
        "changed_files",
        lambda base, head: [ChangedFile("M", path, path)],
    )
    monkeypatch.setattr(
        api_diff,
        "git_file",
        lambda revision, filename: contents.get((revision, filename)),
    )

    findings = api_diff.check("base-tip", "head")

    assert [finding.check for finding in findings] == ["public_api_signature_changed"]


def test_appended_default_parameter_is_compatible() -> None:
    before = _api(
        """
@flashinfer_api
def example(value: int) -> int:
    pass
"""
    )
    after = _api(
        """
@flashinfer_api
def example(value: int, optional: int = 1, *, enabled: bool = True) -> int:
    pass
"""
    )

    assert is_compatible_signature_extension(before, after)


@pytest.mark.parametrize(
    "after_source",
    [
        "def example(renamed: int) -> int: pass",
        "def example(value: str) -> int: pass",
        "def example(value: int = 1) -> int: pass",
        "def example(value: int) -> str: pass",
    ],
)
def test_existing_signature_changes_are_breaking(after_source: str) -> None:
    before = _api(
        """
@flashinfer_api
def example(value: int) -> int:
    pass
"""
    )
    after = _api("@flashinfer_api\n" + after_source)

    assert not is_compatible_signature_extension(before, after)


def test_changed_files_preserve_both_rename_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_diff,
        "git",
        lambda *args: "R100\0flashinfer/old.py\0flashinfer/new.py\0",
    )

    assert api_diff.changed_files("base", "head") == [
        ChangedFile("R100", "flashinfer/old.py", "flashinfer/new.py")
    ]


def test_resolve_merge_base_excludes_unrelated_target_branch_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    run("init", "-q", "-b", "main")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    (tmp_path / "base.txt").write_text("base\n", encoding="utf-8")
    run("add", "base.txt")
    run("commit", "-qm", "base")
    common = run("rev-parse", "HEAD")

    run("switch", "-qc", "pr")
    (tmp_path / "pr.txt").write_text("pr\n", encoding="utf-8")
    run("add", "pr.txt")
    run("commit", "-qm", "pr")
    pr_head = run("rev-parse", "HEAD")

    run("switch", "-q", "main")
    (tmp_path / "main.txt").write_text("main\n", encoding="utf-8")
    run("add", "main.txt")
    run("commit", "-qm", "main")
    main_tip = run("rev-parse", "HEAD")

    monkeypatch.chdir(tmp_path)
    assert resolve_merge_base(main_tip, pr_head) == common


def test_public_module_rename_is_reported_as_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "@flashinfer_api\ndef example(value): pass\n"
    contents = {
        ("base", "flashinfer/old.py"): source,
        ("head", "flashinfer/new.py"): source,
    }
    monkeypatch.setattr(api_diff, "resolve_merge_base", lambda base, head: "base")
    monkeypatch.setattr(
        api_diff,
        "changed_files",
        lambda base, head: [
            ChangedFile("R100", "flashinfer/old.py", "flashinfer/new.py")
        ],
    )
    monkeypatch.setattr(
        api_diff, "git_file", lambda revision, path: contents.get((revision, path))
    )

    findings = api_diff.check("base-tip", "head")

    assert [finding.check for finding in findings] == ["public_module_moved"]


def test_same_signature_legacy_reexports_preserve_moved_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_core = """
@flashinfer_api
def interleave_moe_scales_for_sm90_mixed_gemm(scales, group_size=32): pass

@flashinfer_api
def interleave_moe_weights_for_sm90_mixed_gemm(weight, quant_type="fp4"): pass
"""
    new_core = """
from .prepare import (
    interleave_moe_scales_for_sm90_mixed_gemm,
    interleave_moe_weights_for_sm90_mixed_gemm,
)
"""
    new_prepare = old_core
    contents = {
        ("base", "flashinfer/fused_moe/core.py"): old_core,
        ("head", "flashinfer/fused_moe/core.py"): new_core,
        ("head", "flashinfer/fused_moe/prepare.py"): new_prepare,
    }
    monkeypatch.setattr(api_diff, "resolve_merge_base", lambda base, head: "base")
    monkeypatch.setattr(
        api_diff,
        "changed_files",
        lambda base, head: [
            ChangedFile(
                "M",
                "flashinfer/fused_moe/core.py",
                "flashinfer/fused_moe/core.py",
            ),
            ChangedFile("A", None, "flashinfer/fused_moe/prepare.py"),
        ],
    )
    monkeypatch.setattr(
        api_diff, "git_file", lambda revision, path: contents.get((revision, path))
    )

    assert api_diff.check("base-tip", "head") == []


def test_legacy_reexport_with_changed_signature_remains_breaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "flashinfer/fused_moe/core.py"
    target_path = "flashinfer/fused_moe/prepare.py"
    contents = {
        ("base", path): "@flashinfer_api\ndef moved(value): pass\n",
        ("head", path): "from .prepare import moved\n",
        ("head", target_path): "@flashinfer_api\ndef moved(value, required): pass\n",
    }
    monkeypatch.setattr(api_diff, "resolve_merge_base", lambda base, head: "base")
    monkeypatch.setattr(
        api_diff,
        "changed_files",
        lambda base, head: [
            ChangedFile("M", path, path),
            ChangedFile("A", None, target_path),
        ],
    )
    monkeypatch.setattr(
        api_diff,
        "git_file",
        lambda revision, filename: contents.get((revision, filename)),
    )

    findings = api_diff.check("base-tip", "head")

    assert [finding.check for finding in findings] == ["public_api_removed"]


def test_env_var_reads_include_constant_aliases() -> None:
    source = """
DIRECT = os.getenv("FLASHINFER_DIRECT")
ENV_NAME: str = "FLASHINFER_INDIRECT"
value = os.environ.get(ENV_NAME)
"""

    reads = [
        (source.count("\n", 0, offset) + 1, name)
        for offset, name in _env_var_reads(source)
    ]

    assert reads == [
        (2, "FLASHINFER_DIRECT"),
        (4, "FLASHINFER_INDIRECT"),
    ]


def test_markdown_paths_cover_supported_contexts() -> None:
    markdown = """
Use `pyproject.toml` and `.github/workflows/docs.yml`.

```bash
python scripts/check_pr_document.py
```

| Guide | docs/api/index.rst |
| Link | https://example.com/not/a/repo/path.py |
"""

    assert iter_markdown_paths(markdown) == [
        "pyproject.toml",
        ".github/workflows/docs.yml",
        "scripts/check_pr_document.py",
        "docs/api/index.rst",
    ]


def test_toctree_entries_skip_free_text_and_keep_explicit_titles() -> None:
    rst = """
.. toctree::
   :maxdepth: 1

   Free text caption
   API Reference <api/index>
   tutorials/quickstart
"""

    assert [entry for _line, entry in _parse_toctree_entries(rst)] == [
        "api/index",
        "tutorials/quickstart",
    ]


def test_all_scope_walks_control_flow_and_nested_classes(tmp_path: Path) -> None:
    package = tmp_path / "flashinfer"
    package.mkdir()
    (package / "sample.py").write_text(
        """
if enabled:
    @flashinfer_api
    def guarded():
        pass

class Outer:
    @flashinfer_api
    def method(self):
        pass

    class Inner:
        if enabled:
            @flashinfer_api
            def nested(self):
                pass
""",
        encoding="utf-8",
    )

    found = [
        (module, node.name)
        for _path, module, node in iter_decorated_functions(package, scope="all")
    ]

    assert found == [
        ("flashinfer.sample", "guarded"),
        ("flashinfer.sample.Outer", "method"),
        ("flashinfer.sample.Outer.Inner", "nested"),
    ]
