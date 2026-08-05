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

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

from check_pr_api_diff import extract_public_apis  # noqa: E402
from scripts.pr_checks.check_cross_sources import (
    _env_var_reads,
    iter_markdown_paths,
)  # noqa: E402
from scripts.pr_checks.inspect_sources import iter_decorated_functions  # noqa: E402


def test_api_diff_visits_class_body_once() -> None:
    source = """
class PublicClass:
    @flashinfer_api
    def method(self):
        pass
"""

    apis = extract_public_apis("flashinfer/example.py", source)

    assert set(apis) == {"PublicClass.method"}


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
